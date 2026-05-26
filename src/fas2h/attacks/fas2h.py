"""FA-S2H MVP attack orchestration.

The current implementation is intentionally scoped to the confirmed MVP:
- fixed source carrier indices;
- fixed target evidence banks;
- active loss is `L_inj` only;
- no route amplification loss, no anchor loss, no dynamic refresh.
"""

from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
import re
from typing import Any

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from fas2h.attacks.losses import carrier_target_similarity, injection_loss_logsumexp
from fas2h.attacks.pgd import pgd_step
from fas2h.data.pairs import load_pairs_jsonl
from fas2h.route.bank import build_target_evidence_bank
from fas2h.route.projection import build_projection
from fas2h.route.source_carrier import select_source_carriers
from fas2h.route.target_evidence import select_target_evidence
from fas2h.utils.cache import save_json
from fas2h.utils.image_io import load_image_tensor, save_image_tensor, save_perturbation_visualization
from fas2h.utils.logging import get_logger


def _tensor_to_list(value: torch.Tensor | None) -> Any:
    """Convert tensors to Python lists for JSON output."""
    if value is None:
        return None
    return value.detach().cpu().tolist()


def _slugify(value: str) -> str:
    """Convert free-form text into a filesystem-friendly token."""
    compact = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return compact.strip("_").lower() or "unknown"


class FAS2HAttack:
    """Runnable FA-S2H MVP attack.

    Pipeline:
    1. load source/target images;
    2. extract clean source features and clean target features;
    3. select fixed source carriers;
    4. select fixed target evidence and build fixed banks;
    5. optimize `x_adv` with PGD using `L_inj` only;
    6. save images and route metadata.
    """

    def __init__(self, cfg: Any, models: list[Any]) -> None:
        self.cfg = cfg
        self.models = models
        self.attack_cfg = cfg.attack
        self.runtime_cfg = cfg.runtime
        self.data_cfg = cfg.data
        self.logger = get_logger("fas2h.attack")
        self.projection = build_projection(OmegaConf.to_container(self.attack_cfg.projection, resolve=True))

        for model in self.models:
            model.load()
        self.logger.info("Resolved runtime device: %s", self.models[0].model_device)

    def _method_name(self) -> str:
        """Map the configured selection mode to a readable method name."""
        selection_mode = str(getattr(self.attack_cfg, "selection_mode", "fas2h"))
        if selection_mode == "random_patch":
            return "random_patch_inj"
        return "fas2h_inj"

    def _data_name(self) -> str:
        """Build a short dataset token from the configured pair file."""
        pair_file = Path(str(self.data_cfg.pair_file))
        stem = pair_file.stem
        if stem == "resources_pairs_1000":
            return "res1000"
        return _slugify(stem)

    def _build_auto_run_stem(self) -> str:
        """Build a readable run stem from the current config values."""
        limit = self.data_cfg.limit
        limit_text = "all" if limit is None else str(limit)
        return "_".join(
            [
                _slugify(str(self.attack_cfg.name)),
                self._data_name(),
                f"n{limit_text}",
                f"steps{int(self.attack_cfg.pgd.steps)}",
                f"seed{int(self.runtime_cfg.seed)}",
            ]
        )

    def _run_id(self) -> str:
        """Build a readable run identifier."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        auto_name = bool(getattr(self.runtime_cfg, "auto_name", True))
        stem = self._build_auto_run_stem() if auto_name else _slugify(str(self.attack_cfg.name))
        return f"{stem}_{timestamp}"

    def _resolve_output_dir(self, run_name: str | None = None) -> Path:
        """Resolve and create the output directory for this run."""
        run_id = run_name or self._run_id()
        out_dir = Path(self.runtime_cfg.output_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _select_routes_and_banks(
        self,
        pair_id: str,
        x_src: torch.Tensor,
        x_tar: torch.Tensor,
    ) -> tuple[dict[str, dict[int, Any]], dict[str, dict[int, Any]], dict[str, dict[int, Any]]]:
        """Precompute source carriers and fixed target evidence banks for one pair."""
        source_routes: dict[str, dict[int, Any]] = {}
        target_routes: dict[str, dict[int, Any]] = {}
        target_banks: dict[str, dict[int, Any]] = {}
        shallow_layers = list(self.attack_cfg.shallow_layers)

        with torch.no_grad():
            for model in self.models:
                src_features = model.encode_with_features(x_src, shallow_layers, capture_attn=True)
                tar_features = model.encode_with_features(x_tar, shallow_layers, capture_attn=True)

                source_routes[model.name] = select_source_carriers(
                    feature_bundle=src_features,
                    model_name=model.name,
                    pretrained=model.pretrained,
                    layers=shallow_layers,
                    topk_ratio=float(self.attack_cfg.topk_ratio),
                    projection=self.projection,
                    score_cfg=OmegaConf.to_container(self.attack_cfg.source_score, resolve=True),
                    selection_mode=str(getattr(self.attack_cfg, "selection_mode", "fas2h")),
                    random_seed=int(getattr(self.runtime_cfg, "random_selection_seed", self.runtime_cfg.seed)),
                    pair_id=pair_id,
                )
                target_route_bundle, target_bank_bundle = select_target_evidence(
                    feature_bundle=tar_features,
                    model_name=model.name,
                    pretrained=model.pretrained,
                    layers=shallow_layers,
                    topk_ratio=float(self.attack_cfg.topk_ratio),
                    projection=self.projection,
                    score_cfg=OmegaConf.to_container(self.attack_cfg.target_score, resolve=True),
                    selection_mode=str(getattr(self.attack_cfg, "selection_mode", "fas2h")),
                    random_seed=int(getattr(self.runtime_cfg, "random_selection_seed", self.runtime_cfg.seed)),
                    pair_id=pair_id,
                )
                target_routes[model.name] = target_route_bundle
                target_banks[model.name] = {
                    layer: build_target_evidence_bank(bundle) for layer, bundle in target_bank_bundle.items()
                }

        return source_routes, target_routes, target_banks

    def _compute_injection_objective(
        self,
        x_adv: torch.Tensor,
        source_routes: dict[str, dict[int, Any]],
        target_banks: dict[str, dict[int, Any]],
    ) -> dict[str, Any]:
        """Compute the ensemble-averaged MVP `L_inj` objective and diagnostics."""
        shallow_layers = list(self.attack_cfg.shallow_layers)
        per_term_losses: list[torch.Tensor] = []
        per_model_losses: dict[str, list[torch.Tensor]] = {}
        per_layer_losses: dict[str, dict[str, torch.Tensor]] = {}
        sim_values: list[torch.Tensor] = []
        per_model_sims: dict[str, list[torch.Tensor]] = {}
        per_layer_sims: dict[str, dict[str, torch.Tensor]] = {}

        for model in self.models:
            adv_features = model.encode_with_features(x_adv, shallow_layers, capture_attn=False)
            per_model_losses[model.name] = []
            per_model_sims[model.name] = []
            per_layer_losses[model.name] = {}
            per_layer_sims[model.name] = {}
            for layer in shallow_layers:
                patch_tokens = adv_features.patch_tokens_by_layer[layer]
                source_indices = source_routes[model.name][layer].source_indices
                target_bank = target_banks[model.name][layer].target_bank
                assert source_indices is not None
                assert target_bank is not None

                gather_index = source_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
                adv_carriers = torch.gather(patch_tokens, dim=1, index=gather_index)
                layer_loss = injection_loss_logsumexp(
                    adv_patch_tokens=adv_carriers,
                    target_bank=target_bank,
                    tau=float(self.attack_cfg.loss.tau),
                    projection=self.projection,
                    layer=layer,
                )
                per_term_losses.append(layer_loss)
                per_model_losses[model.name].append(layer_loss)
                per_layer_losses[model.name][f"layer_{layer}"] = layer_loss

                layer_sim = carrier_target_similarity(
                    adv_patch_tokens=adv_carriers,
                    target_bank=target_bank,
                    projection=self.projection,
                    layer=layer,
                )
                sim_values.append(layer_sim)
                per_model_sims[model.name].append(layer_sim)
                per_layer_sims[model.name][f"layer_{layer}"] = layer_sim

        if not per_term_losses:
            raise RuntimeError("No injection-loss terms were produced for the ensemble.")
        total_loss = torch.stack(per_term_losses).mean()
        global_sim = torch.stack(sim_values).mean()
        per_model_loss_scalar = {
            name: torch.stack(values).mean() for name, values in per_model_losses.items() if values
        }
        per_model_sim_scalar = {
            name: torch.stack(values).mean() for name, values in per_model_sims.items() if values
        }
        return {
            "loss_total": total_loss,
            "loss_inj": total_loss,
            "carrier_target_sim": global_sim,
            "per_model_loss_inj": per_model_loss_scalar,
            "per_layer_loss_inj": per_layer_losses,
            "per_model_sim": per_model_sim_scalar,
            "per_layer_sim": per_layer_sims,
        }

    def _save_route_metadata(
        self,
        pair_dir: Path,
        source_routes: dict[str, dict[int, Any]],
        target_routes: dict[str, dict[int, Any]],
    ) -> None:
        """Save selected source carriers and target evidence metadata."""
        selection_mode = str(getattr(self.attack_cfg, "selection_mode", "fas2h"))
        source_payload: dict[str, Any] = {
            "selection_mode": selection_mode,
            "selected_by": "score" if selection_mode == "fas2h" else "random",
            "scores_logged_for_diagnosis_only": selection_mode == "random_patch",
            "topk_ratio": float(self.attack_cfg.topk_ratio),
            "models": {},
        }
        target_payload: dict[str, Any] = {
            "selection_mode": selection_mode,
            "selected_by": "score" if selection_mode == "fas2h" else "random",
            "scores_logged_for_diagnosis_only": selection_mode == "random_patch",
            "topk_ratio": float(self.attack_cfg.topk_ratio),
            "models": {},
        }
        for model_name, layer_bundles in source_routes.items():
            source_payload["models"][model_name] = {}
            for layer, bundle in layer_bundles.items():
                layer_key = f"layer_{layer}"
                source_payload["models"][model_name][layer_key] = []
                assert bundle.source_indices is not None
                indices = bundle.source_indices[0].detach().cpu().tolist()
                for rank, patch_idx in enumerate(indices, start=1):
                    idx = int(patch_idx)
                    source_payload["models"][model_name][layer_key].append(
                        {
                            "rank": rank,
                            "index": idx,
                            "total_score": float(bundle.meta["full_source_scores"][0][idx]),
                            "attn_score": float(bundle.meta["attn_scores"][0][idx]),
                            "cls_coupling_score": float(bundle.meta["cls_coupling_scores"][0][idx]),
                            "global_score": float(bundle.meta["global_scores"][0][idx]),
                        }
                    )
        for model_name, layer_bundles in target_routes.items():
            target_payload["models"][model_name] = {}
            for layer, bundle in layer_bundles.items():
                layer_key = f"layer_{layer}"
                target_payload["models"][model_name][layer_key] = []
                assert bundle.target_indices is not None
                indices = bundle.target_indices[0].detach().cpu().tolist()
                for rank, patch_idx in enumerate(indices, start=1):
                    idx = int(patch_idx)
                    target_payload["models"][model_name][layer_key].append(
                        {
                            "rank": rank,
                            "index": idx,
                            "total_score": float(bundle.meta["full_target_scores"][0][idx]),
                            "path_score": float(bundle.rollout_scores[0][idx]) if bundle.rollout_scores is not None else None,
                            "semantic_score": float(bundle.meta["semantic_scores"][0][idx]),
                        }
                    )

        save_json(pair_dir / "source_carriers.json", source_payload)
        save_json(pair_dir / "target_evidence.json", target_payload)

    def precompute_pair(self, pair: dict[str, Any], run_dir: Path) -> dict[str, Any]:
        """Precompute and save route metadata without running PGD."""
        device = self.models[0].model_device
        x_src = load_image_tensor(pair["source_path"]).unsqueeze(0).to(device)
        x_tar = load_image_tensor(pair["target_path"]).unsqueeze(0).to(device)
        source_routes, target_routes, target_banks = self._select_routes_and_banks(
            pair_id=pair["pair_id"],
            x_src=x_src,
            x_tar=x_tar,
        )

        pair_dir = run_dir / "samples" / pair["pair_id"]
        pair_dir.mkdir(parents=True, exist_ok=True)
        self._save_route_metadata(pair_dir=pair_dir, source_routes=source_routes, target_routes=target_routes)
        save_json(
            pair_dir / "metadata.json",
            {
                "pair_id": pair["pair_id"],
                "source_path": pair["source_path"],
                "target_path": pair["target_path"],
                "target_keywords": pair.get("target_keywords"),
                "note": "target_keywords are metadata only and are not used in optimization.",
                "static_target_bank": bool(self.attack_cfg.bank.static_target_bank),
                "static_source_carrier": bool(self.attack_cfg.bank.static_source_carrier),
                "num_models": len(self.models),
                "num_layers": len(self.attack_cfg.shallow_layers),
            },
        )
        return {
            "source_routes": source_routes,
            "target_routes": target_routes,
            "target_banks": target_banks,
        }

    def run_pair(self, pair: dict[str, Any], run_dir: Path) -> dict[str, Any]:
        """Run the full MVP attack for a single source-target pair."""
        device = self.models[0].model_device
        x_src = load_image_tensor(pair["source_path"]).unsqueeze(0).to(device)
        x_tar = load_image_tensor(pair["target_path"]).unsqueeze(0).to(device)

        source_routes, target_routes, target_banks = self._select_routes_and_banks(
            pair_id=pair["pair_id"],
            x_src=x_src,
            x_tar=x_tar,
        )
        pair_dir = run_dir / "samples" / pair["pair_id"]
        pair_dir.mkdir(parents=True, exist_ok=True)

        self._save_route_metadata(pair_dir=pair_dir, source_routes=source_routes, target_routes=target_routes)

        x_adv = x_src.clone()
        if bool(self.attack_cfg.pgd.random_start):
            eps = float(self.attack_cfg.pgd.eps)
            x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)).clamp(0.0, 1.0)

        loss_log: list[dict[str, Any]] = []
        step_iter = tqdm(
            range(int(self.attack_cfg.pgd.steps)),
            desc=f"{pair['pair_id']} steps",
            leave=False,
            disable=not bool(getattr(self.runtime_cfg, "progress", True)),
        )
        for step_idx in step_iter:
            x_adv.requires_grad_(True)
            objective = self._compute_injection_objective(
                x_adv=x_adv,
                source_routes=source_routes,
                target_banks=target_banks,
            )
            loss = objective["loss_total"]
            grad = torch.autograd.grad(loss, x_adv)[0]
            with torch.no_grad():
                x_adv = pgd_step(
                    x_adv=x_adv,
                    grad=grad,
                    x_src=x_src,
                    eps=float(self.attack_cfg.pgd.eps),
                    step_size=float(self.attack_cfg.pgd.step_size),
                    clamp_min=float(self.attack_cfg.pgd.clamp_min),
                    clamp_max=float(self.attack_cfg.pgd.clamp_max),
                )
            linf = float((x_adv - x_src).abs().max().detach().cpu().item())
            pixel_min = float(x_adv.min().detach().cpu().item())
            pixel_max = float(x_adv.max().detach().cpu().item())
            mean_abs_delta = float((x_adv - x_src).abs().mean().detach().cpu().item())
            loss_value = float(objective["loss_total"].detach().cpu().item())
            loss_inj_value = float(objective["loss_inj"].detach().cpu().item())
            sim_value = float(objective["carrier_target_sim"].detach().cpu().item())
            per_model_loss = {k: float(v.detach().cpu().item()) for k, v in objective["per_model_loss_inj"].items()}
            per_layer_loss: dict[str, dict[str, float]] = {}
            for model_name, layer_map in objective["per_layer_loss_inj"].items():
                per_layer_loss[model_name] = {
                    layer_name: float(layer_loss.detach().cpu().item()) for layer_name, layer_loss in layer_map.items()
                }
            per_model_sim = {k: float(v.detach().cpu().item()) for k, v in objective["per_model_sim"].items()}
            per_layer_sim: dict[str, dict[str, float]] = {}
            for model_name, layer_map in objective["per_layer_sim"].items():
                per_layer_sim[model_name] = {
                    layer_name: float(layer_sim.detach().cpu().item()) for layer_name, layer_sim in layer_map.items()
                }
            loss_log.append(
                {
                    "step": float(step_idx),
                    "loss": loss_value,
                    "loss_total": loss_value,
                    "loss_inj": loss_inj_value,
                    "carrier_target_sim": sim_value,
                    "linf": linf,
                    "pixel_min": pixel_min,
                    "pixel_max": pixel_max,
                    "mean_abs_delta": mean_abs_delta,
                    "per_model_loss_inj": per_model_loss,
                    "per_layer_loss_inj": per_layer_loss,
                    "per_model_carrier_target_sim": per_model_sim,
                    "per_layer_carrier_target_sim": per_layer_sim,
                }
            )
            if bool(getattr(self.runtime_cfg, "progress", True)):
                step_iter.set_postfix(loss=f"{loss_value:.4f}", sim=f"{sim_value:.4f}", linf=f"{linf:.4f}")

        delta = x_adv - x_src
        save_image_tensor(x_adv, pair_dir / "adv.png")
        save_perturbation_visualization(delta, pair_dir / "perturbation.png", eps=float(self.attack_cfg.pgd.eps))
        save_json(pair_dir / "loss_log.json", {"losses": loss_log})
        steps = [int(item["step"]) for item in loss_log]
        losses = [float(item["loss_inj"]) for item in loss_log]
        sims = [float(item["carrier_target_sim"]) for item in loss_log]
        linfs = [float(item["linf"]) for item in loss_log]
        pixel_mins = [float(item["pixel_min"]) for item in loss_log]
        pixel_maxs = [float(item["pixel_max"]) for item in loss_log]
        deltas = [float(item["mean_abs_delta"]) for item in loss_log]
        slice_n = max(1, math.ceil(len(losses) * 0.1))
        loss_start = losses[0]
        loss_end = losses[-1]
        loss_delta = loss_end - loss_start
        loss_argmin_index = min(range(len(losses)), key=lambda i: losses[i])
        per_model_loss_start = loss_log[0]["per_model_loss_inj"]
        per_model_loss_end = loss_log[-1]["per_model_loss_inj"]
        per_model_loss_delta = {
            model_name: float(per_model_loss_end[model_name] - per_model_loss_start[model_name])
            for model_name in per_model_loss_start.keys()
        }
        per_layer_loss_start = loss_log[0]["per_layer_loss_inj"]
        per_layer_loss_end = loss_log[-1]["per_layer_loss_inj"]
        per_layer_loss_delta: dict[str, dict[str, float]] = {}
        for model_name, layers in per_layer_loss_start.items():
            per_layer_loss_delta[model_name] = {}
            for layer_name in layers.keys():
                per_layer_loss_delta[model_name][layer_name] = float(
                    per_layer_loss_end[model_name][layer_name] - per_layer_loss_start[model_name][layer_name]
                )
        per_model_sim_start = loss_log[0]["per_model_carrier_target_sim"]
        per_model_sim_end = loss_log[-1]["per_model_carrier_target_sim"]
        per_model_sim_delta = {
            model_name: float(per_model_sim_end[model_name] - per_model_sim_start[model_name])
            for model_name in per_model_sim_start.keys()
        }
        per_layer_sim_start = loss_log[0]["per_layer_carrier_target_sim"]
        per_layer_sim_end = loss_log[-1]["per_layer_carrier_target_sim"]
        per_layer_sim_delta: dict[str, dict[str, float]] = {}
        for model_name, layers in per_layer_sim_start.items():
            per_layer_sim_delta[model_name] = {}
            for layer_name in layers.keys():
                per_layer_sim_delta[model_name][layer_name] = float(
                    per_layer_sim_end[model_name][layer_name] - per_layer_sim_start[model_name][layer_name]
                )

        metrics_payload: dict[str, Any] = {
            "pair_id": pair["pair_id"],
            "method": self._method_name(),
            "selection_mode": str(getattr(self.attack_cfg, "selection_mode", "fas2h")),
            "shallow_layers": list(self.attack_cfg.shallow_layers),
            "topk_ratio": float(self.attack_cfg.topk_ratio),
            "seed": int(self.runtime_cfg.seed),
            "random_selection_seed": int(getattr(self.runtime_cfg, "random_selection_seed", self.runtime_cfg.seed)),
            "steps": int(self.attack_cfg.pgd.steps),
            "eps": float(self.attack_cfg.pgd.eps),
            "step_size": float(self.attack_cfg.pgd.step_size),
            "inj_loss_start": loss_start,
            "inj_loss_end": loss_end,
            "inj_loss_delta": loss_delta,
            "inj_loss_relative_change": None if loss_start == 0 else float(loss_delta / abs(loss_start)),
            "inj_loss_min": float(min(losses)),
            "inj_loss_argmin_step": int(steps[loss_argmin_index]),
            "carrier_target_sim_start": sims[0],
            "carrier_target_sim_end": sims[-1],
            "carrier_target_sim_delta": float(sims[-1] - sims[0]),
            "sim_increase_success": bool(sims[-1] > sims[0]),
            "linf_max": float(max(linfs)),
            "linf_end": float(linfs[-1]),
            "linf_valid": bool(max(linfs) <= float(self.attack_cfg.pgd.eps) + 1e-6),
            "pixel_min_global": float(min(pixel_mins)),
            "pixel_max_global": float(max(pixel_maxs)),
            "pixel_range_valid": bool(min(pixel_mins) >= 0.0 and max(pixel_maxs) <= 1.0),
            "mean_abs_delta_end": float(deltas[-1]),
            "monotonic_decrease_ratio": float(
                sum(losses[i + 1] < losses[i] for i in range(len(losses) - 1)) / max(1, len(losses) - 1)
            ),
            "first_10pct_mean_loss": float(sum(losses[:slice_n]) / slice_n),
            "final_10pct_mean_loss": float(sum(losses[-slice_n:]) / slice_n),
            "smooth_decrease_success": bool((sum(losses[-slice_n:]) / slice_n) < (sum(losses[:slice_n]) / slice_n)),
            "per_model_loss_delta": per_model_loss_delta,
            "per_layer_loss_delta": per_layer_loss_delta,
            "per_model_sim_delta": per_model_sim_delta,
            "per_layer_sim_delta": per_layer_sim_delta,
            "notes": "",
        }
        save_json(pair_dir / "metrics.json", metrics_payload)
        save_json(
            pair_dir / "metadata.json",
            {
                "pair_id": pair["pair_id"],
                "method": self._method_name(),
                "selection_mode": str(getattr(self.attack_cfg, "selection_mode", "fas2h")),
                "random_selection_seed": int(getattr(self.runtime_cfg, "random_selection_seed", self.runtime_cfg.seed)),
                "source_path": pair["source_path"],
                "target_path": pair["target_path"],
                "target_keywords": pair.get("target_keywords"),
                "target_keywords_note": "target_keywords are metadata only and are not used by the attack loss.",
                "attack_name": self.attack_cfg.name,
                "active_loss": self.attack_cfg.loss.active,
                "projection_type": self.attack_cfg.projection.type,
                "models": [
                    {
                        "name": model.name,
                        "model_name": model.model_name,
                        "pretrained": model.pretrained,
                    }
                    for model in self.models
                ],
            },
        )

        return {
            "pair_id": pair["pair_id"],
            "method": self._method_name(),
            "selection_mode": str(getattr(self.attack_cfg, "selection_mode", "fas2h")),
            "final_loss": loss_log[-1]["loss_inj"] if loss_log else None,
            "metrics": metrics_payload,
            "pair_dir": str(pair_dir),
        }

    def run(self, limit: int | None = None, run_name: str | None = None) -> list[dict[str, Any]]:
        """Run the attack across pairs listed in the configured JSONL file."""
        run_dir = self._resolve_output_dir(run_name=run_name)
        OmegaConf.save(config=self.cfg, f=run_dir / "config_used.yaml")

        pairs = load_pairs_jsonl(self.data_cfg.pair_file)
        if limit is not None:
            pairs = pairs[:limit]
        elif self.data_cfg.limit is not None:
            pairs = pairs[: int(self.data_cfg.limit)]

        results = []
        pair_iter = tqdm(
            pairs,
            desc="pairs",
            disable=not bool(getattr(self.runtime_cfg, "progress", True)),
        )
        for pair in pair_iter:
            self.logger.info("Running FA-S2H MVP for pair_id=%s", pair["pair_id"])
            results.append(self.run_pair(pair=deepcopy(pair), run_dir=run_dir))
        summary_rows: list[dict[str, Any]] = []
        for result in results:
            metrics = result["metrics"]
            summary_rows.append(
                {
                    "run_id": run_dir.name,
                    "pair_id": result["pair_id"],
                    "method": result["method"],
                    "selection_mode": result["selection_mode"],
                    "shallow_layers": "|".join(str(v) for v in self.attack_cfg.shallow_layers),
                    "inj_loss_start": metrics["inj_loss_start"],
                    "inj_loss_end": metrics["inj_loss_end"],
                    "inj_loss_delta": metrics["inj_loss_delta"],
                    "inj_loss_relative_change": metrics["inj_loss_relative_change"],
                    "carrier_target_sim_start": metrics["carrier_target_sim_start"],
                    "carrier_target_sim_end": metrics["carrier_target_sim_end"],
                    "carrier_target_sim_delta": metrics["carrier_target_sim_delta"],
                    "linf_max": metrics["linf_max"],
                    "linf_valid": metrics["linf_valid"],
                    "pixel_range_valid": metrics["pixel_range_valid"],
                    "mean_abs_delta_end": metrics["mean_abs_delta_end"],
                    "monotonic_decrease_ratio": metrics["monotonic_decrease_ratio"],
                    "smooth_decrease_success": metrics["smooth_decrease_success"],
                    "notes": metrics.get("notes", ""),
                }
            )
        if summary_rows:
            summary_path = run_dir / "summary.csv"
            with summary_path.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
        summary_json = {
            "run_id": run_dir.name,
            "method": self._method_name(),
            "selection_mode": str(getattr(self.attack_cfg, "selection_mode", "fas2h")),
            "shallow_layers": list(self.attack_cfg.shallow_layers),
            "num_pairs": len(summary_rows),
            "loss_decrease_rate": (
                sum(row["inj_loss_end"] < row["inj_loss_start"] for row in summary_rows) / len(summary_rows)
                if summary_rows
                else None
            ),
            "sim_increase_rate": (
                sum(row["carrier_target_sim_end"] > row["carrier_target_sim_start"] for row in summary_rows) / len(summary_rows)
                if summary_rows
                else None
            ),
            "linf_valid_rate": (
                sum(bool(row["linf_valid"]) for row in summary_rows) / len(summary_rows) if summary_rows else None
            ),
            "pixel_valid_rate": (
                sum(bool(row["pixel_range_valid"]) for row in summary_rows) / len(summary_rows)
                if summary_rows
                else None
            ),
        }
        save_json(run_dir / "summary.json", summary_json)
        readme_lines = [
            f"# Run {run_dir.name}",
            "",
            f"- method: {self._method_name()}",
            f"- selection_mode: {getattr(self.attack_cfg, 'selection_mode', 'fas2h')}",
            f"- shallow_layers: {list(self.attack_cfg.shallow_layers)}",
            f"- topk_ratio: {float(self.attack_cfg.topk_ratio)}",
            f"- steps: {int(self.attack_cfg.pgd.steps)}",
            f"- eps: {float(self.attack_cfg.pgd.eps)}",
            f"- step_size: {float(self.attack_cfg.pgd.step_size)}",
            f"- seed: {int(self.runtime_cfg.seed)}",
            f"- random_selection_seed: {int(getattr(self.runtime_cfg, 'random_selection_seed', self.runtime_cfg.seed))}",
            "",
            "NOTE: This run reports only surrogate-level metrics. It does not prove closed-model attack success.",
        ]
        (run_dir / "README_run.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
        return results
