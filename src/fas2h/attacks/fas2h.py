"""FA-S2H MVP attack orchestration.

The current implementation is intentionally scoped to the confirmed MVP:
- fixed source carrier indices;
- fixed target evidence banks;
- active loss is `L_inj` only;
- no route amplification loss, no anchor loss, no dynamic refresh.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import torch
from omegaconf import OmegaConf

from fas2h.attacks.losses import injection_loss_logsumexp
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
                )
                target_route_bundle, target_bank_bundle = select_target_evidence(
                    feature_bundle=tar_features,
                    model_name=model.name,
                    pretrained=model.pretrained,
                    layers=shallow_layers,
                    topk_ratio=float(self.attack_cfg.topk_ratio),
                    projection=self.projection,
                    score_cfg=OmegaConf.to_container(self.attack_cfg.target_score, resolve=True),
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
    ) -> torch.Tensor:
        """Compute the ensemble-averaged MVP `L_inj` objective."""
        shallow_layers = list(self.attack_cfg.shallow_layers)
        per_term_losses: list[torch.Tensor] = []

        for model in self.models:
            adv_features = model.encode_with_features(x_adv, shallow_layers, capture_attn=False)
            for layer in shallow_layers:
                patch_tokens = adv_features.patch_tokens_by_layer[layer]
                source_indices = source_routes[model.name][layer].source_indices
                target_bank = target_banks[model.name][layer].target_bank
                assert source_indices is not None
                assert target_bank is not None

                gather_index = source_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
                adv_carriers = torch.gather(patch_tokens, dim=1, index=gather_index)
                per_term_losses.append(
                    injection_loss_logsumexp(
                        adv_patch_tokens=adv_carriers,
                        target_bank=target_bank,
                        tau=float(self.attack_cfg.loss.tau),
                        projection=self.projection,
                        layer=layer,
                    )
                )

        if not per_term_losses:
            raise RuntimeError("No injection-loss terms were produced for the ensemble.")
        return torch.stack(per_term_losses).mean()

    def _save_route_metadata(
        self,
        pair_dir: Path,
        source_routes: dict[str, dict[int, Any]],
        target_routes: dict[str, dict[int, Any]],
    ) -> None:
        """Save selected source carriers and target evidence metadata."""
        source_payload: dict[str, Any] = {}
        target_payload: dict[str, Any] = {}
        for model_name, layer_bundles in source_routes.items():
            source_payload[model_name] = {}
            for layer, bundle in layer_bundles.items():
                source_payload[model_name][str(layer)] = {
                    "source_indices": _tensor_to_list(bundle.source_indices),
                    "source_scores": _tensor_to_list(bundle.source_scores),
                    "comment": bundle.meta.get("comment"),
                }
        for model_name, layer_bundles in target_routes.items():
            target_payload[model_name] = {}
            for layer, bundle in layer_bundles.items():
                target_payload[model_name][str(layer)] = {
                    "target_indices": _tensor_to_list(bundle.target_indices),
                    "target_scores": _tensor_to_list(bundle.target_scores),
                    "rollout_scores": _tensor_to_list(bundle.rollout_scores),
                    "comment": bundle.meta.get("comment"),
                }

        save_json(pair_dir / "source_carriers.json", source_payload)
        save_json(pair_dir / "target_evidence.json", target_payload)

    def precompute_pair(self, pair: dict[str, Any], run_dir: Path) -> dict[str, Any]:
        """Precompute and save route metadata without running PGD."""
        device = self.models[0].model_device
        x_src = load_image_tensor(pair["source_path"]).unsqueeze(0).to(device)
        x_tar = load_image_tensor(pair["target_path"]).unsqueeze(0).to(device)
        source_routes, target_routes, target_banks = self._select_routes_and_banks(x_src=x_src, x_tar=x_tar)

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

        source_routes, target_routes, target_banks = self._select_routes_and_banks(x_src=x_src, x_tar=x_tar)
        pair_dir = run_dir / "samples" / pair["pair_id"]
        pair_dir.mkdir(parents=True, exist_ok=True)

        self._save_route_metadata(pair_dir=pair_dir, source_routes=source_routes, target_routes=target_routes)

        x_adv = x_src.clone()
        if bool(self.attack_cfg.pgd.random_start):
            eps = float(self.attack_cfg.pgd.eps)
            x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)).clamp(0.0, 1.0)

        loss_log: list[dict[str, float]] = []
        for step_idx in range(int(self.attack_cfg.pgd.steps)):
            x_adv.requires_grad_(True)
            loss = self._compute_injection_objective(
                x_adv=x_adv,
                source_routes=source_routes,
                target_banks=target_banks,
            )
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
            loss_log.append({"step": float(step_idx), "loss": float(loss.detach().cpu().item())})

        delta = x_adv - x_src
        save_image_tensor(x_adv, pair_dir / "adv.png")
        save_perturbation_visualization(delta, pair_dir / "perturbation.png", eps=float(self.attack_cfg.pgd.eps))
        save_json(pair_dir / "loss_log.json", {"losses": loss_log})
        save_json(
            pair_dir / "metadata.json",
            {
                "pair_id": pair["pair_id"],
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
            "final_loss": loss_log[-1]["loss"] if loss_log else None,
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
        for pair in pairs:
            self.logger.info("Running FA-S2H MVP for pair_id=%s", pair["pair_id"])
            results.append(self.run_pair(pair=deepcopy(pair), run_dir=run_dir))
        return results
