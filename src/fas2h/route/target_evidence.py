"""Target-side final-aware shallow evidence scoring and selection."""

from __future__ import annotations

import hashlib
import math

import torch
import torch.nn.functional as F

from fas2h.features import BankBundle, FeatureBundle, RouteBundle
from fas2h.route.rollout import compute_rollout, extract_cls_to_patch_rollout


def _stable_random_topk_indices(
    num_patches: int,
    topk: int,
    seed: int,
    pair_id: str,
    model_name: str,
    layer: int,
) -> torch.Tensor:
    """Build deterministic random patch indices for one `(pair, model, layer)`."""
    key = f"{seed}|{pair_id}|{model_name}|{layer}|target"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    local_seed = int(digest[:16], 16) % (2**31 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(local_seed)
    return torch.randperm(num_patches, generator=generator)[:topk]


def compute_target_scores(
    feature_bundle: FeatureBundle,
    layer: int,
    projection,
    lambda_path: float,
    lambda_sem: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the FA-S2H target evidence score `E_{m,j}^l`.

    Formula:
    `E = lambda_path * rollout + lambda_sem * cos(P_l z_j^l, sg(g))`

    NOTE(fas2h): target evidence is called "final-aware" because shallow target
    tokens are scored partly by how much downstream rollout suggests they feed
    into the final CLS route, while the final global feature itself remains
    detached and is not directly optimized as an attack target.
    NOTE(fas2h): the MVP does not use final-layer patch tokens as direct attack
    targets. Final semantics are guidance only.
    """
    patch_tokens = feature_bundle.patch_tokens_by_layer[layer]
    final_global = feature_bundle.final_global
    if final_global is None:
        raise ValueError("FeatureBundle.final_global is required for target score computation.")

    rollout_matrix = compute_rollout(
        attentions_by_layer=feature_bundle.attentions_by_layer,
        start_layer=layer + 1,
        end_layer=feature_bundle.num_layers - 1,
    )
    rollout_scores = extract_cls_to_patch_rollout(rollout_matrix)
    projected_patch = projection.project(layer, patch_tokens)
    projected_global = projection.project(layer, final_global.detach()).unsqueeze(1).expand_as(projected_patch)
    semantic_scores = F.cosine_similarity(projected_patch, projected_global, dim=-1)
    total_scores = lambda_path * rollout_scores + lambda_sem * semantic_scores
    return total_scores, rollout_scores, semantic_scores


def select_target_evidence(
    feature_bundle: FeatureBundle,
    model_name: str,
    pretrained: str,
    layers: list[int],
    topk_ratio: float,
    projection,
    score_cfg: dict,
    selection_mode: str = "fas2h",
    random_seed: int = 42,
    pair_id: str = "unknown_pair",
) -> tuple[dict[int, RouteBundle], dict[int, BankBundle]]:
    """Select top-k target evidence tokens and build detached fixed banks."""
    num_patches = feature_bundle.num_patches
    topk = max(1, min(num_patches, math.floor(topk_ratio * num_patches)))

    route_bundles: dict[int, RouteBundle] = {}
    bank_bundles: dict[int, BankBundle] = {}
    for layer in layers:
        total_scores, rollout_scores, semantic_scores = compute_target_scores(
            feature_bundle=feature_bundle,
            layer=layer,
            projection=projection,
            lambda_path=float(score_cfg["lambda_path"]),
            lambda_sem=float(score_cfg["lambda_sem"]),
        )
        if selection_mode == "fas2h":
            top_scores, top_indices = torch.topk(total_scores, k=topk, dim=-1)
        elif selection_mode == "random_patch":
            indices_per_batch = []
            for batch_idx in range(total_scores.shape[0]):
                indices_per_batch.append(
                    _stable_random_topk_indices(
                        num_patches=num_patches,
                        topk=topk,
                        seed=random_seed + batch_idx,
                        pair_id=pair_id,
                        model_name=model_name,
                        layer=layer,
                    )
                )
            top_indices = torch.stack(indices_per_batch, dim=0).to(total_scores.device)
            top_scores = torch.gather(total_scores, dim=-1, index=top_indices)
        else:
            raise ValueError(f"Unsupported selection_mode: {selection_mode}")
        patch_tokens = feature_bundle.patch_tokens_by_layer[layer]
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
        target_bank = torch.gather(patch_tokens, dim=1, index=gather_index).detach()

        route_bundles[layer] = RouteBundle(
            target_indices=top_indices,
            target_scores=top_scores,
            rollout_scores=rollout_scores,
            layer=layer,
            model_name=model_name,
            pretrained=pretrained,
            meta={
                "full_target_scores": total_scores.detach().cpu(),
                "semantic_scores": semantic_scores.detach().cpu(),
                "topk": topk,
                "selection_mode": selection_mode,
                "selected_by": "score" if selection_mode == "fas2h" else "random",
                "scores_logged_for_diagnosis_only": selection_mode == "random_patch",
                "comment": (
                    "Attention rollout is an approximation, not causal proof. "
                    "The selected target evidence tokens are kept fixed during PGD in the MVP."
                ),
            },
        )
        bank_bundles[layer] = BankBundle(
            target_bank=target_bank,
            target_indices=top_indices.detach(),
            target_scores=top_scores.detach(),
            layer=layer,
            model_name=model_name,
            pretrained=pretrained,
            detached=True,
            meta={
                "rollout_scores": rollout_scores.detach().cpu(),
                "semantic_scores": semantic_scores.detach().cpu(),
                "full_target_scores": total_scores.detach().cpu(),
            },
        )
    return route_bundles, bank_bundles
