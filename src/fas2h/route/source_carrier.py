"""Source-side shortcut carrier scoring and top-k selection."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from fas2h.features import FeatureBundle, RouteBundle
from fas2h.route.rollout import average_heads


def compute_source_scores(
    feature_bundle: FeatureBundle,
    layer: int,
    projection,
    alpha_attn: float,
    beta_cls: float,
    gamma_global: float,
) -> torch.Tensor:
    """Compute the FA-S2H source shortcut score `S_{m,i}^l`.

    Formula:
    `S = alpha * A + beta * cos(z_i^l, c^l) + gamma * cos(P_l z_i^l, sg(g))`

    Shapes:
    - patch tokens `[B, N, D]`
    - CLS token `[B, D]`
    - attention `[B, H, N+1, N+1]`

    NOTE(fas2h): these are called "shortcut carriers" because they are high-scoring
    shallow patches that appear to support the model's semantic route from local
    evidence toward the final image representation.
    """
    patch_tokens = feature_bundle.patch_tokens_by_layer[layer]
    cls_tokens = feature_bundle.cls_tokens_by_layer[layer]
    final_global = feature_bundle.final_global
    attention = feature_bundle.attentions_by_layer[layer]

    if final_global is None:
        raise ValueError("FeatureBundle.final_global is required for source score computation.")

    cls_to_patch_attn = average_heads(attention)[:, 0, 1:]
    cls_sim = F.cosine_similarity(
        patch_tokens,
        cls_tokens.unsqueeze(1).expand_as(patch_tokens),
        dim=-1,
    )
    projected_patch = projection.project(layer, patch_tokens)
    projected_global = projection.project(layer, final_global.detach()).unsqueeze(1).expand_as(projected_patch)
    global_sim = F.cosine_similarity(projected_patch, projected_global, dim=-1)
    return alpha_attn * cls_to_patch_attn + beta_cls * cls_sim + gamma_global * global_sim


def select_source_carriers(
    feature_bundle: FeatureBundle,
    model_name: str,
    pretrained: str,
    layers: list[int],
    topk_ratio: float,
    projection,
    score_cfg: dict,
) -> dict[int, RouteBundle]:
    """Select fixed source carrier indices for the MVP attack.

    MVP behavior:
    - source carrier indices are fixed during PGD;
    - the score is a proxy for shortcut route selection, not a proof of causality.
    """
    num_patches = feature_bundle.num_patches
    topk = max(1, min(num_patches, math.floor(topk_ratio * num_patches)))

    bundles: dict[int, RouteBundle] = {}
    for layer in layers:
        full_scores = compute_source_scores(
            feature_bundle=feature_bundle,
            layer=layer,
            projection=projection,
            alpha_attn=float(score_cfg["alpha_attn"]),
            beta_cls=float(score_cfg["beta_cls"]),
            gamma_global=float(score_cfg["gamma_global"]),
        )
        top_scores, top_indices = torch.topk(full_scores, k=topk, dim=-1)
        bundles[layer] = RouteBundle(
            source_indices=top_indices,
            source_scores=top_scores,
            layer=layer,
            model_name=model_name,
            pretrained=pretrained,
            meta={
                "full_source_scores": full_scores.detach().cpu(),
                "topk": topk,
                "comment": (
                    "This is MVP behavior: source carrier indices are fixed during PGD. "
                    "These indices are a score-based proxy for shallow shortcut routes."
                ),
            },
        )
    return bundles
