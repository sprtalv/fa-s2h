"""Loss functions used by the FA-S2H MVP attack."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fas2h.features import FeatureBundle, RouteBundle
from fas2h.route.rollout import cls_to_selected_patch_rollout


def injection_loss_logsumexp(
    adv_patch_tokens: torch.Tensor,
    target_bank: torch.Tensor,
    tau: float,
    projection,
    layer: int,
) -> torch.Tensor:
    """Compute the MVP shallow shortcut injection loss `L_inj`.

    Shapes:
    - `adv_patch_tokens`: `[B, K_src, D]`
    - `target_bank`: `[B, K_tgt, D]`

    Formula:
    `- mean_i log sum_j exp(cos(P_l z_adv_i, P_l z_tar_j) / tau)`

    NOTE(fas2h): the target bank must be detached. This keeps the target-side
    evidence fixed and ensures optimization only flows through `x_adv`.
    NOTE(fas2h): log-sum-exp softly aggregates over multiple target evidence
    tokens, which is less brittle than hard nearest-neighbor matching.
    NOTE(fas2h): this is still a shallow-token injection loss rather than a
    direct final/global feature alignment objective.
    """
    projected_adv = projection.project(layer, adv_patch_tokens)
    projected_bank = projection.project(layer, target_bank.detach())
    adv_norm = F.normalize(projected_adv, dim=-1)
    bank_norm = F.normalize(projected_bank, dim=-1)
    logits = adv_norm @ bank_norm.transpose(-1, -2)
    logits = logits / tau
    return -torch.logsumexp(logits, dim=-1).mean()


def carrier_target_similarity(
    adv_patch_tokens: torch.Tensor,
    target_bank: torch.Tensor,
    projection,
    layer: int,
) -> torch.Tensor:
    """Compute mean of per-carrier max cosine similarity to target bank."""
    projected_adv = projection.project(layer, adv_patch_tokens)
    projected_bank = projection.project(layer, target_bank.detach())
    adv_norm = F.normalize(projected_adv, dim=-1)
    bank_norm = F.normalize(projected_bank, dim=-1)
    sims = adv_norm @ bank_norm.transpose(-1, -2)
    return sims.max(dim=-1).values.mean()


def _ensure_route_weight_sum_one(eta_attn: float, eta_cls: float, eta_rollout: float) -> tuple[float, float, float]:
    """Validate route weights and return them as floats."""
    total = float(eta_attn) + float(eta_cls) + float(eta_rollout)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "Route weights must sum to 1.0 exactly for reproducible objective semantics: "
            f"eta_attn={eta_attn}, eta_cls={eta_cls}, eta_rollout={eta_rollout}, sum={total}"
        )
    return float(eta_attn), float(eta_cls), float(eta_rollout)


def _mean_selected_cls_attention(
    attention: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    """Mean CLS-to-selected-patch attention score for one layer."""
    if attention.dim() == 4:
        cls_to_patch = attention.mean(dim=1)[:, 0, 1:]
    elif attention.dim() == 3:
        cls_to_patch = attention[:, 0, 1:]
    else:
        raise ValueError(f"Expected attention rank 3 or 4, got {tuple(attention.shape)}")
    selected = torch.gather(cls_to_patch, dim=1, index=selected_indices.long())
    return selected.mean()


def _mean_selected_cls_coupling(
    patch_tokens: torch.Tensor,
    cls_tokens: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    """Mean cosine coupling between selected patch tokens and shallow CLS token."""
    gather_index = selected_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1]).long()
    selected_patch = torch.gather(patch_tokens, dim=1, index=gather_index)
    cls_expanded = cls_tokens.unsqueeze(1).expand_as(selected_patch)
    return F.cosine_similarity(selected_patch, cls_expanded, dim=-1).mean()


def _mean_selected_rollout(
    feature_bundle: FeatureBundle,
    layer: int,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    """Mean rollout proxy from selected shallow carriers to final CLS."""
    selected_rollout = cls_to_selected_patch_rollout(
        attentions_by_layer=feature_bundle.attentions_by_layer,
        layer=layer,
        num_layers=feature_bundle.num_layers,
        selected_indices=selected_indices,
    )
    return selected_rollout.mean()


def route_loss(
    adv_feature_bundles: dict[str, FeatureBundle],
    source_routes: dict[str, dict[int, RouteBundle]],
    layers: list[int],
    eta_attn: float,
    eta_cls: float,
    eta_rollout: float,
) -> dict[str, object]:
    """Compute shortcut route amplification loss and diagnostics.

    Formula:
    `L_route = - mean_{m,l,i}(eta_attn*A + eta_cls*cos + eta_rollout*rollout)`
    where all terms are computed from current `x_adv` features.
    """
    w_attn, w_cls, w_rollout = _ensure_route_weight_sum_one(eta_attn, eta_cls, eta_rollout)
    layer_losses: list[torch.Tensor] = []
    attn_scores: list[torch.Tensor] = []
    cls_scores: list[torch.Tensor] = []
    rollout_scores: list[torch.Tensor] = []
    per_model_loss_route: dict[str, torch.Tensor] = {}
    per_layer_loss_route: dict[str, dict[str, torch.Tensor]] = {}

    for model_name, feature_bundle in adv_feature_bundles.items():
        if model_name not in source_routes:
            raise KeyError(f"Missing source route bundles for model {model_name}")
        model_layer_losses: list[torch.Tensor] = []
        per_layer_loss_route[model_name] = {}
        for layer in layers:
            if layer not in source_routes[model_name]:
                raise KeyError(f"Missing source route for model={model_name}, layer={layer}")
            route_bundle = source_routes[model_name][layer]
            source_indices = route_bundle.source_indices
            if source_indices is None:
                raise ValueError(f"source_indices is required for model={model_name}, layer={layer}")
            if layer not in feature_bundle.patch_tokens_by_layer:
                raise KeyError(f"Missing patch tokens for model={model_name}, layer={layer}")
            if layer not in feature_bundle.cls_tokens_by_layer:
                raise KeyError(f"Missing cls tokens for model={model_name}, layer={layer}")
            if layer not in feature_bundle.attentions_by_layer:
                raise KeyError(
                    f"Missing attention for model={model_name}, layer={layer}. "
                    "Route loss requires attention extraction on current x_adv."
                )

            attn_score = _mean_selected_cls_attention(
                attention=feature_bundle.attentions_by_layer[layer],
                selected_indices=source_indices,
            )
            cls_score = _mean_selected_cls_coupling(
                patch_tokens=feature_bundle.patch_tokens_by_layer[layer],
                cls_tokens=feature_bundle.cls_tokens_by_layer[layer],
                selected_indices=source_indices,
            )
            rollout_score = _mean_selected_rollout(
                feature_bundle=feature_bundle,
                layer=layer,
                selected_indices=source_indices,
            )
            layer_score = (w_attn * attn_score) + (w_cls * cls_score) + (w_rollout * rollout_score)
            layer_loss = -layer_score

            layer_losses.append(layer_loss)
            model_layer_losses.append(layer_loss)
            attn_scores.append(attn_score)
            cls_scores.append(cls_score)
            rollout_scores.append(rollout_score)
            per_layer_loss_route[model_name][f"layer_{layer}"] = layer_loss

        if not model_layer_losses:
            raise RuntimeError(f"No route loss terms were produced for model={model_name}")
        per_model_loss_route[model_name] = torch.stack(model_layer_losses).mean()

    if not layer_losses:
        raise RuntimeError("No route loss terms were produced for the ensemble.")

    return {
        "loss_route": torch.stack(layer_losses).mean(),
        "route_attn_score": torch.stack(attn_scores).mean() if attn_scores else None,
        "route_cls_coupling_score": torch.stack(cls_scores).mean() if cls_scores else None,
        "route_rollout_score": torch.stack(rollout_scores).mean() if rollout_scores else None,
        "per_model_loss_route": per_model_loss_route,
        "per_layer_loss_route": per_layer_loss_route,
    }


def anchor_loss(*args, **kwargs) -> torch.Tensor:
    """Placeholder for future mid-layer anchor loss.

    TODO(fas2h): mid-layer semantic anchor is intentionally disabled in the MVP.
    """
    raise NotImplementedError("Anchor loss is not active in the MVP.")
