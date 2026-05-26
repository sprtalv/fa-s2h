"""Loss functions used by the FA-S2H MVP attack."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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


def route_loss(*args, **kwargs) -> torch.Tensor:
    """Placeholder for future route amplification loss.

    TODO(fas2h): route amplification loss is intentionally disabled in the MVP.
    """
    raise NotImplementedError("Route amplification loss is not active in the MVP.")


def anchor_loss(*args, **kwargs) -> torch.Tensor:
    """Placeholder for future mid-layer anchor loss.

    TODO(fas2h): mid-layer semantic anchor is intentionally disabled in the MVP.
    """
    raise NotImplementedError("Anchor loss is not active in the MVP.")
