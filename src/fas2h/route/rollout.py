"""Attention rollout utilities for target evidence selection and route loss.

Rollout is used here as a proxy for downstream path contribution. It is not an
exact causal attribution method; the README documents this limitation explicitly.
"""

from __future__ import annotations

from typing import Mapping

import torch


def average_heads(attn: torch.Tensor) -> torch.Tensor:
    """Average attention over heads.

    Args:
    - `attn`: tensor of shape `[B, H, T, T]` or `[B, T, T]`.

    Returns:
    - tensor of shape `[B, T, T]`.
    """
    if attn.dim() == 4:
        return attn.mean(dim=1)
    if attn.dim() == 3:
        return attn
    raise ValueError(f"Expected attention tensor rank 3 or 4, got shape {tuple(attn.shape)}")


def add_residual_and_normalize(attn: torch.Tensor) -> torch.Tensor:
    """Apply `Norm(A + I)` with row normalization.

    Args:
    - `attn`: tensor of shape `[B, T, T]`.

    Returns:
    - row-normalized tensor of shape `[B, T, T]`.
    """
    if attn.dim() != 3:
        raise ValueError(f"Expected `[B, T, T]`, got shape {tuple(attn.shape)}")
    identity = torch.eye(attn.shape[-1], device=attn.device, dtype=attn.dtype).unsqueeze(0)
    attn = attn + identity
    denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return attn / denom


def compute_rollout(
    attentions_by_layer: Mapping[int, torch.Tensor],
    start_layer: int,
    end_layer: int,
) -> torch.Tensor:
    """Compute attention rollout between two 0-based layers, inclusive.

    Args:
    - `attentions_by_layer[layer]`: tensors shaped `[B, H, T, T]` or `[B, T, T]`.
    - `start_layer`: first layer used in the rollout product.
    - `end_layer`: last layer used in the rollout product.

    Returns:
    - rollout matrix of shape `[B, T, T]`.

    NOTE(fas2h): for target evidence scoring at shallow layer `l`, the MVP uses
    attention blocks from `l + 1` to the final layer.
    """
    if start_layer > end_layer:
        sample = average_heads(next(iter(attentions_by_layer.values())))
        identity = torch.eye(sample.shape[-1], device=sample.device, dtype=sample.dtype).unsqueeze(0)
        return identity.expand(sample.shape[0], -1, -1).clone()

    matrices = []
    for layer_idx in range(start_layer, end_layer + 1):
        if layer_idx not in attentions_by_layer:
            raise KeyError(f"Missing attention for layer {layer_idx} in rollout computation.")
        matrices.append(add_residual_and_normalize(average_heads(attentions_by_layer[layer_idx])))

    rollout = matrices[0]
    for matrix in matrices[1:]:
        rollout = rollout @ matrix
    return rollout


def extract_cls_to_patch_rollout(rollout: torch.Tensor) -> torch.Tensor:
    """Extract CLS-to-patch rollout scores.

    Args:
    - `rollout`: rollout matrix of shape `[B, T, T]`, where token `0` is CLS.

    Returns:
    - tensor of shape `[B, N]` with `N = T - 1`.
    """
    if rollout.dim() != 3:
        raise ValueError(f"Expected rollout shape `[B, T, T]`, got {tuple(rollout.shape)}")
    return rollout[:, 0, 1:]


def gather_selected_patch_scores(
    patch_scores: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather per-patch scores for selected patch indices.

    Args:
    - `patch_scores`: tensor `[B, N]`.
    - `selected_indices`: tensor `[B, K]` with patch-space indices in `[0, N)`.

    Returns:
    - tensor `[B, K]`.
    """
    if patch_scores.dim() != 2:
        raise ValueError(f"Expected patch_scores shape `[B, N]`, got {tuple(patch_scores.shape)}")
    if selected_indices.dim() != 2:
        raise ValueError(f"Expected selected_indices shape `[B, K]`, got {tuple(selected_indices.shape)}")
    if patch_scores.shape[0] != selected_indices.shape[0]:
        raise ValueError(
            "Batch size mismatch between patch_scores and selected_indices: "
            f"{patch_scores.shape[0]} vs {selected_indices.shape[0]}"
        )
    return torch.gather(patch_scores, dim=1, index=selected_indices.long())


def cls_to_selected_patch_rollout(
    attentions_by_layer: Mapping[int, torch.Tensor],
    layer: int,
    num_layers: int,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute rollout proxy `CLS <- selected patch` for one shallow layer.

    The rollout product uses blocks `[layer + 1, ..., num_layers - 1]`.
    The returned tensor preserves batch and selection dimensions.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    rollout_matrix = compute_rollout(
        attentions_by_layer=attentions_by_layer,
        start_layer=layer + 1,
        end_layer=num_layers - 1,
    )
    cls_to_patch = extract_cls_to_patch_rollout(rollout_matrix)
    return gather_selected_patch_scores(cls_to_patch, selected_indices=selected_indices)
