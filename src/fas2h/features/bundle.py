"""Dataclasses used by the FA-S2H MVP pipeline.

The current repository targets a research MVP rather than a final paper release.
The bundles below are used to make tensor semantics explicit across model wrappers,
route selection, target bank construction, and PGD optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class FeatureBundle:
    """Feature container returned by a vision surrogate wrapper.

    Fields:
    - `patch_tokens_by_layer[layer]`: tensor of shape `[B, N, D]`.
      `N` is the number of patch tokens for the image encoder.
    - `cls_tokens_by_layer[layer]`: tensor of shape `[B, D]`.
    - `attentions_by_layer[layer]`: tensor of shape `[B, H, N+1, N+1]`.
      Index `0` is the CLS token, so patch `i` lives at attention index `i + 1`.
    - `final_global`: pooled global image feature of shape `[B, D_out]`.
      This is used only as detached semantic guidance in the MVP.
    - `num_layers`: number of transformer blocks in the visual encoder.
    - `num_patches`: number of patch tokens `N`.
    - `patch_grid`: optional metadata such as `{"height": 14, "width": 14}`.
    - `model_meta`: metadata describing the surrogate wrapper and exact checkpoint.
    """

    patch_tokens_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    cls_tokens_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    attentions_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    final_global: Optional[torch.Tensor] = None
    num_layers: int = 0
    num_patches: int = 0
    patch_grid: Optional[dict[str, Any]] = None
    model_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def attn_by_layer(self) -> dict[int, torch.Tensor]:
        """Backward-compatible alias used by earlier Stage-1 modules."""
        return self.attentions_by_layer


@dataclass
class RouteBundle:
    """Selection results for one `(pair, model, layer)` route component.

    Fields:
    - `source_indices`: source carrier indices of shape `[B, K_src]` or `None`.
    - `source_scores`: source carrier scores of shape `[B, K_src]` or `None`.
    - `target_indices`: target evidence indices of shape `[B, K_tgt]` or `None`.
    - `target_scores`: target evidence scores of shape `[B, K_tgt]` or `None`.
    - `rollout_scores`: rollout-to-patch scores of shape `[B, N]` or `None`.
    - `layer`: 0-based transformer layer index.
    - `model_name`: surrogate model identifier.
    - `pretrained`: exact pretrained checkpoint tag.
    - `meta`: additional diagnostic data, including full score tensors when useful.
    """

    source_indices: Optional[torch.Tensor] = None
    source_scores: Optional[torch.Tensor] = None
    target_indices: Optional[torch.Tensor] = None
    target_scores: Optional[torch.Tensor] = None
    rollout_scores: Optional[torch.Tensor] = None
    layer: Optional[int] = None
    model_name: Optional[str] = None
    pretrained: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BankBundle:
    """Fixed target evidence bank for one `(pair, model, layer)`.

    Fields:
    - `target_bank`: detached tensor of shape `[B, K_tgt, D]`.
    - `target_indices`: selected target evidence indices of shape `[B, K_tgt]`.
    - `target_scores`: selected target evidence scores of shape `[B, K_tgt]`.
    - `layer`: 0-based transformer layer index.
    - `model_name`: surrogate model identifier.
    - `pretrained`: exact pretrained checkpoint tag.
    - `detached`: whether the stored bank has been detached from autograd.
    - `meta`: diagnostic data such as full target score tensors and rollout scores.
    """

    target_bank: Optional[torch.Tensor] = None
    target_indices: Optional[torch.Tensor] = None
    target_scores: Optional[torch.Tensor] = None
    layer: Optional[int] = None
    model_name: Optional[str] = None
    pretrained: Optional[str] = None
    detached: bool = True
    meta: dict[str, Any] = field(default_factory=dict)
