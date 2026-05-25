"""OpenCLIP ViT wrapper with real attention extraction for FA-S2H MVP.

This module does not modify installed site-packages. Instead, it reproduces the
VisionTransformer forward pass locally so that real attention weights can be
captured from each `ResidualAttentionBlock`. This is necessary because the
default OpenCLIP implementation calls `MultiheadAttention(..., need_weights=False)`.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from fas2h.features import FeatureBundle
from fas2h.models.wrappers.base import BaseVisionWrapper
from fas2h.viz.patch_grid import infer_patch_grid


class OpenCLIPVisionWrapper(BaseVisionWrapper):
    """Frozen OpenCLIP vision wrapper with shallow token and attention extraction."""

    def __init__(self, name: str, model_name: str, pretrained: str, device: str = "cpu") -> None:
        super().__init__(name=name, model_name=model_name, pretrained=pretrained, device=device)
        self.model = None
        self.visual = None
        self.image_size: tuple[int, int] = (224, 224)
        self.image_mean = (0.48145466, 0.4578275, 0.40821073)
        self.image_std = (0.26862954, 0.26130258, 0.27577711)
        self.num_layers = 0

    def load(self) -> None:
        """Load the exact OpenCLIP model/checkpoint and freeze parameters."""
        if self.model is not None:
            return

        try:
            import open_clip
            from open_clip.transformer import VisionTransformer
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is required for OpenCLIPVisionWrapper."
            ) from exc

        model, _, _ = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        model = model.to(self.device).eval()
        for param in model.parameters():
            param.requires_grad_(False)

        visual = model.visual
        if not isinstance(visual, VisionTransformer):
            raise TypeError(
                f"FA-S2H MVP currently expects VisionTransformer backbones, got {type(visual)!r} "
                f"for model {self.model_name}/{self.pretrained}."
            )

        self.model = model
        self.visual = visual
        self.dtype = next(model.parameters()).dtype
        image_size = getattr(visual, "image_size", (224, 224))
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = tuple(image_size)
        self.image_mean = tuple(getattr(visual, "image_mean", self.image_mean))
        self.image_std = tuple(getattr(visual, "image_std", self.image_std))
        self.num_layers = len(visual.transformer.resblocks)

    def preprocess_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """Resize and normalize `[0, 1]` image tensors for OpenCLIP.

        FA-S2H keeps perturbations in pixel space `[0, 1]`.
        OpenCLIP normalization is applied only inside this wrapper.
        """
        self.load()
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected image tensor shape [B, 3, H, W], got {tuple(images.shape)}")

        images = images.to(device=self.device, dtype=self.dtype)
        if tuple(images.shape[-2:]) != self.image_size:
            images = F.interpolate(
                images,
                size=self.image_size,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

        mean = torch.tensor(self.image_mean, device=self.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.image_std, device=self.device, dtype=images.dtype).view(1, 3, 1, 1)
        return (images - mean) / std

    def _run_block_with_attn(
        self,
        block: torch.nn.Module,
        x: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one residual attention block and return output plus real attention weights."""
        q_x = block.ln_1(x)
        attn_out, attn_weights = block.attn(
            q_x,
            q_x,
            q_x,
            need_weights=True,
            average_attn_weights=False,
        )
        if attn_weights is None:
            raise RuntimeError(
                f"Attention extraction failed for {self.model_name}/{self.pretrained} at layer {layer_idx}: "
                "MultiheadAttention returned no weights."
            )

        x = x + block.ls_1(attn_out)
        x = x + block.ls_2(block.mlp(block.ln_2(x)))

        if attn_weights.dim() != 4:
            raise RuntimeError(
                f"Expected attention tensor [B, H, T, T] at layer {layer_idx}, got {tuple(attn_weights.shape)}"
            )
        return x, attn_weights

    def _run_block_without_attn(self, block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run one residual attention block without requesting attention weights."""
        return block(x)

    def encode_with_features(
        self,
        images: torch.Tensor,
        layers: Iterable[int],
        capture_attn: bool = True,
    ) -> FeatureBundle:
        """Encode images and collect shallow tokens, final global features, and attention weights."""
        self.load()
        assert self.visual is not None

        requested_layers = sorted(set(int(layer) for layer in layers))
        if not requested_layers:
            raise ValueError("At least one layer must be requested.")
        if requested_layers[0] < 0 or requested_layers[-1] >= self.num_layers:
            raise ValueError(
                f"Requested layers {requested_layers} are out of range for a {self.num_layers}-layer model."
            )

        x = self.visual._embeds(self.preprocess_tensor(images))
        patch_tokens_by_layer: dict[int, torch.Tensor] = {}
        cls_tokens_by_layer: dict[int, torch.Tensor] = {}
        attentions_by_layer: dict[int, torch.Tensor] = {}

        for layer_idx, block in enumerate(self.visual.transformer.resblocks):
            if capture_attn:
                x, attn_weights = self._run_block_with_attn(block, x, layer_idx)
                attentions_by_layer[layer_idx] = attn_weights
            else:
                x = self._run_block_without_attn(block, x)

            if layer_idx in requested_layers:
                cls_tokens_by_layer[layer_idx] = x[:, 0, :]
                patch_tokens_by_layer[layer_idx] = x[:, 1:, :]

        # NOTE(fas2h): the MVP uses the visual-tower pooled feature before the
        # final CLIP projection so that identity projection can be applied to
        # shallow patch tokens and detached global guidance in the same space.
        pooled, _ = self.visual._pool(x)

        any_patch_tokens = next(iter(patch_tokens_by_layer.values()))
        num_patches = any_patch_tokens.shape[1]
        patch_grid_tuple = infer_patch_grid(num_patches)
        patch_grid = None
        if patch_grid_tuple is not None:
            patch_grid = {
                "height": patch_grid_tuple[0],
                "width": patch_grid_tuple[1],
                "num_patches": num_patches,
            }

        return FeatureBundle(
            patch_tokens_by_layer=patch_tokens_by_layer,
            cls_tokens_by_layer=cls_tokens_by_layer,
            attentions_by_layer=attentions_by_layer,
            final_global=pooled,
            num_layers=self.num_layers,
            num_patches=num_patches,
            patch_grid=patch_grid,
            model_meta={
                "wrapper": "OpenCLIPVisionWrapper",
                "name": self.name,
                "model_name": self.model_name,
                "pretrained": self.pretrained,
                "device": str(self.device),
                "dtype": str(self.dtype),
                "image_size": list(self.image_size),
            },
        )
