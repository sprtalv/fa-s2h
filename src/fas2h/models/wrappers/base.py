"""Base wrapper interfaces for frozen vision surrogates used by FA-S2H."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import torch

from fas2h.features import FeatureBundle


class BaseVisionWrapper(ABC):
    """Abstract interface for a frozen surrogate vision encoder.

    Public methods are intentionally narrow:
    - `load()` prepares the exact model/checkpoint.
    - `preprocess_tensor()` accepts image tensors in `[0, 1]`.
    - `encode_with_features()` returns shallow tokens, final global features,
      and optionally real attention weights for route selection.
    """

    def __init__(self, name: str, model_name: str, pretrained: str, device: str = "cpu") -> None:
        self.name = name
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = torch.device(device)
        self.dtype = torch.float32

    @abstractmethod
    def load(self) -> None:
        """Load the exact surrogate weights and freeze the model."""

    @abstractmethod
    def preprocess_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """Resize and normalize image tensors in `[0, 1]` for the backend model.

        Args:
        - `images`: tensor of shape `[B, 3, H, W]` or `[3, H, W]`.

        Returns:
        - normalized tensor of shape `[B, 3, H_model, W_model]`.
        """

    @abstractmethod
    def encode_with_features(
        self,
        images: torch.Tensor,
        layers: Iterable[int],
        capture_attn: bool = True,
    ) -> FeatureBundle:
        """Encode images and return shallow tokens, final global feature, and attentions.

        Args:
        - `images`: tensor in `[0, 1]`.
        - `layers`: 0-based transformer layers to expose in the `FeatureBundle`.
        - `capture_attn`: whether to return real attention weights.
          The MVP uses `True` for clean source/target route selection and `False`
          during PGD steps to save memory because `L_inj` does not need attention.
        """

    @property
    def model_device(self) -> torch.device:
        """Return the effective device used by the wrapped model."""
        return self.device

    @property
    def model_dtype(self) -> torch.dtype:
        """Return the floating-point dtype used by the wrapped model."""
        return self.dtype
