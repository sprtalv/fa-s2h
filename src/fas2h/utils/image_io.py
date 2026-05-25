"""Image loading and saving utilities for FA-S2H."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fas2h.utils.io import ensure_parent_dir


def load_image_tensor(path: str | Path) -> torch.Tensor:
    """Load an RGB image and return a tensor in `[0, 1]` with shape `[3, H, W]`."""
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_image_tensor(image: torch.Tensor, path: str | Path) -> None:
    """Save a tensor image in `[0, 1]` to disk."""
    ensure_parent_dir(path)
    image = image.detach().cpu().clamp(0.0, 1.0)
    if image.dim() == 4:
        image = image[0]
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def save_perturbation_visualization(delta: torch.Tensor, path: str | Path, eps: float) -> None:
    """Save a simple perturbation visualization in `[0, 1]`.

    This is SMOKE_TEST-style visualization for debugging and is not a paper-level
    analysis artifact.
    """
    scaled = (delta.detach().cpu() / max(eps, 1e-8)) * 0.5 + 0.5
    save_image_tensor(scaled.clamp(0.0, 1.0), path)
