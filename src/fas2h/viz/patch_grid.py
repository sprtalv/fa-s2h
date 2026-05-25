"""Utilities for patch-grid metadata inference."""

from __future__ import annotations

import math
from typing import Optional, Tuple


def infer_patch_grid(num_patches: int) -> Optional[Tuple[int, int]]:
    """Infer patch grid (H, W) if number of patches is a perfect square.

    NOTE(fas2h): this helper supports common ViT patch layouts such as 14x14 (196 patches).
    """
    if num_patches <= 0:
        return None
    side = int(math.isqrt(num_patches))
    if side * side != num_patches:
        return None
    return side, side
