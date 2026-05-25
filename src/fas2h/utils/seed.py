"""Random seed utilities."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed across Python, NumPy, and torch when available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        # NOTE(fas2h): allow seed utility import even without torch installed.
        pass
