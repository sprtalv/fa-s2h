#!/usr/bin/env python3
"""Evaluation entrypoint placeholder for FA-S2H MVP."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> int:
    """Print the current evaluation status.

    TODO(fas2h): black-box evaluation is intentionally outside the MVP boundary.
    """
    del cfg
    print("Black-box eval is not implemented in MVP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
