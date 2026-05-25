#!/usr/bin/env python3
"""Stage-1 ablation entrypoint (placeholder)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# NOTE(fas2h): allow running script directly from repo root without editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fas2h.utils.config import load_config


def resolve_config_path(config_dir: Path, config_name: str) -> Path:
    return config_dir / f"{config_name}.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FA-S2H ablation (Stage 1 placeholder).")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--config-name", type=str, default="attack/fas2h_skeleton")
    args = parser.parse_args()

    cfg_path = resolve_config_path(args.config_dir, args.config_name)
    cfg = load_config(cfg_path)

    print("[Stage 1] run_ablation placeholder")
    print(f"config={cfg_path}")
    print(f"experiment={cfg.get('experiment', {}).get('name')}")
    print("pipeline=iterate ablation configs -> precompute route -> run attack -> run eval")
    print("TODO(fas2h): implement ablation sweep orchestration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
