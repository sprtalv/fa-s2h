#!/usr/bin/env python3
"""Verify exact OpenCLIP surrogate model/checkpoint availability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_model_cfg(config_path: Path) -> dict[str, Any]:
    """Load model config as a plain dictionary."""
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def extract_surrogates(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract surrogate entries from either direct or composed model config."""
    if "model" in cfg:
        return cfg["model"].get("surrogates", [])
    return cfg.get("surrogates", [])


def build_pretrained_index(pretrained_pairs: list[Any]) -> dict[str, set[str]]:
    """Build `model_name -> pretrained_tags` index from `open_clip.list_pretrained()`."""
    index: dict[str, set[str]] = {}
    for item in pretrained_pairs:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            model_name = str(item[0])
            pretrained = str(item[1])
            index.setdefault(model_name, set()).add(pretrained)
    return index


def main() -> int:
    """Run model availability and load checks."""
    parser = argparse.ArgumentParser(description="Validate and load OpenCLIP surrogate models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/clip_3surrogate.yaml"),
        help="Path to surrogate model config YAML.",
    )
    args = parser.parse_args()

    try:
        import open_clip
    except ImportError as exc:
        print("ERROR: open_clip_torch is not installed or failed to import.")
        print(f"Details: {exc}")
        return 2

    cfg = load_model_cfg(args.config)
    surrogates = extract_surrogates(cfg)
    if not surrogates:
        print(f"ERROR: no surrogates found in config: {args.config}")
        return 2

    available = open_clip.list_pretrained()
    available_index = build_pretrained_index(available)
    print(f"Loaded {len(available)} OpenCLIP pretrained entries from current open_clip version.")
    print(f"Checking {len(surrogates)} configured surrogates...")

    all_ok = True
    for item in surrogates:
        name = item["name"]
        model_name = item["model_name"]
        pretrained = item["pretrained"]
        print("-" * 80)
        print(f"name={name} | model_name={model_name} | pretrained={pretrained}")

        candidates = sorted(available_index.get(model_name, set()))
        if pretrained not in available_index.get(model_name, set()):
            all_ok = False
            print("status=FAILED (not found in open_clip.list_pretrained())")
            print(f"available_pretrained_for_{model_name}={candidates or '<none>'}")
            continue

        try:
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            print("status=SUCCESS (model load/create passed)")
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            print(f"status=FAILED (load error: {exc})")
            print(f"available_pretrained_for_{model_name}={candidates or '<none>'}")

    print("-" * 80)
    if not all_ok:
        print("Result: FAILED. Some exact model/pretrained pairs are unavailable or failed to load.")
        return 1
    print("Result: SUCCESS. All configured surrogates are available and loadable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
