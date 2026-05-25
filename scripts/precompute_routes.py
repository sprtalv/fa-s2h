#!/usr/bin/env python3
"""Hydra entrypoint for precomputing source carriers and target banks."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fas2h.attacks.fas2h import FAS2HAttack
from fas2h.data.pairs import load_pairs_jsonl
from fas2h.models.registry import build_surrogate_wrappers
from fas2h.utils.logging import get_logger
from fas2h.utils.seed import set_seed


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> int:
    """Precompute and save route metadata without running PGD."""
    logger = get_logger("fas2h.precompute")
    set_seed(int(cfg.runtime.seed))
    wrappers = build_surrogate_wrappers(cfg, device=str(cfg.runtime.device))
    attack = FAS2HAttack(cfg=cfg, models=wrappers)
    run_dir = attack._resolve_output_dir(run_name=f"{cfg.attack.name}_precompute")
    OmegaConf.save(config=cfg, f=run_dir / "config_used.yaml")

    pairs = load_pairs_jsonl(cfg.data.pair_file)
    if cfg.data.limit is not None:
        pairs = pairs[: int(cfg.data.limit)]
    for pair in pairs:
        attack.precompute_pair(pair=pair, run_dir=run_dir)
        logger.info("Precomputed routes for pair_id=%s", pair["pair_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
