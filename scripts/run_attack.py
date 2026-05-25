#!/usr/bin/env python3
"""Hydra entrypoint for the FA-S2H MVP attack."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fas2h.attacks.fas2h import FAS2HAttack
from fas2h.models.registry import build_surrogate_wrappers
from fas2h.utils.logging import get_logger
from fas2h.utils.seed import set_seed


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> int:
    """Run FA-S2H MVP on configured JSONL pairs."""
    logger = get_logger("fas2h.run_attack")
    set_seed(int(cfg.runtime.seed))
    wrappers = build_surrogate_wrappers(cfg, device=str(cfg.runtime.device))
    attack = FAS2HAttack(cfg=cfg, models=wrappers)
    results = attack.run()
    logger.info("Completed %d pair(s).", len(results))
    for result in results:
        logger.info("pair_id=%s final_loss=%s output=%s", result["pair_id"], result["final_loss"], result["pair_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
