"""Configuration helpers for loading YAML/Hydra-style config files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load config file into a plain dictionary.

    NOTE(fas2h): this helper exists for non-Hydra utility paths; main scripts now use Hydra composition.
    TODO(fas2h): switch scripts to unified Hydra entrypoint for overrides/sweeps.
    """
    cfg = OmegaConf.load(Path(config_path))
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
