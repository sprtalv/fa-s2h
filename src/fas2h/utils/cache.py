"""Filesystem cache helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save JSON payload with UTF-8 encoding."""
    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON payload from file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_pt(path: str | Path, tensor_obj: Any) -> None:
    """Save torch object to disk."""
    import torch

    out = Path(path)
    ensure_dir(out.parent)
    torch.save(tensor_obj, out)


def load_pt(path: str | Path) -> Any:
    """Load torch object from disk (CPU map location)."""
    import torch

    return torch.load(Path(path), map_location="cpu")
