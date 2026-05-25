"""General IO helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_parent_dir(path: str | Path) -> Path:
    """Create parent directory for target file path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
