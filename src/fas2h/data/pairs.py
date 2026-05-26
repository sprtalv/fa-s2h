"""JSONL pair loader for FA-S2H experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("pair_id", "source_path", "target_path")


def resolve_data_path(path_text: str, base_dir: Path) -> Path:
    """Resolve a possibly relative image path.

    Resolution policy:
    - absolute paths are used directly;
    - relative paths are first resolved against the JSONL directory;
    - if that does not exist, they are resolved against the repo working directory.
    """
    path = Path(path_text)
    if path.is_absolute():
        return path

    jsonl_relative = (base_dir / path).resolve()
    if jsonl_relative.exists():
        return jsonl_relative
    return (Path.cwd() / path).resolve()


def load_pairs_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate source-target pairs from a JSONL file.

    Required fields:
    - `pair_id`
    - `source_path`
    - `target_path`

    Optional fields:
    - `target_keywords`

    NOTE(fas2h): `target_keywords` are metadata only for future evaluation and
    are intentionally not used by the MVP attack loss.
    """
    pair_file = Path(path)
    if not pair_file.exists():
        raise FileNotFoundError(f"Pair file not found: {pair_file}")

    rows: list[dict[str, Any]] = []
    with pair_file.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            for field in REQUIRED_FIELDS:
                if field not in sample:
                    raise ValueError(f"Missing required field '{field}' at line {line_idx} in {pair_file}")

            source_path = resolve_data_path(sample["source_path"], pair_file.parent)
            target_path = resolve_data_path(sample["target_path"], pair_file.parent)
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Source image for pair {sample['pair_id']} does not exist: {source_path}"
                )
            if not target_path.exists():
                raise FileNotFoundError(
                    f"Target image for pair {sample['pair_id']} does not exist: {target_path}"
                )

            normalized = dict(sample)
            normalized["source_path"] = str(source_path)
            normalized["target_path"] = str(target_path)
            rows.append(normalized)
    return rows
