#!/usr/bin/env python3
"""Search lightweight experiment metadata across active, archive, and graveyard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
STATUSES = ("active", "archive", "graveyard")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for experiment search."""
    parser = argparse.ArgumentParser(description="Find experiments by keyword, date prefix, status, or metric key.")
    parser.add_argument("--keyword", help="Substring match against experiment directory name.")
    parser.add_argument("--date", help="Date prefix in YYYYMMDD format.")
    parser.add_argument("--status", choices=STATUSES, help="Restrict search to one experiment status.")
    parser.add_argument("--metric", help="Metric key to print from summary.json if present.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root. Intended for testing; defaults to the current repo root.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    """Load summary.json when present and valid."""
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def iter_experiments(root: Path, status_filter: str | None) -> list[tuple[str, Path]]:
    """List experiments from the configured status directories."""
    statuses = (status_filter,) if status_filter else STATUSES
    results: list[tuple[str, Path]] = []
    for status in statuses:
        status_dir = root / "experiments" / status
        if not status_dir.exists():
            continue
        for child in sorted(status_dir.iterdir()):
            if child.is_dir() and child.name != "_template":
                results.append((status, child))
    return results


def matches(exp_name: str, keyword: str | None, date_prefix: str | None) -> bool:
    """Apply name and date filters."""
    if keyword and keyword not in exp_name:
        return False
    if date_prefix and not exp_name.startswith(date_prefix):
        return False
    return True


def format_metric(summary: dict[str, Any], metric_name: str | None) -> str:
    """Format metrics output for the search result line."""
    metrics = summary.get("metrics", {})
    if metric_name:
        if metric_name in metrics:
            return f"{metric_name}={metrics[metric_name]}"
        return f"{metric_name}=N/A"
    if isinstance(metrics, dict) and metrics:
        preview = ", ".join(f"{key}={value}" for key, value in sorted(metrics.items())[:3])
        return preview
    return "metrics=none"


def main() -> int:
    """Search experiments and print concise result lines."""
    args = parse_args()
    repo_root = args.root.resolve()

    rows = []
    for status, exp_dir in iter_experiments(repo_root, args.status):
        if not matches(exp_dir.name, args.keyword, args.date):
            continue
        summary = load_summary(exp_dir)
        rows.append((status, exp_dir, summary))

    for status, exp_dir, summary in rows:
        metric_text = format_metric(summary, args.metric)
        print(f"[{status}] {exp_dir.name} :: {metric_text}")

    if not rows:
        print("No experiments matched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
