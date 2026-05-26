#!/usr/bin/env python3
"""Print a concise summary for one experiment directory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for experiment summarization."""
    parser = argparse.ArgumentParser(description="Summarize metadata and lightweight outputs for one experiment.")
    parser.add_argument("--exp", type=Path, required=True, help="Path to the experiment directory.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root. Intended for testing; defaults to the current repo root.",
    )
    return parser.parse_args()


def read_text_if_exists(path: Path) -> str:
    """Return file contents or an empty string."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def read_json_if_exists(path: Path) -> dict[str, Any]:
    """Return decoded JSON or an empty dict."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def last_metrics_row(path: Path) -> dict[str, str] | None:
    """Return the final row from a CSV metrics file."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return rows[-1] if rows else None


def log_file_count(path: Path) -> int:
    """Count regular files inside a log directory."""
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for child in path.rglob("*") if child.is_file())


def jsonl_line_count(path: Path) -> int:
    """Count lines in a JSONL file without printing its content."""
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> int:
    """Print a concise experiment summary."""
    args = parse_args()
    repo_root = args.root.resolve()
    exp_dir = args.exp.resolve() if args.exp.is_absolute() else (repo_root / args.exp).resolve()

    if not exp_dir.exists() or not exp_dir.is_dir():
        print(f"ERROR: experiment directory does not exist: {args.exp}", file=sys.stderr)
        return 2

    print(f"Experiment: {exp_dir.name}")
    print(f"Path: {exp_dir}")

    git_commit = read_text_if_exists(exp_dir / "git_commit.txt") or "unknown"
    data_version = read_text_if_exists(exp_dir / "data_version.txt") or "missing"
    readme_text = read_text_if_exists(exp_dir / "README.md")
    failures_text = read_text_if_exists(exp_dir / "failures.md")
    summary = read_json_if_exists(exp_dir / "summary.json")

    print(f"Git commit: {git_commit}")
    print("Data version:")
    print(data_version)
    print(f"Summary status: {summary.get('status', 'unknown')}")
    print(f"Summary metrics: {summary.get('metrics', {})}")
    print(f"Summary result: {summary.get('result_summary', '')}")
    command_text = read_text_if_exists(exp_dir / "command.sh")
    if command_text:
        print("Command preview:")
        print("\n".join(command_text.splitlines()[:8]))
    logs_dir = exp_dir / "logs"
    print(f"Logs: {logs_dir} ({log_file_count(logs_dir)} files)")

    if readme_text:
        print("README preview:")
        print("\n".join(readme_text.splitlines()[:12]))
    if failures_text:
        print("Failure preview:")
        print("\n".join(failures_text.splitlines()[:12]))

    metrics_row = last_metrics_row(exp_dir / "metrics.csv")
    if metrics_row is not None:
        print(f"metrics.csv last row: {metrics_row}")

    for filename in ("captions.jsonl", "judge_scores.jsonl"):
        file_path = exp_dir / filename
        if file_path.exists():
            print(f"{filename}: present ({jsonl_line_count(file_path)} lines)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
