#!/usr/bin/env python3
"""Move an experiment from active to archive or graveyard with an audit note."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
VALID_DESTINATIONS = {"archive", "graveyard"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for archive moves."""
    parser = argparse.ArgumentParser(description="Move an experiment into archive or graveyard.")
    parser.add_argument("--exp", type=Path, required=True, help="Path to the experiment directory to move.")
    parser.add_argument("--to", required=True, choices=sorted(VALID_DESTINATIONS), help="Destination status.")
    parser.add_argument("--reason", required=True, help="Short reason for archiving or discarding the experiment.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root. Intended for testing; defaults to the current repo root.",
    )
    return parser.parse_args()


def append_archive_note(exp_dir: Path, destination: str, reason: str) -> None:
    """Append archive metadata to README.md or a fallback note file."""
    timestamp = datetime.now(timezone.utc).isoformat()
    note = (
        "\n## Archive Note\n\n"
        f"- archived_at_utc: `{timestamp}`\n"
        f"- destination: `{destination}`\n"
        f"- reason: {reason}\n"
    )
    readme_path = exp_dir / "README.md"
    target_path = readme_path if readme_path.exists() else exp_dir / "archive_note.md"
    existing = target_path.read_text(encoding="utf-8") if target_path.exists() else "# Archive Note\n"
    if note.strip() not in existing:
        target_path.write_text(existing.rstrip() + note, encoding="utf-8")


def update_summary(exp_dir: Path, destination: str, reason: str) -> None:
    """Update summary.json if it exists and is valid JSON."""
    summary_path = exp_dir / "summary.json"
    if not summary_path.exists():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    summary["status"] = destination
    summary["archive"] = destination == "archive"
    summary["deprecated"] = destination == "graveyard"
    summary["failure_reason"] = reason if destination == "graveyard" else summary.get("failure_reason", "")
    summary["archive_reason"] = reason
    summary["archived_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    """Archive or discard an experiment directory without overwriting targets."""
    args = parse_args()
    repo_root = args.root.resolve()
    experiments_dir = repo_root / "experiments"
    exp_path = args.exp.resolve() if args.exp.is_absolute() else (repo_root / args.exp).resolve()

    if not exp_path.exists() or not exp_path.is_dir():
        print(f"ERROR: experiment directory does not exist: {args.exp}", file=sys.stderr)
        return 2
    try:
        exp_path.relative_to(experiments_dir)
    except ValueError:
        print("ERROR: experiment directory must be inside experiments/.", file=sys.stderr)
        return 2

    destination_dir = experiments_dir / args.to / exp_path.name
    if destination_dir.exists():
        print(f"ERROR: destination already exists: {destination_dir}", file=sys.stderr)
        return 2

    append_archive_note(exp_path, args.to, args.reason)
    update_summary(exp_path, args.to, args.reason)
    shutil.move(str(exp_path), str(destination_dir))
    print(destination_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
