#!/usr/bin/env python3
"""Create a lightweight experiment directory with reproducibility metadata."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
ACTIVE_DIR = EXPERIMENTS_DIR / "active"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for experiment creation."""
    parser = argparse.ArgumentParser(
        description="Create a new experiment folder under experiments/active with template metadata files.",
    )
    parser.add_argument("--name", required=True, help="Short experiment name, for example fas2h_clip3_topk32_seed42.")
    parser.add_argument("--config", required=True, help="Path to an existing config file that will be copied.")
    parser.add_argument("--data-version", default="unknown", help="Dataset snapshot, tag, or free-form version note.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for the experiment.")
    parser.add_argument(
        "--command",
        default="",
        help="Optional exact command to write into command.sh. If omitted, a placeholder command is generated.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root. Intended for testing; defaults to the current repo root.",
    )
    return parser.parse_args()


def current_timestamp() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def resolve_git_commit(repo_root: Path) -> str:
    """Return the current git commit id or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def default_command(config_path: Path, seed: int) -> str:
    """Build a simple placeholder command for the copied config."""
    rel_config = config_path.as_posix()
    return (
        "# TODO(fas2h): replace with the exact command if this template differs from the real run.\n"
        f"# Config copied from {rel_config}\n"
        f"python scripts/run_attack.py --config-name config runtime.seed={seed}"
    )


def write_text(path: Path, content: str) -> None:
    """Write a UTF-8 text file."""
    path.write_text(content, encoding="utf-8")


def main() -> int:
    """Create a new experiment directory and metadata files."""
    args = parse_args()
    repo_root = args.root.resolve()
    experiments_dir = repo_root / "experiments"
    active_dir = experiments_dir / "active"

    config_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    if not config_path.exists() or not config_path.is_file():
        print(f"ERROR: config file does not exist: {args.config}", file=sys.stderr)
        return 2
    try:
        config_path.relative_to(repo_root)
    except ValueError:
        print("ERROR: config file must be inside the repository root.", file=sys.stderr)
        return 2

    stamp = current_timestamp()
    exp_name = f"{stamp.strftime('%Y%m%d')}_{args.name}"
    exp_dir = active_dir / exp_name
    if exp_dir.exists():
        print(f"ERROR: experiment directory already exists: {exp_dir}", file=sys.stderr)
        return 2

    exp_dir.mkdir(parents=True, exist_ok=False)
    (exp_dir / "logs").mkdir()

    shutil.copy2(config_path, exp_dir / "config.yaml")

    command_body = args.command.strip() or default_command(config_path.relative_to(repo_root), args.seed)
    write_text(
        exp_dir / "command.sh",
        "#!/usr/bin/env bash\n"
        "# Record the exact command used for this experiment.\n\n"
        "set -euo pipefail\n\n"
        f"{command_body}\n",
    )

    git_commit = resolve_git_commit(repo_root)
    write_text(exp_dir / "git_commit.txt", f"{git_commit}\n")
    write_text(
        exp_dir / "data_version.txt",
        f"data_version: {args.data_version}\n"
        f"created_at_utc: {stamp.isoformat()}\n"
        f"git_commit: {git_commit}\n",
    )
    write_text(
        exp_dir / "README.md",
        "# Experiment Record\n\n"
        "## Purpose\n\n"
        "TODO(fas2h): describe the experiment goal.\n\n"
        "## Key Changes\n\n"
        "- TODO(fas2h): list config or code differences.\n\n"
        "## Current Conclusion\n\n"
        "- TODO(fas2h): summarize the current result.\n",
    )
    write_text(
        exp_dir / "failures.md",
        "# Failure Records\n\n"
        "- Date: `YYYY-MM-DD`\n"
        "- Stage: `setup | precompute | attack | eval | analysis`\n"
        "- Symptom: `TBD`\n"
        "- Suspected cause: `TBD`\n"
        "- Resolution or next step: `TODO(fas2h)`\n",
    )
    summary = {
        "status": "active",
        "archive": False,
        "deprecated": False,
        "seed": args.seed,
        "data_version": args.data_version,
        "config_source": str(config_path.relative_to(repo_root)),
        "command": command_body,
        "metrics": {},
        "result_summary": "",
        "failure_reason": "",
        "notes": ["TODO(fas2h): fill in key findings after the run."],
    }
    write_text(exp_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(exp_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
