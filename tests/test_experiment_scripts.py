from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def make_repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "configs" / "attack").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "experiments" / "active").mkdir(parents=True)
    (root / "experiments" / "archive").mkdir(parents=True)
    (root / "experiments" / "graveyard").mkdir(parents=True)
    (root / "configs" / "attack" / "fas2h.yaml").write_text("name: fas2h\n", encoding="utf-8")
    return root


def run_script(script_name: str, repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    return subprocess.run(
        [sys.executable, str(script_path), "--root", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_new_exp_creates_expected_files(tmp_path: Path) -> None:
    repo_root = make_repo_root(tmp_path)

    result = run_script(
        "new_exp.py",
        repo_root,
        "--name",
        "fas2h_clip3_topk32_seed42",
        "--config",
        "configs/attack/fas2h.yaml",
        "--data-version",
        "toy-v1",
        "--seed",
        "42",
    )

    exp_dir = Path(result.stdout.strip())
    assert exp_dir.exists()
    assert (exp_dir / "config.yaml").exists()
    assert (exp_dir / "command.sh").exists()
    assert (exp_dir / "git_commit.txt").read_text(encoding="utf-8").strip() == "unknown"
    assert "data_version: toy-v1" in (exp_dir / "data_version.txt").read_text(encoding="utf-8")

    summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "active"
    assert summary["seed"] == 42


def test_archive_exp_moves_directory(tmp_path: Path) -> None:
    repo_root = make_repo_root(tmp_path)
    create_result = run_script(
        "new_exp.py",
        repo_root,
        "--name",
        "fas2h_archive_case",
        "--config",
        "configs/attack/fas2h.yaml",
        "--data-version",
        "toy-v1",
        "--seed",
        "7",
    )
    exp_dir = Path(create_result.stdout.strip())

    move_result = run_script(
        "archive_exp.py",
        repo_root,
        "--exp",
        str(exp_dir),
        "--to",
        "graveyard",
        "--reason",
        "transfer failed",
    )

    destination = Path(move_result.stdout.strip())
    assert destination.exists()
    assert not exp_dir.exists()
    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "graveyard"
    assert summary["failure_reason"] == "transfer failed"
    assert "Archive Note" in (destination / "README.md").read_text(encoding="utf-8")


def test_find_exp_reports_matching_experiment(tmp_path: Path) -> None:
    repo_root = make_repo_root(tmp_path)
    create_result = run_script(
        "new_exp.py",
        repo_root,
        "--name",
        "fas2h_find_case",
        "--config",
        "configs/attack/fas2h.yaml",
        "--data-version",
        "toy-v1",
        "--seed",
        "11",
    )
    exp_dir = Path(create_result.stdout.strip())
    summary_path = exp_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics"] = {"asr": 0.4}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = run_script("find_exp.py", repo_root, "--keyword", "fas2h_find", "--metric", "asr")

    assert exp_dir.name in result.stdout
    assert "asr=0.4" in result.stdout
