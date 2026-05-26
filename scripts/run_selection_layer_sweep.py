#!/usr/bin/env python3
"""Run FA-S2H selection vs random_patch across shallow-layer sweeps."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SweepCase:
    method: str
    selection_mode: str
    shallow_layers: str


CASES = [
    SweepCase(method="fas2h_inj_l0123", selection_mode="fas2h", shallow_layers="[0,1,2,3]"),
    SweepCase(method="random_patch_inj_l0123", selection_mode="random_patch", shallow_layers="[0,1,2,3]"),
    SweepCase(method="fas2h_inj_l234", selection_mode="fas2h", shallow_layers="[2,3,4]"),
    SweepCase(method="random_patch_inj_l234", selection_mode="random_patch", shallow_layers="[2,3,4]"),
    SweepCase(method="fas2h_inj_l345", selection_mode="fas2h", shallow_layers="[3,4,5]"),
    SweepCase(method="random_patch_inj_l345", selection_mode="random_patch", shallow_layers="[3,4,5]"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run six diagnostic FA-S2H sweep combinations.")
    parser.add_argument("--output-root", default="outputs/runs_selection_layer_sweep", help="Run output root.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-selection-seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--eps", type=float, default=0.062745098)
    parser.add_argument("--step-size", type=float, default=0.0039215686)
    parser.add_argument("--topk-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child runs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    failures: list[tuple[str, int]] = []
    completed: list[str] = []

    for case in CASES:
        cmd = [
            args.python,
            str(project_root / "scripts" / "run_attack.py"),
            "--config-name",
            "config",
            "attack=fas2h_mvp",
            "data=resources_pairs_1000",
            f"runtime.output_dir={args.output_root}",
            f"runtime.seed={args.seed}",
            f"runtime.random_selection_seed={args.random_selection_seed}",
            f"runtime.device={args.device}",
            "runtime.progress=true",
            f"data.limit={args.limit}",
            f"attack.selection_mode={case.selection_mode}",
            f"attack.shallow_layers={case.shallow_layers}",
            f"attack.topk_ratio={args.topk_ratio}",
            f"attack.pgd.steps={args.steps}",
            f"attack.pgd.eps={args.eps}",
            f"attack.pgd.step_size={args.step_size}",
            "attack.pgd.random_start=false",
            f"attack.name={case.method}",
        ]
        print("[RUN]", " ".join(cmd))
        result = subprocess.run(cmd, cwd=project_root)
        if result.returncode != 0:
            failures.append((case.method, result.returncode))
        else:
            completed.append(case.method)

    print("\nCompleted runs:")
    for run_name in completed:
        print("-", run_name)

    if failures:
        print("\nFailed runs:")
        for method, code in failures:
            print(f"- {method}: returncode={code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
