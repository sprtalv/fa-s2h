#!/usr/bin/env python3
"""Run the agreed Stage-1 + Stage-2 route-loss sweep.

This script intentionally keeps the sweep matrix small and explicit:
- Stage 1: compare `inj_only` with several `lambda_route` values under balanced eta
- Stage 2: fix `lambda_route=0.1` and compare route component weightings

Use `--dry-run` first to verify commands and naming before launching the full sweep.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SweepCase:
    """One route-loss sweep case."""

    run_tag: str
    objective: str
    lambda_route: float
    eta_attn: float
    eta_cls: float
    eta_rollout: float


CASES = [
    SweepCase(
        run_tag="inj_only_l123456_s300_eps16_seed42",
        objective="inj_only",
        lambda_route=0.0,
        eta_attn=1.0 / 3.0,
        eta_cls=1.0 / 3.0,
        eta_rollout=1.0 / 3.0,
    ),
    SweepCase(
        run_tag="inj_route005_eta333_l123456_s300_eps16_seed42",
        objective="inj_plus_route",
        lambda_route=0.05,
        eta_attn=1.0 / 3.0,
        eta_cls=1.0 / 3.0,
        eta_rollout=1.0 / 3.0,
    ),
    SweepCase(
        run_tag="inj_route01_eta333_l123456_s300_eps16_seed42",
        objective="inj_plus_route",
        lambda_route=0.10,
        eta_attn=1.0 / 3.0,
        eta_cls=1.0 / 3.0,
        eta_rollout=1.0 / 3.0,
    ),
    SweepCase(
        run_tag="inj_route02_eta333_l123456_s300_eps16_seed42",
        objective="inj_plus_route",
        lambda_route=0.20,
        eta_attn=1.0 / 3.0,
        eta_cls=1.0 / 3.0,
        eta_rollout=1.0 / 3.0,
    ),
    SweepCase(
        run_tag="inj_route03_eta333_l123456_s300_eps16_seed42",
        objective="inj_plus_route",
        lambda_route=0.30,
        eta_attn=1.0 / 3.0,
        eta_cls=1.0 / 3.0,
        eta_rollout=1.0 / 3.0,
    ),
    SweepCase(
        run_tag="inj_route01_eta_attn_bias_l123456_s300_eps16_seed42",
        objective="inj_plus_route",
        lambda_route=0.10,
        eta_attn=0.50,
        eta_cls=0.25,
        eta_rollout=0.25,
    ),
    SweepCase(
        run_tag="inj_route01_eta_cls_bias_l123456_s300_eps16_seed42",
        objective="inj_plus_route",
        lambda_route=0.10,
        eta_attn=0.25,
        eta_cls=0.50,
        eta_rollout=0.25,
    ),
    SweepCase(
        run_tag="inj_route01_eta_rollout_bias_l123456_s300_eps16_seed42",
        objective="inj_plus_route",
        lambda_route=0.10,
        eta_attn=0.25,
        eta_cls=0.25,
        eta_rollout=0.50,
    ),
    SweepCase(
        run_tag="inj_route01_eta_attn_strong_l123456_s300_eps16_seed42",
        objective="inj_plus_route",
        lambda_route=0.10,
        eta_attn=0.60,
        eta_cls=0.20,
        eta_rollout=0.20,
    ),
    SweepCase(
        run_tag="inj_route01_eta_rollout_strong_l123456_s300_eps16_seed42",
        objective="inj_plus_route",
        lambda_route=0.10,
        eta_attn=0.20,
        eta_cls=0.20,
        eta_rollout=0.60,
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the fixed Stage-1 + Stage-2 route sweep.")
    parser.add_argument("--output-root", default="outputs/runs_route_loss_stage12", help="Run output root.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-selection-seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--eps", type=float, default=0.062745098)
    parser.add_argument("--step-size", type=float, default=0.0039215686)
    parser.add_argument("--topk-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def build_command(project_root: Path, args: argparse.Namespace, case: SweepCase) -> list[str]:
    """Build one child `run_attack.py` command."""
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
        "runtime.run_name_include_params=false",
        f"runtime.run_tag={case.run_tag}",
        f"data.limit={args.limit}",
        "attack.selection_mode=fas2h",
        "attack.shallow_layers=[1,2,3,4,5,6]",
        f"attack.topk_ratio={args.topk_ratio}",
        f"attack.pgd.steps={args.steps}",
        f"attack.pgd.eps={args.eps}",
        f"attack.pgd.step_size={args.step_size}",
        "attack.pgd.random_start=false",
        f"attack.objective={case.objective}",
        f"attack.active_loss={case.objective}",
        f"attack.loss.active={case.objective}",
        f"attack.loss.lambda_route={case.lambda_route}",
        f"attack.loss.route.enabled={'true' if case.objective == 'inj_plus_route' else 'false'}",
        f"attack.loss.route.eta_attn={case.eta_attn}",
        f"attack.loss.route.eta_cls={case.eta_cls}",
        f"attack.loss.route.eta_rollout={case.eta_rollout}",
        f"attack.name={case.run_tag}",
    ]
    return cmd


def main() -> int:
    """Run or print the route sweep."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    failures: list[tuple[str, int]] = []
    completed: list[str] = []

    for case in CASES:
        cmd = build_command(project_root, args, case)
        print("[RUN]", " ".join(cmd))
        if args.dry_run:
            completed.append(case.run_tag)
            continue

        result = subprocess.run(cmd, cwd=project_root)
        if result.returncode != 0:
            failures.append((case.run_tag, result.returncode))
        else:
            completed.append(case.run_tag)

    print("\nPlanned or completed runs:")
    for run_tag in completed:
        print("-", run_tag)

    if failures:
        print("\nFailed runs:")
        for run_tag, code in failures:
            print(f"- {run_tag}: returncode={code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
