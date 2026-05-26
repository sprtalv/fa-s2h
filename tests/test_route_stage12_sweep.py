from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_route_stage12_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_route_stage12_sweep", SCRIPT_PATH)
assert SPEC is not None
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CASES = MODULE.CASES
SweepCase = MODULE.SweepCase
build_command = MODULE.build_command
parse_args = MODULE.parse_args


def test_route_stage12_case_names_are_unique() -> None:
    run_tags = [case.run_tag for case in CASES]
    assert len(run_tags) == len(set(run_tags))


def test_route_stage12_cases_have_valid_eta_sum() -> None:
    for case in CASES:
        assert abs(case.eta_attn + case.eta_cls + case.eta_rollout - 1.0) < 1e-9


def test_route_stage12_builds_expected_inj_only_command(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["run_route_stage12_sweep.py", "--dry-run"],
    )
    args = parse_args()
    project_root = Path("/tmp/fas2h")
    case = SweepCase(
        run_tag="inj_only_l123456_s300_eps16_seed42",
        objective="inj_only",
        lambda_route=0.0,
        eta_attn=1.0 / 3.0,
        eta_cls=1.0 / 3.0,
        eta_rollout=1.0 / 3.0,
    )
    cmd = build_command(project_root, args, case)
    joined = " ".join(cmd)
    assert "runtime.run_tag=inj_only_l123456_s300_eps16_seed42" in joined
    assert "attack.objective=inj_only" in joined
    assert "attack.loss.route.enabled=false" in joined


def test_route_stage12_builds_expected_route_command(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["run_route_stage12_sweep.py", "--dry-run"],
    )
    args = parse_args()
    project_root = Path("/tmp/fas2h")
    case = SweepCase(
        run_tag="inj_route01_eta_rollout_strong_l123456_s300_eps16_seed42",
        objective="inj_plus_route",
        lambda_route=0.10,
        eta_attn=0.20,
        eta_cls=0.20,
        eta_rollout=0.60,
    )
    cmd = build_command(project_root, args, case)
    joined = " ".join(cmd)
    assert "attack.objective=inj_plus_route" in joined
    assert "attack.loss.lambda_route=0.1" in joined
    assert "attack.loss.route.enabled=true" in joined
    assert "attack.loss.route.eta_rollout=0.6" in joined
