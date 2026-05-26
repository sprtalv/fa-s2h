from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fas2h.attacks.fas2h import FAS2HAttack


def make_attack(auto_name: bool = True) -> FAS2HAttack:
    attack = object.__new__(FAS2HAttack)
    attack.attack_cfg = SimpleNamespace(
        name="fas2h_mvp",
        objective="inj_only",
        shallow_layers=[1, 2, 3, 4, 5, 6],
        pgd=SimpleNamespace(steps=200, eps=16 / 255),
        loss=SimpleNamespace(active="inj_only", lambda_route=0.0, route=SimpleNamespace(enabled=False)),
    )
    attack.data_cfg = SimpleNamespace(pair_file="configs/data/resources_pairs_1000.jsonl", limit=5)
    attack.runtime_cfg = SimpleNamespace(seed=42, auto_name=auto_name, run_name_include_params=False, run_tag=None)
    return attack


def test_auto_run_stem_uses_config_values() -> None:
    attack = make_attack(auto_name=True)
    stem = attack._build_auto_run_stem()
    assert stem == "fas2h_mvp_res1000_n5_steps200_seed42"


def test_run_id_uses_attack_name_when_auto_naming_is_disabled() -> None:
    attack = make_attack(auto_name=False)
    run_id = attack._run_id()
    assert run_id.startswith("fas2h_mvp_")
    assert "res1000" not in run_id


def test_param_run_stem_includes_objective_layers_steps_eps_seed() -> None:
    attack = make_attack(auto_name=True)
    attack.runtime_cfg.run_name_include_params = True
    stem = attack._build_auto_run_stem()
    assert stem == "inj_only_l123456_s200_eps16_seed42"


def test_run_tag_overrides_auto_stem() -> None:
    attack = make_attack(auto_name=True)
    attack.runtime_cfg.run_tag = "inj_route005_l123456_s300_eps16_seed42"
    run_id = attack._run_id()
    assert run_id.startswith("inj_route005_l123456_s300_eps16_seed42_")
