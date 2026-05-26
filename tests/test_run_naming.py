from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fas2h.attacks.fas2h import FAS2HAttack


def make_attack(auto_name: bool = True) -> FAS2HAttack:
    attack = object.__new__(FAS2HAttack)
    attack.attack_cfg = SimpleNamespace(name="fas2h_mvp", pgd=SimpleNamespace(steps=200))
    attack.data_cfg = SimpleNamespace(pair_file="configs/data/resources_pairs_1000.jsonl", limit=5)
    attack.runtime_cfg = SimpleNamespace(seed=42, auto_name=auto_name)
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
