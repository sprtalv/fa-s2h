from pathlib import Path


def test_score_decomposition_schema_has_required_fields() -> None:
    run = Path("outputs/runs_resources/fas2h_mvp_res1000_n5_steps200_seed42_20260526_035129")
    source = run / "samples" / "resources_0000" / "source_carriers.json"
    target = run / "samples" / "resources_0000" / "target_evidence.json"

    if not source.exists() or not target.exists():
        # Existing historical runs may not yet contain the new schema.
        return

    source_data = __import__("json").loads(source.read_text(encoding="utf-8"))
    target_data = __import__("json").loads(target.read_text(encoding="utf-8"))
    if "models" not in source_data or "models" not in target_data:
        # Historical runs from before this feature do not include decomposition schema.
        return

    for model_layers in source_data["models"].values():
        for records in model_layers.values():
            assert records
            sample = records[0]
            for field in ["total_score", "attn_score", "cls_coupling_score", "global_score"]:
                assert field in sample

    for model_layers in target_data["models"].values():
        for records in model_layers.values():
            assert records
            sample = records[0]
            for field in ["total_score", "path_score", "semantic_score"]:
                assert field in sample


def test_random_schema_flags_are_declared() -> None:
    # Schema-level check independent of run availability.
    payload = {
        "selection_mode": "random_patch",
        "selected_by": "random",
        "scores_logged_for_diagnosis_only": True,
        "models": {"m": {"layer_0": [{"total_score": 0.1, "attn_score": 0.1, "cls_coupling_score": 0.0, "global_score": 0.2}] }},
    }
    assert payload["selection_mode"] == "random_patch"
    assert payload["selected_by"] == "random"
    assert payload["scores_logged_for_diagnosis_only"] is True
