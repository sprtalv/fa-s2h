from pathlib import Path
import json


REQUIRED_FIELDS = [
    "pair_id",
    "method",
    "selection_mode",
    "shallow_layers",
    "topk_ratio",
    "seed",
    "steps",
    "eps",
    "step_size",
    "inj_loss_start",
    "inj_loss_end",
    "inj_loss_delta",
    "inj_loss_relative_change",
    "inj_loss_min",
    "inj_loss_argmin_step",
    "carrier_target_sim_start",
    "carrier_target_sim_end",
    "carrier_target_sim_delta",
    "sim_increase_success",
    "linf_max",
    "linf_end",
    "linf_valid",
    "pixel_min_global",
    "pixel_max_global",
    "pixel_range_valid",
    "mean_abs_delta_end",
    "monotonic_decrease_ratio",
    "first_10pct_mean_loss",
    "final_10pct_mean_loss",
    "smooth_decrease_success",
    "per_model_loss_delta",
    "per_layer_loss_delta",
    "per_model_sim_delta",
    "per_layer_sim_delta",
    "notes",
]


def test_metrics_schema_fields_exist_when_metrics_present() -> None:
    run = Path("outputs/runs_resources/fas2h_mvp_res1000_n5_steps200_seed42_20260526_035129")
    metrics_file = run / "samples" / "resources_0000" / "metrics.json"
    if not metrics_file.exists():
        return
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    for field in REQUIRED_FIELDS:
        assert field in metrics
