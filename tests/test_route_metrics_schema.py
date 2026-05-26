from pathlib import Path


LOSS_LOG_REQUIRED_FIELDS = [
    "step",
    "loss_total",
    "loss_inj",
    "loss_route",
    "carrier_target_sim",
    "route_attn_score",
    "route_cls_coupling_score",
    "route_rollout_score",
    "linf",
    "pixel_min",
    "pixel_max",
    "mean_abs_delta",
    "per_model_loss_inj",
    "per_model_loss_route",
    "per_layer_loss_inj",
    "per_layer_loss_route",
]

METRICS_REQUIRED_FIELDS = [
    "pair_id",
    "method",
    "objective",
    "active_loss",
    "selection_mode",
    "shallow_layers",
    "steps",
    "eps",
    "step_size",
    "topk_ratio",
    "seed",
    "lambda_route",
    "route_eta_attn",
    "route_eta_cls",
    "route_eta_rollout",
    "inj_loss_start",
    "inj_loss_end",
    "inj_loss_delta",
    "route_loss_start",
    "route_loss_end",
    "route_loss_delta",
    "total_loss_start",
    "total_loss_end",
    "total_loss_delta",
    "carrier_target_sim_start",
    "carrier_target_sim_end",
    "carrier_target_sim_delta",
    "route_attn_start",
    "route_attn_end",
    "route_attn_delta",
    "route_cls_coupling_start",
    "route_cls_coupling_end",
    "route_cls_coupling_delta",
    "route_rollout_start",
    "route_rollout_end",
    "route_rollout_delta",
    "linf_max",
    "linf_end",
    "linf_valid",
    "pixel_min_global",
    "pixel_max_global",
    "pixel_range_valid",
    "mean_abs_delta_end",
    "monotonic_decrease_ratio",
    "first_10pct_mean_total_loss",
    "final_10pct_mean_total_loss",
    "smooth_decrease_success",
    "per_model_inj_loss_delta",
    "per_model_route_loss_delta",
    "per_layer_inj_loss_delta",
    "per_layer_route_loss_delta",
]


def test_route_schema_field_names_exist_in_attack_writer() -> None:
    source = Path("src/fas2h/attacks/fas2h.py").read_text(encoding="utf-8")

    for field in LOSS_LOG_REQUIRED_FIELDS:
        assert f'"{field}"' in source

    for field in METRICS_REQUIRED_FIELDS:
        assert f'"{field}"' in source


def test_route_unavailable_fields_can_be_null_in_schema() -> None:
    sample = {
        "route_attn_start": None,
        "route_attn_end": None,
        "route_attn_delta": None,
        "route_cls_coupling_start": None,
        "route_cls_coupling_end": None,
        "route_cls_coupling_delta": None,
        "route_rollout_start": None,
        "route_rollout_end": None,
        "route_rollout_delta": None,
        "per_model_route_loss_delta": {"openclip_vit_b32": None},
        "per_layer_route_loss_delta": {"openclip_vit_b32": {"layer_1": None}},
    }
    for key in sample:
        assert key in sample
