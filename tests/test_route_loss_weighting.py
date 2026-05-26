import pytest
import torch

from fas2h.attacks.losses import route_loss
from fas2h.features import FeatureBundle, RouteBundle


def _setup_inputs() -> tuple[dict[str, FeatureBundle], dict[str, dict[int, RouteBundle]], list[int]]:
    features = FeatureBundle(
        patch_tokens_by_layer={
            1: torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]], dtype=torch.float32),
        },
        cls_tokens_by_layer={
            1: torch.tensor([[0.2, 0.3]], dtype=torch.float32),
        },
        attentions_by_layer={
            1: torch.rand(1, 2, 4, 4, dtype=torch.float32),
            2: torch.rand(1, 2, 4, 4, dtype=torch.float32),
        },
        final_global=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        num_layers=3,
        num_patches=3,
    )
    source_routes = {
        "m1": {
            1: RouteBundle(source_indices=torch.tensor([[0, 2]], dtype=torch.long), layer=1, model_name="m1"),
        }
    }
    return {"m1": features}, source_routes, [1]


def test_route_loss_weight_changes_total_value() -> None:
    bundles, routes, layers = _setup_inputs()

    attn_only = route_loss(
        adv_feature_bundles=bundles,
        source_routes=routes,
        layers=layers,
        eta_attn=1.0,
        eta_cls=0.0,
        eta_rollout=0.0,
    )
    cls_only = route_loss(
        adv_feature_bundles=bundles,
        source_routes=routes,
        layers=layers,
        eta_attn=0.0,
        eta_cls=1.0,
        eta_rollout=0.0,
    )

    assert not torch.allclose(attn_only["loss_route"], cls_only["loss_route"])


def test_route_loss_rejects_invalid_eta_sum() -> None:
    bundles, routes, layers = _setup_inputs()

    with pytest.raises(ValueError, match="must sum to 1.0"):
        route_loss(
            adv_feature_bundles=bundles,
            source_routes=routes,
            layers=layers,
            eta_attn=0.5,
            eta_cls=0.5,
            eta_rollout=0.2,
        )
