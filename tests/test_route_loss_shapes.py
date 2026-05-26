import torch

from fas2h.attacks.losses import route_loss
from fas2h.features import FeatureBundle, RouteBundle


def _make_feature_bundle() -> FeatureBundle:
    # 4 layers total, use shallow layers 1 and 2.
    patch_l1 = torch.tensor([[[0.1, 0.0, 0.2], [0.3, 0.1, 0.4], [0.5, 0.3, 0.2], [0.7, 0.2, 0.9]]], dtype=torch.float32)
    patch_l2 = torch.tensor([[[0.2, 0.1, 0.0], [0.4, 0.2, 0.5], [0.6, 0.3, 0.1], [0.8, 0.4, 1.0]]], dtype=torch.float32)
    cls_l1 = torch.tensor([[0.2, 0.1, 0.3]], dtype=torch.float32)
    cls_l2 = torch.tensor([[0.3, 0.2, 0.4]], dtype=torch.float32)

    # [B, H, T, T], T = N + 1 = 5
    attn_l1 = torch.rand(1, 2, 5, 5, dtype=torch.float32)
    attn_l2 = torch.rand(1, 2, 5, 5, dtype=torch.float32)
    attn_l3 = torch.rand(1, 2, 5, 5, dtype=torch.float32)

    return FeatureBundle(
        patch_tokens_by_layer={1: patch_l1, 2: patch_l2},
        cls_tokens_by_layer={1: cls_l1, 2: cls_l2},
        attentions_by_layer={1: attn_l1, 2: attn_l2, 3: attn_l3},
        final_global=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        num_layers=4,
        num_patches=4,
    )


def test_route_loss_returns_finite_scalar_and_scores() -> None:
    features = _make_feature_bundle()
    source_routes = {
        "m1": {
            1: RouteBundle(source_indices=torch.tensor([[0, 2]], dtype=torch.long), layer=1, model_name="m1"),
            2: RouteBundle(source_indices=torch.tensor([[1, 3]], dtype=torch.long), layer=2, model_name="m1"),
        }
    }
    result = route_loss(
        adv_feature_bundles={"m1": features},
        source_routes=source_routes,
        layers=[1, 2],
        eta_attn=1.0 / 3.0,
        eta_cls=1.0 / 3.0,
        eta_rollout=1.0 / 3.0,
    )

    assert "loss_route" in result
    assert torch.is_tensor(result["loss_route"])
    assert result["loss_route"].ndim == 0
    assert torch.isfinite(result["loss_route"]).item()

    assert torch.is_tensor(result["route_attn_score"])
    assert torch.is_tensor(result["route_cls_coupling_score"])
    assert torch.is_tensor(result["route_rollout_score"])

    assert "m1" in result["per_model_loss_route"]
    assert "layer_1" in result["per_layer_loss_route"]["m1"]
    assert "layer_2" in result["per_layer_loss_route"]["m1"]
