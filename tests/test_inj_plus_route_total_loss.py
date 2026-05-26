from types import SimpleNamespace

import torch

from fas2h.attacks.fas2h import FAS2HAttack


def _make_attack(objective: str, lambda_route: float, route_enabled: bool) -> FAS2HAttack:
    attack = object.__new__(FAS2HAttack)
    attack.cfg = SimpleNamespace(
        logging=SimpleNamespace(
            record_route_attn_score=False,
            record_route_cls_coupling_score=False,
            record_route_rollout_score=False,
        )
    )
    attack.attack_cfg = SimpleNamespace(
        objective=objective,
        active_loss=objective,
        shallow_layers=[1, 2],
        loss=SimpleNamespace(
            active=objective,
            lambda_route=lambda_route,
            route=SimpleNamespace(
                enabled=route_enabled,
                eta_attn=1.0 / 3.0,
                eta_cls=1.0 / 3.0,
                eta_rollout=1.0 / 3.0,
            ),
        ),
    )
    attack.models = [SimpleNamespace(name="m1")]
    attack._compute_adv_features = lambda x_adv, capture_attn: {"m1": object()}  # type: ignore[method-assign]
    attack._compute_injection_objective = lambda adv_feature_bundles, source_routes, target_banks: {  # type: ignore[method-assign]
        "loss_total": torch.tensor(-2.0),
        "loss_inj": torch.tensor(-2.0),
        "carrier_target_sim": torch.tensor(0.4),
        "per_model_loss_inj": {},
        "per_layer_loss_inj": {},
        "per_model_sim": {},
        "per_layer_sim": {},
    }
    return attack


def test_lambda_zero_keeps_total_equal_inj() -> None:
    attack = _make_attack(objective="inj_only", lambda_route=0.0, route_enabled=False)
    out = attack._compute_objective(
        x_adv=torch.zeros(1, 3, 4, 4),
        source_routes={},
        target_banks={},
    )
    assert torch.allclose(out["loss_total"], out["loss_inj"])


def test_lambda_positive_adds_route_term(monkeypatch) -> None:
    attack = _make_attack(objective="inj_plus_route", lambda_route=0.1, route_enabled=True)

    def _fake_route_loss(*args, **kwargs):
        return {
            "loss_route": torch.tensor(-3.0),
            "route_attn_score": torch.tensor(0.2),
            "route_cls_coupling_score": torch.tensor(0.3),
            "route_rollout_score": torch.tensor(0.4),
            "per_model_loss_route": {"m1": torch.tensor(-3.0)},
            "per_layer_loss_route": {"m1": {"layer_1": torch.tensor(-3.0), "layer_2": torch.tensor(-3.0)}},
        }

    monkeypatch.setattr("fas2h.attacks.fas2h.route_loss", _fake_route_loss)

    out = attack._compute_objective(
        x_adv=torch.zeros(1, 3, 4, 4),
        source_routes={"m1": {}},
        target_banks={"m1": {}},
    )
    expected = torch.tensor(-2.0) + 0.1 * torch.tensor(-3.0)
    assert torch.allclose(out["loss_total"], expected)
