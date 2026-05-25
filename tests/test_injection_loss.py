import torch

from fas2h.attacks.losses import injection_loss_logsumexp
from fas2h.route.projection import IdentityProjection


def test_injection_loss_is_scalar_and_finite_with_grad_flow() -> None:
    projection = IdentityProjection()
    adv_tokens = torch.randn(1, 3, 8, requires_grad=True)
    target_bank = torch.randn(1, 4, 8)

    loss = injection_loss_logsumexp(
        adv_patch_tokens=adv_tokens,
        target_bank=target_bank,
        tau=0.07,
        projection=projection,
        layer=0,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert adv_tokens.grad is not None
    assert adv_tokens.grad.abs().sum() > 0
    assert target_bank.requires_grad is False
