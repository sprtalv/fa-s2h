import torch

from fas2h.attacks.losses import carrier_target_similarity
from fas2h.route.projection import IdentityProjection


def test_carrier_target_similarity_is_finite_scalar() -> None:
    adv = torch.randn(1, 4, 8)
    bank = torch.randn(1, 5, 8)
    sim = carrier_target_similarity(
        adv_patch_tokens=adv,
        target_bank=bank,
        projection=IdentityProjection(),
        layer=0,
    )
    assert sim.ndim == 0
    assert torch.isfinite(sim)


def test_similarity_increases_when_adv_equals_target_tokens() -> None:
    torch.manual_seed(0)
    bank = torch.randn(1, 5, 8)
    adv_random = torch.randn(1, 4, 8)
    adv_match = bank[:, :4, :].clone()
    sim_random = carrier_target_similarity(adv_random, bank, IdentityProjection(), 0)
    sim_match = carrier_target_similarity(adv_match, bank, IdentityProjection(), 0)
    assert sim_match > sim_random
