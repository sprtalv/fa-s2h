from fas2h import __version__
from fas2h.attacks.losses import injection_loss_logsumexp
from fas2h.features import BankBundle, FeatureBundle, RouteBundle
from fas2h.route.cache_keys import build_cache_key
from fas2h.route.rollout import add_residual_and_normalize, average_heads, compute_rollout, extract_cls_to_patch_rollout


def test_basic_imports() -> None:
    assert isinstance(__version__, str)
    assert FeatureBundle is not None
    assert RouteBundle is not None
    assert BankBundle is not None
    assert callable(build_cache_key)
    assert callable(injection_loss_logsumexp)
    assert callable(average_heads)
    assert callable(add_residual_and_normalize)
    assert callable(compute_rollout)
    assert callable(extract_cls_to_patch_rollout)
