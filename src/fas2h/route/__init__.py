"""Route selection, rollout, projection, and target-bank helpers."""

from .bank import build_target_evidence_bank
from .cache_keys import build_cache_key
from .projection import IdentityProjection, build_projection
from .rollout import add_residual_and_normalize, average_heads, compute_rollout, extract_cls_to_patch_rollout
from .source_carrier import select_source_carriers
from .target_evidence import select_target_evidence

__all__ = [
    "select_source_carriers",
    "select_target_evidence",
    "average_heads",
    "add_residual_and_normalize",
    "compute_rollout",
    "extract_cls_to_patch_rollout",
    "IdentityProjection",
    "build_projection",
    "build_target_evidence_bank",
    "build_cache_key",
]
