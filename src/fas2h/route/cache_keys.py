"""Deterministic cache-key utilities for pair/model/layer-specific route artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def build_cache_key(
    pair_id: str,
    model_name: str,
    pretrained: str,
    layers: list[int],
    topk_ratio: float,
    alpha_attn: float,
    beta_cls: float,
    gamma_global: float,
    lambda_path: float,
    projection_type: str,
    source_path: str,
    target_path: str,
) -> str:
    """Build a deterministic key for route/bank caches.

    The cache key includes image paths because FA-S2H banks are pair-specific.
    If future experiments need stronger invalidation, these paths can be swapped
    for file hashes without changing the call sites.
    """
    payload = {
        "pair_id": pair_id,
        "model_name": model_name,
        "pretrained": pretrained,
        "layers": sorted(int(layer) for layer in layers),
        "topk_ratio": float(topk_ratio),
        "alpha_attn": float(alpha_attn),
        "beta_cls": float(beta_cls),
        "gamma_global": float(gamma_global),
        "lambda_path": float(lambda_path),
        "projection_type": projection_type,
        "source_path": str(Path(source_path)),
        "target_path": str(Path(target_path)),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]
    return f"fas2h_{digest}"
