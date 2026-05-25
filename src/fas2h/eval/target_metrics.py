"""Targeted-attack metric interfaces."""

from __future__ import annotations

from typing import Any


def compute_target_metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compute targeted metrics placeholder.

    TODO(fas2h): implement KMR, AvgSim, ASR, LLM judge metrics later.
    """
    return {"status": "placeholder", "message": "target metrics are not implemented in the MVP"}
