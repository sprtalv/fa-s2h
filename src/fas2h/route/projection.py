"""Projection helpers for FA-S2H route scoring and injection loss.

The MVP uses identity projection only. A dedicated interface is still useful so
future learned layer-specific projections can be added without silently changing
the MVP assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IdentityProjection:
    """Identity projection used by the FA-S2H MVP."""

    projection_type: str = "identity"

    def project(self, layer: int, tensor: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        NOTE(fas2h): this is intentional MVP behavior. Future work may replace
        it with a learned layer-specific projection `P_l`.
        """
        del layer
        return tensor


def build_projection(cfg: dict) -> IdentityProjection:
    """Build the configured projection object."""
    projection_type = cfg.get("type", "identity")
    if projection_type != "identity":
        raise ValueError(
            f"Unsupported projection type {projection_type!r}; MVP implements identity projection only."
        )
    return IdentityProjection(projection_type=projection_type)
