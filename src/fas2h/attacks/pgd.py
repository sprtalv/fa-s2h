"""Projected gradient descent helpers for FA-S2H."""

from __future__ import annotations

import torch


def clamp_image(x: torch.Tensor, clamp_min: float = 0.0, clamp_max: float = 1.0) -> torch.Tensor:
    """Clamp image tensor to the valid pixel range `[0, 1]` by default."""
    return x.clamp(clamp_min, clamp_max)


def project_linf(x_adv: torch.Tensor, x_src: torch.Tensor, eps: float) -> torch.Tensor:
    """Project `x_adv` into the `L_inf` ball around `x_src`.

    NOTE(fas2h): `eps=16/255` is interpreted in pixel scale, not normalized model space.
    """
    return torch.max(torch.min(x_adv, x_src + eps), x_src - eps)


def pgd_step(
    x_adv: torch.Tensor,
    grad: torch.Tensor,
    x_src: torch.Tensor,
    eps: float,
    step_size: float,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    """Run one gradient-descent PGD step for the FA-S2H MVP objective."""
    x_next = x_adv - step_size * grad.sign()
    x_next = project_linf(x_next, x_src=x_src, eps=eps)
    x_next = clamp_image(x_next, clamp_min=clamp_min, clamp_max=clamp_max)
    return x_next.detach()
