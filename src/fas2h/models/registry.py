"""Registry helpers for resolving model configs into runnable wrappers."""

from __future__ import annotations

import warnings
from typing import Any

import torch

from fas2h.models.wrappers.openclip_wrapper import OpenCLIPVisionWrapper


def resolve_runtime_device(device: str) -> str:
    """Resolve runtime device string.

    `auto` prefers CUDA when available because the MVP attack extracts
    multi-layer tokens and real attention weights from several surrogates.
    """
    if device == "auto":
        if torch.cuda.is_available():
            try:
                _ = torch.empty(1, device="cuda")
                return "cuda"
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"CUDA was reported available but could not be initialized; falling back to CPU. Details: {exc}",
                    stacklevel=2,
                )
        return "cpu"
    return device


def build_surrogate_wrappers(cfg: dict[str, Any], device: str = "auto") -> list[OpenCLIPVisionWrapper]:
    """Build wrappers from composed Hydra config or plain model config.

    Supported layouts:
    - `cfg["model"]["surrogates"]` from the composed Hydra root config.
    - `cfg["surrogates"]` from a direct model config file.
    """
    model_cfg = cfg.get("model", cfg)
    surrogates = model_cfg.get("surrogates", [])
    resolved_device = resolve_runtime_device(device)

    wrappers: list[OpenCLIPVisionWrapper] = []
    for item in surrogates:
        backend = item.get("backend")
        if backend != "open_clip":
            raise ValueError(
                f"Unsupported backend {backend!r}; FA-S2H MVP currently implements OpenCLIP only."
            )

        wrappers.append(
            OpenCLIPVisionWrapper(
                name=item["name"],
                model_name=item["model_name"],
                pretrained=item["pretrained"],
                device=resolved_device,
            )
        )
    return wrappers
