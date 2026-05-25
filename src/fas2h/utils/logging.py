"""Logging helpers for FA-S2H scripts."""

from __future__ import annotations

import logging


def get_logger(name: str = "fas2h") -> logging.Logger:
    """Create a lightweight console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
