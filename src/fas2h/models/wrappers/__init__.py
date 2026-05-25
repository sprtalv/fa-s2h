"""Wrapper implementations for vision surrogate backends."""

from .base import BaseVisionWrapper
from .openclip_wrapper import OpenCLIPVisionWrapper

__all__ = ["BaseVisionWrapper", "OpenCLIPVisionWrapper"]
