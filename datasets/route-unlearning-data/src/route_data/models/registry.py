"""Backend registry (coding plan sections 6.2, 3.6).

Backends are registered by name and must return the same VisionResponse
schema. The factory records the model *role* in every run manifest to prevent
circular evaluation (a model must not be evaluated by itself in the same role).
"""

from __future__ import annotations

from typing import Callable

from ..config import ModelConfig
from .base import VisionLanguageModel

_FACTORIES: dict[str, Callable[[ModelConfig], VisionLanguageModel]] = {}


def register_backend(name: str):
    def decorator(fn: Callable[[ModelConfig], VisionLanguageModel]):
        if name in _FACTORIES:
            raise ValueError(f"Backend already registered: {name}")
        _FACTORIES[name] = fn
        return fn

    return decorator


def available_backends() -> list[str]:
    return sorted(_FACTORIES)


def ensure_backends_loaded() -> list[str]:
    """Trigger lazy backend imports and return the registered names."""
    try:
        from . import mllama  # noqa: F401
    except ImportError:
        pass
    try:
        from . import stub  # noqa: F401
    except ImportError:
        pass
    return available_backends()


def create_backend(config: ModelConfig) -> VisionLanguageModel:
    """Instantiate a backend from a validated model config."""
    if config.backend not in _FACTORIES:
        # Import built-in backends lazily so registry population happens on
        # first use without heavy imports at module load time.
        ensure_backends_loaded()

    if config.backend not in _FACTORIES:
        raise KeyError(
            f"Unknown model backend {config.backend!r}. "
            f"Available: {available_backends()}"
        )
    return _FACTORIES[config.backend](config)
