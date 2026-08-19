"""Trainable VLM adapter package for multi-model unlearning.

Provides the :class:`TrainableVLMAdapter` abstraction and per-model-family
implementations that isolate model-specific loading, rendering, batching,
LoRA scope, and structural metadata behind a uniform interface.

The generic pipeline (datasets, objectives, evaluation) selects a model
profile and adapter — it does not branch on model names.
"""

from __future__ import annotations

from .base import (
    ModelFamilyProfile,
    NeuronSpec,
    TrainableVLMAdapter,
)
from .registry import (
    create_adapter,
    load_profile_from_yaml,
    register_adapter,
    available_adapters,
)

__all__ = [
    "ModelFamilyProfile",
    "NeuronSpec",
    "TrainableVLMAdapter",
    "create_adapter",
    "load_profile_from_yaml",
    "register_adapter",
    "available_adapters",
]
