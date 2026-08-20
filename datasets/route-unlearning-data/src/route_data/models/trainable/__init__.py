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
    adapter_families,
    available_adapters,
    clear_cache,
    compute_profile_sha256,
    create_adapter,
    load_profile_from_yaml,
    register_adapter_family,
    register_model_key,
    validate_environment_compatibility,
    validate_research_profile,
    validate_structural_metadata,
)

__all__ = [
    "ModelFamilyProfile",
    "NeuronSpec",
    "TrainableVLMAdapter",
    "adapter_families",
    "available_adapters",
    "clear_cache",
    "compute_profile_sha256",
    "create_adapter",
    "load_profile_from_yaml",
    "register_adapter_family",
    "register_model_key",
    "validate_environment_compatibility",
    "validate_research_profile",
    "validate_structural_metadata",
]
