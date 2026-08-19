"""Adapter registry for trainable VLM models.

Maps model-family keys to adapter factory callables and provides
profile loading from YAML configuration files.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from .base import ModelFamilyProfile, TrainableVLMAdapter

logger = logging.getLogger(__name__)

# key → factory callable: () -> TrainableVLMAdapter
_ADAPTER_FACTORIES: dict[str, Callable[[], TrainableVLMAdapter]] = {}

# key -> cached adapter instance
_ADAPTER_CACHE: dict[str, TrainableVLMAdapter] = {}


def register_adapter(key: str):
    """Decorator to register a trainable adapter factory by model key.

    Usage::

        @register_adapter("qwen35_9b")
        def _create_qwen35() -> TrainableVLMAdapter:
            return Qwen35Adapter()
    """
    def decorator(fn: Callable[[], TrainableVLMAdapter]):
        if key in _ADAPTER_FACTORIES:
            raise ValueError(f"Trainable adapter already registered for key: {key!r}")
        _ADAPTER_FACTORIES[key] = fn
        return fn
    return decorator


def create_adapter(key: str) -> TrainableVLMAdapter:
    """Create or return a cached adapter for the given model key.

    Raises ``KeyError`` if no adapter is registered for *key*.
    """
    if key not in _ADAPTER_CACHE:
        if key not in _ADAPTER_FACTORIES:
            _ensure_builtin_adapters_loaded()
        if key not in _ADAPTER_FACTORIES:
            raise KeyError(
                f"Unknown trainable adapter key: {key!r}. "
                f"Available: {sorted(_ADAPTER_FACTORIES)}"
            )
        _ADAPTER_CACHE[key] = _ADAPTER_FACTORIES[key]()
    return _ADAPTER_CACHE[key]


def available_adapters() -> list[str]:
    """Return sorted list of registered adapter keys."""
    _ensure_builtin_adapters_loaded()
    return sorted(_ADAPTER_FACTORIES.keys())


def load_profile_from_yaml(path: str | Path) -> ModelFamilyProfile:
    """Load a :class:`ModelFamilyProfile` from a YAML configuration file.

    The YAML schema mirrors the ``ModelFamilyProfile`` dataclass fields
    with nested sections: ``model``, ``candidate_protocol``, ``lora``,
    ``structural``, ``compatibility``, ``access``.
    """
    import yaml

    with open(path) as fh:
        data = yaml.safe_load(fh)

    model = data["model"]
    candidate = data.get("candidate_protocol", {})
    lora = data.get("lora", {})
    structural = data.get("structural", {})
    compat = data.get("compatibility", {})
    access = data.get("access", {})

    return ModelFamilyProfile(
        key=data["key"],
        model_id=model["id"],
        revision=model["revision"],
        processor_id=model.get("processor_id", model["id"]),
        processor_revision=model.get("processor_revision", model["revision"]),
        adapter_name=model.get("adapter", data["key"]),
        trust_remote_code=model.get("trust_remote_code", False),
        dtype=model.get("dtype", "bfloat16"),
        attn_implementation=model.get("attn_implementation", "sdpa"),
        candidate_positive=candidate.get("positive", "Yes"),
        candidate_negative=candidate.get("negative", "No"),
        lora_rank=lora.get("rank", 8),
        lora_alpha=lora.get("alpha", 16),
        lora_dropout=lora.get("dropout", 0.05),
        lora_scope=lora.get("scope", "language_attention_only"),
        lora_target_leaf_names=tuple(lora.get("target_leaf_names", [])),
        lora_scope_regex=lora.get("scope_regex", ""),
        r2mu_candidate_layers=tuple(structural.get("r2mu_candidate_layers", ())),
        r2mu_n_select_layers=structural.get("r2mu_n_select_layers", 0),
        supports_prompting=data.get("supports_prompting", True),
        supports_candidate_margin=data.get("supports_candidate_margin", True),
        supports_ga=data.get("supports_ga", True),
        supports_gd=data.get("supports_gd", True),
        supports_kl=data.get("supports_kl", True),
        supports_npo=data.get("supports_npo", True),
        supports_mmunlearner=data.get("supports_mmunlearner", True),
        supports_manu=data.get("supports_manu", True),
        supports_r2mu=data.get("supports_r2mu", True),
        min_transformers_version=compat.get("min_transformers"),
        tested_transformers_version=compat.get("tested_transformers", ""),
        requires_hf_auth=access.get("requires_hf_auth", False),
    )


def _ensure_builtin_adapters_loaded() -> None:
    """Lazily import built-in adapter modules to trigger registration."""
    _try_import("qwen35")
    _try_import("glm46v")
    _try_import("internvl35")
    _try_import("phi4mm")
    _try_import("gemma3")


def _try_import(module_name: str) -> None:
    try:
        __import__(
            f"route_data.models.trainable.{module_name}",
            fromlist=[module_name],
        )
    except ImportError:
        pass


def clear_cache() -> None:
    """Clear the adapter cache. Useful for testing."""
    _ADAPTER_CACHE.clear()
