"""Gemma3-12B-it trainable adapter stub.

Larger secondary validation model.  Requires HuggingFace license acceptance.

Official loading path: ``Gemma3ForConditionalGeneration`` + ``AutoProcessor``.
Requires Transformers >= 4.50.0.

Notes:

- Image handling: 896×896 / 256-token images — let the official processor
  handle normalization; do not manually emulate.
- Memory/resource preflight needed before KL/R²MU runs (12B scale).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .base import ModelFamilyProfile
from .hf_chat import HuggingFaceChatAdapter
from .registry import register_adapter_family, register_model_key

logger = logging.getLogger(__name__)


@register_adapter_family("gemma3")
class Gemma3Adapter(HuggingFaceChatAdapter):
    """Trainable adapter for Gemma3-12B-it (stub).

    Inherits standard HF chat-template operations from
    :class:`HuggingFaceChatAdapter`.  Structural methods require
    real-model discovery.

    The profile is **required** and must come from a YAML file.
    """

    def __init__(self, profile: ModelFamilyProfile):
        self._profile = profile

    @property
    def profile(self) -> ModelFamilyProfile:
        return self._profile

    # ------------------------------------------------------------------ #
    # Gemma-specific hooks
    # ------------------------------------------------------------------ #

    def _model_auto_class(self):
        """Gemma3 uses ``Gemma3ForConditionalGeneration``."""
        try:
            from transformers import Gemma3ForConditionalGeneration
            return Gemma3ForConditionalGeneration
        except ImportError:
            raise ImportError(
                "Gemma3ForConditionalGeneration not available. "
                "Requires transformers >= 4.50.0."
            )

    # ------------------------------------------------------------------ #
    # Methods requiring real-model discovery (stubs)
    # ------------------------------------------------------------------ #

    def resolve_lora_targets(self, model: torch.nn.Module) -> list[str]:
        raise NotImplementedError(
            "Gemma3 LoRA targets require runtime module tree inspection."
        )

    def language_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        raise NotImplementedError(
            "Gemma3 language layer path requires runtime inspection."
        )

    def language_hidden_size(self, model: torch.nn.Module) -> int:
        raise NotImplementedError(
            "Gemma3 hidden size requires runtime inspection."
        )

    def to_eval_backend(self, **kwargs) -> Any:
        """Convert to a generic :class:`AdapterEvalBackend`."""
        from ..adapter_eval_backend import AdapterEvalBackend

        return AdapterEvalBackend(adapter=self, **kwargs)


register_model_key("gemma3_12b", "gemma3")
