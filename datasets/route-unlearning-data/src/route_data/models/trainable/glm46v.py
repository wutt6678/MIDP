"""GLM-4.6V-Flash trainable adapter stub.

Highest-priority new model.  This stub provides the adapter skeleton;
the exact module tree, LoRA scope regex, and structural metadata must
be discovered on the real model before flipping support flags.

Official loading path: ``Glm4vForConditionalGeneration`` + ``AutoProcessor``.
Requires Transformers >= 5.0.0rc0.

GLM-specific behaviour (to be verified on real model):

- Drop ``token_type_ids`` from forward inputs (official quick-start).
- Chat template follows standard HF pattern (no ``enable_thinking``).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .base import ModelFamilyProfile
from .hf_chat import HuggingFaceChatAdapter
from .registry import register_adapter_family, register_model_key

logger = logging.getLogger(__name__)


@register_adapter_family("glm46v")
class GLM46VAdapter(HuggingFaceChatAdapter):
    """Trainable adapter for GLM-4.6V-Flash (stub).

    Inherits standard HF chat-template operations from
    :class:`HuggingFaceChatAdapter`.  Structural methods require
    real-model discovery before they can be implemented.

    The profile is **required** and must come from a YAML file.
    """

    def __init__(self, profile: ModelFamilyProfile):
        self._profile = profile

    @property
    def profile(self) -> ModelFamilyProfile:
        return self._profile

    # ------------------------------------------------------------------ #
    # GLM-specific hooks
    # ------------------------------------------------------------------ #

    def _model_auto_class(self):
        """GLM uses ``Glm4vForConditionalGeneration``."""
        try:
            from transformers import Glm4vForConditionalGeneration
            return Glm4vForConditionalGeneration
        except ImportError:
            raise ImportError(
                "Glm4vForConditionalGeneration not available. "
                "Requires transformers >= 5.0.0rc0."
            )

    def sanitize_model_inputs(
        self, inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """GLM requires dropping ``token_type_ids`` from forward inputs."""
        inputs.pop("token_type_ids", None)
        return inputs

    # ------------------------------------------------------------------ #
    # Methods requiring real-model discovery (stubs)
    # ------------------------------------------------------------------ #

    def resolve_lora_targets(self, model: torch.nn.Module) -> list[str]:
        raise NotImplementedError(
            "GLM-4.6V LoRA targets require runtime module tree inspection. "
            "Discover the language-tower attention projection names and "
            "freeze the scope_regex before enabling."
        )

    def language_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        raise NotImplementedError(
            "GLM-4.6V language layer path requires runtime inspection."
        )

    def language_hidden_size(self, model: torch.nn.Module) -> int:
        raise NotImplementedError(
            "GLM-4.6V hidden size requires runtime inspection."
        )

    def to_eval_backend(self, **kwargs) -> Any:
        """Convert to a generic :class:`AdapterEvalBackend`."""
        from ..adapter_eval_backend import AdapterEvalBackend

        return AdapterEvalBackend(adapter=self, **kwargs)


register_model_key("glm46v_flash", "glm46v")
