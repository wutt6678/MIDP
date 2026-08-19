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

from .base import ModelFamilyProfile, TrainableVLMAdapter
from .hf_chat import HuggingFaceChatAdapter
from .registry import register_adapter

logger = logging.getLogger(__name__)

_GLM46V_PROFILE = ModelFamilyProfile(
    key="glm46v_flash",
    model_id="zai-org/GLM-4.6V-Flash",
    revision="<PIN_EXACT_HF_COMMIT_SHA>",
    processor_id="zai-org/GLM-4.6V-Flash",
    processor_revision="<PIN_EXACT_HF_COMMIT_SHA>",
    adapter_name="glm46v",
    trust_remote_code=False,
    dtype="bfloat16",
    attn_implementation="sdpa",
    candidate_positive="Yes",
    candidate_negative="No",
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_scope="language_attention_only",
    lora_target_leaf_names=("q_proj", "k_proj", "v_proj", "o_proj"),
    lora_scope_regex="<DISCOVER_ON_REAL_MODEL_AND_FREEZE>",
    r2mu_candidate_layers=(),
    r2mu_n_select_layers=0,
    supports_prompting=False,
    supports_candidate_margin=False,
    supports_ga=False,
    supports_gd=False,
    supports_kl=False,
    supports_npo=False,
    supports_mmunlearner=False,
    supports_manu=False,
    supports_r2mu=False,
    min_transformers_version="5.0.0rc0",
    tested_transformers_version="5.14.1",
    requires_hf_auth=False,
)


class GLM46VAdapter(HuggingFaceChatAdapter):
    """Trainable adapter for GLM-4.6V-Flash (stub).

    Inherits standard HF chat-template operations from
    :class:`HuggingFaceChatAdapter`.  Structural methods require
    real-model discovery before they can be implemented.
    """

    @property
    def profile(self) -> ModelFamilyProfile:
        return _GLM46V_PROFILE

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
        raise NotImplementedError(
            "GLM-4.6V eval backend not yet implemented. "
            "Requires a VisionLanguageModel backend for GLM."
        )


@register_adapter("glm46v_flash")
def _create_glm46v() -> TrainableVLMAdapter:
    return GLM46VAdapter()
