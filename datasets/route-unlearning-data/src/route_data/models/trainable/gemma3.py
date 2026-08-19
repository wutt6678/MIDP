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

from .base import ModelFamilyProfile, TrainableVLMAdapter
from .hf_chat import HuggingFaceChatAdapter
from .registry import register_adapter

logger = logging.getLogger(__name__)

_GEMMA3_PROFILE = ModelFamilyProfile(
    key="gemma3_12b",
    model_id="google/gemma-3-12b-it",
    revision="<PIN_EXACT_HF_COMMIT_SHA>",
    processor_id="google/gemma-3-12b-it",
    processor_revision="<PIN_EXACT_HF_COMMIT_SHA>",
    adapter_name="gemma3",
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
    min_transformers_version="4.50.0",
    tested_transformers_version="5.14.1",
    requires_hf_auth=True,
)


class Gemma3Adapter(HuggingFaceChatAdapter):
    """Trainable adapter for Gemma3-12B-it (stub).

    Inherits standard HF chat-template operations from
    :class:`HuggingFaceChatAdapter`.  Structural methods require
    real-model discovery.
    """

    @property
    def profile(self) -> ModelFamilyProfile:
        return _GEMMA3_PROFILE

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
        raise NotImplementedError(
            "Gemma3 eval backend not yet implemented."
        )


@register_adapter("gemma3_12b")
def _create_gemma3() -> TrainableVLMAdapter:
    return Gemma3Adapter()
