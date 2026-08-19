"""InternVL3.5-8B-HF trainable adapter stub.

Architecture contrast model.  Uses the HF-format checkpoint with
``trust_remote_code=True``.  The language backbone is Qwen3-family
with 36 language layers.

Key facts from official config:

- ``AutoModelForImageTextToText`` + ``AutoProcessor``
- ``trust_remote_code=True``
- Transformers >= 4.52.1
- 36 language layers (Qwen3 backbone)
- Do NOT add InternVL's optional thinking-mode system prompt.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .base import ModelFamilyProfile, TrainableVLMAdapter
from .hf_chat import HuggingFaceChatAdapter
from .registry import register_adapter

logger = logging.getLogger(__name__)

_INTERNVL35_PROFILE = ModelFamilyProfile(
    key="internvl35_8b_hf",
    model_id="OpenGVLab/InternVL3_5-8B-HF",
    revision="<PIN_EXACT_HF_COMMIT_SHA>",
    processor_id="OpenGVLab/InternVL3_5-8B-HF",
    processor_revision="<PIN_EXACT_HF_COMMIT_SHA>",
    adapter_name="internvl35",
    trust_remote_code=True,
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
    min_transformers_version="4.52.1",
    tested_transformers_version="5.14.1",
    requires_hf_auth=False,
)


class InternVL35Adapter(HuggingFaceChatAdapter):
    """Trainable adapter for InternVL3.5-8B-HF (stub).

    Inherits standard HF chat-template operations.  The language backbone
    is Qwen3-family, so LoRA targets and MANU specs are similar to Qwen3.5
    but must be scoped to the language submodule only.
    """

    @property
    def profile(self) -> ModelFamilyProfile:
        return _INTERNVL35_PROFILE

    # ------------------------------------------------------------------ #
    # Methods requiring real-model discovery (stubs)
    # ------------------------------------------------------------------ #

    def resolve_lora_targets(self, model: torch.nn.Module) -> list[str]:
        raise NotImplementedError(
            "InternVL3.5 LoRA targets require runtime module tree inspection. "
            "The language backbone is Qwen3-family; discover the exact path "
            "to language-tower attention projections."
        )

    def language_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        raise NotImplementedError(
            "InternVL3.5 language layer path requires runtime inspection. "
            "Expected 36 layers in the Qwen3 language backbone."
        )

    def language_hidden_size(self, model: torch.nn.Module) -> int:
        raise NotImplementedError(
            "InternVL3.5 hidden size requires runtime inspection."
        )

    def to_eval_backend(self, **kwargs) -> Any:
        raise NotImplementedError(
            "InternVL3.5 eval backend not yet implemented."
        )


@register_adapter("internvl35_8b_hf")
def _create_internvl35() -> TrainableVLMAdapter:
    return InternVL35Adapter()
