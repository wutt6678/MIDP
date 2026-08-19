"""Phi-4-multimodal-instruct trainable adapter stub.

Dedicated adapter required — Phi does NOT follow the standard
``AutoModelForImageTextToText`` loading path.

Key differences from Qwen/GLM/InternVL:

- Loads with ``AutoModelForCausalLM`` (not image-text-to-text).
- Custom multimodal fields: ``input_image_embeds``,
  ``image_attention_mask``, ``image_sizes``, ``input_mode``.
- Fused ``qkv_proj`` + ``o_proj`` (not separate q/k/v/o).
- Fused ``gate_up_proj`` + ``down_proj`` MLP (not gate/up/down).
- Bundled vision/speech LoRA structures must be detected and avoided.
- Official: Transformers 4.47–4.48; MIDP env uses 5.14 — compatibility P0.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .base import ModelFamilyProfile, NeuronSpec, TrainableVLMAdapter
from .registry import register_adapter

logger = logging.getLogger(__name__)

_PHI4MM_PROFILE = ModelFamilyProfile(
    key="phi4_mm",
    model_id="microsoft/Phi-4-multimodal-instruct",
    revision="<PIN_EXACT_HF_COMMIT_SHA>",
    processor_id="microsoft/Phi-4-multimodal-instruct",
    processor_revision="<PIN_EXACT_HF_COMMIT_SHA>",
    adapter_name="phi4mm",
    trust_remote_code=True,
    dtype="bfloat16",
    attn_implementation="sdpa",
    candidate_positive="Yes",
    candidate_negative="No",
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_scope="language_attention_only",
    lora_target_leaf_names=("qkv_proj", "o_proj"),
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
    min_transformers_version="4.47.0",
    tested_transformers_version="5.14.1",
    requires_hf_auth=False,
)


class Phi4MMAdapter(TrainableVLMAdapter):
    """Trainable adapter for Phi-4-multimodal-instruct (stub).

    Does NOT inherit from :class:`HuggingFaceChatAdapter` because Phi
    uses a fundamentally different rendering and batching path.
    All methods are stubs requiring real-model discovery.
    """

    @property
    def profile(self) -> ModelFamilyProfile:
        return _PHI4MM_PROFILE

    # ------------------------------------------------------------------ #
    # Multimodal key management (Phi-specific)
    # ------------------------------------------------------------------ #

    def required_multimodal_keys(self) -> frozenset[str]:
        """Phi uses ``input_image_embeds`` instead of ``pixel_values``."""
        return frozenset({"input_image_embeds"})

    def image_indexed_keys(self) -> frozenset[str]:
        return frozenset({"input_image_embeds", "image_attention_mask", "image_sizes"})

    # ------------------------------------------------------------------ #
    # All abstract methods — stubs
    # ------------------------------------------------------------------ #

    def load_model_processor(self, **kwargs) -> tuple[Any, Any]:
        raise NotImplementedError(
            "Phi-4-MM requires AutoModelForCausalLM loading path. "
            "Implement after environment compatibility is verified."
        )

    def build_prefix(self, processor, *, image, prompt) -> dict[str, Any]:
        raise NotImplementedError("Phi-4-MM prefix building not yet implemented.")

    def build_supervised_example(
        self, processor, *, image, prompt, answer_text,
    ) -> dict[str, Any]:
        raise NotImplementedError("Phi-4-MM supervised example not yet implemented.")

    def candidate_token_ids(self, processor, text: str) -> list[int]:
        raise NotImplementedError("Phi-4-MM candidate token resolution not yet implemented.")

    def collate(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "Phi-4-MM requires a dedicated collator modeled on "
            "the official vision fine-tuning semantics."
        )

    def append_candidate(
        self, prefix: dict[str, Any], candidate_token_ids: list[int],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Phi-4-MM candidate append not yet implemented.")

    def resolve_lora_targets(self, model: torch.nn.Module) -> list[str]:
        raise NotImplementedError(
            "Phi-4-MM uses fused qkv_proj. Must inspect runtime module tree "
            "and verify no bundled vision/speech LoRA structures conflict."
        )

    def language_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        raise NotImplementedError(
            "Phi-4-MM language layer path requires runtime inspection. "
            "Expected 32 language layers."
        )

    def language_hidden_size(self, model: torch.nn.Module) -> int:
        raise NotImplementedError("Phi-4-MM hidden size requires runtime inspection.")

    def manu_neuron_specs(self, model: torch.nn.Module) -> list[NeuronSpec]:
        raise NotImplementedError(
            "Phi-4-MM uses fused gate_up_proj + down_proj. "
            "Write dedicated indexing logic."
        )

    def to_eval_backend(self, **kwargs) -> Any:
        raise NotImplementedError("Phi-4-MM eval backend not yet implemented.")


@register_adapter("phi4_mm")
def _create_phi4mm() -> TrainableVLMAdapter:
    return Phi4MMAdapter()
