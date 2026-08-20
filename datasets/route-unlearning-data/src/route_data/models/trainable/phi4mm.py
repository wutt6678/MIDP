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
from .registry import register_adapter_family, register_model_key

logger = logging.getLogger(__name__)


@register_adapter_family("phi4mm")
class Phi4MMAdapter(TrainableVLMAdapter):
    """Trainable adapter for Phi-4-multimodal-instruct (stub).

    Does NOT inherit from :class:`HuggingFaceChatAdapter` because Phi
    uses a fundamentally different rendering and batching path.
    All methods are stubs requiring real-model discovery.

    The profile is **required** and must come from a YAML file.
    """

    def __init__(self, profile: ModelFamilyProfile):
        self._profile = profile

    @property
    def profile(self) -> ModelFamilyProfile:
        return self._profile

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
        """Convert to a generic :class:`AdapterEvalBackend`."""
        from ..adapter_eval_backend import AdapterEvalBackend

        return AdapterEvalBackend(adapter=self, **kwargs)


register_model_key("phi4_mm", "phi4mm")
