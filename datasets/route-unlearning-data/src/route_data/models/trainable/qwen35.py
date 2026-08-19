"""Qwen3.5 trainable adapter for the MIDP unlearning pipeline.

Ports the existing Qwen-specific logic from ``unlearning_harness.py`` into
the :class:`TrainableVLMAdapter` interface.  Supports both Qwen3.5-9B
(reference model) and Qwen3.5-4B (scale ablation) through different profiles.

Key Qwen-specific behaviours:

- ``enable_thinking=False`` in ``apply_chat_template`` to suppress the
  thinking block.
- Image-indexed keys: ``image_grid_thw``, ``image_sizes``, ``pixel_values``.
- ``mm_token_type_ids`` is a sequence-indexed text tensor.
- Pad token ID is 0.
- Language tower lives at ``model.model.layers`` with self-attention
  projections ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .base import ModelFamilyProfile, NeuronSpec, TrainableVLMAdapter
from .hf_chat import HuggingFaceChatAdapter
from .registry import register_adapter

logger = logging.getLogger(__name__)

# Default Qwen3.5-9B profile (frozen for reproducibility)
_QWEN35_9B_PROFILE = ModelFamilyProfile(
    key="qwen35_9b",
    model_id="Qwen/Qwen3.5-9B",
    revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    processor_id="Qwen/Qwen3.5-9B",
    processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    adapter_name="qwen35",
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
    lora_scope_regex=r"^model\.layers\.\d+\.self_attn\.",
    r2mu_candidate_layers=(7, 14, 21, 25),
    r2mu_n_select_layers=4,
    supports_prompting=True,
    supports_candidate_margin=True,
    supports_ga=True,
    supports_gd=True,
    supports_kl=True,
    supports_npo=True,
    supports_mmunlearner=True,
    supports_manu=True,
    supports_r2mu=True,
    min_transformers_version="5.0.0rc0",
    tested_transformers_version="5.14.1",
    requires_hf_auth=False,
)


class Qwen35Adapter(HuggingFaceChatAdapter):
    """Trainable adapter for the Qwen3.5 model family.

    Implements the :class:`TrainableVLMAdapter` interface by delegating
    standard HF chat-template operations to :class:`HuggingFaceChatAdapter`
    and providing Qwen-specific structural metadata.
    """

    def __init__(self, profile: ModelFamilyProfile | None = None):
        self._profile = profile or _QWEN35_9B_PROFILE

    @property
    def profile(self) -> ModelFamilyProfile:
        return self._profile

    # ------------------------------------------------------------------ #
    # Chat template hooks
    # ------------------------------------------------------------------ #

    def chat_template_kwargs(self) -> dict[str, Any]:
        """Qwen3.5 requires ``enable_thinking=False`` to suppress the
        ``<think>...</think>`` block in the assistant prefix."""
        return {"enable_thinking": False}

    # ------------------------------------------------------------------ #
    # Multimodal key management
    # ------------------------------------------------------------------ #

    def image_indexed_keys(self) -> frozenset[str]:
        """Keys whose dim 0 is num_images/tiles, not batch."""
        return frozenset({"image_grid_thw", "image_sizes", "pixel_values"})

    def required_multimodal_keys(self) -> frozenset[str]:
        return frozenset({"pixel_values"})

    # ------------------------------------------------------------------ #
    # Pad token
    # ------------------------------------------------------------------ #

    def pad_token_id(self, processor: Any) -> int:
        """Qwen pad token ID is 0."""
        return 0

    def _default_pad_token_id(self) -> int:
        return 0

    # ------------------------------------------------------------------ #
    # Sequence-indexed keys (for collation)
    # ------------------------------------------------------------------ #

    def _sequence_indexed_keys_in_batch(
        self, batch: list[dict[str, Any]],
    ) -> set[str]:
        """Qwen uses ``mm_token_type_ids`` as a sequence-indexed tensor."""
        keys: set[str] = set()
        if "mm_token_type_ids" in batch[0]:
            keys.add("mm_token_type_ids")
        return keys

    # ------------------------------------------------------------------ #
    # LoRA target resolution
    # ------------------------------------------------------------------ #

    def resolve_lora_targets(
        self,
        model: torch.nn.Module,
    ) -> list[str]:
        """Resolve language-tower attention targets for Qwen3.5.

        The Qwen3.5 language model structure is::

            model.model.layers.{i}.self_attn.{q,k,v,o}_proj

        The scope regex ensures only language-tower modules are selected.
        """
        return self._resolve_targets_with_regex(
            model,
            leaf_names=self._profile.lora_target_leaf_names,
            scope_regex=self._profile.lora_scope_regex,
        )

    # ------------------------------------------------------------------ #
    # Language layer access
    # ------------------------------------------------------------------ #

    def language_layers(
        self,
        model: torch.nn.Module,
    ) -> list[torch.nn.Module]:
        """Return the Qwen3.5 language transformer layers.

        Path: ``model.model.layers``.
        """
        return list(model.model.layers)

    def language_hidden_size(
        self,
        model: torch.nn.Module,
    ) -> int:
        """Return the hidden dimension of the Qwen3.5 language model."""
        return model.config.hidden_size

    # ------------------------------------------------------------------ #
    # MANU neuron specifications
    # ------------------------------------------------------------------ #

    def manu_neuron_specs(
        self,
        model: torch.nn.Module,
    ) -> list[NeuronSpec]:
        """Return MANU neuron specs for Qwen3.5 language-backbone MLPs.

        Qwen3.5 uses a gated MLP with ``gate_proj``, ``up_proj``, and
        ``down_proj``.  The intermediate size is ``model.config.intermediate_size``.
        """
        layers = self.language_layers(model)
        intermediate_size = model.config.intermediate_size

        specs: list[NeuronSpec] = []
        for i, layer in enumerate(layers):
            layer_name = f"model.layers.{i}.mlp"
            specs.append(NeuronSpec(
                layer_name=layer_name,
                neuron_count=intermediate_size,
                input_projection_names=("gate_proj", "up_proj"),
                output_projection_name="down_proj",
                input_axis=0,   # neuron index along intermediate dim
                output_axis=1,  # output dim maps to hidden_size
            ))
        return specs

    # ------------------------------------------------------------------ #
    # Eval backend conversion
    # ------------------------------------------------------------------ #

    def to_eval_backend(
        self,
        *,
        model: torch.nn.Module,
        processor: Any,
        model_config: Any,
        adapter_metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Convert to a :class:`QwenHFBackend` for post-unlearning evaluation."""
        from ..qwen import QwenHFBackend

        backend = QwenHFBackend(model_config)
        backend._model = model
        backend._processor = processor
        backend._adapter_metadata = adapter_metadata
        return backend


# ------------------------------------------------------------------ #
# Registration
# ------------------------------------------------------------------ #

@register_adapter("qwen35_9b")
def _create_qwen35_9b() -> TrainableVLMAdapter:
    return Qwen35Adapter(_QWEN35_9B_PROFILE)


# Qwen3.5-4B shares the same adapter class with a different profile.
# The profile is loaded from YAML at runtime; register a placeholder
# factory that accepts a profile override.
@register_adapter("qwen35_4b")
def _create_qwen35_4b() -> TrainableVLMAdapter:
    """Factory for Qwen3.5-4B scale ablation.

    Uses the same adapter class as Qwen3.5-9B but with a profile
    that pins the 4B checkpoint.
    """
    profile_4b = ModelFamilyProfile(
        key="qwen35_4b",
        model_id="Qwen/Qwen3.5-4B",
        revision="<PIN_EXACT_HF_COMMIT_SHA>",
        processor_id="Qwen/Qwen3.5-4B",
        processor_revision="<PIN_EXACT_HF_COMMIT_SHA>",
        adapter_name="qwen35",
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
        lora_scope_regex=r"^model\.layers\.\d+\.self_attn\.",
        r2mu_candidate_layers=(7, 14, 21, 25),
        r2mu_n_select_layers=4,
        supports_prompting=True,
        supports_candidate_margin=True,
        supports_ga=True,
        supports_gd=True,
        supports_kl=True,
        supports_npo=True,
        supports_mmunlearner=True,
        supports_manu=True,
        supports_r2mu=True,
        min_transformers_version="5.0.0rc0",
        tested_transformers_version="5.14.1",
        requires_hf_auth=False,
    )
    return Qwen35Adapter(profile_4b)
