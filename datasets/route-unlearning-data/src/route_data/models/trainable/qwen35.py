"""Qwen3.5 trainable adapter for the MIDP unlearning pipeline.

Ports the existing Qwen-specific logic from ``unlearning_harness.py`` into
the :class:`TrainableVLMAdapter` interface.  Supports both Qwen3.5-9B
(reference model) and Qwen3.5-4B (scale ablation) through different profiles.

Key Qwen-specific behaviours:

- ``enable_thinking=False`` in ``apply_chat_template`` to suppress the
  thinking block.
- Image-indexed keys: ``image_grid_thw``, ``image_sizes``, ``pixel_values``.
- ``mm_token_type_ids`` is a sequence-indexed text tensor.
- Pad token ID is resolved from the processor (not hard-coded).
- Language tower lives at ``model.model.language_model.layers`` (composite
  multimodal model) with self-attention projections
  ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``.
- Language config lives at ``model.config.text_config`` (composite model)
  or ``model.config`` (standalone).  The adapter resolves both.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .base import ModelFamilyProfile, NeuronSpec
from .hf_chat import HuggingFaceChatAdapter
from .registry import register_adapter_family, register_model_key

logger = logging.getLogger(__name__)


@register_adapter_family("qwen35")
class Qwen35Adapter(HuggingFaceChatAdapter):
    """Trainable adapter for the Qwen3.5 model family.

    Implements the :class:`TrainableVLMAdapter` interface by delegating
    standard HF chat-template operations to :class:`HuggingFaceChatAdapter`
    and providing Qwen-specific structural metadata.

    The profile is **required** and must come from a YAML file.  There are
    no hard-coded internal defaults.
    """

    def __init__(self, profile: ModelFamilyProfile):
        self._profile = profile

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
        """Resolve pad token ID from the processor (fail-closed)."""
        tokenizer = getattr(processor, "tokenizer", processor)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            raise RuntimeError("Qwen processor has no pad_token_id")
        return int(pad_id)

    def _default_pad_token_id(self) -> int:
        """Fail-closed: pad token must come from processor."""
        raise RuntimeError(
            "pad_token_id must be resolved from processor"
        )

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
    # Language config resolution (P0 — composite vs standalone)
    # ------------------------------------------------------------------ #

    @staticmethod
    def language_config(model: torch.nn.Module) -> Any:
        """Resolve the language sub-config from a (possibly composite) model.

        For Qwen3.5 VL the text config lives at ``model.config.text_config``.
        For standalone language models it is ``model.config`` itself.
        """
        cfg = getattr(model.config, "text_config", None)
        if cfg is not None:
            return cfg
        return model.config

    # ------------------------------------------------------------------ #
    # LoRA target resolution
    # ------------------------------------------------------------------ #

    def resolve_lora_targets(
        self,
        model: torch.nn.Module,
    ) -> list[str]:
        """Resolve language-tower attention targets for Qwen3.5.

        The Qwen3.5 composite language model structure is::

            model.model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj

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

        Path: ``model.model.language_model.layers`` (composite multimodal).
        """
        language_model = getattr(model.model, "language_model", None)
        if language_model is None:
            raise RuntimeError(
                "Qwen composite model has no model.language_model attribute"
            )
        layers = getattr(language_model, "layers", None)
        if layers is None:
            raise RuntimeError(
                "Qwen language_model has no layers attribute"
            )
        return list(layers)

    def language_hidden_size(
        self,
        model: torch.nn.Module,
    ) -> int:
        """Return the hidden dimension of the Qwen3.5 language model.

        Uses :meth:`language_config` to resolve the correct config node.
        """
        cfg = self.language_config(model)
        return cfg.hidden_size

    def language_intermediate_size(
        self,
        model: torch.nn.Module,
    ) -> int:
        """Return the MLP intermediate dimension of the Qwen3.5 language model."""
        cfg = self.language_config(model)
        return int(cfg.intermediate_size)

    # ------------------------------------------------------------------ #
    # MANU neuron specifications
    # ------------------------------------------------------------------ #

    def manu_neuron_specs(
        self,
        model: torch.nn.Module,
    ) -> list[NeuronSpec]:
        """Return MANU neuron specs for Qwen3.5 language-backbone MLPs.

        Qwen3.5 uses a gated MLP with ``gate_proj``, ``up_proj``, and
        ``down_proj``.  The intermediate size comes from the language
        config (which may be nested under ``text_config``).
        """
        layers = self.language_layers(model)
        cfg = self.language_config(model)
        intermediate_size = cfg.intermediate_size

        specs: list[NeuronSpec] = []
        for i, layer in enumerate(layers):
            layer_name = f"model.language_model.layers.{i}.mlp"
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
        """Convert to an :class:`AdapterEvalBackend` for evaluation.

        Uses the generic adapter-backed backend (§1.2 of multi-model plan)
        which delegates prefix construction and candidate tokenization to
        this adapter, ensuring model-agnostic evaluation.
        """
        from ..adapter_eval_backend import AdapterEvalBackend

        return AdapterEvalBackend(
            adapter=self,
            model=model,
            processor=processor,
            model_config=model_config,
            adapter_metadata=adapter_metadata,
        )


# Register model keys that share the qwen35 adapter family
register_model_key("qwen35_9b", "qwen35")
register_model_key("qwen35_4b", "qwen35")
