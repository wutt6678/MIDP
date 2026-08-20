"""Gemma 3 12B IT trainable adapter.

Larger model with gated access.  Structural metadata discovered from
the real pinned checkpoint (revision 96b6f1eccf38110c56df3a15bffe176da04bfd80).

Official loading path: ``AutoModelForImageTextToText`` + ``AutoProcessor``.

Gemma-specific behaviour:

- Processor emits ``token_type_ids`` (similar to Qwen's ``mm_token_type_ids``).
- ``pixel_values`` shape: ``(1, 3, 896, 896)`` — larger image than Qwen/GLM.
- Language tower: 48 layers, hidden_size=3840, intermediate_size=15360.
- Attention projections: q_proj, k_proj, v_proj, o_proj (4 per layer).
- MLP projections: gate_proj, up_proj, down_proj (standard).
- Expected LoRA target count: 48 × 4 = 192 (language attention only).
- pad_token_id=0, eos_token_id=1.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import torch

from .base import ModelFamilyProfile, NeuronSpec
from .hf_chat import HuggingFaceChatAdapter
from .registry import register_adapter_family, register_model_key

logger = logging.getLogger(__name__)


@register_adapter_family("gemma3")
class Gemma3Adapter(HuggingFaceChatAdapter):
    """Trainable adapter for Gemma 3 12B IT."""

    def __init__(self, profile: ModelFamilyProfile):
        self._profile = profile

    @property
    def profile(self) -> ModelFamilyProfile:
        return self._profile

    # ------------------------------------------------------------------ #
    # Gemma-specific hooks
    # ------------------------------------------------------------------ #

    def _model_auto_class(self):
        """Gemma uses AutoModelForImageTextToText."""
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText

    def sanitize_model_inputs(
        self, inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Gemma may need token_type_ids dropped in some contexts."""
        # Keep token_type_ids — Gemma uses them for multimodal masking.
        return inputs

    def required_multimodal_keys(self) -> frozenset[str]:
        return frozenset({"pixel_values"})

    def image_indexed_keys(self) -> frozenset[str]:
        """Gemma pixel_values is batch-indexed (1, C, H, W)."""
        return frozenset()

    # ------------------------------------------------------------------ #
    # Structural methods (discovered from real model)
    # ------------------------------------------------------------------ #

    def resolve_lora_targets(self, model: torch.nn.Module) -> list[str]:
        """Resolve language-attention-only LoRA targets.

        Discovered: 48 language layers at ``model.language_model.layers.{i}``
        with attention projections ``self_attn.{q,k,v,o}_proj``.
        Expected count: 48 × 4 = 192.
        """
        scope_regex = self._profile.lora_scope_regex
        targets = []
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and re.match(scope_regex, name):
                targets.append(name)
        targets.sort()

        vision_targets = [
            t for t in targets
            if "visual" in t.lower() or "vision" in t.lower()
            or "projector" in t.lower() or "connector" in t.lower()
        ]
        if vision_targets:
            raise RuntimeError(
                f"Gemma LoRA scope matched {len(vision_targets)} "
                f"vision/projector targets: {vision_targets[:5]}"
            )

        logger.info(f"Gemma LoRA targets: {len(targets)} (language attention only)")
        return targets

    def language_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        layer_path = self._profile.language_layer_path
        layers_container = dict(model.named_modules())[layer_path]
        return [mod for _, mod in sorted(layers_container.named_children(), key=lambda x: int(x[0]) if x[0].isdigit() else 0)]

    def language_hidden_size(self, model: torch.nn.Module) -> int:
        return self._profile.language_hidden_size

    def language_intermediate_size(self, model: torch.nn.Module) -> int:
        return self._profile.intermediate_size

    def manu_neuron_specs(self, model: torch.nn.Module) -> list[NeuronSpec]:
        layers = self.language_layers(model)
        intermediate_size = self.language_intermediate_size(model)

        specs: list[NeuronSpec] = []
        for i, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
                specs.append(NeuronSpec(
                    layer_name=f"model.language_model.layers.{i}",
                    neuron_count=intermediate_size,
                    input_projection_names=("gate_proj", "up_proj"),
                    output_projection_name="down_proj",
                    input_axis=0,
                    output_axis=1,
                ))
        return specs

    def to_eval_backend(self, **kwargs) -> Any:
        from ..adapter_eval_backend import AdapterEvalBackend
        return AdapterEvalBackend(adapter=self, **kwargs)


register_model_key("gemma3_12b", "gemma3")
