"""InternVL3.5-8B-HF trainable adapter.

Architecture contrast model: different vision stack from Qwen/GLM but
shares Qwen3 language backbone.  Structural metadata discovered from
the real pinned checkpoint (revision 741a7d03020411e666c6109218ab71e08151ef86).

Official loading path: ``AutoModelForMultimodalLM`` with
``trust_remote_code=True`` + ``AutoProcessor``.

InternVL-specific behaviour:

- Processor emits only ``pixel_values`` (no ``mm_token_type_ids``,
  no ``image_grid_thw``, no ``image_sizes``).
- ``pixel_values`` shape: ``(1, 3, 448, 448)`` — standard image tensor.
- Language tower: 36 layers, hidden_size=4096, intermediate_size=12288.
- Attention projections: q_proj, k_proj, v_proj, o_proj (4 per layer).
- MLP projections: gate_proj, up_proj, down_proj (standard).
- Expected LoRA target count: 36 × 4 = 144 (language attention only).
- pad_token_id=151643, eos_token_id=151645.
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


@register_adapter_family("internvl35")
class InternVL35Adapter(HuggingFaceChatAdapter):
    """Trainable adapter for InternVL3.5-8B-HF."""

    def __init__(self, profile: ModelFamilyProfile):
        self._profile = profile

    @property
    def profile(self) -> ModelFamilyProfile:
        return self._profile

    # ------------------------------------------------------------------ #
    # InternVL-specific hooks
    # ------------------------------------------------------------------ #

    def _model_auto_class(self):
        """InternVL uses AutoModelForMultimodalLM with trust_remote_code."""
        from transformers import AutoModelForMultimodalLM
        return AutoModelForMultimodalLM

    def required_multimodal_keys(self) -> frozenset[str]:
        """InternVL only requires pixel_values."""
        return frozenset({"pixel_values"})

    def image_indexed_keys(self) -> frozenset[str]:
        """InternVL pixel_values is batch-indexed (1, C, H, W)."""
        return frozenset()

    # ------------------------------------------------------------------ #
    # Structural methods (discovered from real model)
    # ------------------------------------------------------------------ #

    def resolve_lora_targets(self, model: torch.nn.Module) -> list[str]:
        """Resolve language-attention-only LoRA targets.

        Discovered: 36 language layers at ``model.language_model.layers.{i}``
        with attention projections ``self_attn.{q,k,v,o}_proj``.
        Expected count: 36 × 4 = 144.
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
                f"InternVL LoRA scope matched {len(vision_targets)} "
                f"vision/projector targets: {vision_targets[:5]}"
            )

        logger.info(
            f"InternVL LoRA targets: {len(targets)} "
            f"(language attention only)"
        )
        return targets

    def language_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        """Return the 36 language transformer layers."""
        layer_path = self._profile.language_layer_path
        layers_container = dict(model.named_modules())[layer_path]
        return [mod for _, mod in sorted(layers_container.named_children(), key=lambda x: int(x[0]) if x[0].isdigit() else 0)]

    def language_hidden_size(self, model: torch.nn.Module) -> int:
        return self._profile.language_hidden_size

    def language_intermediate_size(self, model: torch.nn.Module) -> int:
        return self._profile.intermediate_size

    def manu_neuron_specs(self, model: torch.nn.Module) -> list[NeuronSpec]:
        """MANU specs for InternVL language MLP (standard gate/up/down)."""
        layers = self.language_layers(model)
        _hidden_size = self.language_hidden_size(model)
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
        """Convert to a generic :class:`AdapterEvalBackend`."""
        from ..adapter_eval_backend import AdapterEvalBackend
        return AdapterEvalBackend(adapter=self, **kwargs)


register_model_key("internvl35_8b_hf", "internvl35")
