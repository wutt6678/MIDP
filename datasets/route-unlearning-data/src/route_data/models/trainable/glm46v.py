"""GLM-4.6V-Flash trainable adapter.

Highest-priority new model.  Structural metadata discovered from the
real pinned checkpoint (revision 411bb4d77144a3f03accbf4b780f5acb8b7cde4e).

Official loading path: ``AutoModelForMultimodalLM`` (or
``Glm4vForConditionalGeneration``) + ``AutoProcessor``.
Requires Transformers >= 5.0.0rc0.

GLM-specific behaviour:

- Processor emits ``mm_token_type_ids`` (similar to Qwen).
- Processor does NOT emit ``image_sizes`` (unlike Qwen).
- Language tower: 40 layers, hidden_size=4096, intermediate_size=13696.
- Attention projections: q_proj, k_proj, v_proj, o_proj (4 per layer).
- MLP projections: gate_proj, up_proj, down_proj (standard, not fused).
- Expected LoRA target count: 40 × 4 = 160 (language attention only).
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


@register_adapter_family("glm46v")
class GLM46VAdapter(HuggingFaceChatAdapter):
    """Trainable adapter for GLM-4.6V-Flash.

    Inherits standard HF chat-template operations from
    :class:`HuggingFaceChatAdapter`.  Structural methods use the
    runtime-discovered values from the pinned checkpoint.
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
        """GLM uses ``AutoModelForMultimodalLM`` (preferred) or concrete class."""
        try:
            from transformers import AutoModelForMultimodalLM
            return AutoModelForMultimodalLM
        except ImportError:
            try:
                from transformers import Glm4vForConditionalGeneration
                return Glm4vForConditionalGeneration
            except ImportError:
                raise ImportError(
                    "Neither AutoModelForMultimodalLM nor "
                    "Glm4vForConditionalGeneration available. "
                    "Requires transformers >= 5.0.0rc0."
                )

    def sanitize_model_inputs(
        self, inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """GLM requires dropping ``token_type_ids`` from forward inputs."""
        inputs.pop("token_type_ids", None)
        return inputs

    def required_multimodal_keys(self) -> frozenset[str]:
        """GLM requires pixel_values (no image_sizes)."""
        return frozenset({"pixel_values"})

    def image_indexed_keys(self) -> frozenset[str]:
        """GLM pixel_values has dim 0 = num_tiles, not batch."""
        return frozenset({"image_grid_thw", "pixel_values"})

    # ------------------------------------------------------------------ #
    # Structural methods (discovered from real model)
    # ------------------------------------------------------------------ #

    def resolve_lora_targets(self, model: torch.nn.Module) -> list[str]:
        """Resolve language-attention-only LoRA targets.

        Discovered: 40 language layers at ``model.language_model.layers.{i}``
        with attention projections ``self_attn.{q,k,v,o}_proj``.
        Expected count: 40 × 4 = 160.
        """
        scope_regex = self._profile.lora_scope_regex
        targets = []
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                if re.match(scope_regex, name):
                    targets.append(name)
        targets.sort()

        # Post-discovery assertion: vision/projector/connector count = 0.
        vision_targets = [
            t for t in targets
            if "visual" in t.lower() or "vision" in t.lower()
            or "projector" in t.lower() or "connector" in t.lower()
        ]
        if vision_targets:
            raise RuntimeError(
                f"GLM LoRA scope matched {len(vision_targets)} "
                f"vision/projector targets: {vision_targets[:5]}"
            )

        logger.info(
            f"GLM LoRA targets: {len(targets)} "
            f"(language attention only, scope_regex={scope_regex})"
        )
        return targets

    def language_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        """Return the 40 language transformer layers."""
        layer_path = self._profile.language_layer_path  # "model.language_model.layers"
        layers_container = dict(model.named_modules())[layer_path]
        layers = []
        for name, mod in sorted(layers_container.named_children(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            layers.append(mod)
        return layers

    def language_hidden_size(self, model: torch.nn.Module) -> int:
        """Return 4096 (discovered from text_config)."""
        return self._profile.language_hidden_size

    def language_intermediate_size(self, model: torch.nn.Module) -> int:
        """Return 13696 (discovered from text_config)."""
        return self._profile.intermediate_size

    def manu_neuron_specs(self, model: torch.nn.Module) -> list[NeuronSpec]:
        """Return MANU neuron specs for GLM language MLP layers.

        GLM uses standard gate_proj/up_proj/down_proj (not fused).
        """
        layers = self.language_layers(model)
        if not layers:
            raise RuntimeError("No language layers found")

        hidden_size = self.language_hidden_size(model)
        intermediate_size = self.language_intermediate_size(model)

        specs: list[NeuronSpec] = []
        for i, layer in enumerate(layers):
            # Verify the layer has the expected MLP structure.
            mlp = None
            for attr in ("mlp", "feed_forward"):
                if hasattr(layer, attr):
                    mlp = getattr(layer, attr)
                    break
            if mlp is None:
                continue

            # Check for standard gate/up/down projections.
            has_gate = hasattr(mlp, "gate_proj")
            has_up = hasattr(mlp, "up_proj")
            has_down = hasattr(mlp, "down_proj")

            if has_gate and has_up and has_down:
                specs.append(NeuronSpec(
                    layer_name=f"model.language_model.layers.{i}",
                    neuron_count=intermediate_size,
                    input_projection_names=("gate_proj", "up_proj"),
                    output_projection_name="down_proj",
                    input_axis=0,   # intermediate dim is axis 0 for gate/up
                    output_axis=1,  # hidden dim is axis 1 for down
                ))

        return specs

    def to_eval_backend(self, **kwargs) -> Any:
        """Convert to a generic :class:`AdapterEvalBackend`."""
        from ..adapter_eval_backend import AdapterEvalBackend

        return AdapterEvalBackend(adapter=self, **kwargs)


register_model_key("glm46v_flash", "glm46v")
