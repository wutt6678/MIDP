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

from .base import ModelFamilyProfile
from .hf_chat import HuggingFaceChatAdapter
from .registry import register_adapter_family, register_model_key

logger = logging.getLogger(__name__)


@register_adapter_family("internvl35")
class InternVL35Adapter(HuggingFaceChatAdapter):
    """Trainable adapter for InternVL3.5-8B-HF (stub).

    Inherits standard HF chat-template operations.  The language backbone
    is Qwen3-family, so LoRA targets and MANU specs are similar to Qwen3.5
    but must be scoped to the language submodule only.

    The profile is **required** and must come from a YAML file.
    """

    def __init__(self, profile: ModelFamilyProfile):
        self._profile = profile

    @property
    def profile(self) -> ModelFamilyProfile:
        return self._profile

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
        """Convert to a generic :class:`AdapterEvalBackend`."""
        from ..adapter_eval_backend import AdapterEvalBackend

        return AdapterEvalBackend(adapter=self, **kwargs)


register_model_key("internvl35_8b_hf", "internvl35")
