"""Phi-4-multimodal-instruct trainable adapter.

Independent replication model.  Structural metadata discovered from
the real pinned checkpoint (revision 93f923e1a7727d1c4f446756212d9d3e8fcc5d81).

Official loading path: ``AutoModelForCausalLM`` (NOT image-text-to-text)
with ``trust_remote_code=True``.

**Environment requirements:**

- ``transformers >= 4.47, < 4.49`` (custom code incompatible with 5.x).
- SDPA monkey-patches (no flash_attn available on this system):
  1. ``PreTrainedModel._check_and_enable_flash_attn_2`` → noop.
  2. ``modeling_flash_attention_utils._flash_attention_forward`` → SDPA.
  3. ``vision_siglip_navit.SiglipFlashAttention2._flash_attention_forward``
     → SDPA.

Phi-specific behaviour:

- Loads with ``AutoModelForCausalLM`` (not image-text-to-text).
- Custom multimodal fields: ``input_image_embeds``,
  ``image_attention_mask``, ``image_sizes``, ``input_mode``.
- Fused ``qkv_proj`` + ``o_proj`` (not separate q/k/v/o).
- Fused ``gate_up_proj`` + ``down_proj`` MLP (not gate/up/down).
- Bundled vision/speech LoRA structures: target LoraLayer wrappers directly
  (NOT ``.base_layer``); PEFT ``update_layer()`` adds our adapter alongside.
- Image token: ``<|endoftext10|>`` (ID 200010).
- Language tower: 32 layers, hidden_size=3072, intermediate_size=8192.
- Expected LoRA target count: 32 × 2 = 64 (language attention only).
- pad_token_id=199999, eos_token_id=199999.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import torch

from .base import ModelFamilyProfile, NeuronSpec, TrainableVLMAdapter
from .registry import register_adapter_family, register_model_key

logger = logging.getLogger(__name__)

# Image token for Phi-4-MM
PHI_IMAGE_TOKEN = "<|endoftext10|>"
PHI_IMAGE_TOKEN_ID = 200010


class _PhiInnerModelWrapper(torch.nn.Module):
    """Wrapper around Phi's inner model for unlearning.

    Replicates the outer ``Phi4MMForCausalLM.forward()`` logic that
    the normal model uses to:

    1. Interpret ``input_mode`` to activate the correct bundled LoRA
       adapter (vision / speech / language).
    2. Set ``audio_projection_mode`` accordingly.
    3. Add the ``lm_head`` to produce logits (the inner model only
       returns hidden states).

    ``input_mode`` is consumed and removed before passing kwargs to
    the inner model.

    Multi-adapter composition
    -------------------------
    Instead of calling ``outer.set_lora_adapter()`` (which activates
    only ONE adapter), we set ``_active_adapter`` on every LoraLayer
    to a *list* of adapters.  PEFT's forward sums contributions from
    all active adapters::

        result = base(x) + sum(lora_B(lora_A(x)) * scaling
                               for each active_adapter)

    So for visual input::

        active = ["vision", "unlearning"]
        → base + vision_LoRA + unlearning_LoRA

    For text-only input::

        active = ["unlearning"]
        → base + unlearning_LoRA

    Invariant
    ---------
    - Outer Phi model (baseline): ``input_mode`` retained.
    - Inner unlearning wrapper: ``input_mode`` interpreted, then removed.
    """

    # InputMode enum values (from processing_phi4mm.py)
    _LANGUAGE = 0
    _VISION = 1
    _SPEECH = 2
    _VISION_SPEECH = 3

    def __init__(
        self,
        inner_model: torch.nn.Module,
        lm_head: torch.nn.Module,
    ):
        super().__init__()
        self.inner_model = inner_model
        self.lm_head = lm_head

    def _set_active_adapters(self, adapter_names: list[str]) -> None:
        """Set active adapters on all LoraLayers in the inner model.

        Uses ``_active_adapter`` directly (not ``set_adapter()``) to
        avoid PEFT's unmerge/merge side effects.
        """
        from peft.tuners.lora.layer import LoraLayer
        for mod in self.inner_model.modules():
            if isinstance(mod, LoraLayer):
                mod._active_adapter = adapter_names

    def forward(self, **kwargs):
        # --- Consume input_mode (P0-1) ---
        input_mode = kwargs.pop("input_mode", None)
        if input_mode is not None:
            if isinstance(input_mode, torch.Tensor):
                input_mode = input_mode[0].item()
            input_mode = int(input_mode)

            if input_mode in (self._VISION, self._VISION_SPEECH):
                kwargs["audio_projection_mode"] = "vision"
                self._set_active_adapters(["vision", "unlearning"])
            elif input_mode == self._SPEECH:
                kwargs["audio_projection_mode"] = "speech"
                self._set_active_adapters(["speech", "unlearning"])
            elif input_mode == self._LANGUAGE:
                kwargs["audio_projection_mode"] = "speech"
                self._set_active_adapters(["unlearning"])
        else:
            kwargs.setdefault("audio_projection_mode", "speech")
            # Default to unlearning-only for text-only input
            self._set_active_adapters(["unlearning"])

        outputs = self.inner_model(**kwargs)
        # Add logits from the LM head (inner model only returns hidden states)
        from transformers.modeling_outputs import CausalLMOutputWithPast
        logits = self.lm_head(outputs[0])

        # Compute loss if labels are provided (replicates the outer
        # Phi4MMForCausalLM loss computation).
        loss = None
        labels = kwargs.get("labels")
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def _apply_sdpa_patches() -> None:
    """Monkey-patch transformers to use SDPA instead of flash_attn.

    Required because Phi-4-MM's custom code forces flash_attention_2
    but flash_attn is not installable on this system (CUDA version
    mismatch).  These patches redirect all flash attention calls to
    PyTorch's native ``scaled_dot_product_attention``.
    """
    import torch.nn.functional as F
    from transformers.modeling_utils import PreTrainedModel

    # 1. Skip flash attention availability check
    @classmethod
    def _noop_fa2(cls, config, **kwargs):  # type: ignore[misc]
        return config

    PreTrainedModel._check_and_enable_flash_attn_2 = _noop_fa2

    # 2. Replace language model's _flash_attention_forward with SDPA
    import transformers.modeling_flash_attention_utils as mfa

    def _sdpa_flash(
        query_states, key_states, value_states,
        attention_mask, query_length,
        dropout=0.0, softmax_scale=None, sliding_window=None, **kwargs,
    ):
        # [batch, num_heads, seq_len, head_dim]
        q2 = query_states.transpose(1, 2)
        k2 = key_states.transpose(1, 2)
        v2 = value_states.transpose(1, 2)

        # Determine causal masking:
        # - Prefill (query_length > 1): causal = True
        # - Cached decode (query_length == 1): causal = False
        #   (the single query token attends to all prior keys)
        is_causal = query_length > 1 if query_length is not None else q2.shape[1] > 1

        # Build SDPA attention mask from the incoming attention_mask.
        # The incoming mask shape varies:
        #   [batch, key_length]             (2D padding mask)
        #   [batch, 1, query_length, key_length]  (4D expanded)
        #   [batch, query_length, key_length]     (3D)
        sdpa_attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Padding mask: [batch, key_len] → [batch, 1, 1, key_len]
                sdpa_attn_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                # [batch, query_len, key_len] → [batch, 1, query_len, key_len]
                sdpa_attn_mask = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 4:
                sdpa_attn_mask = attention_mask

            # Convert additive masks (0/-inf) to boolean for SDPA
            if sdpa_attn_mask is not None and sdpa_attn_mask.dtype != torch.bool:
                sdpa_attn_mask = sdpa_attn_mask > -1e4

            # If causal + mask, combine: where mask is True, still apply causal
            if is_causal and sdpa_attn_mask is not None:
                # Use the explicit mask (it should already encode causal)
                is_causal = False

        out = F.scaled_dot_product_attention(
            q2, k2, v2,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout,
            is_causal=is_causal if sdpa_attn_mask is None else False,
            scale=softmax_scale,
        )
        return out.transpose(1, 2)

    mfa._flash_attention_forward = _sdpa_flash


def _patch_vision_attention() -> None:
    """Patch the vision encoder's flash attention to use SDPA.

    Must be called AFTER the model is loaded (the vision module is
    only imported during model loading with ``trust_remote_code=True``).
    """
    import sys

    import torch.nn.functional as F

    for mod_name, mod in list(sys.modules.items()):
        if mod is not None and "vision_siglip_navit" in mod_name:
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if isinstance(attr, type) and hasattr(attr, "_flash_attention_forward"):

                    def _sdpa_vision(
                        self,
                        query_states, key_states, value_states,
                        attention_mask, query_length,
                        dropout=0.0, softmax_scale=None,
                    ):
                        causal = self.is_causal and query_length != 1
                        q2 = query_states.transpose(1, 2)
                        k2 = key_states.transpose(1, 2)
                        v2 = value_states.transpose(1, 2)

                        # Pass attention_mask through to SDPA.
                        # The vision encoder may produce 2D padding masks
                        # [batch, key_len] or 4D expanded masks.
                        sdpa_attn_mask = None
                        if attention_mask is not None and attention_mask.dim() >= 2:
                            if attention_mask.dim() == 2:
                                sdpa_attn_mask = attention_mask[:, None, None, :]
                            elif attention_mask.dim() == 3:
                                sdpa_attn_mask = attention_mask[:, None, :, :]
                            elif attention_mask.dim() == 4:
                                sdpa_attn_mask = attention_mask
                            # Convert additive masks (0/-inf) to boolean
                            if sdpa_attn_mask.dtype != torch.bool:
                                sdpa_attn_mask = sdpa_attn_mask > -1e4
                            causal = False  # explicit mask supersedes causal

                        out = F.scaled_dot_product_attention(
                            q2, k2, v2,
                            attn_mask=sdpa_attn_mask,
                            dropout_p=dropout, is_causal=causal,
                            scale=softmax_scale,
                        )
                        return out.transpose(1, 2)

                    attr._flash_attention_forward = _sdpa_vision
                    logger.debug("Patched %s._flash_attention_forward → SDPA", attr_name)
                    break
            break


@register_adapter_family("phi4mm")
class Phi4MMAdapter(TrainableVLMAdapter):
    """Trainable adapter for Phi-4-multimodal-instruct.

    Does NOT inherit from :class:`HuggingFaceChatAdapter` because Phi
    uses a fundamentally different rendering and batching path.
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
        return frozenset({"input_image_embeds", "image_attention_mask", "image_sizes"})

    def image_indexed_keys(self) -> frozenset[str]:
        return frozenset({"input_image_embeds", "image_attention_mask", "image_sizes"})

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def load_model_processor(self, **kwargs) -> tuple[Any, Any]:
        """Load Phi-4-MM with SDPA patches.

        Applies monkey-patches to bypass flash_attn requirement before
        loading, then patches vision attention after loading.

        Fails closed if the current transformers version is outside the
        supported range ``[min, max_exclusive)``.
        """
        # P0-3: fail-closed environment check BEFORE loading
        from .registry import validate_environment_compatibility

        env_errors = validate_environment_compatibility(self._profile)
        if env_errors:
            raise RuntimeError(
                "Incompatible Phi-4-MM environment:\n"
                + "\n".join(f"  - {e}" for e in env_errors)
            )

        from transformers import AutoModelForCausalLM, AutoProcessor

        model_id = kwargs.get("model_id", self._profile.model_id)
        revision = kwargs.get("revision", self._profile.revision)
        processor_revision = kwargs.get("processor_revision", self._profile.processor_revision)
        dtype_str = kwargs.get("dtype", self._profile.dtype)
        device = kwargs.get("device", "cuda:0")

        dtype = getattr(torch, dtype_str, torch.bfloat16)

        # Apply SDPA patches BEFORE loading
        _apply_sdpa_patches()

        logger.info("Loading Phi-4-MM processor (revision %s)...", processor_revision)
        processor = AutoProcessor.from_pretrained(
            model_id, revision=processor_revision, trust_remote_code=True,
        )

        logger.info("Loading Phi-4-MM model (revision %s)...", revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=dtype,
            trust_remote_code=True, device_map=device,
        )

        # Patch vision attention AFTER loading
        _patch_vision_attention()

        # Disable gradient checkpointing on the vision SiglipEncoder.
        # SiglipEncoder is an nn.Module (NOT PreTrainedModel) that
        # references _gradient_checkpointing_func in its forward(),
        # but that method is only provided by PreTrainedModel.
        # Since we don't train the vision tower, disabling is safe.
        for _mod in model.modules():
            if type(_mod).__name__ == "SiglipEncoder":
                _mod.gradient_checkpointing = False
                logger.debug("Disabled gradient_checkpointing on SiglipEncoder")

        # Disable gradient checkpointing on the audio processor.
        # The audio processor uses checkpoint() which breaks the grad
        # flow when inputs don't require grad (frozen audio tower).
        for _mod in model.modules():
            if hasattr(_mod, "gradient_checkpointing_disable") and callable(_mod.gradient_checkpointing_disable):
                try:
                    _mod.gradient_checkpointing_disable()
                    logger.debug("Disabled gradient_checkpointing on %s", type(_mod).__name__)
                except Exception:
                    pass

        logger.info("Phi-4-MM loaded successfully with SDPA patches")
        return model, processor

    # ------------------------------------------------------------------ #
    # Prefix building
    # ------------------------------------------------------------------ #

    def build_prefix(
        self, processor: Any, *, image: Any, prompt: str,
    ) -> dict[str, Any]:
        """Build multimodal prefix using Phi's processor.

        Uses ``<|endoftext10|>`` as the image token and the standard
        Phi chat template: ``<|user|>\\n<image>\\n<prompt><|end|><|assistant|>\\n``.

        When ``image is None`` (name_only probes), the image token is
        omitted and the processor is called without ``images=``.
        """
        if image is None:
            chat = [{"role": "user", "content": prompt}]
            prompt_text = processor.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True,
            )
            inputs = processor(text=prompt_text, return_tensors="pt")
        else:
            chat = [
                {"role": "user", "content": f"{PHI_IMAGE_TOKEN}\n{prompt}"},
            ]
            prompt_text = processor.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True,
            )
            inputs = processor(
                text=prompt_text, images=[image], return_tensors="pt",
            )
        # Squeeze batch dim for text tensors (consistent with other adapters).
        # Keep image-indexed tensors and 1-D tensors (like ``input_mode``)
        # unsqueezed — the inner model expects them with their batch dim.
        image_indexed = self.image_indexed_keys()
        result = {}
        for k, v in inputs.items():
            if torch.is_tensor(v) and k not in image_indexed and v.dim() > 1 and v.shape[0] == 1:
                result[k] = v.squeeze(0)
            else:
                result[k] = v
        # NOTE: ``input_mode`` is PRESERVED (P0-1).
        # The outer Phi4MMForCausalLM consumes it to activate the correct
        # bundled LoRA adapter.  The inner _PhiInnerModelWrapper also
        # consumes it and strips it before calling the inner model.
        return result

    def build_supervised_example(
        self, processor: Any, *, image: Any, prompt: str, answer_text: str,
    ) -> dict[str, Any]:
        """Build a supervised training example with labels.

        When ``image is None`` (name_only probes), the image token is
        omitted and the processor is called without ``images=``.
        """
        if image is None:
            chat = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer_text},
            ]
            full_text = processor.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False,
            )
            inputs = processor(text=full_text, return_tensors="pt")
        else:
            chat = [
                {"role": "user", "content": f"{PHI_IMAGE_TOKEN}\n{prompt}"},
                {"role": "assistant", "content": answer_text},
            ]
            full_text = processor.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False,
            )
            inputs = processor(
                text=full_text, images=[image], return_tensors="pt",
            )
        result = {}
        image_indexed = self.image_indexed_keys()
        for k, v in inputs.items():
            if torch.is_tensor(v) and k not in image_indexed and v.dim() > 1 and v.shape[0] == 1:
                result[k] = v.squeeze(0)
            else:
                result[k] = v
        # NOTE: ``input_mode`` is PRESERVED (P0-1).  See build_prefix().

        # Build labels: mask out the prompt portion
        input_ids = result["input_ids"]
        # Find where the assistant response starts
        if image is None:
            prompt_chat = [{"role": "user", "content": prompt}]
        else:
            prompt_chat = [
                {"role": "user", "content": f"{PHI_IMAGE_TOKEN}\n{prompt}"},
            ]
        prompt_text = processor.tokenizer.apply_chat_template(
            prompt_chat, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = processor.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        result["labels"] = labels
        return result

    def candidate_token_ids(self, processor: Any, text: str) -> list[int]:
        """Resolve candidate text to token IDs."""
        return processor.tokenizer.encode(text, add_special_tokens=False)

    def collate(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a batch of training examples.

        Image-indexed tensors (``input_image_embeds``,
        ``image_attention_mask``, ``image_sizes``) already carry the
        correct leading dimension and must NOT be unsqueezed again.
        Only sequence tensors that were deliberately squeezed in
        ``build_prefix()`` need the batch dim restored.
        """
        if len(batch) == 1:
            image_indexed = self.image_indexed_keys()
            result = {}
            for k, v in batch[0].items():
                if torch.is_tensor(v) and k not in image_indexed:
                    result[k] = v.unsqueeze(0)
                else:
                    result[k] = v
            return result
        raise NotImplementedError("Phi-4-MM batching > 1 not yet supported")

    def append_candidate(
        self, prefix: dict[str, Any], candidate_token_ids: list[int],
    ) -> dict[str, torch.Tensor]:
        """Append candidate tokens to the prefix for scoring."""
        input_ids = prefix["input_ids"]
        candidate_ids = torch.tensor(
            candidate_token_ids, dtype=input_ids.dtype, device=input_ids.device,
        )
        # Match batch dimensions: input_ids is [batch, seq], candidate needs [batch, n]
        if input_ids.dim() == 2:
            candidate_ids = candidate_ids.unsqueeze(0)
        new_ids = torch.cat([input_ids, candidate_ids], dim=-1)
        result = dict(prefix)
        result["input_ids"] = new_ids

        # Extend attention_mask if present
        if "attention_mask" in result:
            mask = result["attention_mask"]
            if mask.dim() == 2:
                new_mask = torch.ones(
                    mask.shape[0], len(candidate_token_ids),
                    dtype=mask.dtype, device=mask.device,
                )
            else:
                new_mask = torch.ones(
                    len(candidate_token_ids),
                    dtype=mask.dtype, device=mask.device,
                )
            result["attention_mask"] = torch.cat([mask, new_mask], dim=-1)

        return result

    # ------------------------------------------------------------------ #
    # LoRA target resolution
    # ------------------------------------------------------------------ #

    def resolve_lora_targets(self, model: torch.nn.Module) -> list[str]:
        """Resolve LoRA target modules.

        Phi ships with bundled vision/speech LoRA adapters that wrap
        ``qkv_proj`` and ``o_proj`` in PEFT ``LoraLayer`` instances.
        We target these LoraLayer wrappers directly (NOT ``.base_layer``).
        PEFT's ``update_layer()`` multi-adapter path adds our adapter
        alongside the existing vision/speech adapters.
        """
        scope_regex = self._profile.lora_scope_regex
        targets = []
        for name, mod in model.named_modules():
            # Skip sub-modules inside existing LoRA adapters
            if ".lora_A." in name or ".lora_B." in name or ".base_layer" in name:
                continue
            # Accept nn.Linear or PEFT LoraLayer (already-wrapped modules)
            is_linear = isinstance(mod, torch.nn.Linear)
            is_lora_layer = False
            try:
                from peft.tuners.lora import LoraLayer
                is_lora_layer = isinstance(mod, LoraLayer)
            except ImportError:
                pass
            if (is_linear or is_lora_layer) and re.match(scope_regex, name):
                targets.append(name)
        targets.sort()

        # Post-discovery assertion: no vision/projector targets
        vision_targets = [
            t for t in targets
            if "visual" in t.lower() or "vision" in t.lower()
            or "projector" in t.lower() or "connector" in t.lower()
            or "speech" in t.lower()
        ]
        if vision_targets:
            raise RuntimeError(
                f"Phi LoRA scope matched {len(vision_targets)} "
                f"vision/speech/projector targets: {vision_targets[:5]}"
            )

        return targets

    def get_inner_peft_model(self, model: torch.nn.Module) -> torch.nn.Module | None:
        """Return the inner model with injected PEFT layers.

        Phi-4-MM internally calls ``get_peft_model(self.model, ...)`` during
        ``__init__`` to attach vision and speech LoRA adapters, then stores
        the ``peft_config`` directly on ``self.model`` (a ``Phi4MMModel``)
        without keeping the ``PeftModel`` wrapper.

        The returned model has ``peft_config`` with existing adapters
        (``vision``, ``speech``) and LoraLayer wrappers on attention modules.
        Use ``LoraModel.inject_adapter()`` to add our language adapter.
        """
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "peft_config"):
            peft_config = getattr(inner, "peft_config", {})
            if peft_config:  # has existing adapters (vision, speech)
                return inner
        return None

    # ------------------------------------------------------------------ #
    # Structural metadata
    # ------------------------------------------------------------------ #

    def language_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        """Return the language model's transformer layers."""
        layers = []
        for name, mod in model.named_modules():
            if re.match(r"^model\.layers\.\d+$", name):
                layers.append((name, mod))
        layers.sort(key=lambda x: int(x[0].split(".")[-1]))
        return [mod for _, mod in layers]

    def language_hidden_size(self, model: torch.nn.Module) -> int:
        return self._profile.language_hidden_size

    def language_intermediate_size(self, model: torch.nn.Module) -> int:
        return self._profile.intermediate_size

    def manu_neuron_specs(self, model: torch.nn.Module) -> list[NeuronSpec]:
        """MANU specs for Phi language MLP (fused gate_up_proj + down_proj)."""
        layers = self.language_layers(model)
        _hidden_size = self.language_hidden_size(model)
        intermediate_size = self.language_intermediate_size(model)

        specs: list[NeuronSpec] = []
        for i, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            if hasattr(mlp, "gate_up_proj") and hasattr(mlp, "down_proj"):
                specs.append(NeuronSpec(
                    layer_name=f"model.layers.{i}",
                    neuron_count=intermediate_size,
                    input_projection_names=("gate_up_proj",),
                    output_projection_name="down_proj",
                    input_axis=1,   # down_proj: intermediate -> hidden
                    output_axis=0,  # down_proj: intermediate -> hidden
                    is_fused_up=True,
                    fused_up_input_axis=0,   # gate_up_proj: hidden -> intermediate
                    fused_up_output_axis=1,  # gate_up_proj: hidden -> intermediate
                ))
        return specs

    def independent_forward_kwargs(
        self,
        prefix: dict[str, Any],
        candidate_token_ids: list[int],
    ) -> dict[str, torch.Tensor]:
        """Build forward kwargs independently from ``append_candidate``.

        Explicitly constructs the full forward dict with Phi-specific
        fields (``input_image_embeds``, ``image_attention_mask``,
        ``image_sizes``) passed through verbatim.  Used for P0-5 scorer
        equivalence verification.
        """
        input_ids = prefix["input_ids"]
        device = input_ids.device
        dtype = input_ids.dtype

        cand_ids = torch.tensor(
            [candidate_token_ids], dtype=dtype, device=device,
        )
        full_input_ids = torch.cat([input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids, cand_ids], dim=1)

        forward_kwargs: dict[str, Any] = {"input_ids": full_input_ids}

        # Attention mask
        if "attention_mask" in prefix:
            mask = prefix["attention_mask"]
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            new_mask = torch.ones(
                mask.shape[0], len(candidate_token_ids),
                dtype=mask.dtype, device=device,
            )
            forward_kwargs["attention_mask"] = torch.cat([mask, new_mask], dim=1)
        else:
            forward_kwargs["attention_mask"] = torch.ones_like(full_input_ids)

        # Phi-specific multimodal fields — pass through verbatim
        for key in ("input_image_embeds", "image_attention_mask", "image_sizes"):
            if key in prefix:
                forward_kwargs[key] = prefix[key]

        return forward_kwargs

    def to_eval_backend(self, **kwargs) -> Any:
        """Convert to a generic :class:`AdapterEvalBackend`."""
        from ..adapter_eval_backend import AdapterEvalBackend

        return AdapterEvalBackend(adapter=self, **kwargs)


register_model_key("phi4_mm", "phi4mm")
