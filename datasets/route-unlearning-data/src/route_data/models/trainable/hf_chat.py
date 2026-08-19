"""Shared HuggingFace chat-template adapter base class.

Provides default implementations of :class:`TrainableVLMAdapter` methods
for models that follow the standard HuggingFace ``processor.apply_chat_template``
convention.  Qwen, GLM, InternVL, and Gemma all use this pattern; only
Phi-4-MM requires a genuinely different rendering path.

Subclasses override only the model-specific hooks:

* ``_model_class()`` — the ``transformers`` auto-model class to load
* ``_chat_template_kwargs()`` — extra kwargs for ``apply_chat_template``
* ``_resolve_lora_targets()`` — language-tower attention module names
* ``_language_layer_path()`` — attribute path to language transformer layers
"""

from __future__ import annotations

import logging
import re
from typing import Any

import torch

from .base import TrainableVLMAdapter

logger = logging.getLogger(__name__)


class HuggingFaceChatAdapter(TrainableVLMAdapter):
    """Base adapter for standard HF chat-template multimodal models.

    Provides concrete implementations of:

    - :meth:`build_prefix`
    - :meth:`build_supervised_example`
    - :meth:`candidate_token_ids`
    - :meth:`collate`
    - :meth:`append_candidate`
    - :meth:`load_model_processor`
    - :meth:`pad_token_id`

    Subclasses must implement the abstract methods from
    :class:`TrainableVLMAdapter` that are genuinely architecture-specific:

    - :meth:`resolve_lora_targets`
    - :meth:`language_layers`
    - :meth:`language_hidden_size`
    - :meth:`to_eval_backend`

    And may override these hooks:

    - :meth:`chat_template_kwargs` (default: ``{}``)
    - :meth:`image_indexed_keys` (default: Qwen-style set)
    - :meth:`required_multimodal_keys` (default: ``{"pixel_values"}``)
    - :meth:`sanitize_model_inputs` (default: identity)
    """

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #

    def chat_template_kwargs(self) -> dict[str, Any]:
        """Extra keyword arguments for ``processor.apply_chat_template``.

        Override per model family.  For example Qwen uses
        ``enable_thinking=False``; most other models need nothing extra.
        """
        return {}

    # ------------------------------------------------------------------ #
    # Model / processor loading (generic HF auto-class route)
    # ------------------------------------------------------------------ #

    def load_model_processor(
        self,
        *,
        model_id: str,
        revision: str,
        processor_revision: str,
        dtype: str,
        device: str,
        training: bool = False,
    ) -> tuple[Any, Any]:
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor

        torch_dtype = getattr(torch, dtype)
        snapshot_download(model_id, revision=revision)

        model_cls = self._model_auto_class()
        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": self.profile.trust_remote_code,
            "revision": revision,
        }
        if device:
            model_kwargs["device_map"] = device
        if self.profile.attn_implementation:
            model_kwargs["attn_implementation"] = self.profile.attn_implementation

        model = model_cls.from_pretrained(model_id, **model_kwargs)
        processor = AutoProcessor.from_pretrained(
            self.profile.processor_id,
            revision=processor_revision,
            trust_remote_code=self.profile.trust_remote_code,
        )

        if not training:
            model.eval()
        else:
            model.gradient_checkpointing_enable()

        logger.info(
            f"Loaded {self.profile.key}: {model_id} rev={revision[:12]}"
        )
        return model, processor

    def _model_auto_class(self):
        """Return the ``transformers`` auto-model class for this family.

        Default: ``AutoModelForImageTextToText``.  Override for families
        that require a specific class (e.g. Phi uses ``AutoModelForCausalLM``).
        """
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText

    # ------------------------------------------------------------------ #
    # Prefix construction
    # ------------------------------------------------------------------ #

    def build_prefix(
        self,
        processor: Any,
        *,
        image: Any,
        prompt: str,
    ) -> dict[str, Any]:
        user_messages = self._build_user_messages(image=image, prompt=prompt)
        extra_kw = self.chat_template_kwargs()

        try:
            prefix = processor.apply_chat_template(
                user_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **extra_kw,
            )
        except TypeError:
            # Fallback for processors that don't accept extra kwargs
            prefix = processor.apply_chat_template(
                user_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

        return self._process_prefix_output(prefix)

    def _build_user_messages(
        self, *, image: Any, prompt: str,
    ) -> list[dict[str, Any]]:
        """Build the user-content message list for the chat template.

        Default includes an image content part when *image* is not ``None``.
        Override for text-only or multi-image layouts.
        """
        if image is not None:
            return [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }]

    def _process_prefix_output(self, prefix: dict[str, Any]) -> dict[str, Any]:
        """Post-process the raw processor output into adapter prefix format.

        Default: squeeze batch dim from text-aligned tensors, preserve
        image-indexed tensors as-is, and record ``_prefix_len``.
        """
        image_indexed = self.image_indexed_keys()
        result: dict[str, Any] = {}
        for key, value in prefix.items():
            if torch.is_tensor(value) and key not in image_indexed:
                result[key] = value.squeeze(0)
            elif torch.is_tensor(value):
                result[key] = value

        result["_prefix_len"] = result["input_ids"].shape[0]
        return result

    # ------------------------------------------------------------------ #
    # Full supervised example
    # ------------------------------------------------------------------ #

    def build_supervised_example(
        self,
        processor: Any,
        *,
        image: Any,
        prompt: str,
        answer_text: str,
    ) -> dict[str, Any]:
        # Build prefix first (for _prefix_len and metadata)
        prefix = self.build_prefix(processor, image=image, prompt=prompt)
        prefix_len = prefix["_prefix_len"]

        # Build full conversation text
        messages = self._build_full_conversation(
            image=image, prompt=prompt, answer_text=answer_text,
        )
        extra_kw = self.chat_template_kwargs()

        try:
            full_prompt = processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                **extra_kw,
            )
        except TypeError:
            full_prompt = processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
            )

        # Tokenize full conversation with processor (includes image)
        full_tokens = processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
            truncation=False,
            padding=False,
        )

        # Process tensors
        image_indexed = self.image_indexed_keys()
        result: dict[str, Any] = {}
        for key, value in full_tokens.items():
            if torch.is_tensor(value) and key not in image_indexed:
                result[key] = value.squeeze(0)
            elif torch.is_tensor(value):
                result[key] = value

        # Build labels: -100 for prefix and padding positions
        labels = result["input_ids"].clone()
        labels[:prefix_len] = -100
        result["labels"] = labels

        # Carry forward metadata from prefix
        result["_prefix_len"] = prefix_len
        result["_pad_token_id"] = self.pad_token_id(processor)
        result["_correct_answer_token_ids"] = self.candidate_token_ids(
            processor, answer_text,
        )
        result["_answer_label"] = (answer_text == self.profile.candidate_positive)
        result["_yes_token_ids"] = self.candidate_token_ids(
            processor, self.profile.candidate_positive,
        )
        result["_no_token_ids"] = self.candidate_token_ids(
            processor, self.profile.candidate_negative,
        )

        return result

    def _build_full_conversation(
        self, *, image: Any, prompt: str, answer_text: str,
    ) -> list[dict[str, Any]]:
        """Build the full user+assistant conversation for supervised training."""
        if image is not None:
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        else:
            user_content = [
                {"type": "text", "text": prompt},
            ]
        return [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text}],
            },
        ]

    # ------------------------------------------------------------------ #
    # Candidate token resolution
    # ------------------------------------------------------------------ #

    def candidate_token_ids(
        self,
        processor: Any,
        text: str,
    ) -> list[int]:
        tokenizer = getattr(processor, "tokenizer", processor)
        ids = tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt",
        )
        if ids.numel() > 0:
            return ids[0].tolist()
        # Fallback via vocabulary lookup
        tid = tokenizer.vocab.get(text)
        if tid is not None:
            return [tid]
        raise RuntimeError(
            f"Cannot resolve token IDs for text: {text!r}"
        )

    # ------------------------------------------------------------------ #
    # Collation
    # ------------------------------------------------------------------ #

    def collate(
        self,
        batch: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        pad_id = self._collate_pad_token_id(batch)
        batch_size = len(batch)

        # Validate required keys
        self._validate_batch_keys(batch)

        # Determine max sequence length
        max_len = max(item["input_ids"].shape[0] for item in batch)

        padded_input_ids: list[torch.Tensor] = []
        padded_attention_mask: list[torch.Tensor] = []
        padded_labels: list[torch.Tensor] = []

        # Track which sequence-indexed keys need padding
        seq_keys = self._sequence_indexed_keys_in_batch(batch)
        padded_seq: dict[str, list[torch.Tensor]] = {k: [] for k in seq_keys}

        for item in batch:
            seq_len = item["input_ids"].shape[0]
            pad_len = max_len - seq_len

            padded_input_ids.append(
                torch.cat([
                    item["input_ids"],
                    torch.full((pad_len,), pad_id, dtype=torch.long),
                ])
            )
            padded_attention_mask.append(
                torch.cat([
                    item["attention_mask"],
                    torch.zeros(pad_len, dtype=torch.long),
                ])
            )
            padded_labels.append(
                torch.cat([
                    item["labels"],
                    torch.full((pad_len,), -100, dtype=torch.long),
                ])
            )
            for key in seq_keys:
                if key in item:
                    padded_seq[key].append(
                        torch.cat([
                            item[key],
                            torch.zeros(pad_len, dtype=item[key].dtype),
                        ])
                    )

        result: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }
        for key in seq_keys:
            if padded_seq[key]:
                result[key] = torch.stack(padded_seq[key])

        # Visual tensors: concatenate along image/tile dimension
        self._collate_visual_tensors(batch, result, batch_size)

        # Shape assertions
        self._assert_collate_shapes(result, batch_size, max_len)

        # Preserve metadata
        for key in ("_prefix_len", "_correct_answer_token_ids", "_answer_label",
                     "_yes_token_ids", "_no_token_ids"):
            if key in batch[0]:
                result[key] = [item[key] for item in batch]

        return result

    def _collate_pad_token_id(self, batch: list[dict[str, Any]]) -> int:
        """Resolve pad token ID for collation.

        Requires ``_pad_token_id`` in each batch item (set by
        ``build_supervised_example``).  Validates consistency across
        the batch.  No generic fallback to 0.
        """
        if "_pad_token_id" not in batch[0]:
            raise RuntimeError(
                f"{self.profile.key}: batch items must contain "
                f"'_pad_token_id'. Ensure build_supervised_example "
                f"stores it."
            )
        pad_id = batch[0]["_pad_token_id"]
        if not isinstance(pad_id, int) or pad_id < 0:
            raise RuntimeError(
                f"{self.profile.key}: _pad_token_id must be a "
                f"non-negative integer, got {pad_id!r}"
            )
        # Validate consistency across batch
        for i, item in enumerate(batch):
            if item.get("_pad_token_id") != pad_id:
                raise RuntimeError(
                    f"{self.profile.key}: inconsistent _pad_token_id "
                    f"in batch: item 0 has {pad_id}, item {i} has "
                    f"{item.get('_pad_token_id')!r}"
                )
        return pad_id

    def _default_pad_token_id(self) -> int:
        """Deprecated — pad token must come from the batch."""
        raise RuntimeError(
            f"{self.profile.key}: _default_pad_token_id() should not "
            f"be called. Pad token must come from the batch items."
        )

    def _sequence_indexed_keys_in_batch(
        self, batch: list[dict[str, Any]],
    ) -> set[str]:
        """Return sequence-indexed tensor keys present in the batch.

        These are text-aligned (same length as input_ids) and must be
        padded alongside input_ids/attention_mask/labels.
        """
        # Default: mm_token_type_ids if present (Qwen-style)
        keys: set[str] = set()
        if "mm_token_type_ids" in batch[0]:
            keys.add("mm_token_type_ids")
        return keys

    def _sequence_indexed_keys_for_scoring(
        self, prefix: dict[str, Any],
    ) -> set[str]:
        """Return sequence-indexed tensor keys in a single prefix dict.

        Used by :meth:`append_candidate` to know which tensors need
        extending with candidate tokens.  Default: ``mm_token_type_ids``
        if present.  Override for models with different text-aligned
        tensors.
        """
        keys: set[str] = set()
        if "mm_token_type_ids" in prefix:
            keys.add("mm_token_type_ids")
        return keys

    def _collate_visual_tensors(
        self,
        batch: list[dict[str, Any]],
        result: dict[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        """Concatenate visual tensors from batch items into *result*.

        Uses :meth:`image_indexed_keys` to determine which keys are
        image-indexed and should be concatenated along dim 0.
        """
        for key in sorted(self.image_indexed_keys()):
            tensor_list = [
                item[key] for item in batch if key in item
            ]
            if len(tensor_list) == batch_size and all(
                torch.is_tensor(t) for t in tensor_list
            ):
                result[key] = torch.cat(tensor_list, dim=0)

    def _validate_batch_keys(self, batch: list[dict[str, Any]]) -> None:
        required = {"input_ids", "attention_mask", "labels"}
        for i, item in enumerate(batch):
            missing = required - set(item.keys())
            if missing:
                raise RuntimeError(
                    f"Sample {i} missing required keys: {sorted(missing)}. "
                    f"Available: {sorted(item.keys())}"
                )

    def _assert_collate_shapes(
        self,
        result: dict[str, torch.Tensor],
        batch_size: int,
        seq_len: int,
    ) -> None:
        for key in ("input_ids", "attention_mask", "labels"):
            assert result[key].shape == (batch_size, seq_len), (
                f"{key} shape mismatch: {result[key].shape}"
            )

    # ------------------------------------------------------------------ #
    # Append candidate (for scoring)
    # ------------------------------------------------------------------ #

    def append_candidate(
        self,
        prefix: dict[str, Any],
        candidate_token_ids: list[int],
    ) -> dict[str, torch.Tensor]:
        if not candidate_token_ids:
            raise ValueError("candidate_token_ids must be non-empty")

        prefix_input_ids = prefix["input_ids"]
        device = prefix_input_ids.device
        dtype = prefix_input_ids.dtype

        cand_ids = torch.tensor(
            [candidate_token_ids], dtype=dtype, device=device,
        )

        forward: dict[str, torch.Tensor] = {
            "input_ids": torch.cat([prefix_input_ids, cand_ids], dim=1),
            "attention_mask": torch.cat(
                [prefix["attention_mask"], torch.ones_like(cand_ids)], dim=1,
            ),
        }

        # Extend sequence-indexed text tensors
        for key in self._sequence_indexed_keys_for_scoring(prefix):
            if key in prefix:
                prefix_mm = prefix[key]
                cand_mm = torch.zeros_like(cand_ids, dtype=prefix_mm.dtype)
                forward[key] = torch.cat([prefix_mm, cand_mm], dim=1)

        # Pass through visual tensors unchanged
        for key in self._visual_tensor_keys(prefix):
            if key in prefix:
                forward[key] = prefix[key]

        # Apply model-specific sanitization
        forward = self.sanitize_model_inputs(forward)

        return forward

    def _visual_tensor_keys(self, prefix: dict[str, Any]) -> list[str]:
        """Return the visual tensor keys present in *prefix*.

        Uses :meth:`image_indexed_keys` to determine which keys are
        visual/image-indexed, plus any additional tensor keys that are
        not text-aligned.
        """
        img_keys = self.image_indexed_keys()
        keys: list[str] = []
        for k in sorted(img_keys):
            if k in prefix:
                keys.append(k)
        return keys

    # ------------------------------------------------------------------ #
    # LoRA target resolution (shared regex-based helper)
    # ------------------------------------------------------------------ #

    def _resolve_targets_with_regex(
        self,
        model: torch.nn.Module,
        *,
        leaf_names: tuple[str, ...],
        scope_regex: str,
    ) -> list[str]:
        """Resolve LoRA targets by matching leaf names within a scope regex.

        Parameters
        ----------
        model:
            The base model.
        leaf_names:
            Target leaf module names (e.g. ``("q_proj", "v_proj")``).
        scope_regex:
            Regex that must match the parent path to select a module.
            E.g. ``r"^model\\.language_model\\."`` restricts to the language tower.
        """
        if not scope_regex:
            raise ValueError(
                f"{self.profile.key}: lora_scope_regex must be set to "
                f"restrict LoRA targets to the language tower"
            )
        pattern = re.compile(scope_regex)
        targets: list[str] = []
        for name, _module in model.named_modules():
            leaf = name.rsplit(".", 1)[-1] if "." in name else name
            if leaf in leaf_names and pattern.match(name):
                targets.append(name)
        return sorted(targets)
