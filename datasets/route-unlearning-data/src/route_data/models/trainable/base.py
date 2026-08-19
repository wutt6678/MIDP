"""Trainable VLM adapter abstraction for multi-model unlearning.

This module defines the interface that each model family must implement
to participate in the MIDP unlearning pipeline. The adapter isolates
model-specific loading, rendering, batching, LoRA scope, and structural
metadata behind a uniform interface.

The generic pipeline (datasets, objectives, evaluation) selects a model
profile and adapter, not branch on model names throughout the training code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ModelFamilyProfile:
    """Frozen reproducibility-critical configuration for a model family.

    All final research profiles must pin an exact immutable Hugging Face
    revision for both model and processor. Reject ``main``, ``None``, or
    a mutable branch in final experiment mode.
    """

    key: str
    model_id: str
    revision: str
    processor_id: str
    processor_revision: str
    adapter_name: str
    trust_remote_code: bool
    dtype: str
    attn_implementation: str | None

    # Candidate protocol
    candidate_positive: str
    candidate_negative: str

    # LoRA configuration
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_scope: str  # e.g. "language_attention_only"
    lora_target_leaf_names: tuple[str, ...]
    lora_scope_regex: str  # regex to restrict targets to language tower

    # R2MU configuration
    r2mu_candidate_layers: tuple[int, ...]
    r2mu_n_select_layers: int

    # Method support flags
    supports_prompting: bool = True
    supports_candidate_margin: bool = True
    supports_ga: bool = True
    supports_gd: bool = True
    supports_kl: bool = True
    supports_npo: bool = True
    supports_mmunlearner: bool = True
    supports_manu: bool = True
    supports_r2mu: bool = True

    # Environment
    min_transformers_version: str | None = None
    tested_transformers_version: str = ""
    requires_hf_auth: bool = False

    def validate_revision_immutable(self) -> None:
        """Reject mutable branch references in research mode."""
        for name, val in [
            ("revision", self.revision),
            ("processor_revision", self.processor_revision),
        ]:
            if not val:
                raise ValueError(f"{name} must be a non-empty immutable SHA")
            if val in ("main", "master", "HEAD", "dev", "develop"):
                raise ValueError(
                    f"{name}={val!r} is a mutable branch; "
                    f"pin an exact HuggingFace commit SHA"
                )


@dataclass(frozen=True)
class NeuronSpec:
    """Structural specification for MANU neuron selection.

    Describes a language-backbone MLP layer with explicit axis information
    so that neuron pruning can be performed without model-specific hacks.
    """

    layer_name: str
    neuron_count: int
    input_projection_names: tuple[str, ...]
    output_projection_name: str
    input_axis: int
    output_axis: int


class TrainableVLMAdapter(ABC):
    """Abstract interface for trainable multimodal model adapters.

    Each model family (Qwen, GLM, InternVL, Phi, Gemma) implements this
    interface to participate in the MIDP unlearning pipeline. The generic
    pipeline delegates model-specific operations to the adapter.
    """

    @property
    @abstractmethod
    def profile(self) -> ModelFamilyProfile:
        """Return the frozen model profile."""

    # ------------------------------------------------------------------ #
    # Model and processor loading
    # ------------------------------------------------------------------ #

    @abstractmethod
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
        """Load the base model and processor.

        Returns
        -------
        model : torch.nn.Module
            The base model in eval mode (or training mode if *training*).
        processor : Any
            The processor for tokenization and image processing.
        """

    # ------------------------------------------------------------------ #
    # Prefix and supervised example construction
    # ------------------------------------------------------------------ #

    @abstractmethod
    def build_prefix(
        self,
        processor: Any,
        *,
        image: Any,
        prompt: str,
    ) -> dict[str, Any]:
        """Build the multimodal assistant prefix for candidate scoring.

        The prefix must be exactly the first ``_prefix_len`` tokens of the
        full supervised sequence. This is a P0 gate for scoring correctness.

        Returns
        -------
        prefix : dict
            Contains at minimum ``input_ids``, ``attention_mask``, and
            model-specific multimodal tensors. Must include ``_prefix_len``.
        """

    @abstractmethod
    def build_supervised_example(
        self,
        processor: Any,
        *,
        image: Any,
        prompt: str,
        answer_text: str,
    ) -> dict[str, Any]:
        """Build a full supervised training example.

        The full sequence is prefix + answer tokens. Labels must have
        ``-100`` for prefix and padding positions.

        Returns
        -------
        example : dict
            Contains ``input_ids``, ``attention_mask``, ``labels``,
            ``_prefix_len``, ``_correct_answer_token_ids``,
            ``_answer_label``, ``_yes_token_ids``, ``_no_token_ids``,
            and model-specific multimodal tensors.
        """

    # ------------------------------------------------------------------ #
    # Candidate token resolution
    # ------------------------------------------------------------------ #

    @abstractmethod
    def candidate_token_ids(
        self,
        processor: Any,
        text: str,
    ) -> list[int]:
        """Resolve a candidate string to token IDs.

        The default implementation uses the processor's tokenizer to
        encode the text without special tokens.
        """

    # ------------------------------------------------------------------ #
    # Collation
    # ------------------------------------------------------------------ #

    @abstractmethod
    def collate(
        self,
        batch: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """Collate a list of examples into a padded batch.

        Text-aligned tensors are right-padded. Visual tensors follow
        model-specific batching semantics. The pad token ID must be
        obtained from the model/processor/profile, not hard-coded.
        """

    # ------------------------------------------------------------------ #
    # Candidate append (for scoring)
    # ------------------------------------------------------------------ #

    @abstractmethod
    def append_candidate(
        self,
        prefix: dict[str, Any],
        candidate_token_ids: list[int],
    ) -> dict[str, torch.Tensor]:
        """Append candidate token IDs to a prefix for forward pass.

        Returns a dict suitable for ``model(**prepared)``. Text tensors
        are extended; visual tensors are passed through unchanged.
        """

    # ------------------------------------------------------------------ #
    # LoRA target resolution
    # ------------------------------------------------------------------ #

    @abstractmethod
    def resolve_lora_targets(
        self,
        model: torch.nn.Module,
    ) -> list[str]:
        """Resolve LoRA target module names for the language tower only.

        Must return fully-qualified module names that exist in the model.
        Must exclude vision encoder, projector, and multimodal modules.

        Returns
        -------
        targets : list[str]
            Module names for LoRA attachment.
        """

    # ------------------------------------------------------------------ #
    # Language layer access (for R2MU, MANU)
    # ------------------------------------------------------------------ #

    @abstractmethod
    def language_layers(
        self,
        model: torch.nn.Module,
    ) -> list[torch.nn.Module]:
        """Return the language transformer layers (excluding vision)."""

    @abstractmethod
    def language_hidden_size(
        self,
        model: torch.nn.Module,
    ) -> int:
        """Return the hidden dimension of the language transformer."""

    def manu_neuron_specs(
        self,
        model: torch.nn.Module,
    ) -> list[NeuronSpec]:
        """Return MANU neuron specifications for language-backbone MLPs.

        Default raises NotImplementedError; override for models that
        support MANU.
        """
        raise NotImplementedError(
            f"{self.profile.key} does not support MANU"
        )

    # ------------------------------------------------------------------ #
    # Multimodal key management
    # ------------------------------------------------------------------ #

    def required_multimodal_keys(self) -> frozenset[str]:
        """Return the set of required multimodal tensor keys.

        Default: ``{"pixel_values"}``. Override for models that use
        different tensor names (e.g., Phi uses ``input_image_embeds``).
        """
        return frozenset({"pixel_values"})

    def image_indexed_keys(self) -> frozenset[str]:
        """Return keys whose dim 0 is num_images/tiles, not batch.

        These keys must NOT be squeezed when processing single examples.
        """
        return frozenset({"image_grid_thw", "image_sizes", "pixel_values"})

    def sanitize_model_inputs(
        self,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Remove or transform inputs for model-specific forward pass.

        Default: identity. Override for model-specific sanitization
        (e.g., GLM drops ``token_type_ids``).
        """
        return inputs

    # ------------------------------------------------------------------ #
    # Eval backend conversion
    # ------------------------------------------------------------------ #

    @abstractmethod
    def to_eval_backend(
        self,
        *,
        model: torch.nn.Module,
        processor: Any,
        model_config: Any,
        adapter_metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Convert to a VisionLanguageModel for post-unlearning evaluation.

        Returns a backend compatible with the existing eval pipeline.
        """

    # ------------------------------------------------------------------ #
    # Pad token resolution
    # ------------------------------------------------------------------ #

    def pad_token_id(self, processor: Any) -> int:
        """Resolve the pad token ID from the processor.

        Default implementation looks up the tokenizer's pad_token_id.
        """
        tokenizer = getattr(processor, "tokenizer", processor)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            raise RuntimeError(
                f"Cannot resolve pad_token_id for {self.profile.key}"
            )
        return pad_id

    # ------------------------------------------------------------------ #
    # Prefix/full alignment verification (P0 gate)
    # ------------------------------------------------------------------ #

    def verify_prefix_alignment(
        self,
        processor: Any,
        *,
        image: Any,
        prompt: str,
        answer_text: str = "Yes",
    ) -> bool:
        """Verify that prefix is exactly the first prefix_len tokens of full.

        This is a P0 gate: ``supervised["input_ids"][:prefix_len]`` must
        equal ``prefix["input_ids"]``.
        """
        prefix = self.build_prefix(processor, image=image, prompt=prompt)
        full = self.build_supervised_example(
            processor, image=image, prompt=prompt, answer_text=answer_text,
        )
        prefix_len = prefix["_prefix_len"]
        prefix_ids = prefix["input_ids"]
        full_ids = full["input_ids"][:prefix_len]

        # Handle batched vs unbatched
        if prefix_ids.dim() == 2:
            prefix_ids = prefix_ids[0]
        if full_ids.dim() == 2:
            full_ids = full_ids[0]

        return torch.equal(prefix_ids, full_ids)
