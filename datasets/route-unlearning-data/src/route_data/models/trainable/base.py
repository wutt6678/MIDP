"""Trainable VLM adapter abstraction for multi-model unlearning.

This module defines the interface that each model family must implement
to participate in the MIDP unlearning pipeline. The adapter isolates
model-specific loading, rendering, batching, LoRA scope, and structural
metadata behind a uniform interface.

The generic pipeline (datasets, objectives, evaluation) selects a model
profile and adapter, not branch on model names throughout the training code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _remap_adapter_key(ckpt_key: str, adapter_name: str) -> str:
    """Remap a single-adapter checkpoint key to a named-adapter key.

    Checkpoint keys from ``PeftModel.save_pretrained()`` use the format::

        base_model.model.<layer>.lora_A.weight

    But the live model with a named adapter expects::

        base_model.model.<layer>.lora_A.<adapter_name>.weight

    This function inserts the adapter name before the final
    ``.weight`` (or ``.bias``) component.
    """
    import re
    # Match .lora_A.<param> or .lora_B.<param> at the end
    m = re.match(r"^(.*\.lora_[AB]\.)(\w+)$", ckpt_key)
    if m:
        prefix, param = m.group(1), m.group(2)
        # If the key already has the adapter name, return as-is
        if adapter_name in prefix:
            return ckpt_key
        return f"{prefix}{adapter_name}.{param}"
    return ckpt_key


def _load_adapter_manual(
    base_model: torch.nn.Module,
    checkpoint_dir: Path,
    adapter_name: str,
) -> torch.nn.Module:
    """Manually create a PEFT adapter and copy checkpoint tensors.

    This avoids the peak GPU memory of ``PeftModel.from_pretrained()``
    by reading checkpoint tensors on CPU and copying them one-by-one.
    The base model stays on its original device — only the checkpoint
    loading path is changed.

    This does NOT guarantee recovery from every GPU OOM (the base model
    itself must already fit).  It only reduces the *checkpoint loading*
    peak.
    """
    import json as _json

    from peft import LoraConfig as _LC
    from peft import get_peft_model
    from safetensors import safe_open as _sf

    cfg_path = checkpoint_dir / "adapter_config.json"
    with open(cfg_path) as _f:
        _cfg = _json.load(_f)
    lora_cfg = _LC(
        r=_cfg.get("r", 8),
        lora_alpha=_cfg.get("lora_alpha", 16),
        lora_dropout=_cfg.get("lora_dropout", 0.0),
        target_modules=_cfg.get("target_modules", []),
        task_type=_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(base_model, lora_cfg, adapter_name=adapter_name)

    # Read checkpoint on CPU and copy tensors one-by-one.
    ckpt_path = checkpoint_dir / "adapter_model.safetensors"
    if ckpt_path.is_file():
        ckpt_data = {}
        with _sf(str(ckpt_path), framework="pt", device="cpu") as _f:
            for _k in list(_f.keys()):
                ckpt_data[_k] = _f.get_tensor(_k)
        live_params = dict(model.named_parameters())
        for _k, _t in ckpt_data.items():
            if _k in live_params:
                live_params[_k].data.copy_(_t)
            else:
                _remapped = _remap_adapter_key(_k, adapter_name)
                if _remapped in live_params:
                    live_params[_remapped].data.copy_(_t)
        del ckpt_data

    return model


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

    # Structural metadata (P0-5: runtime validation)
    language_layer_path: str = ""
    language_hidden_size: int = 0
    intermediate_size: int = 0
    num_language_layers: int = 0
    lora_expected_target_modules: int = 0  # P0-SHARED-01: exact expected count

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
    max_transformers_version_exclusive: str | None = None
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

    For fused up-projections (e.g. Phi's ``gate_up_proj``), set
    ``is_fused_up=True`` and provide ``fused_up_input_axis`` /
    ``fused_up_output_axis``.  The standard ``input_axis`` /
    ``output_axis`` fields still describe the *down* projection.
    """

    layer_name: str
    neuron_count: int
    input_projection_names: tuple[str, ...]
    output_projection_name: str
    input_axis: int
    output_axis: int
    # Optional fused up-projection support (e.g. Phi gate_up_proj)
    is_fused_up: bool = False
    fused_up_input_axis: int = 0
    fused_up_output_axis: int = 0


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

    def independent_forward_kwargs(
        self,
        prefix: dict[str, Any],
        candidate_token_ids: list[int],
    ) -> dict[str, torch.Tensor]:
        """Build forward kwargs independently from ``append_candidate``.

        Used for P0-5 scorer equivalence verification: the shared scorer
        (via ``append_candidate``) is compared against this independent
        construction to ensure both produce identical results.

        Default implementation delegates to ``append_candidate``. Override
        for models that need explicit field pass-through verification
        (e.g. Phi's ``input_image_embeds``, ``image_attention_mask``).
        """
        return self.append_candidate(prefix, candidate_token_ids)

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

    def get_inner_peft_model(self, model: torch.nn.Module) -> torch.nn.Module | None:
        """Return the inner PEFT model if the model already has LoRA adapters.

        Some model families (e.g. Phi-4-MM) ship with bundled LoRA adapters
        for non-language modalities.  In that case, ``get_peft_model()``
        cannot be called again; instead, ``add_adapter()`` must be called on
        the existing PeftModel to add our language adapter alongside the
        bundled ones.

        Returns ``None`` for model families that do not have pre-existing
        PEFT wrapping (the default).
        """
        return None

    # ------------------------------------------------------------------ #
    # Adapter lifecycle hooks (P0-05)
    # ------------------------------------------------------------------ #

    def attach_unlearning_adapter(
        self,
        model: torch.nn.Module,
        *,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: list[str],
        adapter_name: str = "unlearning",
    ) -> torch.nn.Module:
        """Attach the unlearning LoRA adapter to the model.

        Default implementation uses ``peft.get_peft_model()`` for standard
        models.  Override for models with bundled adapters (e.g. Phi-4-MM).

        Returns the model with the unlearning adapter attached.
        """
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, lora_cfg, adapter_name=adapter_name)

    def save_unlearning_adapter(
        self,
        model: torch.nn.Module,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Save only the unlearning adapter checkpoint.

        Saves adapter_config.json and adapter_model.safetensors directly
        in output_dir (not in a named subdirectory).

        Returns metadata dict with checkpoint file list, SHA-256, etc.
        """
        import hashlib

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use PEFT's save_pretrained but handle the subdirectory structure
        model.save_pretrained(str(output_dir))

        # PEFT may save into a subdirectory named after the adapter.
        # Move files to the top level if needed.  Stale files from a previous
        # run MUST be replaced: the former `if not dest.exists()` guard
        # silently discarded the freshly saved checkpoint (the rmtree below
        # then deleted it), leaving the previous run's weights on disk to be
        # loaded and evaluated by later phases.
        for sub in output_dir.iterdir():
            if sub.is_dir() and (sub / "adapter_config.json").exists():
                for f in sub.iterdir():
                    if not f.is_file():
                        continue
                    dest = output_dir / f.name
                    if dest.exists():
                        dest.unlink()
                    f.rename(dest)
                import shutil
                shutil.rmtree(sub, ignore_errors=True)
                break

        # Fail closed: a checkpoint weight file must exist at the top level
        # after flattening, otherwise downstream loads would silently reuse
        # stale weights.
        if not any(p.is_file() and p.suffix in (".safetensors", ".bin")
                   for p in output_dir.iterdir()):
            raise RuntimeError(
                f"save_unlearning_adapter: no checkpoint weight file found "
                f"in {output_dir} after save/flatten")

        # Compute SHA-256 of checkpoint files
        metadata: dict[str, Any] = {"files": []}
        total_sha = hashlib.sha256()
        for fname in sorted(output_dir.iterdir()):
            if fname.is_file() and fname.suffix in (".safetensors", ".bin", ".json"):
                fhash = hashlib.sha256()
                with open(fname, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        fhash.update(chunk)
                file_sha = fhash.hexdigest()
                total_sha.update(file_sha.encode())
                metadata["files"].append({
                    "name": fname.name,
                    "sha256": file_sha,
                    "size": fname.stat().st_size,
                })
        metadata["checkpoint_sha256"] = total_sha.hexdigest()
        return metadata

    def load_unlearning_adapter(
        self,
        base_model: torch.nn.Module,
        checkpoint_dir: Path,
        *,
        adapter_name: str = "unlearning",
    ) -> torch.nn.Module:
        """Load a saved unlearning adapter onto a fresh base model.

        Uses ``PeftModel.from_pretrained()`` then verifies that all
        checkpoint tensors were actually restored.  If the checkpoint
        was saved as a single-adapter model (keys lack the adapter
        name suffix), we manually remap and copy the weights.
        """
        from peft import PeftModel

        ckpt_src = Path(checkpoint_dir)
        try:
            # Try loading directly first
            model = PeftModel.from_pretrained(
                base_model, str(ckpt_src), adapter_name=adapter_name,
            )
        except torch.cuda.OutOfMemoryError:
            model = _load_adapter_manual(
                base_model, ckpt_src, adapter_name,
            )
        except RuntimeError as exc:
            # Only retry on CUDA OOM — other RuntimeErrors (shape mismatch,
            # corrupted checkpoint, device mismatch) must propagate.
            if "out of memory" not in str(exc).lower():
                raise
            logger.info(
                "Retrying adapter load with manual tensor copy "
                "to reduce GPU memory peak"
            )
            torch.cuda.empty_cache()
            model = _load_adapter_manual(
                base_model, ckpt_src, adapter_name,
            )

        # Bidirectional checkpoint verification (P0-SHARED-03).
        # The checkpoint may use keys without the adapter name suffix
        # (e.g. ``lora_A.weight``) while the live model expects
        # ``lora_A.<adapter_name>.weight``.
        from safetensors import safe_open

        ckpt_path = Path(checkpoint_dir) / "adapter_model.safetensors"
        if ckpt_path.is_file():
            ckpt_data = {}
            with safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
                for k in list(f.keys()):
                    ckpt_data[k] = f.get_tensor(k)

            live_params = dict(model.named_parameters())

            # Enumerate live unlearning adapter keys.
            live_unlearning_keys = {
                k for k in live_params if adapter_name in k
            }

            copied = 0
            copied_live_keys: set[str] = set()
            missing_checkpoint_keys: list[str] = []
            for ckpt_key, ckpt_tensor in ckpt_data.items():
                if ckpt_key in live_params:
                    live_params[ckpt_key].data.copy_(ckpt_tensor)
                    copied += 1
                    copied_live_keys.add(ckpt_key)
                else:
                    remapped = _remap_adapter_key(ckpt_key, adapter_name)
                    if remapped in live_params:
                        live_params[remapped].data.copy_(ckpt_tensor)
                        copied += 1
                        copied_live_keys.add(remapped)
                    else:
                        missing_checkpoint_keys.append(ckpt_key)

            unexpected_live_keys = sorted(
                live_unlearning_keys - copied_live_keys
            )

            # Bidirectional validation:
            #   checkpoint tensors == live unlearning tensors == copied
            ckpt_count = len(ckpt_data)
            live_count = len(live_unlearning_keys)
            if (
                copied != ckpt_count
                or copied != live_count
                or missing_checkpoint_keys
                or unexpected_live_keys
            ):
                raise RuntimeError(
                    f"Bidirectional adapter load validation FAILED:\n"
                    f"  checkpoint_count={ckpt_count}\n"
                    f"  live_count={live_count}\n"
                    f"  copied_count={copied}\n"
                    f"  missing_checkpoint_keys={len(missing_checkpoint_keys)}"
                    f" (first: {missing_checkpoint_keys[:3]})\n"
                    f"  unexpected_live_keys={len(unexpected_live_keys)}"
                    f" (first: {unexpected_live_keys[:3]})"
                )

            logger.info(
                f"Bidirectional adapter load verified: "
                f"{copied}/{ckpt_count} checkpoint, "
                f"{copied}/{live_count} live unlearning tensors restored "
                f"from {ckpt_path.name}"
            )

        return model

    def snapshot_protected_parameters(
        self,
        model: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Snapshot protected (non-trainable) parameters before training.

        Returns a dict mapping parameter names to detached CPU copies.
        Default: all parameters with ``requires_grad=False``.
        """
        snapshot: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                snapshot[name] = param.detach().cpu().clone()
        return snapshot

    def verify_protected_parameters(
        self,
        snapshot: dict[str, torch.Tensor],
        model: torch.nn.Module,
    ) -> dict[str, Any]:
        """Verify protected parameters are bitwise unchanged after training.

        Returns a report dict with ``pass`` (bool), ``n_changed`` (int),
        and ``changed_names`` (list[str]).
        """
        changed: list[str] = []
        for name, old_val in snapshot.items():
            param = dict(model.named_parameters()).get(name)
            if param is None:
                changed.append(name)
                continue
            new_val = param.detach().cpu()
            if not torch.equal(old_val, new_val):
                changed.append(name)
        return {
            "pass": len(changed) == 0,
            "n_total": len(snapshot),
            "n_changed": len(changed),
            "changed_names": changed[:20],  # cap for readability
        }

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

    def language_intermediate_size(
        self,
        model: torch.nn.Module,
    ) -> int:
        """Return the intermediate (MLP) dimension of the language transformer.

        Default raises NotImplementedError; override for models that
        support structural validation or MANU.
        """
        raise NotImplementedError(
            f"{self.profile.key} does not implement language_intermediate_size"
        )

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
