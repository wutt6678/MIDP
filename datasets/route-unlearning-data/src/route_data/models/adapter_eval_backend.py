"""Generic adapter-backed evaluation backend.

Wraps any :class:`TrainableVLMAdapter` + loaded model + processor into a
:class:`VisionLanguageModel` suitable for the frozen 500-probe baseline
evaluation pipeline.

Design goals (from multi-model integration plan §1.2):

- **Model-agnostic**: no branching on model names in the evaluator.
- **Shared scorer**: uses :func:`score_candidate_sequence_tensor` with the
  adapter for identical train/eval logic.
- **Adapter-driven prefix**: delegates to ``adapter.build_prefix()`` for
  model-specific prompt rendering and multimodal tensor construction.
- **Adapter-driven candidate tokenization**: delegates to
  ``adapter.candidate_token_ids()`` for model-specific token resolution.
- **Adapter-aware fingerprints**: includes profile key, adapter metadata,
  and processor/tokenizer class in the cache key.
- **Frozen 128-token name_only protocol**: respects the generation budget
  passed via ``max_new_tokens``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any

import torch

from .base import CandidateScore, VisionLanguageModel, VisionResponse
from .scoring import score_candidate_sequence_tensor


class AdapterEvalBackend(VisionLanguageModel):
    """Generic :class:`VisionLanguageModel` wrapping a TrainableVLMAdapter.

    Parameters
    ----------
    adapter:
        The :class:`TrainableVLMAdapter` instance (e.g. Qwen35Adapter,
        GLM46VAdapter).  Provides model-specific prefix construction,
        candidate tokenization, and collation.
    model:
        Pre-loaded model (typically a ``PeftModel`` wrapping the base,
        or the bare base model for pre-unlearning baselines).
    processor:
        Pre-loaded processor matching the base model.
    model_config:
        :class:`~route_data.config.ModelConfig` for generation parameters.
    adapter_metadata:
        Optional adapter provenance dict (checkpoint SHA, LoRA hyperparams).
    resolved_revision:
        The resolved base-model revision string for fingerprinting.
    """

    def __init__(
        self,
        adapter: Any,
        model: Any,
        processor: Any,
        model_config: Any,
        *,
        adapter_metadata: dict[str, Any] | None = None,
        resolved_revision: str | None = None,
    ) -> None:
        self._adapter = adapter
        self._model = model
        self._processor = processor
        self._config = model_config
        self._adapter_metadata = adapter_metadata or {}
        self._resolved_revision = resolved_revision or getattr(model_config, "revision", "unresolved")

        # Ensure model is in eval mode.
        self._model.eval()

    # ------------------------------------------------------------------ #
    # Prefix construction (delegates to adapter)
    # ------------------------------------------------------------------ #

    def _build_prefix(self, image: Any, prompt: str) -> dict[str, Any]:
        """Build the multimodal assistant prefix via the adapter.

        Delegates to ``adapter.build_prefix()`` which handles model-specific
        chat template rendering and multimodal tensor construction.
        The result is unsqueezed to batch dim=1 for scoring compatibility.
        """
        prefix = self._adapter.build_prefix(self._processor, image=image, prompt=prompt)

        # Move tensors to model device and unsqueeze text tensors to batch dim.
        # build_prefix() squeezes batch dim for training; scoring needs 2D.
        device = self._model.get_input_embeddings().weight.device
        image_indexed = self._adapter.image_indexed_keys()
        result: dict[str, Any] = {}
        for k, v in prefix.items():
            if k.startswith("_"):
                result[k] = v
                continue
            if torch.is_tensor(v):
                v = v.to(device)
                if k not in image_indexed and v.dim() == 1:
                    v = v.unsqueeze(0)
            result[k] = v
        return result

    # ------------------------------------------------------------------ #
    # Generation (delegates to adapter for prefix, shared path for generate)
    # ------------------------------------------------------------------ #

    def generate(
        self, image: Any, prompt: str, *, max_new_tokens: int | None = None,
    ) -> VisionResponse:
        return self.generate_batch([(image, prompt)], max_new_tokens=max_new_tokens)[0]

    def generate_batch(
        self, items: list[tuple], *, max_new_tokens: int | None = None,
    ) -> list[VisionResponse]:
        images = [x[0] for x in items]
        prompts = [x[1] for x in items]

        # Build prefix for each item via the adapter.
        # For batched generation, we process one at a time (the baseline
        # evaluator calls generate() per probe anyway).
        responses: list[VisionResponse] = []
        for image, prompt in zip(images, prompts):
            prefix = self._build_prefix(image, prompt)
            # Filter out metadata keys (starting with '_') before passing
            # to model.generate() — the model doesn't accept them.
            model_inputs = {k: v for k, v in prefix.items() if not k.startswith("_")}
            input_len = model_inputs["input_ids"].shape[1]

            effective_max_new_tokens = (
                max_new_tokens
                if max_new_tokens is not None
                else getattr(self._config.generation, "max_new_tokens", 128)
            )

            started = time.perf_counter()
            with torch.inference_mode():
                output = self._model.generate(
                    **model_inputs,
                    do_sample=getattr(self._config.generation, "do_sample", False),
                    temperature=(
                        getattr(self._config.generation, "temperature", 0.0)
                        if getattr(self._config.generation, "do_sample", False)
                        else None
                    ),
                    max_new_tokens=effective_max_new_tokens,
                )
            latency_ms = (time.perf_counter() - started) * 1000.0

            new_tokens = output[0, input_len:]
            generated_token_count = len(new_tokens)
            hit_max = generated_token_count >= effective_max_new_tokens

            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            eos_reached = (
                generated_token_count > 0
                and eos_token_id is not None
                and new_tokens[-1].item() == eos_token_id
            )
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            is_text_only = image is None
            responses.append(
                VisionResponse(
                    text=text,
                    metadata={
                        "latency_ms": latency_ms,
                        "input_mode": "text_only" if is_text_only else "multimodal",
                        "image_present": not is_text_only,
                        "generated_token_count": generated_token_count,
                        "hit_max_new_tokens": hit_max,
                        "eos_reached": eos_reached,
                    },
                )
            )
        return responses

    # ------------------------------------------------------------------ #
    # Candidate scoring (shared scorer + adapter-driven prefix)
    # ------------------------------------------------------------------ #

    def score_candidates(
        self, image: Any, prompt: str, candidates: list[str],
    ) -> VisionResponse:
        """Score candidates using the shared differentiable scorer.

        Uses ``score_candidate_sequence_tensor`` with ``adapter=self._adapter``
        for model-agnostic forward construction.
        """
        if not candidates:
            raise ValueError("candidates must be non-empty")

        tokenizer = getattr(self._processor, "tokenizer", self._processor)

        # Validate candidate tokenization via the adapter.
        cand_token_map: dict[str, list[int]] = {}
        for cand in candidates:
            ids = self._adapter.candidate_token_ids(self._processor, cand)
            assert len(ids) > 0, f"candidate tokenized to empty sequence: {cand!r}"
            cand_token_map[cand] = ids

        # Distinct token sequences for distinct candidates.
        unique_id_tuples = {tuple(v) for v in cand_token_map.values()}
        assert len(unique_id_tuples) == len(candidates), (
            f"candidates did not produce distinct token sequences: {cand_token_map}"
        )

        # Build multimodal prefix ONCE via the adapter.
        prefix = self._build_prefix(image, prompt)
        prefix_len = prefix["input_ids"].shape[1]
        assert prefix_len > 0, "prefix length must be > 0"

        scores: list[CandidateScore] = []
        debug_info: list[dict] = []

        for candidate in candidates:
            cand_ids_list = cand_token_map[candidate]

            with torch.inference_mode():
                log_prob_tensor = score_candidate_sequence_tensor(
                    self._model, prefix, cand_ids_list, adapter=self._adapter,
                )
            log_prob = log_prob_tensor.item()

            assert math.isfinite(log_prob), (
                f"non-finite log probability for {candidate!r}: {log_prob}"
            )

            scores.append(
                CandidateScore(
                    candidate=candidate,
                    log_probability=log_prob,
                )
            )
            debug_info.append({
                "candidate": candidate,
                "candidate_token_ids": cand_ids_list,
                "candidate_decoded": tokenizer.decode(cand_ids_list),
                "prefix_length": prefix_len,
                "full_length": prefix_len + len(cand_ids_list),
                "scored_positions": list(range(prefix_len - 1, prefix_len - 1 + len(cand_ids_list))),
                "scorer": "score_candidate_sequence_tensor",
                "adapter": self._adapter.profile.key,
            })

        # Cross-candidate sanity check.
        if len(scores) >= 2:
            log_probs = [s.log_probability for s in scores]
            if len({round(lp, 8) for lp in log_probs}) == 1:
                import warnings
                warnings.warn(
                    f"All candidate log probabilities are identical: {log_probs}. "
                    "This indicates the model is completely uncertain (50/50 probability).",
                    UserWarning,
                )

        metadata: dict = {
            "adapter_family": self._adapter.profile.key,
            "scoring_debug": debug_info,
        }
        return VisionResponse(
            text="",
            candidate_scores=scores,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Fingerprint (adapter-aware)
    # ------------------------------------------------------------------ #

    def fingerprint(self) -> dict[str, str]:
        import transformers

        profile = self._adapter.profile
        payload = {
            "backend": "adapter_eval_backend",
            "adapter_family": profile.key,
            "model_id": profile.model_id,
            "revision": self._resolved_revision,
            "processor_revision": profile.processor_revision,
            "dtype": profile.dtype,
            "attn": profile.attn_implementation or "default",
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        }

        # Processor/tokenizer class for cache differentiation.
        try:
            proc = self._processor
            payload["processor_class"] = type(proc).__name__
            if hasattr(proc, "tokenizer"):
                payload["tokenizer_class"] = type(proc.tokenizer).__name__
            if hasattr(proc, "chat_template"):
                tpl = proc.chat_template or ""
                payload["chat_template_hash"] = hashlib.sha256(
                    tpl.encode()
                ).hexdigest()[:16]
        except Exception:
            pass

        # Adapter metadata (checkpoint SHA, LoRA hyperparams).
        if self._adapter_metadata:
            for key in (
                "adapter_checkpoint_path",
                "adapter_checkpoint_sha",
                "adapter_config_sha",
                "checkpoint_name",
                "checkpoint_step",
                "lora_rank",
                "lora_alpha",
                "lora_target_modules",
                "base_fingerprint_id",
            ):
                val = self._adapter_metadata.get(key)
                if val is not None:
                    payload[key] = val

        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        return {k: str(v) for k, v in payload.items()} | {
            "fingerprint_id": digest,
        }
