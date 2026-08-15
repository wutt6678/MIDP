"""Qwen3.5 Hugging Face backend for route-data dataset construction.

Prepared for Qwen/Qwen3.5-9B.  This backend intentionally mirrors the
VisionLanguageModel interface used by the route-unlearning-data package.

Important:
- Qwen3.5 thinking is disabled in the chat template.
- Candidate answers are scored via prefix-token teacher forcing: the
  multimodal prefix is constructed once with ``processor.apply_chat_template``
  and candidate token IDs are explicitly appended.  This makes the
  conditional probability mathematically explicit:

      log P(c | x) = sum_j log P(c_j | x, c_<j)

  where x is the exact image-conditioned assistant prefix.
- The frozen binary candidate protocol uses ``"Yes"`` / ``"No"``
  (capitalized, no leading space).  Both candidates are validated at
  initialization to tokenize to non-empty, distinct sequences.
- No model is loaded until a generation/scoring method or fingerprint()
  requiring revision resolution is called.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from ..config import ModelConfig
from .base import CandidateScore, VisionLanguageModel, VisionResponse
from .registry import register_backend
from .scoring import gather_sequence_log_probs

# Frozen binary candidate protocol.
# The Qwen3.5 chat template ends the assistant prefix after the empty
# <think>...</think> block; the model naturally produces capitalized
# "Yes" / "No" as the first token.  Do NOT change these without
# re-running the diagnostic in scripts/diagnose_qwen_logits.py.
POSITIVE_CANDIDATE = "Yes"
NEGATIVE_CANDIDATE = "No"
BINARY_CANDIDATES: tuple[str, ...] = (POSITIVE_CANDIDATE, NEGATIVE_CANDIDATE)

_DTYPE_MAP = {
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
}


@register_backend("qwen_hf")
def _create(config: ModelConfig) -> VisionLanguageModel:
    return QwenHFBackend(config)


class QwenHFBackend(VisionLanguageModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._resolved_revision: str | None = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor

        cfg = self.config
        torch.manual_seed(cfg.seed)

        local_dir = snapshot_download(cfg.model_id, revision=cfg.revision)
        self._resolved_revision = cfg.revision or _read_head_commit(local_dir)

        torch_dtype = getattr(torch, _DTYPE_MAP[cfg.dtype])
        kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": cfg.trust_remote_code,
            "revision": cfg.revision,
        }
        if cfg.device_map:
            kwargs["device_map"] = cfg.device_map
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation

        if cfg.quantization.enabled:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=cfg.quantization.mode in {"nf4", "fp4"},
                load_in_8bit=cfg.quantization.mode == "int8",
                bnb_4bit_quant_type=cfg.quantization.mode,
                bnb_4bit_compute_dtype=getattr(
                    torch, _DTYPE_MAP[cfg.quantization.compute_dtype]
                ),
                bnb_4bit_use_double_quant=cfg.quantization.double_quant,
            )

        self._model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_id, **kwargs
        )
        self._processor = AutoProcessor.from_pretrained(
            cfg.resolved_processor_id,
            revision=cfg.revision,
            trust_remote_code=cfg.trust_remote_code,
        )
        self._model.eval()

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def processor(self):
        self._load()
        return self._processor

    def _render(self, prompt: str, *, image: bool = True) -> str:
        """Render chat template for Qwen3.5.

        Args:
            prompt: The user prompt text.
            image: If True, include image placeholder (multimodal path).
                   If False, text-only path with no image token.
        """
        if image:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }]
        else:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }]
        try:
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )

    def _build_prefix(self, image, prompt: str) -> dict:
        """Construct the prefix tensor dict via the official route.

        For multimodal (image is not None): uses image + text chat template.
        For text-only (image is None): uses text-only chat template with no
        image token, no pixel_values, no image_grid_thw.

        Uses ``processor.apply_chat_template`` with ``enable_thinking=False``
        so the assistant prefix ends after the empty ``<think>...</think>``
        block.  The returned dict contains ``input_ids`` and
        ``attention_mask`` ready for ``model(**prefix)``.
        """
        if image is not None:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]
        else:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }]
        try:
            prefix = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            prefix = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        device = self.model.get_input_embeddings().weight.device
        return {k: v.to(device) for k, v in prefix.items()}

    def _prepare(self, images, prompts, *, padding_side: str):
        """Prepare batch inputs for generation.

        Handles both multimodal (images present) and text-only (all images
        are None) cases. For text-only, the processor is called without
        the images argument to avoid passing [None].
        """
        processor = self.processor
        # Determine if this is a text-only batch (all images are None).
        all_none = all(img is None for img in images)
        texts = [self._render(prompt, image=not all_none) for prompt in prompts]
        tokenizer = getattr(processor, "tokenizer", None)
        old_side = getattr(tokenizer, "padding_side", "right") if tokenizer else None
        if tokenizer is not None:
            tokenizer.padding_side = padding_side
        try:
            if all_none:
                # Text-only path: no images argument to processor.
                batch = processor(
                    text=texts,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                # Multimodal path: pass images list.
                batch = processor(
                    images=list(images),
                    text=texts,
                    padding=True,
                    return_tensors="pt",
                )
        finally:
            if tokenizer is not None and old_side is not None:
                tokenizer.padding_side = old_side

        device = self.model.get_input_embeddings().weight.device
        return {k: v.to(device) for k, v in batch.items()}

    def generate(self, image, prompt: str) -> VisionResponse:
        return self.generate_batch([(image, prompt)])[0]

    def generate_batch(self, items: list[tuple]) -> list[VisionResponse]:
        import torch

        images = [x[0] for x in items]
        prompts = [x[1] for x in items]
        inputs = self._prepare(images, prompts, padding_side="left")
        input_len = inputs["input_ids"].shape[1]
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=self.config.generation.do_sample,
                temperature=(
                    self.config.generation.temperature
                    if self.config.generation.do_sample
                    else None
                ),
                max_new_tokens=self.config.generation.max_new_tokens,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        tokenizer = self.processor.tokenizer
        responses = []
        for row in output:
            text = tokenizer.decode(row[input_len:], skip_special_tokens=True)
            responses.append(
                VisionResponse(
                    text=text,
                    metadata={"latency_ms": latency_ms / max(1, len(items))},
                )
            )
        return responses

    def score_candidates(
        self, image, prompt: str, candidates: list[str]
    ) -> VisionResponse:
        if not candidates:
            raise ValueError("candidates must be non-empty")

        import torch

        tokenizer = self.processor.tokenizer

        # ── Validate candidate tokenization ─────────────────────────────
        cand_token_map: dict[str, list[int]] = {}
        for cand in candidates:
            ids = tokenizer.encode(cand, add_special_tokens=False)
            assert len(ids) > 0, (
                f"candidate tokenized to empty sequence: {cand!r}"
            )
            cand_token_map[cand] = ids

        # Distinct token sequences for distinct candidates.
        unique_id_tuples = {tuple(v) for v in cand_token_map.values()}
        assert len(unique_id_tuples) == len(candidates), (
            f"candidates did not produce distinct token sequences: "
            f"{cand_token_map}"
        )

        # ── Build multimodal prefix ONCE ────────────────────────────────
        prefix = self._build_prefix(image, prompt)
        prefix_input_ids = prefix["input_ids"]
        prefix_len = prefix_input_ids.shape[1]
        assert prefix_len > 0, "prefix length must be > 0"

        scores: list[CandidateScore] = []
        debug_info: list[dict] = []

        for candidate in candidates:
            cand_ids_list = cand_token_map[candidate]
            cand_ids_tensor = torch.tensor(
                [cand_ids_list], dtype=prefix_input_ids.dtype,
                device=prefix_input_ids.device,
            )

            # ── Explicitly append candidate IDs to prefix ───────────────
            full_input_ids = torch.cat(
                [prefix_input_ids, cand_ids_tensor], dim=1
            )
            full_attention_mask = torch.cat(
                [
                    prefix["attention_mask"],
                    torch.ones_like(cand_ids_tensor),
                ],
                dim=1,
            )
            # Extend mm_token_type_ids for candidate tokens (text-only, type 0).
            full_mm_token_type_ids = None
            if "mm_token_type_ids" in prefix:
                prefix_mm_ids = prefix["mm_token_type_ids"]
                cand_mm_ids = torch.zeros_like(
                    cand_ids_tensor, dtype=prefix_mm_ids.dtype
                )
                full_mm_token_type_ids = torch.cat(
                    [prefix_mm_ids, cand_mm_ids], dim=1
                )
            full_len = full_input_ids.shape[1]
            m = len(cand_ids_list)

            # ── Alignment assertions ────────────────────────────────────
            assert full_len == prefix_len + m, (
                f"full length {full_len} != prefix {prefix_len} + "
                f"candidate {m}"
            )
            target_slice = full_input_ids[0, prefix_len:]
            assert target_slice.tolist() == cand_ids_list, (
                f"target candidate IDs {target_slice.tolist()} do not "
                f"match appended IDs {cand_ids_list}"
            )

            forward_kwargs = {
                "input_ids": full_input_ids,
                "attention_mask": full_attention_mask,
            }
            # Forward multimodal tensors from prefix if present.
            # Qwen3.5 requires mm_token_type_ids for M-RoPE computation.
            for key in ("pixel_values", "image_sizes", "image_grid_thw"):
                if key in prefix:
                    forward_kwargs[key] = prefix[key]
            # Use extended mm_token_type_ids (prefix + text-only candidate).
            if full_mm_token_type_ids is not None:
                forward_kwargs["mm_token_type_ids"] = full_mm_token_type_ids

            with torch.inference_mode():
                outputs = self.model(**forward_kwargs)

            # Score candidate positions: logit at prefix_len-1 predicts
            # the first candidate token, logit at prefix_len predicts the
            # second, etc.
            pred_rows = outputs.logits[0, prefix_len - 1: prefix_len - 1 + m, :]
            target_ids = full_input_ids[0, prefix_len: prefix_len + m]
            log_prob = gather_sequence_log_probs(
                pred_rows, target_ids.to(pred_rows.device)
            )

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
                "full_length": full_len,
                "scored_positions": list(
                    range(prefix_len - 1, prefix_len - 1 + m)
                ),
            })

        # Cross-candidate sanity: log probs should not be accidentally
        # identical (would indicate a scoring bug).
        # Note: identical logits can occur when the model is completely uncertain
        # (e.g., 50/50 probability), which is valid behavior.
        if len(scores) >= 2:
            log_probs = [s.log_probability for s in scores]
            # Warn if all log probs are identical (may indicate model uncertainty)
            if len({round(lp, 8) for lp in log_probs}) == 1:
                import warnings
                warnings.warn(
                    f"All candidate log probabilities are identical: {log_probs}. "
                    "This indicates the model is completely uncertain (50/50 probability).",
                    UserWarning
                )

        metadata: dict = {
            "thinking_disabled": True,
            "scoring_debug": debug_info,
        }
        return VisionResponse(
            text="",
            candidate_scores=scores,
            metadata=metadata,
        )

    def fingerprint(self) -> dict[str, str]:
        import torch
        import transformers

        cfg = self.config
        payload = {
            "backend": "qwen_hf",
            "model_id": cfg.model_id,
            "revision": self._resolved_revision or cfg.revision or "unresolved",
            "dtype": cfg.dtype,
            "quantization": (
                cfg.quantization.mode if cfg.quantization.enabled else "none"
            ),
            "attn": cfg.attn_implementation,
            "thinking": "disabled",
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        return {k: str(v) for k, v in payload.items()} | {
            "fingerprint_id": digest
        }


def _read_head_commit(local_dir: str) -> str | None:
    """Resolve the commit hash for a ``snapshot_download`` directory.

    Tries common ref files first (``refs/main``, ``refs/HEAD``), then
    falls back to the snapshot directory name itself — ``snapshot_download``
    places files under ``…/snapshots/<commit_hash>/``.
    """
    # 1. Try well-known ref files inside the snapshot root.
    for rel in ("refs/main", "refs/HEAD"):
        ref_path = Path(local_dir) / rel
        if ref_path.is_file():
            try:
                value = ref_path.read_text().strip()
                if value:
                    return value
            except OSError:
                pass

    # 2. Fall back to the basename of the snapshot directory, which
    #    snapshot_download names after the resolved commit hash.
    stem = Path(local_dir).name
    # Commit hashes are 40-char hex (or 7+ char short hashes).
    if len(stem) >= 7 and all(c in "0123456789abcdef" for c in stem):
        return stem
    return None
