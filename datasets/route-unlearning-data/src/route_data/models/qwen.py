"""Qwen3.5 Hugging Face backend for route-data dataset construction.

Prepared for Qwen/Qwen3.5-9B.  This backend intentionally mirrors the
VisionLanguageModel interface used by the route-unlearning-data package.

Important:
- Qwen3.5 thinking is disabled in the chat template.
- Candidate answers are scored as full token sequences, not first-token logits.
- No model is loaded until a generation/scoring method or fingerprint() requiring
  revision resolution is called.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..config import ModelConfig
from .base import CandidateScore, VisionLanguageModel, VisionResponse
from .registry import register_backend
from .scoring import gather_sequence_log_probs


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
        import os
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

    def _render(self, prompt: str) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
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

    def _prepare(self, images, prompts, *, padding_side: str):
        processor = self.processor
        texts = [self._render(prompt) for prompt in prompts]
        tokenizer = getattr(processor, "tokenizer", None)
        old_side = getattr(tokenizer, "padding_side", "right") if tokenizer else None
        if tokenizer is not None:
            tokenizer.padding_side = padding_side
        try:
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

        scores: list[CandidateScore] = []
        for candidate in candidates:
            # Render chat prompt first, then append candidate to the rendered
            # assistant-generation prefix. This avoids inserting the candidate
            # inside the user's natural-language prompt.
            rendered = self._render(prompt)
            processor = self.processor
            tokenizer = processor.tokenizer

            # Encode rendered prompt + candidate directly.
            batch = processor(
                images=[image],
                text=[rendered + candidate],
                padding=True,
                return_tensors="pt",
            )
            device = self.model.get_input_embeddings().weight.device
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]

            cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
            m = len(cand_ids)
            if m == 0:
                raise ValueError(f"candidate tokenized to empty sequence: {candidate!r}")

            target_ids = input_ids[0, -m:]
            with torch.inference_mode():
                outputs = self.model(**batch)
            pred_rows = outputs.logits[0, -m - 1:-1, :]
            log_prob = gather_sequence_log_probs(
                pred_rows, target_ids.to(pred_rows.device)
            )
            scores.append(
                CandidateScore(
                    candidate=candidate,
                    log_probability=log_prob,
                )
            )

        return VisionResponse(
            text="",
            candidate_scores=scores,
            metadata={"thinking_disabled": True},
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
