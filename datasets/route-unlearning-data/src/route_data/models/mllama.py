"""Hugging Face Mllama backend for Llama 3.2 Vision (coding plan sections
2.1, 6.1, 6.3, 5.3).

Implementation notes enforced here:

- one user turn containing an image item and a text item; no system message
  for the main benchmark;
- ``processor.apply_chat_template(..., add_generation_prompt=True)``;
- only newly generated tokens are decoded (slice after the input length);
- inputs are moved to the device of the embedding module (``model.device`` is
  not reliable for sharded modules);
- left padding for batched generation, right padding for scoring forwards;
- 4-bit loading via bitsandbytes is optional and fails with an actionable
  message when unavailable.
"""

from __future__ import annotations

import hashlib
import json
import time

from ..config import ModelConfig
from .base import CandidateScore, VisionLanguageModel, VisionResponse
from .registry import register_backend
from .scoring import gather_sequence_log_probs

_DTYPE_MAP = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}


@register_backend("mllama_hf")
def _create(config: ModelConfig) -> VisionLanguageModel:
    return MllamaHFBackend(config)


class MllamaHFBackend(VisionLanguageModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._resolved_revision: str | None = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _load(self):
        if self._model is not None:
            return
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        cfg = self.config
        torch.manual_seed(cfg.seed)

        # Resolve and record the exact commit SHA used (plan section 5.2).
        local_dir = snapshot_download(cfg.model_id, revision=cfg.revision, allow_patterns=None)

        self._resolved_revision = cfg.revision or _read_head_commit(local_dir)

        torch_dtype = getattr(torch, _DTYPE_MAP[cfg.dtype])
        kwargs: dict = {
            "torch_dtype": torch_dtype,
            "attn_implementation": cfg.attn_implementation,
            "trust_remote_code": cfg.trust_remote_code,
            "revision": cfg.revision,
        }
        if cfg.device_map:
            kwargs["device_map"] = cfg.device_map
        if cfg.quantization.enabled:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "Quantized loading requires a transformers version exposing "
                    "BitsAndBytesConfig."
                ) from exc
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "4-bit/8-bit inference requires the optional dependency "
                    "bitsandbytes (install with `pip install bitsandbytes`)."
                ) from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=cfg.quantization.mode in {"nf4", "fp4"},
                load_in_8bit=cfg.quantization.mode == "int8",
                bnb_4bit_quant_type=cfg.quantization.mode,
                bnb_4bit_compute_dtype=getattr(torch, _DTYPE_MAP[cfg.quantization.compute_dtype]),
                bnb_4bit_use_double_quant=cfg.quantization.double_quant,
            )
        try:
            self._model = MllamaForConditionalGeneration.from_pretrained(cfg.model_id, **kwargs)
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover
            raise RuntimeError(
                "Out of memory while loading the model. Try the 4-bit profile "
                "(configs/model/llama32_11b_vision_4bit.yaml) or multi-GPU "
                "device_map. Progress from any completed shards is preserved."
            ) from exc
        self._processor = AutoProcessor.from_pretrained(
            cfg.resolved_processor_id, revision=cfg.revision, trust_remote_code=cfg.trust_remote_code
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

    # ------------------------------------------------------------------ #
    # Input preparation (plan section 6.3)
    # ------------------------------------------------------------------ #

    def _prepare(self, images: list, prompts: list[str], padding_side: str):
        processor = self.processor
        messages_batch = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            for prompt in prompts
        ]
        texts = [
            processor.apply_chat_template(m, add_generation_prompt=True) for m in messages_batch
        ]
        prev_side = getattr(processor.tokenizer, "padding_side", "right")
        processor.tokenizer.padding_side = padding_side
        try:
            inputs = processor(images=list(images), text=texts, padding=True, return_tensors="pt")
        finally:
            processor.tokenizer.padding_side = prev_side
        # Do not assume model.device covers sharded modules; anchor on the
        # text-embedding module (plan section 6.3).
        device = self.model.get_input_embeddings().weight.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate(self, image, prompt: str) -> VisionResponse:
        return self.generate_batch([(image, prompt)])[0]

    def generate_batch(self, items: list[tuple]) -> list[VisionResponse]:
        import torch

        cfg = self.config
        images = [img for img, _ in items]
        prompts = [p for _, p in items]
        inputs = self._prepare(images, prompts, padding_side="left")
        input_len = inputs["input_ids"].shape[1]
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=cfg.generation.do_sample,
                temperature=cfg.generation.temperature if cfg.generation.do_sample else None,
                max_new_tokens=cfg.generation.max_new_tokens,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        responses = []
        for row in output:
            new_tokens = row[input_len:]
            text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(
                VisionResponse(text=text, metadata={"latency_ms": latency_ms / len(items)})
            )
        return responses

    # ------------------------------------------------------------------ #
    # Candidate sequence scoring (plan section 8.4)
    # ------------------------------------------------------------------ #

    def score_candidates(self, image, prompt: str, candidates: list[str]) -> VisionResponse:
        return self.score_candidates_batch([(image, prompt, candidates)])[0]

    def score_candidates_batch(self, items: list[tuple]) -> list[VisionResponse]:
        import torch

        responses = []
        for image, prompt, candidates in items:
            if not candidates:
                raise ValueError("candidates must be non-empty")
            scores: list[CandidateScore] = []
            for candidate in candidates:
                inputs = self._prepare([image], [prompt + candidate], padding_side="right")
                input_ids = inputs["input_ids"]
                cand_tokens = self.processor.tokenizer.encode(candidate, add_special_tokens=False)
                m = len(cand_tokens)
                target_ids = input_ids[0, -m:]
                if target_ids.tolist() != cand_tokens:
                    # Tokenization of prompt+candidate differs from the
                    # standalone candidate (merging at the boundary). Fall
                    # back to the sliced tail which is what the model sees.
                    cand_tokens = target_ids.tolist()
                with torch.inference_mode():
                    outputs = self.model(**inputs)
                # Logit at position t predicts token t+1: take the rows that
                # predict each candidate token.
                pred_rows = outputs.logits[0, -m - 1 : -1, :]
                log_prob = gather_sequence_log_probs(
                    pred_rows, target_ids.to(pred_rows.device)
                )
                scores.append(CandidateScore(candidate=candidate, log_probability=log_prob))
            responses.append(VisionResponse(text="", candidate_scores=scores, metadata={}))
        return responses

    # ------------------------------------------------------------------ #
    # Fingerprint (plan sections 5.2, 8.5)
    # ------------------------------------------------------------------ #

    def fingerprint(self) -> dict[str, str]:
        import torch
        import transformers

        cfg = self.config
        payload = {
            "backend": "mllama_hf",
            "model_id": cfg.model_id,
            "revision": self._resolved_revision or cfg.revision or "unresolved",
            "dtype": cfg.dtype,
            "quantization": cfg.quantization.mode if cfg.quantization.enabled else "none",
            "attn": cfg.attn_implementation,
            "generation": {
                "do_sample": cfg.generation.do_sample,
                "temperature": cfg.generation.temperature,
                "max_new_tokens": cfg.generation.max_new_tokens,
            },
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        return {k: str(v) for k, v in payload.items()} | {"fingerprint_id": digest}


def _read_head_commit(local_dir: str) -> str | None:
    """Best-effort resolution of the checked-out commit SHA."""
    import os

    for rel in (".cache/huggingface/download_metadata", "refs/main"):
        path = os.path.join(local_dir, rel)
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    value = f.read().strip()
                if value:
                    return value
            except OSError:
                continue
    return None
