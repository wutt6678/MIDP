"""Pluggable vision-language model adapters.

Each adapter exposes a minimal interface used by the evaluation loop:

    load(cfg)                                  load model + processor
    score_batch(images, questions) -> (yes, no)   first-token logits scores
    generate(image, question, max_new_tokens)     free-form sanity check

The Yes/No decision is made from the logits of the first generated token
(max of Yes/ Yes-token logits vs. No/ No-token logits), so evaluation needs
only a single forward pass per batch.

Register a new model family with the ``@register_model("name")`` decorator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from PIL import Image

from .config import ModelConfig

_REGISTRY: dict[str, type["ModelAdapter"]] = {}

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def register_model(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def get_model_adapter(name: str) -> "ModelAdapter":
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model adapter '{name}'. "
                         f"Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def get_yes_no_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Token ids counted as Yes / No evidence (with & without leading space)."""
    yes_ids, no_ids = [], []
    for word, bucket in [("Yes", yes_ids), (" Yes", yes_ids),
                         ("No", no_ids), (" No", no_ids)]:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            bucket.append(ids[0])
    if not yes_ids or not no_ids:
        raise RuntimeError(f"Could not resolve Yes/No tokens: {yes_ids} / {no_ids}")
    return sorted(set(yes_ids)), sorted(set(no_ids))


class ModelAdapter(ABC):
    name: str = ""

    @abstractmethod
    def load(self, cfg: ModelConfig) -> None:
        """Load processor/tokenizer and model according to cfg."""

    @abstractmethod
    def build_prompt(self, image: Image.Image, question: str) -> str:
        """Render the model-specific chat prompt for one (image, question)."""

    @abstractmethod
    def score_batch(self, images: list[Image.Image], questions: list[str]):
        """Return (yes_scores, no_scores) tensors for one batch."""

    # ---- shared helpers ----

    def device(self) -> torch.device:
        return getattr(self, "_device", torch.device("cpu"))

    def _device_from_cfg(self, cfg: ModelConfig) -> torch.device:
        return torch.device(cfg.device_map
                            if cfg.device_map.startswith("cuda") else "cpu")

    def _yes_no_from_next_logits(self, logits: torch.Tensor,
                                 attention_mask: torch.Tensor):
        """Extract Yes/No scores from the last non-pad position (next-token)."""
        seq_lens = attention_mask.sum(dim=1) - 1
        next_logits = logits[torch.arange(logits.shape[0], device=logits.device),
                             seq_lens]
        yes_score = next_logits[:, self.yes_ids].max(dim=1).values
        no_score = next_logits[:, self.no_ids].max(dim=1).values
        return yes_score, no_score

    @torch.no_grad()
    def generate(self, image: Image.Image, question: str,
                 max_new_tokens: int = 16) -> str:
        """Free-form generation sanity check on one image."""
        prompt = self.build_prompt(image, question)
        inputs = self._encode([image], [prompt]).to(self.device())
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False)
        return self._tokenizer().decode(out[0][inputs.input_ids.shape[1]:],
                                        skip_special_tokens=True)

    # ---- hooks used by generate(); implemented by subclasses ----

    @abstractmethod
    def _encode(self, images: list, prompts: list[str]):
        """Tokenize + preprocess a batch (no padding target device yet)."""

    @abstractmethod
    def _tokenizer(self):
        """Return the tokenizer used for decoding."""


@register_model("mllama")
class MllamaAdapter(ModelAdapter):
    """Llama-3.2-Vision family (MllamaForConditionalGeneration)."""

    def load(self, cfg: ModelConfig) -> None:
        from transformers import MllamaForConditionalGeneration, MllamaProcessor
        dtype = _DTYPES[cfg.dtype]
        print(f"[load] adapter=mllama model={cfg.model_id} dtype={cfg.dtype} "
              f"device_map={cfg.device_map}")
        self.processor = MllamaProcessor.from_pretrained(cfg.model_id)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            cfg.model_id, torch_dtype=dtype, device_map=cfg.device_map
        )
        self.model.eval()
        self.yes_ids, self.no_ids = get_yes_no_token_ids(self.processor.tokenizer)
        self._device = self._device_from_cfg(cfg)

    def build_prompt(self, image: Image.Image, question: str) -> str:
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def _encode(self, images: list, prompts: list[str]):
        return self.processor(images=images, text=prompts, padding=True,
                              return_tensors="pt", add_special_tokens=False)

    def _tokenizer(self):
        return self.processor.tokenizer

    @torch.no_grad()
    def score_batch(self, images: list[Image.Image], questions: list[str]):
        prompts = [self.build_prompt(img, q) for img, q in zip(images, questions)]
        inputs = self._encode(images, prompts).to(self.device())
        logits = self.model(**inputs).logits  # (B, T, V)
        return self._yes_no_from_next_logits(logits, inputs.attention_mask)


@register_model("auto")
class AutoVLMAdapter(ModelAdapter):
    """Best-effort generic adapter via AutoProcessor + AutoModelForImageTextToText.

    Works for chat-template-based VLMs supported by transformers' Auto classes
    (e.g. Llava, Idefics3, Qwen2-VL). The model must accept interleaved
    image/text chat messages and expose a text tokenizer.
    """

    def load(self, cfg: ModelConfig) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        dtype = _DTYPES[cfg.dtype]
        print(f"[load] adapter=auto model={cfg.model_id} dtype={cfg.dtype} "
              f"device_map={cfg.device_map}")
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_id, torch_dtype=dtype, device_map=cfg.device_map
        )
        self.model.eval()
        self.yes_ids, self.no_ids = get_yes_no_token_ids(self._tokenizer())
        self._device = self._device_from_cfg(cfg)

    def build_prompt(self, image: Image.Image, question: str) -> str:
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def _encode(self, images: list, prompts: list[str]):
        return self.processor(images=images, text=prompts, padding=True,
                              return_tensors="pt")

    def _tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    @torch.no_grad()
    def score_batch(self, images: list[Image.Image], questions: list[str]):
        prompts = [self.build_prompt(img, q) for img, q in zip(images, questions)]
        inputs = self._encode(images, prompts).to(self.device())
        logits = self.model(**inputs).logits  # (B, T, V)
        return self._yes_no_from_next_logits(logits, inputs.attention_mask)


@register_model("qwen")
class QwenAdapter(AutoVLMAdapter):
    """Qwen VLMs: Qwen3.5 (Qwen3_5ForConditionalGeneration) and Qwen3-VL
    (Qwen3VLForConditionalGeneration).

    These load like the ``auto`` adapter, but the thinking-capable variants
    *think by default* (emit reasoning before answering). The evaluation
    scores the logits of the first generated token, so thinking must be
    disabled to make that token the Yes/No answer. We therefore render the
    prompt with ``enable_thinking=False`` (a harmless no-op for non-thinking
    variants like Qwen3-VL-*-Instruct).
    """

    def build_prompt(self, image: Image.Image, question: str) -> str:
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        try:
            return self.processor.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            # Chat templates that don't accept the enable_thinking kwarg.
            return self.processor.apply_chat_template(
                messages, add_generation_prompt=True)
