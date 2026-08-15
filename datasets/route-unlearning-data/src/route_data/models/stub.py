"""Deterministic stub backend (coding plan section 6.2, Phase 0).

A lightweight, dependency-free backend used for unit tests, dry runs and
offline pipeline validation. It never loads a real model: responses are a
pure function of (prompt, candidate, seed, image size) so every run is
bit-reproducible. It is registered both as ``stub`` and ``example_vlm``
(the plan's placeholder name for a replaceable third-party VLM).
"""

from __future__ import annotations

import hashlib
import json
import math

from ..config import ModelConfig
from .base import CandidateScore, VisionLanguageModel, VisionResponse
from .registry import register_backend


def _unit(*parts: object) -> float:
    """Deterministic float in [0, 1) derived from the given parts."""
    payload = json.dumps([str(p) for p in parts], ensure_ascii=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _image_signature(image) -> str:
    """Cheap stable signature of an image-like object (size + mode)."""
    size = getattr(image, "size", None) or (0, 0)
    mode = getattr(image, "mode", "unknown")
    return f"{int(size[0])}x{int(size[1])}:{mode}"


class StubVisionModel(VisionLanguageModel):
    """Hash-based dummy VLM. Deterministic and fully offline."""

    def __init__(self, config: ModelConfig, role_name: str = "stub") -> None:
        self.config = config
        self.role_name = role_name
        # P0-4: stable immutable revision so score manifests carry a
        # deterministic resolved_revision for the stub backend.
        self._resolved_revision = getattr(config, "revision", None) or "stub-v1"

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(self, image, prompt: str) -> VisionResponse:
        sig = _image_signature(image)
        p_yes = _unit(self.config.seed, sig, prompt, "yes")
        text = "yes" if p_yes >= 0.5 else "no"
        return VisionResponse(
            text=text,
            candidate_scores=None,
            metadata={"backend": self.role_name, "p_yes": p_yes},
        )

    def score_candidates(
        self, image, prompt: str, candidates: list[str]
    ) -> VisionResponse:
        sig = _image_signature(image)
        # One deterministic posterior per query; candidate log-probabilities
        # encode it so downstream softmax recovers exactly this distribution
        # (full spread keeps the confidence-band ladder exercisable).
        posterior = {
            candidate: _unit(self.config.seed, sig, prompt, candidate)
            for candidate in candidates
        }
        total = sum(posterior.values())
        p_by_candidate = {c: p / total for c, p in posterior.items()}
        scores = [
            CandidateScore(
                candidate=candidate,
                log_probability=math.log(max(p_by_candidate[candidate], 1e-6)),
            )
            for candidate in candidates
        ]
        return VisionResponse(text="", candidate_scores=scores, metadata={})

    # ------------------------------------------------------------------ #
    # Fingerprint (plan sections 5.2, 8.5)
    # ------------------------------------------------------------------ #

    def fingerprint(self) -> dict[str, str]:
        payload = {
            "backend": self.role_name,
            "model_id": self.config.model_id,
            "revision": self._resolved_revision,
            "seed": self.config.seed,
            "dtype": self.config.dtype,
            "attn_implementation": self.config.attn_implementation,
            "processor_id": self.config.resolved_processor_id,
            "quantization": {
                "enabled": self.config.quantization.enabled,
                "mode": self.config.quantization.mode,
                "compute_dtype": self.config.quantization.compute_dtype,
                "double_quant": self.config.quantization.double_quant,
            },
            "generation": {
                "do_sample": self.config.generation.do_sample,
                "temperature": self.config.generation.temperature,
                "max_new_tokens": self.config.generation.max_new_tokens,
            },
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        return {k: str(v) for k, v in payload.items()} | {"fingerprint_id": digest}


@register_backend("stub")
def _create_stub(config: ModelConfig) -> VisionLanguageModel:
    return StubVisionModel(config, role_name="stub")


@register_backend("example_vlm")
def _create_example_vlm(config: ModelConfig) -> VisionLanguageModel:
    return StubVisionModel(config, role_name="example_vlm")
