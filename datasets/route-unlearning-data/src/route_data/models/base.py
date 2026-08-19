"""Model adapter interface (coding plan section 6.2).

The backend is configurable and replaceable without touching dataset-
construction or evaluation code. Every backend returns the same
:class:`VisionResponse` schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CandidateScore:
    """Sequence-level score for one answer candidate (e.g. "Yes")."""

    candidate: str
    log_probability: float


@dataclass(frozen=True)
class VisionResponse:
    text: str
    candidate_scores: list[CandidateScore] | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteHooks:
    """Optional hooks for later mechanistic route analysis (plan section 18.3).

    Disabled by default in dataset construction because storing activations is
    expensive; the interface exists now so later experiments do not require
    rewriting the input pipeline.
    """

    return_hidden_states: bool = False
    return_attentions: bool = False
    return_token_indices: bool = False


class VisionLanguageModel(ABC):
    @abstractmethod
    def generate(self, image, prompt: str, *, max_new_tokens: int | None = None) -> VisionResponse:
        """Deterministic text generation for one image + prompt.

        Parameters
        ----------
        max_new_tokens:
            Optional override for the generation budget.  When *None*
            the backend's default ``GenerationConfig.max_new_tokens`` is
            used.  Text-only probe families (e.g. ``name_only``) pass a
            larger budget (e.g. 64) to avoid artificial truncation.
        """

    @abstractmethod
    def score_candidates(self, image, prompt: str, candidates: list[str]) -> VisionResponse:
        """Score full candidate sequences conditioned on image + prompt."""

    @abstractmethod
    def fingerprint(self) -> dict[str, str]:
        """Stable identity used for caching and run manifests."""

    def generate_batch(self, items: list[tuple], *, max_new_tokens: int | None = None) -> list[VisionResponse]:
        """Batched generation; default loops over :meth:`generate`."""
        return [self.generate(image, prompt, max_new_tokens=max_new_tokens) for image, prompt in items]

    def score_candidates_batch(self, items: list[tuple]) -> list[VisionResponse]:
        """Batched scoring; default loops over :meth:`score_candidates`."""
        return [self.score_candidates(image, prompt, cands) for image, prompt, cands in items]
