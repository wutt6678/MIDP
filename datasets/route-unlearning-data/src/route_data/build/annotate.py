"""Benchmark annotation with the frozen CelebA protocol (plan sections 11, 12.2).

Applies the *frozen* CelebA evaluation protocol to benchmark images to produce
per-image, per-attribute weak labels. Three rules from the plan are enforced:

- **Three-tier labeling** (plan 11.1): high-confidence automatic, human-verified,
  and unlabeled/uncertain. A binary label is never forced for coverage; samples
  whose calibrated confidence is below the accept threshold keep ``label=None``.
- **Frozen calibration** (plan 11.2): one calibrator per attribute, fit on CelebA
  validation only, applied frozen here. Calibrated scores are confidence
  indicators, not guaranteed probabilities, because of domain shift.
- **Namespaces** (plan 12.2): CelebA-style predictions are written under
  ``extended_attributes.celeba40.*`` and never overwrite source annotations such
  as ``source_attributes.fairface.*``.

Only attributes that passed the CelebA gate (plan 8.10) may become
high-confidence automatic labels; the gated set is supplied by the caller so the
policy stays explicit and configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from ..config import BuildConfig, ConfigError
from ..constants.attribute_taxonomy import group_of, is_reliability_flagged
from ..constants.celeba_attributes import CELEBA_ATTRIBUTE_SET
from ..data.schemas import AttributeObservation, CanonicalSample
from ..eval.calibration import Calibrator, load_calibrators

# Namespace for CelebA-style predictions added to benchmark images (plan 12.2).
CELEBA40_NAMESPACE = "extended_attributes.celeba40"

# Default decision-confidence bands (must satisfy 0 <= medium <= high <= 1).
DEFAULT_BANDS: dict[str, float] = {"high": 0.85, "medium": 0.60}


class AnnotationError(ValueError):
    """Raised when annotation inputs are inconsistent or misconfigured."""


def celeba40_key(attribute: str) -> str:
    """Namespaced key for a CelebA-style prediction on a benchmark image."""
    return f"{CELEBA40_NAMESPACE}.{attribute}"


def validate_bands(bands: Mapping[str, float]) -> dict[str, float]:
    high = float(bands.get("high", DEFAULT_BANDS["high"]))
    medium = float(bands.get("medium", DEFAULT_BANDS["medium"]))
    if not (0.0 <= medium <= high <= 1.0):
        raise ConfigError(
            f"confidence_bands must satisfy 0<=medium<=high<=1, got {dict(bands)}"
        )
    return {"high": high, "medium": medium}


def confidence_band(decision_confidence: float, bands: Mapping[str, float]) -> str:
    """Map a decision confidence in [0.5, 1] to a coarse confidence band."""
    if decision_confidence >= bands["high"]:
        return "high"
    if decision_confidence >= bands["medium"]:
        return "medium"
    return "low"


def decision_from_probability(p_positive: float) -> tuple[bool, float]:
    """Return ``(label, decision_confidence)`` from a calibrated P(positive).

    Confidence is measured as distance from chance, so a strong negative
    prediction (``p_positive=0.1``) is just as confident as a strong positive.
    """
    p = float(p_positive)
    label = p >= 0.5
    return label, max(p, 1.0 - p)


# --------------------------------------------------------------------------- #
# Annotation policy
# --------------------------------------------------------------------------- #


@dataclass
class AnnotationPolicy:
    """Explicit, configurable rules for turning scores into weak labels."""

    gated_attributes: frozenset[str] | None = None
    bands: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BANDS))
    min_auto_accept_score: float = 0.85
    # Manual corrections: (sample_id, attribute) -> confirmed bool label.
    human_overrides: dict[tuple[str, str], bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bands = validate_bands(self.bands)
        if not (0.5 <= self.min_auto_accept_score <= 1.0):
            raise ConfigError(
                f"min_auto_accept_score must be in [0.5, 1.0], got {self.min_auto_accept_score}"
            )
        if self.gated_attributes is not None:
            unknown = set(self.gated_attributes) - CELEBA_ATTRIBUTE_SET
            if unknown:
                raise AnnotationError(f"Unknown gated attributes: {sorted(unknown)}")
            self.gated_attributes = frozenset(self.gated_attributes)

    def accepts_attribute(self, attribute: str) -> bool:
        """Whether ``attribute`` is allowed to become an automatic label."""
        if attribute not in CELEBA_ATTRIBUTE_SET:
            raise AnnotationError(f"Unknown CelebA attribute: {attribute}")
        return self.gated_attributes is None or attribute in self.gated_attributes

    def human_override(self, sample_id: str, attribute: str) -> bool | None:
        return self.human_overrides.get((sample_id, attribute))

    @classmethod
    def from_build_config(
        cls,
        build: BuildConfig,
        *,
        gated_attributes: frozenset[str] | None = None,
        human_overrides: dict[tuple[str, str], bool] | None = None,
    ) -> "AnnotationPolicy":
        bands = build.confidence_bands or DEFAULT_BANDS
        return cls(
            gated_attributes=gated_attributes,
            bands=dict(bands),
            min_auto_accept_score=build.min_auto_accept_score,
            human_overrides=human_overrides or {},
        )


# --------------------------------------------------------------------------- #
# Annotator
# --------------------------------------------------------------------------- #


class BenchmarkAnnotator:
    """Turns frozen-protocol scores into provenance-rich attribute observations."""

    def __init__(
        self,
        policy: AnnotationPolicy,
        calibrators: Mapping[str, Calibrator] | None = None,
        model_fingerprint: str | None = None,
        prompt_registry_hash: str | None = None,
    ):
        self.policy = policy
        self.calibrators = dict(calibrators or {})
        self.model_fingerprint = model_fingerprint
        self.prompt_registry_hash = prompt_registry_hash

    # -- calibration ----------------------------------------------------- #

    def calibrated_score(self, attribute: str, raw_score: float) -> float:
        """Apply the frozen per-attribute calibrator (identity if absent)."""
        calibrator = self.calibrators.get(attribute)
        if calibrator is None:
            return float(raw_score)
        value = calibrator.predict([float(raw_score)])[0]
        return float(value)

    # -- single observation --------------------------------------------- #

    def observe(
        self, attribute: str, raw_score: float, *, sample_id: str | None = None
    ) -> AttributeObservation:
        """Produce one weak-label observation for an image + attribute."""
        self.policy.accepts_attribute(attribute)
        calibrated = self.calibrated_score(attribute, raw_score)
        label, confidence = decision_from_probability(calibrated)

        override = self.policy.human_override(sample_id or "", attribute)
        if override is not None:
            # Human-verified tier: an annotator confirmed/corrected the model.
            return AttributeObservation(
                name=celeba40_key(attribute),
                label=bool(override),
                score=round(calibrated, 6),
                source="human_verified_model",
                model_fingerprint=self.model_fingerprint,
                prompt_id=self.prompt_registry_hash,
                confidence_band=confidence_band(confidence, self.policy.bands),
                attribute_class=group_of(attribute),
            ).validate()

        band = confidence_band(confidence, self.policy.bands)
        accepted = (
            self.policy.accepts_attribute(attribute)
            and confidence >= self.policy.min_auto_accept_score
            and band == "high"
        )
        if accepted:
            return AttributeObservation(
                name=celeba40_key(attribute),
                label=label,
                score=round(calibrated, 6),
                source="source_model",
                model_fingerprint=self.model_fingerprint,
                prompt_id=self.prompt_registry_hash,
                confidence_band=band,
                attribute_class=group_of(attribute),
            ).validate()

        # Unlabeled/uncertain tier: keep the score, withhold the label.
        return AttributeObservation(
            name=celeba40_key(attribute),
            label=None,
            score=round(calibrated, 6),
            source="derived",
            model_fingerprint=self.model_fingerprint,
            prompt_id=self.prompt_registry_hash,
            confidence_band=band,
            attribute_class=group_of(attribute),
        ).validate()

    # -- image / sample level ------------------------------------------- #

    def annotate_image(
        self, scores: Mapping[str, float], *, sample_id: str | None = None
    ) -> dict[str, AttributeObservation]:
        """Annotate all attributes for one image; returns namespaced obs."""
        return {
            celeba40_key(attr): self.observe(attr, raw, sample_id=sample_id)
            for attr, raw in scores.items()
        }

    def annotate_sample(
        self, sample: CanonicalSample, scores: Mapping[str, float]
    ) -> CanonicalSample:
        """Return a copy of ``sample`` with CelebA-40 observations merged in.

        Source annotations are never overwritten: only namespaced CelebA keys
        are written, and existing source_* entries are preserved untouched.
        """
        obs = dict(sample.visual_attributes)
        obs.update(self.annotate_image(scores, sample_id=sample.source_sample_id))
        return replace(sample, visual_attributes=obs)

    # -- reporting ------------------------------------------------------- #

    def caveats_for(self, attribute: str) -> list[str]:
        """Caveats that must travel with an attribute in cards/reports."""
        caveats: list[str] = []
        if is_reliability_flagged(attribute):
            caveats.append(
                f"{attribute} is a low-reliability or subjective CelebA label; "
                "it inherits the original CelebA definition and limitations."
            )
        return caveats


# --------------------------------------------------------------------------- #
# Helpers for wiring runner outputs into the annotator
# --------------------------------------------------------------------------- #


def load_frozen_calibrators(calibrators_path: str | None) -> dict[str, Calibrator]:
    """Load frozen per-attribute calibrators (empty dict when not configured)."""
    if not calibrators_path:
        return {}
    return load_calibrators(calibrators_path)


def predictions_to_scores(rows: Any) -> dict[str, dict[str, float]]:
    """Collapse CelebaRunner prediction rows into ``{sample_id: {attr: p}}``.

    Accepts an iterable of mappings or a pandas DataFrame. Only rows produced
    with candidate scoring carry ``p_positive``; rows without it are skipped so
    generation-only runs do not silently fabricate scores.
    """
    records: list[dict[str, Any]]
    if hasattr(rows, "to_dict"):
        records = rows.to_dict(orient="records")
    else:
        records = list(rows)
    out: dict[str, dict[str, float]] = {}
    for row in records:
        p = row.get("p_positive")
        if p is None:
            continue
        sample_id = str(row.get("sample_id"))
        attr = row.get("attribute")
        if not sample_id or not attr:
            continue
        out.setdefault(sample_id, {})[str(attr)] = float(p)
    return out
