"""Annotation policy, confidence bands, and weak-label gating (plan 11.3)."""

from __future__ import annotations

import pytest

from route_data.build.annotate import (
    AnnotationError,
    AnnotationPolicy,
    BenchmarkAnnotator,
    confidence_band,
    decision_from_probability,
    validate_bands,
)
from route_data.config import BuildConfig, ConfigError
from route_data.eval.calibration import PlattCalibrator

BANDS = {"high": 0.85, "medium": 0.60}


class TestDecisionFromProbability:
    def test_high_probability_is_positive(self):
        label, conf = decision_from_probability(0.9)
        assert label is True
        assert conf == pytest.approx(0.9)

    def test_low_probability_is_negative(self):
        label, conf = decision_from_probability(0.1)
        assert label is False
        assert conf == pytest.approx(0.9)

    def test_tie_breaks_positive_with_half_confidence(self):
        label, conf = decision_from_probability(0.5)
        assert label is True
        assert conf == pytest.approx(0.5)


class TestConfidenceBands:
    def test_band_boundaries(self):
        assert confidence_band(0.9, BANDS) == "high"
        assert confidence_band(0.85, BANDS) == "high"
        assert confidence_band(0.7, BANDS) == "medium"
        assert confidence_band(0.6, BANDS) == "medium"
        assert confidence_band(0.55, BANDS) == "low"

    def test_invalid_bands_raise(self):
        with pytest.raises(ConfigError, match="confidence_bands"):
            validate_bands({"high": 0.5, "medium": 0.9})


class TestAnnotationPolicy:
    def test_min_auto_accept_score_bounds(self):
        with pytest.raises(ConfigError, match="min_auto_accept_score"):
            AnnotationPolicy(min_auto_accept_score=0.4)
        with pytest.raises(ConfigError, match="min_auto_accept_score"):
            AnnotationPolicy(min_auto_accept_score=1.2)

    def test_unknown_gated_attribute_rejected(self):
        with pytest.raises(AnnotationError, match="Unknown gated"):
            AnnotationPolicy(gated_attributes=frozenset({"not_a_real_attribute"}))

    def test_accepts_attribute_gating(self):
        policy = AnnotationPolicy(gated_attributes=frozenset({"Smiling"}))
        assert policy.accepts_attribute("Smiling") is True
        assert policy.accepts_attribute("Wearing_Hat") is False

    def test_unknown_attribute_raises(self):
        policy = AnnotationPolicy()
        with pytest.raises(AnnotationError, match="Unknown CelebA attribute"):
            policy.accepts_attribute("teleportation")

    def test_from_build_config_carries_values(self):
        build = BuildConfig(
            confidence_bands={"high": 0.9, "medium": 0.7},
            min_auto_accept_score=0.9,
        )
        policy = AnnotationPolicy.from_build_config(build)
        assert policy.bands == {"high": 0.9, "medium": 0.7}
        assert policy.min_auto_accept_score == pytest.approx(0.9)


class TestBenchmarkAnnotator:
    def _annotator(self, **policy_kwargs) -> BenchmarkAnnotator:
        return BenchmarkAnnotator(AnnotationPolicy(**policy_kwargs))

    def test_high_confidence_label_is_accepted(self):
        obs = self._annotator().observe("Smiling", 0.95)
        assert obs.label is True
        assert obs.source == "source_model"
        assert obs.confidence_band == "high"
        assert obs.name == "extended_attributes.celeba40.Smiling"

    def test_low_confidence_label_is_withheld(self):
        obs = self._annotator().observe("Smiling", 0.55)
        assert obs.label is None
        assert obs.source == "derived"
        assert obs.score == pytest.approx(0.55)

    def test_gated_attribute_is_not_auto_accepted(self):
        annotator = self._annotator(gated_attributes=frozenset({"Wearing_Hat"}))
        obs = annotator.observe("Smiling", 0.99)
        assert obs.label is None

    def test_human_override_wins(self):
        annotator = self._annotator(
            human_overrides={("s1", "Smiling"): False}
        )
        obs = annotator.observe("Smiling", 0.99, sample_id="s1")
        assert obs.label is False
        assert obs.source == "human_verified_model"

    def test_calibrator_applied_only_to_configured_attribute(self):
        steep = PlattCalibrator(weight=100.0, bias=0.0, n_fit=10)
        annotator = BenchmarkAnnotator(
            AnnotationPolicy(), calibrators={"Smiling": steep}
        )
        assert annotator.calibrated_score("Smiling", 1.0) > 0.99
        assert annotator.calibrated_score("Wearing_Hat", 0.2) == pytest.approx(0.2)
