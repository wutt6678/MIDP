"""Deterministic stub backend tests (plan section 6.2, Phase 0)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from route_data.config import ModelConfig
from route_data.models.registry import available_backends, create_backend, ensure_backends_loaded
from route_data.models.stub import StubVisionModel


def _config(seed: int = 17) -> ModelConfig:
    return ModelConfig(backend="stub", model_id="local/stub-vlm-v1", device_map=None, seed=seed)


def _image():
    return SimpleNamespace(size=(96, 96), mode="RGB")


class TestStubDeterminism:
    def test_score_candidates_is_bit_reproducible(self):
        a = StubVisionModel(_config())
        b = StubVisionModel(_config())
        prompt = "Is this person wearing glasses? Answer yes or no."
        ra = a.score_candidates(_image(), prompt, [" yes", " no"])
        rb = b.score_candidates(_image(), prompt, [" yes", " no"])
        assert [cs.log_probability for cs in ra.candidate_scores] == [
            cs.log_probability for cs in rb.candidate_scores
        ]

    def test_generate_is_reproducible_and_binary(self):
        model = StubVisionModel(_config())
        prompt = "Is this person smiling? Answer yes or no."
        first = model.generate(_image(), prompt).text
        assert first in {"yes", "no"}
        assert model.generate(_image(), prompt).text == first

    def test_different_seed_changes_scores(self):
        prompt = "Is this person wearing a hat? Answer yes or no."
        a = StubVisionModel(_config(seed=17)).score_candidates(_image(), prompt, [" yes", " no"])
        b = StubVisionModel(_config(seed=99)).score_candidates(_image(), prompt, [" yes", " no"])
        assert [cs.log_probability for cs in a.candidate_scores] != [
            cs.log_probability for cs in b.candidate_scores
        ]


class TestCandidateDistribution:
    @pytest.mark.parametrize("candidates", [[" yes", " no"], ["a", "b", "c"]])
    def test_candidate_probabilities_sum_to_one(self, candidates):
        model = StubVisionModel(_config())
        resp = model.score_candidates(_image(), "prompt", candidates)
        probs = {
            cs.candidate: math.exp(cs.log_probability) for cs in resp.candidate_scores
        }
        assert set(probs) == set(candidates)
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)

    def test_scores_spread_across_confidence_bands(self):
        """Over many queries the stub must reach the high band (>=0.85)."""
        model = StubVisionModel(_config())
        best = 0.0
        for i in range(200):
            resp = model.score_candidates(_image(), f"prompt {i}", [" yes", " no"])
            probs = {
                cs.candidate: math.exp(cs.log_probability) for cs in resp.candidate_scores
            }
            best = max(best, max(probs.values()))
        assert best >= 0.85


class TestFingerprint:
    def test_fingerprint_stable_across_instances(self):
        fp_a = StubVisionModel(_config()).fingerprint()
        fp_b = StubVisionModel(_config()).fingerprint()
        assert fp_a == fp_b
        assert len(fp_a["fingerprint_id"]) == 16

    def test_fingerprint_changes_with_seed(self):
        fp_a = StubVisionModel(_config(seed=17)).fingerprint()
        fp_b = StubVisionModel(_config(seed=18)).fingerprint()
        assert fp_a["fingerprint_id"] != fp_b["fingerprint_id"]


class TestRegistry:
    def test_stub_and_example_vlm_registered(self):
        names = available_backends()
        assert "stub" in names
        assert "example_vlm" in names

    def test_create_backend_dispatches_to_stub(self):
        ensure_backends_loaded()
        backend = create_backend(_config())
        assert isinstance(backend, StubVisionModel)
