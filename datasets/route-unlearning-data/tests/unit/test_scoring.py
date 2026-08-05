"""Candidate-sequence scoring utilities (plan section 8.4)."""

from __future__ import annotations

import math

import pytest
import torch

from route_data.models.scoring import (
    binary_probability,
    gather_sequence_log_probs,
    normalize_binary_scores,
)


class TestNormalizeBinaryScores:
    def test_equal_scores_are_uniform(self):
        out = normalize_binary_scores({"yes": 0.0, "no": 0.0})
        assert out["yes"] == pytest.approx(0.5)
        assert out["no"] == pytest.approx(0.5)

    def test_softmax_is_shift_invariant(self):
        out = normalize_binary_scores({"yes": 1000.0, "no": 1000.0})
        assert out["yes"] == pytest.approx(0.5)

    def test_probabilities_sum_to_one(self):
        out = normalize_binary_scores({"a": -2.0, "b": 1.0, "c": 0.5})
        assert sum(out.values()) == pytest.approx(1.0)

    def test_single_candidate_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            normalize_binary_scores({"yes": 1.0})


class TestBinaryProbability:
    def test_equal_scores(self):
        assert binary_probability(0.0, 0.0) == pytest.approx(0.5)

    def test_positive_advantage(self):
        expected = math.e / (math.e + 1.0)
        assert binary_probability(1.0, 0.0) == pytest.approx(expected)

    def test_large_scores_are_numerically_stable(self):
        assert binary_probability(1e3, 1e3) == pytest.approx(0.5)


class TestGatherSequenceLogProbs:
    def test_matches_manual_log_softmax_reference(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 7)
        targets = torch.tensor([0, 4, 6])
        expected = float(
            torch.log_softmax(logits, dim=-1)
            .gather(-1, targets.unsqueeze(-1))
            .sum()
            .item()
        )
        assert gather_sequence_log_probs(logits, targets) == pytest.approx(expected)

    def test_wrong_dims_raise(self):
        with pytest.raises(ValueError, match="Expected logits"):
            gather_sequence_log_probs(torch.zeros(3), torch.tensor([0, 1, 2]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            gather_sequence_log_probs(torch.zeros(3, 5), torch.tensor([0, 1]))

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="Empty candidate"):
            gather_sequence_log_probs(torch.zeros(0, 5), torch.zeros(0, dtype=torch.long))

    def test_target_outside_vocabulary_raises(self):
        with pytest.raises(ValueError, match="vocabulary"):
            gather_sequence_log_probs(torch.zeros(2, 5), torch.tensor([0, 7]))
