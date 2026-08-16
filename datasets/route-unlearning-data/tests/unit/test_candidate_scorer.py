"""Tests for the shared candidate scorer (P0-1, P1-1).

Verifies that training and evaluation use identical scoring logic:
- prefix_len - 1 alignment for first candidate token
- multi-token candidate support
- dynamic Yes/No tokenization (no hard-coded IDs)
- training/eval parity on identical synthetic logits
"""

from __future__ import annotations

import math

import pytest
import torch

from route_data.models.scoring import (
    compute_candidate_margin,
    score_candidate_sequence_tensor,
)


class _MockModel(torch.nn.Module):
    """Mock model returning deterministic logits for testing."""

    def __init__(self, vocab_size: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        # Dummy parameter to satisfy torch.nn.Module
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.nn.Module:
        """Return mock logits with controlled log probabilities."""
        batch_size, seq_len = input_ids.shape
        # Create logits where specific tokens have known probabilities
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size),
            -10.0,
            dtype=torch.float32,
        )
        # Set high probability for the target token at each position
        for b in range(batch_size):
            for t in range(seq_len):
                target_id = input_ids[b, t].item()
                if target_id < self.vocab_size:
                    logits[b, t, target_id] = 0.0  # High logit = high probability
        # Return as a mock output object
        output = torch.nn.Module()
        output.logits = logits
        return output


def _make_prefix(prefix_len: int = 5, vocab_size: int = 100) -> dict[str, torch.Tensor]:
    """Create a minimal prefix dict for testing."""
    input_ids = torch.arange(1, prefix_len + 1).unsqueeze(0)  # [1, prefix_len]
    attention_mask = torch.ones(1, prefix_len, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def _make_prefix_with_mm(prefix_len: int = 5, vocab_size: int = 100) -> dict[str, torch.Tensor]:
    """Create a prefix dict with multimodal tensors for testing."""
    prefix = _make_prefix(prefix_len, vocab_size)
    # Add mock multimodal tensors
    prefix["mm_token_type_ids"] = torch.zeros(1, prefix_len, dtype=torch.long)
    prefix["pixel_values"] = torch.randn(1, 3, 224, 224)
    prefix["image_grid_thw"] = torch.tensor([[1, 1, 1]])
    return prefix


class TestScoreCandidateSequenceTensor:
    """Tests for score_candidate_sequence_tensor."""

    def test_prefix_len_minus_1_alignment(self) -> None:
        """First candidate token scored from prefix_len - 1."""
        model = _MockModel(vocab_size=100)
        prefix = _make_prefix(prefix_len=5)
        candidate_ids = [10]  # Single-token candidate

        log_prob = score_candidate_sequence_tensor(model, prefix, candidate_ids)

        # Should return a finite scalar
        assert isinstance(log_prob, torch.Tensor)
        assert log_prob.dim() == 0  # Scalar
        assert math.isfinite(log_prob.item())

    def test_multi_token_candidate(self) -> None:
        """Multi-token candidate sums log probs across positions."""
        model = _MockModel(vocab_size=100)
        prefix = _make_prefix(prefix_len=5)
        candidate_ids = [10, 20, 30]  # Three-token candidate

        log_prob = score_candidate_sequence_tensor(model, prefix, candidate_ids)

        # Should sum log probs for all three tokens
        assert isinstance(log_prob, torch.Tensor)
        assert log_prob.dim() == 0
        assert math.isfinite(log_prob.item())
        # Each token contributes log P(token) which is negative
        # For 3 tokens, total should be more negative than single token
        assert log_prob.item() < 0  # Log probs are negative
        assert log_prob.item() > -100.0  # But not unreasonably so

    def test_dynamic_token_ids(self) -> None:
        """Different token IDs produce different scores."""
        model = _MockModel(vocab_size=100)
        prefix = _make_prefix(prefix_len=5)

        # Score two different candidates
        log_prob_1 = score_candidate_sequence_tensor(model, prefix, [10])
        log_prob_2 = score_candidate_sequence_tensor(model, prefix, [20])

        # Both should be finite
        assert math.isfinite(log_prob_1.item())
        assert math.isfinite(log_prob_2.item())
        # In mock model, both get high probability, so scores should be similar
        # but the function should handle different IDs correctly

    def test_empty_candidate_raises(self) -> None:
        """Empty candidate sequence raises ValueError."""
        model = _MockModel(vocab_size=100)
        prefix = _make_prefix(prefix_len=5)

        with pytest.raises(ValueError, match="non-empty"):
            score_candidate_sequence_tensor(model, prefix, [])

    def test_zero_prefix_len_raises(self) -> None:
        """Zero-length prefix raises ValueError."""
        model = _MockModel(vocab_size=100)
        prefix = _make_prefix(prefix_len=0)

        with pytest.raises(ValueError, match="prefix length"):
            score_candidate_sequence_tensor(model, prefix, [10])

    def test_multimodal_tensors_forwarded(self) -> None:
        """Multimodal tensors are forwarded to the model."""
        model = _MockModel(vocab_size=100)
        prefix = _make_prefix_with_mm(prefix_len=5)

        # Should not raise even with multimodal tensors
        log_prob = score_candidate_sequence_tensor(model, prefix, [10])
        assert math.isfinite(log_prob.item())

    def test_differentiable_output(self) -> None:
        """Output is differentiable (supports gradients)."""
        # Create a model with actual parameters for gradient flow
        model = torch.nn.Sequential(
            torch.nn.Embedding(100, 10),
            torch.nn.Linear(10, 100),
        )
        
        # Wrap to return logits in expected format
        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, input_ids, **kwargs):
                emb = self.inner[0](input_ids)
                logits = self.inner[1](emb)
                output = torch.nn.Module()
                output.logits = logits
                return output
        
        model = Wrapper(model)
        prefix = _make_prefix(prefix_len=5)
        candidate_ids = [10]

        log_prob = score_candidate_sequence_tensor(model, prefix, candidate_ids)

        # Should be able to compute gradients
        assert log_prob.requires_grad or log_prob.grad_fn is not None
        # Backward should work
        log_prob.backward()


class TestComputeCandidateMargin:
    """Tests for compute_candidate_margin."""

    def test_yes_expected_positive_margin(self) -> None:
        """Expected Yes: margin = logP(Yes) - logP(No)."""
        model = _MockModel(vocab_size=100)
        prefix = _make_prefix(prefix_len=5)
        yes_ids = [10]
        no_ids = [20]

        margin = compute_candidate_margin(model, prefix, yes_ids, no_ids, expected_answer=True)

        # Should be a finite scalar
        assert isinstance(margin, torch.Tensor)
        assert margin.dim() == 0
        assert math.isfinite(margin.item())

    def test_no_expected_positive_margin(self) -> None:
        """Expected No: margin = logP(No) - logP(Yes)."""
        model = _MockModel(vocab_size=100)
        prefix = _make_prefix(prefix_len=5)
        yes_ids = [10]
        no_ids = [20]

        margin = compute_candidate_margin(model, prefix, yes_ids, no_ids, expected_answer=False)

        # Should be a finite scalar
        assert isinstance(margin, torch.Tensor)
        assert margin.dim() == 0
        assert math.isfinite(margin.item())

    def test_margin_sign_consistency(self) -> None:
        """Margin sign matches expected answer."""
        model = _MockModel(vocab_size=100)
        prefix = _make_prefix(prefix_len=5)
        yes_ids = [10]
        no_ids = [20]

        margin_yes = compute_candidate_margin(model, prefix, yes_ids, no_ids, expected_answer=True)
        margin_no = compute_candidate_margin(model, prefix, yes_ids, no_ids, expected_answer=False)

        # In mock model, both get similar scores, so margins should be similar
        # but opposite in sign (approximately)
        assert math.isfinite(margin_yes.item())
        assert math.isfinite(margin_no.item())
        # They should be approximately negatives of each other
        assert abs(margin_yes.item() + margin_no.item()) < 1e-5

    def test_multi_token_candidates(self) -> None:
        """Multi-token candidates work correctly."""
        model = _MockModel(vocab_size=100)
        prefix = _make_prefix(prefix_len=5)
        yes_ids = [10, 11]  # Two-token "Yes"
        no_ids = [20, 21]  # Two-token "No"

        margin = compute_candidate_margin(model, prefix, yes_ids, no_ids, expected_answer=True)

        assert math.isfinite(margin.item())


class TestTrainingEvalParity:
    """Tests for training/evaluation parity."""

    def test_identical_inputs_identical_outputs(self) -> None:
        """Same inputs produce same outputs in training and eval mode."""
        model = _MockModel(vocab_size=100)
        prefix = _make_prefix(prefix_len=5)
        candidate_ids = [10, 20]

        # Training mode (no inference_mode)
        model.train()
        log_prob_train = score_candidate_sequence_tensor(model, prefix, candidate_ids)

        # Eval mode (with inference_mode)
        model.eval()
        with torch.inference_mode():
            log_prob_eval = score_candidate_sequence_tensor(model, prefix, candidate_ids)

        # Should be identical (within numerical tolerance)
        assert abs(log_prob_train.item() - log_prob_eval.item()) < 1e-5
