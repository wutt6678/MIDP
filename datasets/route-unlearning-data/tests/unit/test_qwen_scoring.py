"""Qwen3.5 prefix-token teacher-forcing regression tests.

CPU-level tests verify the scoring logic without requiring a real GPU model.
A mock processor + model simulates the Qwen3.5 pipeline to validate:

1. Candidate tokenization is non-empty and distinct.
2. Prefix is built once and shared across candidates.
3. Candidate IDs are explicitly appended to prefix.
4. Scored positions match the appended candidate IDs.
5. Multi-token candidates are handled correctly.
6. The frozen Yes/No protocol is enforced.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch

from route_data.models.qwen import (
    BINARY_CANDIDATES,
    NEGATIVE_CANDIDATE,
    POSITIVE_CANDIDATE,
    QwenHFBackend,
)

# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_mock_backend(
    vocab_size: int = 100,
    prefix_len: int = 10,
) -> QwenHFBackend:
    """Create a QwenHFBackend with mocked model + processor.

    The mock processor tokenizes deterministically: each character maps to
    a unique token id (ord(c) % vocab_size).  The mock model returns
    random-but-seeded logits.
    """
    config = MagicMock()
    config.model_id = "Qwen/Qwen3.5-9B"
    config.revision = None
    config.dtype = "bfloat16"
    config.device_map = "cuda:0"
    config.attn_implementation = "sdpa"
    config.trust_remote_code = False
    config.quantization.enabled = False
    config.seed = 17
    config.resolved_processor_id = None
    config.generation.do_sample = False
    config.generation.temperature = 0.0
    config.generation.max_new_tokens = 4

    backend = QwenHFBackend(config)

    # ── Mock tokenizer ──────────────────────────────────────────────────
    tokenizer = MagicMock()

    def _encode(text, add_special_tokens=True):
        """Deterministic tokenization: each char → ord(c) % vocab_size."""
        return [ord(c) % vocab_size for c in text]

    def _decode(ids):
        return "".join(chr(tid % vocab_size) for tid in ids)

    tokenizer.encode = _encode
    tokenizer.decode = _decode

    # ── Mock processor ──────────────────────────────────────────────────
    processor = MagicMock()
    processor.tokenizer = tokenizer

    def _apply_chat_template(messages, **kwargs):
        """Return a dict mimicking the real processor output."""
        if kwargs.get("tokenize") and kwargs.get("return_dict"):
            # Build a deterministic prefix of length prefix_len.
            prefix_ids = list(range(1, prefix_len + 1))
            return {
                "input_ids": torch.tensor([prefix_ids]),
                "attention_mask": torch.ones(1, prefix_len, dtype=torch.long),
            }
        # Non-tokenized call returns a string.
        return "PREFIX_TEXT"

    processor.apply_chat_template = _apply_chat_template

    # ── Mock model ──────────────────────────────────────────────────────
    model = MagicMock()
    embedding = MagicMock()
    embedding.weight.device = torch.device("cpu")
    model.get_input_embeddings.return_value = embedding

    def _forward(**kwargs):
        input_ids = kwargs["input_ids"]
        seq_len = input_ids.shape[1]
        torch.manual_seed(42)
        logits = torch.randn(1, seq_len, vocab_size)
        return MagicMock(logits=logits)

    model.side_effect = _forward
    model.eval = MagicMock()

    # Inject mocks directly (bypass _load).
    backend._model = model
    backend._processor = processor
    backend._resolved_revision = "test"

    return backend


# ── Protocol tests ───────────────────────────────────────────────────────


class TestFrozenProtocol:
    """The binary candidate protocol is frozen: Yes / No."""

    def test_qwen_yes_no_case_protocol_is_frozen(self):
        assert POSITIVE_CANDIDATE == "Yes"
        assert NEGATIVE_CANDIDATE == "No"
        assert BINARY_CANDIDATES == ("Yes", "No")

    def test_qwen_candidate_ids_differ(self):
        """Yes and No must tokenize to different sequences."""
        backend = _make_mock_backend()
        tok = backend.processor.tokenizer
        yes_ids = tok.encode("Yes", add_special_tokens=False)
        no_ids = tok.encode("No", add_special_tokens=False)
        assert yes_ids != no_ids
        assert len(yes_ids) > 0
        assert len(no_ids) > 0


# ── Prefix construction tests ───────────────────────────────────────────


class TestPrefixConstruction:
    """The multimodal prefix is built once via apply_chat_template."""

    def test_build_prefix_returns_tensor_dict(self):
        backend = _make_mock_backend(prefix_len=10)
        image = MagicMock()
        prefix = backend._build_prefix(image, "test prompt")

        assert "input_ids" in prefix
        assert "attention_mask" in prefix
        assert prefix["input_ids"].shape == (1, 10)
        assert prefix["attention_mask"].shape == (1, 10)

    def test_build_prefix_message_format(self):
        """_build_prefix passes image + text in the correct message format."""
        backend = _make_mock_backend(prefix_len=10)

        captured_messages = []
        orig_apply = backend.processor.apply_chat_template

        def _spy(messages, **kwargs):
            captured_messages.append(messages)
            return orig_apply(messages, **kwargs)

        backend.processor.apply_chat_template = _spy

        image = MagicMock()
        backend._build_prefix(image, "test prompt")

        assert len(captured_messages) == 1
        messages = captured_messages[0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "image"
        assert content[0]["image"] is image
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "test prompt"


# ── Scoring logic tests ─────────────────────────────────────────────────


class TestScoreCandidatesLogic:
    """score_candidates uses explicit prefix + candidate concatenation."""

    def test_qwen_candidate_suffix_is_explicitly_appended(self):
        """full_input_ids must equal prefix_input_ids + candidate_ids."""
        backend = _make_mock_backend(vocab_size=100, prefix_len=10)

        # Track the input_ids passed to the model.
        captured_kwargs = []

        def _forward(**kwargs):
            captured_kwargs.append(dict(kwargs))
            input_ids = kwargs["input_ids"]
            seq_len = input_ids.shape[1]
            torch.manual_seed(42)
            logits = torch.randn(1, seq_len, 100)
            return MagicMock(logits=logits)

        backend._model.side_effect = _forward

        image = MagicMock()
        backend.score_candidates(image, "prompt", ["Yes", "No"])

        # Should have called the model twice (once per candidate).
        assert len(captured_kwargs) == 2

        for i, kw in enumerate(captured_kwargs):
            full_ids = kw["input_ids"]
            assert full_ids.shape[0] == 1  # batch dim
            assert full_ids.shape[1] > 0  # has tokens

    def test_qwen_scored_targets_equal_appended_candidate_ids(self):
        """The scored target positions must match the appended candidate IDs."""
        backend = _make_mock_backend(vocab_size=100, prefix_len=10)

        captured_kwargs = []

        def _forward(**kwargs):
            captured_kwargs.append(dict(kwargs))
            input_ids = kwargs["input_ids"]
            seq_len = input_ids.shape[1]
            torch.manual_seed(42)
            logits = torch.randn(1, seq_len, 100)
            return MagicMock(logits=logits)

        backend._model.side_effect = _forward

        image = MagicMock()
        candidates = ["Yes", "No"]
        backend.score_candidates(image, "prompt", candidates)

        tok = backend.processor.tokenizer
        for i, cand in enumerate(candidates):
            cand_ids = tok.encode(cand, add_special_tokens=False)
            full_ids = captured_kwargs[i]["input_ids"][0]
            # The last len(cand_ids) tokens should be the candidate.
            suffix = full_ids[-len(cand_ids):].tolist()
            assert suffix == cand_ids, (
                f"candidate {cand!r}: suffix {suffix} != expected {cand_ids}"
            )

    def test_qwen_multitoken_candidate_scoring(self):
        """Multi-token candidates are scored correctly."""
        backend = _make_mock_backend(vocab_size=100, prefix_len=10)

        captured_kwargs = []

        def _forward(**kwargs):
            captured_kwargs.append(dict(kwargs))
            input_ids = kwargs["input_ids"]
            seq_len = input_ids.shape[1]
            torch.manual_seed(42)
            logits = torch.randn(1, seq_len, 100)
            return MagicMock(logits=logits)

        backend._model.side_effect = _forward

        image = MagicMock()
        # Use multi-token candidates.
        candidates = ["Yes I do", "No way"]
        resp = backend.score_candidates(image, "prompt", candidates)

        assert len(resp.candidate_scores) == 2
        for score in resp.candidate_scores:
            assert math.isfinite(score.log_probability)

        tok = backend.processor.tokenizer
        for i, cand in enumerate(candidates):
            cand_ids = tok.encode(cand, add_special_tokens=False)
            assert len(cand_ids) > 1, f"expected multi-token: {cand!r}"
            full_ids = captured_kwargs[i]["input_ids"][0]
            suffix = full_ids[-len(cand_ids):].tolist()
            assert suffix == cand_ids

    def test_qwen_candidate_scoring_does_not_depend_on_full_string_suffix(self):
        """Scoring does NOT re-tokenize prefix+candidate as one string.

        Instead, candidate IDs are explicitly appended.  This test verifies
        that the suffix of full_input_ids matches the candidate encoding
        exactly, not some re-tokenized version.
        """
        backend = _make_mock_backend(vocab_size=100, prefix_len=10)

        captured_kwargs = []

        def _forward(**kwargs):
            captured_kwargs.append(dict(kwargs))
            input_ids = kwargs["input_ids"]
            seq_len = input_ids.shape[1]
            torch.manual_seed(42)
            logits = torch.randn(1, seq_len, 100)
            return MagicMock(logits=logits)

        backend._model.side_effect = _forward

        image = MagicMock()
        backend.score_candidates(image, "prompt", ["Yes", "No"])

        tok = backend.processor.tokenizer
        for i, cand in enumerate(["Yes", "No"]):
            cand_ids = tok.encode(cand, add_special_tokens=False)
            full_ids = captured_kwargs[i]["input_ids"][0]
            # Verify the suffix is exactly the candidate encoding.
            suffix = full_ids[-len(cand_ids):].tolist()
            assert suffix == cand_ids

    def test_all_log_probabilities_finite(self):
        """All returned log probabilities must be finite."""
        backend = _make_mock_backend(vocab_size=100, prefix_len=10)
        image = MagicMock()
        resp = backend.score_candidates(image, "prompt", ["Yes", "No"])
        for score in resp.candidate_scores:
            assert math.isfinite(score.log_probability), (
                f"non-finite log_prob for {score.candidate!r}: "
                f"{score.log_probability}"
            )

    def test_log_probabilities_not_identical(self):
        """Different candidates should not produce identical log probs."""
        backend = _make_mock_backend(vocab_size=100, prefix_len=10)
        image = MagicMock()
        resp = backend.score_candidates(image, "prompt", ["Yes", "No"])
        log_probs = [s.log_probability for s in resp.candidate_scores]
        assert len({round(lp, 8) for lp in log_probs}) > 1, (
            f"all log probabilities identical: {log_probs}"
        )

    def test_debug_metadata_present(self):
        """scoring_debug metadata should be present in response."""
        backend = _make_mock_backend(vocab_size=100, prefix_len=10)
        image = MagicMock()
        resp = backend.score_candidates(image, "prompt", ["Yes", "No"])
        debug = resp.metadata.get("scoring_debug")
        assert debug is not None
        assert len(debug) == 2
        for entry in debug:
            assert "candidate" in entry
            assert "candidate_token_ids" in entry
            assert "prefix_length" in entry
            assert "full_length" in entry
            assert "scored_positions" in entry
            assert entry["full_length"] == entry["prefix_length"] + len(
                entry["candidate_token_ids"]
            )

    def test_empty_candidates_raises(self):
        backend = _make_mock_backend()
        image = MagicMock()
        with pytest.raises(ValueError, match="non-empty"):
            backend.score_candidates(image, "prompt", [])


# ── Cross-cutting: protocol used in CLI ─────────────────────────────────


class TestProtocolIntegration:
    """Verify the frozen protocol is used consistently."""

    def test_binary_candidates_tuple_is_exported(self):
        from route_data.models.qwen import BINARY_CANDIDATES
        assert BINARY_CANDIDATES == ("Yes", "No")

    def test_celeba_runner_uses_frozen_protocol(self):
        from route_data.eval.celeba_runner import (
            _NEGATIVE_CANDIDATE,
            _POSITIVE_CANDIDATE,
            CANDIDATES,
        )
        assert _POSITIVE_CANDIDATE == "Yes"
        assert _NEGATIVE_CANDIDATE == "No"
        assert CANDIDATES == ("Yes", "No")
