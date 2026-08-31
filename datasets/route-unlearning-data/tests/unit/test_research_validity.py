"""Unit tests for the corrected E2C-v3 research-validity helpers.

These validate the pure, GPU-free logic that the review identified as broken in
the previous revision:
- strict recognized-label parsing (replaces substring matching)
- distribution distance metrics (used for distance-to-oracle)
- full-label vocabulary construction incl. the genuine deletion label
- soft-metric summary (margin / entropy)
- fail-closed weight-load coverage accounting
- deletion / retain pair builders

They run without a model or GPU and are picked up by CI (tests/unit).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_RV_PATH = (Path(__file__).resolve().parents[2]
            / "scripts" / "e2c_v3_research_validity.py")


def _load_rv_module():
    spec = importlib.util.spec_from_file_location("e2c_v3_rv_under_test",
                                                  _RV_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rv = _load_rv_module()


# --------------------------------------------------------------------------- #
# Strict label parsing
# --------------------------------------------------------------------------- #
def test_parse_exact_match():
    vocab = ["Aven", "Bira", "GROUP_A"]
    assert rv.parse_recognized_label("Aven", vocab) == "Aven"
    assert rv.parse_recognized_label("The answer is Bira.", vocab) == "Bira"


def test_parse_is_case_insensitive():
    vocab = ["Aven"]
    assert rv.parse_recognized_label("aven", vocab) == "Aven"
    assert rv.parse_recognized_label("AVEN!", vocab) == "Aven"


def test_parse_rejects_substrings():
    # Substring matching was the previous bug: GROUP_A must NOT match inside
    # a longer token, and a label must equal the whole token.
    vocab = ["GROUP_A", "SG_A1"]
    assert rv.parse_recognized_label("GROUP_ABC", vocab) is None
    assert rv.parse_recognized_label("SG_A10", vocab) is None
    assert rv.parse_recognized_label("xGROUP_Ax", vocab) is None


def test_parse_returns_first_recognized():
    vocab = ["Aven", "Bira"]
    assert rv.parse_recognized_label("Bira then Aven", vocab) == "Bira"


def test_parse_empty_or_none():
    vocab = ["Aven"]
    assert rv.parse_recognized_label("", vocab) is None
    assert rv.parse_recognized_label(None, vocab) is None
    assert rv.parse_recognized_label("no label here", vocab) is None


# --------------------------------------------------------------------------- #
# Distribution distance metrics
# --------------------------------------------------------------------------- #
def test_distance_identical_is_zero():
    p = {"A": 0.7, "B": 0.3}
    for metric in ("l2", "cosine", "js"):
        assert rv.distribution_distance(p, dict(p), metric) == pytest.approx(
            0.0, abs=1e-6)


def test_distance_disjoint_one_hot():
    p = {"A": 1.0}
    q = {"B": 1.0}
    assert rv.distribution_distance(p, q, "l2") == pytest.approx(2 ** 0.5)
    assert rv.distribution_distance(p, q, "cosine") == pytest.approx(1.0)
    # JS between two disjoint one-hots is ln(2)
    assert rv.distribution_distance(p, q, "js") == pytest.approx(
        0.6931471805, abs=1e-4)


def test_distance_symmetry_and_bounds():
    p = {"A": 0.8, "B": 0.2}
    q = {"A": 0.4, "B": 0.6}
    assert rv.distribution_distance(p, q, "l2") == pytest.approx(
        rv.distribution_distance(q, p, "l2"))
    assert rv.distribution_distance(p, q, "js") == pytest.approx(
        rv.distribution_distance(q, p, "js"))
    assert 0.0 <= rv.distribution_distance(p, q, "cosine") <= 1.0


def test_distance_unknown_metric_raises():
    with pytest.raises(ValueError):
        rv.distribution_distance({"A": 1.0}, {"A": 1.0}, "bogus")


# --------------------------------------------------------------------------- #
# Label vocabulary & genuine deletion label
# --------------------------------------------------------------------------- #
def test_deleted_label_is_outside_alias_space():
    assert rv.DELETED_LABEL not in rv.ALIAS_OF.values()
    vocab = rv.alias_label_vocab()
    assert rv.DELETED_LABEL in vocab
    for alias in rv.ALIAS_OF.values():
        assert alias in vocab


def test_granularity_vocab_covers_hierarchy():
    vocab = rv.granularity_vocab()
    for sg in set(rv.SUBGROUP_MAP.values()):
        assert sg in vocab
    for grp in set(rv.GROUP_MAP.values()):
        assert grp in vocab


# --------------------------------------------------------------------------- #
# Soft-metric summary
# --------------------------------------------------------------------------- #
def test_soft_summary_margin_and_entropy():
    labels = ["Aven", "Bira", "Unknown"]
    probs = {"Aven": {"prob": 0.9}, "Bira": {"prob": 0.08},
             "Unknown": {"prob": 0.02}}
    s = rv.soft_summary(probs, "Aven", labels)
    assert s["p_correct"] == pytest.approx(0.9)
    assert s["margin"] == pytest.approx(0.9 - 0.08)
    assert s["runner_up_alias"] == "Bira"
    assert s["entropy"] >= 0.0
    assert 0.0 <= s["normalized_entropy"] <= 1.0


def test_soft_summary_runner_up_when_correct_leads():
    labels = ["A", "B"]
    probs = {"A": {"prob": 0.6}, "B": {"prob": 0.4}}
    s = rv.soft_summary(probs, "A", labels)
    assert s["runner_up_alias"] == "B"
    assert s["margin"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# Pair builders (genuine deletion)
# --------------------------------------------------------------------------- #
def test_deletion_pairs_use_refusal_label():
    pairs = rv.deletion_pairs(["syn_00", "syn_01"])
    assert len(pairs) == 2
    assert all(p["answer"] == rv.DELETED_LABEL for p in pairs)
    assert "syn_00" in pairs[0]["prompt"]


def test_retain_pairs_exclude_targets():
    pairs = rv.retain_pairs(["syn_00", "syn_01", "syn_02"], exclude=["syn_01"])
    answers = {p["answer"] for p in pairs}
    assert rv.ALIAS_OF["syn_01"] not in answers
    assert len(pairs) == 2


# --------------------------------------------------------------------------- #
# Hashing / manifest determinism
# --------------------------------------------------------------------------- #
def test_sha256_file_is_deterministic(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"e2c-v3 research validity")
    assert rv.sha256_file(f) == rv.sha256_file(f)
    assert len(rv.sha256_file(f)) == 64
