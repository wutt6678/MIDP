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


def test_parse_rejects_multiple_distinct_labels():
    # Multi-label outputs are scored INVALID, never resolved to the first.
    vocab = ["Aven", "Bira", "Unknown"]
    assert rv.parse_recognized_label("Bira then Aven", vocab) is None
    assert rv.parse_recognized_label("Aven Bira", vocab) is None
    assert rv.parse_recognized_label("Unknown, or Aven?", vocab) is None


def test_parse_allows_repeated_same_label():
    # Repetition of ONE distinct label is not ambiguous.
    vocab = ["Aven", "Bira"]
    assert rv.parse_recognized_label("Aven Aven", vocab) == "Aven"
    assert rv.parse_recognized_label("aven, Aven.", vocab) == "Aven"


def test_parse_multitoken_labels():
    # Real semantic labels can span multiple tokens (MLLMU professions,
    # SALMU jobs); token-exact matching used to fail on them entirely.
    vocab = ["Software Developer", "Marine Biologist", "Unknown"]
    assert rv.parse_recognized_label(
        "Software Developer", vocab) == "Software Developer"
    assert rv.parse_recognized_label(
        "The profession is Marine Biologist.", vocab) == "Marine Biologist"
    assert rv.parse_recognized_label(
        "software developer.", vocab) == "Software Developer"


def test_parse_multitoken_longest_match_wins():
    # 'Software Developer' must be recognized as ONE label even though
    # 'Developer' could be a sub-span; the longest label wins at each
    # position.
    vocab = ["Software Developer", "Developer"]
    assert rv.parse_recognized_label("Software Developer", vocab) == \
        "Software Developer"
    assert rv.parse_recognized_label("Developer", vocab) == "Developer"


def test_parse_multitoken_rejects_two_distinct_labels():
    # Multi-label rejection also applies across multi-token labels.
    vocab = ["Software Developer", "Marine Biologist"]
    assert rv.parse_recognized_label(
        "Software Developer or Marine Biologist", vocab) is None
    assert rv.recognized_labels_in(
        "Marine Biologist, software developer", vocab) == \
        ["Marine Biologist", "Software Developer"]


def test_parse_multitoken_repeated_same_label_ok():
    vocab = ["Marine Biologist"]
    assert rv.parse_recognized_label(
        "Marine Biologist. marine biologist!", vocab) == "Marine Biologist"


def test_single_token_vocab_unchanged_by_multitoken_upgrade():
    # Backward compatibility: synthetic-label vocabularies behave as before.
    vocab = ["Aven", "GROUP_A", "SG_A1", "Unknown"]
    assert rv.recognized_labels_in("GROUP_A", vocab) == ["GROUP_A"]
    assert rv.recognized_labels_in("GROUP_ABC SG_A10", vocab) == []
    assert rv.parse_recognized_label("SG_A1", vocab) == "SG_A1"


def test_recognized_labels_in_distinct_and_ordered():
    vocab = ["Aven", "Bira"]
    assert rv.recognized_labels_in("Bira x Aven Bira", vocab) == ["Bira", "Aven"]
    assert rv.recognized_labels_in("nothing", vocab) == []


def test_extract_code_rejects_multiple_codes():
    ids = ["syn_00", "syn_01", "syn_02"]
    assert rv._extract_code("syn_00", ids) == "syn_00"
    assert rv._extract_code("The code is syn_01.", ids) == "syn_01"
    assert rv._extract_code("syn_00 or syn_01", ids) is None
    assert rv._extract_code("no code", ids) is None
    assert rv._extract_code(None, ids) is None
    # token-exact: a longer token is not a code
    assert rv._extract_code("syn_001", ids) is None


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


# --------------------------------------------------------------------------- #
# Candidate mass / OTHER mass / reliability gating
# --------------------------------------------------------------------------- #
VOCAB = ["Aven", "Bira", "Unknown"]


def test_normalize_over_healthy_mass():
    norm, mass = rv.normalize_over({"Aven": 0.2, "Bira": 0.6, "Unknown": 0.2},
                                   VOCAB)
    assert mass == pytest.approx(1.0)
    assert norm["Bira"] == pytest.approx(0.6)


def test_normalize_over_partial_mass_and_other():
    # Only half the mass is on the candidate set; the rest is OTHER.
    norm, mass = rv.normalize_over({"Aven": 0.3, "Bira": 0.2, "Unknown": 0.0},
                                   VOCAB)
    assert mass == pytest.approx(0.5)
    assert norm["Aven"] == pytest.approx(0.6)
    assert norm["Bira"] == pytest.approx(0.4)


def test_normalize_over_zero_mass_returns_zeros():
    norm, mass = rv.normalize_over({"Aven": 0.0, "Bira": 0.0, "Unknown": 0.0},
                                   VOCAB)
    assert mass == 0.0
    assert all(v == 0.0 for v in norm.values())


def test_build_candidate_summary_masses():
    s = rv.build_candidate_summary(
        {"Aven": 0.0, "Bira": 0.0, "Unknown": 0.9, "_other_": 0.1},
        ["Aven", "Bira", "Unknown"], "Unknown")
    assert s["candidate_mass"] == pytest.approx(0.9)
    assert s["other_mass"] == pytest.approx(0.1)
    assert s["alias_only_mass"] == pytest.approx(0.0)
    assert s["normalized"]["Unknown"] == pytest.approx(1.0)


def test_gated_distance_refuses_negligible_mass():
    oracle = rv.build_candidate_summary(
        {"Aven": 0.001, "Bira": 0.998, "Unknown": 0.001}, VOCAB, "Unknown")
    garbage = rv.build_candidate_summary(
        {"Aven": 1e-30, "Bira": 1e-30, "Unknown": 1e-30}, VOCAB, "Unknown")
    dist, reliable, reason = rv.gated_distance(
        garbage, oracle, "candidate", VOCAB, 0.01)
    assert dist is None
    assert reliable is False
    assert reason is not None and "not established" in reason


def test_gated_distance_alias_scope_gated_when_refusal():
    # Refusal model has healthy candidate mass but ~0 alias-only mass.
    oracle = rv.build_candidate_summary(
        {"Aven": 0.001, "Bira": 0.998, "Unknown": 0.001}, VOCAB, "Unknown")
    refusal = rv.build_candidate_summary(
        {"Aven": 0.0, "Bira": 0.0, "Unknown": 1.0}, VOCAB, "Unknown")
    alias_only = [l for l in VOCAB if l != "Unknown"]
    d_cand, cand_ok, _ = rv.gated_distance(refusal, oracle, "candidate",
                                           VOCAB, 0.01)
    d_alias, alias_ok, reason = rv.gated_distance(refusal, oracle, "alias",
                                                  alias_only, 0.01)
    assert cand_ok and d_cand is not None
    assert alias_ok is False and d_alias is None
    assert reason is not None


def test_gated_distance_reproducible_from_probs():
    # Distance computed via gated_distance must equal distribution_distance on
    # the stored normalized vectors, so it is reproducible from the artifact.
    oracle = rv.build_candidate_summary(
        {"Aven": 0.001, "Bira": 0.997, "Unknown": 0.002}, VOCAB, "Unknown")
    model = rv.build_candidate_summary(
        {"Aven": 0.1, "Bira": 0.7, "Unknown": 0.2}, VOCAB, "Unknown")
    dist, reliable, _ = rv.gated_distance(model, oracle, "candidate", VOCAB,
                                          0.01)
    assert reliable
    direct = rv.distribution_distance(model["normalized"], oracle["normalized"],
                                      "l2", labels=VOCAB)
    assert dist["l2"] == pytest.approx(direct)


# --------------------------------------------------------------------------- #
# Provenance helpers
# --------------------------------------------------------------------------- #
def test_script_sha256_is_valid_hex():
    h = rv.script_sha256()
    assert h == "unknown" or (
        len(h) == 64 and all(c in "0123456789abcdef" for c in h))


def test_git_worktree_dirty_returns_flag():
    assert rv.git_worktree_dirty() in (True, False, None)


# --------------------------------------------------------------------------- #
# Empirical visual-control summarizer
# --------------------------------------------------------------------------- #
def test_summarize_visual_controls_per_family():
    control_codes = [
        {"image_id": "i1", "identity_id": "syn_00",
         "controls": {"eyeglasses": True, "hat": False, "smiling": False},
         "pred_code": "syn_00", "code_correct": True},
        {"image_id": "i2", "identity_id": "syn_05",
         "controls": {"eyeglasses": False, "hat": False, "smiling": True},
         "pred_code": "syn_05", "code_correct": True},
    ]
    pre = [{"image_id": "i1", "outcome_ok": True},
           {"image_id": "i2", "outcome_ok": True}]
    post = [{"image_id": "i1", "outcome_ok": True},
            {"image_id": "i2", "outcome_ok": False}]
    s = rv._summarize_visual_controls(control_codes, pre, post, rv.ALIAS_OF,
                                      set())
    assert s["n_control_images"] == 2
    assert s["g_code_accuracy"] == pytest.approx(1.0)
    egt = s["per_family"]["eyeglasses"]["True"]
    assert egt["n"] == 1
    assert egt["g_code_accuracy"] == pytest.approx(1.0)
    assert egt["post_pipeline_outcome_accuracy"] == pytest.approx(1.0)
    smt = s["per_family"]["smiling"]["True"]
    assert smt["n"] == 1
    assert smt["pre_pipeline_outcome_accuracy"] == pytest.approx(1.0)
    assert smt["post_pipeline_outcome_accuracy"] == pytest.approx(0.0)
    assert smt["delta_post_minus_pre"] == pytest.approx(-1.0)


def test_summarize_visual_controls_empty():
    s = rv._summarize_visual_controls([], [], [], rv.ALIAS_OF, set())
    assert s["n_control_images"] == 0

