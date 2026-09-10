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


# --------------------------------------------------------------------------- #
# Punctuation inside labels: the MLLMU broad-SOC-title bug
# --------------------------------------------------------------------------- #
BROAD_SWE = "Software and Web Developers, Programmers, and Testers"
BROAD_MUSEUM = "Archivists, Curators, and Museum Technicians"


def test_a_label_containing_commas_is_recognized_when_emitted_verbatim():
    """The regression test for a 16-GPU-hour misdiagnosis.

    ``recognized_labels_in`` stripped punctuation from the OUTPUT's tokens but
    split labels with a bare ``l.lower().split()``, so a label containing a comma
    could never match.  Both of these are the CORRECT post-edit answer for two of
    the five MLLMU pilot sets, and the model emitted them verbatim; they were
    scored unparseable, reporting 0.6 strict accuracy over 30 target rows that
    were every one correct.
    """
    vocab = [BROAD_SWE, BROAD_MUSEUM, "Software Developer", "Unknown"]
    assert rv.parse_recognized_label(BROAD_SWE, vocab) == BROAD_SWE
    assert rv.parse_recognized_label(BROAD_MUSEUM, vocab) == BROAD_MUSEUM
    # and in the trailing-period form a model naturally produces
    assert rv.parse_recognized_label(BROAD_SWE + ".", vocab) == BROAD_SWE
    assert rv.parse_recognized_label(BROAD_SWE.lower(), vocab) == BROAD_SWE


def test_a_broad_title_does_not_also_report_its_nested_detailed_label():
    """Fixing the comma must not create a multi-label false positive.

    'Software Developer' is a subsequence of the broad title's words.  If both
    were returned, ``parse_recognized_label`` would reject the row as ambiguous
    and the fix would trade one wrong verdict for another.
    """
    vocab = [BROAD_SWE, "Software Developer", "Unknown"]
    assert rv.recognized_labels_in(BROAD_SWE, vocab) == [BROAD_SWE]
    assert rv.parse_recognized_label(BROAD_SWE, vocab) == BROAD_SWE
    # the detailed label on its own still resolves to itself
    assert rv.parse_recognized_label("Software Developer", vocab) == \
        "Software Developer"


def test_a_punctuation_only_label_cannot_swallow_the_output():
    """A label that cleans to an empty span would match at every position, so
    degenerate vocab entries are dropped rather than allowed to match.

    The trigger needs punctuation on BOTH sides: a degenerate label can only
    match a text token that also cleans to empty, so an output containing a bare
    punctuation token is what makes the guard observable.  Asserting only on a
    clean output passes with the guard deleted.
    """
    vocab = [",", "Aven"]
    assert rv.recognized_labels_in("Aven", vocab) == ["Aven"]
    assert rv.parse_recognized_label("Aven", vocab) == "Aven"
    # the output has a punctuation-only token; the degenerate label must not
    # claim it, or two distinct labels are recognized and the row is rejected
    assert rv.recognized_labels_in("Aven , Aven", vocab) == ["Aven"]
    assert rv.parse_recognized_label("Aven , Aven", vocab) == "Aven"
    assert rv.recognized_labels_in("nothing recognizable here", vocab) == []


def _pre_fix_recognized_labels_in(text, vocab):
    """The matcher exactly as it was BEFORE the fix: ``clean`` applied to the
    text's tokens, labels split with a bare ``l.lower().split()``.  Reproduced
    here only to prove the invariant below detects it -- a guard that cannot
    fire is worse than no guard, because it reads as coverage."""
    def clean(t):
        return t.strip().strip(".,!?;:'\"()[]{}").lower()

    tokens = [clean(t) for t in text.strip().split()]
    spans = sorted(((tuple(lab.lower().split()), lab) for lab in vocab),
                   key=lambda x: (-len(x[0]), x[1]))
    out, seen, i = [], set(), 0
    while i < len(tokens):
        for span, lab in spans:
            n = len(span)
            if tokens[i:i + n] == list(span):
                if lab not in seen:
                    seen.add(lab)
                    out.append(lab)
                i += n
                break
        else:
            i += 1
    return out


def test_check_vocab_parseable_flags_the_historical_asymmetry(monkeypatch):
    """``check_vocab_parseable`` must catch the bug it was written for.

    Restoring the pre-fix matcher and re-running the check has to name both
    comma-bearing titles.  Without this the invariant is only ever observed
    returning an empty list, which is indistinguishable from a check that does
    nothing.
    """
    vocab = [BROAD_SWE, BROAD_MUSEUM, "Software Developer", "Unknown"]
    assert rv.check_vocab_parseable(vocab) == []

    monkeypatch.setattr(rv, "recognized_labels_in",
                        _pre_fix_recognized_labels_in)
    flagged = rv.check_vocab_parseable(vocab)
    assert sorted(flagged) == sorted([BROAD_SWE, BROAD_MUSEUM]), (
        "the invariant did not detect the punctuation asymmetry it exists to "
        f"detect; flagged={flagged}")
    # punctuation-free labels were unaffected even under the broken matcher,
    # which is why SALMU and CelebA-numeric never surfaced this
    assert "Software Developer" not in flagged
    assert "Unknown" not in flagged


def _frozen_vocabularies():
    """The three committed label vocabularies, read from their frozen artifacts
    and built the way ``dataset_ctx`` builds each one."""
    import json
    root = Path(__file__).resolve().parents[2]
    gdir = root / "e2c_granularity" / "manifests"
    salmu = json.loads((gdir / "matrix_salmu.json").read_text())["vocab"]
    mllmu = json.loads((root / "e2c_mllmu" / "manifests"
                        / "matrix_mllmu.json").read_text())["vocab"]
    nm = json.loads((gdir / "numeric_manifest.json").read_text())
    cel = json.loads((gdir / "matrix_celeba_numeric.json").read_text())
    celeba = sorted(set(nm["alias_of"].values())
                    | {a["target"] for e in cel["sets"]
                       for a in e["assignments"].values()} | {"Unknown"})
    return {"salmu": salmu, "celeba_numeric": celeba, "mllmu": mllmu}


def test_every_label_in_every_frozen_vocabulary_round_trips():
    """The design-time half, checked against the committed artifacts.

    A label the parser cannot recognize makes any set targeting it unpassable by
    construction.  Asserted for all three frozen vocabularies so a future label
    with punctuation in it fails here, on CPU, instead of after the oracles have
    been trained.
    """
    vocabs = _frozen_vocabularies()
    for ds, vocab in sorted(vocabs.items()):
        bad = rv.check_vocab_parseable(vocab)
        assert bad == [], (
            f"{ds}: these vocab labels cannot be recognized when emitted "
            f"verbatim, so any set targeting them is unpassable by "
            f"construction: {bad}")
    # the MLLMU vocabulary is the one that actually contains commas
    assert any("," in lab for lab in vocabs["mllmu"])


def test_the_punctuation_fix_is_a_no_op_on_punctuation_free_labels():
    """Why repairing MLLMU does not re-score two already-executed datasets.

    SALMU and CelebA-numeric matrices are frozen, executed and committed; their
    verdicts are evidence.  Both vocabularies are punctuation-free, and for those
    the pre-fix and post-fix matchers must agree on EVERY output -- otherwise
    this fix would silently move results nobody re-ran.  Asserted over the real
    frozen vocabularies and a family of output shapes, not a hand-picked example.
    """
    punct = ".,!?;:'\"()[]{}"
    for ds, vocab in sorted(_frozen_vocabularies().items()):
        contaminated = [lab for lab in vocab
                        if any(ch in punct for ch in lab)]
        if ds == "mllmu":
            # the dataset the fix was FOR is the one with punctuation in it
            assert contaminated, "expected comma-bearing broad SOC titles"
            continue
        assert contaminated == [], (
            f"{ds} gained punctuation-bearing labels, so the no-op argument no "
            f"longer covers it and its committed verdicts must be re-checked: "
            f"{contaminated}")
        for lab in vocab:
            for text in (lab, lab.lower(), lab.upper(), lab + ".",
                         f"the answer is {lab}.", f"{lab}, certainly"):
                assert rv.recognized_labels_in(text, vocab) == \
                    _pre_fix_recognized_labels_in(text, vocab), (
                        f"{ds}: matcher changed on {text!r}")


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


# --------------------------------------------------------------------------- #
# save_unlearning_adapter: stale checkpoints must be REPLACED, never kept
# --------------------------------------------------------------------------- #
def test_save_unlearning_adapter_replaces_stale_checkpoint(tmp_path):
    """Regression: the flatten step used `if not dest.exists()`, so re-running
    a pipeline into an existing output dir silently discarded the fresh
    checkpoint (rmtree deleted it) and later phases evaluated STALE weights."""
    from route_data.models.trainable.base import TrainableVLMAdapter

    out = tmp_path / "adapter_final"
    out.mkdir()
    # stale top-level checkpoint from a "previous run"
    (out / "adapter_model.safetensors").write_bytes(b"STALE-WEIGHTS")
    (out / "adapter_config.json").write_text('{"stale": true}')
    # fresh save that PEFT placed into an adapter-named subdirectory
    sub = out / "e2c_test_adapter"
    sub.mkdir()
    (sub / "adapter_config.json").write_text('{"fresh": true}')
    (sub / "adapter_model.safetensors").write_bytes(b"FRESH-WEIGHTS")

    class _FakeModel:
        def save_pretrained(self, _dir):  # no-op: files pre-created above
            pass

    # the method does not use `self`; call it unbound
    meta = TrainableVLMAdapter.save_unlearning_adapter(
        None, _FakeModel(), out)

    assert (out / "adapter_model.safetensors").read_bytes() == b"FRESH-WEIGHTS"
    assert not sub.exists()  # subdirectory flattened away
    assert meta["checkpoint_sha256"]
    names = {f["name"] for f in meta["files"]}
    assert "adapter_model.safetensors" in names


def test_save_unlearning_adapter_fails_closed_without_weights(tmp_path):
    from route_data.models.trainable.base import TrainableVLMAdapter

    out = tmp_path / "empty_final"

    class _FakeModel:
        def save_pretrained(self, _dir):  # saves nothing at all
            pass

    with pytest.raises(RuntimeError, match="no checkpoint weight file"):
        TrainableVLMAdapter.save_unlearning_adapter(None, _FakeModel(), out)

