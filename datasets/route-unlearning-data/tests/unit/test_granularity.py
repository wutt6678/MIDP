"""CPU tests for the E2C-v3 granularity phase (G0/G1 plan iterations).

Covers: frozen numeric schema functions, taxonomic DAG + depth-error
classification, boundary tags, vocab collision audit, assignment
validation, deterministic matrix builders (SALMU 63 / numeric 84 cells),
frozen pass criteria, and the transformation-aware hard-eval engine with a
STUB model backend (plan G1: "Run fixture tests with a stub model").
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gx = _load("gx_lib_under_test", "e2c_v3_granularity.py")
gxm = _load("gx_runner_under_test", "e2c_v3_granularity_matrix.py")


# ------------------------------------------------------------------ #
# frozen numeric schema
# ------------------------------------------------------------------ #
def test_numeric_bins_and_rounding_frozen_policy():
    fs = gx.NUMERIC_SCHEMA["years_experience"]
    assert gx.narrow_interval(fs, 37) == (35, 39)
    assert gx.broad_interval(fs, 37) == (30, 39)
    assert gx.round_value(fs, 12) == 10 and gx.round_value(fs, 13) == 15
    fa = gx.NUMERIC_SCHEMA["activity_count"]
    assert gx.narrow_interval(fa, 126) == (120, 129)
    assert gx.broad_interval(fa, 126) == (100, 199)
    assert gx.round_value(fa, 125) == 130  # ties-up, frozen
    assert gx.round_value(fa, 126) == 130
    assert gx.category_of(fs, 14) == "entry-level"
    assert gx.category_of(fs, 37) == "experienced"


def test_numeric_label_format_and_parse_roundtrip():
    fs = gx.NUMERIC_SCHEMA["years_experience"]
    lab = gx.fmt_label(fs, "narrow", interval=(35, 39))
    assert lab == "35–39 years"
    p = gx.parse_numeric_label(lab, gx.NUMERIC_SCHEMA)
    assert p == {"kind": "bin", "width_class": "narrow",
                 "field": "years_experience", "lo": 35, "hi": 39}
    pe = gx.parse_numeric_label("19 years", gx.NUMERIC_SCHEMA)
    assert pe == {"kind": "exact", "field": "years_experience", "value": 19}
    assert gx.parse_numeric_label("Unknown", gx.NUMERIC_SCHEMA) is None
    assert gx.parse_numeric_label("Oden", gx.NUMERIC_SCHEMA) is None


def test_numeric_transformation_validity():
    fs = gx.NUMERIC_SCHEMA["years_experience"]
    ok, _ = gx.valid_numeric_transformation(
        "exact_to_narrow", fs, 19, "19 years", "15–19 years")
    assert ok
    ok, _ = gx.valid_numeric_transformation(
        "exact_to_broad", fs, 20, "20 years", "20–29 years")
    assert ok
    ok, why = gx.valid_numeric_transformation(
        "exact_to_narrow", fs, 19, "19 years", "20–24 years")
    assert not ok and "frozen-policy" in why
    ok, _ = gx.valid_numeric_transformation(
        "narrow_to_broad", fs, 34, "30–34 years", "30–39 years")
    assert ok
    ok, why = gx.valid_numeric_transformation(
        "narrow_to_broad", fs, 34, "35–39 years", "30–39 years")
    assert not ok and "source is not the narrow bin" in why


def test_classify_numeric_depth_and_secondary_metrics():
    schema = gx.NUMERIC_SCHEMA
    fs = schema["years_experience"]
    # correct
    r = gx.classify_numeric("15–19 years", "15–19 years", fs, 19, schema)
    assert r["classification"] == "correct" and r["contains_exact_value"]
    # over-abstraction: broad bin for a narrow request
    r = gx.classify_numeric("10–19 years", "15–19 years", fs, 19, schema)
    assert r["classification"] == "over_abstraction"
    assert r["contains_exact_value"] and r["abstraction_width_error"] == 5
    assert r["adjacent_bin_error"] is False
    # under-abstraction: exact value for a narrow request
    r = gx.classify_numeric("19 years", "15–19 years", fs, 19, schema)
    assert r["classification"] == "under_abstraction"
    # adjacent-bin error at the same resolution
    r = gx.classify_numeric("20–24 years", "15–19 years", fs, 19, schema)
    assert r["classification"] == "wrong_branch"
    assert r["adjacent_bin_error"] is True
    # wrong field is wrong_branch
    r = gx.classify_numeric("120–129 activities", "15–19 years", fs, 19,
                            schema)
    assert r["classification"] == "wrong_branch"


def test_boundary_tags():
    fs = gx.NUMERIC_SCHEMA["years_experience"]
    vals = {14, 15, 19, 20, 38}
    assert "narrow_lower" in gx.boundary_tags(fs, 15, vals)
    assert "narrow_upper" in gx.boundary_tags(fs, 19, vals)
    assert "broad_upper" in gx.boundary_tags(fs, 19, vals)
    assert "adjacent_across_broad_boundary" in gx.boundary_tags(fs, 19, vals)
    assert "adjacent_across_broad_boundary" in gx.boundary_tags(fs, 20, vals)
    assert "narrow_interior" in gx.boundary_tags(fs, 17, vals | {17})
    assert "sparse_region" in gx.boundary_tags(fs, 38, vals)


# ------------------------------------------------------------------ #
# taxonomy DAG + classification
# ------------------------------------------------------------------ #
_HIER = {
    "i1": ["fashion/clothing designer", "design professional", "media"],
    "i2": ["textile designer", "design professional", "media"],
    "i3": ["forensic psychologist", "therapy professional", "healthcare"],
}


def test_label_dag_and_classification():
    dag = gx.build_label_dag(_HIER)
    assert gx.classify_taxonomic("design professional",
                                 "design professional", dag) == "correct"
    # expected level1, output level2 -> over-abstraction
    assert gx.classify_taxonomic("media", "design professional",
                                 dag) == "over_abstraction"
    # expected level1, output the original job -> under-abstraction
    assert gx.classify_taxonomic("fashion/clothing designer",
                                 "design professional",
                                 dag) == "under_abstraction"
    # expected level1 design, output therapy (other branch) -> wrong
    assert gx.classify_taxonomic("therapy professional",
                                 "design professional",
                                 dag) == "wrong_branch"
    assert gx.classify_taxonomic("Unknown", "design professional",
                                 dag) == "refusal"
    assert gx.classify_taxonomic(None, "design professional",
                                 dag) == "unparseable"


def test_label_dag_rejects_contradiction():
    bad = {"a": ["x", "y", "z"], "b": ["y", "x", "w"]}  # x<y and y<x
    with pytest.raises(ValueError):
        gx.build_label_dag(bad)


def test_sibling_controls():
    ctl = gx.sibling_controls("i1", _HIER, targets={"i1"})
    assert ctl["sibling"] == ["i2"]
    assert ctl["cousin"] == []
    assert ctl["unrelated"] == ["i3"]
    # co-targeted sibling disappears from controls
    ctl2 = gx.sibling_controls("i1", _HIER, targets={"i1", "i2"})
    assert ctl2["sibling"] == []


# ------------------------------------------------------------------ #
# vocab collisions
# ------------------------------------------------------------------ #
def test_vocab_collision_audit():
    col = gx.check_vocab_collisions(
        ["media", "media professional", "Media", "Oden"])
    assert ("media", "Media") in col["hard_collisions"] or \
           ("Media", "media") in col["hard_collisions"]
    assert ("media", "media professional") in \
        col["nested_longest_match_wins"]


# ------------------------------------------------------------------ #
# deterministic matrix builders
# ------------------------------------------------------------------ #
def test_salmu_matrix_deterministic_and_valid():
    with open(_ROOT / "e2c_salmu" / "manifests" / "salmu_manifest.json") as f:
        sm = json.load(f)
    m1, ctx1 = gx.build_salmu_matrix(sm)
    m2, _ = gx.build_salmu_matrix(sm)
    assert m1 == m2
    assert m1["n_sets"] == 21 and m1["n_cells"] == 63
    modes = [e["mode"] for e in m1["sets"]]
    assert modes.count("single_level1") == 6
    assert modes.count("single_level2") == 6
    assert modes.count("simultaneous_same_depth_l1") == 3
    assert modes.count("simultaneous_same_depth_l2") == 3
    assert modes.count("simultaneous_mixed_depth") == 3
    # groups are disjoint (no reuse between L1 and L2 single groups)
    A, B = m1["groups"]["A_level1"], m1["groups"]["B_level2"]
    assert set(A).isdisjoint(B) and len(A) == len(B) == 6
    for entry in m1["sets"]:
        assert gx.validate_set(entry, ctx1) == []
    # mixed sets contain exactly one refusal control
    for entry in m1["sets"]:
        if entry["mode"] == "simultaneous_mixed_depth":
            ops = [a["operation"] for a in entry["assignments"].values()]
            assert sorted(ops) == ["refusal", "taxonomic", "taxonomic"]
            depths = sorted(a["target_depth"] for a in
                            entry["assignments"].values()
                            if a["operation"] == "taxonomic")
            assert depths == [1, 2]


def test_numeric_matrix_frozen_and_valid():
    nm = gx.build_numeric_manifest()
    assert len(nm["identity_ids"]) == 24
    assert gx.validate_numeric_boundary_coverage(nm) == []
    m1 = gx.build_numeric_matrix(nm)
    m2 = gx.build_numeric_matrix(gx.build_numeric_manifest())
    assert m1 == m2
    assert m1["n_sets"] == 28 and m1["n_cells"] == 84
    modes = [e["mode"] for e in m1["sets"]]
    assert modes.count("single_exact_to_narrow") == 8
    assert modes.count("single_exact_to_broad") == 8
    assert modes.count("single_exact_to_rounded") == 6
    assert modes.count("simultaneous_same_resolution") == 3
    assert modes.count("simultaneous_mixed_resolution") == 3
    vocab = sorted(set(nm["alias_of"].values())
                   | {a["target"] for e in m1["sets"]
                      for a in e["assignments"].values()} | {"Unknown"})
    ctx = {"kind": "numeric", "identity_ids": nm["identity_ids"],
           "baseline_alias_of": nm["alias_of"], "schema": nm["schema"],
           "vocab": vocab}
    for entry in m1["sets"]:
        assert gx.validate_set(entry, ctx) == []
    issues, col = gx.validate_vocab(vocab)
    assert issues == [] and col["hard_collisions"] == []


# ------------------------------------------------------------------ #
# frozen pass criteria + stub-backend engine fixture (plan G1)
# ------------------------------------------------------------------ #
class _StubBackend:
    def __init__(self, responses):
        self.responses = responses  # prompt-substring -> text

    def generate(self, _img, prompt, max_new_tokens=12):
        for key, text in self.responses.items():
            if key in prompt:
                return argparse.Namespace(text=text)
        return argparse.Namespace(text="garbage output")


class _StubModel:
    def eval(self):
        return self


class _StubSession:
    def __init__(self, responses):
        self._backend = _StubBackend(responses)
        self.model = _StubModel()

    def backend(self):
        return self._backend


def _numeric_ctx_and_entry():
    nm = gx.build_numeric_manifest()
    mat = gx.build_numeric_matrix(nm)
    entry = next(e for e in mat["sets"]
                 if e["set_id"] == "gx_num_s_exact_to_narrow_Y02")
    vocab = sorted(set(nm["alias_of"].values())
                   | {a["target"] for e in mat["sets"]
                      for a in e["assignments"].values()} | {"Unknown"})
    ctx = {"kind": "numeric", "identity_ids": nm["identity_ids"],
           "code_of": nm["code_of"], "baseline_alias_of": nm["alias_of"],
           "profiles": nm["profiles"], "schema": nm["schema"],
           "vocab": vocab}
    return ctx, entry


def test_hard_eval_stub_perfect_cell():
    ctx, entry = _numeric_ctx_and_entry()
    responses = {}
    for iid in ctx["identity_ids"]:
        exp = gxm.expected_label(ctx, entry, iid)
        responses[ctx["code_of"][iid]] = f"{exp}."
    args = argparse.Namespace(max_gen_tokens=12)
    hard = gxm._hard_eval(_StubSession(responses), ctx, entry, args)
    assert all(p["correct_post_edit"] for p in hard)
    tgt = next(p for p in hard if p["group"] == "target")
    assert tgt["classification"] == "correct"
    # Y02 exact value 15 -> frozen narrow bin 15-19
    assert tgt["expected_post_edit"] == "15–19 years"
    assert tgt["parsed_label"] == "15–19 years"


def test_hard_eval_stub_detects_depth_errors_and_leaks():
    ctx, entry = _numeric_ctx_and_entry()
    target_iid = next(iter(entry["assignments"]))
    responses = {}
    for iid in ctx["identity_ids"]:
        if iid == target_iid:
            responses[ctx["code_of"][iid]] = "10–19 years"  # over-abstract
        else:
            responses[ctx["code_of"][iid]] = \
                gxm.expected_label(ctx, entry, iid)
    args = argparse.Namespace(max_gen_tokens=12)
    hard = gxm._hard_eval(_StubSession(responses), ctx, entry, args)
    tgt = next(p for p in hard if p["group"] == "target")
    assert tgt["classification"] == "over_abstraction"
    assert not tgt["correct_post_edit"]
    assert tgt["contains_exact_value"] is True  # 15 in 10-19


def test_pass_criteria_frozen_gates():
    ctx, entry = _numeric_ctx_and_entry()
    ids = ctx["identity_ids"]
    t = next(iter(entry["assignments"]))

    def mk(pred_overrides=None, soft_overrides=None):
        hard = []
        for iid in ids:
            exp = gxm.expected_label(ctx, entry, iid)
            p = {"identity_id": iid, "parsed_label": exp,
                 "recognized_labels": [exp], "multi_label_ambiguous": False,
                 "group": "target" if iid in entry["assignments"]
                 else "retain",
                 "expected_post_edit": exp, "correct_post_edit": True,
                 "source_leaked": False, "classification": "correct"}
            if pred_overrides and iid in pred_overrides:
                p.update(pred_overrides[iid])
            hard.append(p)
        soft = {}
        for iid in ids:
            exp = gxm.expected_label(ctx, entry, iid)
            src = (entry["assignments"][iid]["source"]
                   if iid in entry["assignments"] else exp)
            probs = {l: 0.0 for l in ctx["vocab"]}
            probs[exp] = 0.999
            probs[src] = 1e-6
            s = {"probs": probs, "candidate_mass": 0.9995,
                 "other_mass": 0.0005}
            if soft_overrides and iid in soft_overrides:
                s.update(soft_overrides[iid])
            soft[iid] = s
        return hard, soft

    hard, soft = mk()
    crit = gxm._pass_criteria(hard, soft, entry, ctx)
    assert crit["cell_pass"] and crit["failed_criteria"] == []

    # p(source) too high -> fail
    hard, soft = mk(soft_overrides={t: {"probs": {
        **{l: 0.0 for l in ctx["vocab"]},
        gxm.expected_label(ctx, entry, t): 0.55,
        entry["assignments"][t]["source"]: 0.44}}})
    crit = gxm._pass_criteria(hard, soft, entry, ctx)
    assert not crit["cell_pass"]
    assert "max_target_p_source<=0.01" in crit["failed_criteria"]
    assert "min_target_p_desired>=0.90" in crit["failed_criteria"]

    # wrong-branch output -> fail
    hard, soft = mk(pred_overrides={t: {
        "parsed_label": "995 activities", "recognized_labels":
        ["995 activities"], "correct_post_edit": False,
        "classification": "wrong_branch"}})
    crit = gxm._pass_criteria(hard, soft, entry, ctx)
    assert not crit["cell_pass"]
    assert "wrong_branch_rate==0" in crit["failed_criteria"]
    assert "strict_expected_accuracy==1.0" in crit["failed_criteria"]
