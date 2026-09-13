"""The protocol-repair study tool must compare like with like.

This script produces a FILED scientific attribution, so its failure modes are
worse than a crash: an arm read from the wrong directory, or read with fewer
seeds than its partners, yields a comparison that still prints confident
numbers.  The tests here pin the guards that make that impossible, and pin that
the tool reads the frozen pilot cells as its baseline arm rather than
re-deriving or re-running anything.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
OUT_BASE = _ROOT / "e2c_granularity" / "outputs" / "mllmu"
TOOL_NAME = "e2c_v3_study_protocol_repair.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(
        "study_protocol_repair_under_test", _SCRIPTS / TOOL_NAME)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cell(seed, sid="gx_mll_151252", mass=0.995, tag=None):
    """A minimal cell in the shape the runner commits."""
    return {
        "cell_id": f"{sid}__seed{seed}" + (f"__{tag}" if tag else ""),
        "set_id": sid, "seed": seed,
        "assignments": {"003": "x", "004": "y"},
        "edit_protocol": {"ul_steps": 500, "retain_repeat": 3,
                          "cell_tag": tag},
        "criteria": {
            "cell_pass": True, "failed_criteria": [],
            "strict_expected_accuracy": 1.0,
            "min_candidate_mass": mass,
            "min_target_p_desired": mass, "max_target_p_source": 1e-5,
            "retain_acc": 1.0},
        "soft": {"003": {"candidate_mass": mass},
                 "004": {"candidate_mass": mass + 0.001}},
        "hard_preds": [
            {"identity_id": "003", "group": "target", "correct_post_edit": True,
             "expected_post_edit": "a", "parsed_label": "a"},
            # TWO same_leaf partners, as the real 151252 cells have: the leak
            # the retention claim rests on is "1 of 2", and a fixture with a
            # single same_leaf row could not distinguish that from "0 of 1".
            {"identity_id": "024", "group": "same_leaf",
             "correct_post_edit": True,
             "expected_post_edit": "Software Developer",
             "parsed_label": "Software Developer"},
            {"identity_id": "025", "group": "same_leaf",
             "correct_post_edit": True,
             "expected_post_edit": "Software Developer",
             "parsed_label": "Software Developer"}],
    }


def test_the_baseline_arm_is_the_frozen_pilot_cells_not_a_study_dir(tool):
    """``baseline`` must read the pilot's own two-level ``seed_*`` dirs.

    If it read a study dir the whole factor chain would compare protoB against
    protoA and attribute nothing, while still printing a plausible table.
    """
    assert tool.ARMS["baseline"] is None
    base = tool.cell_paths("gx_mll_151252", None)
    study = tool.cell_paths("gx_mll_151252", "protoB")
    assert base and study
    assert not (set(base) & set(study)), "an arm's cells appear in another's"
    assert all("study_" not in p for p in base)
    assert all("study_protoB" in p for p in study)


def test_each_edge_of_the_factor_chain_moves_exactly_one_knob(tool):
    """The attribution rests on the design, so the design is checked, not
    merely described: baseline->protoB differs only in retain_repeat and
    protoB->protoA differs only in ul_steps."""
    assert set(tool.PROTOCOL_DESIGN) == set(tool.ARMS)
    chain = tool.verify_factor_chain()
    assert sorted(chain) == ["baseline_to_protoB", "protoB_to_protoA"]
    assert chain["baseline_to_protoB"]["isolates"] == "retain_repeat"
    assert chain["baseline_to_protoB"]["held"] == {"ul_steps": 500}
    assert chain["protoB_to_protoA"]["isolates"] == "ul_steps"
    assert chain["protoB_to_protoA"]["held"] == {"retain_repeat": 9}


def test_a_confounded_edge_is_refused(tool):
    """The guard must have teeth: an edge that moves two knobs is the confound
    protoA already had, and it must stop the study rather than let it file an
    attribution it cannot support."""
    saved = dict(tool.PROTOCOL_DESIGN["protoB"])
    try:
        # protoB moving ul_steps too would make both edges confounded
        tool.PROTOCOL_DESIGN["protoB"]["ul_steps"] = 1000
        with pytest.raises(RuntimeError, match="confounded"):
            tool.verify_factor_chain()
    finally:
        tool.PROTOCOL_DESIGN["protoB"] = saved
    assert tool.verify_factor_chain()


def test_arm_summary_reads_the_leak_that_makes_the_retention_claim(tool):
    """The whole retain_repeat attribution rests on ONE leaking row in ONE
    cell, so the extractor must surface it by identity rather than as a
    ratio -- 1/2 is not evidence, '024 emitted the broad title instead of
    Software Developer' is."""
    c = _cell(123)
    c["hard_preds"][1].update(correct_post_edit=False,
                            parsed_label="Software and Web Developers, "
                                         "Programmers, and Testers")
    c["criteria"]["retain_acc"] = 0.9697
    got = _summary_of(tool, c)
    assert got["same_leaf_correct"] == 1
    assert got["same_leaf_total"] == 2
    assert got["retain_acc"] == 0.9697
    assert len(got["same_leaf_leaks"]) == 1
    leak = got["same_leaf_leaks"][0]
    assert leak["identity_id"] == "024"
    assert leak["expected"] == "Software Developer"
    assert "Software and Web Developers" in leak["got"]


def _summary_of(tool, cell):
    """Run ``arm_summary`` against one cell without touching the evidence
    tree, by pointing the loader at it."""
    real = tool.load_arm
    tool.load_arm = lambda sid, tag: {cell["seed"]: cell}
    try:
        return tool.arm_summary(cell["set_id"], tag=None)[str(cell["seed"])]
    finally:
        tool.load_arm = real


def test_the_sub_matrix_declares_only_the_sets_the_study_ran(tool):
    """Coverage is exact by construction, so the sub-matrix must not quietly
    keep the other three pilot sets: a 6-cell arm evaluated against a 15-cell
    design would report missing cells instead of a verdict."""
    matrix = {"sets": [{"set_id": s} for s in
                       ("gx_mll_151252", "gx_mll_254012", "gx_mll_191023",
                        "gx_mll_192041", "gx_mll_193091")],
              "n_sets": 5, "n_cells": 15, "edit_seeds": tool.SEEDS}
    sub = tool.sub_matrix(matrix)
    assert [e["set_id"] for e in sub["sets"]] == tool.SETS
    assert sub["n_sets"] == 2
    assert sub["n_cells"] == 2 * len(tool.SEEDS)
    # the original is untouched -- deepcopy, not mutation
    assert matrix["n_cells"] == 15
    assert len(matrix["sets"]) == 5


def test_the_averaging_counterfactual_is_computed_not_asserted(tool):
    """The claim that the pre-992efa9 gate would have passed these cells is
    the study's sharpest point, so it is derived from the same rows the gate
    saw.  A positive mean over rows that include inversions is exactly what
    averaging hides, and a negative mean is what it would have caught."""
    rows = [{"delta_retrain_l2": d} for d in (-0.5812, 1.3511, -0.1918,
                                              1.2994, -0.2075, 1.2884)]
    entry = tool.averaging_counterfactual(
        {"gx_mll_254012": {"per_target": rows}})["gx_mll_254012"]
    assert entry["n_rows"] == 6
    assert entry["n_rows_inverting"] == 3
    assert entry["mean_delta_retrain_over_rows"] == pytest.approx(
        sum(r["delta_retrain_l2"] for r in rows) / 6)
    # the mean is positive although three of six rows invert: this is precisely
    # the case an every-row criterion exists to catch
    assert entry["averaging_gate_would_pass"] is True

    caught = tool.averaging_counterfactual({"s": {"per_target": [
        {"delta_retrain_l2": -0.5}, {"delta_retrain_l2": -0.25}]}})
    assert caught["s"]["averaging_gate_would_pass"] is False
    assert caught["s"]["n_rows_inverting"] == 2

    # a set with no established rows must invent neither a mean nor a verdict
    empty = tool.averaging_counterfactual(
        {"s": {"per_target": [{"delta_retrain_l2": None}]}})
    assert empty["s"]["mean_delta_retrain_over_rows"] is None
    assert empty["s"]["averaging_gate_would_pass"] is False
    assert empty["s"]["n_rows"] == 0


def test_it_writes_only_its_own_report(tool):
    """The study files beside the sealed pilot reports, never over them."""
    assert tool.REPORT_NAME == "study_protoB_summary.json"
    sealed = ("g6_pilot_gates.json", "granularity_summary.json",
              "run_manifest.json", "study_protoA_summary.json")
    assert tool.REPORT_NAME not in sealed


@pytest.mark.skipif(not OUT_BASE.is_dir(),
                    reason="study evidence not present")
def test_the_real_arms_are_complete_and_the_attribution_holds(tool):
    """End to end on the committed cells: three arms x two sets x three seeds,
    and the two findings the study exists to make."""
    arms = {sid: {n: tool.arm_summary(sid, t) for n, t in tool.ARMS.items()}
            for sid in tool.SETS}
    for sid in tool.SETS:
        for name in tool.ARMS:
            # key=int: the seeds are numeric strings, and a lexicographic sort
            # would order them 123, 17, 42 -- asserting against the design's
            # own order only works if both sides are sorted the same way
            assert sorted(arms[sid][name], key=int) == \
                [str(s) for s in tool.SEEDS]

    # retain_repeat alone repairs the leak, at 500 steps
    assert arms["gx_mll_151252"]["baseline"]["123"]["same_leaf_correct"] == 1
    assert arms["gx_mll_151252"]["protoB"]["123"]["same_leaf_correct"] == 2
    assert arms["gx_mll_151252"]["protoB"]["123"]["retain_acc"] == 1.0

    # and alone it breaks cells the frozen recipe passed
    assert arms["gx_mll_151252"]["baseline"]["17"]["cell_pass"] is True
    assert arms["gx_mll_151252"]["protoB"]["17"]["cell_pass"] is False
    assert (arms["gx_mll_151252"]["protoB"]["17"]["min_candidate_mass"]
            < arms["gx_mll_151252"]["baseline"]["17"]["min_candidate_mass"])

    # ul_steps on top of retain_repeat restores them
    assert all(arms["gx_mll_151252"]["protoA"][str(s)]["cell_pass"]
               for s in tool.SEEDS)

    # 254012 never clears the floor under any recipe
    for name in tool.ARMS:
        for s in tool.SEEDS:
            assert (arms["gx_mll_254012"][name][str(s)]["min_candidate_mass"]
                    < tool.MASS_FLOOR)
            assert arms["gx_mll_254012"][name][str(s)]["cell_pass"] is False
