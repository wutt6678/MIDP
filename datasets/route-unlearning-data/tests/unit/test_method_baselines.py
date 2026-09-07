"""CPU tests for the E2C-v3 granularity METHOD baselines (MB0-MB2).

What this suite pins down:

* the FROZEN METHOD TABLE -- every method the comparison was asked for is a
  row, the two rows that are not independent methods (the counterfactual
  alias and the two reference rows) say so, and the ascent/descent pair sets
  each objective consumes come from the one frozen data spec;
* BUDGET DISCIPLINE -- identical pairs, repeats, steps, warmup, lr, seed and
  LoRA initialization across every trainable method, with the fresh-retrain
  oracle declared as the single sanctioned exception, and a hard gate that
  refuses to run a full-budget comparison at a budget the design did not
  freeze;
* PROMPT-ONLY HONESTY -- a policy may state the abstraction rule but never
  the answer, demonstrations are resolved at the requested member's own
  resolution, and a prompt row changes no weights so its retention is 1.0 by
  construction rather than by achievement;
* the METRIC SET the comparison reports, computed with the SAME frozen
  per-cell criteria function the matrix cells use;
* MB1 driven end to end by a stub model, so the pairs each trainer actually
  received, the fail-closed checkpoint reload, the alias row, the prompt rows
  and the cached-record validation are all checked without a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import types
from pathlib import Path
from typing import ClassVar

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mb = _load("mb_runner_under_test", "e2c_v3_method_baselines.py")
gx, gxm, mx, rv, rd = mb.gx, mb.gxm, mb.mx, mb.rv, mb.rd

DS_NUM = "celeba_numeric"
DS_SAL = "salmu"
NARROW = "gx_num_s_exact_to_narrow_Y02"
BROAD = "gx_num_s_exact_to_broad_Y04"
SAL_L1 = "gx_sal_s_L1_00060576"
SAL_L2 = "gx_sal_s_L2_00039880"
SAL_MIX = "gx_sal_mix_0"


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    """The frozen inputs are read through repo-relative paths."""
    monkeypatch.chdir(_ROOT)


def _design(ds):
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(_ROOT)
        matrix = gxm.load_frozen(ds)
        return mb.build_mb_manifest(ds, matrix, gxm.dataset_ctx(ds, matrix)), \
            matrix, gxm.dataset_ctx(ds, matrix)


@pytest.fixture(scope="session")
def num_design():
    return _design(DS_NUM)


@pytest.fixture(scope="session")
def sal_design():
    return _design(DS_SAL)


def _set_of(man, set_id):
    return next(s for s in man["sets"] if s["set_id"] == set_id)


def _entry_of(matrix, set_id):
    return next(e for e in matrix["sets"] if e["set_id"] == set_id)


def _mutate(obj):
    """A deep copy: negative tests mutate, they never touch a fixture."""
    return json.loads(json.dumps(obj))


# ------------------------------------------------------------------ #
# the frozen method table: every requested method, nothing smuggled in
# ------------------------------------------------------------------ #
#: the seven methods the comparison was asked for, and the row that carries
#: each one
REQUESTED_METHODS = {
    "direct supervised fine-tuning": "sft_target",
    "CF/GD": ("cf_relabel", "gd_distribution"),
    "GA plus retain descent": "ga_retain_descent",
    "NPO": "npo",
    "KL-anchored editing": "kl_anchored_edit",
    "prompting-only policy": "prompt_only",
    "fresh matched retraining": "matched_retrain",
}


def test_every_requested_method_is_a_row_in_the_frozen_table():
    for requested, rows in REQUESTED_METHODS.items():
        for row in (rows if isinstance(rows, tuple) else (rows,)):
            assert row in mb.MB_METHODS, f"{requested} -> {row} is missing"


def test_the_table_adds_only_rows_it_explains():
    extra = set(mb.MB_METHODS) - {
        r for rows in REQUESTED_METHODS.values()
        for r in (rows if isinstance(rows, tuple) else (rows,))}
    # kl_ascent_anchor is rv.train_kl exactly as shipped for deletion, kept so
    # the claim "it cannot install a label" is measured rather than asserted
    assert extra == {"kl_ascent_anchor"}
    assert mb.MB_METHODS["kl_ascent_anchor"]["descent_set"] == []
    assert "NO descent term" in mb.MB_METHODS["kl_ascent_anchor"]["note"]


def test_counterfactual_relabeling_is_declared_as_an_alias_not_a_method():
    cf = mb.MB_METHODS["cf_relabel"]
    assert cf["trainer"] == "alias"
    assert cf["aliased_to"] == "sft_target"
    assert cf["trains"] is False
    assert "coincide by construction" in cf["note"]
    assert "never as independent evidence" in cf["note"]


def test_reference_rows_are_never_called_methods():
    assert sorted(mb.MB_REFERENCES) == ["baseline_route", "loo_retrain"]
    assert sorted(mb.REFERENCE_ROWS) == sorted(mb.MB_REFERENCES)
    for rid, r in mb.MB_REFERENCES.items():
        assert r["trains"] is False
        assert rid not in mb.MB_METHODS


def test_the_row_order_puts_dependencies_before_the_rows_that_read_them():
    order = list(mb.ROW_ORDER)
    assert order[0] == "baseline_route"           # prompt rows inherit it
    assert order.index("sft_target") < order.index("cf_relabel")
    assert len(order) == len(set(order)) == 13
    assert set(order) == (((set(mb.MB_METHODS) - {"prompt_only"})
                           | set(mb.MB_REFERENCES))
                          | {f"prompt_only__{r}" for r in mb.PROMPT_ROW_ROLES})
    # prompt_only is ONE method but three frozen policies, so it is executed as
    # one row per policy and never as a fourth, unlabeled row
    assert "prompt_only" not in order
    # one row per instruction policy; the canonical-prompt control IS the
    # baseline_route row and is never re-run as a "policy"
    assert [r for r in order if r.startswith("prompt_only")] == \
        [f"prompt_only__{r}" for r in mb.PROMPT_ROW_ROLES]
    assert "canonical_control" not in mb.PROMPT_ROW_ROLES
    assert "canonical_control" in mb.PROMPT_POLICY_ROLES


def test_smoke_rows_are_a_subset_of_the_design_and_cover_every_row_shape():
    assert set(mb.SMOKE_ROWS) <= set(mb.ROW_ORDER)
    shapes = {"reference": "baseline_route", "reuse": "sft_target",
              "alias": "cf_relabel", "trained": "gd_distribution",
              "trained_new_objective": "kl_anchored_edit",
              "prompt": "prompt_only__rule_statement"}
    assert set(shapes.values()) == set(mb.SMOKE_ROWS)


# ------------------------------------------------------------------ #
# the representative subset (never the whole 147-cell matrix)
# ------------------------------------------------------------------ #
def test_the_representative_subset_is_one_set_per_transformation_type(
        num_design, sal_design):
    nman, _nmatrix, _ = num_design
    sman, _smatrix, _ = sal_design
    assert [s["set_id"] for s in sman["sets"]] == mb.MB_REP_SETS[DS_SAL]
    assert [s["set_id"] for s in nman["sets"]] == mb.MB_REP_SETS[DS_NUM]
    # the representatives are the same ones GX2B/GX2S already used
    assert mb.MB_REP_SETS == {ds: list(v) for ds, v in
                              gxm.BALANCED_REP_SETS.items()}
    modes = {s["mode"] for s in sman["sets"]} | {s["mode"] for s in nman["sets"]}
    assert "single_level1" in modes and "single_level2" in modes
    assert "simultaneous_mixed_depth" in modes
    assert "single_exact_to_narrow" in modes and "single_exact_to_broad" in modes


def test_the_subset_is_a_small_fraction_of_the_matrix(num_design, sal_design):
    nman, nmatrix, _ = num_design
    sman, smatrix, _ = sal_design
    assert len(smatrix["sets"]) == 21 and len(nmatrix["sets"]) == 28
    # 21 x 3 seeds + 28 x 3 seeds = the 147-cell matrix this pass refuses to
    # multiply the baselines across
    assert len(smatrix["sets"]) * len(smatrix["edit_seeds"]) == 63
    assert len(nmatrix["sets"]) * len(nmatrix["edit_seeds"]) == 84
    assert sman["n_sets"] + nman["n_sets"] == 5
    assert "NOT multiplied" in sman["rep_set_policy"]
    assert sman["edit_seed"] == nman["edit_seed"] == gxm.ORACLE_SEED == 17


def test_a_single_edit_seed_is_declared_and_the_matrix_seeds_are_not_used(
        num_design):
    man, matrix, _ = num_design
    assert man["edit_seed"] == 17
    assert matrix["edit_seeds"] == [17, 42, 123]
    assert "one edit seed" in man["rep_set_policy"] or man["edit_seed"] == 17


def test_representative_sets_all_have_the_artifacts_the_metrics_need(
        num_design, sal_design):
    for man, _matrix, _ctx in (num_design, sal_design):
        for s in man["sets"]:
            av = s["available_artifacts"]
            assert all(av.values()), (s["set_id"], av)


# ------------------------------------------------------------------ #
# the metric set the comparison must report
# ------------------------------------------------------------------ #
REQUESTED_METRICS = [
    "transformation_success", "source_label_suppression", "retention",
    "sibling_preservation", "wrong_branch_rate", "candidate_validity",
    "matched_vs_loo_oracle_distance", "training_time",
    "trainable_parameters",
]


def test_every_requested_metric_is_required_and_reported():
    assert mb.MB_REQUIRED_METRICS == REQUESTED_METRICS
    assert set(mb.MB_REQUIRED_METRICS) <= set(mb.MB_REPORTED_METRICS)
    # the frozen per-cell criteria ride along so the yardstick is visible
    assert "frozen_cell_criteria" in mb.MB_REPORTED_METRICS


def test_mb0_fails_if_the_report_cannot_emit_a_required_metric(num_design):
    man, _matrix, ctx = num_design
    broken = _mutate(man)
    broken["required_metrics"] = [*man["required_metrics"], "not_emitted"]
    issues = mb.validate_mb_manifest(DS_NUM, broken, ctx)
    assert any("not_emitted" in i for i in issues)


def test_the_scoring_limitation_is_recorded_in_the_design(num_design):
    man, _matrix, _ctx = num_design
    lim = man["scoring_limitation"]
    assert lim["missing_termination_event"] is True
    assert "SCORE SUM" in lim["consequence"]
    assert "full_sequence_label_probs" in lim["same_scorer_as"]
    assert lim == mb.MB_SCORING_LIMITATION


# ------------------------------------------------------------------ #
# one frozen data spec, identical for every trainable method
# ------------------------------------------------------------------ #
def test_the_data_spec_is_the_frozen_edit_recipe_not_the_callers_args(
        num_design):
    man, _matrix, _ctx = num_design
    s = _set_of(man, NARROW)
    spec = s["data_spec"]
    assert spec["budget"] == {"steps": mx.UL_STEPS, "warmup": mx.UL_WARMUP,
                              "lr": mx.UL_LR, "seed": mb.MB_SEED}
    assert spec["repeats"] == {
        "target": mx.UL_REPEAT * gxm.TARGET_BOOST,
        "source": mx.UL_REPEAT * gxm.TARGET_BOOST,
        "retain": mx.UL_REPEAT * gxm.RETAIN_REPEAT}
    # the builder takes no args at all: --smoke cannot freeze a cheap design
    import inspect
    assert list(inspect.signature(mb.data_spec).parameters) == ["ctx", "entry"]
    assert list(inspect.signature(mb.build_mb_manifest).parameters) == \
        ["ds", "matrix", "ctx"]


def test_target_and_source_pairs_are_the_two_sides_of_one_association(
        num_design):
    man, matrix, ctx = num_design
    entry = _entry_of(matrix, NARROW)
    spec = _set_of(man, NARROW)["data_spec"]
    t = "01"
    prompt = rd.CODE_TO_ALIAS_PROMPT.format(code=ctx["code_of"][t])
    assert spec["target_pairs"] == [
        {"prompt": prompt, "answer": entry["assignments"][t]["target"]}]
    assert spec["source_pairs"] == [
        {"prompt": prompt, "answer": entry["assignments"][t]["source"]}]
    assert spec["target_pairs"][0]["answer"] == "15\u201319 years"
    assert spec["source_pairs"][0]["answer"] == "15 years"


def test_the_retain_set_is_every_member_the_edit_does_not_transform(
        num_design):
    man, matrix, ctx = num_design
    entry = _entry_of(matrix, NARROW)
    spec = _set_of(man, NARROW)["data_spec"]
    assert len(spec["retain_pairs"]) == len(entry["retain_ids"]) == 23
    prompts = {p["prompt"] for p in spec["retain_pairs"]}
    # the transformed member is NOT retention-trained to its old label
    assert rd.CODE_TO_ALIAS_PROMPT.format(
        code=ctx["code_of"]["01"]) not in prompts
    for p in spec["retain_pairs"]:
        code = p["prompt"].split("Identity code: ")[1].split(".")[0]
        iid = next(i for i in ctx["identity_ids"] if ctx["code_of"][i] == code)
        assert p["answer"] == ctx["baseline_alias_of"][iid]


def test_no_source_pair_is_the_refusal_label_in_either_dataset(num_design,
                                                              sal_design):
    for man, _matrix, _ctx in (num_design, sal_design):
        for s in man["sets"]:
            for p in s["data_spec"]["source_pairs"]:
                assert p["answer"] != gx.DELETED_LABEL


def test_the_mixed_set_keeps_its_refusal_control_out_of_the_transformation(
        sal_design):
    man, matrix, _ctx = sal_design
    s = _set_of(man, SAL_MIX)
    assert s["transformation_targets"] == ["00036363", "00039880"]
    assert s["refusal_controls"] == ["00060576"]
    spec = s["data_spec"]
    # the refusal control IS an assignment, so it has a target pair (the row
    # is trained on the whole frozen cell recipe) but it is never counted
    # inside transformation success
    assert len(spec["target_pairs"]) == 3
    assert len(spec["source_pairs"]) == 3
    assert mb.transformation_targets(_entry_of(matrix, SAL_MIX)) == \
        ["00036363", "00039880"]


def test_the_reference_assignment_never_picks_a_refusal_control(sal_design):
    _man, matrix, _ctx = sal_design
    entry = _entry_of(matrix, SAL_MIX)
    a = mb._reference_assignment(entry)
    assert a["operation"] == "taxonomic"
    assert a["target_depth"] in (1, 2)


# ------------------------------------------------------------------ #
# budget discipline: identical where it is methodologically meaningful
# ------------------------------------------------------------------ #
def test_every_trainable_method_declares_the_edit_budget(num_design):
    man, _matrix, _ctx = num_design
    trainable = [m for m, s in man["methods"].items() if s["trains"]]
    assert sorted(trainable) == ["ga_retain_descent", "gd_distribution",
                                 "kl_anchored_edit", "kl_ascent_anchor", "npo"]
    for mid in trainable:
        assert man["methods"][mid]["budget"] == "edit"
        assert man["methods"][mid]["budget_mismatch_ok"] is False
    for mid in ("matched_retrain", "loo_retrain"):
        spec = (man["methods"].get(mid) or man["references"][mid])
        assert spec["budget"] == "oracle"
        assert spec["budget_mismatch_ok"] is True


def test_the_budget_policy_names_the_one_sanctioned_exception(num_design):
    man, _matrix, _ctx = num_design
    pol = man["budget_policy"]
    assert pol["sanctioned_exception"]["rows"] == ["matched_retrain",
                                                  "loo_retrain"]
    for field in ("steps", "warmup", "lr", "edit seed", "LoRA configuration"):
        assert any(field in f for f in pol["identical_across_trainable_methods"])
    assert pol["may_differ"] == [
        "the objective and which pair set feeds which side of it"]


def test_objectives_only_name_pair_sets_the_frozen_spec_contains(num_design):
    man, _matrix, ctx = num_design
    for mid, m in man["methods"].items():
        for side in ("ascent_set", "descent_set"):
            v = m.get(side)
            names = v if isinstance(v, list) else ([v] if v else [])
            assert set(names) <= {"target", "source", "retain"}, (mid, side)
        assert m.get("distribution_set") in (None, "target")
    broken = _mutate(man)
    broken["methods"]["npo"]["descent_set"] = ["target", "holdout"]
    assert any("holdout" in i for i in
               mb.validate_mb_manifest(DS_NUM, broken, ctx))


def test_the_budget_gate_refuses_a_full_run_at_an_unfrozen_budget(num_design):
    man, _matrix, _ctx = num_design
    spec = _set_of(man, NARROW)["data_spec"]
    ok = argparse.Namespace(ul_steps=mx.UL_STEPS, ul_warmup=mx.UL_WARMUP,
                            ul_lr=mx.UL_LR, ul_repeat=mx.UL_REPEAT)
    assert mb.check_budget_matches_design(ok, spec, smoke=False) == []
    cheap = argparse.Namespace(ul_steps=20, ul_warmup=2, ul_lr=mx.UL_LR,
                               ul_repeat=mx.UL_REPEAT)
    with pytest.raises(RuntimeError) as exc:
        mb.check_budget_matches_design(cheap, spec, smoke=False)
    assert "same budget" in str(exc.value)
    # --smoke is allowed to be cheap and says so
    assert mb.check_budget_matches_design(cheap, spec, smoke=True) == []


def test_a_wrong_repeat_is_also_a_budget_difference(num_design):
    man, _matrix, _ctx = num_design
    spec = _set_of(man, NARROW)["data_spec"]
    args = argparse.Namespace(ul_steps=mx.UL_STEPS, ul_warmup=mx.UL_WARMUP,
                              ul_lr=mx.UL_LR, ul_repeat=5)
    with pytest.raises(RuntimeError) as exc:
        mb.check_budget_matches_design(args, spec, smoke=False)
    assert "repeats" in str(exc.value)


def test_executed_repeats_follow_the_caller_and_the_design_does_not():
    args = argparse.Namespace(ul_repeat=1)
    assert mb.executed_repeats(args) == {"target": gxm.TARGET_BOOST,
                                        "source": gxm.TARGET_BOOST,
                                        "retain": gxm.RETAIN_REPEAT}
    args = argparse.Namespace(ul_repeat=mx.UL_REPEAT)
    assert mb.executed_repeats(args) == {
        "target": mx.UL_REPEAT * gxm.TARGET_BOOST,
        "source": mx.UL_REPEAT * gxm.TARGET_BOOST,
        "retain": mx.UL_REPEAT * gxm.RETAIN_REPEAT}


def test_the_row_recipe_records_both_the_design_and_the_executed_budget(
        num_design):
    man, _matrix, _ctx = num_design
    cheap = argparse.Namespace(ul_steps=20, ul_warmup=2, ul_lr=mx.UL_LR,
                               ul_repeat=mx.UL_REPEAT)
    full = argparse.Namespace(ul_steps=mx.UL_STEPS, ul_warmup=mx.UL_WARMUP,
                              ul_lr=mx.UL_LR, ul_repeat=mx.UL_REPEAT)
    smoke = mb.row_recipe(man, NARROW, "gd_distribution", cheap)
    real = mb.row_recipe(man, NARROW, "gd_distribution", full)
    assert smoke != real            # a cheap row can never satisfy a full one
    assert smoke["design_budget"] == real["design_budget"]
    assert smoke["steps"] == 20 and real["steps"] == mx.UL_STEPS
    assert real["objective"] == man["methods"]["gd_distribution"]["objective"]
    assert real["seed"] == mb.MB_SEED


def test_the_prompt_role_is_part_of_the_recipe(num_design):
    man, _matrix, _ctx = num_design
    args = argparse.Namespace(ul_steps=mx.UL_STEPS, ul_warmup=mx.UL_WARMUP,
                              ul_lr=mx.UL_LR, ul_repeat=mx.UL_REPEAT)
    a = mb.row_recipe(man, NARROW, "prompt_only__rule_statement", args)
    b = mb.row_recipe(man, NARROW, "prompt_only__coarsen_instruction", args)
    assert a["prompt_role"] == "rule_statement"
    assert b["prompt_role"] == "coarsen_instruction"
    assert a != b
    assert mb.row_recipe(man, NARROW, "sft_target", args)["prompt_role"] is None


# ------------------------------------------------------------------ #
# prompt-only honesty: state the rule, never the answer
# ------------------------------------------------------------------ #
def test_no_policy_prompt_names_the_answer_it_is_measured_on(num_design,
                                                             sal_design):
    for man, matrix, ctx in (num_design, sal_design):
        for s in man["sets"]:
            entry = _entry_of(matrix, s["set_id"])
            labels = {a["target"] for a in entry["assignments"].values()} | \
                {a["source"] for a in entry["assignments"].values()}
            for t, roles in s["prompt_policies"]["by_target"].items():
                for role in mb.PROMPT_POLICY_ROLES:
                    text = roles[role].lower()
                    for lab in labels:
                        assert lab.lower() not in text, (s["set_id"], t, role,
                                                         lab)


def test_the_canonical_control_is_the_frozen_prompt(num_design):
    man, _matrix, ctx = num_design
    s = _set_of(man, NARROW)
    roles = s["prompt_policies"]["by_target"]["01"]
    assert roles["canonical_control"] == rd.CODE_TO_ALIAS_PROMPT.format(
        code=ctx["code_of"]["01"])
    assert s["prompt_policies"]["canonical_control_is"].startswith(
        "the baseline_route reference row")


def test_policies_state_the_requested_resolution(num_design, sal_design):
    nman, _nm, _nctx = num_design
    narrow = _set_of(nman, NARROW)["prompt_policies"]["by_target"]["01"]
    assert "5-year" in narrow["rule_statement"]
    assert "5-year" in narrow["coarsen_instruction"]
    assert "multiples of 5" in narrow["rule_statement"]
    broad = _set_of(nman, BROAD)["prompt_policies"]["by_target"]["03"]
    assert "10-year" in broad["rule_statement"]
    assert "10-year" in broad["fewshot_other_branch"]

    sman, _sm, _sctx = sal_design
    l1 = _set_of(sman, SAL_L1)["prompt_policies"]["by_target"]["00060576"]
    assert "level 1" in l1["rule_statement"]
    assert "level-1" in l1["coarsen_instruction"]
    l2 = _set_of(sman, SAL_L2)["prompt_policies"]["by_target"]["00039880"]
    assert "level 2" in l2["rule_statement"]
    assert "level-2" in l2["coarsen_instruction"]
    assert "level-2" in l2["fewshot_other_branch"]


def test_a_mixed_depth_set_gives_each_member_its_own_resolution(sal_design):
    """The two transformed members of mix_0 ask for different depths, so the
    policy text AND the demonstrations must differ per member."""
    man, _matrix, ctx = sal_design
    s = _set_of(man, SAL_MIX)
    by_target = s["prompt_policies"]["by_target"]
    assert "level-1" in by_target["00036363"]["coarsen_instruction"]
    assert "level-2" in by_target["00039880"]["coarsen_instruction"]
    demos_1 = by_target["00036363"]["demonstrations"]
    demos_2 = by_target["00039880"]["demonstrations"]
    # every demonstration is resolved at the resolution its member asked for
    for d in demos_1:
        assert ctx["hierarchy_of"][d["identity_id"]][1] == d["coarse_label"]
    for d in demos_2:
        assert ctx["hierarchy_of"][d["identity_id"]][2] == d["coarse_label"]
    assert {d["coarse_label"] for d in demos_1} != \
        {d["coarse_label"] for d in demos_2}


def test_demonstrations_come_from_a_different_branch_or_bin(num_design,
                                                            sal_design):
    for man, matrix, ctx in (num_design, sal_design):
        for s in man["sets"]:
            entry = _entry_of(matrix, s["set_id"])
            labels = {a["target"] for a in entry["assignments"].values()} | \
                {a["source"] for a in entry["assignments"].values()}
            for roles in s["prompt_policies"]["by_target"].values():
                demos = roles["demonstrations"]
                assert len(demos) == mb.N_DEMOS
                for d in demos:
                    assert d["identity_id"] not in entry["assignments"]
                    assert d["coarse_label"] not in labels
                    assert d["own_label"] == ctx["baseline_alias_of"][
                        d["identity_id"]]
                    # a demonstration that teaches abstraction must actually
                    # abstract: its answer differs from its own label
                    assert d["coarse_label"] != d["own_label"]


def test_the_demo_selection_rule_is_frozen_and_deterministic(num_design):
    man, matrix, ctx = num_design
    entry = _entry_of(matrix, NARROW)
    a = mb._reference_assignment(entry)
    labels = {"15\u201319 years"}
    first = mb._demo_pool(ctx, entry, a, labels)
    again = mb._demo_pool(ctx, entry, a, labels)
    assert first == again
    codes = [d["code"] for d in first]
    assert codes == sorted(codes)                  # sorted by code, frozen
    assert len(first) == _set_of(man, NARROW)["prompt_policies"][
        "n_demo_pool_by_target"]["01"]


def test_a_refusal_control_is_excluded_from_the_prompt_policies(sal_design):
    man, _matrix, _ctx = sal_design
    pol = _set_of(man, SAL_MIX)["prompt_policies"]
    assert pol["refusal_controls_excluded"] == ["00060576"]
    assert "00060576" not in pol["by_target"]


def test_mb0_fails_when_a_policy_hands_over_the_answer(num_design):
    man, _matrix, ctx = num_design
    broken = _mutate(man)
    s = _set_of(broken, NARROW)
    s["prompt_policies"]["by_target"]["01"]["rule_statement"] = (
        "Identity code: GRN_01. Report 15\u201319 years.")
    issues = mb.validate_mb_manifest(DS_NUM, broken, ctx)
    assert any("CONTAINS" in i and "rule_statement" in i for i in issues)


def test_mb0_fails_when_a_demonstration_is_a_transformed_member(num_design):
    man, _matrix, ctx = num_design
    broken = _mutate(man)
    s = _set_of(broken, NARROW)
    demos = s["prompt_policies"]["by_target"]["01"]["demonstrations"]
    demos[0]["identity_id"] = "01"
    issues = mb.validate_mb_manifest(DS_NUM, broken, ctx)
    assert any("is a transformed member" in i for i in issues)


def test_mb0_fails_when_a_demonstration_is_at_the_wrong_resolution(
        sal_design):
    man, _matrix, ctx = sal_design
    broken = _mutate(man)
    s = _set_of(broken, SAL_MIX)
    # hand the depth-2 member the depth-1 demonstrations of its sibling
    demos = s["prompt_policies"]["by_target"]["00036363"]["demonstrations"]
    s["prompt_policies"]["by_target"]["00039880"]["demonstrations"] = _mutate(
        demos)
    issues = mb.validate_mb_manifest(DS_SAL, broken, ctx)
    assert any("requested resolution" in i for i in issues)


def test_mb0_fails_when_a_demonstration_answer_is_one_of_the_sets_labels(
        num_design):
    man, _matrix, ctx = num_design
    broken = _mutate(man)
    s = _set_of(broken, NARROW)
    s["prompt_policies"]["by_target"]["01"]["demonstrations"][0][
        "coarse_label"] = "15\u201319 years"
    issues = mb.validate_mb_manifest(DS_NUM, broken, ctx)
    assert any("is one of this set's labels" in i for i in issues)


def test_mb0_fails_when_there_are_not_enough_demonstrations(num_design):
    man, _matrix, ctx = num_design
    broken = _mutate(man)
    s = _set_of(broken, NARROW)
    s["prompt_policies"]["by_target"]["01"]["demonstrations"] = []
    issues = mb.validate_mb_manifest(DS_NUM, broken, ctx)
    assert any("demonstrations available" in i for i in issues)


def test_the_over_application_prompt_is_resolvable_for_every_member(
        num_design, sal_design):
    """Mode (b) applies one policy to members it was not aimed at, so the
    frozen text must render for every identity, target or not."""
    for man, matrix, ctx in (num_design, sal_design):
        for s in man["sets"]:
            entry = _entry_of(matrix, s["set_id"])
            for role in mb.PROMPT_ROW_ROLES:
                for iid in ctx["identity_ids"]:
                    text = mb._policy_prompt_for(ctx, s, entry, role, iid)
                    assert ctx["code_of"][iid] in text
                    assert "{" not in text and "}" not in text
                # a transformed member gets its OWN resolved policy text
                t = s["transformation_targets"][0]
                assert mb._policy_prompt_for(ctx, s, entry, role, t) == \
                    s["prompt_policies"]["by_target"][t][role]


def test_mode_a_leaves_untouched_members_on_the_canonical_prompt(num_design):
    man, _matrix, ctx = num_design
    s = _set_of(man, NARROW)
    fn = mb._mode_a_prompt_fn(ctx, s, "rule_statement")
    assert fn("01") == s["prompt_policies"]["by_target"]["01"][
        "rule_statement"]
    other = next(i for i in ctx["identity_ids"] if i != "01")
    assert fn(other) == rd.CODE_TO_ALIAS_PROMPT.format(
        code=ctx["code_of"][other])


# ------------------------------------------------------------------ #
# MB0 design gate: what it refuses to freeze
# ------------------------------------------------------------------ #
def test_a_clean_design_validates_without_issues(num_design, sal_design):
    for ds, (man, _matrix, ctx) in ((DS_NUM, num_design),
                                    (DS_SAL, sal_design)):
        assert mb.validate_mb_manifest(ds, man, ctx) == []


def test_mb0_refuses_a_set_with_no_transformation_target(num_design):
    man, _matrix, ctx = num_design
    broken = _mutate(man)
    s = _set_of(broken, NARROW)
    s["transformation_targets"] = []
    s["assignments"]["01"]["operation"] = "refusal"
    issues = mb.validate_mb_manifest(DS_NUM, broken, ctx)
    assert any("no transformation target" in i for i in issues)


def test_mb0_refuses_an_empty_retain_set(num_design):
    man, _matrix, ctx = num_design
    broken = _mutate(man)
    _set_of(broken, NARROW)["data_spec"]["retain_pairs"] = []
    assert any("empty retain set" in i
               for i in mb.validate_mb_manifest(DS_NUM, broken, ctx))


def test_mb0_refuses_target_pairs_that_do_not_cover_the_assignments(
        num_design):
    man, _matrix, ctx = num_design
    broken = _mutate(man)
    _set_of(broken, NARROW)["data_spec"]["target_pairs"] = []
    assert any("do not cover every assignment" in i
               for i in mb.validate_mb_manifest(DS_NUM, broken, ctx))


def test_mb0_refuses_a_refusal_pair_disguised_as_a_transformation(
        num_design):
    man, _matrix, ctx = num_design
    broken = _mutate(man)
    _set_of(broken, NARROW)["data_spec"]["source_pairs"][0][
        "answer"] = gx.DELETED_LABEL
    assert any("this comparison is about transformation" in i
               for i in mb.validate_mb_manifest(DS_NUM, broken, ctx))


def test_mb0_refuses_a_data_spec_at_a_different_seed(num_design):
    man, _matrix, ctx = num_design
    broken = _mutate(man)
    _set_of(broken, NARROW)["data_spec"]["budget"]["seed"] = 42
    assert any("frozen edit seed" in i
               for i in mb.validate_mb_manifest(DS_NUM, broken, ctx))


def test_mb0_refuses_an_unknown_budget_label(num_design):
    man, _matrix, ctx = num_design
    broken = _mutate(man)
    broken["methods"]["npo"]["budget"] = "generous"
    assert any("unknown budget labels" in i
               for i in mb.validate_mb_manifest(DS_NUM, broken, ctx))


def test_mb0_refuses_to_let_the_counterfactual_alias_go_undeclared(
        num_design):
    man, _matrix, ctx = num_design
    broken = _mutate(man)
    del broken["methods"]["cf_relabel"]["aliased_to"]
    assert any("must declare its alias" in i
               for i in mb.validate_mb_manifest(DS_NUM, broken, ctx))


def test_mb0_refuses_a_kl_ascent_row_that_gained_a_descent_term(num_design):
    man, _matrix, ctx = num_design
    broken = _mutate(man)
    broken["methods"]["kl_ascent_anchor"]["descent_set"] = ["target"]
    broken["methods"]["kl_ascent_anchor"]["installs_target"] = True
    issues = mb.validate_mb_manifest(DS_NUM, broken, ctx)
    assert any("gained a descent term" in i for i in issues)


def test_mb0_refuses_a_method_that_does_not_install_the_target(num_design):
    man, _matrix, ctx = num_design
    for mid in ("sft_target", "gd_distribution", "ga_retain_descent", "npo",
                "kl_anchored_edit"):
        broken = _mutate(man)
        broken["methods"][mid]["installs_target"] = False
        assert any("does not install the target label" in i
                   for i in mb.validate_mb_manifest(DS_NUM, broken, ctx)), mid


def test_installs_target_is_derived_not_asserted(num_design):
    man, _matrix, _ctx = num_design
    assert man["methods"]["gd_distribution"]["installs_target"] is True
    assert man["methods"]["gd_distribution"]["distribution_set"] == "target"
    assert man["methods"]["ga_retain_descent"]["installs_target"] is True
    assert man["methods"]["kl_ascent_anchor"]["installs_target"] is False
    assert man["methods"]["prompt_only"]["installs_target"] is False
    assert man["references"]["baseline_route"]["installs_target"] is False


def test_mb0_refuses_a_missing_frozen_cell_or_oracle(num_design):
    man, _matrix, ctx = num_design
    for key, fragment in (("edited_cell_checkpoint", "no frozen edited cell"),
                          ("matched_retrain_soft", "matched-vs-LOO oracle"),
                          ("loo_retrain_checkpoint", "matched-vs-LOO oracle")):
        broken = _mutate(man)
        _set_of(broken, NARROW)["available_artifacts"][key] = False
        assert any(fragment in i
                   for i in mb.validate_mb_manifest(DS_NUM, broken, ctx)), key


# ------------------------------------------------------------------ #
# MB0 freeze / verify-not-rewrite / byte determinism
# ------------------------------------------------------------------ #
def _mb0(tmp_path, monkeypatch, ds, man):
    monkeypatch.setattr(mb, "MB_MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(mb, "MB_OUT_ROOT", tmp_path / "outputs")
    matrix = gxm.load_frozen(ds)
    ctx = gxm.dataset_ctx(ds, matrix)
    monkeypatch.setattr(mb, "build_mb_manifest", lambda *a, **k: man)
    return mb.build_or_verify(ds, matrix, ctx)


def test_mb0_freezes_the_design_the_first_time(tmp_path, monkeypatch,
                                               num_design):
    man, _matrix, _ctx = num_design
    got = _mb0(tmp_path, monkeypatch, DS_NUM, man)
    path = tmp_path / "manifests" / f"mb_manifest_{DS_NUM}.json"
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == man
    assert got == man
    validation = json.loads((tmp_path / "outputs" / DS_NUM
                             / "mb0_validation.json").read_text())
    assert validation["issues"] == []
    assert validation["n_sets"] == man["n_sets"]
    assert validation["trained_methods"] == sorted(mb.MB_TRAINABLE)
    # deterministic bytes: a committed file carries no timestamp
    assert "generated_at" not in validation
    assert "updated_at" not in validation


def test_mb0_verifies_and_never_rewrites_a_committed_design(tmp_path,
                                                            monkeypatch,
                                                            num_design):
    man, _matrix, _ctx = num_design
    _mb0(tmp_path, monkeypatch, DS_NUM, man)
    path = tmp_path / "manifests" / f"mb_manifest_{DS_NUM}.json"
    before = path.read_bytes()
    drifted_man = _mutate(man)
    drifted_man["edit_seed"] = 42
    path.write_text(json.dumps(drifted_man, indent=2) + "\n",
                    encoding="utf-8")
    drifted = path.read_bytes()
    assert drifted != before
    with pytest.raises(RuntimeError) as exc:
        _mb0(tmp_path, monkeypatch, DS_NUM, man)
    assert "must not drift" in str(exc.value)
    # the committed bytes are evidence: a failed verification changes nothing
    assert path.read_bytes() == drifted != before


def test_mb0_is_byte_deterministic(tmp_path, monkeypatch, num_design):
    man, _matrix, _ctx = num_design
    _mb0(tmp_path, monkeypatch, DS_NUM, man)
    first = (tmp_path / "manifests" / f"mb_manifest_{DS_NUM}.json").read_bytes()
    v1 = (tmp_path / "outputs" / DS_NUM / "mb0_validation.json").read_bytes()
    (tmp_path / "manifests" / f"mb_manifest_{DS_NUM}.json").unlink()
    _mb0(tmp_path, monkeypatch, DS_NUM, man)
    assert (tmp_path / "manifests"
            / f"mb_manifest_{DS_NUM}.json").read_bytes() == first
    assert (tmp_path / "outputs" / DS_NUM
            / "mb0_validation.json").read_bytes() == v1


def test_mb0_fails_closed_on_a_broken_design_before_writing_anything(
        tmp_path, monkeypatch, num_design):
    man, _matrix, _ctx = num_design
    broken = _mutate(man)
    broken["methods"]["cf_relabel"].pop("aliased_to")
    with pytest.raises(RuntimeError) as exc:
        _mb0(tmp_path, monkeypatch, DS_NUM, broken)
    assert "MB0 validation failed" in str(exc.value)
    assert not (tmp_path / "manifests"
                / f"mb_manifest_{DS_NUM}.json").exists()
    assert not (tmp_path / "outputs" / DS_NUM
                / "mb0_validation.json").exists()


def test_load_frozen_mb_reads_the_committed_design(tmp_path, monkeypatch,
                                                   num_design):
    man, _matrix, _ctx = num_design
    _mb0(tmp_path, monkeypatch, DS_NUM, man)
    assert mb.load_frozen_mb(DS_NUM) == man


# ------------------------------------------------------------------ #
# cached rows are validated, never trusted because they are readable
# ------------------------------------------------------------------ #
def test_the_cache_policy_declares_what_must_match():
    pol = mb.MB_CACHE_VALIDATION
    for field in ("kind", "dataset", "set_id", "seed", "row_id", "recipe",
                  "checkpoint_sha256"):
        assert field in pol["required_to_match"]
    assert "provenance.mb_manifest_sha256" in pol["required_to_match"]
    assert "provenance.shared_scoring_script_sha256" in \
        pol["required_to_match"]
    # the runner's own hash is recorded but not required: a bug fix to this
    # script must not invalidate a measurement the frozen scorer produced
    assert "provenance.runner_script_sha256" in pol["recorded_not_required"]
    assert "quarantine" in pol["on_mismatch"]


def _row_rec(ds, sid, row_id, recipe, ckpt_sha, man_sha):
    return {"kind": mb.ROW_KIND, "dataset": ds, "set_id": sid,
            "row_id": row_id, "row_type": "method", "seed": mb.MB_SEED,
            "recipe": recipe, "checkpoint_sha256": ckpt_sha,
            "provenance": {"mb_manifest_sha256": man_sha,
                           "shared_scoring_script_sha256": rv.script_sha256(),
                           "runner_script_sha256": "irrelevant",
                           "git_commit": "deadbeef"}}


def test_a_matching_cached_row_is_reusable(tmp_path, monkeypatch, num_design):
    man, _matrix, _ctx = num_design
    monkeypatch.setattr(mb, "MB_MANIFEST_DIR", tmp_path / "manifests")
    (tmp_path / "manifests").mkdir(parents=True)
    mpath = tmp_path / "manifests" / f"mb_manifest_{DS_NUM}.json"
    mpath.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    args = argparse.Namespace(ul_steps=mx.UL_STEPS, ul_warmup=mx.UL_WARMUP,
                              ul_lr=mx.UL_LR, ul_repeat=mx.UL_REPEAT)
    recipe = mb.row_recipe(man, NARROW, "npo", args)
    rec = _row_rec(DS_NUM, NARROW, "npo", recipe, "sha-of-ckpt",
                   rv.sha256_file(mpath))
    assert mb.validate_cached_row(rec, DS_NUM, NARROW, "npo", recipe,
                                  "sha-of-ckpt") == []


def test_every_stale_field_is_reported_as_a_reason(tmp_path, monkeypatch,
                                                   num_design):
    man, _matrix, _ctx = num_design
    monkeypatch.setattr(mb, "MB_MANIFEST_DIR", tmp_path / "manifests")
    (tmp_path / "manifests").mkdir(parents=True)
    mpath = tmp_path / "manifests" / f"mb_manifest_{DS_NUM}.json"
    mpath.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    args = argparse.Namespace(ul_steps=mx.UL_STEPS, ul_warmup=mx.UL_WARMUP,
                              ul_lr=mx.UL_LR, ul_repeat=mx.UL_REPEAT)
    recipe = mb.row_recipe(man, NARROW, "npo", args)
    good = _row_rec(DS_NUM, NARROW, "npo", recipe, "sha", rv.sha256_file(mpath))
    cases = {
        "kind": dict(good, kind="something_else"),
        "dataset": dict(good, dataset=DS_SAL),
        "set_id": dict(good, set_id=BROAD),
        "row_id": dict(good, row_id="ga_retain_descent"),
        "seed": dict(good, seed=42),
        "recipe": dict(good, recipe=dict(recipe, steps=20)),
        "checkpoint_sha256": dict(good, checkpoint_sha256="other-bytes"),
        "provenance.mb_manifest_sha256": dict(
            good, provenance=dict(good["provenance"],
                                  mb_manifest_sha256="0" * 64)),
        "provenance.shared_scoring_script_sha256": dict(
            good, provenance=dict(good["provenance"],
                                  shared_scoring_script_sha256="0" * 64)),
    }
    for reason, rec in cases.items():
        got = mb.validate_cached_row(rec, DS_NUM, NARROW, "npo", recipe, "sha")
        assert got == [reason], (reason, got)


def test_a_missing_manifest_makes_every_cached_row_stale(tmp_path,
                                                         monkeypatch,
                                                         num_design):
    man, _matrix, _ctx = num_design
    monkeypatch.setattr(mb, "MB_MANIFEST_DIR", tmp_path / "nowhere")
    args = argparse.Namespace(ul_steps=mx.UL_STEPS, ul_warmup=mx.UL_WARMUP,
                              ul_lr=mx.UL_LR, ul_repeat=mx.UL_REPEAT)
    recipe = mb.row_recipe(man, NARROW, "npo", args)
    rec = _row_rec(DS_NUM, NARROW, "npo", recipe, "sha", "0" * 64)
    assert mb.validate_cached_row(rec, DS_NUM, NARROW, "npo", recipe,
                                  "sha") == ["provenance.mb_manifest_sha256"]


def test_a_replaced_checkpoint_invalidates_the_row_that_measured_it(
        tmp_path, monkeypatch, num_design):
    man, _matrix, _ctx = num_design
    monkeypatch.setattr(mb, "MB_MANIFEST_DIR", tmp_path / "manifests")
    (tmp_path / "manifests").mkdir(parents=True)
    mpath = tmp_path / "manifests" / f"mb_manifest_{DS_NUM}.json"
    mpath.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    args = argparse.Namespace(ul_steps=mx.UL_STEPS, ul_warmup=mx.UL_WARMUP,
                              ul_lr=mx.UL_LR, ul_repeat=mx.UL_REPEAT)
    recipe = mb.row_recipe(man, NARROW, "npo", args)
    rec = _row_rec(DS_NUM, NARROW, "npo", recipe, "sha-of-old-bytes",
                   rv.sha256_file(mpath))
    assert mb.validate_cached_row(rec, DS_NUM, NARROW, "npo", recipe,
                                  "sha-of-new-bytes") == ["checkpoint_sha256"]
    # a row with no checkpoint on disk cannot be checked against bytes, so
    # the checkpoint field alone is not a reason
    assert mb.validate_cached_row(rec, DS_NUM, NARROW, "npo", recipe,
                                  None) == []


def test_quarantine_is_deterministic_and_keeps_the_evidence(tmp_path):
    path = tmp_path / "mb_row.json"
    path.write_text(json.dumps({"row_id": "npo"}), encoding="utf-8")
    dest = mb._quarantine(path, ["recipe", "checkpoint_sha256"])
    digest = hashlib.sha256(b"checkpoint_sha256|recipe").hexdigest()[:8]
    assert dest.name == f"mb_row.stale-{digest}.json"
    assert not path.exists()
    assert json.loads(dest.read_text(encoding="utf-8")) == {"row_id": "npo"}
    # the same reasons always produce the same name, so a repeated quarantine
    # never accumulates anonymous copies
    other = tmp_path / "mb_row.json"
    other.write_text("{}", encoding="utf-8")
    assert mb._quarantine(other, ["checkpoint_sha256", "recipe"]).name == \
        dest.name


# ------------------------------------------------------------------ #
# MB1 driven by a stub model: no GPU, but the real runner code path
# ------------------------------------------------------------------ #
def _key_of(ckpt_path):
    """Which weights a row evaluated: route_h | edited_h | row_id | oracle."""
    return Path(str(ckpt_path)).parts[-3]


def _gran_tree(tmp_path, ds, man, ctx, *, cell_record=True,
               oracle_soft=True):
    """A fake granularity output tree: baseline route h, frozen cell E and the
    four oracle families for every representative set.

    Each oracle family gets the distribution its own recipe implies for a
    transformed member: the matched families installed the coarser label, the
    leave-one-out fine-tune kept the baseline association, and the fresh
    leave-one-out retrain never saw the code at all.
    """
    gran = tmp_path / "gran" / ds
    fin = gran / "route_h" / "adapter_final"
    fin.mkdir(parents=True, exist_ok=True)
    (fin / "adapter_model.safetensors").write_bytes(b"baseline-route-h")
    families = (("matched_finetune", "matched_{sid}", "target"),
                ("loo_finetune", "loo_{sid}", "own"),
                ("matched_retrain", "matched_retrain_{sid}", "target"),
                ("loo_retrain", "loo_retrain_{sid}", "deleted"))
    for s in man["sets"]:
        sid = s["set_id"]
        cell = gran / "cells" / sid / f"seed_{mb.MB_SEED}"
        cfin = cell / "edited_h" / "adapter_final"
        cfin.mkdir(parents=True, exist_ok=True)
        (cfin / "adapter_model.safetensors").write_bytes(f"E-{sid}".encode())
        if cell_record:
            (cell / "cell_results.json").write_text(json.dumps({
                "cell_id": f"{sid}__seed{mb.MB_SEED}", "set_id": sid,
                "checkpoint_sha256": rv.sha256_file(
                    cfin / "adapter_model.safetensors"),
                "criteria": {"cell_pass": True, "failed_criteria": [],
                             "checks": {"stub": True}},
            }, indent=2) + "\n", encoding="utf-8")
        for fam, pattern, target_label in families:
            dirname = pattern.format(sid=sid)
            d = gran / "oracles" / dirname
            afin = d / "adapter_final"
            afin.mkdir(parents=True, exist_ok=True)
            (afin / "adapter_model.safetensors").write_bytes(dirname.encode())
            (d / "oracle_results.json").write_text(json.dumps({
                "family": fam, "set_id": sid, "stub": True}),
                encoding="utf-8")
            if not oracle_soft:
                continue
            soft = {}
            for iid in ctx["identity_ids"]:
                if iid in s["transformation_targets"]:
                    lab = {"target": s["assignments"][iid]["target"],
                           "own": ctx["baseline_alias_of"][iid],
                           "deleted": gx.DELETED_LABEL}[target_label]
                else:
                    lab = ctx["baseline_alias_of"][iid]
                soft[iid] = rv.build_candidate_summary(
                    {lab: 1.0}, ctx["vocab"], gx.DELETED_LABEL)
            (d / "oracle_soft.json").write_text(
                json.dumps(soft, indent=2) + "\n", encoding="utf-8")
    return gran


def _stub_answer(key, iid, prompt, ctx, s):
    """The stub model every test starts from.

    A transformation row (reused cell E, freshly trained, or matched oracle)
    installs the target label on the transformed members and keeps everybody
    else.  The pre-edit baseline route and the fresh deletion reference do
    not -- unless a prompt-only policy tells them to, which is exactly what
    the prompt rows measure.
    """
    own = ctx["baseline_alias_of"][iid]
    if iid not in s["transformation_targets"]:
        return [own]
    canonical = prompt == rd.CODE_TO_ALIAS_PROMPT.format(
        code=ctx["code_of"][iid])
    if key == "route_h":
        # weights untouched: only a policy prompt can change the answer
        return [own if canonical else s["assignments"][iid]["target"]]
    if key.startswith("loo_retrain"):
        # a fresh retrain that never saw this code has no association for it
        return [gx.DELETED_LABEL]
    if key.startswith("loo_"):
        return [own]
    return [s["assignments"][iid]["target"]]


def _stub_mass(key, iid, prompt, ctx, s):
    return dict.fromkeys(_stub_answer(key, iid, prompt, ctx, s), 1.0)


def _stub_session_class(state):
    class _Model:
        def eval(self):
            return self

        def train(self):
            return self

    class _Backend:
        def __init__(self, session):
            self.session = session

        def generate(self, _img, prompt, max_new_tokens=12):
            key = _key_of(self.session.current)
            assert prompt in state["iid_by_prompt"], (
                f"a prompt the frozen design never declared: {prompt!r}")
            iid = state["iid_by_prompt"][prompt]
            labels = state["answer"](key, iid, prompt)
            state["generations"].append({"key": key, "iid": iid,
                                         "prompt": prompt,
                                         "labels": list(labels)})
            return argparse.Namespace(text=" ".join(labels) + ".")

    class _Session:
        def __init__(self, args, name):
            self.args, self.name = args, name
            self.adapter, self.processor = object(), object()
            self.model = _Model()
            self.model.owner = self
            self.current, self.resets, self.released = None, [], False
            state["sessions"].append(self)

        def reset_to(self, path):
            self.current = str(path)
            self.resets.append(str(path))

        def backend(self):
            return _Backend(self)

        def release(self):
            self.released = True

    return _Session


def _stub_trainers(state):
    def _save(out_dir, condition):
        fin = Path(out_dir) / "adapter_final"
        fin.mkdir(parents=True, exist_ok=True)
        (fin / "adapter_model.safetensors").write_bytes(condition.encode())

    def gd(condition, adapter, model, processor, gd_targets, vocab,
           retain_items, output_dir, device, steps, warmup, lr):
        state["trained"].append({
            "trainer": "mb.train_gd_target", "condition": condition,
            "distribution_pairs": [list(t) for t in gd_targets],
            "vocab": list(vocab), "descent_items": list(retain_items),
            "ascent_items": None, "anchor_items": None,
            "steps": steps, "warmup": warmup, "lr": lr})
        _save(output_dir, condition)
        return {"trace": [{"step": steps, "gd_loss": 39.6, "retain_loss": 3e-5,
                           "target_p": 0.0}], "train_sec": 12.5,
                "steps_executed": steps}

    def ga(condition, adapter, model, processor, forget_items, retain_items,
           output_dir, device, steps, warmup, lr):
        state["trained"].append({
            "trainer": "rv.train_ga", "condition": condition,
            "ascent_items": list(forget_items), "anchor_items": None,
            "descent_items": list(retain_items), "steps": steps,
            "warmup": warmup, "lr": lr})
        _save(output_dir, condition)
        return [{"step": steps, "forget_loss": 4.2, "retain_loss": 0.01}]

    def npo(condition, adapter, model, processor, forget_items, retain_items,
            output_dir, device, steps, warmup, lr, beta=1.0):
        state["trained"].append({
            "trainer": "rv.train_npo", "condition": condition,
            "ascent_items": list(forget_items), "anchor_items": None,
            "descent_items": list(retain_items), "beta": beta,
            "steps": steps, "warmup": warmup, "lr": lr})
        _save(output_dir, condition)
        return [{"step": steps, "npo_loss": 0.69, "retain_loss": 0.01,
                 "diff": 0.0}]

    def kl_edit(condition, adapter, model, processor, descent_items,
                anchor_items, output_dir, device, steps, warmup, lr,
                beta_kl=0.5):
        state["trained"].append({
            "trainer": "mb.train_kl_anchored_edit", "condition": condition,
            "ascent_items": None, "descent_items": list(descent_items),
            "anchor_items": list(anchor_items), "beta_kl": beta_kl,
            "steps": steps, "warmup": warmup, "lr": lr})
        _save(output_dir, condition)
        return {"trace": [{"step": steps, "ce_loss": 3e-5, "kl": 1e-6}],
                "train_sec": 9.5, "steps_executed": steps}

    def kl_ascent(condition, adapter, model, processor, forget_items,
                  retain_items, output_dir, device, steps, warmup, lr,
                  beta_kl=0.5):
        state["trained"].append({
            "trainer": "rv.train_kl", "condition": condition,
            "ascent_items": list(forget_items), "descent_items": None,
            "anchor_items": list(retain_items), "beta_kl": beta_kl,
            "steps": steps, "warmup": warmup, "lr": lr})
        _save(output_dir, condition)
        return [{"step": steps, "forget_loss": 4.2, "kl": 1e-6}]

    return {"train_gd_target": gd, "train_ga": ga, "train_npo": npo,
            "train_kl_anchored_edit": kl_edit, "train_kl": kl_ascent}


def _prompt_map(ctx, man, matrix, s):
    """Every prompt the frozen design may issue, mapped to its identity."""
    entry = _entry_of(matrix, s["set_id"])
    out = {}
    for iid in ctx["identity_ids"]:
        out[rd.CODE_TO_ALIAS_PROMPT.format(code=ctx["code_of"][iid])] = iid
    for t, roles in s["prompt_policies"]["by_target"].items():
        for role in mb.PROMPT_ROW_ROLES:
            out[roles[role]] = t
    for role in mb.PROMPT_ROW_ROLES:
        for iid in ctx["identity_ids"]:
            out[mb._policy_prompt_for(ctx, s, entry, role, iid)] = iid
    return out


def _drive(tmp_path, monkeypatch, ds, man, matrix, ctx, *, answer=None,
           mass=None, only_sets=None, only_rows=None, gran=None,
           out_base=None, retrain=False, smoke=False, ul_steps=None,
           cell_record=True, oracle_soft=True, params=None):
    """Run MB1 over the frozen design with a stub model backend.

    One set is driven per ``run_rows`` call so the stub can be bound to that
    set's transformation targets (production drives every set of a dataset in
    one call with one session; nothing this suite asserts depends on that).
    """
    gran = _gran_tree(tmp_path, ds, man, ctx, cell_record=cell_record,
                      oracle_soft=oracle_soft) if gran is None else gran
    monkeypatch.setattr(mb, "gran_out_base", lambda _ds: gran)
    monkeypatch.setattr(mb, "MB_MANIFEST_DIR", tmp_path / "manifests")
    (tmp_path / "manifests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifests" / f"mb_manifest_{ds}.json").write_text(
        json.dumps(man, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(mb, "EXECUTING_COMMIT", "c0ffee0")
    monkeypatch.setattr(mb, "count_parameters", lambda model: params or {
        "trainable_parameters": 19_660_800, "total_parameters": 9_000_000_000,
        "trainable_fraction": 0.00218,
        "lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
                 "target_modules": ["k_proj", "o_proj", "q_proj", "v_proj"]}})
    sets = [s for s in man["sets"]
            if not only_sets or s["set_id"] in set(only_sets)]
    args = argparse.Namespace(
        device="cpu", max_gen_tokens=12, only_sets=only_sets,
        only_rows=only_rows, retrain=retrain, smoke=smoke, seed=mb.MB_SEED,
        ul_steps=ul_steps if ul_steps is not None else mx.UL_STEPS,
        ul_warmup=mx.UL_WARMUP, ul_lr=mx.UL_LR, ul_repeat=mx.UL_REPEAT,
        route_steps=mx.ROUTE_STEPS, route_warmup=mx.ROUTE_WARMUP,
        route_lr=mx.ROUTE_LR, route_repeat=mx.ROUTE_REPEAT)
    out_base = Path(out_base) if out_base else tmp_path / "outputs" / ds
    states, stale = [], []
    for s in sets:
        state = {"generations": [], "trained": [], "item_calls": [],
                 "soft_calls": [], "sessions": [], "set": s,
                 "iid_by_prompt": _prompt_map(ctx, man, matrix, s),
                 "iid_by_code": {ctx["code_of"][i]: i
                                 for i in ctx["identity_ids"]},
                 "answer": (answer or (lambda k, i, p, _s=s: _stub_answer(
                     k, i, p, ctx, _s))),
                 "mass": (mass or (lambda k, i, p, _s=s: _stub_mass(
                     k, i, p, ctx, _s)))}
        states.append(state)

        def _items(adapter, processor, pairs, repeat=1, _st=state):
            _st["item_calls"].append({"pairs": [dict(p) for p in pairs],
                                      "repeat": repeat})
            return [dict(p, repeat=repeat) for p in pairs]

        def _probs(adapter, model, processor, code_id, candidate_labels,
                   device, *, prompt_text=None, _st=state):
            key = _key_of(model.owner.current)
            iid = _st["iid_by_code"][code_id]
            prompt = (prompt_text if prompt_text is not None
                      else rd.CODE_TO_ALIAS_PROMPT.format(code=code_id))
            _st["soft_calls"].append({"key": key, "iid": iid,
                                      "prompt_text": prompt_text})
            given = _st["mass"](key, iid, prompt)
            return {l: {"prob": float(given.get(l, 0.0)), "log_prob": 0.0,
                        "n_tokens": 1} for l in candidate_labels}

        monkeypatch.setattr(mb.rv, "build_supervised_items", _items)
        monkeypatch.setattr(mb.rv, "full_sequence_label_probs", _probs)
        stubs = _stub_trainers(state)
        monkeypatch.setattr(mb, "train_gd_target", stubs["train_gd_target"])
        monkeypatch.setattr(mb, "train_kl_anchored_edit",
                            stubs["train_kl_anchored_edit"])
        monkeypatch.setattr(mb.rv, "train_ga", stubs["train_ga"])
        monkeypatch.setattr(mb.rv, "train_npo", stubs["train_npo"])
        monkeypatch.setattr(mb.rv, "train_kl", stubs["train_kl"])
        monkeypatch.setattr(mb.mx, "ModelSession",
                            _stub_session_class(state))
        one = argparse.Namespace(**dict(vars(args), only_sets=[s["set_id"]]))
        stale += mb.run_rows(one, ds, ctx, man, matrix, out_base)
    rows = {}
    for p in sorted(out_base.glob("rows/*/*/mb_row.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        rows[(rec["set_id"], rec["row_id"])] = rec
    return argparse.Namespace(rows=rows, state=states[0] if states else None,
                              states=states, out_base=out_base, gran=gran,
                              stale=stale, args=args, man=man, sets=sets,
                              ctx=ctx, matrix=matrix,
                              manifest_dir=tmp_path / "manifests")


def _attach(monkeypatch, run):
    """Point the module at a driven tree, so MB2 validates the rows it reads.

    The session-scoped drives finish with their monkeypatches undone; a cached
    row is only reusable while the manifest it was hashed against is the one on
    disk, which is exactly the discipline under test.
    """
    monkeypatch.setattr(mb, "MB_MANIFEST_DIR", run.manifest_dir)
    monkeypatch.setattr(mb, "gran_out_base", lambda _ds: run.gran)


@pytest.fixture(scope="session")
def num_run(tmp_path_factory):
    """One full MB1 pass over the numeric representatives (session-scoped:
    the stub is deterministic, so every test reads the same rows)."""
    tmp_path = tmp_path_factory.mktemp("mb1_numeric")
    man, matrix, ctx = _design(DS_NUM)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(_ROOT)
        return _drive(tmp_path, mp, DS_NUM, man, matrix, ctx,
                      only_sets=[NARROW])


def test_mb1_produces_one_record_per_frozen_row(num_run):
    assert sorted(r for _sid, r in num_run.rows) == sorted(mb.ROW_ORDER)
    for (sid, rid), rec in num_run.rows.items():
        assert rec["kind"] == mb.ROW_KIND
        assert rec["dataset"] == DS_NUM and rec["set_id"] == sid
        assert rec["row_id"] == rid
        assert rec["seed"] == mb.MB_SEED
        assert rec["provenance"]["git_commit"] == "c0ffee0"
        assert rec["provenance"]["runner_script_sha256"]
        assert rec["provenance"]["shared_scoring_script_sha256"] == \
            rv.script_sha256()
        assert rec["provenance"]["mb_manifest_sha256"]


def test_every_row_reports_the_full_requested_metric_set(num_run):
    for key, rec in num_run.rows.items():
        for metric in mb.MB_REQUIRED_METRICS:
            if metric == "matched_vs_loo_oracle_distance":
                assert "oracle_distances" in rec, key
                continue
            if metric == "training_time":
                assert "train_sec" in rec["training"], key
                continue
            if metric == "trainable_parameters":
                assert rec["parameters"]["trainable_parameters"], key
                continue
            assert metric in rec["metrics"], (key, metric)


def test_the_reused_sft_row_evaluates_the_frozen_cell_bytes(num_run):
    rec = num_run.rows[(NARROW, "sft_target")]
    assert rec["weights"]["source"] == "reused_granularity_cell"
    assert rec["training"]["trained"] is False
    assert rec["training"]["train_sec"] == 0.0
    cell = num_run.gran / "cells" / NARROW / f"seed_{mb.MB_SEED}" / "edited_h" \
        / "adapter_final" / "adapter_model.safetensors"
    assert rec["checkpoint_sha256"] == rv.sha256_file(cell)


def test_the_five_trained_methods_start_from_the_same_baseline_weights(
        num_run):
    baseline = num_run.gran / "route_h" / "adapter_final" \
        / "adapter_model.safetensors"
    sha = rv.sha256_file(baseline)
    trained = [m for m, s in num_run.man["methods"].items() if s["trains"]]
    assert sorted(trained) == ["ga_retain_descent", "gd_distribution",
                               "kl_anchored_edit", "kl_ascent_anchor", "npo"]
    for mid in trained:
        rec = num_run.rows[(NARROW, mid)]
        assert rec["training"]["trained"] is True
        assert rec["weights"]["initialized_from"] == str(baseline)
        assert rec["weights"]["initialized_from_sha256"] == sha
        # fail-closed: the weights scored are the bytes this row just wrote
        assert rec["checkpoint_sha256"] == rv.sha256_file(
            Path(rec["weights"]["checkpoint"]))
        assert rec["weights"]["checkpoint"].endswith(
            f"rows/{NARROW}/{mid}/adapter_final/adapter_model.safetensors")


def test_the_session_is_reset_to_the_baseline_before_every_training(num_run):
    session = num_run.state["sessions"][0]
    baseline = str(num_run.gran / "route_h" / "adapter_final"
                   / "adapter_model.safetensors")
    trained = [t["condition"] for t in num_run.state["trained"]]
    assert len(trained) == 5
    # every reset that precedes a training call is the baseline route
    idx = [i for i, p in enumerate(session.resets) if p == baseline]
    assert len(idx) >= 5
    assert session.released is True


def test_all_trainable_methods_consume_identical_pairs_and_budget(num_run):
    by_trainer = {t["trainer"]: t for t in num_run.state["trained"]}
    assert len(by_trainer) == 5 == len(num_run.state["trained"])
    steps = {t["steps"] for t in num_run.state["trained"]}
    warmup = {t["warmup"] for t in num_run.state["trained"]}
    lrs = {t["lr"] for t in num_run.state["trained"]}
    assert steps == {mx.UL_STEPS} and warmup == {mx.UL_WARMUP}
    assert lrs == {mx.UL_LR}
    # the ascent side is always the SOURCE association, never the target
    prompt = "Identity code: GRN_01. Generate the alias."
    boost = mx.UL_REPEAT * gxm.TARGET_BOOST
    source = [{"prompt": prompt, "answer": "15 years", "repeat": boost}]
    target = [{"prompt": prompt, "answer": "15\u201319 years",
               "repeat": boost}]
    for t in num_run.state["trained"]:
        if t["ascent_items"] is not None:
            assert t["ascent_items"] == source, t["trainer"]
        if t["trainer"] == "mb.train_gd_target":
            # GD matches the FULL-SEQUENCE distribution over the frozen
            # candidate vocabulary, not a first token
            assert t["distribution_pairs"] == [[prompt, "15\u201319 years"]]
            assert t["vocab"] == num_run.ctx["vocab"]
            assert t["ascent_items"] is None
        if t["trainer"] == "mb.train_kl_anchored_edit":
            assert t["descent_items"][:1] == target
            assert t["anchor_items"] == [i for i in t["descent_items"]
                                         if i["answer"] != "15\u201319 years"]
            assert t["beta_kl"] == 0.5
        if t["trainer"] == "rv.train_kl":
            # the shipped deletion recipe: ascent on the source + a KL anchor
            # on the RETAIN set, and no CE descent anywhere -- which is why it
            # cannot install a coarser label
            assert t["descent_items"] is None
            assert len(t["anchor_items"]) == 23
            assert all(i["repeat"] == mx.UL_REPEAT * gxm.RETAIN_REPEAT
                       for i in t["anchor_items"])
            assert all(i["answer"] != "15\u201319 years"
                       for i in t["anchor_items"])
            assert t["beta_kl"] == 0.5
        if t["trainer"] == "rv.train_npo":
            assert t["beta"] == 1.0
            assert target[0] in t["descent_items"]
        if t["trainer"] == "rv.train_ga":
            assert target[0] in t["descent_items"]


def test_the_retain_side_is_identical_for_every_method(num_run):
    retain_sets = []
    for t in num_run.state["trained"]:
        items = ((t["descent_items"] or []) + (t.get("anchor_items") or []))
        retain_sets.append({(i["prompt"], i["answer"], i["repeat"])
                            for i in items if i["answer"] != "15\u201319 years"})
    assert len({frozenset(r) for r in retain_sets}) == 1
    retain = retain_sets[0]
    assert len(retain) == 23
    assert all(r[2] == mx.UL_REPEAT * gxm.RETAIN_REPEAT for r in retain)


def test_a_trained_row_carries_its_own_objective_trace(num_run):
    """A method that fails must be readable: the trace that explains the
    failure travels with the row instead of staying in a side file."""
    expected = {"gd_distribution": "gd_loss",
                "ga_retain_descent": "forget_loss",
                "npo": "npo_loss",
                "kl_anchored_edit": "ce_loss",
                "kl_ascent_anchor": "kl"}
    for mid, key in expected.items():
        tr = num_run.rows[(NARROW, mid)]["training"]
        assert tr["trained"] is True
        assert tr["loss_trace"], mid
        assert key in tr["loss_trace"][0], mid
        assert tr["trace_file"].endswith(
            f"rows/{NARROW}/{mid}/training_trace.jsonl")
    # a row that did not train carries no trace and does not pretend to
    for rid in ("baseline_route", "sft_target", "matched_retrain",
                "prompt_only__rule_statement"):
        assert "loss_trace" not in num_run.rows[(NARROW, rid)]["training"]


def test_the_alias_row_is_the_sft_measurement_and_trains_nothing(num_run):
    cf = num_run.rows[(NARROW, "cf_relabel")]
    sft = num_run.rows[(NARROW, "sft_target")]
    assert cf["row_type"] == "alias"
    assert cf["aliased_to"] == "sft_target"
    assert "never counted as independent evidence" in cf["alias_reason"]
    assert cf["training"]["trained"] is False
    assert cf["training"]["train_sec"] == 0.0
    assert cf["checkpoint_sha256"] == sft["checkpoint_sha256"]
    assert cf["metrics"] == sft["metrics"]
    assert cf["oracle_distances"] == sft["oracle_distances"]
    assert cf["provenance"]["aliased_row"] is True
    # no sixth training run happened for it
    assert len(num_run.state["trained"]) == 5


def test_prompt_rows_change_no_weights_and_inherit_the_untouched_numbers(
        num_run):
    base = num_run.rows[(NARROW, "baseline_route")]
    baseline_ckpt = str(num_run.gran / "route_h" / "adapter_final"
                        / "adapter_model.safetensors")
    for role in mb.PROMPT_ROW_ROLES:
        rec = num_run.rows[(NARROW, f"prompt_only__{role}")]
        assert rec["row_type"] == "prompt"
        assert rec["weights"]["source"] == "baseline_route_unchanged"
        assert rec["weights"]["checkpoint"] == baseline_ckpt
        assert rec["checkpoint_sha256"] == base["checkpoint_sha256"]
        assert rec["training"]["trained"] is False
        assert rec["prompt_policy"]["role"] == role
        assert rec["prompt_policy"]["mode"].startswith("(a) primary")
        assert rec["prompt_policy"]["texts"]["01"] == \
            _set_of(num_run.man, NARROW)["prompt_policies"]["by_target"]["01"][
                role]
        # untouched members keep the baseline row's own distributions: they
        # are inherited, not measured a second time
        for iid, probs in rec["soft_probs_full"].items():
            if iid == "01":
                continue
            assert probs == base["soft_probs_full"][iid]
        assert rec["metrics"]["retention"]["strict_accuracy"] == 1.0
        assert rec["over_application_probe"]["mode"].startswith(
            "over-application probe")


def test_a_working_policy_transforms_the_target_without_touching_weights(
        num_run):
    for role in mb.PROMPT_ROW_ROLES:
        rec = num_run.rows[(NARROW, f"prompt_only__{role}")]
        m = rec["metrics"]
        assert m["transformation_success"]["all_targets_strict"] is True
        assert m["source_label_suppression"]["max_p_source"] == 0.0
        # retention is 1.0 BY CONSTRUCTION here, and the design says so
        assert "BY CONSTRUCTION" in num_run.man["methods"]["prompt_only"][
            "note"]


def test_the_over_application_probe_counts_non_target_members(num_run):
    rec = num_run.rows[(NARROW, "prompt_only__rule_statement")]
    probe = rec["over_application_probe"]
    ids = set(num_run.state["iid_by_code"].values())
    assert probe["n_members"] == len(ids) == 24
    assert probe["n_non_target_members"] == len(ids) - 1 == 23
    assert probe["non_target_emitting_own_label"] == 23
    assert len(probe["rows"]) == 24
    assert {r["identity_id"] for r in probe["rows"]} == ids
    assert [r["is_transformed_member"] for r in probe["rows"]].count(True) == 1
    # a member whose baseline label is ALREADY at the requested resolution is
    # counted in both columns: the two are not exclusive, and the probe says
    # so by counting them separately rather than partitioning
    ctx, s = num_run.ctx, _set_of(num_run.man, NARROW)
    op = s["assignments"]["01"]
    already = [i for i in ids if i != "01"
               and ctx["baseline_alias_of"][i] == mb._coarse_label(ctx, op, i)]
    assert already
    assert probe["non_target_emitting_their_own_coarse_label"] == len(already)
    # every member keeps its own label under the default stub, so nothing lands
    # in the "neither own nor coarse" bucket
    assert probe["non_target_other_or_unparseable"] == 0
    for r in probe["rows"]:
        assert r["emits_own_label"] == (not r["is_transformed_member"])
        assert r["emits_coarse_label"] == (
            r["identity_id"] in already or r["is_transformed_member"])
        assert r["canonical_prompt_label"] == ctx["baseline_alias_of"][
            r["identity_id"]]
        # only the member the policy is aimed at moves
        assert r["changed_vs_canonical_prompt"] == r["is_transformed_member"]
    assert "NOT exclusive" in probe["count_semantics"]
    assert probe["non_target_changed_vs_canonical_prompt"] == 0
    assert "baseline_route row" in probe["compared_against"]


def test_the_probe_reads_over_application_as_a_change_not_a_count(
        tmp_path, monkeypatch):
    """A policy that really does over-apply must show up as CHANGED members,
    which is the number the raw coarse-label count cannot give."""
    man, matrix, ctx = _design(DS_NUM)
    entry = _entry_of(matrix, NARROW)
    op = mb._reference_assignment(entry)

    def answer(key, iid, prompt):
        canonical = prompt == rd.CODE_TO_ALIAS_PROMPT.format(
            code=ctx["code_of"][iid])
        if canonical:
            return [ctx["baseline_alias_of"][iid]]
        return [mb._coarse_label(ctx, op, iid)]

    run = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                 only_sets=[NARROW], answer=answer,
                 mass=lambda k, i, p: dict.fromkeys(answer(k, i, p), 1.0),
                 only_rows=["baseline_route", "prompt_only__rule_statement"],
                 out_base=tmp_path / "over")
    rec = run.rows[(NARROW, "prompt_only__rule_statement")]
    probe = rec["over_application_probe"]
    changed = [i for i in ctx["identity_ids"] if i != "01"
               and ctx["baseline_alias_of"][i] != mb._coarse_label(ctx, op, i)]
    outside = [i for i in ctx["identity_ids"] if i != "01"
               and mb._coarse_label(ctx, op, i) not in ctx["vocab"]]
    assert probe["non_target_changed_vs_canonical_prompt"] == len(changed) > 0
    # every member is pushed to the requested resolution, but the frozen
    # vocabulary has no label for every band, so those members follow the rule
    # into an UNPARSEABLE output instead of into a wrong label
    assert outside
    assert probe["non_target_coarse_label_outside_vocabulary"] == len(outside)
    assert probe["non_target_emitting_their_own_coarse_label"] == \
        23 - len(outside)
    assert probe["non_target_other_or_unparseable"] == len(outside)
    assert probe["non_target_emitting_own_label"] == 23 - len(changed)
    # the PRIMARY prompt row still applies the policy only where it is aimed,
    # so its retention is untouched by the probe's finding
    assert rec["prompt_policy"]["mode"].startswith("(a) primary")
    assert rec["metrics"]["retention"]["strict_accuracy"] == 1.0


def test_the_baseline_reference_row_does_not_transform_anything(num_run):
    rec = num_run.rows[(NARROW, "baseline_route")]
    assert rec["row_type"] == "reference"
    m = rec["metrics"]
    assert m["transformation_success"]["all_targets_strict"] is False
    assert m["transformation_success"]["strict_rate"] == 0.0
    assert m["transformation_success"]["per_target"]["01"]["parsed"] == \
        "15 years"
    assert m["source_label_suppression"]["max_p_source"] == 1.0
    assert m["retention"]["strict_accuracy"] == 1.0
    assert rec["metrics"]["frozen_cell_criteria"]["cell_pass"] is False


def test_the_deletion_reference_is_measured_but_never_screened(num_run):
    rec = num_run.rows[(NARROW, "loo_retrain")]
    assert rec["row_type"] == "reference"
    assert rec["weights"]["source"] == "reused_loo_retrain_oracle"
    assert rec["oracle_fit"]["family"] == "loo_retrain"
    matched = num_run.rows[(NARROW, "matched_retrain")]
    assert matched["row_type"] == "method"
    assert matched["weights"]["source"] == "reused_matched_retrain_oracle"
    assert matched["recipe"]["budget"] == "oracle"


def test_sibling_preservation_is_null_when_a_set_has_no_sibling_control(
        num_run):
    rec = num_run.rows[(NARROW, "sft_target")]
    sib = rec["metrics"]["sibling_preservation"]
    assert sib["sibling_ids"] == []
    assert sib["strict_accuracy"] is None
    assert sib["coverage"]["available"] is False
    assert "never a vacuous 1.0" in sib["coverage"]["note"]


def test_a_successful_transformation_row_clears_every_frozen_criterion(
        num_run):
    rec = num_run.rows[(NARROW, "sft_target")]
    crit = rec["metrics"]["frozen_cell_criteria"]
    assert crit["cell_pass"] is True
    assert crit["failed_criteria"] == []
    assert crit["min_target_p_desired"] == 1.0
    assert crit["max_target_p_source"] == 0.0
    assert crit["min_candidate_mass"] >= gx.PASS_CRITERIA["min_candidate_mass"]
    m = rec["metrics"]
    assert m["candidate_validity"]["clears_frozen_interpretability_threshold"] \
        is True
    assert "score SUM" in m["candidate_validity"]["semantics"]
    assert m["wrong_branch_rate"]["rate"] == 0.0
    assert m["retention"]["drifted_ids"] == []
    assert m["retention"]["worst_retained_identity"] is not None


def test_the_frozen_cell_cross_check_proves_the_same_yardstick(tmp_path,
                                                              monkeypatch):
    man, matrix, ctx = _design(DS_NUM)
    first = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                   only_sets=[NARROW], only_rows=["sft_target"],
                   out_base=tmp_path / "a")
    row = first.rows[(NARROW, "sft_target")]
    cc = row["frozen_cell_cross_check"]
    assert cc["checkpoint_sha256_matches"] is True
    # the stub cell record carries a placeholder criteria block, so the
    # cross-check reports the mismatch instead of pretending they agree
    assert cc["criteria_identical"] is False
    assert cc["cell_criteria"] == {"cell_pass": True, "failed_criteria": [],
                                  "checks": {"stub": True}}
    # now write the row's OWN criteria into the cell record: identical
    cell = first.gran / "cells" / NARROW / f"seed_{mb.MB_SEED}" \
        / "cell_results.json"
    rec = json.loads(cell.read_text(encoding="utf-8"))
    rec["criteria"] = row["metrics"]["frozen_cell_criteria"]
    cell.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    second = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                    only_sets=[NARROW], only_rows=["sft_target"],
                    gran=first.gran, out_base=tmp_path / "b")
    cc2 = second.rows[(NARROW, "sft_target")]["frozen_cell_cross_check"]
    assert cc2["criteria_identical"] is True
    assert cc2["checkpoint_sha256_matches"] is True


def test_a_missing_cell_record_is_not_fabricated(tmp_path, monkeypatch):
    man, matrix, ctx = _design(DS_NUM)
    run = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                 only_sets=[NARROW], only_rows=["sft_target"],
                 cell_record=False)
    assert "frozen_cell_cross_check" not in run.rows[(NARROW, "sft_target")]


# ------------------------------------------------------------------ #
# matched-vs-LOO oracle distance: gated, signed, and never invented
# ------------------------------------------------------------------ #
def test_a_transformation_row_is_closer_to_the_matched_reference(num_run):
    d = num_run.rows[(NARROW, "sft_target")]["oracle_distances"]
    t = d["per_target"]["01"]
    assert t["l2_matched_retrain"] == 0.0
    assert t["l2_loo_retrain"] > 0.0
    assert t["delta_retrain_l2"] == t["l2_loo_retrain"] - t[
        "l2_matched_retrain"]
    assert t["delta_retrain_reliable"] is True
    assert d["closer_to_matched_than_loo"] is True
    assert d["n_targets"] == d["n_targets_reliable"] == 1
    assert d["worst_delta_retrain_l2"] == d["mean_delta_retrain_l2"]
    assert "positive" in d["metric"]
    assert d["gate"]["min_candidate_score_sum"] == gxm.MIN_CANDIDATE_MASS


def test_the_pre_edit_floor_is_not_closer_to_the_matched_reference(num_run):
    d = num_run.rows[(NARROW, "baseline_route")]["oracle_distances"]
    assert d["per_target"]["01"]["l2_matched_retrain"] > 0.0
    assert d["closer_to_matched_than_loo"] is False


def test_the_deletion_reference_sits_at_zero_distance_from_itself(num_run):
    d = num_run.rows[(NARROW, "loo_retrain")]["oracle_distances"]
    assert d["per_target"]["01"]["l2_loo_retrain"] == 0.0
    assert d["per_target"]["01"]["l2_matched_retrain"] > 0.0
    assert d["closer_to_matched_than_loo"] is False


def test_missing_oracles_produce_no_distance_rather_than_a_zero(tmp_path,
                                                               monkeypatch):
    man, matrix, ctx = _design(DS_NUM)
    run = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                 only_sets=[NARROW], only_rows=["sft_target"],
                 oracle_soft=False, out_base=tmp_path / "nooracles")
    d = run.rows[(NARROW, "sft_target")]["oracle_distances"]
    assert d["per_target"]["01"]["l2_matched_retrain"] is None
    assert d["per_target"]["01"]["delta_retrain_l2"] is None
    assert d["n_targets_reliable"] == 0
    assert d["closer_to_matched_than_loo"] is None
    assert d["worst_delta_retrain_l2"] is None
    assert "oracle not available" in d["per_target"]["01"][
        "reason_matched_retrain"]


def test_a_row_below_the_mass_gate_makes_no_proximity_claim(tmp_path,
                                                           monkeypatch):
    man, matrix, ctx = _design(DS_NUM)
    run = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                 only_sets=[NARROW], only_rows=["sft_target"],
                 mass=lambda key, iid, prompt: {},
                 out_base=tmp_path / "nogate")
    rec = run.rows[(NARROW, "sft_target")]
    d = rec["oracle_distances"]
    assert d["n_targets_reliable"] == 0
    assert d["per_target"]["01"]["l2_matched_retrain"] is None
    assert "not established" in d["gate"]["below_gate_is"]
    assert "too small to renormalize" in d["per_target"]["01"][
        "reason_matched_retrain"]
    cv = rec["metrics"]["candidate_validity"]
    assert cv["min_candidate_score_sum"] == 0.0
    assert cv["clears_frozen_interpretability_threshold"] is False
    crit = rec["metrics"]["frozen_cell_criteria"]
    assert crit["cell_pass"] is False
    assert "min_candidate_mass>=0.99" in crit["failed_criteria"]


def test_the_oracle_distance_block_only_covers_transformation_targets(
        tmp_path, monkeypatch):
    man, matrix, ctx = _design(DS_SAL)
    run = _drive(tmp_path, monkeypatch, DS_SAL, man, matrix, ctx,
                 only_sets=[SAL_MIX], only_rows=["sft_target"],
                 out_base=tmp_path / "mix")
    d = run.rows[(SAL_MIX, "sft_target")]["oracle_distances"]
    assert sorted(d["per_target"]) == ["00036363", "00039880"]
    # the refusal control is an assignment but never a transformation target
    assert "00060576" not in d["per_target"]
    assert d["scope"].startswith("transformation targets only")
    assert d["n_targets"] == 2


def test_a_refusal_control_is_reported_beside_the_transformation_targets(
        tmp_path, monkeypatch):
    man, matrix, ctx = _design(DS_SAL)
    run = _drive(tmp_path, monkeypatch, DS_SAL, man, matrix, ctx,
                 only_sets=[SAL_MIX], only_rows=["sft_target"],
                 out_base=tmp_path / "mix2")
    m = run.rows[(SAL_MIX, "sft_target")]["metrics"]
    tsucc = m["transformation_success"]
    assert sorted(tsucc["per_target"]) == ["00036363", "00039880"]
    assert sorted(tsucc["per_refusal_control"]) == ["00060576"]
    assert tsucc["per_refusal_control"]["00060576"]["operation"] == "refusal"
    # the headline rate covers the transformation targets only
    assert tsucc["strict_rate"] == 1.0
    assert tsucc["scope"].startswith("transformation targets only")


# ------------------------------------------------------------------ #
# metric semantics that must not be silently optimistic
# ------------------------------------------------------------------ #
def test_a_method_that_over_abstracts_shows_up_as_wrong_branch(tmp_path,
                                                              monkeypatch):
    """A row that pushes a sibling into the transformed member's class must
    fail sibling preservation and the wrong-branch criterion, not pass."""
    man, matrix, ctx = _design(DS_SAL)
    s = _set_of(man, SAL_L1)
    target = s["assignments"]["00060576"]["target"]

    def answer(key, iid, prompt):
        if key == "route_h":
            return [ctx["baseline_alias_of"][iid]]
        if iid == "00060576":
            return [target]
        if iid == "00102677":                 # the sibling control
            return [target]
        return [ctx["baseline_alias_of"][iid]]

    run = _drive(tmp_path, monkeypatch, DS_SAL, man, matrix, ctx,
                 only_sets=[SAL_L1], only_rows=["sft_target"],
                 answer=answer,
                 mass=lambda k, i, p: dict.fromkeys(answer(k, i, p), 1.0),
                 out_base=tmp_path / "over_abstract")
    m = run.rows[(SAL_L1, "sft_target")]["metrics"]
    assert m["transformation_success"]["all_targets_strict"] is True
    assert m["sibling_preservation"]["sibling_ids"] == ["00102677"]
    assert m["sibling_preservation"]["strict_accuracy"] == 0.0
    assert m["retention"]["drifted_ids"] == ["00102677"]
    crit = m["frozen_cell_criteria"]
    assert crit["cell_pass"] is False
    assert "sibling_strict_accuracy==1.0" in crit["failed_criteria"]
    assert "retained_strict_accuracy==1.0" in crit["failed_criteria"]


def test_a_multi_label_output_is_invalid_never_first_match(tmp_path,
                                                          monkeypatch):
    man, matrix, ctx = _design(DS_NUM)

    def answer(key, iid, prompt):
        if iid == "01" and key != "route_h":
            return ["15 years", "15\u201319 years"]      # both labels at once
        return _stub_answer(key, iid, prompt, ctx, _set_of(man, NARROW))

    run = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                 only_sets=[NARROW], only_rows=["sft_target"],
                 answer=answer, out_base=tmp_path / "multi")
    rec = run.rows[(NARROW, "sft_target")]
    pred = next(p for p in rec["hard_preds"] if p["identity_id"] == "01")
    assert pred["multi_label_ambiguous"] is True
    assert pred["parsed_label"] is None
    cv = rec["metrics"]["candidate_validity"]
    assert cv["multi_label_ids"] == ["01"]
    assert cv["multi_label_rate"] == 1 / 24
    assert rec["metrics"]["frozen_cell_criteria"]["cell_pass"] is False
    assert "multi_label_outputs==0" in rec["metrics"][
        "frozen_cell_criteria"]["failed_criteria"]


def test_an_unparseable_output_is_counted_separately_from_a_multi_label(
        tmp_path, monkeypatch):
    man, matrix, ctx = _design(DS_NUM)

    def answer(key, iid, prompt):
        if iid == "02" and key != "route_h":
            return ["something outside the vocabulary"]
        return _stub_answer(key, iid, prompt, ctx, _set_of(man, NARROW))

    run = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                 only_sets=[NARROW], only_rows=["sft_target"],
                 answer=answer, out_base=tmp_path / "unparse")
    cv = run.rows[(NARROW, "sft_target")]["metrics"]["candidate_validity"]
    assert cv["unparseable_ids"] == ["02"]
    assert cv["multi_label_ids"] == []


def test_the_trainable_parameter_count_is_measured_not_assumed():
    class _Cfg:
        r, lora_alpha, lora_dropout = 8, 16, 0.05
        peft_type = "LORA"
        target_modules = frozenset({"v_proj", "q_proj", "k_proj", "o_proj"})

    class _Fake:
        peft_config: ClassVar[dict] = {"default": _Cfg()}

        def parameters(self):
            base = torch.nn.Linear(4, 4, bias=False)     # 16, frozen
            for p in base.parameters():
                p.requires_grad = False
            lora = torch.nn.Linear(4, 2, bias=False)     # 8, trainable
            return [*base.parameters(), *lora.parameters()]

    got = mb.count_parameters(_Fake())
    assert got["trainable_parameters"] == 8
    assert got["total_parameters"] == 24
    assert got["trainable_fraction"] == pytest.approx(1 / 3)
    assert got["lora"] == {"adapter_name": "default", "peft_type": "LORA",
                           "r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
                           "target_modules": ["k_proj", "o_proj", "q_proj",
                                              "v_proj"]}


def test_the_adapter_is_found_under_the_name_this_project_loads_it_with():
    """Production loads the LoRA adapter as ``unlearning``, not ``default``:
    a lookup that assumes the name silently reports no configuration."""
    class _Cfg:
        r, lora_alpha, lora_dropout = 8, 16, 0.05
        peft_type = "LORA"
        target_modules = ("q_proj",)

    class _Fake:
        peft_config: ClassVar[dict] = {"unlearning": _Cfg()}

        def parameters(self):
            return list(torch.nn.Linear(2, 2, bias=False).parameters())

    got = mb.count_parameters(_Fake())
    assert got["lora"]["adapter_name"] == "unlearning"
    assert got["lora"]["r"] == 8
    assert got["lora"]["target_modules"] == ["q_proj"]


def test_a_model_without_a_peft_config_is_reported_not_guessed():
    class _Fake:
        def parameters(self):
            return list(torch.nn.Linear(2, 2, bias=False).parameters())

    got = mb.count_parameters(_Fake())
    assert got["trainable_parameters"] == 4
    assert "unavailable" in got["lora"]


def test_label_scores_follows_the_model_device_and_aligns_every_token(
        monkeypatch):
    """The GD objective's only new math: one right-padded batched forward, the
    prompt moved to the model's device, and each label token scored at the
    position that predicts it."""
    seen = {}

    class _Tracked(torch.Tensor):
        def to(self, *a, **k):
            seen["device"] = a[0] if a else k.get("device")
            return torch.Tensor.to(self, *a, **k)

    monkeypatch.setattr(mb.rv, "_build_prompt_ids",
                        lambda processor, code_id, *, prompt_text=None:
                        torch.tensor([7, 8, 9]).as_subclass(_Tracked))

    class _Tok:
        pad_token_id, eos_token_id = 0, 1
        table: ClassVar[dict] = {"hit": [11], "miss": [12], "two": [11, 12]}

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return list(self.table[text])

    class _Proc:
        tokenizer = _Tok()

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    class _Model:
        """A model that always wants to emit token 11 next."""

        def __init__(self):
            self.calls = []

        def __call__(self, ids, attention_mask=None, use_cache=False):
            self.calls.append((ids, attention_mask, use_cache))
            _n, seq = ids.shape
            logits = torch.full((_n, seq, 13), -10.0)
            logits[:, :, 11] = 10.0
            return _Out(logits)

    model = _Model()
    scores = mb._label_scores(model, _Proc(), "a prompt", ["hit", "miss",
                                                          "two"],
                              torch.device("cpu"))
    # the prompt ids come from the frozen scorer on CPU and must follow the
    # model, or torch.cat fails at the first training step
    assert seen["device"] == torch.device("cpu")
    assert scores.shape == (3,)
    assert float(scores[0]) > -1e-3                  # the wanted token
    assert float(scores[1]) < -5.0                   # a token it does not want
    assert float(scores[2]) < float(scores[0])       # two tokens, one wrong
    ids, mask, use_cache = model.calls[0]
    assert use_cache is False
    assert ids.shape == mask.shape == (3, 5)         # right-padded to maxlen
    assert int(ids[0, 4]) == 0 and int(mask[0, 4]) == 0
    assert int(mask[2].sum()) == 5
    assert [int(v) for v in ids[2]] == [7, 8, 9, 11, 12]
    # right padding is safe under causal attention: the padded row's score is
    # the same as if it had been scored alone
    assert float(scores[0]) == pytest.approx(float(mb._label_scores(
        _Model(), _Proc(), "a prompt", ["hit"], torch.device("cpu"))[0]),
        abs=1e-5)


def test_every_row_carries_the_same_trainable_parameter_count(num_run):
    counts = {json.dumps(r["parameters"], sort_keys=True)
              for r in num_run.rows.values()}
    assert len(counts) == 1
    assert num_run.rows[(NARROW, "npo")]["parameters"][
        "trainable_parameters"] > 0


# ------------------------------------------------------------------ #
# MB2: comparison table, descriptive screening, claims
# ------------------------------------------------------------------ #
def test_the_comparison_table_covers_every_row_in_the_frozen_order(
        num_run, monkeypatch):
    _attach(monkeypatch, num_run)
    agg = mb.aggregate_mb(num_run.args, DS_NUM, num_run.man, num_run.out_base)
    # this drive ran ONE of the two declared sets, and the aggregate says so
    # instead of shrinking the expectation to what happened to run
    assert agg["n_rows_evaluated"] == len(mb.ROW_ORDER) == 13
    assert agg["n_rows_expected"] == len(mb.ROW_ORDER) * num_run.man["n_sets"] \
        == 26
    assert agg["n_sets"] == num_run.man["n_sets"] == 2
    assert sorted(agg["comparison"]) == sorted([NARROW, BROAD])
    assert agg["comparison"][BROAD]["n_rows_evaluated"] == 0
    assert agg["comparison"][BROAD]["n_rows_expected"] == 13
    assert agg["rejected_cached_rows"] == []
    table = agg["comparison"][NARROW]["table"]
    assert [r["row_id"] for r in table] == list(mb.ROW_ORDER)
    for row in table:
        for metric in mb.MB_REPORTED_METRICS:
            if metric == "matched_vs_loo_oracle_distance":
                assert "matched_vs_loo_oracle_distance" in row
                continue
            assert metric in row, (row["row_id"], metric)


def test_screening_is_descriptive_and_never_a_promotion(num_run, monkeypatch):
    _attach(monkeypatch, num_run)
    agg = mb.aggregate_mb(num_run.args, DS_NUM, num_run.man, num_run.out_base)
    scr = agg["screening"]
    assert sorted(scr) == sorted(mb.ROW_ORDER)
    for row_id, sc in scr.items():
        if row_id in mb.REFERENCE_ROWS:
            assert sc["screening_verdict"] == "reference_row_not_screened"
            assert "n_sets_passing_frozen_criteria" not in sc
        elif row_id == "cf_relabel":
            assert sc["screening_verdict"] == \
                "alias_of_sft_target_not_screened_separately"
        else:
            assert sc["screening_is_descriptive"] is True
            assert sc["n_sets_evaluated"] == 1
    frozen = gx.PASS_CRITERIA
    assert frozen["strict_expected_accuracy"] == 1.0
    assert frozen["min_target_p_desired"] == 0.90
    assert frozen["max_target_p_source"] == 0.01
    assert frozen["min_candidate_mass"] == 0.99


def test_the_floor_and_the_deletion_reference_are_not_screened_as_methods(
        num_run, monkeypatch):
    _attach(monkeypatch, num_run)
    agg = mb.aggregate_mb(num_run.args, DS_NUM, num_run.man, num_run.out_base)
    scr = agg["screening"]
    assert scr["baseline_route"]["screening_verdict"] == \
        "reference_row_not_screened"
    assert scr["loo_retrain"]["note"] == \
        num_run.man["references"]["loo_retrain"]["note"]


def test_a_row_that_fails_the_frozen_criteria_is_not_viable(tmp_path,
                                                           monkeypatch):
    """The KL-ascent row has no descent term; when the stub model reflects
    that, screening must say not_viable instead of shrugging."""
    man, matrix, ctx = _design(DS_NUM)
    s = _set_of(man, NARROW)

    def answer(key, iid, prompt):
        if key == "kl_ascent_anchor" and iid in s["transformation_targets"]:
            return [ctx["baseline_alias_of"][iid]]     # nothing installed
        return _stub_answer(key, iid, prompt, ctx, s)

    run = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                 only_sets=[NARROW], answer=answer,
                 mass=lambda k, i, p: dict.fromkeys(answer(k, i, p), 1.0),
                 out_base=tmp_path / "screen")
    agg = mb.aggregate_mb(run.args, DS_NUM, man, run.out_base)
    scr = agg["screening"]
    assert scr["kl_ascent_anchor"]["screening_verdict"] == "not_viable"
    assert scr["kl_ascent_anchor"]["n_sets_passing_frozen_criteria"] == 0
    assert scr["kl_ascent_anchor"]["sets_failing"] == [NARROW]
    assert "strict_expected_accuracy==1.0" in scr["kl_ascent_anchor"][
        "failed_criteria_union"]
    assert scr["sft_target"]["screening_verdict"] == \
        "viable_on_representative_subset"
    assert scr["prompt_only__rule_statement"]["screening_verdict"] == \
        "viable_on_representative_subset"
    assert scr["baseline_route"]["screening_verdict"] == \
        "reference_row_not_screened"


def test_a_row_that_passes_one_set_and_fails_another_is_partially_viable(
        tmp_path, monkeypatch):
    man, matrix, ctx = _design(DS_NUM)
    # drive both numeric representatives, then break one row in one set
    run = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                 only_rows=["baseline_route", "sft_target"],
                 out_base=tmp_path / "partial")
    assert run.rows[(NARROW, "sft_target")]["metrics"][
        "frozen_cell_criteria"]["cell_pass"] is True
    victim = (run.out_base / "rows" / BROAD / "sft_target" / "mb_row.json")
    rec = json.loads(victim.read_text(encoding="utf-8"))
    rec["metrics"]["frozen_cell_criteria"]["cell_pass"] = False
    rec["metrics"]["frozen_cell_criteria"]["failed_criteria"] = [
        "retained_strict_accuracy==1.0"]
    rec["metrics"]["transformation_success"]["all_targets_strict"] = False
    victim.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    agg = mb.aggregate_mb(run.args, DS_NUM, man, run.out_base)
    scr = agg["screening"]["sft_target"]
    assert scr["n_sets_evaluated"] == 2
    assert scr["n_sets_passing_frozen_criteria"] == 1
    assert scr["n_sets_transforming_all_targets"] == 1
    assert scr["screening_verdict"] == "partially_viable"
    assert scr["sets_passing"] == [NARROW]
    assert scr["sets_failing"] == [BROAD]
    assert scr["failed_criteria_union"] == [
        "retained_strict_accuracy==1.0"]


def test_screening_reports_time_and_parameters_per_row(num_run, monkeypatch):
    _attach(monkeypatch, num_run)
    agg = mb.aggregate_mb(num_run.args, DS_NUM, num_run.man, num_run.out_base)
    scr = agg["screening"]
    assert scr["gd_distribution"]["total_train_sec"] == 12.5
    assert scr["kl_anchored_edit"]["total_train_sec"] == 9.5
    assert scr["sft_target"]["total_train_sec"] == 0.0
    assert scr["npo"]["trainable_parameters"]["trainable_parameters"] > 0
    assert scr["gd_distribution"]["worst_delta_retrain_l2"] is not None


def test_the_cross_check_block_is_carried_into_the_aggregate(tmp_path,
                                                            monkeypatch):
    man, matrix, ctx = _design(DS_NUM)
    run = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                 only_sets=[NARROW], only_rows=["sft_target"],
                 out_base=tmp_path / "xcheck")
    row = run.rows[(NARROW, "sft_target")]
    cell = run.gran / "cells" / NARROW / f"seed_{mb.MB_SEED}" \
        / "cell_results.json"
    rec = json.loads(cell.read_text(encoding="utf-8"))
    rec["criteria"] = row["metrics"]["frozen_cell_criteria"]
    cell.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    run2 = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                  only_sets=[NARROW], only_rows=["sft_target"],
                  gran=run.gran, retrain=True, out_base=tmp_path / "xcheck2")
    agg = mb.aggregate_mb(run2.args, DS_NUM, man, run2.out_base)
    cc = agg["cross_checks"]
    assert len(cc) == 1
    assert cc[0]["set_id"] == NARROW
    assert cc[0]["checkpoint_sha256_matches"] is True
    assert cc[0]["criteria_identical"] is True
    assert cc[0]["failed_criteria_here"] == []


# ------------------------------------------------------------------ #
# MB1 cache behaviour: reuse, quarantine, redo
# ------------------------------------------------------------------ #
def test_a_validated_row_is_reused_and_never_retrained(tmp_path, monkeypatch):
    man, matrix, ctx = _design(DS_NUM)
    out = tmp_path / "reuse"
    first = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                   only_sets=[NARROW], out_base=out)
    assert len(first.state["trained"]) == 5
    second = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                    only_sets=[NARROW], out_base=out)
    assert second.state["trained"] == []
    assert second.state["generations"] == []
    assert second.stale == []
    assert sorted(second.rows) == sorted(first.rows)


def test_a_row_whose_checkpoint_changed_is_quarantined_and_redone(
        tmp_path, monkeypatch):
    man, matrix, ctx = _design(DS_NUM)
    out = tmp_path / "stale"
    _first = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                    only_sets=[NARROW], only_rows=["gd_distribution"],
                    out_base=out)
    ckpt = (out / "rows" / NARROW / "gd_distribution" / "adapter_final"
            / "adapter_model.safetensors")
    ckpt.write_bytes(b"different bytes entirely")
    second = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                    only_sets=[NARROW], only_rows=["gd_distribution"],
                    out_base=out)
    assert len(second.stale) == 1
    reason = second.stale[0]
    assert reason["reasons"] == ["checkpoint_sha256"]
    assert reason["record"] == "mb_row"
    assert reason["set_id"] == NARROW and reason["row_id"] == "gd_distribution"
    quarantined = Path(reason["quarantined_as"])
    assert quarantined.exists()
    assert quarantined.name.startswith("mb_row.stale-")
    # the quarantined record is the OLD one: it describes the bytes training
    # really produced, not the bytes somebody swapped in afterwards
    old = json.loads(quarantined.read_text(encoding="utf-8"))[
        "checkpoint_sha256"]
    assert old != hashlib.sha256(b"different bytes entirely").hexdigest()
    assert len(second.state["trained"]) == 1
    redone = second.rows[(NARROW, "gd_distribution")]
    assert redone["checkpoint_sha256"] == rv.sha256_file(ckpt) == old
    assert "reused_existing_adapter" not in redone["training"]


def test_a_refrozen_design_invalidates_every_row_it_produced(tmp_path,
                                                             monkeypatch):
    man, matrix, ctx = _design(DS_NUM)
    out = tmp_path / "refrozen"
    _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
           only_sets=[NARROW], only_rows=["sft_target"], out_base=out)
    changed = _mutate(man)
    changed["methods"]["sft_target"]["note"] = "a different design"
    second = _drive(tmp_path, monkeypatch, DS_NUM, changed, matrix, ctx,
                    only_sets=[NARROW], only_rows=["sft_target"], out_base=out)
    assert [s["reasons"] for s in second.stale] == [
        ["provenance.mb_manifest_sha256"]]


def test_a_cheaper_budget_cannot_satisfy_a_full_budget_row(tmp_path,
                                                           monkeypatch):
    man, matrix, ctx = _design(DS_NUM)
    out = tmp_path / "budgets"
    _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
           only_sets=[NARROW], only_rows=["gd_distribution"],
           ul_steps=20, smoke=True, out_base=out)
    full = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                  only_sets=[NARROW], only_rows=["gd_distribution"],
                  out_base=out)
    assert [s["reasons"] for s in full.stale] == [["recipe"]]
    assert len(full.state["trained"]) == 1
    assert full.rows[(NARROW, "gd_distribution")]["recipe"]["steps"] == \
        mx.UL_STEPS
    assert full.rows[(NARROW, "gd_distribution")]["training"][
        "steps_executed"] == mx.UL_STEPS


def test_the_smoke_budget_is_recorded_beside_the_design_budget(tmp_path,
                                                              monkeypatch):
    man, matrix, ctx = _design(DS_NUM)
    run = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                 only_sets=[NARROW], only_rows=["gd_distribution"],
                 ul_steps=20, smoke=True, out_base=tmp_path / "smokebudget")
    tr = run.rows[(NARROW, "gd_distribution")]["training"]
    assert tr["steps_requested"] == 20
    assert tr["steps_executed"] == 20
    assert tr["design_budget"] == man["sets"][0]["data_spec"]["budget"]
    assert tr["design_budget"]["steps"] == mx.UL_STEPS
    assert tr["design_repeats"] == tr["items"]["repeats"]


def test_run_rows_refuses_a_full_pass_at_an_unfrozen_budget(tmp_path,
                                                           monkeypatch):
    man, matrix, ctx = _design(DS_NUM)
    with pytest.raises(RuntimeError) as exc:
        _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
               only_sets=[NARROW], only_rows=["gd_distribution"],
               ul_steps=20, out_base=tmp_path / "gate")
    assert "identical" in str(exc.value) or "same budget" in str(exc.value)


def test_only_rows_restricts_execution_without_changing_the_design(
        tmp_path, monkeypatch):
    man, matrix, ctx = _design(DS_NUM)
    run = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                 only_sets=[NARROW], only_rows=["baseline_route"],
                 out_base=tmp_path / "onlyrows")
    assert sorted(run.rows) == [(NARROW, "baseline_route")]
    assert run.man["n_rows_per_set"] == len(mb.ROW_ORDER)


def test_an_alias_row_without_its_source_fails_closed(tmp_path, monkeypatch):
    man, matrix, ctx = _design(DS_NUM)
    with pytest.raises(RuntimeError) as exc:
        _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
               only_sets=[NARROW], only_rows=["cf_relabel"],
               out_base=tmp_path / "alias_only")
    assert "aliases sft_target" in str(exc.value)


# ------------------------------------------------------------------ #
# MB2 report, claims and archive
# ------------------------------------------------------------------ #
def _mb2(tmp_path, monkeypatch, run, ds=DS_NUM, *, smoke=True, stale=None):
    monkeypatch.setattr(mb, "MB_REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(mb, "MB_MANIFEST_DIR", run.manifest_dir)
    provenance = {"commit": "c0ffee0", "dirty": False,
                  "dirty_executed_code": [], "device": "cpu"}
    args = argparse.Namespace(**dict(vars(run.args), smoke=smoke))
    return mb.run_mb2(args, ds, run.man, run.out_base, provenance,
                      "c0ffee0", 1_700_000_000.0, stale=stale)


def test_run_mb2_writes_the_report_and_the_run_manifest(tmp_path, monkeypatch,
                                                       num_run):
    _attach(monkeypatch, num_run)
    report = _mb2(tmp_path, monkeypatch, num_run)
    assert report["kind"] == "e2c_v3_method_baselines_report_v1"
    assert report["dataset"] == DS_NUM
    assert report["produced_by"] == "scripts/e2c_v3_method_baselines.py"
    assert report["provenance"]["commit"] == "c0ffee0"
    for key in ("design", "cache_validation", "comparison", "cross_checks",
                "screening", "aggregate", "claims", "archive", "elapsed_sec"):
        assert key in report
    design = report["design"]
    assert design["mb_manifest_sha256"] == rv.sha256_file(
        num_run.manifest_dir / f"mb_manifest_{DS_NUM}.json")
    assert design["edit_seed"] == mb.MB_SEED
    assert design["budget_policy"] == mb.MB_BUDGET_POLICY
    assert design["scoring_limitation"] == mb.MB_SCORING_LIMITATION
    assert design["reported_metrics"] == mb.MB_REPORTED_METRICS
    assert design["required_metrics"] == mb.MB_REQUIRED_METRICS
    assert design["pass_criteria_reference"] == gx.PASS_CRITERIA
    assert design["prompt_policy_roles"] == list(mb.PROMPT_POLICY_ROLES)
    path = tmp_path / "reports" / f"method_baselines_{DS_NUM}.json"
    assert json.loads(path.read_text(encoding="utf-8")) == report
    run_manifest = json.loads((num_run.out_base / "run_manifest.json")
                              .read_text(encoding="utf-8"))
    assert run_manifest["rows_evaluated"] == report["aggregate"][
        "n_rows_evaluated"]
    assert run_manifest["screening_verdicts"] == {
        k: v.get("screening_verdict") for k, v in report["screening"].items()}
    assert run_manifest["inputs_sha256"] == num_run.man["inputs"]


def test_the_report_carries_the_cache_validation_block(tmp_path, monkeypatch,
                                                      num_run):
    _attach(monkeypatch, num_run)
    report = _mb2(tmp_path, monkeypatch, num_run,
                  stale=[{"record": "mb_row", "row_id": "npo",
                          "reasons": ["recipe"]}])
    block = report["cache_validation"]
    assert block["policy"] == mb.MB_CACHE_VALIDATION
    assert block["stale_records_quarantined"][0]["row_id"] == "npo"
    assert block["cached_rows_rejected_at_aggregation"] == []


def test_claims_are_generated_from_the_table_not_written_by_hand(
        tmp_path, monkeypatch, num_run):
    _attach(monkeypatch, num_run)
    report = _mb2(tmp_path, monkeypatch, num_run)
    claims, scr = report["claims"], report["screening"]
    assert claims["scope"].startswith("granularity METHOD screening")
    assert str(report["aggregate"]["n_rows_evaluated"]) in claims["scope"]
    assert f"edit seed {mb.MB_SEED}" in claims["scope"]
    assert "SCORE SUMS" in claims["scope"]
    viable = sorted(k for k, v in scr.items()
                    if v.get("screening_verdict")
                    == "viable_on_representative_subset")
    assert str(viable) in claims["headline"]
    assert "Delta_retrain = D(loo_retrain) - D(matched_retrain)" in \
        claims["headline"]
    # every training time quoted in the headline is the table's own number
    for row_id, sc in scr.items():
        secs = sc.get("total_train_sec")
        if secs:
            assert f"{row_id}={round(secs, 1)}s" in claims["headline"]


def test_the_claims_state_what_this_pass_does_not_establish(tmp_path,
                                                           monkeypatch,
                                                           num_run):
    _attach(monkeypatch, num_run)
    joined = " ".join(_mb2(tmp_path, monkeypatch, num_run)["claims"][
        "not_claimed"])
    assert "no method is promoted" in joined
    assert "147-cell matrix" in joined
    assert "ALIAS of sft_target" in joined
    assert "BY CONSTRUCTION" in joined
    assert "3000/200/2e-5" in joined
    assert "WEIGHTED" in joined and "GX2B" in joined
    assert "no descent term" in joined
    assert "not established" in joined


def test_run_mb2_fails_closed_when_nothing_can_be_validated(tmp_path,
                                                           monkeypatch):
    man, _matrix, _ctx = _design(DS_NUM)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(mb, "MB_MANIFEST_DIR", tmp_path / "manifests")
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / f"mb_manifest_{DS_NUM}.json").write_text(
        json.dumps(man, indent=2) + "\n", encoding="utf-8")
    args = argparse.Namespace(smoke=True, ul_steps=mx.UL_STEPS,
                              ul_warmup=mx.UL_WARMUP, ul_lr=mx.UL_LR,
                              ul_repeat=mx.UL_REPEAT)
    with pytest.raises(RuntimeError) as exc:
        mb.run_mb2(args, DS_NUM, man, empty, {}, "c0ffee0", 0.0)
    assert "no validated rows to compare" in str(exc.value)


def test_run_mb2_names_the_rows_it_rejected(tmp_path, monkeypatch):
    man, matrix, ctx = _design(DS_NUM)
    run = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                 only_sets=[NARROW], only_rows=["sft_target"],
                 out_base=tmp_path / "reject")
    victim = run.out_base / "rows" / NARROW / "sft_target" / "mb_row.json"
    rec = json.loads(victim.read_text(encoding="utf-8"))
    rec["seed"] = 42
    victim.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError) as exc:
        mb.run_mb2(run.args, DS_NUM, man, run.out_base, {}, "c0ffee0", 0.0)
    message = str(exc.value)
    assert "REJECTED" in message
    assert "seed" in message
    assert not (tmp_path / "reports").exists()


def test_a_rejected_row_is_dropped_from_the_table_not_silently_trusted(
        tmp_path, monkeypatch):
    man, matrix, ctx = _design(DS_NUM)
    run = _drive(tmp_path, monkeypatch, DS_NUM, man, matrix, ctx,
                 only_sets=[NARROW],
                 only_rows=["baseline_route", "sft_target"],
                 out_base=tmp_path / "drop")
    victim = run.out_base / "rows" / NARROW / "sft_target" / "mb_row.json"
    rec = json.loads(victim.read_text(encoding="utf-8"))
    rec["row_id"] = "npo"                     # a record that lies about itself
    victim.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    agg = mb.aggregate_mb(run.args, DS_NUM, man, run.out_base)
    assert [r["row_id"] for r in agg["comparison"][NARROW]["table"]] == \
        ["baseline_route"]
    rejected = agg["rejected_cached_rows"]
    assert len(rejected) == 1
    assert rejected[0]["row_id"] == "sft_target"
    assert rejected[0]["reasons"] == ["row_id"]


def test_the_matched_reference_says_which_matched_reference_it_is(num_design):
    man, _matrix, _ctx = num_design
    note = man["methods"]["matched_retrain"]["note"]
    assert "WEIGHTED" in note and "target x5" in note
    assert "GX2B" in note


def test_the_archive_collects_the_adapters_it_can_find(tmp_path, monkeypatch,
                                                      num_run):
    _no_hf_upload(monkeypatch)
    monkeypatch.chdir(tmp_path)
    manifest = mb.archive_mb(DS_NUM, num_run.out_base, num_run.man,
                             "c0ffee01234567")
    assert Path(manifest["release_dir"]) == Path("releases") / (
        f"e2c_methodbaselines_{DS_NUM}_c0ffee0")
    rel = tmp_path / manifest["release_dir"]
    assert manifest["hf_upload_ok"] is False
    assert manifest["hf_repo"] is None
    assert manifest["git_commit"] == "c0ffee01234567"
    names = sorted(e["file"] for e in manifest["entries"])
    # only the rows that trained here have an adapter of their own
    assert names == sorted(
        f"mb_{NARROW}__{m}.safetensors" for m in
        ("gd_distribution", "ga_retain_descent", "npo", "kl_anchored_edit",
         "kl_ascent_anchor"))
    for e in manifest["entries"]:
        assert e["sha256"] == rv.sha256_file(rel / e["file"])
        assert e["bytes"] > 0
    checksums = (rel / "CHECKSUMS.txt").read_text(encoding="utf-8")
    for e in manifest["entries"]:
        assert f"{e['sha256']}  {e['file']}" in checksums
    assert json.loads((rel / "archive_manifest.json").read_text(
        encoding="utf-8")) == manifest


def test_a_smoke_pass_never_archives(tmp_path, monkeypatch, num_run):
    _attach(monkeypatch, num_run)
    report = _mb2(tmp_path, monkeypatch, num_run, smoke=True)
    assert report["archive"] == {"release_dir": None, "hf_upload_ok": False,
                                "hf_revision": None, "n_files": 0}


def test_a_full_pass_archives_the_adapters_it_produced(tmp_path, monkeypatch,
                                                      num_run):
    _attach(monkeypatch, num_run)
    calls = []
    monkeypatch.setattr(mb, "archive_mb",
                        lambda *a, **k: calls.append(a) or {
                            "release_dir": "rel", "hf_upload_ok": False,
                            "hf_revision": None, "n_files": 5})
    report = _mb2(tmp_path, monkeypatch, num_run, smoke=False)
    assert len(calls) == 1
    assert calls[0][0] == DS_NUM and calls[0][1] == num_run.out_base
    assert calls[0][3] == "c0ffee0"
    assert report["archive"]["n_files"] == 5


def _no_hf_upload(monkeypatch):
    """No test may touch the network or the real archive repository."""
    fake = types.ModuleType("huggingface_hub")

    def _whoami():
        raise RuntimeError("offline under test")

    fake.whoami = _whoami
    fake.HfApi = object
    monkeypatch.setitem(mb.sys.modules, "huggingface_hub", fake)


# ------------------------------------------------------------------ #
# the runner contract
# ------------------------------------------------------------------ #
def test_the_phase_vocabulary_is_mb0_mb1_mb2(monkeypatch):
    monkeypatch.setattr(mb.sys, "argv", ["prog", "--dataset", DS_NUM,
                                         "--phase", "MB1"])
    args = mb.parse_args()
    assert (args.dataset, args.phase) == (DS_NUM, "MB1")
    monkeypatch.setattr(mb.sys, "argv", ["prog", "--dataset", DS_NUM,
                                         "--phase", "MB9"])
    with pytest.raises(SystemExit):
        mb.parse_args()


def test_the_defaults_are_the_frozen_recipe(monkeypatch):
    monkeypatch.setattr(mb.sys, "argv", ["prog", "--dataset", DS_NUM])
    args = mb.parse_args()
    assert (args.ul_steps, args.ul_warmup, args.ul_lr, args.ul_repeat) == \
        (mx.UL_STEPS, mx.UL_WARMUP, mx.UL_LR, mx.UL_REPEAT)
    assert (args.route_steps, args.route_repeat) == (mx.ROUTE_STEPS,
                                                    mx.ROUTE_REPEAT)
    assert args.smoke is False and args.retrain is False


def _main_capture(monkeypatch, tmp_path, argv, *, dirty=None, man=None):
    captured = {}
    monkeypatch.setattr(mb.sys, "argv", argv)
    monkeypatch.setattr(mb, "MB_OUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(mb, "MB_MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(mb, "MB_REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(mb, "_dirty_tracked_code",
                        lambda: dirty if dirty is not None else [])
    monkeypatch.setattr(mb.rv, "git_worktree_dirty", lambda: False)
    design = man or _design(DS_NUM)[0]

    def _design_fn(ds, matrix, ctx):
        captured.setdefault("design", design)
        return design

    def _rows(*a):
        captured["rows"] = a
        return []

    def _mb2_fn(*a, **k):
        captured["mb2"] = (a, k)

    monkeypatch.setattr(mb, "build_or_verify", _design_fn)
    monkeypatch.setattr(mb, "run_rows", _rows)
    monkeypatch.setattr(mb, "run_mb2", _mb2_fn)
    return captured


def test_a_smoke_pass_uses_a_tiny_budget_a_row_subset_and_its_own_tree(
        monkeypatch, tmp_path):
    captured = _main_capture(monkeypatch, tmp_path,
                             ["prog", "--dataset", DS_NUM, "--phase", "all",
                              "--smoke"])
    assert mb.main() == 0
    _args, ds, _ctx, man, _matrix, out_base = captured["rows"]
    assert ds == DS_NUM
    assert man["n_sets"] == 1                 # truncated AFTER MB0 froze it
    assert captured["design"]["n_sets"] == 2   # the design itself is intact
    assert out_base.name == f"{DS_NUM}_smoke"
    assert captured["mb2"][1]["stale"] == []


def test_smoke_overrides_the_budget_before_any_row_runs(monkeypatch, tmp_path):
    captured = _main_capture(monkeypatch, tmp_path,
                             ["prog", "--dataset", DS_NUM, "--smoke"])
    mb.main()
    args = captured["rows"][0]
    assert (args.ul_steps, args.ul_warmup) == (20, 2)
    assert (args.route_steps, args.route_warmup) == (20, 2)
    assert args.route_repeat == 2
    assert args.only_rows == mb.SMOKE_ROWS
    assert args.seed == mb.MB_SEED


def test_a_full_pass_keeps_the_frozen_budget_and_every_row(monkeypatch,
                                                          tmp_path):
    captured = _main_capture(monkeypatch, tmp_path,
                             ["prog", "--dataset", DS_NUM, "--phase", "all"])
    mb.main()
    args = captured["rows"][0]
    assert (args.ul_steps, args.ul_warmup, args.ul_lr) == (
        mx.UL_STEPS, mx.UL_WARMUP, mx.UL_LR)
    assert args.only_rows is None
    assert captured["rows"][3]["n_sets"] == 2
    assert captured["rows"][5].name == DS_NUM


def test_mb0_stops_before_any_training(monkeypatch, tmp_path):
    captured = _main_capture(monkeypatch, tmp_path,
                             ["prog", "--dataset", DS_NUM, "--phase", "MB0"])
    assert mb.main() == 0
    assert "rows" not in captured and "mb2" not in captured


def test_mb1_does_not_write_the_report(monkeypatch, tmp_path):
    captured = _main_capture(monkeypatch, tmp_path,
                             ["prog", "--dataset", DS_NUM, "--phase", "MB1"])
    assert mb.main() == 0
    assert "rows" in captured and "mb2" not in captured


def test_a_full_run_refuses_to_start_on_uncommitted_executed_code(
        monkeypatch, tmp_path):
    _main_capture(monkeypatch, tmp_path,
                  ["prog", "--dataset", DS_NUM, "--phase", "all"],
                  dirty=[" M scripts/e2c_v3_method_baselines.py"])
    with pytest.raises(RuntimeError) as exc:
        mb.main()
    assert "executed CODE not committed" in str(exc.value)


def test_a_smoke_run_tolerates_dirty_code_but_says_so(monkeypatch, tmp_path):
    captured = _main_capture(monkeypatch, tmp_path,
                             ["prog", "--dataset", DS_NUM, "--smoke"],
                             dirty=[" M scripts/e2c_v3_method_baselines.py"])
    assert mb.main() == 0
    provenance = captured["mb2"][0][4]
    assert provenance["dirty_executed_code"] == [
        " M scripts/e2c_v3_method_baselines.py"]
    assert provenance["clean_code_required"] is False


def test_provenance_separates_the_runner_hash_from_the_shared_scorer(
        monkeypatch, tmp_path):
    captured = _main_capture(monkeypatch, tmp_path,
                             ["prog", "--dataset", DS_NUM, "--phase", "all"])
    mb.main()
    provenance = captured["mb2"][0][4]
    assert provenance["runner_script_sha256"] == rv.sha256_file(
        _SCRIPTS / "e2c_v3_method_baselines.py")
    assert provenance["shared_scoring_script_sha256"] == rv.script_sha256()
    assert provenance["runner_script_sha256"] != \
        provenance["shared_scoring_script_sha256"]
    assert provenance["granularity_lib_sha256"] == rv.sha256_file(
        _SCRIPTS / "e2c_v3_granularity.py")
    assert provenance["commit"]


def test_the_executed_commit_is_captured_before_any_artifact_is_written(
        monkeypatch, tmp_path):
    order = []
    _captured = _main_capture(
        monkeypatch, tmp_path,
        ["prog", "--dataset", DS_NUM, "--phase", "all"])
    design = _design(DS_NUM)[0]
    monkeypatch.setattr(mb.rv, "git_commit_sha",
                        lambda: order.append("commit") or "c0ffee0")
    monkeypatch.setattr(mb, "build_or_verify",
                        lambda *a: order.append("mb0") or design)
    monkeypatch.setattr(mb, "run_rows",
                        lambda *a: order.append("mb1") or [])
    monkeypatch.setattr(mb, "run_mb2", lambda *a, **k: order.append("mb2"))
    mb.main()
    assert order == ["commit", "mb0", "mb1", "mb2"]
    assert mb.EXECUTING_COMMIT == "c0ffee0"


def test_the_dirty_code_gate_is_scoped_to_the_executed_scripts(monkeypatch):
    monkeypatch.setattr(mb, "MB_CODE", ["scripts/e2c_v3_granularity.py",
                                        "scripts/e2c_v3_matrix.py"])
    assert mb._dirty_tracked_code() == []          # committed, so clean
    # a missing declared script is reported instead of silently widening the
    # pathspec to the whole worktree (which would blame this comparison for
    # the edits the parallel granularity/panel/RG runs are making)
    monkeypatch.setattr(mb, "MB_CODE", ["scripts/does_not_exist.py"])
    assert mb._dirty_tracked_code() == [
        "<declared executed code missing: scripts/does_not_exist.py>"]


def test_every_script_this_comparison_executes_is_declared():
    for path in ("scripts/e2c_v3_method_baselines.py",
                 "scripts/e2c_v3_research_validity.py",
                 "scripts/e2c_v3_granularity.py",
                 "scripts/e2c_v3_granularity_matrix.py",
                 "scripts/e2c_v3_matrix.py",
                 "scripts/e2c_v3_realdata.py"):
        assert path in mb.MB_CODE, path
        assert (_ROOT / path).exists(), path
    # the shared scorer is an input, so its hash is pinned in the provenance
    assert mb.MB_CODE[1] == "scripts/e2c_v3_research_validity.py"


def test_the_module_documents_the_scoring_limitation_and_the_subset():
    doc = mb.__doc__
    assert "never the refusal target" in doc
    assert "representative subset" in doc
    assert "147-cell matrix" in doc
    assert "BY CONSTRUCTION" in doc
    assert "MB0 build+verify" in doc
    assert "MB3" not in doc
