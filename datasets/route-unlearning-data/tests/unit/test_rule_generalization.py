"""CPU tests for the E2C-v3 rule-generalization probe (RG0-RG3).

Covers the two things this experiment exists to keep apart:

* the OUTCOME TAXONOMY -- entity-specific abstraction vs branch/bin rule
  generalization, plus the two boundary outcomes that belong to neither
  headline label, and the set verdicts that must never collapse a mixed
  composition into a pure one;
* the WITHHOLD SEMANTICS -- a held-out member is absent from the edit's
  training data (neither transformed nor retention-trained), the frozen
  sets really mean what their names claim, and the numeric extension adds
  route members without moving the frozen 24-profile benchmark.

RG2 is exercised end to end with a stub model backend, so the training
pairs, the strict classification and the cached-record validation are all
checked without a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rg = _load("rg_runner_under_test", "e2c_v3_rule_generalization.py")
gx = rg.gx


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    """The frozen inputs are read through repo-relative paths."""
    monkeypatch.chdir(_ROOT)


def _build(ds):
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(_ROOT)
        return (rg.build_rg_salmu_manifest() if ds == "salmu"
                else rg.build_rg_numeric_manifest())


@pytest.fixture(scope="session")
def sal_man():
    return _build("salmu")


@pytest.fixture(scope="session")
def num_man():
    return _build("celeba_numeric")


@pytest.fixture(scope="session")
def committed_numeric():
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(_ROOT)
        path = rg.gxm.MANIFEST_DIR / "numeric_manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))


def _set_of(man, set_id):
    return next(s for s in man["sets"] if s["set_id"] == set_id)


def _code_of_pid(man, pid):
    return next(man["code_of"][i] for i in man["identity_ids"]
                if man["profiles"][i]["profile_id"] == pid)


def _mutate(man):
    """A deep copy: negative tests mutate, they never touch the fixture."""
    return json.loads(json.dumps(man))


# ------------------------------------------------------------------ #
# the outcome taxonomy
# ------------------------------------------------------------------ #
def test_the_two_headline_outcomes_are_distinct_and_never_alias_each_other():
    headline = {rg.ENTITY_SPECIFIC, rg.RULE_GENERALIZED}
    assert len(headline) == 2
    boundary = {rg.RULE_CONSISTENT, rg.OVERGENERALIZED}
    assert not headline & boundary
    assert rg.INDETERMINATE not in headline | boundary
    verdicts = {rg.VERDICT_ENTITY, rg.VERDICT_RULE, rg.VERDICT_MIXED,
                rg.VERDICT_INDETERMINATE, rg.VERDICT_EDIT_FAILURE}
    assert len(verdicts) == 5


def test_every_outcome_and_verdict_is_documented_in_the_frozen_taxonomy():
    for outcome in (rg.ENTITY_SPECIFIC, rg.RULE_GENERALIZED,
                    rg.RULE_CONSISTENT, rg.OVERGENERALIZED,
                    rg.INDETERMINATE):
        assert rg.TAXONOMY["member_outcomes"][outcome]
    for verdict in (rg.VERDICT_ENTITY, rg.VERDICT_RULE, rg.VERDICT_MIXED,
                    rg.VERDICT_INDETERMINATE, rg.VERDICT_EDIT_FAILURE):
        assert rg.TAXONOMY["set_verdicts"][verdict]


def test_the_taxonomy_declares_the_mislabeling_guards_it_is_tested_against():
    guards = " ".join(rg.TAXONOMY["mislabeling_guards"])
    assert "never reported as either pure outcome" in guards
    # rule_consistent is NOT entity_specific evidence: the guard that keeps
    # an outside member keeping its label from being counted as "only the
    # edited code changed"
    assert "rule_consistent, not entity_specific" in guards
    assert "never override it" in guards           # soft vs strict
    assert "precondition failure, not an outcome" in guards


# ------------------------------------------------------------------ #
# classify_holdout_outcome -- the per-member truth table
# ------------------------------------------------------------------ #
def test_a_branch_member_following_the_rule_generalizes():
    assert rg.classify_holdout_outcome("therapy professional", "alpha",
                                       "therapy professional",
                                       "branch_member") == \
        rg.RULE_GENERALIZED


def test_a_branch_member_keeping_its_own_label_is_entity_specific():
    assert rg.classify_holdout_outcome("alpha", "alpha",
                                       "therapy professional",
                                       "branch_member") == \
        rg.ENTITY_SPECIFIC


def test_an_interior_value_following_the_bin_rule_generalizes():
    for role in ("interior", "interior_upper_edge"):
        assert rg.classify_holdout_outcome("15\u201319 years", "16 years",
                                           "15\u201319 years", role) == \
            rg.RULE_GENERALIZED


def test_an_interior_value_keeping_its_exact_label_is_entity_specific():
    assert rg.classify_holdout_outcome("16 years", "16 years",
                                       "15\u201319 years", "interior") == \
        rg.ENTITY_SPECIFIC


def test_an_outside_value_keeping_its_label_is_rule_consistent_not_entity():
    """The same parse means two different things depending on the role.

    A member OUTSIDE the bin that keeps its own label is following the rule
    (the rule stops at the edge); counting it as entity-specific evidence
    would silently credit the edit with a boundary it never had to respect.
    """
    assert rg.classify_holdout_outcome("20 years", "20 years",
                                       "15\u201319 years",
                                       "boundary_outside") == \
        rg.RULE_CONSISTENT
    assert rg.classify_holdout_outcome("20 years", "20 years",
                                       "15\u201319 years",
                                       "interior") == rg.ENTITY_SPECIFIC


def test_an_outside_value_emitting_the_bin_label_is_a_third_outcome():
    assert rg.classify_holdout_outcome("15\u201319 years", "20 years",
                                       "15\u201319 years",
                                       "boundary_outside") == \
        rg.OVERGENERALIZED
    # ... and an interior member can never be "overgeneralized": inside the
    # bin, emitting the bin label IS the rule
    assert rg.classify_holdout_outcome("15\u201319 years", "16 years",
                                       "15\u201319 years", "interior") == \
        rg.RULE_GENERALIZED


@pytest.mark.parametrize("role", ["branch_member", "interior",
                                  "interior_upper_edge", "boundary_outside"])
def test_anything_else_is_indeterminate_and_is_never_coerced(role):
    for parsed in (None, "Unknown", "some other profession",
                   "engineer (unrecognized)"):
        assert rg.classify_holdout_outcome(parsed, "16 years",
                                           "15\u201319 years", role) == \
            rg.INDETERMINATE


# ------------------------------------------------------------------ #
# set_verdict -- composition, never collapse
# ------------------------------------------------------------------ #
def test_a_failed_edit_is_a_precondition_failure_not_a_generalization():
    outcomes = {"h1": rg.RULE_GENERALIZED, "h2": rg.RULE_GENERALIZED}
    assert rg.set_verdict(False, outcomes) == rg.VERDICT_EDIT_FAILURE


def test_every_member_keeping_its_label_is_entity_specific_abstraction():
    assert rg.set_verdict(True, {"h1": rg.ENTITY_SPECIFIC,
                                 "h2": rg.ENTITY_SPECIFIC}) == \
        rg.VERDICT_ENTITY
    # an outside member that respected the edge does not spoil it
    assert rg.set_verdict(True, {"h1": rg.ENTITY_SPECIFIC,
                                 "h2": rg.RULE_CONSISTENT}) == \
        rg.VERDICT_ENTITY


def test_every_branch_member_following_the_rule_is_rule_generalization():
    assert rg.set_verdict(True, {"h1": rg.RULE_GENERALIZED,
                                 "h2": rg.RULE_GENERALIZED}) == \
        rg.VERDICT_RULE
    assert rg.set_verdict(True, {"h1": rg.RULE_GENERALIZED,
                                 "h2": rg.RULE_CONSISTENT}) == \
        rg.VERDICT_RULE


def test_a_mixed_composition_is_never_collapsed_into_either_pure_label():
    mixed = {"h1": rg.ENTITY_SPECIFIC, "h2": rg.RULE_GENERALIZED}
    assert rg.set_verdict(True, mixed) == rg.VERDICT_MIXED
    # the same holds when the split is between the rule and its boundary
    assert rg.set_verdict(True, {"h1": rg.RULE_GENERALIZED,
                                 "h2": rg.OVERGENERALIZED}) == \
        rg.VERDICT_MIXED
    assert rg.set_verdict(True, {"h1": rg.ENTITY_SPECIFIC,
                                 "h2": rg.OVERGENERALIZED}) == \
        rg.VERDICT_MIXED


@pytest.mark.parametrize("other", [rg.ENTITY_SPECIFIC, rg.RULE_GENERALIZED,
                                   rg.RULE_CONSISTENT, rg.OVERGENERALIZED])
def test_one_indeterminate_member_makes_the_set_indeterminate(other):
    assert rg.set_verdict(True, {"h1": other,
                                 "h2": rg.INDETERMINATE}) == \
        rg.VERDICT_INDETERMINATE


def test_no_members_at_all_is_not_a_vacuous_entity_specific_verdict():
    assert rg.set_verdict(True, {}) == rg.VERDICT_INDETERMINATE


def test_an_all_outside_roster_that_holds_is_entity_specific_by_definition():
    """Documented on purpose: "only the edited code changed" is exactly what
    an all-boundary-outside roster that keeps its labels observed."""
    assert rg.set_verdict(True, {"h1": rg.RULE_CONSISTENT,
                                 "h2": rg.RULE_CONSISTENT}) == \
        rg.VERDICT_ENTITY


# ------------------------------------------------------------------ #
# boundary_pair_verdict -- the discontinuity at one bin edge
# ------------------------------------------------------------------ #
def test_the_boundary_matrix():
    assert rg.boundary_pair_verdict(True, rg.RULE_CONSISTENT) == \
        rg.BOUNDARY_SHARP
    assert rg.boundary_pair_verdict(True, rg.OVERGENERALIZED) == \
        rg.BOUNDARY_VIOLATED
    assert rg.boundary_pair_verdict(False, rg.RULE_CONSISTENT) == \
        rg.BOUNDARY_NO_GEN
    # an inside member that did not follow the rule says nothing about the
    # edge, whatever the outside member did
    assert rg.boundary_pair_verdict(False, rg.OVERGENERALIZED) == \
        rg.BOUNDARY_INDETERMINATE
    assert rg.boundary_pair_verdict(False, rg.ENTITY_SPECIFIC) == \
        rg.BOUNDARY_INDETERMINATE
    assert rg.boundary_pair_verdict(True, rg.INDETERMINATE) == \
        rg.BOUNDARY_INDETERMINATE
    assert rg.boundary_pair_verdict(False, rg.INDETERMINATE) == \
        rg.BOUNDARY_INDETERMINATE


def test_a_sharp_boundary_needs_both_sides_of_the_edge():
    """Sharpness is a PAIR property: inside follows, outside does not."""
    assert rg.boundary_pair_verdict(True, rg.RULE_CONSISTENT) != \
        rg.boundary_pair_verdict(False, rg.RULE_CONSISTENT)


# ------------------------------------------------------------------ #
# SALMU frozen sets: train one member, withhold its branch siblings
# ------------------------------------------------------------------ #
def test_salmu_sets_train_one_member_and_withhold_its_branch_siblings(
        sal_man):
    assert sal_man["n_sets"] == 5
    assert sal_man["n_cells"] == 5 * len(rg.RG_SEEDS)
    for s in sal_man["sets"]:
        t = s["trained"]
        depth = s["rule_depth"]
        # the rule label IS the trained member's own parent at that depth
        assert sal_man["hierarchy_of"][t["identity_id"]][depth] == \
            t["rule_label"]
        assert t["rule_label"] != t["source_alias"]
        for h in s["holdouts"]:
            assert h["shares_rule_at_depth"] is True
            assert sal_man["hierarchy_of"][h["identity_id"]][depth] == \
                t["rule_label"]
            assert h["identity_id"] != t["identity_id"]
            # a held-out member must have its OWN label to keep, otherwise
            # "entity-specific" and "rule-following" are the same string
            assert h["baseline_alias"] != t["rule_label"]


def test_salmu_holdouts_are_absent_from_retention_training(sal_man):
    """The withhold: a held-out member is neither transformed nor retained."""
    for s in sal_man["sets"]:
        withheld = {h["identity_id"] for h in s["holdouts"]}
        assert not set(s["retain_ids"]) & withheld
        assert s["trained"]["identity_id"] not in s["retain_ids"]
        assert len(s["retain_ids"]) == len(sal_man["identity_ids"]) - 1 - \
            len(withheld)
        assert s["retain_ids"]


def test_salmu_covers_both_directions_of_each_level1_sibling_pair(sal_man):
    pairs = {(s["trained"]["identity_id"],
              s["holdouts"][0]["identity_id"])
             for s in sal_man["sets"] if s["rule_depth"] == 1}
    assert pairs == {("00060576", "00102677"), ("00102677", "00060576"),
                     ("00082704", "00106148"), ("00106148", "00082704")}
    for s in sal_man["sets"]:
        if s["rule_depth"] == 1:
            assert s["holdouts"][0]["same_level1_as_trained"] is True


def test_the_level2_media_set_spans_more_than_one_level1_branch(sal_man):
    s = _set_of(sal_man, "rg_sal_L2_media_from_00041286")
    assert s["rule_depth"] == 2
    assert s["trained"]["rule_label"] == "media"
    hold_l1 = {h["level1"] for h in s["holdouts"]}
    assert hold_l1 == {"design professional", "information professional"}
    assert all(h["same_level1_as_trained"] is False for h in s["holdouts"])
    assert sal_man["hierarchy_of"][s["trained"]["identity_id"]][1] == \
        "media professional"
    assert {h["identity_id"] for h in s["holdouts"]} == \
        {"00082704", "00106148", "00110390"}


def test_salmu_manifest_is_deterministic_and_carries_the_frozen_taxonomy(
        sal_man):
    again = _build("salmu")
    assert again == sal_man
    assert sal_man["kind"] == "e2c_v3_rule_generalization_manifest_v1"
    assert sal_man["edit_seeds"] == list(gx.SEEDS_DEFAULT) == [17, 42, 123]
    assert sal_man["produced_by"] == \
        "scripts/e2c_v3_rule_generalization.py"
    assert sal_man["taxonomy"] == rg.TAXONOMY
    assert rg.validate_rg_manifest("salmu", sal_man) == []


# ------------------------------------------------------------------ #
# numeric: the frozen extension and the two requested sets
# ------------------------------------------------------------------ #
def test_the_extension_leaves_the_frozen_benchmark_untouched(
        num_man, committed_numeric):
    frozen_ids = committed_numeric["identity_ids"]
    assert len(frozen_ids) == 24
    for iid in frozen_ids:
        assert num_man["code_of"][iid] == committed_numeric["code_of"][iid]
        assert num_man["alias_of"][iid] == committed_numeric["alias_of"][iid]
        got, want = num_man["profiles"][iid], committed_numeric["profiles"][iid]
        assert got["rg_extension"] is False
        for key in ("profile_id", "field", "exact_value", "baseline_kind",
                    "boundary_tags", "narrow_bin", "broad_bin", "rounded",
                    "category", "unit"):
            assert got[key] == want[key], key
    added = [i for i in num_man["identity_ids"] if i not in set(frozen_ids)]
    assert added == ["24", "25", "26", "27", "28", "29"]
    assert [num_man["code_of"][i] for i in added] == \
        [f"GRN_{i}" for i in added]
    assert all(num_man["profiles"][i]["rg_extension"] for i in added)


def test_the_extension_adds_exactly_the_six_withheld_values(num_man):
    assert num_man["extension"]["new_values"] == [16, 18, 21, 25, 29, 30]
    assert num_man["extension"]["n_new_profiles"] == 6
    added = {"24": 16, "25": 18, "26": 21, "27": 25, "28": 29, "29": 30}
    for iid, value in added.items():
        prof = num_man["profiles"][iid]
        assert prof["exact_value"] == value
        assert prof["field"] == "years_experience"
        assert prof["baseline_kind"] == "exact"
        assert num_man["alias_of"][iid] == f"{value} years"
    # every one of the 30 codes has its OWN baseline label, so "kept its own
    # label" is always distinguishable from "followed the rule"
    assert len(set(num_man["alias_of"].values())) == 30


def test_narrow_set_15_is_the_requested_experiment(num_man):
    s = _set_of(num_man, "rg_num_narrow_15")
    assert s["operation"] == "exact_to_narrow"
    assert s["trained"]["profile_id"] == "Y02"
    assert s["trained"]["exact_value"] == 15
    assert s["trained"]["rule_label"] == "15\u201319 years"
    by_role = {}
    for h in s["holdouts"]:
        by_role.setdefault(h["role"], []).append(h["exact_value"])
    assert sorted(by_role["interior"]) == [16, 18]
    assert by_role["interior_upper_edge"] == [19]
    assert by_role["boundary_outside"] == [20]
    assert [p["edge"] for p in s["boundary_pairs"]] == ["19|20"]
    pair = s["boundary_pairs"][0]
    assert (pair["inside_value"], pair["outside_value"]) == (19, 20)
    assert pair["inside_is_trained_member"] is False


def test_broad_set_20_is_the_requested_experiment(num_man):
    s = _set_of(num_man, "rg_num_broad_20")
    assert s["operation"] == "exact_to_broad"
    assert s["trained"]["profile_id"] == "Y04"
    assert s["trained"]["exact_value"] == 20
    assert s["trained"]["rule_label"] == "20\u201329 years"
    interiors = sorted(h["exact_value"] for h in s["holdouts"]
                       if h["role"].startswith("interior"))
    assert interiors == [21, 25, 29]
    assert sorted(h["exact_value"] for h in s["holdouts"]
                  if h["role"] == "boundary_outside") == [19, 30]
    edges = {p["edge"]: p for p in s["boundary_pairs"]}
    assert set(edges) == {"29|30", "19|20"}
    assert (edges["29|30"]["inside_value"],
            edges["29|30"]["outside_value"]) == (29, 30)
    assert edges["29|30"]["inside_is_trained_member"] is False
    # the 19|20 edge of THIS set has the trained member itself inside the
    # bin, so its "inside emits the rule" side is the edit precondition
    assert edges["19|20"]["inside_is_trained_member"] is True
    assert edges["19|20"]["inside_identity_id"] == \
        s["trained"]["identity_id"]
    assert (edges["19|20"]["inside_value"],
            edges["19|20"]["outside_value"]) == (20, 19)


def test_numeric_holdouts_are_absent_from_retention_training(num_man):
    for s in num_man["sets"]:
        withheld = {h["identity_id"] for h in s["holdouts"]}
        assert not set(s["retain_ids"]) & withheld
        assert s["trained"]["identity_id"] not in s["retain_ids"]
        assert len(s["retain_ids"]) == 30 - 1 - len(withheld)
        # every boundary pair's outside member is itself withheld, so its
        # behavior at the edge is spontaneous
        for p in s["boundary_pairs"]:
            assert p["outside_identity_id"] in withheld


def test_every_requested_value_appears_with_the_requested_role(num_man):
    for sid, want in rg.RG_USER_SPEC.items():
        s = _set_of(num_man, sid)
        assert s["trained"]["exact_value"] == want["trained_value"]
        interiors = {h["exact_value"] for h in s["holdouts"]
                     if h["role"].startswith("interior")}
        assert set(want["interior"]) <= interiors
        assert set(want["boundary_edges"]) <= \
            {p["edge"] for p in s["boundary_pairs"]}


def test_numeric_manifest_is_deterministic_and_validates(num_man):
    assert _build("celeba_numeric") == num_man
    assert num_man["n_sets"] == 2
    assert num_man["n_cells"] == 2 * len(rg.RG_SEEDS)
    assert num_man["schema"] == gx.NUMERIC_SCHEMA
    assert rg.validate_rg_manifest("celeba_numeric", num_man) == []
    issues, _collisions = gx.validate_vocab(num_man["vocab"])
    assert issues == []


def test_the_extension_may_not_collide_with_a_frozen_baseline_label(
        monkeypatch):
    monkeypatch.setattr(rg, "RG_NUMERIC_NEW_PROFILES",
                        list(rg.RG_NUMERIC_NEW_PROFILES)
                        + [("RY15", "years_experience", 15, "exact", "dup")])
    with pytest.raises(ValueError, match="not unique"):
        rg.build_rg_numeric_manifest()


def test_the_extension_may_not_duplicate_a_frozen_profile_value(
        num_man, monkeypatch):
    fake = list(gx.NUMERIC_PROFILES) + \
        [("ZZ16", "years_experience", 16, "exact", "pretend frozen")]
    monkeypatch.setattr(rg.gx, "NUMERIC_PROFILES", fake)
    issues = rg.validate_rg_manifest("celeba_numeric", num_man)
    assert any("duplicates a frozen profile value" in i for i in issues)


# ------------------------------------------------------------------ #
# validate_rg_manifest -- the hard RG0 gate must actually catch things
# ------------------------------------------------------------------ #
def test_salmu_validation_catches_a_holdout_outside_the_branch(sal_man):
    man = _mutate(sal_man)
    man["sets"][0]["holdouts"][0]["shares_rule_at_depth"] = False
    issues = rg.validate_rg_manifest("salmu", man)
    assert any("does not share the rule label" in i for i in issues)


def test_salmu_validation_catches_a_rule_label_that_is_not_the_parent(sal_man):
    man = _mutate(sal_man)
    man["sets"][0]["trained"]["rule_label"] = "healthcare"   # depth 2, not 1
    issues = rg.validate_rg_manifest("salmu", man)
    assert any("not the trained member's own parent" in i for i in issues)


def test_salmu_validation_catches_a_degenerate_holdout(sal_man):
    man = _mutate(sal_man)
    s = man["sets"][0]
    s["holdouts"][0]["baseline_alias"] = s["trained"]["rule_label"]
    issues = rg.validate_rg_manifest("salmu", man)
    assert any("degenerate" in i for i in issues)


def test_salmu_validation_catches_retention_training_a_withheld_member(sal_man):
    """The withhold is enforced structurally, not by intention."""
    man = _mutate(sal_man)
    s = man["sets"][0]
    s["retain_ids"].append(s["holdouts"][0]["identity_id"])
    issues = rg.validate_rg_manifest("salmu", man)
    assert any("retain overlaps" in i for i in issues)


def test_salmu_validation_catches_a_duplicated_holdout(sal_man):
    man = _mutate(sal_man)
    s = man["sets"][-1]                     # the three-holdout media set
    s["holdouts"].append(dict(s["holdouts"][0]))
    issues = rg.validate_rg_manifest("salmu", man)
    assert any("duplicates" in i for i in issues)


def test_salmu_validation_refuses_an_empty_retention_set(sal_man):
    man = _mutate(sal_man)
    man["sets"][0]["retain_ids"] = []
    issues = rg.validate_rg_manifest("salmu", man)
    assert any("no retention-trained members" in i for i in issues)


def test_numeric_validation_catches_an_interior_value_outside_the_bin(num_man):
    man = _mutate(num_man)
    s = _set_of(man, "rg_num_narrow_15")
    next(h for h in s["holdouts"] if h["exact_value"] == 16)["exact_value"] = 25
    issues = rg.validate_rg_manifest("celeba_numeric", man)
    assert any("not inside" in i for i in issues)


def test_numeric_validation_catches_an_upper_edge_that_is_not_the_edge(num_man):
    man = _mutate(num_man)
    s = _set_of(man, "rg_num_narrow_15")
    h = next(x for x in s["holdouts"] if x["role"] == "interior_upper_edge")
    h["exact_value"] = 18                   # inside the bin, but not its edge
    issues = rg.validate_rg_manifest("celeba_numeric", man)
    assert any("upper-edge holdout" in i for i in issues)


def test_numeric_validation_catches_an_outside_member_that_is_inside(num_man):
    man = _mutate(num_man)
    s = _set_of(man, "rg_num_narrow_15")
    next(h for h in s["holdouts"]
         if h["role"] == "boundary_outside")["exact_value"] = 17
    issues = rg.validate_rg_manifest("celeba_numeric", man)
    assert any("INSIDE" in i for i in issues)


def test_numeric_validation_catches_a_non_adjacent_outside_member(num_man):
    man = _mutate(num_man)
    s = _set_of(man, "rg_num_broad_20")
    next(h for h in s["holdouts"]
         if h["exact_value"] == 30)["exact_value"] = 35
    issues = rg.validate_rg_manifest("celeba_numeric", man)
    assert any("not adjacent to" in i for i in issues)


def test_numeric_validation_catches_a_boundary_pair_that_is_not_adjacent(
        num_man):
    man = _mutate(num_man)
    s = _set_of(man, "rg_num_broad_20")
    next(p for p in s["boundary_pairs"] if p["edge"] == "29|30")[
        "outside_value"] = 32
    issues = rg.validate_rg_manifest("celeba_numeric", man)
    assert any("pair 29|30 is not adjacent" in i for i in issues)


def test_numeric_validation_catches_a_pair_outside_a_withheld_member(num_man):
    man = _mutate(num_man)
    s = _set_of(man, "rg_num_broad_20")
    p = next(x for x in s["boundary_pairs"] if x["edge"] == "29|30")
    p["outside_identity_id"] = s["retain_ids"][0]
    issues = rg.validate_rg_manifest("celeba_numeric", man)
    assert any("outside member is not held out" in i for i in issues)


def test_numeric_validation_catches_a_pair_inside_a_stranger(num_man):
    man = _mutate(num_man)
    s = _set_of(man, "rg_num_broad_20")
    p = next(x for x in s["boundary_pairs"] if x["edge"] == "29|30")
    p["inside_identity_id"] = s["retain_ids"][0]
    issues = rg.validate_rg_manifest("celeba_numeric", man)
    assert any("neither trained nor held out" in i for i in issues)


def test_numeric_validation_catches_a_rule_label_that_is_not_the_frozen_bin(
        num_man):
    man = _mutate(num_man)
    s = _set_of(man, "rg_num_narrow_15")
    s["trained"]["rule_label"] = "10\u201319 years"
    issues = rg.validate_rg_manifest("celeba_numeric", man)
    assert any("not the frozen-schema bin" in i for i in issues)


def test_numeric_validation_catches_an_unknown_holdout_role(num_man):
    man = _mutate(num_man)
    s = _set_of(man, "rg_num_narrow_15")
    s["holdouts"][0]["role"] = "cousin"
    issues = rg.validate_rg_manifest("celeba_numeric", man)
    assert any("unknown holdout role" in i for i in issues)


def test_numeric_validation_enforces_the_requested_coverage(num_man):
    man = _mutate(num_man)
    s = _set_of(man, "rg_num_broad_20")
    s["holdouts"] = [h for h in s["holdouts"] if h["exact_value"] != 25]
    s["boundary_pairs"] = [p for p in s["boundary_pairs"]
                           if p["edge"] != "29|30"]
    s["trained"]["exact_value"] = 21
    issues = rg.validate_rg_manifest("celeba_numeric", man)
    assert any("missing interior values [25]" in i for i in issues)
    assert any("missing boundary edges ['29|30']" in i for i in issues)
    assert any("trained value != 20" in i for i in issues)


def test_numeric_validation_catches_retention_training_a_withheld_value(
        num_man):
    man = _mutate(num_man)
    s = _set_of(man, "rg_num_narrow_15")
    s["retain_ids"].append(s["holdouts"][0]["identity_id"])
    issues = rg.validate_rg_manifest("celeba_numeric", man)
    assert any("retain overlaps" in i for i in issues)


# ------------------------------------------------------------------ #
# RG0: freeze / verify-not-rewrite / refuse drift
# ------------------------------------------------------------------ #
@pytest.fixture
def rg0_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "RG_MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(rg, "RG_OUT_ROOT", tmp_path / "outputs")
    return tmp_path


def test_rg0_freezes_then_verifies_without_rewriting(rg0_tmp, num_man):
    got = rg.build_or_verify("celeba_numeric")
    assert got == num_man
    path = rg0_tmp / "manifests" / "rg_manifest_celeba_numeric.json"
    assert path.exists()
    first, mtime = path.read_bytes(), path.stat().st_mtime_ns
    assert rg.build_or_verify("celeba_numeric") == num_man
    assert path.read_bytes() == first
    assert path.stat().st_mtime_ns == mtime       # verified, NOT rewritten


def test_rg0_validation_record_is_byte_deterministic(rg0_tmp):
    rg.build_or_verify("salmu")
    path = rg0_tmp / "outputs" / "salmu" / "rg0_validation.json"
    first = path.read_bytes()
    rg.build_or_verify("salmu")
    assert path.read_bytes() == first
    rec = json.loads(first.decode())
    assert rec["issues"] == []
    assert rec["n_sets"] == 5 and rec["n_cells"] == 15
    assert rec["n_identities"] == 12
    # committed file: no timestamps, so a re-run never dirties the worktree
    text = first.decode()
    for banned in ("generated_at", "updated_at", "timestamp", "started_utc"):
        assert banned not in text
    row = next(r for r in rec["sets"]
               if r["set_id"] == "rg_sal_L2_media_from_00041286")
    assert row["rule_label"] == "media"
    assert [h[1] for h in row["holdouts"]] == ["branch_member"] * 3


def test_rg0_refuses_a_manifest_that_drifted_from_the_builder(rg0_tmp):
    rg.build_or_verify("salmu")
    path = rg0_tmp / "manifests" / "rg_manifest_salmu.json"
    man = json.loads(path.read_text(encoding="utf-8"))
    man["sets"][0]["set_id"] = "rg_sal_renamed_after_the_fact"
    path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must not drift"):
        rg.build_or_verify("salmu")


def test_rg0_fails_closed_on_an_invalid_manifest(rg0_tmp, monkeypatch):
    monkeypatch.setattr(rg, "validate_rg_manifest",
                        lambda ds, man: ["boom: the sets do not mean that"])
    with pytest.raises(RuntimeError, match="RG0 validation failed"):
        rg.build_or_verify("salmu")
    assert not (rg0_tmp / "manifests" / "rg_manifest_salmu.json").exists()
    assert not (rg0_tmp / "outputs").exists()      # nothing written at all


def test_the_ctx_exposes_only_frozen_inputs(num_man, sal_man):
    ctx = rg.rg_dataset_ctx("celeba_numeric", num_man)
    assert ctx["kind"] == "numeric"
    assert len(ctx["identity_ids"]) == 30
    assert ctx["baseline_alias_of"] == num_man["alias_of"]
    assert ctx["profiles"]["24"]["exact_value"] == 16
    assert ctx["schema"] == gx.NUMERIC_SCHEMA
    sctx = rg.rg_dataset_ctx("salmu", sal_man)
    assert sctx["kind"] == "taxonomic"
    assert sctx["dag"] == sal_man["dag"]
    assert list(sctx["vocab"]) == list(sal_man["vocab"])
    assert sctx["hierarchy_of"]["00060576"][1] == "therapy professional"


# ------------------------------------------------------------------ #
# RG2 driven by a stub model: the withhold, the strict classification
# and the cached-record validation -- no GPU
# ------------------------------------------------------------------ #
DS = "celeba_numeric"
NARROW = "rg_num_narrow_15"
BROAD = "rg_num_broad_20"


def _stub_session_class(responses, sessions):
    class _Backend:
        def generate(self, _img, prompt, max_new_tokens=12):
            for code, text in responses.items():
                if code in prompt:
                    return argparse.Namespace(text=text)
            return argparse.Namespace(text="garbage output")

    class _Model:
        def eval(self):
            return self

    class _Session:
        def __init__(self, args, name):
            self.args, self.name = args, name
            self.adapter, self.processor = object(), object()
            self.model = _Model()
            self.resets, self.released = [], False
            sessions.append(self)

        def reset_to(self, path):
            self.resets.append(str(path))

        def backend(self):
            return _Backend()

        def release(self):
            self.released = True

    return _Session


def _drive(tmp_path, monkeypatch, man, set_id, labels_by_code, seed=17,
           out_base=None, probs_override=None, ds=DS):
    """Run ONE (set, seed) cell through ``run_rg_cells`` with a stub model.

    ``labels_by_code`` maps a code to the labels the stub emits (a list of
    TWO labels produces a multi-label, therefore invalid, output); every
    member not listed emits its own baseline label.
    """
    ctx = rg.rg_dataset_ctx(ds, man)
    responses, probs_by_code = {}, {}
    for iid in ctx["identity_ids"]:
        code = ctx["code_of"][iid]
        labs = labels_by_code.get(code) or [ctx["baseline_alias_of"][iid]]
        responses[code] = " ".join(labs) + "."
        probs_by_code[code] = dict.fromkeys(labs, 1.0 / len(labs))
    probs_by_code.update(probs_override or {})
    item_calls, trained_calls, sessions = [], [], []

    def _items(adapter, processor, pairs, repeat=1):
        item_calls.append({"pairs": list(pairs), "repeat": repeat})
        return [dict(p, repeat=repeat) for p in pairs]

    def _train(condition, adapter, model, processor, items, output_dir,
               device, steps=None, warmup=None, lr=None):
        trained_calls.append({"condition": condition, "items": list(items),
                              "steps": steps, "warmup": warmup, "lr": lr})
        final = Path(output_dir) / "adapter_final"
        final.mkdir(parents=True, exist_ok=True)
        (final / "adapter_model.safetensors").write_bytes(
            condition.encode("utf-8"))

    def _probs(adapter, model, processor, code_id, candidate_labels, device,
               *, prompt_text=None):
        given = probs_by_code.get(code_id, {})
        return {l: {"prob": float(given.get(l, 0.0)), "log_prob": 0.0,
                    "n_tokens": 1} for l in candidate_labels}

    monkeypatch.setattr(rg.rv, "build_supervised_items", _items)
    monkeypatch.setattr(rg.rv, "train_supervised", _train)
    monkeypatch.setattr(rg.rv, "full_sequence_label_probs", _probs)
    monkeypatch.setattr(rg.mx, "ModelSession",
                        _stub_session_class(responses, sessions))
    monkeypatch.setattr(rg, "EXECUTING_COMMIT", "c0ffee0")
    monkeypatch.setattr(rg, "RG_MANIFEST_DIR", tmp_path / "manifests")
    (tmp_path / "manifests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifests" / f"rg_manifest_{ds}.json").write_text(
        json.dumps(man, indent=2) + "\n", encoding="utf-8")
    out_base = Path(out_base) if out_base else tmp_path / "outputs"
    one = _mutate(man)
    for s in one["sets"]:
        s["seeds"] = [seed]
    args = argparse.Namespace(device="cpu", max_gen_tokens=12,
                              only_sets=[set_id], only_seeds=[seed],
                              ul_steps=500, ul_warmup=50, ul_lr=2e-5,
                              ul_repeat=50)
    cells_root, stale = rg.run_rg_cells(args, ds, ctx, one, out_base)
    path = (out_base / "cells" / set_id / f"seed_{seed}"
            / "rg_cell_results.json")
    return argparse.Namespace(
        cell=(json.loads(path.read_text(encoding="utf-8"))
              if path.exists() else None),
        path=path, cell_dir=path.parent, ctx=ctx, man=one, args=args,
        out_base=out_base, item_calls=item_calls, sessions=sessions,
        trained_calls=trained_calls, stale=stale, cells_root=cells_root)


def _narrow_setup(num_man):
    s = _set_of(num_man, NARROW)
    code = {h["profile_id"]: h["code"] for h in s["holdouts"]}
    iid = {h["profile_id"]: h["identity_id"] for h in s["holdouts"]}
    base = {h["profile_id"]: h["baseline_alias"] for h in s["holdouts"]}
    return s, s["trained"]["rule_label"], s["trained"]["code"], code, iid, base


def _rule_everywhere(num_man, outside_too=False):
    """The trained member follows the rule; so does every interior member."""
    _s, rule, t_code, code, _iid, _base = _narrow_setup(num_man)
    labels = {t_code: [rule], code["RY16"]: [rule], code["RY18"]: [rule],
              code["Y03"]: [rule]}
    if outside_too:
        labels[code["Y04"]] = [rule]
    return labels


def test_rg2_rule_generalization_with_a_sharp_boundary(tmp_path, monkeypatch,
                                                       num_man):
    labels = _rule_everywhere(num_man)
    s, _rule, _t, _code, iid, base = _narrow_setup(num_man)
    r = _drive(tmp_path, monkeypatch, num_man, NARROW, labels)
    cell = r.cell
    assert cell["edit_success"] is True
    assert cell["verdict"] == rg.VERDICT_RULE
    assert cell["outcomes"] == {iid["RY16"]: rg.RULE_GENERALIZED,
                                iid["RY18"]: rg.RULE_GENERALIZED,
                                iid["Y03"]: rg.RULE_GENERALIZED,
                                iid["Y04"]: rg.RULE_CONSISTENT}
    pair = cell["boundary_pairs"][0]
    assert pair["edge"] == "19|20"
    assert pair["inside_emits_rule"] is True
    assert pair["outside_outcome"] == rg.RULE_CONSISTENT
    assert pair["verdict"] == rg.BOUNDARY_SHARP
    assert cell["retain_controls"] == {"n": len(s["retain_ids"]),
                                       "strict_accuracy": 1.0,
                                       "drifted_ids": []}
    # the outside member kept its own label with no mass on the bin
    y04 = cell["holdouts"][iid["Y04"]]
    assert y04["parsed_label"] == base["Y04"]
    assert y04["soft"]["p_baseline_alias"] == 1.0
    assert y04["soft"]["p_rule_label"] == 0.0
    assert y04["exact_value"] == 20
    assert cell["holdouts"][iid["RY16"]]["soft"]["p_rule_label"] == 1.0
    assert cell["min_candidate_mass"] == 1.0


def test_rg2_withholds_the_held_out_members_from_the_training_data(
        tmp_path, monkeypatch, num_man):
    """THE withhold: nothing about a held-out member is in the edit's data."""
    labels = _rule_everywhere(num_man)
    s, rule, t_code, code, _iid, _base = _narrow_setup(num_man)
    r = _drive(tmp_path, monkeypatch, num_man, NARROW, labels)
    assert len(r.trained_calls) == 1
    call = r.trained_calls[0]
    assert call["condition"] == f"rg_{NARROW}__seed17"
    assert (call["steps"], call["warmup"], call["lr"]) == (500, 50, 2e-5)
    assert len(r.item_calls) == 2
    target, retain = r.item_calls
    # one transformed mapping, boosted exactly like every other cell
    assert [p["answer"] for p in target["pairs"]] == [rule]
    assert t_code in target["pairs"][0]["prompt"]
    assert target["repeat"] == 50 * rg.gxm.TARGET_BOOST
    # retention covers every OTHER member -- and no withheld member
    assert len(retain["pairs"]) == len(s["retain_ids"])
    assert retain["repeat"] == 50 * rg.gxm.RETAIN_REPEAT
    withheld_codes = {s["trained"]["code"], *[code[p] for p in code]}
    trained_prompts = " ".join(p["prompt"] for c in r.item_calls
                               for p in c["pairs"])
    for held in [code[p] for p in code]:
        assert held not in trained_prompts
    assert t_code in trained_prompts
    # every retained member is trained to KEEP its own baseline label
    by_code = {r.ctx["code_of"][i]: r.ctx["baseline_alias_of"][i]
               for i in r.ctx["identity_ids"]}
    for p in retain["pairs"]:
        got = next(c for c in by_code if c in p["prompt"])
        assert p["answer"] == by_code[got]
    assert len(withheld_codes) == 5
    # the cell starts from the established baseline route
    assert r.sessions[0].resets == [str(rg._baseline_ckpt_rg(DS,
                                                             r.out_base))]
    assert r.sessions[0].released is True


def test_rg2_entity_specific_abstraction_keeps_every_member_untouched(
        tmp_path, monkeypatch, num_man):
    _s, rule, t_code, _code, iid, base = _narrow_setup(num_man)
    r = _drive(tmp_path, monkeypatch, num_man, NARROW, {t_code: [rule]})
    cell = r.cell
    assert cell["edit_success"] is True
    assert cell["verdict"] == rg.VERDICT_ENTITY
    assert cell["outcomes"] == {iid["RY16"]: rg.ENTITY_SPECIFIC,
                                iid["RY18"]: rg.ENTITY_SPECIFIC,
                                iid["Y03"]: rg.ENTITY_SPECIFIC,
                                iid["Y04"]: rg.RULE_CONSISTENT}
    assert cell["boundary_pairs"][0]["inside_emits_rule"] is False
    assert cell["boundary_pairs"][0]["verdict"] == rg.BOUNDARY_NO_GEN
    # entity-specific and rule-consistent are reported separately, so the
    # composition shows exactly which question each member answered
    assert cell["holdouts"][iid["Y04"]]["outcome"] == rg.RULE_CONSISTENT
    assert cell["holdouts"][iid["Y03"]]["outcome"] == rg.ENTITY_SPECIFIC
    assert cell["holdouts"][iid["Y03"]]["parsed_label"] == base["Y03"]


def test_rg2_overgeneralization_is_a_third_outcome_and_makes_the_set_mixed(
        tmp_path, monkeypatch, num_man):
    """Crossing the bin edge is neither headline label."""
    labels = _rule_everywhere(num_man, outside_too=True)
    _s, _rule, _t, _code, iid, _base = _narrow_setup(num_man)
    r = _drive(tmp_path, monkeypatch, num_man, NARROW, labels)
    cell = r.cell
    assert cell["edit_success"] is True
    assert cell["outcomes"][iid["Y04"]] == rg.OVERGENERALIZED
    assert cell["outcomes"][iid["RY16"]] == rg.RULE_GENERALIZED
    assert cell["verdict"] == rg.VERDICT_MIXED      # never RULE, never ENTITY
    assert cell["boundary_pairs"][0]["verdict"] == rg.BOUNDARY_VIOLATED


def test_rg2_a_failed_edit_is_a_precondition_failure(tmp_path, monkeypatch,
                                                     num_man):
    """Holdouts that "generalize" mean nothing if the edit itself failed."""
    s, _rule, t_code, _code, iid, _base = _narrow_setup(num_man)
    labels = _rule_everywhere(num_man)
    labels[t_code] = [s["trained"]["source_alias"]]   # the edit did not take
    r = _drive(tmp_path, monkeypatch, num_man, NARROW, labels)
    cell = r.cell
    assert cell["edit_success"] is False
    assert cell["trained"]["parsed_label"] == s["trained"]["source_alias"]
    assert cell["verdict"] == rg.VERDICT_EDIT_FAILURE
    # the member outcomes are still recorded -- the precondition dominates
    # the verdict, it does not erase the measurement
    assert cell["outcomes"][iid["RY16"]] == rg.RULE_GENERALIZED
    assert cell["trained"]["p_source_alias"] == 1.0
    assert cell["trained"]["p_rule_label"] == 0.0
    assert cell["trained"]["edit_success"] is False


def test_rg2_an_ambiguous_held_out_output_is_indeterminate(tmp_path,
                                                           monkeypatch,
                                                           num_man):
    _s, rule, _t_code, code, iid, base = _narrow_setup(num_man)
    labels = _rule_everywhere(num_man)
    labels[code["RY16"]] = [base["RY16"], rule]      # two labels -> invalid
    r = _drive(tmp_path, monkeypatch, num_man, NARROW, labels)
    cell = r.cell
    block = cell["holdouts"][iid["RY16"]]
    assert block["parsed_label"] is None
    assert block["multi_label_invalid"] is True
    assert sorted(block["recognized_labels"]) == sorted([base["RY16"], rule])
    assert block["outcome"] == rg.INDETERMINATE
    assert block["detail"]["classification"] == "unparseable"
    assert cell["verdict"] == rg.VERDICT_INDETERMINATE
    assert cell["outcomes"][iid["RY18"]] == rg.RULE_GENERALIZED


def test_rg2_soft_evidence_never_overrides_the_strict_decision(
        tmp_path, monkeypatch, num_man):
    _s, rule, t_code, code, iid, _base = _narrow_setup(num_man)
    labels = {t_code: [rule]}                        # every holdout keeps its
    override = {code["RY16"]: {rule: 0.97}}          # own label under parsing
    r = _drive(tmp_path, monkeypatch, num_man, NARROW, labels,
               probs_override=override)
    cell = r.cell
    block = cell["holdouts"][iid["RY16"]]
    assert block["parsed_label"] != rule
    assert block["outcome"] == rg.ENTITY_SPECIFIC    # strict wins
    assert block["soft"]["p_rule_label"] == 0.97     # ... and is still shown
    assert block["soft"]["p_baseline_alias"] == 0.0


def test_rg2_retention_drift_is_reported_per_member(tmp_path, monkeypatch,
                                                    num_man):
    s, _rule, _t_code, _code, _iid, _base = _narrow_setup(num_man)
    labels = _rule_everywhere(num_man)
    stray = next(i for i in s["retain_ids"])
    labels[num_man["code_of"][stray]] = [rg.gx.DELETED_LABEL]
    r = _drive(tmp_path, monkeypatch, num_man, NARROW, labels)
    cell = r.cell
    assert cell["retain_controls"]["drifted_ids"] == [stray]
    assert cell["retain_controls"]["strict_accuracy"] == \
        (len(s["retain_ids"]) - 1) / len(s["retain_ids"])
    # a drifted RETAINED member is not a held-out outcome
    assert stray not in cell["outcomes"]
    assert cell["verdict"] == rg.VERDICT_RULE


def test_rg2_the_trained_member_is_not_one_of_the_outcomes(tmp_path,
                                                           monkeypatch,
                                                           num_man):
    s, _rule, _t, _code, iid, _base = _narrow_setup(num_man)
    r = _drive(tmp_path, monkeypatch, num_man, NARROW,
               _rule_everywhere(num_man))
    assert set(r.cell["outcomes"]) == set(iid.values())
    assert s["trained"]["identity_id"] not in r.cell["outcomes"]
    assert r.cell["trained"]["detail"]["classification"]


def test_rg2_records_provenance_that_binds_the_record_to_the_code(
        tmp_path, monkeypatch, num_man):
    r = _drive(tmp_path, monkeypatch, num_man, NARROW,
               _rule_everywhere(num_man))
    prov = r.cell["provenance"]
    assert prov["git_commit"] == "c0ffee0"
    assert prov["runner_script_sha256"] == rg.rv.sha256_file(
        Path(rg.__file__).resolve())
    assert prov["shared_scoring_script_sha256"] == rg.rv.script_sha256()
    assert prov["rg_manifest_sha256"] == rg.rv.sha256_file(
        rg.rg_manifest_path(DS))
    assert r.cell["checkpoint_sha256"] == rg.rv.sha256_file(
        r.cell_dir / "edited_h" / "adapter_final"
        / "adapter_model.safetensors")
    assert r.cell["kind"] == rg.CELL_KIND


# ------------------------------------------------------------------ #
# RG2 on the taxonomic side: a withheld sibling, not a withheld value
# ------------------------------------------------------------------ #
THERAPY = "rg_sal_L1_therapy_from_00060576"
MEDIA = "rg_sal_L2_media_from_00041286"


def test_rg2_a_withheld_sibling_that_follows_the_branch_rule(
        tmp_path, monkeypatch, sal_man):
    s = _set_of(sal_man, THERAPY)
    rule, hid = s["trained"]["rule_label"], s["holdouts"][0]["identity_id"]
    assert rule == "therapy professional"
    labels = {s["trained"]["code"]: [rule], s["holdouts"][0]["code"]: [rule]}
    r = _drive(tmp_path, monkeypatch, sal_man, THERAPY, labels, ds="salmu")
    assert r.cell["edit_success"] is True
    assert r.cell["outcomes"][hid] == rg.RULE_GENERALIZED
    assert r.cell["verdict"] == rg.VERDICT_RULE
    assert r.cell["boundary_pairs"] == []          # no bin edges in a taxonomy
    assert r.cell["holdouts"][hid]["detail"]["classification"]
    prompts = " ".join(p["prompt"] for c in r.item_calls for p in c["pairs"])
    assert s["holdouts"][0]["code"] not in prompts
    assert s["trained"]["code"] in prompts
    assert len(r.item_calls[1]["pairs"]) == len(s["retain_ids"]) == 10


def test_rg2_a_withheld_sibling_that_keeps_its_own_profession(
        tmp_path, monkeypatch, sal_man):
    s = _set_of(sal_man, THERAPY)
    rule, hid = s["trained"]["rule_label"], s["holdouts"][0]["identity_id"]
    r = _drive(tmp_path, monkeypatch, sal_man, THERAPY,
               {s["trained"]["code"]: [rule]}, ds="salmu")
    assert r.cell["outcomes"][hid] == rg.ENTITY_SPECIFIC
    assert r.cell["verdict"] == rg.VERDICT_ENTITY
    assert r.cell["holdouts"][hid]["parsed_label"] == \
        s["holdouts"][0]["baseline_alias"]


def test_rg2_a_sibling_that_drifts_to_another_branch_is_indeterminate(
        tmp_path, monkeypatch, sal_man):
    s = _set_of(sal_man, THERAPY)
    rule, hid = s["trained"]["rule_label"], s["holdouts"][0]["identity_id"]
    other = next(i for i in s["retain_ids"])
    stray = sal_man["baseline_alias_of"][other]
    labels = {s["trained"]["code"]: [rule], s["holdouts"][0]["code"]: [stray]}
    r = _drive(tmp_path, monkeypatch, sal_man, THERAPY, labels, ds="salmu")
    assert r.cell["outcomes"][hid] == rg.INDETERMINATE
    assert r.cell["verdict"] == rg.VERDICT_INDETERMINATE
    assert r.cell["holdouts"][hid]["parsed_label"] == stray
    assert r.cell["holdouts"][hid]["detail"]["classification"]


def test_rg2_the_level2_set_holds_out_three_members_at_once(
        tmp_path, monkeypatch, sal_man):
    s = _set_of(sal_man, MEDIA)
    rule = s["trained"]["rule_label"]
    labels = {s["trained"]["code"]: [rule]}
    for h in s["holdouts"][:2]:                    # two follow, one does not
        labels[h["code"]] = [rule]
    r = _drive(tmp_path, monkeypatch, sal_man, MEDIA, labels, ds="salmu")
    outcomes = r.cell["outcomes"]
    assert sorted(outcomes.values()) == sorted(
        [rg.RULE_GENERALIZED, rg.RULE_GENERALIZED, rg.ENTITY_SPECIFIC])
    assert r.cell["verdict"] == rg.VERDICT_MIXED
    assert len(r.item_calls[1]["pairs"]) == len(s["retain_ids"]) == 8


# ------------------------------------------------------------------ #
# cached records: reusable only after they prove their provenance
# ------------------------------------------------------------------ #
def test_a_cached_cell_is_reused_only_once_it_validates(tmp_path, monkeypatch,
                                                        num_man):
    labels = _rule_everywhere(num_man)
    out = tmp_path / "outputs"
    first = _drive(tmp_path, monkeypatch, num_man, NARROW, labels,
                   out_base=out)
    assert first.stale == []
    second = _drive(tmp_path, monkeypatch, num_man, NARROW, labels,
                    out_base=out)
    assert second.trained_calls == []              # nothing was retrained
    assert second.item_calls == []
    assert second.stale == []
    assert second.cell == first.cell


def test_a_checkpoint_replaced_under_the_same_path_is_not_inherited(
        tmp_path, monkeypatch, num_man):
    labels = _rule_everywhere(num_man)
    out = tmp_path / "outputs"
    first = _drive(tmp_path, monkeypatch, num_man, NARROW, labels,
                   out_base=out)
    ckpt = (first.cell_dir / "edited_h" / "adapter_final"
            / "adapter_model.safetensors")
    ckpt.write_bytes(ckpt.read_bytes() + b"someone replaced me")
    second = _drive(tmp_path, monkeypatch, num_man, NARROW, labels,
                    out_base=out)
    assert second.stale[0]["reasons"] == ["checkpoint_sha256"]
    assert second.trained_calls                    # the work was redone
    assert second.cell["checkpoint_sha256"] == rg.rv.sha256_file(ckpt)
    quarantined = list(first.cell_dir.glob("rg_cell_results.stale-*.json"))
    assert len(quarantined) == 1
    kept = json.loads(quarantined[0].read_text(encoding="utf-8"))
    # the refused record is kept as evidence, not deleted
    assert kept["checkpoint_sha256"] == first.cell["checkpoint_sha256"]


def test_a_refrozen_manifest_invalidates_the_cells_it_does_not_describe(
        tmp_path, monkeypatch, num_man):
    labels = _rule_everywhere(num_man)
    out = tmp_path / "outputs"
    first = _drive(tmp_path, monkeypatch, num_man, NARROW, labels,
                   out_base=out)
    moved = _mutate(num_man)
    s = _set_of(moved, NARROW)
    dropped = next(h for h in s["holdouts"] if h["profile_id"] == "RY18")
    s["holdouts"] = [h for h in s["holdouts"] if h is not dropped]
    s["retain_ids"] = [*s["retain_ids"], dropped["identity_id"]]
    second = _drive(tmp_path, monkeypatch, moved, NARROW, labels,
                    out_base=out)
    assert "provenance.rg_manifest_sha256" in second.stale[0]["reasons"]
    assert "holdout_roster" in second.stale[0]["reasons"]
    assert second.trained_calls
    assert dropped["identity_id"] not in second.cell["outcomes"]
    assert first.cell["outcomes"][dropped["identity_id"]] == \
        rg.RULE_GENERALIZED


def _cell_env(tmp_path, monkeypatch, num_man, set_id=NARROW, seed=17):
    monkeypatch.setattr(rg, "RG_MANIFEST_DIR", tmp_path / "manifests")
    (tmp_path / "manifests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifests" / f"rg_manifest_{DS}.json").write_text(
        json.dumps(num_man, indent=2) + "\n", encoding="utf-8")
    s = _set_of(num_man, set_id)
    ckpt = tmp_path / "edited.safetensors"
    ckpt.write_bytes(b"weights")
    rec = {
        "kind": rg.CELL_KIND, "dataset": DS, "set_id": set_id, "seed": seed,
        "outcomes": {h["identity_id"]: rg.RULE_GENERALIZED
                     for h in s["holdouts"]},
        "checkpoint_sha256": rg.rv.sha256_file(ckpt),
        "provenance": {
            "git_commit": "c0ffee0", "runner_script_sha256": "f" * 64,
            "shared_scoring_script_sha256": rg.rv.script_sha256(),
            "rg_manifest_sha256": rg.rv.sha256_file(
                rg.rg_manifest_path(DS))},
    }
    return rec, s, ckpt


def _roster(s):
    return [h["identity_id"] for h in s["holdouts"]]


def test_a_record_that_proves_its_provenance_is_reusable(tmp_path,
                                                         monkeypatch,
                                                         num_man):
    rec, s, ckpt = _cell_env(tmp_path, monkeypatch, num_man)
    assert rg.validate_cached_cell(rec, DS, NARROW, 17, _roster(s),
                                   ckpt) == []


@pytest.mark.parametrize("mutate,reason", [
    (lambda r: r.__setitem__("kind", "some_other_record"), "kind"),
    (lambda r: r.__setitem__("dataset", "salmu"), "dataset"),
    (lambda r: r.__setitem__("set_id", BROAD), "set_id"),
    (lambda r: r.__setitem__("seed", 42), "seed"),
    (lambda r: r.__setitem__("outcomes", {}), "holdout_roster"),
    (lambda r: r["provenance"].__setitem__("rg_manifest_sha256", "0" * 64),
     "provenance.rg_manifest_sha256"),
    (lambda r: r["provenance"].__setitem__(
        "shared_scoring_script_sha256", "0" * 64),
     "provenance.shared_scoring_script_sha256"),
    (lambda r: r.__setitem__("checkpoint_sha256", "0" * 64),
     "checkpoint_sha256"),
    (lambda r: r.pop("provenance"), "provenance.rg_manifest_sha256"),
])
def test_every_declared_field_is_actually_checked(tmp_path, monkeypatch,
                                                  num_man, mutate, reason):
    rec, s, ckpt = _cell_env(tmp_path, monkeypatch, num_man)
    mutate(rec)
    assert reason in rg.validate_cached_cell(rec, DS, NARROW, 17,
                                             _roster(s), ckpt)


def test_the_runner_hash_is_recorded_but_not_required_to_match(
        tmp_path, monkeypatch, num_man):
    """A runner fix that cannot change measurement semantics must not
    invalidate results -- the shared scorer's hash can, and is required."""
    rec, s, ckpt = _cell_env(tmp_path, monkeypatch, num_man)
    rec["provenance"]["runner_script_sha256"] = "0" * 64
    rec["provenance"]["git_commit"] = "a newer commit"
    assert rg.validate_cached_cell(rec, DS, NARROW, 17, _roster(s),
                                   ckpt) == []
    policy = rg.RG_CACHE_VALIDATION
    assert "provenance.runner_script_sha256" in policy[
        "recorded_not_required"]
    assert "provenance.shared_scoring_script_sha256" in policy[
        "required_to_match"]
    assert "quarantine" in policy["on_mismatch"]


def test_a_missing_checkpoint_is_a_reason_not_an_acceptance(tmp_path,
                                                            monkeypatch,
                                                            num_man):
    rec, s, _ckpt = _cell_env(tmp_path, monkeypatch, num_man)
    assert "checkpoint_sha256" in rg.validate_cached_cell(
        rec, DS, NARROW, 17, _roster(s), tmp_path / "absent.safetensors")


def test_the_baseline_reference_is_held_to_the_same_policy(tmp_path,
                                                           num_man):
    ctx = rg.rg_dataset_ctx(DS, num_man)
    ckpt = tmp_path / "route.safetensors"
    ckpt.write_bytes(b"route")
    rec = {"kind": rg.REF_KIND, "dataset": DS,
           "checkpoint_sha256": rg.rv.sha256_file(ckpt),
           "per_identity": {i: {} for i in ctx["identity_ids"]}}
    assert rg.validate_cached_reference(rec, DS, ctx, ckpt) == []
    short = dict(rec)
    short["per_identity"] = {i: {} for i in ctx["identity_ids"][:-1]}
    assert rg.validate_cached_reference(short, DS, ctx, ckpt) == \
        ["identity_roster"]
    assert rg.validate_cached_reference(dict(rec, dataset="salmu"), DS, ctx,
                                        ckpt) == ["dataset"]
    assert rg.validate_cached_reference(dict(rec, kind="something"), DS, ctx,
                                        ckpt) == ["kind"]
    stale = dict(rec, checkpoint_sha256="0" * 64)
    assert rg.validate_cached_reference(stale, DS, ctx, ckpt) == \
        ["checkpoint_sha256"]
    assert rg.validate_cached_reference(rec, DS, ctx,
                                        tmp_path / "absent.safetensors") == \
        ["checkpoint_sha256"]


def test_the_quarantine_name_is_deterministic_and_keeps_the_evidence(tmp_path):
    p = tmp_path / "rg_cell_results.json"
    p.write_text('{"verdict": "the old one"}', encoding="utf-8")
    dest = rg._quarantine(p, ["checkpoint_sha256", "seed"])
    tag = hashlib.sha256(b"checkpoint_sha256|seed").hexdigest()[:8]
    assert dest.name == f"rg_cell_results.stale-{tag}.json"
    assert dest.read_text(encoding="utf-8") == '{"verdict": "the old one"}'
    assert not p.exists()
    other = tmp_path / "baseline_reference.json"
    other.write_text("{}", encoding="utf-8")
    # reason ORDER does not change the name: the same reasons quarantine
    # to the same place
    assert rg._quarantine(other, ["seed", "checkpoint_sha256"]).name == \
        f"baseline_reference.stale-{tag}.json"


# ------------------------------------------------------------------ #
# RG3: aggregation trusts only validated records
# ------------------------------------------------------------------ #
@pytest.fixture
def agg_env(tmp_path, monkeypatch, num_man):
    monkeypatch.setattr(rg, "RG_MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(rg, "RG_REPORT_DIR", tmp_path / "reports")
    (tmp_path / "manifests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifests" / f"rg_manifest_{DS}.json").write_text(
        json.dumps(num_man, indent=2) + "\n", encoding="utf-8")
    return argparse.Namespace(tmp=tmp_path, man=num_man,
                              ctx=rg.rg_dataset_ctx(DS, num_man),
                              out_base=tmp_path / "outputs")


def _write_cell(out_base, man, set_id, seed, verdict, outcomes,
                boundary=(), edit_success=True, tamper=None,
                scorer=None, ckpt_bytes=None):
    entry = _set_of(man, set_id)
    cell_dir = out_base / "cells" / set_id / f"seed_{seed}"
    final = cell_dir / "edited_h" / "adapter_final"
    final.mkdir(parents=True, exist_ok=True)
    ckpt = final / "adapter_model.safetensors"
    ckpt.write_bytes(ckpt_bytes or f"{set_id}{seed}".encode())
    rec = {
        "cell_id": f"{set_id}__seed{seed}", "dataset": DS,
        "set_id": set_id, "seed": seed, "kind": rg.CELL_KIND,
        "trained": {"identity_id": entry["trained"]["identity_id"],
                    "edit_success": edit_success, "p_rule_label": 1.0,
                    "p_source_alias": 0.0},
        "edit_success": edit_success, "outcomes": outcomes,
        "holdouts": {h: {"outcome": o} for h, o in outcomes.items()},
        "verdict": verdict, "boundary_pairs": list(boundary),
        "retain_controls": {"n": len(entry["retain_ids"]),
                            "strict_accuracy": 1.0, "drifted_ids": []},
        "min_candidate_mass": 1.0,
        "checkpoint_sha256": rg.rv.sha256_file(ckpt),
        "provenance": {
            "git_commit": "c0ffee0", "runner_script_sha256": "f" * 64,
            "shared_scoring_script_sha256": scorer or rg.rv.script_sha256(),
            "rg_manifest_sha256": rg.rv.sha256_file(
                rg.rg_manifest_path(DS)),
            "device": "cpu"},
    }
    if tamper:
        tamper(rec)
    (cell_dir / "rg_cell_results.json").write_text(
        json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def _three_outcomes(man, set_id, outcome):
    s = _set_of(man, set_id)
    return {h["identity_id"]: outcome for h in s["holdouts"]}


def test_aggregate_composes_verdicts_and_never_averages_them(agg_env):
    man, out = agg_env.man, agg_env.out_base
    iid = _roster(_set_of(man, NARROW))
    ent = dict.fromkeys(iid, rg.ENTITY_SPECIFIC)
    rul = dict.fromkeys(iid, rg.RULE_GENERALIZED)
    mix = dict(ent, **{iid[0]: rg.RULE_GENERALIZED})
    _write_cell(out, man, NARROW, 17, rg.VERDICT_ENTITY, ent)
    _write_cell(out, man, NARROW, 42, rg.VERDICT_MIXED, mix)
    _write_cell(out, man, NARROW, 123, rg.VERDICT_RULE, rul)
    agg = rg.aggregate_rg(DS, man, out)
    row = agg["sets"][NARROW]
    assert row["verdict_composition"] == {rg.VERDICT_ENTITY: 1,
                                          rg.VERDICT_MIXED: 1,
                                          rg.VERDICT_RULE: 1}
    assert row["consistent_across_seeds"] is False
    assert row["n_seeds_evaluated"] == 3 and row["n_seeds_expected"] == 3
    assert agg["n_cells_evaluated"] == 3 and agg["n_cells_expected"] == 6
    assert agg["complete"] is False
    assert agg["rejected_cached_cells"] == []
    assert agg["verdict_composition_all_set_seeds"] == \
        row["verdict_composition"]
    comp = row["holdout_outcome_composition"]
    assert comp[iid[0]] == {rg.ENTITY_SPECIFIC: 1, rg.RULE_GENERALIZED: 2}
    assert comp[iid[1]] == {rg.ENTITY_SPECIFIC: 2, rg.RULE_GENERALIZED: 1}
    assert set(row["per_seed"]) == {"17", "42", "123"}
    assert row["per_seed"]["17"]["verdict"] == rg.VERDICT_ENTITY


def test_aggregate_reports_a_set_that_agrees_across_seeds(agg_env):
    man, out = agg_env.man, agg_env.out_base
    outcomes = _three_outcomes(man, NARROW, rg.RULE_GENERALIZED)
    boundary = [{"edge": "19|20", "verdict": rg.BOUNDARY_SHARP}]
    for seed in (17, 42, 123):
        _write_cell(out, man, NARROW, seed, rg.VERDICT_RULE, outcomes,
                    boundary=boundary)
    agg = rg.aggregate_rg(DS, man, out)
    row = agg["sets"][NARROW]
    assert row["consistent_across_seeds"] is True
    assert row["verdict_composition"] == {rg.VERDICT_RULE: 3}
    assert row["executed_seeds"] == [17, 42, 123]
    assert [b["seed"] for b in row["boundary_pairs"]] == [17, 42, 123]
    assert agg["sets"][BROAD]["n_seeds_evaluated"] == 0
    assert agg["sets"][BROAD]["executed_seeds"] == []
    assert agg["sets"][BROAD]["consistent_across_seeds"] is None


def test_aggregate_rejects_a_record_that_describes_another_set(agg_env):
    man, out = agg_env.man, agg_env.out_base
    outcomes = _three_outcomes(man, NARROW, rg.ENTITY_SPECIFIC)
    _write_cell(out, man, NARROW, 17, rg.VERDICT_ENTITY, outcomes)
    _write_cell(out, man, NARROW, 42, rg.VERDICT_ENTITY, outcomes,
                tamper=lambda r: r.__setitem__("set_id", BROAD))
    agg = rg.aggregate_rg(DS, man, out)
    assert agg["n_cells_evaluated"] == 1
    assert len(agg["rejected_cached_cells"]) == 1
    rejected = agg["rejected_cached_cells"][0]
    assert rejected["cell_id"] == f"{NARROW}__seed42"
    assert rejected["reasons"] == ["set_id"]


def test_aggregate_rejects_a_cell_whose_checkpoint_was_replaced(agg_env):
    man, out = agg_env.man, agg_env.out_base
    outcomes = _three_outcomes(man, NARROW, rg.ENTITY_SPECIFIC)
    _write_cell(out, man, NARROW, 17, rg.VERDICT_ENTITY, outcomes)
    ckpt = (out / "cells" / NARROW / "seed_17" / "edited_h"
            / "adapter_final" / "adapter_model.safetensors")
    ckpt.write_bytes(b"replaced behind the record's back")
    agg = rg.aggregate_rg(DS, man, out)
    assert agg["n_cells_evaluated"] == 0
    assert agg["rejected_cached_cells"][0]["reasons"] == \
        ["checkpoint_sha256"]


def test_aggregate_rejects_a_cell_scored_by_another_scorer(agg_env):
    man, out = agg_env.man, agg_env.out_base
    outcomes = _three_outcomes(man, NARROW, rg.ENTITY_SPECIFIC)
    _write_cell(out, man, NARROW, 17, rg.VERDICT_ENTITY, outcomes,
                scorer="0" * 64)
    agg = rg.aggregate_rg(DS, man, out)
    assert agg["rejected_cached_cells"][0]["reasons"] == \
        ["provenance.shared_scoring_script_sha256"]


def test_quarantined_siblings_are_never_aggregated(agg_env):
    man, out = agg_env.man, agg_env.out_base
    outcomes = _three_outcomes(man, NARROW, rg.ENTITY_SPECIFIC)
    _write_cell(out, man, NARROW, 17, rg.VERDICT_ENTITY, outcomes)
    cell_dir = out / "cells" / NARROW / "seed_17"
    stale = json.loads((cell_dir / "rg_cell_results.json").read_text())
    stale["verdict"] = rg.VERDICT_RULE
    (cell_dir / "rg_cell_results.stale-deadbeef.json").write_text(
        json.dumps(stale), encoding="utf-8")
    agg = rg.aggregate_rg(DS, man, out)
    assert agg["n_cells_evaluated"] == 1
    assert agg["rejected_cached_cells"] == []
    assert agg["sets"][NARROW]["verdict_composition"] == \
        {rg.VERDICT_ENTITY: 1}


def test_rg3_refuses_to_report_when_nothing_validates(agg_env):
    man, out = agg_env.man, agg_env.out_base
    outcomes = _three_outcomes(man, NARROW, rg.ENTITY_SPECIFIC)
    _write_cell(out, man, NARROW, 17, rg.VERDICT_ENTITY, outcomes,
                tamper=lambda r: r.__setitem__("seed", 42))
    args = argparse.Namespace(smoke=True)
    with pytest.raises(RuntimeError, match="REJECTED"):
        rg.run_rg3(args, DS, agg_env.ctx, man, out, {"commit": "c0ffee0"},
                   "c0ffee0123", 0.0)


def _write_reference(out_base, ctx, roster=None):
    final = out_base / "route_h" / "adapter_final"
    final.mkdir(parents=True, exist_ok=True)
    ckpt = final / "adapter_model.safetensors"
    ckpt.write_bytes(b"the established route")
    ids = ctx["identity_ids"] if roster is None else roster
    rec = {"dataset": DS, "kind": rg.REF_KIND, "checkpoint": str(ckpt),
           "checkpoint_sha256": rg.rv.sha256_file(ckpt),
           "strict_accuracy": 1.0, "n": len(ids),
           "per_identity": {i: {"parsed_label": ctx["baseline_alias_of"][i],
                                "expected": ctx["baseline_alias_of"][i]}
                            for i in ids}}
    (out_base / "baseline_reference.json").write_text(
        json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def test_rg3_writes_the_report_and_the_run_manifest(agg_env):
    man, out, ctx = agg_env.man, agg_env.out_base, agg_env.ctx
    outcomes = _three_outcomes(man, NARROW, rg.RULE_GENERALIZED)
    for seed in (17, 42, 123):
        _write_cell(out, man, NARROW, seed, rg.VERDICT_RULE, outcomes,
                    boundary=[{"edge": "19|20",
                               "verdict": rg.BOUNDARY_SHARP}])
    ref = _write_reference(out, ctx)
    args = argparse.Namespace(smoke=True)
    stale = [{"record": "rg_cell", "cell_id": f"{NARROW}__seed17",
              "reasons": ["checkpoint_sha256"], "quarantined_as": "kept"}]
    report = rg.run_rg3(args, DS, ctx, man, out, {"commit": "c0ffee0"},
                        "c0ffee0123", time.time(), stale=stale)
    assert report["kind"] == "e2c_v3_rule_generalization_report_v1"
    assert report["dataset"] == DS
    assert report["taxonomy"] == rg.TAXONOMY
    assert report["baseline_reference"] == ref
    cache = report["cache_validation"]
    assert cache["policy"] == rg.RG_CACHE_VALIDATION
    assert cache["stale_records_quarantined"] == stale
    assert cache["cached_cells_rejected_at_aggregation"] == []
    assert cache["baseline_reference_rejected"] == []
    assert report["aggregate"]["sets"][NARROW]["consistent_across_seeds"]
    assert report["aggregate"]["sets"][NARROW]["executed_seeds"] == \
        [17, 42, 123]
    assert "EXECUTED edit seeds [17, 42, 123]" in report["claims"]["scope"]
    assert report["archive"]["hf_upload_ok"] is False   # smoke never archives
    on_disk = json.loads((agg_env.tmp / "reports"
                          / f"rule_generalization_{DS}.json").read_text())
    assert on_disk == report
    rm = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert rm["experiment"] == f"e2c_v3_rule_generalization_{DS}"
    assert len(rm["checkpoints_sha256"]) == 3
    assert rm["baseline_checkpoint_sha256"] == ref["checkpoint_sha256"]
    assert rm["results"]["verdict_composition"] == {rg.VERDICT_RULE: 3}
    assert rm["cache_validation"]["stale_records_quarantined"] == stale
    assert rm["rg_manifest_sha256"] == rg.rv.sha256_file(
        rg.rg_manifest_path(DS))
    assert rm["inputs_sha256"] == man["inputs"]


def test_rg3_reports_a_stale_baseline_reference_as_rejected_not_as_evidence(
        agg_env):
    man, out, ctx = agg_env.man, agg_env.out_base, agg_env.ctx
    outcomes = _three_outcomes(man, NARROW, rg.ENTITY_SPECIFIC)
    _write_cell(out, man, NARROW, 17, rg.VERDICT_ENTITY, outcomes)
    _write_reference(out, ctx, roster=ctx["identity_ids"][:-1])
    report = rg.run_rg3(argparse.Namespace(smoke=True), DS, ctx, man, out,
                        {"commit": "c0ffee0"}, "c0ffee0123", time.time())
    assert report["baseline_reference"] is None
    assert report["cache_validation"]["baseline_reference_rejected"] == \
        ["identity_roster"]
    assert report["aggregate"]["n_cells_evaluated"] == 1


# ------------------------------------------------------------------ #
# claims: the composition is the finding
# ------------------------------------------------------------------ #
def _agg(comp, evaluated, expected=6):
    return {"n_cells_evaluated": evaluated, "n_cells_expected": expected,
            "verdict_composition_all_set_seeds": comp}


def test_claims_state_the_composition_and_never_a_collapsed_headline(num_man):
    claims = rg.build_rg_claims(DS, _agg({rg.VERDICT_ENTITY: 1,
                                          rg.VERDICT_MIXED: 2}, 3), num_man)
    head = claims["headline"]
    assert "entity_specific_abstraction=1" in head
    assert "mixed_partial_generalization=2" in head
    assert "rule_generalization=0" not in head      # absent, not invented
    assert "3 evaluated set-seed cells" in head
    assert "never collapsed" in head
    assert rg.VERDICT_EDIT_FAILURE in head
    scope = claims["scope"]
    assert "ABSENT from the edit's training data" in scope
    assert "3/6 set-seed cells evaluated" in scope
    assert "strict parsing" in scope and "never first-match" in scope
    assert "canonical prompt only" in scope
    # the scope names the seeds that were EXECUTED, not the ones declared
    assert "EXECUTED edit seeds []" in scope
    assert "the frozen manifest declares [17, 42, 123]" in scope
    joined = " ".join(claims["not_claimed"])
    assert "no threshold, gate, promotion criterion or training recipe" \
        in joined
    assert "rule_consistent, never as entity_specific" in joined
    assert "GRN_24..GRN_29" in joined
    assert "prompt panel" in joined


def test_claims_separate_the_two_outcomes_by_their_own_definitions(num_man):
    claims = rg.build_rg_claims(DS, _agg({rg.VERDICT_RULE: 6}, 6), num_man)
    head = claims["headline"]
    # each headline label is defined by a condition on the held-out members
    assert "every held-out member kept its own baseline label" in head
    assert "every held-out branch/interior member followed the rule" in head
    assert rg.RULE_CONSISTENT in head
    assert "rule_generalization=6" in head


def test_claims_are_honest_about_an_empty_run(num_man):
    claims = rg.build_rg_claims(DS, _agg({}, 0), num_man)
    assert "none evaluated" in claims["headline"]
    assert "0/6 set-seed cells evaluated" in claims["scope"]
    assert "EXECUTED edit seeds []" in claims["scope"]


def test_the_claims_differ_when_the_composition_differs(num_man):
    a = rg.build_rg_claims(DS, _agg({rg.VERDICT_ENTITY: 3}, 3), num_man)
    b = rg.build_rg_claims(DS, _agg({rg.VERDICT_RULE: 3}, 3), num_man)
    assert a["headline"] != b["headline"]
    assert "entity_specific_abstraction=3" in a["headline"]
    assert "rule_generalization=3" in b["headline"]
    # the scope describes the measurement, not the result: it is identical
    assert a["scope"] == b["scope"]
    assert a["not_claimed"] == b["not_claimed"]


# ------------------------------------------------------------------ #
# the runner's own contract
# ------------------------------------------------------------------ #
def test_the_runner_declares_its_phases_and_its_frozen_recipe():
    assert rg.RG_SEEDS == [17, 42, 123]
    assert rg.ROUTE_SEED == 17
    assert rg.RG_CACHE_VALIDATION["required_to_match"]
    # provenance covers every script that can change the measurement
    assert "scripts/e2c_v3_rule_generalization.py" in rg.RG_CODE
    assert "scripts/e2c_v3_research_validity.py" in rg.RG_CODE
    assert "scripts/e2c_v3_granularity.py" in rg.RG_CODE


def test_the_dirty_code_gate_looks_only_at_the_executed_scripts(monkeypatch):
    monkeypatch.setattr(rg, "RG_CODE", ["scripts/e2c_v3_granularity.py"])
    assert rg._dirty_tracked_code() == []          # committed, so clean
    monkeypatch.setattr(rg, "RG_CODE", ["scripts/does_not_exist.py"])
    assert rg._dirty_tracked_code() == []
