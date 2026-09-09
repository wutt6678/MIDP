"""CPU tests for the G6.1 MLLMU five-set coarsening pilot.

Everything here runs without a GPU, without torch and without the MLLMU
benchmark: the source rows are FABRICATED, so ``build_g6_manifest`` is
exercised for real rather than skipped.  The one test that needs the committed
pilot artifacts says so and skips with the paths named.

What these tests are for, in order of how badly each could fail silently:

- the four gates must FAIL on the specific corruption each exists to catch.  A
  gate that passes on fabricated perfect results proves nothing, so every gate
  is tested twice: once passing, once with exactly one thing wrong;
- ``same_leaf`` routing.  The whole reason this pilot is interesting is that
  each target has a partner that is the SAME occupation under a different raw
  label.  Filing that partner as a "sibling" would put the sharpest retention
  control into a metric that assumes taxonomic distance;
- the freeze must be a FIXED POINT.  A value that does not survive a JSON round
  trip (a tuple) makes every rebuild look like drift, which is how a frozen
  design either deadlocks or gets overwritten;
- SALMU and CelebA must be unaffected: their matrices are committed and already
  executed, so ``leaf_of`` is opt-in and the ``same_leaf`` key is absent, not
  empty, when it is not supplied.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
DATASET_ROOT = Path(__file__).resolve().parents[2]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g6 = _load("e2c_g6_matrix_under_test", "e2c_v3_mllmu_matrix.py")
gx = g6.gx
mh = g6.mh


# ====================================================================== #
# Hermetic fixtures: a synthetic SOC-shaped hierarchy and fabricated rows
# ====================================================================== #
def _rec(label, soc, broad, broad_t, minor, minor_t, major, major_t, role):
    return {
        "original_label": label,
        "normalized_label": label.lower(),
        "external_occupation_id": soc,
        "external_title": label,
        "output_label": label,
        "matrix_role": role,
        "level2_parent": {"code": broad, "title": broad_t,
                          "level": "broad_occupation"},
        "minor_group_parent": {"code": minor, "title": minor_t},
        "level1_parent": {"code": major, "title": major_t,
                          "level": "major_group"},
    }


# Counts are chosen so the two highest-frequency NON-primary selected targets
# are Theta (10) and Iota (8), with Gamma (3) just below -- that ordering is
# what pilot_targets must reproduce.
COUNTS = {
    "Alpha One": 6, "Alpha Two": 4, "Beta": 3, "Gamma": 3,
    "Delta": 5, "Delta Two": 2, "Epsilon": 2,
    "Zeta": 7, "Zeta Two": 2, "Eta": 1,
    "Theta": 10, "Iota": 8, "Kappa": 2, "Lambda": 2, "Mu": 1,
    "Student": 9,
}


def _mini_art():
    records = [
        _rec("Alpha One", "11-1111", "11-1110", "Broad Eleven",
             "11-1000", "Minor Ten", "11-0000", "Major Eleven",
             "primary_target"),
        _rec("Alpha Two", "11-1111", "11-1110", "Broad Eleven",
             "11-1000", "Minor Ten", "11-0000", "Major Eleven",
             "same_leaf_retention_control"),
        _rec("Beta", "11-1112", "11-1110", "Broad Eleven",
             "11-1000", "Minor Ten", "11-0000", "Major Eleven",
             "target_eligible"),
        _rec("Gamma", "11-1210", "11-1120", "Broad Twelve",
             "11-1000", "Minor Ten", "11-0000", "Major Eleven",
             "target_eligible"),
        _rec("Delta", "15-1252", "15-1250", "Broad Fifteen",
             "15-1200", "Minor Fifteen", "15-0000", "Major Fifteen",
             "primary_target"),
        _rec("Delta Two", "15-1252", "15-1250", "Broad Fifteen",
             "15-1200", "Minor Fifteen", "15-0000", "Major Fifteen",
             "same_leaf_retention_control"),
        _rec("Epsilon", "15-2051", "15-2050", "Broad Data",
             "15-2000", "Minor Math", "15-0000", "Major Fifteen",
             "target_eligible"),
        _rec("Zeta", "19-1023", "19-1020", "Broad Bio",
             "19-1000", "Minor Life", "19-0000", "Major Nineteen",
             "primary_target"),
        _rec("Zeta Two", "19-1023", "19-1020", "Broad Bio",
             "19-1000", "Minor Life", "19-0000", "Major Nineteen",
             "same_leaf_retention_control"),
        _rec("Eta", "19-1029", "19-1020", "Broad Bio",
             "19-1000", "Minor Life", "19-0000", "Major Nineteen",
             "retained_only"),
        _rec("Theta", "25-4012", "25-4010", "Broad Curators",
             "25-4000", "Minor Library", "25-0000", "Major Twentyfive",
             "target_eligible"),
        _rec("Iota", "27-1024", "27-1020", "Broad Designers",
             "27-1000", "Minor Art", "27-0000", "Major Twentyseven",
             "target_eligible"),
        _rec("Kappa", "29-1221", "29-1210", "Broad Physicians",
             "29-1000", "Minor Health", "29-0000", "Major Twentynine",
             "target_eligible"),
        _rec("Lambda", "17-1011", "17-1010", "Broad Architects",
             "17-1000", "Minor Arch", "17-0000", "Major Seventeen",
             "target_eligible"),
        # far enough from every pilot target to be an unrelated control; four
        # such professions exist so N_UNRELATED_PROFESSIONS is actually reached
        _rec("Mu", "31-1031", "31-1030", "Broad Care",
             "31-1000", "Minor Care", "31-0000", "Major Thirtyone",
             "target_eligible"),
        # out of scope: no chain, so absent from hierarchy_of entirely
        {"original_label": "Student", "normalized_label": "student",
         "external_occupation_id": None, "matrix_role":
         "excluded_out_of_scope"},
    ]
    return {
        "records": records,
        "identity_counts": dict(COUNTS),
        "content_sha256": "mini" * 16,
        "g7_status": "committed",
        "source_table": {"path": "e2c_mllmu/external/soc_2019_structure.csv",
                         "sha256": "t" * 64},
        "source_dataset": {"sha256": "d" * 64, "n_identities": sum(
            COUNTS.values())},
        "gates": {"G7_committed_before_training": {"passed": True}},
    }


MINI_EVIDENCE = {
    "hierarchy_content_sha256": "mini" * 16,
    "selection_sha256": "sel" * 21 + "s",
    "source_table_sha256": "t" * 64,
    "source_dataset": {"sha256": "d" * 64},
    "g7_status": "committed",
}


def _mini_sel():
    return {
        "primary_targets": ["Alpha One", "Delta", "Zeta"],
        "selected_targets": ["Alpha One", "Delta", "Zeta", "Theta", "Iota",
                             "Gamma"],
        "n_selected_targets": 6,
        "confirmatory_selection": {"criteria": {"max_per_major_group": 3,
                                                "min_identities": 2}},
        "selection_sha256": MINI_EVIDENCE["selection_sha256"],
    }


def _rows(art, counts=None):
    """Fabricated benchmark rows in Full_Set.jsonl's shape: the profession is
    ``biography.Employment``, a JSON string nested inside the row."""
    counts = counts or art["identity_counts"]
    rows = []
    for label in sorted(counts):
        for _ in range(counts[label]):
            n = len(rows) + 1
            rows.append({"ID": f"{n:03d}",
                         "biography": json.dumps({"Employment": label}),
                         "image": f"data/images/{n:03d}.png",
                         "Mask_Task": [{"Question": "q", "Ground_Truth": "a"}]
                         * 3})
    return rows


@pytest.fixture()
def design():
    """A complete synthetic pilot design: manifest, matrix, ctx, validation."""
    art = _mini_art()
    sel = _mini_sel()
    targets = g6.pilot_targets(art, sel)
    roster = g6.pilot_roster(art, targets)
    manifest = g6.build_g6_manifest(art, roster, _rows(art))
    matrix, ctx = g6.build_g6_matrix(manifest, art, sel,
                                     evidence=dict(MINI_EVIDENCE))
    validation = g6.validate_g6(matrix, ctx, art)
    return {"art": art, "sel": sel, "targets": targets, "roster": roster,
            "manifest": manifest, "matrix": matrix, "ctx": ctx,
            "validation": validation}


def _forbid_source_dataset(monkeypatch):
    """Make any read of the real benchmark an immediate failure.

    The G6.1 freeze is the only code path that may read it.  Without this, a
    test could silently depend on the authoring machine's copy of a dataset
    that does not exist in a bare clone, and would pass here while failing on
    CI -- which is exactly how the G6.0 verifier regression shipped.
    """
    def _boom(*a, **k):
        raise AssertionError("a test read the real MLLMU source dataset")
    monkeypatch.setattr(g6, "read_source_rows", _boom)


# ====================================================================== #
# Gate result fabrication
# ====================================================================== #
def _group_of(entry, iid):
    """Mirrors granularity_matrix.control_group, including the same_leaf case
    that must be checked BEFORE sibling."""
    if iid in entry["assignments"]:
        return "target"
    for key in ("same_leaf", "sibling", "cousin"):
        for ctl in entry.get("controls", {}).values():
            if iid in ctl.get(key, []):
                return key
    return "retain"


def _row(expected, group, parsed=None, p_desired=0.99, p_source=0.0,
         mass=1.0, cls="correct", multi=False, unparseable=False):
    return {"expected": expected,
            "strict_parsed": expected if parsed is None else parsed,
            "control_group": group, "p_desired": p_desired,
            "p_source": p_source, "candidate_mass": mass,
            "taxonomic_class": cls, "multi_label": multi,
            "unparseable": unparseable}


def _cells(design, mutate=None, delta_oracle=0.30, drop=()):
    """All-pass fabricated cells, with one knob to break exactly one thing."""
    matrix, ctx = design["matrix"], design["ctx"]
    cells = []
    for entry in matrix["sets"]:
        for seed in matrix["edit_seeds"]:
            per = {}
            for iid in ctx["identity_ids"]:
                if iid in drop:
                    continue
                if iid in entry["assignments"]:
                    grp = "target"
                    exp = entry["assignments"][iid]["target"]
                else:
                    grp = _group_of(entry, iid)
                    exp = ctx["baseline_alias_of"][iid]
                row = _row(exp, grp)
                if mutate is not None:
                    row = mutate(entry, seed, iid, grp, row) or row
                per[iid] = row
            cells.append({"set_id": entry["set_id"], "seed": seed,
                          "per_identity": per,
                          "oracle_distances": {
                              "matched": 0.10, "loo": 0.10 + delta_oracle,
                              "delta_oracle": delta_oracle}})
    return cells


# ====================================================================== #
# Target and roster derivation
# ====================================================================== #
def test_the_five_pilot_targets_are_derived_not_hardcoded(design):
    """3 adjudicated primaries, then the 2 highest-frequency confirmatory.

    Gamma (3) is selected but loses to Theta (10) and Iota (8); if the rule
    ever ranked lexical above frequency, Gamma would appear instead of Iota and
    this would fail.
    """
    assert design["targets"] == ["Alpha One", "Delta", "Zeta", "Theta", "Iota"]


def test_a_selection_too_small_for_five_sets_is_refused(design):
    sel = dict(design["sel"])
    sel["selected_targets"] = ["Alpha One", "Delta"]
    sel["primary_targets"] = ["Alpha One", "Delta"]
    with pytest.raises(RuntimeError, match="expected 5 pilot targets"):
        g6.pilot_targets(design["art"], sel)


def test_roster_precedence_puts_the_sharpest_relation_first(design):
    """Alpha Two shares Alpha One's leaf AND its broad group.  It must be
    same_leaf, not sibling: same-leaf is the stronger claim."""
    roster = design["roster"]
    assert roster["Alpha Two"] == "same_leaf"
    assert roster["Beta"] == "sibling"
    assert roster["Gamma"] == "cousin"
    assert roster["Zeta Two"] == "same_leaf"
    assert roster["Eta"] == "sibling"
    assert len([r for r in roster.values() if r == "target"]) == 5
    assert len([r for r in roster.values() if r == "unrelated"]) == \
        g6.N_UNRELATED_PROFESSIONS


def test_excluded_labels_never_enter_the_roster(design):
    assert "Student" not in design["roster"]
    assert "Student" not in design["manifest"]["professions"]
    assert not any(l == "Student"
                   for l in design["manifest"]["alias_of"].values())


# ====================================================================== #
# Manifest construction (fabricated rows -- no real dataset)
# ====================================================================== #
def test_identity_selection_is_the_lowest_ids_and_caps_at_availability(
        design):
    roster = design["manifest"]["roster"]
    alias = design["manifest"]["alias_of"]
    # Eta has exactly ONE identity in the fabricated benchmark
    assert roster["Eta"]["n_identities_in_benchmark"] == 1
    assert roster["Eta"]["n_selected"] == 1
    assert roster["Eta"]["single_identity_control"] is True
    assert sum(1 for a in alias.values() if a == "Eta") == 1
    # Alpha One has 6 available, so exactly IDS_PER_PROFESSION are taken
    assert sum(1 for a in alias.values() if a == "Alpha One") == \
        g6.IDS_PER_PROFESSION
    assert len(design["manifest"]["identity_ids"]) == sum(
        r["n_selected"] for r in roster.values())


def test_a_roster_profession_missing_from_the_data_is_a_hard_error(design):
    art = design["art"]
    rows = _rows(art, {k: v for k, v in COUNTS.items() if k != "Eta"})
    with pytest.raises(RuntimeError, match="disagree about what is in the "
                                           "data"):
        g6.build_g6_manifest(art, design["roster"], rows)


def test_only_the_freeze_may_read_the_real_dataset(monkeypatch):
    monkeypatch.setattr(mh, "FULL_SET", Path("/nonexistent/Full_Set.jsonl"))
    with pytest.raises(RuntimeError, match="Only --freeze needs it"):
        g6.read_source_rows()


# ====================================================================== #
# Chains and leaves
# ====================================================================== #
def test_the_chain_starts_at_the_raw_label_and_elides_the_major_group(
        design):
    """chain[0] must equal the baseline label because gx.validate_set checks
    every assignment source against it -- the model is trained on the
    benchmark's own wording, and the SOC titles are the TARGETS."""
    h = g6.hierarchy_of_g6(design["manifest"], design["art"])
    alias = design["manifest"]["alias_of"]
    for iid, chain in h.items():
        assert len(chain) == 3
        assert chain[0] == alias[iid]
    by = {r["original_label"]: r for r in g6.retained_records(design["art"])}
    rec = by[alias[design["manifest"]["identity_ids"][0]]]
    chain = h[design["manifest"]["identity_ids"][0]]
    assert chain[1] == rec["level2_parent"]["title"]
    assert chain[2] == rec["minor_group_parent"]["title"]
    # the major group is metadata, never a coarsening target
    assert rec["level1_parent"]["title"] not in chain
    assert rec["level1_parent"]["title"] not in design["matrix"]["vocab"]


def test_leaf_of_uses_the_soc_code_not_the_label_text(design):
    leaf = g6.leaf_of_g6(design["manifest"], design["art"])
    alias = design["manifest"]["alias_of"]
    a1 = next(i for i in alias if alias[i] == "Alpha One")
    a2 = next(i for i in alias if alias[i] == "Alpha Two")
    assert alias[a1] != alias[a2]
    assert leaf[a1] == leaf[a2] == "11-1111"


def test_same_leaf_partners_are_not_reported_as_siblings(design):
    """The defect this pilot exists to stop: two raw labels on one SOC leaf are
    the same occupation, and calling one a sibling of the other asserts a
    taxonomic distance that does not exist."""
    entry = next(e for e in design["matrix"]["sets"]
                 if e["target_label"] == "Alpha One")
    ctl = next(iter(entry["controls"].values()))
    a2 = [i for i, a in design["manifest"]["alias_of"].items()
          if a == "Alpha Two"]
    assert ctl["same_leaf"] == sorted(a2)
    assert not (set(ctl["same_leaf"]) & set(ctl["sibling"]))
    beta = [i for i, a in design["manifest"]["alias_of"].items()
            if a == "Beta"]
    assert ctl["sibling"] == sorted(beta)


def test_without_leaf_of_the_salmu_shape_is_unchanged():
    """leaf_of is opt-in.  SALMU and CelebA pass nothing, so the same_leaf key
    must be ABSENT rather than empty -- an added key would drift their
    committed matrices, which have already been executed."""
    h = {"i1": ["jobA", "L1", "L2"], "i2": ["jobA", "L1", "L2"],
         "i3": ["jobB", "L1", "L2"], "i4": ["jobC", "L9", "L8"]}
    old = gx.sibling_controls("i1", h, {"i1"})
    assert "same_leaf" not in old
    assert old["sibling"] == ["i2", "i3"]
    new = gx.sibling_controls("i1", h, {"i1"},
                              leaf_of={"i1": "X", "i2": "X", "i3": "Y",
                                       "i4": "Z"})
    assert new["same_leaf"] == ["i2"]
    assert new["sibling"] == ["i3"]
    assert new["unrelated"] == ["i4"]


def test_a_set_with_no_sibling_records_null_and_names_what_does_apply(design):
    """Two different nulls, and the note must tell them apart.

    Delta has a same-leaf partner but no sibling: the note has to say the
    same-branch identity is a same_leaf control, not imply there is nothing.
    Theta has neither: the note falls back to the generic wording.  Alpha One
    HAS a sibling (Beta), so it gets no note at all.
    """
    notes = design["validation"]["control_notes"]
    sid = {e["target_label"]: e["set_id"] for e in design["matrix"]["sets"]}

    delta = next(iter(notes[sid["Delta"]].values()))
    assert "sibling metric null, not 1.0" in delta
    assert "same_leaf control" in delta
    assert "unrelated controls apply" in delta

    theta = next(iter(notes[sid["Theta"]].values()))
    assert "sibling metric null, not 1.0" in theta
    assert "same_leaf" not in theta

    assert sid["Alpha One"] not in notes


def test_sibling_availability_separates_null_from_unmeasured(design):
    avail = g6.sibling_availability(design["matrix"])
    measured = {k for k, v in avail.items() if v["sibling_available"]}
    assert len(measured) == 2          # Alpha One -> Beta, Zeta -> Eta
    for v in avail.values():
        if v["sibling_available"]:
            assert v["sibling_metric"] == "measured"
            assert v["sibling_null_reason"] is None
            assert v["siblings"]
        else:
            assert v["sibling_metric"] is None
            assert "null, never 1.0" in v["sibling_null_reason"]
            assert v["siblings"] == []


# ====================================================================== #
# Matrix shape and validation
# ====================================================================== #
def test_five_sets_one_per_target_and_every_other_identity_retained(design):
    matrix = design["matrix"]
    assert matrix["n_sets"] == 5
    assert matrix["n_cells"] == 5 * len(g6.EDIT_SEEDS)
    assert [e["target_label"] for e in matrix["sets"]] == design["targets"]
    ids = set(design["manifest"]["identity_ids"])
    for entry in matrix["sets"]:
        targets = set(entry["assignments"])
        assert set(entry["retain_ids"]) == ids - targets
        for iid, a in entry["assignments"].items():
            assert a["operation"] == "taxonomic"
            assert a["source"] == entry["target_label"]
            assert a["target"] == entry["coarser_label"]
            assert a["target_depth"] == g6.PILOT_TARGET_DEPTH > \
                a["source_depth"]
            # every target identity really does bear the target profession
            assert design["manifest"]["alias_of"][iid] == entry["target_label"]


def test_a_dropped_same_leaf_partner_fails_validation(design):
    """The invariant that matters: an AVAILABLE control must not silently go
    unmeasured while the gates still report a pass."""
    manifest = json.loads(json.dumps(design["manifest"]))
    alias = manifest["alias_of"]
    dropped = [i for i, a in alias.items() if a == "Alpha Two"]
    manifest["identity_ids"] = [i for i in manifest["identity_ids"]
                                if i not in dropped]
    for i in dropped:
        del alias[i]
        del manifest["code_of"][i]
    matrix, ctx = g6.build_g6_matrix(manifest, design["art"], design["sel"],
                                     evidence=dict(MINI_EVIDENCE))
    with pytest.raises(RuntimeError, match="same-leaf partner"):
        g6.validate_g6(matrix, ctx, design["art"])


def test_a_target_with_no_same_leaf_partner_is_recorded_not_failed(design):
    cov = design["validation"]["same_leaf_coverage"]
    theta = next(k for k, v in cov.items() if v["target_label"] == "Theta")
    assert cov[theta]["same_leaf_partners_in_hierarchy"] == []
    assert cov[theta]["all_present_in_manifest"] is True
    assert "no same-leaf control available" in cov[theta]["note"]
    alpha = next(k for k, v in cov.items() if v["target_label"] == "Alpha One")
    assert cov[alpha]["same_leaf_partners_in_hierarchy"] == ["Alpha Two"]
    assert cov[alpha]["note"] is None


# ====================================================================== #
# The four gates: each must pass clean AND fail on exactly one corruption
# ====================================================================== #
def test_all_four_gates_pass_on_a_clean_fabricated_run(design, monkeypatch):
    _forbid_source_dataset(monkeypatch)
    res = g6.evaluate_gates(_cells(design), design["matrix"])
    assert res["passed"] is True
    assert res["failed_gates"] == []
    assert res["proceed_to_full_matrix"] is True
    assert set(res["gates"]) == {"behavioral", "retention",
                                 "sibling_coverage", "matched_oracle"}


def test_the_behavioral_gate_catches_a_target_that_did_not_move(design):
    def break_target(entry, seed, iid, grp, row):
        if grp == "target" and seed == g6.EDIT_SEEDS[0]:
            # still fluent, still a real occupation -- just not the coarser
            # one this set asked for
            return _row(row["expected"], grp,
                        parsed=entry["target_label"],
                        cls="under_abstraction")
        return row
    res = g6.evaluate_gates(_cells(design, mutate=break_target),
                            design["matrix"])
    g = res["gates"]["behavioral"]
    assert g["passed"] is False
    assert g["strict_expected_accuracy"] < 1.0
    assert any("strict-parsed" in f for f in g["failures"])
    assert res["proceed_to_full_matrix"] is False


def test_the_behavioral_gate_catches_a_residual_exact_label(design):
    """Fluent AND correct-looking, but the exact label still carries mass."""
    def leak(entry, seed, iid, grp, row):
        if grp == "target":
            return _row(row["expected"], grp, p_source=0.20)
        return row
    g = g6.evaluate_gates(_cells(design, mutate=leak),
                          design["matrix"])["gates"]["behavioral"]
    assert g["passed"] is False
    assert any("p_source" in f for f in g["failures"])


def test_the_behavioral_gate_catches_a_wrong_branch_answer(design):
    """The specific way this experiment can fail while still looking fluent:
    the model emits a real occupation that is not an ancestor of the target's
    own.  It parses, it is in the vocab, and it is not the exact label -- so
    only the taxonomic classification catches it."""
    def wrong_branch(entry, seed, iid, grp, row):
        if grp == "target":
            return _row(row["expected"], grp, parsed="Broad Care",
                        cls="wrong_branch")
        return row
    g = g6.evaluate_gates(_cells(design, mutate=wrong_branch),
                          design["matrix"])["gates"]["behavioral"]
    assert g["passed"] is False
    assert g["n_wrong_branch"] == g["n_target_rows"]
    # Assert on the message ONLY the wrong-branch arm produces.  The
    # strict-parse failure also embeds the taxonomic class, so matching on the
    # bare word "wrong_branch" passes even with this arm deleted.
    assert any("classified wrong_branch" in f for f in g["failures"])


def test_the_behavioral_gate_rejects_multi_label_and_unparseable(design):
    for kw, needle in (({"multi": True}, "multi-label"),
                       ({"unparseable": True}, "unparseable")):
        def broken(entry, seed, iid, grp, row, _kw=kw):
            if grp == "target":
                return _row(row["expected"], grp, **_kw)
            return row
        g = g6.evaluate_gates(_cells(design, mutate=broken),
                              design["matrix"])["gates"]["behavioral"]
        assert g["passed"] is False, kw
        assert any(needle in f for f in g["failures"]), (needle, g["failures"])


def test_the_behavioral_gate_catches_a_thin_candidate_distribution(design):
    def thin(entry, seed, iid, grp, row):
        if grp == "target":
            return _row(row["expected"], grp, mass=0.50)
        return row
    g = g6.evaluate_gates(_cells(design, mutate=thin),
                          design["matrix"])["gates"]["behavioral"]
    assert g["passed"] is False
    assert any("candidate_mass" in f for f in g["failures"])


def test_the_retention_gate_catches_a_dragged_same_leaf_partner(design):
    """The failure this pilot is designed to detect: an edit that generalises
    on OCCUPATION rather than on label string drags the same-leaf partner
    along with the target."""
    def drag(entry, seed, iid, grp, row):
        if grp == "same_leaf" and row["expected"] == "Alpha Two":
            return _row(row["expected"], grp, parsed=entry["coarser_label"])
        return row
    res = g6.evaluate_gates(_cells(design, mutate=drag), design["matrix"])
    g = res["gates"]["retention"]
    assert g["passed"] is False
    per = g["per_control_group"]
    assert per["same_leaf"]["strict_accuracy"] < 1.0
    # the other groups still pass, and that must be VISIBLE, not averaged away
    assert per["retain"]["strict_accuracy"] == 1.0
    assert any("Alpha Two" in f or "expected" in f for f in g["failures"])


def test_retention_reports_groups_separately_and_never_pools_them(design):
    res = g6.evaluate_gates(_cells(design), design["matrix"])
    per = res["gates"]["retention"]["per_control_group"]
    assert {"same_leaf", "sibling", "cousin", "retain"} <= set(per)
    for name, g in per.items():
        assert g["n_rows"] > 0, name
        assert g["strict_accuracy"] == 1.0, name
    assert "never pooled" in res["gates"]["retention"]["note"]


def test_sibling_coverage_passes_with_nulls_and_fails_on_a_skipped_sibling(
        design):
    res = g6.evaluate_gates(_cells(design), design["matrix"])
    g = res["gates"]["sibling_coverage"]
    assert g["passed"] is True
    assert g["n_sets_with_sibling"] == 2
    assert g["n_sets_sibling_null"] == 3
    nulls = [v for v in g["per_set"].values() if v["sibling_metric"] is None]
    # a null is never reported as a perfect score
    assert all(v["strict_accuracy"] is None for v in nulls)
    assert all(1.0 not in (v["strict_accuracy"],) for v in nulls)

    # now drop the sibling rows from the evaluated results entirely
    sib_ids = {i for i, a in design["manifest"]["alias_of"].items()
               if a in ("Beta", "Eta")}
    res2 = g6.evaluate_gates(_cells(design, drop=sib_ids), design["matrix"])
    g2 = res2["gates"]["sibling_coverage"]
    assert g2["passed"] is False
    assert any("were not evaluated" in f for f in g2["failures"])


def test_sibling_coverage_fails_when_an_evaluated_sibling_is_wrong(design):
    def wrong(entry, seed, iid, grp, row):
        if grp == "sibling":
            return _row(row["expected"], grp, parsed="Something Else")
        return row
    g = g6.evaluate_gates(_cells(design, mutate=wrong),
                          design["matrix"])["gates"]["sibling_coverage"]
    assert g["passed"] is False
    assert any("sibling" in f and "expected" in f for f in g["failures"])


def test_the_matched_oracle_gate_needs_a_positive_delta(design):
    good = g6.evaluate_gates(_cells(design, delta_oracle=0.30),
                             design["matrix"])["gates"]["matched_oracle"]
    assert good["passed"] is True
    assert all(v["mean_delta_oracle"] > 0
               for v in good["per_set"].values())

    # delta == 0 exactly: the edit is no closer to the matched oracle than to
    # LOO, so it is not evidence of controlled coarsening
    flat = g6.evaluate_gates(_cells(design, delta_oracle=0.0),
                             design["matrix"])["gates"]["matched_oracle"]
    assert flat["passed"] is False
    assert len(flat["failures"]) == 5
    assert any("<= 0" in f for f in flat["failures"])

    inverted = g6.evaluate_gates(_cells(design, delta_oracle=-0.25),
                                 design["matrix"])["gates"]["matched_oracle"]
    assert inverted["passed"] is False


def test_the_matched_oracle_gate_fails_when_a_set_was_never_run(design):
    cells = [c for c in _cells(design)
             if c["set_id"] != design["matrix"]["sets"][0]["set_id"]]
    g = g6.evaluate_gates(cells, design["matrix"])["gates"]["matched_oracle"]
    assert g["passed"] is False
    assert any("no cells evaluated" in f for f in g["failures"])


def test_one_failing_gate_blocks_the_full_matrix(design):
    res = g6.evaluate_gates(_cells(design, delta_oracle=-1.0),
                            design["matrix"])
    assert res["passed"] is False
    assert res["failed_gates"] == ["matched_oracle"]
    assert res["proceed_to_full_matrix"] is False
    # the other three still passed on their own merits
    assert res["gates"]["behavioral"]["passed"] is True
    assert res["gates"]["retention"]["passed"] is True
    assert res["gates"]["sibling_coverage"]["passed"] is True


# ====================================================================== #
# Freeze determinism
# ====================================================================== #
def test_a_frozen_artifact_must_survive_a_json_round_trip(tmp_path):
    """The bug this pins: gx.check_vocab_collisions returns TUPLES, json writes
    them as arrays and reads them back as lists, so an unnormalized freeze
    reports drift against itself forever."""
    p = tmp_path / "m.json"
    obj = {"pairs": [["a", "b"]], "n": 1}
    assert g6._write_frozen(p, {"pairs": [("a", "b")], "n": 1}, "thing") \
        is True
    assert p.exists()
    # rebuilding with tuples again must be accepted, not reported as drift
    assert g6._write_frozen(p, {"pairs": [("a", "b")], "n": 1}, "thing") \
        is False
    assert json.loads(p.read_text()) == obj


def test_a_genuinely_different_rebuild_is_refused(tmp_path):
    p = tmp_path / "m.json"
    g6._write_frozen(p, {"n_sets": 5}, "matrix")
    with pytest.raises(RuntimeError, match="must never drift"):
        g6._write_frozen(p, {"n_sets": 6}, "matrix")
    # and the refused write left the committed bytes alone
    assert json.loads(p.read_text()) == {"n_sets": 5}


def test_key_order_does_not_look_like_drift(tmp_path):
    p = tmp_path / "m.json"
    g6._write_frozen(p, {"a": 1, "b": 2}, "matrix")
    assert g6._write_frozen(p, {"b": 2, "a": 1}, "matrix") is False


def test_the_matrix_is_deterministic_across_rebuilds(design):
    a, ctx_a = g6.build_g6_matrix(design["manifest"], design["art"],
                                  design["sel"],
                                  evidence=dict(MINI_EVIDENCE))
    b, ctx_b = g6.build_g6_matrix(design["manifest"], design["art"],
                                  design["sel"],
                                  evidence=dict(MINI_EVIDENCE))
    assert g6._canonical(a) == g6._canonical(b)
    assert g6._canonical(ctx_a) == g6._canonical(ctx_b)
    # and the frozen design carries no timestamp
    assert "timestamp" not in g6._canonical(a)


# ====================================================================== #
# Provenance
# ====================================================================== #
def test_the_worktree_state_reports_both_determinations(tmp_path):
    ws = g6.worktree_state(exclude_prefixes=("e2c_mllmu/outputs/g6/",))
    assert ws["git_commit"]
    assert "dirty_tracked_only" in ws and "dirty_including_untracked" in ws
    # the exclusion is REPORTED, never applied silently
    assert ws["excluded_prefixes"] == ["e2c_mllmu/outputs/g6/"]
    assert "exclusion_note" in ws
    assert isinstance(ws["untracked_outside_exclusions"], list)


def test_every_field_the_launch_sequence_requires_is_in_the_manifest(design,
                                                                    tmp_path):
    cells = _cells(design)
    gates = g6.evaluate_gates(cells, design["matrix"])
    prov = {
        "executing_commit": "b29fadc",
        "source_dataset_sha256": "d" * 64,
        "source_dataset_matches_frozen": True,
        "hierarchy": dict(MINI_EVIDENCE),
        "clean_worktree": g6.worktree_state(exclude_prefixes=()),
    }
    per_set = g6.per_set_provenance(design["matrix"], cells)
    hashes = {"outputs_sha256": {"a.json": "0" * 64},
              "checkpoints_sha256": {}}
    manifest = {
        "executing_commit": prov["executing_commit"],
        "frozen_inputs": {
            "hierarchy_content_sha256": MINI_EVIDENCE[
                "hierarchy_content_sha256"],
            "selection_sha256": MINI_EVIDENCE["selection_sha256"],
            "source_table_sha256": MINI_EVIDENCE["source_table_sha256"],
            "source_dataset_sha256": prov["source_dataset_sha256"]},
        "clean_worktree_state": prov["clean_worktree"],
        "sets": per_set, "gates": gates, **hashes,
    }
    # the six things the launch sequence demands of every run manifest
    assert manifest["executing_commit"]
    assert manifest["frozen_inputs"]["hierarchy_content_sha256"]
    assert manifest["frozen_inputs"]["selection_sha256"]
    assert manifest["frozen_inputs"]["source_dataset_sha256"]
    assert "dirty_tracked_only" in manifest["clean_worktree_state"]
    assert manifest["checkpoints_sha256"] == {}
    assert len(per_set) == len(cells)
    for row in per_set:
        assert {"set_id", "seed", "target_label", "target_soc",
                "coarser_label", "target_identity_ids"} <= set(row)
        assert row["seed"] in g6.EDIT_SEEDS


def test_validation_output_survives_a_json_round_trip(design):
    """Everything stored in a frozen artifact must come back unchanged.

    ``gx.check_vocab_collisions`` returns TUPLES; json writes them as arrays
    and reads them back as lists.  Left unnormalized, the freeze reports drift
    against the file it wrote one second earlier and can never reach a fixed
    point -- which either deadlocks a re-freeze or, worse, invites someone to
    delete committed evidence to get past it.
    """
    v = design["validation"]
    assert json.loads(json.dumps(v)) == json.loads(json.dumps(
        json.loads(json.dumps(v))))
    for pair in v["vocab_collisions"]["nested_longest_match_wins"]:
        assert isinstance(pair, list), f"tuple leaked into frozen data: {pair}"
    for pair in v["vocab_collisions"]["hard_collisions"]:
        assert isinstance(pair, list)
    # and the real vocab does have nesting, so the check is not over an
    # empty list
    assert v["vocab_collisions"]["nested_longest_match_wins"]


def test_a_target_with_no_controls_at_all_is_reported_not_crashed():
    """A lone target: no sibling, no cousin, no unrelated, no same_leaf.

    This is the case that turns ``ctl.get("same_leaf")`` into ``ctl["same_leaf"]``
    from a style question into a crash -- the ``or`` chain short-circuits on any
    non-empty group, so indexing only fails once every group is empty.  SALMU
    has no such identity today, which is exactly why it needs pinning here.
    """
    h = {"t1": ["jobA", "L1", "L2"]}
    entry = {"set_id": "s", "mode": "single_level1",
             "assignments": {"t1": {"operation": "taxonomic",
                                    "source": "jobA", "target": "L1",
                                    "source_depth": 0, "target_depth": 1}},
             "retain_ids": []}
    ctx = {"kind": "taxonomic", "identity_ids": ["t1"],
           "baseline_alias_of": {"t1": "jobA"}, "hierarchy_of": h,
           "dag": gx.build_label_dag(h), "vocab": ["jobA", "L1", "L2",
                                                   gx.DELETED_LABEL]}
    issues = gx.validate_set(entry, ctx)
    assert any("NO controls available at all" in i for i in issues)
    # and with leaf_of supplied the same input must not raise either
    entry2 = json.loads(json.dumps(entry))
    ctx2 = dict(ctx, leaf_of={"t1": "X"})
    issues2 = gx.validate_set(entry2, ctx2)
    assert any("NO controls available at all" in i for i in issues2)
    assert entry2["controls"]["t1"]["same_leaf"] == []


def test_control_group_reports_same_leaf_before_sibling():
    """The runner's report column, which is where a misrouted same-leaf control
    would actually be read.

    Importing the matrix runner pulls in torch; that cost is accepted here
    because ``control_group`` is the function that decides which metric a
    retention row lands in, and leaving it untested would let the same-leaf
    fix be undone in the report while the gates still looked correct.
    """
    gm = _load("e2c_gxm_under_test", "e2c_v3_granularity_matrix.py")
    entry = {"set_id": "s", "assignments": {"t1": {}},
             "controls": {"t1": {"same_leaf": ["p1"], "sibling": ["s1"],
                                 "cousin": ["c1"], "unrelated": ["u1"]}}}
    assert gm.control_group({}, entry, "t1") == "target"
    assert gm.control_group({}, entry, "p1") == "same_leaf"
    assert gm.control_group({}, entry, "s1") == "sibling"
    assert gm.control_group({}, entry, "c1") == "cousin"
    assert gm.control_group({}, entry, "other") == "retain"
    # an identity listed in BOTH must resolve to same_leaf, not sibling
    both = {"set_id": "s", "assignments": {"t1": {}},
            "controls": {"t1": {"same_leaf": ["x"], "sibling": ["x"],
                                "cousin": [], "unrelated": []}}}
    assert gm.control_group({}, both, "x") == "same_leaf"


def test_worktree_state_separates_tracked_from_untracked(tmp_path,
                                                         monkeypatch):
    """A clean tracked tree can still hide an untracked module that changes
    what executes, so both determinations are recorded and the untracked one
    must not simply mirror the tracked one.

    Done in a throwaway repository because dirtying the real checkout would
    fail the post-preflight cleanliness check that CI runs.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "tracked.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], cwd=repo, check=True)
    monkeypatch.setattr(mh, "REPO_ROOT", repo)

    ws = g6.worktree_state()
    assert ws["dirty_tracked_only"] is False
    assert ws["dirty_including_untracked"] is False

    (repo / "sitecustomize.py").write_text("# shadows an installed module\n")
    ws2 = g6.worktree_state()
    assert ws2["dirty_tracked_only"] is False
    assert ws2["dirty_including_untracked"] is True
    assert any("sitecustomize.py" in l
               for l in ws2["untracked_outside_exclusions"])

    # excluding a prefix is legitimate, but the exclusion is REPORTED and the
    # unfiltered determination still says dirty
    ws3 = g6.worktree_state(exclude_prefixes=("sitecustomize.py",))
    assert ws3["n_excluded"] == 1
    assert ws3["untracked_outside_exclusions"] == []
    assert ws3["dirty_including_untracked"] is True
    assert ws3["dirty_tracked_only"] is False

    # a MODIFIED tracked file is dirty under both
    (repo / "tracked.py").write_text("x = 2\n")
    ws4 = g6.worktree_state()
    assert ws4["dirty_tracked_only"] is True
    assert ws4["dirty_including_untracked"] is True


def _runner_cell(design, set_idx=0, seed=17, soft_override=None,
                 fam_override=None):
    """A cell in the GX RUNNER's on-disk schema, not the gate schema.

    The two are deliberately different: the runner stores per-identity hard
    predictions, soft probabilities and PER-IDENTITY oracle distances, while the
    gates want one flat row per identity and one cell-level delta.  Testing the
    adapter against a faithful runner cell is what makes the wiring between a
    real GPU run and the four gates a tested path rather than a hoped-for one.
    """
    entry = design["matrix"]["sets"][set_idx]
    ctx = design["ctx"]
    hard = []
    for iid in ctx["identity_ids"]:
        grp = _group_of(entry, iid)
        exp = (entry["assignments"][iid]["target"]
               if iid in entry["assignments"]
               else ctx["baseline_alias_of"][iid])
        hard.append({"identity_id": iid, "raw": exp, "parsed_label": exp,
                     "recognized_labels": [exp],
                     "multi_label_ambiguous": False, "group": grp,
                     "expected_post_edit": exp, "correct_post_edit": True,
                     "source_leaked": False, "classification": "correct"})
    soft = {iid: {"p_expected": 0.99, "p_baseline_alias": 0.001,
                  "candidate_mass": 1.0, "other_mass": 0.0}
            for iid in ctx["identity_ids"]}
    if soft_override:
        for iid, patch in soft_override.items():
            soft.setdefault(iid, {}).update(patch)
    fam = {}
    for iid in ctx["identity_ids"]:
        fam[iid] = {
            "matched_finetune": {"distance": {"l2": 0.10}, "reliable": True,
                                 "reason": None},
            "loo_finetune": {"distance": {"l2": 0.40}, "reliable": True,
                             "reason": None},
            "delta_ft_l2": 0.30, "delta_reliable": True}
    if fam_override:
        for iid, patch in fam_override.items():
            fam.setdefault(iid, {}).update(patch)
    return {"cell_id": f"{entry['set_id']}__seed{seed}",
            "dataset": "mllmu", "set_id": entry["set_id"],
            "mode": entry["mode"], "seed": seed,
            "assignments": entry["assignments"],
            "hard_preds": hard, "soft": soft, "oracle_families": fam}


def test_the_adapter_maps_runner_cells_onto_the_gate_schema(design):
    cell = _runner_cell(design)
    (adapted,) = g6.cells_for_gates([cell])
    assert adapted["set_id"] == cell["set_id"]
    assert adapted["seed"] == 17
    row = next(iter(adapted["per_identity"].values()))
    assert {"expected", "strict_parsed", "control_group", "p_desired",
            "p_source", "candidate_mass", "taxonomic_class", "multi_label",
            "unparseable"} <= set(row)
    # A single set's cells are NOT enough for the gates to pass: the
    # matched-oracle and sibling-coverage gates both fail on a set that was
    # never run, which is the behaviour the other tests pin.  So the clean-run
    # assertion needs every set present.
    all_cells = g6.cells_for_gates(
        [_runner_cell(design, set_idx=i)
         for i in range(len(design["matrix"]["sets"]))])
    res = g6.evaluate_gates(all_cells, design["matrix"])
    assert res["passed"] is True, res["failed_gates"]
    assert res["n_cells_evaluated"] == len(design["matrix"]["sets"])


def test_the_adapter_carries_same_leaf_groups_through_from_the_runner(design):
    """The runner's own ``group`` column must reach the retention gate as
    same_leaf, or the sharpest control silently lands in the general retain
    pool and its failure is averaged away."""
    entry = design["matrix"]["sets"][0]
    tid = min(entry["assignments"])
    partner = next(iter(entry["controls"][tid]["same_leaf"]))
    (adapted,) = g6.cells_for_gates([_runner_cell(design)])
    assert adapted["per_identity"][partner]["control_group"] == "same_leaf"
    g = g6.evaluate_gates([adapted], design["matrix"])["gates"]["retention"]
    assert g["per_control_group"]["same_leaf"]["n_rows"] >= 1


def test_the_adapter_reads_p_source_from_the_baseline_alias(design):
    """For a target identity the baseline alias IS the exact label the edit must
    remove, so mass there is the residual the behavioral gate rejects.  Mapping
    the wrong soft field would make that check vacuous."""
    entry = design["matrix"]["sets"][0]
    tid = min(entry["assignments"])
    cell = _runner_cell(design, soft_override={tid: {"p_baseline_alias": 0.42,
                                                     "p_expected": 0.99}})
    (adapted,) = g6.cells_for_gates([cell])
    assert adapted["per_identity"][tid]["p_source"] == 0.42
    g = g6.evaluate_gates([adapted], design["matrix"])["gates"]["behavioral"]
    assert g["passed"] is False
    assert any("p_source" in f for f in g["failures"])


def test_the_adapter_averages_delta_over_targets_only(design):
    """A retained identity's oracle distance says nothing about whether the
    TRANSFORMATION matched, so including one would let a control row dilute the
    measurement the matched-oracle gate exists to make."""
    entry = design["matrix"]["sets"][0]
    n_targets = len(entry["assignments"])
    retained = [i for i in design["ctx"]["identity_ids"]
                if i not in entry["assignments"]]
    # push every retained row's delta wildly negative: the cell delta must not
    # move, because only target rows are averaged
    fam = {i: {"delta_ft_l2": -9.0} for i in retained}
    (adapted,) = g6.cells_for_gates([_runner_cell(design, fam_override=fam)])
    d = adapted["oracle_distances"]
    assert d["n_target_rows"] == n_targets
    assert d["n_rows_used"] == n_targets
    assert d["delta_oracle"] == 0.30


def test_an_unreliable_oracle_row_is_dropped_and_counted(design):
    entry = design["matrix"]["sets"][0]
    tids = sorted(entry["assignments"])
    fam = {tids[0]: {"delta_ft_l2": None}}
    (adapted,) = g6.cells_for_gates([_runner_cell(design, fam_override=fam)])
    d = adapted["oracle_distances"]
    assert d["n_rows_used"] == len(tids) - 1
    assert d["n_target_rows"] == len(tids)
    assert d["delta_oracle"] == 0.30


def test_no_reliable_delta_at_all_fails_rather_than_passing(design):
    """A missing or unreliable oracle is not a pass.  Averaging over an empty
    set would either raise or -- worse -- report a plausible number computed
    from nothing."""
    entry = design["matrix"]["sets"][0]
    fam = {i: {"delta_ft_l2": None} for i in sorted(entry["assignments"])}
    cells = g6.cells_for_gates([_runner_cell(design, fam_override=fam)])
    assert cells[0]["oracle_distances"]["delta_oracle"] is None
    g = g6.evaluate_gates(cells, design["matrix"])["gates"]["matched_oracle"]
    assert g["passed"] is False
    assert any("no reliable delta_oracle" in f for f in g["failures"])
    assert g["per_set"][entry["set_id"]]["mean_delta_oracle"] is None


# ====================================================================== #
# Runner wiring
# ====================================================================== #
def test_the_runner_reaches_the_same_frozen_design_as_freeze():
    """``build_or_verify`` and ``--freeze`` must agree, or the runner would
    report drift in a design nothing had changed."""
    _committed_design_or_skip()
    gm = _load("e2c_gxm_under_test", "e2c_v3_granularity_matrix.py")
    rebuilt = gm.g6m.rebuild_frozen_matrix(verify=False)
    committed = json.loads(gm.g6m.G6_MATRIX_PATH.read_text(encoding="utf-8"))
    assert gm.g6m._canonical(rebuilt["matrix"]) == \
        gm.g6m._canonical(committed)
    assert gm.load_frozen("mllmu") == committed


def test_the_runner_ctx_carries_leaf_of_for_mllmu_only():
    _committed_design_or_skip()
    gm = _load("e2c_gxm_under_test", "e2c_v3_granularity_matrix.py")
    matrix = gm.load_frozen("mllmu")
    ctx = gm.dataset_ctx("mllmu", matrix)
    assert ctx["kind"] == "taxonomic"
    assert "leaf_of" in ctx and len(ctx["leaf_of"]) == len(
        ctx["identity_ids"])
    assert len(set(ctx["leaf_of"].values())) < len(ctx["leaf_of"]), \
        "the pilot must contain at least one same-leaf pair"
    # SALMU's ctx must NOT gain the key: its matrix is committed and executed
    salmu_ctx = gm.dataset_ctx("salmu", gm.load_frozen("salmu"))
    assert "leaf_of" not in salmu_ctx


# ====================================================================== #
# The committed pilot design
# ====================================================================== #
def _committed_design_or_skip():
    missing = [str(p) for p in (g6.G6_MANIFEST_PATH, g6.G6_MATRIX_PATH,
                                g6.HIERARCHY_PATH, g6.SELECTION_PATH)
               if not p.exists()]
    if missing:
        pytest.skip(f"frozen G6 artifacts absent from this checkout: "
                    f"{missing}")
    return g6.load_frozen_g6()


def test_the_committed_pilot_is_the_five_duplicate_leaf_cases():
    manifest, matrix = _committed_design_or_skip()
    art = g6.load_hierarchy()
    sel = g6.load_selection()
    assert matrix["pilot_targets"] == g6.pilot_targets(art, sel)
    assert matrix["n_sets"] == 5
    # each of the five is a raw label that shares its SOC leaf with another
    # retained raw label -- the reason they were adjudicated
    ret = g6.retained_records(art)
    for entry in matrix["sets"]:
        partners = [r["original_label"] for r in ret
                    if r["external_occupation_id"] == entry["target_soc"]
                    and r["original_label"] != entry["target_label"]]
        assert partners, entry["target_label"]
        assert all(p in manifest["professions"] for p in partners)


def test_the_committed_pilot_records_its_g6_0_licence():
    _manifest, matrix = _committed_design_or_skip()
    art = g6.load_hierarchy()
    sel = g6.load_selection()
    ev = matrix["g6_0_evidence"]
    assert ev["hierarchy_content_sha256"] == art["content_sha256"]
    assert ev["selection_sha256"] == sel["selection_sha256"]
    assert ev["source_table_sha256"] == art["source_table"]["sha256"]
    assert ev["source_dataset_sha256"] == art["source_dataset"]["sha256"]
    # and the structural limitation is stated in the frozen file itself
    avail = matrix["g6x0_validation"]["sibling_availability"]
    assert sum(1 for v in avail.values() if v["sibling_available"]) == 2
    assert sum(1 for v in avail.values() if not v["sibling_available"]) == 3


def test_the_committed_pilot_verifies_without_the_source_dataset(monkeypatch):
    _committed_design_or_skip()
    monkeypatch.setattr(mh, "FULL_SET", Path("/nonexistent/Full_Set.jsonl"))
    monkeypatch.setattr(g6, "read_source_rows",
                        lambda *a, **k: pytest.fail(
                            "--verify read the source dataset"))
    rep = g6.verify(recount_source=False)
    assert rep["ok"] is True
    assert rep["source_recount"] is None
    assert rep["n_sets"] == 5


def test_verify_refuses_a_pilot_frozen_against_a_different_hierarchy(
        monkeypatch):
    _committed_design_or_skip()
    real_ev = g6.frozen_hierarchy_evidence

    def shifted(*a, **k):
        ev = real_ev(*a, **k)
        ev = dict(ev)
        ev["hierarchy_content_sha256"] = "f" * 64
        return ev
    monkeypatch.setattr(g6, "frozen_hierarchy_evidence", shifted)
    with pytest.raises(RuntimeError, match="frozen against a different "
                                           "hierarchy"):
        g6.verify()
