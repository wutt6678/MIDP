"""Route-dependent forgetting, v4: one denominator, and the honesty about when
it was chosen.

v3 ran to completion.  Twenty-eight cells, two datasets, six trained adapters,
and both authoritative checks green: RF2 re-hashed every binding it could and
reported none it could not, and RF2P reproduced the filed scores exactly.  Its
mediation verdict failed on both datasets, and on PPUBench it failed for a reason
the design had already ruled out of order:

  * ``hybrid_conflict_probe`` is declared ``auxiliary: true``, ``decisive:
    false``, and "reported separately from every mediated gate";
  * the report echoes that as ``auxiliary_conditions_reported_separately``;
  * and ``gate_to_verdict`` maps ``no_unparseable_or_multi_label_outputs`` -- a
    count over EVERY row the run produced, those twelve included -- to
    ``mediation``.

Each of the three is defensible.  Holding all three at once is not, and the run
found out by failing a verdict on rows the design had said were outside it: nine
unparseable outputs, all nine in the auxiliary probe, with sixty-four
non-auxiliary rows clean.

v4 changes that one denominator and nothing else.  What the tests below pin is
therefore narrow, and deliberately so:

  1. the scope is a FIELD OF THE DESIGN, so a reader can list it and a test can
     pin it, rather than a choice the aggregation makes once rows exist;
  2. the auxiliary rows are REPORTED, not dropped -- excluding a row from a
     verdict is a statement about the verdict, not about the row;
  3. the amendment is NOT A WAIVER: a non-auxiliary unparseable output still
     fails it, which is what SALMU did;
  4. ``direct_code_following_rate`` did NOT move.  It is measured entirely on
     auxiliary rows and still feeds mediation, because it is not a hygiene count
     -- it is the ceiling v2 added, and scoping it out would undo a repair;
  5. v4 changes NO MEASUREMENT: same conditions, same thresholds, same gate map,
     same cells, same rows, same result kind -- so the filed v3 cells are read
     rather than re-run, and a v4 cell would be the same bytes;
  6. a cell another design produced is CHECKED before it is consumed, and the
     pairing is DISCLOSED, because ``input_verification`` records the
     pre-registration a cell came from without ever comparing it;
  7. the amendment SAYS it came after the outcome, at the top level of its own
     design, and carries the derived evidence from BOTH datasets -- one that
     moves and one that does not, which is what makes it a correction rather
     than a tailor.

Every test here is CPU-only.  The ones that read the committed v3 record or a
frozen v4 manifest say so and skip with the absent path named.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
_RF = _ROOT / "e2c_route_forgetting"


def _load():
    spec = importlib.util.spec_from_file_location(
        "route_forgetting_v4_under_test",
        _SCRIPTS / "e2c_v3_route_dependent_forgetting.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rf():
    return _load()


# --------------------------------------------------------------------------
# what a checkout has to have
# --------------------------------------------------------------------------
#
# Split by what the test READS.  The gate arithmetic and the spec table need
# nothing but the runner, so those tests run in a bare clone.  The amendment's
# evidence is computed from the committed v3 record, and the lineage check from a
# frozen v4 manifest, so the tests that touch either are gated and name what is
# missing rather than failing on it.

_V3_MANIFESTS = {ds: _RF / "manifests" / f"rf_pilot_{ds}_v3.json"
                 for ds in ("ppubench", "salmu")}
_V3_REPORTS = {ds: _RF / "reports" / f"rf_report_{ds}_v3.json"
               for ds in ("ppubench", "salmu")}
_V4_MANIFESTS = {ds: _RF / "manifests" / f"rf_pilot_{ds}_v4.json"
                 for ds in ("ppubench", "salmu")}
_V3_CELLS = {ds: sorted((_RF / "outputs" / ds).glob("*/cell_results.json"))
             for ds in ("ppubench", "salmu")}


def _absent(paths):
    return [str(p) for p in paths if not Path(p).is_file()]


def _gate(paths, what):
    missing = _absent(paths)
    return pytest.mark.skipif(
        bool(missing),
        reason=(f"{what} reads {len(missing)} committed artifact(s) this "
                f"checkout does not have: " + ", ".join(missing)))


_needs_v3_record = _gate([*_V3_MANIFESTS.values(), *_V3_REPORTS.values()],
                         "the v4 scope amendment's evidence")

#: A glob over an absent directory is empty rather than an error, so "no cells"
#: is probed as its own condition: a gate built only on the paths a glob returned
#: would pass in the one checkout it exists to describe.
_NO_CELLS = [ds for ds, paths in sorted(_V3_CELLS.items()) if not paths]
_needs_v3_cells = pytest.mark.skipif(
    bool(_NO_CELLS),
    reason=("re-reading the filed cells needs the cells v3 filed, and this "
            "checkout has none for " + ", ".join(_NO_CELLS)))


# --------------------------------------------------------------------------
# a hermetic design, and a fake model that derails where a test says
# --------------------------------------------------------------------------
#
# Mirrors the v2 module's helpers rather than importing them: two test modules
# that share a fixture by import order are two modules that break together.

ALIASES = {"001": "Oden", "002": "Seri", "003": "Kael", "004": "Dax"}
DELETED = "Unknown"
SELECTION = {"dataset": "ppubench", "set_id": "fs_001", "targets": ["001"],
             "selection_rule": "fixture", "n_candidates": 7, "n_considered": 4,
             "candidates": []}
FAKE_INVENTORY = {"dataset": "ppubench", "n_forget_sets": 7, "sets": {},
                  "n_reusable_edited_h": 21, "n_edited_h_required": 21,
                  "shared_route_checkpoints": {}, "complete": True}

#: An output that names no candidate label.  This is what the real auxiliary
#: probe produced -- "Bubba", "Mikah", "K" -- and it is why the count matters:
#: the probe did not answer wrongly, it left the vocabulary.
DERAILED = "Bubba"

E2E_OK = {"observed": 1.0, "predicted": 1.0, "n_images": 12}


def _write_png(path, identity_index, salt=0):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4),
              (identity_index, salt % 251, (salt * 7) % 251)).save(path)
    return str(path)


def _disk_manifest(tmp_path, n_test=3, identity_ids=("001", "002", "003",
                                                     "004")):
    """A manifest whose ``image_uri`` values point at files that exist."""
    items = []
    for k, iid in enumerate(identity_ids):
        for split, n in (("train", 2), ("test", n_test)):
            for i in range(n):
                uri = _write_png(tmp_path / "img" / f"{iid}_{split}_{i}.png",
                                 k, salt=k * 31 + i * 7 + len(split))
                items.append({"identity_id": iid, "split": split,
                              "image_uri": uri})
    return {
        "dataset": "hermetic",
        "identity_ids": list(identity_ids),
        "alias_of": {i: ALIASES[i] for i in identity_ids},
        "code_of": {i: f"RID_{i}" for i in identity_ids},
        "deleted_label": DELETED,
        "forget_identity_ids": ["001"],
        "items": items,
        "seed": 17,
    }


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _design(rf, tmp_path, version="v4", man=None):
    """One hermetic design of one version, over the same images.

    Built through the same constructor ``verify_manifest`` rebuilds through, so
    a difference between two versions' designs is a difference the frozen
    artifacts would carry and not an artifact of the fixture.
    """
    man = man if man is not None else _disk_manifest(tmp_path)
    build = getattr(rf, f"build_design_{version}")
    return build("ppubench", "fs_001", ["001"], [17, 42, 123], [17, 42, 123],
                 [17], man=man, images=rf.held_out_images(man)), man


def _cells(design, kind):
    return [c for c in design["cells"] if c["kind"] == kind]


def _identity_of_code(code, man):
    for iid in man["identity_ids"]:
        if man["code_of"][iid] == code:
            return iid
    return None


def _wrong_label(man, iid, not_this=None):
    for i, alias in sorted(man["alias_of"].items()):
        if i != iid and alias != not_this:
            return alias
    return DELETED


def _fill_router(rows, man):
    out = []
    for r in rows:
        code = man["code_of"][r["image_identity_id"]]
        out.append({**r, "routed_code": code, "routed_code_correct": True,
                    "routable": True, "observation_pending": False,
                    "filled_by": "fixture"})
    return out


def _fake_generation(row, man, forget, behaves, forced, derail):
    """The text a model with this behaviour would have emitted for one row.

    ``derail`` names CONDITIONS rather than rows, and a row whose condition is in
    it emits ``DERAILED`` whatever else the behaviour says.  That is the one knob
    these tests turn: the amendment is about which rows an unparseable output was
    counted against, so the fixture has to be able to put one exactly where a
    test says and nowhere else.
    """
    cond = row["condition"]
    if cond in derail:
        return DERAILED
    if cond == "baseline_forced_code":
        # The BEFORE, and it comes first because this condition is also a
        # forced-code one: the UNEDITED h produces every code's own alias,
        # forgotten codes included.  A fixture that answered Unknown here would
        # be exactly the adapter the baseline cell exists to rule out, and the
        # mediation verdict would fail for a reason the test did not mean.
        return man["alias_of"][row["forced_identity_id"]]
    if cond in forced:
        iid = row["forced_identity_id"]
        if behaves == "arbitrary":
            return _wrong_label(man, iid)
        return DELETED if iid in forget else man["alias_of"][iid]
    if cond == "natural_mediated":
        iid = _identity_of_code(row["routed_code"], man)
        return DELETED if iid in forget else man["alias_of"][iid]
    iid = row["image_identity_id"]
    if behaves == "arbitrary":
        return _wrong_label(man, iid, row.get("forced_label"))
    if behaves == "captured" and row.get("forced_label"):
        return row["forced_label"]
    return man["alias_of"][iid]


def _scored(rf, design, man, kind, behaves="design", derail=(), cell=0,
            conditions=None):
    """One cell's rows, generated by the fake model and scored by the real one."""
    rows = _cells(design, kind)[cell]["rows"]
    forget = set(design["forget_identity_ids"])
    forced = rf.FORCED_CODE_CONDITIONS_V3
    if kind == "natural":
        rows = _fill_router(rows, man)
    filled = []
    for r in rows:
        field = rf.generation_field(r["condition"], conditions)
        filled.append({**r, field: _fake_generation(r, man, forget, behaves,
                                                    forced, derail)})
    scored, missing = rf.score_rows_v2(filled, design["vocab"], conditions)
    assert not missing, f"the fixture left rows ungenerated: {missing}"
    return scored


KINDS = ("baseline", "intervention", "natural", "direct", "hybrid")


def _all_scored(rf, design, man, behaves="design", derail=()):
    return {kind: _scored(rf, design, man, kind, behaves, derail)
            for kind in KINDS}


def _gates(rf, spec, rows, accuracies=None):
    """The gates of ONE spec over ONE set of rows.

    Called with two specs on the same rows by the tests that matter, because the
    amendment's whole content is a difference between two denominators and the
    only honest way to show it is to compute both.
    """
    return rf.evaluate_gates_for(
        spec, rows["intervention"], rows["natural"], rows["direct"],
        rows["hybrid"], accuracies if accuracies is not None else {"17": 1.0},
        E2E_OK, baseline_rows=rows["baseline"])


def _verdicts(rf, spec, gates, man, dataset="ppubench"):
    applicability = rf.gate_applicability_v2(rf.image_content_audit(man),
                                             dataset, spec)
    return rf.verdicts_from_gates(gates, applicability, None, spec)


HYGIENE = "no_unparseable_or_multi_label_outputs"
DIAGNOSTIC = "auxiliary_output_hygiene"


@pytest.fixture(scope="module")
def derailed_run(rf, tmp_path_factory):
    """One hermetic run whose auxiliary probe leaves the vocabulary.

    The real PPUBench run's shape: every unparseable output is in
    ``hybrid_conflict_probe``, and every row a mediated verdict rests on is
    clean.  Built once for the module because the design hashes real image bytes.
    """
    tmp = tmp_path_factory.mktemp("derailed")
    design, man = _design(rf, tmp)
    rows = _all_scored(rf, design, man, derail=("hybrid_conflict_probe",))
    return {"rf": rf, "design": design, "man": man, "rows": rows, "tmp": tmp}


# ==========================================================================
# item 1 -- the denominator is a field of the design
# ==========================================================================

def test_the_same_rows_reach_two_verdicts_under_two_declared_scopes(
        rf, derailed_run):
    """THE amendment, as a computation rather than as a claim.

    One set of rows, two specs, two verdicts.  Nothing else differs: the
    thresholds are the same objects, the gate-to-verdict map is the same object,
    the conditions are the same object.  So the difference in the mediation
    verdict is attributable to the denominator and to nothing else -- which is
    the only way to show that a scope change and not a standards change is what
    moved it.
    """
    rows = derailed_run["rows"]
    aux = [r for r in rows["hybrid"] if r["unparseable"]]
    assert aux and len(aux) == len(rows["hybrid"]), \
        "the fixture has to derail every auxiliary row it was asked to"
    assert not any(r["unparseable"] for kind in KINDS if kind != "hybrid"
                   for r in rows[kind]), \
        "and it has to leave every other row clean, or the two scopes would not " \
        "be the only difference"

    v3 = _gates(rf, rf.PILOT_SPEC_V3, rows)
    v4 = _gates(rf, rf.PILOT_SPEC_V4, rows)

    assert v3[HYGIENE]["passed"] is False
    assert v3[HYGIENE]["value"]["unparseable"] == len(aux)
    assert v3[HYGIENE]["n"] == sum(len(rows[k]) for k in KINDS)
    assert "row_scope" not in v3[HYGIENE], \
        "v3's bytes are frozen; a field it never declared cannot appear in a " \
        "gate it computed"
    assert DIAGNOSTIC not in v3

    assert v4[HYGIENE]["passed"] is True
    assert v4[HYGIENE]["value"] == {"unparseable": 0, "multi_label_ambiguous": 0}
    assert v4[HYGIENE]["n"] == v3[HYGIENE]["n"] - len(rows["hybrid"])
    assert v4[HYGIENE]["row_scope"] == rf.HYGIENE_SCOPE_NON_AUXILIARY
    assert v4[HYGIENE]["conditions_excluded_from_the_count"] == \
        list(rf.AUXILIARY_CONDITIONS_V4)
    assert v4[HYGIENE]["n_rows_excluded"] == len(rows["hybrid"])
    assert v4[HYGIENE]["n_rows_in_the_whole_run"] == v3[HYGIENE]["n"], \
        "both denominators are named, so what was excluded is a stated fact"

    man = derailed_run["man"]
    assert _verdicts(rf, rf.PILOT_SPEC_V3, v3, man)["mediation"]["state"] == \
        "fail"
    assert _verdicts(rf, rf.PILOT_SPEC_V4, v4, man)["mediation"]["state"] == \
        "pass"
    # ... and the thresholds did not move to get there
    assert v3[HYGIENE]["threshold"] == v4[HYGIENE]["threshold"]
    assert rf.V4_THRESHOLD_DIFFERENCE == ()
    assert rf.V4_GATE_MAP_DIFFERENCE == ()


def test_the_scope_is_declared_in_the_design_and_not_left_to_the_code(rf):
    """A denominator chosen in the aggregation code is chosen after the rows
    exist.  Declared on the spec, it is frozen with everything else."""
    assert rf.PILOT_SPEC_V4.hygiene_gate_row_scope == \
        rf.HYGIENE_SCOPE_NON_AUXILIARY
    assert rf.PILOT_SPEC_V2.hygiene_gate_row_scope == \
        rf.PILOT_SPEC_V3.hygiene_gate_row_scope == rf.HYGIENE_SCOPE_ALL_ROWS
    assert rf.DECLARED_HYGIENE_SCOPE_V4["which_gate_it_scopes"] == HYGIENE
    assert rf.DECLARED_HYGIENE_SCOPE_V4["hygiene_gate_row_scope"] == \
        rf.PILOT_SPEC_V4.hygiene_gate_row_scope
    # Empty for the versions whose bytes are frozen, which is what keeps those
    # bytes reproducible: a merged ``**{}`` adds no key.
    assert rf.PILOT_SPEC_V2.declared_hygiene_scope == {}
    assert rf.PILOT_SPEC_V3.declared_hygiene_scope == {}
    assert rf.PILOT_SPEC_V2.declared_cell_compatibility == {}
    assert rf.PILOT_SPEC_V3.declared_cell_compatibility == {}


def test_a_row_scope_no_design_declared_is_refused(rf, derailed_run):
    """An unnamed scope is a verdict nobody pre-registered.

    Refused rather than defaulted: a ``else: verdict_rows = every_row`` would
    have made a typo in the spec silently reproduce the v3 behaviour, and the
    report would have carried a scope string the arithmetic ignored.
    """
    bogus = rf.PILOT_SPEC_V4._replace(hygiene_gate_row_scope="most_rows")
    with pytest.raises(RuntimeError, match="most_rows") as exc:
        _gates(rf, bogus, derailed_run["rows"])
    message = str(exc.value)
    assert rf.HYGIENE_SCOPE_ALL_ROWS in message
    assert rf.HYGIENE_SCOPE_NON_AUXILIARY in message, \
        "a refusal that does not name the scopes it accepts sends the reader " \
        "back to the source"


# ==========================================================================
# item 2 -- the auxiliary rows are reported, not dropped
# ==========================================================================

def test_the_auxiliary_rows_hygiene_is_reported_in_full(rf, derailed_run):
    """Excluding a row from a verdict is a statement about the verdict.

    What happened to the row is still a finding, and it is a more interesting
    one than the count that replaced it: the probe did not answer wrongly, it
    left the candidate vocabulary, which says the direct pathway ignores a code
    it was never trained on by answering something else entirely.
    """
    v4 = _gates(rf, rf.PILOT_SPEC_V4, derailed_run["rows"])
    assert DIAGNOSTIC in v4
    diag = v4[DIAGNOSTIC]
    assert diag["is_a_gate"] is False
    assert diag["feeds_no_verdict"] is True
    assert diag["passed"] is None, \
        "a diagnostic with a pass or a fail is a gate somebody forgot to map"
    assert diag["absent_from_gate_to_verdict"] is True
    assert DIAGNOSTIC not in rf.GATE_TO_VERDICT_V4
    assert diag["conditions"] == list(rf.AUXILIARY_CONDITIONS_V4)
    assert diag["n"] == len(derailed_run["rows"]["hybrid"])
    assert diag["value"]["unparseable"] == diag["n"]
    assert diag["rate_unparseable"] == 1.0
    assert diag["example_raw_outputs"] and \
        set(diag["example_raw_outputs"]) == {DERAILED}
    assert diag["example_row_ids"]
    assert set(diag["distinct_expected_labels_missed"]) == set(ALIASES.values()), \
        "every identity's alias went unproduced, which is what makes this a " \
        "property of the probe rather than of one image"


def test_no_verdict_and_no_summary_reads_the_diagnostic(rf, derailed_run):
    """``verdicts_from_gates`` iterates the gate map, so an entry outside it is
    invisible to every verdict -- which is the property, asserted rather than
    assumed from how the loop happens to be written."""
    rows = derailed_run["rows"]
    v4 = _gates(rf, rf.PILOT_SPEC_V4, rows)
    verdicts = _verdicts(rf, rf.PILOT_SPEC_V4, v4, derailed_run["man"])
    for name in rf.VERDICT_NAMES:
        assert DIAGNOSTIC not in verdicts[name]["gates"]
        assert DIAGNOSTIC not in verdicts[name]["failed_gates"]
    # The CLI summary filters on the same flag, so a diagnostic cannot appear in
    # a list of failed gates by virtue of carrying ``passed: None``.
    failed = [n for n, g in v4.items()
              if g.get("is_a_gate", True) and not g["passed"]]
    assert DIAGNOSTIC not in failed
    assert HYGIENE not in failed


# ==========================================================================
# item 3 -- the amendment is not a waiver
# ==========================================================================

def test_one_unparseable_output_outside_the_probe_still_fails_v4(rf, tmp_path):
    """SALMU's case, reproduced.

    The real run had twenty unparseable outputs: nineteen in the auxiliary probe
    and one in ``direct_control``, which is not auxiliary.  Scoping the count
    rescued PPUBench and left SALMU exactly where it was, and a test that only
    pinned the rescue would have pinned the half of the amendment that flatters
    it.
    """
    design, man = _design(rf, tmp_path)
    rows = _all_scored(rf, design, man,
                       derail=("hybrid_conflict_probe", "direct_control"))
    assert sum(1 for r in rows["direct"] if r["unparseable"]) == \
        len(rows["direct"])
    v4 = _gates(rf, rf.PILOT_SPEC_V4, rows)
    assert v4[HYGIENE]["passed"] is False
    assert v4[HYGIENE]["value"]["unparseable"] == len(rows["direct"])
    assert v4[HYGIENE]["n_rows_excluded"] == len(rows["hybrid"])
    assert _verdicts(rf, rf.PILOT_SPEC_V4, v4, man)["mediation"]["state"] == \
        "fail"
    assert HYGIENE in _verdicts(rf, rf.PILOT_SPEC_V4, v4,
                                man)["mediation"]["failed_gates"]


# ==========================================================================
# item 4 -- what v4 refuses to move
# ==========================================================================

def test_the_direct_code_ceiling_still_feeds_mediation(rf, tmp_path):
    """``direct_code_following_rate`` is measured ENTIRELY on auxiliary rows.

    A rule that said "auxiliary rows carry no verdict" would have moved it, and
    moving it would have undone the repair v2 made: an accuracy floor alone is
    satisfied by a direct adapter that answers from the code, so the ceiling is
    what makes ``direct_image_accuracy`` mean anything.  v4's claim is therefore
    narrower than the slogan and is the one that is true -- auxiliary rows are
    outside every mediated HYGIENE count.
    """
    assert rf.GATE_TO_VERDICT_V4["direct_code_following_rate"] == "mediation"
    assert rf.GATE_TO_VERDICT_V4["direct_code_following_rate"] == \
        rf.GATE_TO_VERDICT_V3["direct_code_following_rate"]
    design, man = _design(rf, tmp_path)
    rows = _all_scored(rf, design, man, behaves="captured")
    assert all(r["names_forced_code_label"] for r in rows["hybrid"]), \
        "the fixture has to produce a pathway actually captured by the code"
    v4 = _gates(rf, rf.PILOT_SPEC_V4, rows)
    assert v4["direct_code_following_rate"]["passed"] is False
    assert v4["direct_code_following_rate"]["n"] == len(rows["hybrid"]), \
        "still every auxiliary row: this gate counts the probe, it is not a " \
        "hygiene count over it"
    assert HYGIENE not in _verdicts(rf, rf.PILOT_SPEC_V4, v4,
                                    man)["mediation"]["failed_gates"]
    assert "direct_code_following_rate" in _verdicts(
        rf, rf.PILOT_SPEC_V4, v4, man)["mediation"]["failed_gates"], \
        "hygiene passes and mediation still fails, because the ceiling is a " \
        "verdict-bearing measurement of the auxiliary rows and not a check on " \
        "whether they were readable"


# ==========================================================================
# item 5 -- v4 changes no measurement
# ==========================================================================

def test_v4_and_v3_design_the_same_experiment(rf, tmp_path):
    """Same cells, same rows, same thresholds, same gate map, same conditions.

    Compared as artifacts rather than asserted from the constants, because the
    constants are what a later edit would move: two designs built over the same
    images by the same construction path have to agree on everything except the
    kind and the supersession record.
    """
    man = _disk_manifest(tmp_path)
    v3, _ = _design(rf, tmp_path, "v3", man=man)
    v4, _ = _design(rf, tmp_path, "v4", man=man)
    same = ("cells", "n_cells", "n_rows_total", "cells_by_kind", "conditions",
            "gate_thresholds", "gate_to_verdict", "vocab", "n_vocab",
            "decisive_conditions", "prompts", "route_protocol", "mediator_is_hard",
            "checkpoint_requirements", "bootstrap_plan", "router_seeds",
            "edit_seeds", "direct_seeds", "forget_identity_ids",
            "retained_identity_ids", "held_out_g", "deleted_label")
    for field in same:
        assert v3[field] == v4[field], field
    assert v3["kind"] != v4["kind"]
    assert rf.V4_CONDITION_TABLE_DIFFERENCE["conditions_added"] == ()
    assert rf.V4_CONDITION_TABLE_DIFFERENCE["conditions_removed"] == ()
    assert rf.V4_CONDITION_TABLE_DIFFERENCE["condition_entries_changed"] == ()
    assert rf.CONDITIONS_V4 is rf.CONDITIONS_V3, \
        "the condition table is the same object, so a row built under either " \
        "version carries the same execution text and the two cells cannot differ"


def test_the_two_specs_differ_in_exactly_the_fields_v4_declares(rf):
    """Enumerated rather than described.

    A ``PilotSpec`` is the single place one version differs from another, so the
    difference between two versions is a set of field names -- and a set a test
    can pin is a set a later edit cannot grow by accident.
    """
    differing = {f for f in rf.PilotSpec._fields
                 if getattr(rf.PILOT_SPEC_V3, f)
                 != getattr(rf.PILOT_SPEC_V4, f)}
    assert differing == {
        "version", "kind", "prereg_kind", "supersession_kind",
        "hygiene_gate_row_scope", "declared_hygiene_scope",
        "declared_cell_compatibility", "superseded_filenames"}, differing
    assert rf.PILOT_SPEC_V4.result_kind == rf.PILOT_SPEC_V3.result_kind, \
        "the result kind is deliberately NOT among the differences: see the " \
        "reuse test below"
    assert rf.PILOT_SPEC_V4.thresholds is rf.PILOT_SPEC_V3.thresholds
    assert rf.PILOT_SPEC_V4.gate_to_verdict is rf.PILOT_SPEC_V3.gate_to_verdict
    assert rf.PILOT_SPEC_V4.conditions is rf.PILOT_SPEC_V3.conditions
    assert rf.PILOT_SPEC_V4.embed_live_checkpoint_status is False, \
        "the v3 repair is carried forward, not re-litigated"
    assert rf.PHASES_BY_VERSION["v4"] == rf.PHASES_BY_VERSION["v3"]


def test_v4_reuses_v3s_result_kind_and_can_read_its_cells(rf, tmp_path):
    """Forced, not convenient.

    ``load_cell_result`` refuses a cell whose kind is not the aggregating spec's,
    so a new kind would have refused all twenty-eight filed cells and repeated
    twenty-six GPU phase invocations -- six of them 3000-step training runs -- to
    write byte-identical rows back.  A version whose only claim is that one gate
    counts a different subset of the SAME rows has no business re-measuring them.
    """
    assert rf.RESULT_KIND_V4 == rf.RESULT_KIND_V3
    assert rf.RESULT_KIND_V4 == "route_dependent_forgetting_result_v3"
    cell = {"cell_id": "hybrid__d17", "kind": rf.RESULT_KIND_V3, "n_rows": 0,
            "rows": [], "phase": "RF1D"}
    path = rf.cell_result_path("ppubench", "hybrid__d17", tmp_path)
    rf.atomic_write_json(path, cell)
    assert rf.load_cell_result("ppubench", "hybrid__d17", tmp_path,
                               rf.RESULT_KIND_V4)["cell_id"] == "hybrid__d17"
    # ... and a cell of a genuinely different kind is still refused, in words
    # that name the kind it was found to have
    rf.atomic_write_json(path, {**cell, "kind": rf.RESULT_KIND_V2})
    with pytest.raises(RuntimeError, match=rf.RESULT_KIND_V2):
        rf.load_cell_result("ppubench", "hybrid__d17", tmp_path,
                            rf.RESULT_KIND_V4)


def test_a_v4_report_lands_beside_the_v3_one(rf, tmp_path):
    """The corrective analysis does not overwrite the analysis it corrects.

    ``report_path`` formats with the spec's version, so v4 files ``_v4`` names and
    the committed v3 reports stay byte-identical.  A correction that replaced the
    record it corrected would leave no way to see that the two disagree.
    """
    for kind in rf.REPORT_FILENAMES:
        v3 = rf.report_path("salmu", kind, tmp_path, rf.PILOT_SPEC_V3)
        v4 = rf.report_path("salmu", kind, tmp_path, rf.PILOT_SPEC_V4)
        assert v4 != v3 and "_v4" in v4.name and "_v4" not in v3.name
        assert v4.name == rf.REPORT_FILENAMES[kind].format(dataset="salmu",
                                                           version="v4")
    # and the default is whatever the current spec is, one constant rather than
    # a name per phase -- so a version that stops being current stops being the
    # default with no second edit anywhere.
    assert rf.report_path("salmu", "RF2", tmp_path) == \
        rf.report_path("salmu", "RF2", tmp_path, rf.LATEST_PILOT_SPEC)
    # v4 is no longer the current one: v5 supersedes it for a STATUS reason
    # rather than a scientific one.  v4's re-analysis of v3's cells is correct
    # and stays committed, but it is corrective, and v4's own bytes say a
    # confirmatory result requires a run executed under the corrected scope from
    # the start.  v5 is that run, and what this test pins -- that a v4 report
    # lands BESIDE the v3 one rather than over it -- is unchanged by it.
    assert rf.LATEST_PILOT_SPEC is rf.PILOT_SPEC_V5
    assert rf.report_path("salmu", "RF2", tmp_path).name == \
        "rf_report_salmu_v5.json"


# ==========================================================================
# item 6 -- a cell another design produced is checked, and the pairing is
#          disclosed
# ==========================================================================

def _prereg_with_ancestor(this="a" * 64, ancestor="b" * 64):
    return {
        "design_sha256": this,
        "cell_design_lineage": {
            "ancestor": {"ancestor_version": "v3", "design_sha256": ancestor,
                         "manifest": "e2c_route_forgetting/manifests/x.json"},
            "what_a_v4_report_must_disclose": "disclose",
            "what_is_checked_before_a_cell_is_consumed": "check",
            "rule": "cells_are_consumed_from_the_ancestor_design_and_checked",
            "states": ("produced_under_this_design",
                       "reused_from_the_ancestor_design",
                       "unrecognized_design"),
        },
    }


def _stamped(rf, design, sha, version, kind):
    """A cell whose provenance names a design hash this test chose."""
    entry = next(c for c in design["cells"] if c["kind"] == kind)
    return entry["cell_id"], {
        "cell_id": entry["cell_id"], "kind": rf.RESULT_KIND_V4,
        "kind_of_cell": kind, "n_rows": entry["n_rows"],
        "rows": [], "scored": [],
        "run_provenance": {"preregistration_design_sha256": sha,
                           "pilot_version": version,
                           "preregistration_kind": rf.PREREG_V4_KIND,
                           "executing_commit": "0" * 40},
    }


def test_the_lineage_check_names_the_design_behind_every_cell(rf, tmp_path):
    """Reusing a result kind removed the only check that stood between an
    aggregate and a cell some other design produced.

    So the check moves to the thing that actually identifies a design: the hash
    each cell recorded for the pre-registration it ran against.  Both readings
    are named separately, because "v4 produced this" and "v3 produced this and v4
    is allowed to read it" are different claims, and a report that collapsed them
    would look self-produced.
    """
    design, _man = _design(rf, tmp_path)
    this, ancestor = "a" * 64, "b" * 64
    prereg = _prereg_with_ancestor(this, ancestor)
    cells = OrderedDict()
    for kind, sha, version in (("baseline", this, "v4"), ("hybrid", this, "v4"),
                               ("intervention", ancestor, "v3"),
                               ("direct", ancestor, "v3")):
        cid, doc = _stamped(rf, design, sha, version, kind)
        cells[cid] = doc
    got = rf.verify_cell_design_lineage(rf.PILOT_SPEC_V4, prereg, cells)
    assert got["n_cells"] == 4
    assert got["n_cells_by_state"] == {
        "produced_under_this_design": 2,
        "reused_from_the_ancestor_design": 2}
    assert got["this_design_sha256"] == this
    assert got["ancestor"]["design_sha256"] == ancestor
    assert got["ancestor"]["ancestor_version"] == "v3"
    for cid, doc in cells.items():
        assert got["per_cell"][cid]["recorded_pilot_version"] == \
            doc["run_provenance"]["pilot_version"]
    assert got["the_cells_were_not_relabelled"]
    assert rf.RESULT_KIND_VERSION[rf.RESULT_KIND_V4] == "v3", \
        "the reused kind still says which version produced the rows"


def test_the_lineage_check_refuses_a_cell_no_declared_design_produced(
        rf, tmp_path):
    """Refused rather than annotated.

    An aggregate that reported verdicts over rows produced under a design nobody
    named would be a result with no pre-registration behind it, which is the one
    thing this runner exists to make impossible.
    """
    design, _man = _design(rf, tmp_path)
    prereg = _prereg_with_ancestor()
    cid, doc = _stamped(rf, design, "c" * 64, "v2", "hybrid")
    with pytest.raises(RuntimeError, match=cid) as exc:
        rf.verify_cell_design_lineage(rf.PILOT_SPEC_V4, prereg, {cid: doc})
    assert "a" * 64 in str(exc.value) and "b" * 64 in str(exc.value), \
        "the refusal has to name both designs it would have accepted"
    # a cell of another RESULT kind is refused too, and in its own words
    with pytest.raises(RuntimeError, match="another result kind"):
        rf.verify_cell_design_lineage(
            rf.PILOT_SPEC_V4, prereg,
            {cid: {**doc, "kind": rf.RESULT_KIND_V2}})
    # ... and a pre-registration that declared a policy but names no ancestor is
    # refused rather than quietly accepting anything
    with pytest.raises(RuntimeError, match="names no ancestor design"):
        rf.verify_cell_design_lineage(
            rf.PILOT_SPEC_V4, {"design_sha256": "a" * 64,
                               "cell_design_lineage": {"ancestor": {}}},
            {cid: doc})


def test_the_lineage_check_is_silent_for_a_version_that_declared_no_policy(
        rf, tmp_path):
    """v2 and v3 never read another design's cells, so their reports must not
    gain a block describing a check they did not perform."""
    design, _man = _design(rf, tmp_path)
    cid, doc = _stamped(rf, design, "a" * 64, "v4", "hybrid")
    for spec in (rf.PILOT_SPEC_V2, rf.PILOT_SPEC_V3):
        assert rf.verify_cell_design_lineage(
            spec, {"design_sha256": "a" * 64}, {cid: doc}) is None
    assert rf.ANCESTOR_MANIFEST_BY_VERSION.get("v3") is None
    assert rf.ancestor_manifest_path(rf.PILOT_SPEC_V3, "ppubench") is None
    assert rf.SCOPE_AMENDMENT_EVIDENCE_BY_VERSION.get("v3") is None


# ==========================================================================
# item 7 -- the amendment says it came after the outcome, and shows its work
# ==========================================================================

def _pilot_v4(rf, tmp_path, monkeypatch, dataset="ppubench"):
    """A hermetic v4 pilot pre-registration.

    The design is built over fabricated images and stubbed matrix state, but the
    scope amendment's evidence is NOT stubbed: it is computed from the committed
    v3 record, because an amendment whose evidence a test invented would be
    testing the invention.
    """
    man = _disk_manifest(tmp_path)
    monkeypatch.setattr(rf, "load_manifest", lambda ds: man)
    monkeypatch.setattr(rf, "pilot_forget_set",
                        lambda ds, seeds: dict(SELECTION))
    monkeypatch.setattr(rf, "_promotion_forget_sets",
                        lambda ds: [f"fs_{i:03d}" for i in range(1, 8)])
    monkeypatch.setattr(rf, "reuse_inventory",
                        lambda ds, sets, seeds: dict(FAKE_INVENTORY))
    monkeypatch.setattr(rf, "verify_checkpoints_v2", lambda req: {
        "complete": False, "n_present": 0, "n_required": len(req["roles"]),
        "n_files_required": 0, "present": {},
        "absent": {role: {} for role in req["roles"]},
        "must_be_trained": sorted(req["roles"]),
        "n_must_be_trained": len(req["roles"]),
        "unexpectedly_absent": [], "note": "stub",
        "hashed_not_just_existence_checked": "stub"})
    policy = rf.pilot_seed_policy_v2(dataset, man=man)
    return rf.build_pilot_preregistration_v4(
        dataset, "fs_001", ["001"], policy["router_seeds"],
        policy["edit_seeds"], policy["direct_seeds"], man=man,
        images=rf.held_out_images(man), selection=dict(SELECTION))


@_needs_v3_record
def test_the_design_says_the_amendment_came_after_the_outcome(
        rf, tmp_path, monkeypatch):
    """Top level, and not buried in the rationale.

    A reader deciding whether an artifact was pre-registered should not have to
    open the block that justifies the change to find out that the change was made
    after the result.  ``executed`` stays False for the reason it always has: it
    is a field of the design, and a design that flipped it when work arrived
    would stop reproducing -- which is the v2 defect.
    """
    block = _pilot_v4(rf, tmp_path, monkeypatch)
    assert block["kind"] == rf.PREREG_V4_KIND
    assert block["post_outcome_scope_amendment"] is True
    assert block["preregistered"] is True and block["executed"] is False
    assert block["frozen_gates"]["hygiene_gate_row_scope"] == \
        rf.HYGIENE_SCOPE_NON_AUXILIARY
    assert block["frozen_gates"]["which_gate_it_scopes"] == HYGIENE
    assert block["frozen_gates"]["post_outcome_scope_amendment"] is True
    assert block["frozen_gates"]["not_adjustable_afterwards"] is True
    assert block["scope_amendment"]["what_this_block_is"]
    assert block["scope_amendment"]["this_manifests_dataset"] == "ppubench"
    lineage = block["cell_design_lineage"]
    assert lineage["ancestor"]["ancestor_version"] == "v3"
    assert lineage["accepted_result_kinds"] == (rf.RESULT_KIND_V4,)
    assert rf.design_sha256(block)


@_needs_v3_record
def test_the_amendment_evidence_names_both_datasets_and_only_one_moves(
        rf, tmp_path, monkeypatch):
    """The test of a scope correction is whether it moves the verdicts it should
    move and leaves the others where they were.

    An amendment justified by one dataset's numbers is an amendment tailored to
    that dataset, so both are recorded in both manifests and both are asserted
    here: PPUBench's mediation failed on this gate alone and is rescued, SALMU's
    failed on two gates and is not.
    """
    ev = _pilot_v4(rf, tmp_path, monkeypatch)["scope_amendment"]
    assert sorted(ev["per_dataset"]) == ["ppubench", "salmu"]
    assert ev["datasets_whose_mediation_verdict_moves"] == ["ppubench"]
    assert ev["datasets_whose_mediation_verdict_does_not_move"] == ["salmu"]
    assert ev["thresholds_that_moved"] == []
    assert ev["gate_mappings_that_moved"] == []

    pp = ev["per_dataset"]["ppubench"]
    assert pp["v3_hygiene_gate"]["row_scope"] == rf.HYGIENE_SCOPE_ALL_ROWS
    assert pp["v3_hygiene_gate"]["passed"] is False
    assert pp["the_same_counts_split"]["auxiliary"]["unparseable"] == \
        pp["v3_hygiene_gate"]["value"]["unparseable"], \
        "every unparseable output in that run was an auxiliary row"
    assert pp["the_same_counts_split"]["non_auxiliary"]["unparseable"] == 0
    assert pp["rows"]["counted_under_the_v4_scope"] == \
        pp["rows"]["in_the_whole_run"] - pp["rows"]["auxiliary"]
    assert pp["hygiene_gate_under_the_v4_scope"]["passed"] is True
    assert pp["mediation"]["state_under_v3"] == "fail"
    assert pp["mediation"]["state_under_v4"] == "pass"
    assert pp["mediation"]["moved_by_the_amendment"] is True

    sa = ev["per_dataset"]["salmu"]
    assert sa["mediation"]["state_under_v4"] == "fail"
    assert sa["mediation"]["moved_by_the_amendment"] is False
    assert "direct_image_accuracy" in \
        sa["mediation"]["mediated_gates_that_still_fail_under_v4"], \
        "all three direct models miss the frozen accuracy floor, and no " \
        "denominator changes that"
    assert HYGIENE in sa["mediation"]["mediated_gates_that_still_fail_under_v4"]
    assert sa["the_same_counts_split"]["non_auxiliary"]["unparseable"] == 1, \
        "one non-auxiliary direct output is strictly unparseable"
    direct = sa["the_other_gates_that_feed_mediation_are_untouched"][
        "direct_image_accuracy"]
    assert direct["passed"] is False
    assert sorted(direct["levels_that_failed"]) == [17, 42, 123], \
        "every direct seed failed the floor, so pooling hid nothing here and " \
        "the per-seed shape is what shows it"
    assert ev["auxiliary_conditions"] == list(rf.AUXILIARY_CONDITIONS_V4)
    assert ev["n_gates_in_the_v4_map"] == ev["n_gates_in_the_v3_map"]


@_needs_v3_record
def test_the_evidence_is_bound_to_the_bytes_it_was_read_from(
        rf, tmp_path, monkeypatch):
    """A rationale that cites bytes nothing re-hashes is a rationale nobody can
    check later, so the two v3 reports are named, hashed, and bound as inputs of
    the freeze."""
    block = _pilot_v4(rf, tmp_path, monkeypatch)
    for ds, src in block["scope_amendment"]["sources"].items():
        report = rf.report_path(ds, "RF2", None, rf.PILOT_SPEC_V3)
        manifest = rf.prereg_path_for(rf.PILOT_SPEC_V3, ds)
        assert src["v3_report"] == rf._rel(report)
        assert src["v3_report_sha256"] == _sha(report)
        assert src["v3_manifest_sha256"] == _sha(manifest)
        assert src["v3_report_kind"] == rf.RESULT_KIND_V3
    # The ancestor is this manifest's own dataset, and its recorded design hash
    # is read out of that file's bytes rather than recomputed: a superseded pilot
    # reports drift by design, so verifying it would be asking a question whose
    # answer is already known and unrelated.
    own = rf.prereg_path_for(rf.PILOT_SPEC_V3, "ppubench")
    assert block["cell_design_lineage"]["ancestor"]["design_sha256"] == \
        json.loads(own.read_text(encoding="utf-8"))["design_sha256"]
    assert block["scope_amendment"]["sources"]["ppubench"][
        "v3_design_sha256"] == \
        block["cell_design_lineage"]["ancestor"]["design_sha256"]
    bound = [rf._rel(p) for p in
             rf.EXTRA_FREEZE_INPUTS_BY_VERSION["v4"]("ppubench")]
    assert sorted(bound) == sorted(
        rf._rel(rf.report_path(ds, "RF2", None, rf.PILOT_SPEC_V3))
        for ds in ("ppubench", "salmu")), bound
    # Both reports are bound in BOTH manifests: the evidence is the pair, and a
    # pair cited from one side is a pair nobody can check from the other.
    assert len(bound) == 2


@_needs_v3_record
def test_the_supersession_record_names_v3_and_carries_every_earlier_repair(rf):
    """A reader who finds a v3 manifest in the tree has to see why it is not the
    design to run, and why v4 -- which changed a rule after seeing a result --
    is."""
    record = rf.supersession_record(rf.PILOT_SPEC_V4)
    assert record["kind"] == rf.SUPERSESSION_KIND_V4
    assert record["authoritative_design"] == "v4"
    assert record["n_repairs"] == len(rf.SUPERSESSION_ITEMS) + \
        len(rf.SUPERSESSION_ITEMS_V3) + len(rf.SUPERSESSION_ITEMS_V4)
    assert len(rf.SUPERSESSION_ITEMS_V4) == 5
    for item in ("the_hygiene_gate_has_one_denominator",
                 "auxiliary_output_hygiene_is_reported_and_feeds_nothing",
                 "direct_code_following_rate_stays_where_it_was",
                 "a_cell_from_another_design_is_checked_before_it_is_consumed",
                 "a_post_outcome_amendment_says_it_is_one"):
        assert item in record["repairs"], item
        assert set(record["repairs"][item]) == {"v3", "v4"}, \
            "each item names the two versions it moves between"
    assert "after a run's verdicts were read" in record["policy"], (
        "the policy has to say WHEN the amendment was made, not only that it "
        "was made")
    assert "corrective" in record["policy"]
    # every superseded artifact has notes of its own, so describing one with
    # another kind's limitations is impossible rather than merely avoided
    assert len(record["superseded"]) == 6
    assert {e["kind"] for e in record["superseded"]} == \
        {rf.PREREG_KIND, rf.PREREG_V2_KIND, rf.PREREG_V3_KIND}
    for entry in record["superseded"]:
        assert entry["status"] == "superseded_by_v4"
        assert entry["bytes_preserved"] is True and entry["executed"] is False
        assert rf.SUPERSEDED_NOTES[entry["kind"]]
    v3notes = rf.SUPERSEDED_NOTES[rf.PREREG_V3_KIND]
    reused = v3notes["its_cells_are_consumed_by_v4_and_are_not_relabelled"]
    for phrase in ("same bytes", "same kind string", "run_provenance"):
        assert phrase in reused, (phrase, reused)
    assert "scope reason" in v3notes["why_it_is_superseded"]


@_needs_v3_record
def test_an_amendment_refuses_without_the_evidence_it_cites(rf, monkeypatch):
    """An amendment whose evidence cannot be read is an assertion, so the
    constructor refuses to build one rather than recording a rationale with holes
    in it."""
    for ds in ("ppubench", "salmu"):
        p = rf.report_path(ds, "RF2", None, rf.PILOT_SPEC_V3)
        assert p.is_file(), p
        monkeypatch.setattr(rf, "REPORTS_DIR", rf.Path("e2c_absent_reports"))
        with pytest.raises(RuntimeError, match="scope amendment cites"):
            rf.v4_scope_amendment_evidence(ds)
        monkeypatch.undo()


# ==========================================================================
# the filed record, re-read under v4
# ==========================================================================

@pytest.fixture(scope="module")
def verifiable_v4(rf):
    """Skip unless a frozen v4 pilot verifies to ZERO problems HERE.

    The v4 manifests bind the runner that froze them, so editing that file after
    the freeze makes them report drift -- the documented state of a superseded
    artifact, and a reason not to assert rather than a defect.
    """
    missing = _absent(_V4_MANIFESTS.values())
    if missing:
        pytest.skip("the v4 pilots are not frozen in this checkout: "
                    + ", ".join(missing))
    why = []
    for ds, path in sorted(_V4_MANIFESTS.items()):
        got = rf.verify_manifest(path)
        if not got["valid"]:
            why.append(f"{ds}: " + "; ".join(got["problems"]))
    if why:
        pytest.skip("a frozen v4 pilot does not verify to zero problems here -- "
                    + " | ".join(why))
    return {ds: json.loads(p.read_text(encoding="utf-8"))
            for ds, p in _V4_MANIFESTS.items()}


@_needs_v3_record
def test_the_committed_v4_pilots_declare_the_amendment(rf, verifiable_v4):
    """Pinned against the artifacts that were actually frozen, not a fixture.

    A repair that holds on a hermetic pilot and not on the committed one is a
    repair to the fixture.
    """
    for ds, doc in sorted(verifiable_v4.items()):
        assert doc["kind"] == rf.PREREG_V4_KIND
        assert doc["dataset"] == ds
        assert doc["post_outcome_scope_amendment"] is True
        assert doc["executed"] is False and doc["preregistered"] is True
        assert doc["frozen_gates"]["hygiene_gate_row_scope"] == \
            rf.HYGIENE_SCOPE_NON_AUXILIARY
        assert doc["scope_amendment"]["this_manifests_dataset"] == ds
        assert sorted(doc["scope_amendment"]["per_dataset"]) == \
            ["ppubench", "salmu"]
        assert doc["cell_design_lineage"]["ancestor"]["design_sha256"] == \
            json.loads(_V3_MANIFESTS[ds].read_text(
                encoding="utf-8"))["design_sha256"]
        v3 = json.loads(_V3_REPORTS[ds].read_text(encoding="utf-8"))
        assert doc["n_cells"] == v3["n_cells"]
        assert doc["n_rows_total"] == v3["n_rows"], \
            "v4 designs the same experiment v3 ran, so its cell and row counts " \
            "are the counts the filed cells have"


@_needs_v3_record
@_needs_v3_cells
def test_rf2_rereads_the_filed_v3_cells_under_v4(rf, verifiable_v4, tmp_path):
    """THE corrective analysis, end to end and with no GPU.

    The cells filed under v3 are read as they stand -- same bytes, same kind
    string, same provenance naming v3 and the commit that executed them -- and
    the one thing that differs is the denominator the hygiene gate counts.  The
    verdicts have to come out the way the frozen evidence predicted, and the
    report has to say whose cells they are.
    """
    for ds, doc in sorted(verifiable_v4.items()):
        out = tmp_path / ds
        out.mkdir()
        rep = rf.phase_rf2(ds, prereg=doc, out=out)
        assert rep["kind"] == rf.RESULT_KIND_V4
        assert rep["preregistration_design_sha256"] == doc["design_sha256"]
        assert rep["n_cells"] == doc["n_cells"]

        lineage = rep["cell_design_lineage"]
        assert lineage["n_cells"] == doc["n_cells"]
        assert sum(lineage["n_cells_by_state"].values()) == doc["n_cells"]
        assert set(lineage["n_cells_by_state"]) <= set(lineage["states"])
        assert all(e["result_kind"] == rf.RESULT_KIND_V3
                   for e in lineage["per_cell"].values()), \
            "the cells are not relabelled: they still say which kind they are"

        gates = rep["gates"]
        assert gates[HYGIENE]["row_scope"] == rf.HYGIENE_SCOPE_NON_AUXILIARY
        assert gates[HYGIENE]["n"] == \
            gates[HYGIENE]["n_rows_in_the_whole_run"] - \
            gates[HYGIENE]["n_rows_excluded"]
        assert gates[DIAGNOSTIC]["is_a_gate"] is False
        assert rep["n_entries_in_gates"] == \
            rep["n_gates_the_verdicts_read"] + 1
        assert rep["entries_in_gates_that_are_not_gates"] == [DIAGNOSTIC]

        # The frozen evidence predicted exactly this, before these gates ran.
        predicted = doc["scope_amendment"]["per_dataset"][ds]["mediation"]
        assert rep["verdicts"]["mediation"]["state"] == \
            predicted["state_under_v4"], (ds, predicted)
        assert sorted(rep["verdicts"]["mediation"]["failed_gates"]) == sorted(
            predicted["mediated_gates_that_still_fail_under_v4"])
        # RF2 re-hashed what it could and named what it could not
        verification = rep["input_verification"]
        assert verification["n_bindings_not_recheckable"] == \
            len(verification["bindings_not_recheckable"])
        assert sorted(p.name for p in out.iterdir()) == \
            [f"rf_report_{ds}_v4.json"]
        # ... and the v3 report it corrects is untouched
        assert _sha(_V3_REPORTS[ds]) == \
            doc["scope_amendment"]["sources"][ds]["v3_report_sha256"]
