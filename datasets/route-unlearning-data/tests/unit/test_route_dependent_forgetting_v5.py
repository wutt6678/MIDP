"""Route-dependent forgetting, v5: a run that is prospective rather than a
re-reading of a finished one.

v4 ran to completion and its analysis is correct.  What it cannot be is
confirmatory, and its own bytes say so: every one of the 28 cells it aggregated
was filed by v3, every one of them resolves to
``reused_from_the_ancestor_design``, and the runner states in the v4 block that a
confirmatory result requires a run executed under the corrected scope from the
start.  Re-running the same cells under a new scope name is not a second
measurement of anything; it is the same measurement read twice.

v5 is that run.  What the tests below pin is what makes it one:

  1. the scope is FROZEN rather than restated -- v5's thresholds, gate map and
     conditions are v4's objects, by ``is`` and not by equality, so "the scope did
     not move" is a property of the module rather than a claim about it;
  2. v5 inherits v4's scope keys but NOT v4's amendment claim, because
     ``post_outcome_scope_amendment: true`` is a statement about v4's history and
     inheriting it verbatim would make every v5 artifact say it amended a scope
     after reading its own outcome;
  3. there is no ancestor, so ``reused_from_the_ancestor_design`` is unreachable
     and a cell v1-v4 filed is REFUSED rather than aggregated;
  4. the calibration split is disjoint from the test images IN CONTENT, which is
     the only disjointness that matters -- PPUBench's held-out filenames are bytes
     that also appear in its train split, so a split that partitioned filenames
     would partition nothing;
  5. the grid varies the SCHEDULE only, C0 is read out of the frozen protocol
     rather than typed beside it, and the selection rule REFUSES rather than
     picking the best of a lot that cannot clear the confirmatory floor;
  6. the forget sets are DERIVED, and a set an earlier design's committed manifest
     names is excluded -- which is the defect this version exists to remove;
  7. the design's cells are qualified by forget set in the id, in every row's
     ``cell_id`` and in every ``row_id`` prefix together, so fifteen intervention
     cells do not collapse into three dictionary slots;
  8. the freeze-time listing of what already existed is filed BESIDE the manifest
     and not inside it, because a listing covered by ``design_sha256`` would stop
     reproducing the moment the run it constrains began -- that is the v2 defect,
     arriving through a check added to prevent a different one.

Every test here is CPU-only.  The ones that read the committed tree say so and
skip with the absent path named, because a check that passes in a checkout
missing the thing it checks is a check that passes everywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location(
        "route_forgetting_v5_under_test",
        _SCRIPTS / "e2c_v3_route_dependent_forgetting.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: Loaded once, at import, because the gates below need the runner's own path
#: constants to say which committed artifact is missing -- and a gate that typed
#: those paths would be a second source for them.
_RF = _load()


@pytest.fixture(scope="module")
def rf():
    return _RF


# --------------------------------------------------------------------------
# what a checkout has to have
# --------------------------------------------------------------------------
#
# The scope, the grid and the selection rule are properties of the runner and
# need nothing but the module, so they run in a bare clone.  The development
# split's disjointness is a property of the committed images, and the forget-set
# rule's answer is a property of the committed matrix and manifests, so the tests
# that read either are gated and name what is missing.

_MANIFESTS = {ds: _RF.DATASET_ROOT / _RF.MANIFEST_PATHS[ds]
              for ds in ("ppubench", "salmu")}
_MATRICES = {ds: _RF.DATASET_ROOT / _RF.MATRIX_MANIFEST_DIR / f"matrix_{ds}.json"
             for ds in ("ppubench", "salmu")}
_CELLS = {ds: _RF.DATASET_ROOT / _RF.MATRIX_CELLS[ds]
          for ds in ("ppubench", "salmu")}
_PILOTS = {ds: _RF.DATASET_ROOT / _RF.MANIFEST_DIR / f"rf_pilot_{ds}_v4.json"
           for ds in ("ppubench", "salmu")}


def _absent(paths):
    return [str(p) for p in paths if not Path(p).is_file()]


def _gate(paths, what):
    missing = _absent(paths)
    return pytest.mark.skipif(
        bool(missing),
        reason=(f"{what} reads {len(missing)} committed artifact(s) this "
                f"checkout does not have: " + ", ".join(missing)))


_needs_images = _gate(list(_MANIFESTS.values()), "the development split")
#: A glob over an absent directory is empty rather than an error, so "no cells"
#: is probed as its own condition: a gate built only on the paths a glob returned
#: would pass in the one checkout it exists to describe.
_EMPTY = [ds for ds, p in sorted(_CELLS.items()) if not p.is_dir()]
_needs_tree = pytest.mark.skipif(
    bool(_absent([*_MANIFESTS.values(), *_MATRICES.values(), *_PILOTS.values()])
         or _EMPTY),
    reason=("the forget-set rule reads the committed manifests, matrix and "
            "matrix cells; missing: "
            + ", ".join([*_absent([*_MANIFESTS.values(), *_MATRICES.values(),
                                   *_PILOTS.values()]), *_EMPTY])))


# --------------------------------------------------------------------------
# a hermetic manifest whose filenames carry the image index
# --------------------------------------------------------------------------
#
# The development split is defined on the index the generator wrote into the
# filename, so a stub whose names do not carry one is a stub the split cannot be
# tested against.  Five identities, eight train images and three test images
# each: j=0..5 fit, j=6,7 development, j=8..10 test -- the same 6:2:3 ratio the
# committed data has, at a size a test can build in a temporary directory.

ALIASES = OrderedDict((("001", "Oden"), ("002", "Seri"), ("003", "Kael"),
                       ("004", "Dax"), ("005", "Rue")))
IDENTITIES = tuple(ALIASES)
N_TRAIN = 8
N_TEST = 3
#: Read rather than retyped, so a change to how many images the rule holds out
#: moves these tests instead of leaving them asserting the old arithmetic.
N_DEV = _RF.N_DEVELOPMENT_IMAGES_PER_IDENTITY_V5
N_FIT = N_TRAIN - N_DEV
#: The design tests run against ``salmu`` rather than a made-up dataset name,
#: because ``required_checkpoints_v5`` indexes ``MATRIX_CELLS`` and ``ROUTE_DIRS``
#: by dataset and a name in neither is a KeyError rather than a design.
DS = "salmu"


def _write_image(path, token):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + token.encode("utf-8"))
    return str(path)


def _disk_manifest(tmp_path, n_train=N_TRAIN, n_test=N_TEST,
                   identity_ids=IDENTITIES, duplicate_dev_bytes=False,
                   unnumbered=None):
    """A manifest with real image bytes on disk and an index in every filename.

    ``duplicate_dev_bytes`` reproduces the one condition the split exists to
    refuse: a development image whose BYTES also appear in the test split.  It is
    a parameter rather than a separate fixture because the refusal is about the
    bytes and not about the filenames, and a fixture that also changed the names
    would not have isolated it.
    """
    items, first_test = [], {}
    for iid in identity_ids:
        for j in range(n_train):
            items.append({"identity_id": iid, "split": "train",
                          "image_uri": _write_image(
                              tmp_path / "img" / f"SAL_{iid}_{j:02d}.png",
                              f"{iid}-train-{j}")})
        for j in range(n_test):
            k = n_train + j
            first_test.setdefault(iid, f"{iid}-test-0")
            items.append({"identity_id": iid, "split": "test",
                          "image_uri": _write_image(
                              tmp_path / "img" / f"SAL_{iid}_{k:02d}.png",
                              f"{iid}-test-{j}")})
    if duplicate_dev_bytes:
        # Overwrite the first development image of the first identity with the
        # bytes of that identity's first test image.  Same filename, different
        # contents, so a split that compared filenames would see no overlap.
        iid = identity_ids[0]
        j = n_train - _RF.N_DEVELOPMENT_IMAGES_PER_IDENTITY_V5
        Path(tmp_path / "img" / f"SAL_{iid}_{j:02d}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + first_test[iid].encode("utf-8"))
    if unnumbered is not None:
        items.append({"identity_id": identity_ids[0], "split": "train",
                      "image_uri": _write_image(tmp_path / "img" / unnumbered,
                                                "not-an-index")})
    return {
        "dataset": "hermetic",
        "identity_ids": list(identity_ids),
        "alias_of": {i: ALIASES[i] for i in identity_ids},
        "code_of": {i: f"RID_{i}" for i in identity_ids},
        "deleted_label": "Unknown",
        "forget_identity_ids": [identity_ids[0]],
        "items": items,
        "seed": 17,
    }


# --------------------------------------------------------------------------
# a calibration selection on disk
# --------------------------------------------------------------------------

PASSING = {
    "C0_incumbent": {17: 0.92, 42: 0.94},
    "C1_half_lr": {17: 0.95, 42: 0.95},
    "C2_half_as_many_steps_again": {17: 0.88, 42: 0.90},
    "C3_double_lr": {17: 0.60, 42: 0.70},
}


def _filed_selection(rf, monkeypatch, tmp_path, accuracies=None):
    """A selection shaped exactly like the one RFC files, at a stub path.

    Built by APPLYING the frozen rule to invented measurements rather than by
    writing the selection document out by hand: a hand-written stub would let the
    document and the rule disagree, and the design constructor would then be
    tested against a shape nothing produces.
    """
    out = rf.select_direct_schedule(accuracies or PASSING)
    p = tmp_path / "rf_calibration_selection_hermetic_v5.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    monkeypatch.setattr(rf, "calibration_selection_path",
                        lambda ds, path=None: p)
    return out


def _v5_design(rf, tmp_path, monkeypatch, man=None, n_sets=5,
               router_seeds=(17, 42, 123), edit_seeds=(17, 42, 123),
               direct_seeds=(17, 42, 123)):
    """A hermetic v5 design over ``n_sets`` synthetic forget sets."""
    man = man if man is not None else _disk_manifest(tmp_path)
    _filed_selection(rf, monkeypatch, tmp_path / "sel")
    forget_sets = [{"set_id": f"fs_test_{i:03d}",
                    "targets": [IDENTITIES[i % len(IDENTITIES)]]}
                   for i in range(n_sets)]
    forget_ids = sorted({t for s in forget_sets for t in s["targets"]})
    design = rf.build_design_v5(
        DS, forget_sets, forget_ids, list(router_seeds), list(edit_seeds),
        list(direct_seeds), man=man, images=rf.held_out_images(man))
    return design, man, forget_sets


def _cells(design, kind):
    return [c for c in design["cells"] if c["kind"] == kind]


# ==========================================================================
# item 1 -- the scope is frozen, and "frozen" means it is v4's object
# ==========================================================================


def test_the_scope_is_v4s_object_and_not_a_copy_of_it(rf):
    """``is``, not ``==``: a copy that happens to agree today is a scope that can
    be moved in one place and left standing in another, and no test comparing
    values would notice."""
    spec = rf.PILOT_SPEC_V5
    assert spec.thresholds is rf.GATE_THRESHOLDS_V4
    assert spec.gate_to_verdict is rf.GATE_TO_VERDICT_V4
    assert spec.conditions is rf.CONDITIONS_V4
    assert spec.forced_code_conditions is rf.FORCED_CODE_CONDITIONS_V4
    assert spec.mediated_conditions is rf.MEDIATED_CONDITIONS_V4
    assert spec.auxiliary_conditions is rf.AUXILIARY_CONDITIONS_V4
    assert spec.decisive_conditions is rf.DECISIVE_CONDITIONS_V4
    assert spec.hygiene_gate_row_scope == rf.HYGIENE_SCOPE_NON_AUXILIARY
    # and it is the same object v4 aggregates under, so a threshold edited for v5
    # would move v4's reports too -- which is the point of sharing rather than a
    # hazard of it
    assert spec.thresholds is rf.PILOT_SPEC_V4.thresholds


def test_v5_inherits_v4s_scope_but_not_its_amendment_claim(rf):
    """The one deliberate departure from reusing v4's declaration by identity.

    ``post_outcome_scope_amendment`` is a statement about v4's HISTORY and not
    about the scope.  Inheriting ``DECLARED_HYGIENE_SCOPE_V4`` verbatim would put
    ``true`` at the top level of every v5 artifact, so a v5 design would announce
    that it had amended a scope after reading its own outcome -- the one thing v5
    exists to not have done.  So every scope-DEFINING key is inherited with v4's
    own value and the three keys that describe the amendment are not inherited at
    all, and both halves of that are asserted.
    """
    v4, v5 = rf.DECLARED_HYGIENE_SCOPE_V4, rf.DECLARED_HYGIENE_SCOPE_V5
    not_inherited = rf._V5_KEYS_NOT_INHERITED_FROM_V4_SCOPE
    assert set(not_inherited) == {"post_outcome_scope_amendment",
                                  "what_that_means",
                                  "the_evidence_for_it_is_recorded_in_this_design"}
    # every key that DEFINES the denominator is inherited with v4's own value
    for k, want in v4.items():
        if k in not_inherited:
            continue
        assert k in v5, f"scope key {k!r} was dropped rather than inherited"
        assert v5[k] == want, f"scope key {k!r} moved from {want!r}"
    # the amendment's two explanatory keys are gone, and the flag itself is
    # RE-DECLARED as false rather than left absent: a reader who looks for the
    # field v4 put at the top level of its design should find an answer there
    # rather than have to interpret a missing key
    assert "what_that_means" not in v5
    assert "the_evidence_for_it_is_recorded_in_this_design" not in v5
    assert v5["post_outcome_scope_amendment"] is False
    assert v4["post_outcome_scope_amendment"] is True
    assert v5["scope_inherited_from"] == "v4"
    assert v5["why_the_amendment_fields_are_not_inherited"]
    assert v5["what_v5_is_instead"]
    assert rf.PILOT_SPEC_V5.declared_hygiene_scope is v5
    assert rf.PILOT_SPEC_V4.declared_hygiene_scope is v4


def test_v5_registers_no_scope_amendment_evidence(rf):
    """The absence IS the statement.

    v4 registers a function that derives its amendment evidence from both
    datasets' committed records.  v5 registers none, so a reader who asks what
    evidence v5 has that its scope was amended after an outcome gets nothing
    rather than a document explaining that there is no such evidence -- which
    would be a document about an amendment.
    """
    assert rf.SCOPE_AMENDMENT_EVIDENCE_BY_VERSION.get("v4") is not None
    assert rf.SCOPE_AMENDMENT_EVIDENCE_BY_VERSION.get("v5") is None


def test_v4s_phases_are_unchanged_and_v5_adds_exactly_rfc(rf):
    """A version that declared no calibration has no RFC, and declaring v5 did not
    add one to it."""
    assert "RFC" in rf.PHASES_BY_VERSION["v5"]
    assert "RFC" not in rf.PHASES_BY_VERSION["v4"]
    assert rf.PILOT_SPEC_V4.calibration_phase is False
    assert rf.PILOT_SPEC_V5.calibration_phase is True
    assert rf._phases_for(rf.PILOT_SPEC_V4) == rf.PHASES_BY_VERSION["v4"]
    v4, v5 = set(rf.PHASES_BY_VERSION["v4"]), set(rf.PHASES_BY_VERSION["v5"])
    assert v5 - v4 == {"RFC"}, f"v5 adds {sorted(v5 - v4)} and not only RFC"
    assert v4 - v5 == set(), f"v5 drops {sorted(v4 - v5)}"


def test_rfc_is_refused_for_a_version_that_did_not_declare_it(rf):
    """``--phase`` accepts every phase for every version, because the choices are
    one list.  What refuses is the dispatcher, and it refuses by naming the phases
    the version does have rather than by failing to find one."""
    args = rf._build_parser().parse_args(
        ["--dataset", DS, "--design-version", "v4", "--phase", "RFC"])
    with pytest.raises(RuntimeError, match="has no calibration phase"):
        rf._run_phase_for(rf.PILOT_SPEC_V4, args, "RFC")


# ==========================================================================
# item 3 -- no ancestor, so reuse is unreachable
# ==========================================================================


def test_v5_declares_no_ancestor(rf):
    assert rf.ancestor_manifest_path(rf.PILOT_SPEC_V5, "salmu") is None
    assert rf.ancestor_manifest_path(rf.PILOT_SPEC_V5, "ppubench") is None
    assert not rf.ANCESTOR_MANIFEST_BY_VERSION.get("v5")
    assert rf.DECLARED_CELL_COMPATIBILITY_V5["ancestor"] is None
    assert rf.DECLARED_CELL_COMPATIBILITY_V5["rule"] == "this_design_only"
    # v4 still declares one, and still can: declaring v5 did not edit v4
    assert rf.ANCESTOR_MANIFEST_BY_VERSION.get("v4")


def test_a_cell_filed_under_another_design_is_refused_not_aggregated(rf):
    """The check that makes the declaration operative.

    ``accepted`` holds this design's hash and, where one is declared, the
    ancestor's.  With no ancestor there is one accepted hash, so a cell stamped
    with another design's is refused -- and the declaration's own text says the
    zero reuse count is a consequence of what v5 declared rather than a
    measurement v5 happened to make.  This asserts the consequence.
    """
    prereg = {"design_sha256": "v" * 64,
              "cell_design_lineage": {"ancestor": None}}
    cells = {"natural__fs_1__g17__h42": {
        "kind": rf.RESULT_KIND_V5,
        "run_provenance": {"preregistration_design_sha256": "w" * 64}}}
    with pytest.raises(RuntimeError, match="cannot aggregate") as exc:
        rf.verify_cell_design_lineage(rf.PILOT_SPEC_V5, prereg, cells)
    msg = str(exc.value)
    assert "natural__fs_1__g17__h42 records design" in msg
    assert "its own design" in msg
    assert "ancestor" not in msg.split("consumes cells produced under")[1], \
        "with no ancestor declared, the refusal must not offer one"
    # a cell under THIS design is accepted, so the refusal is about the hash and
    # not about v5 refusing cells in general
    own = {"natural__fs_1__g17__h42": {
        "kind": rf.RESULT_KIND_V5,
        "run_provenance": {"preregistration_design_sha256": "v" * 64}}}
    out = rf.verify_cell_design_lineage(rf.PILOT_SPEC_V5, prereg, own)
    assert dict(out["n_cells_by_state"]) == {"produced_under_this_design": 1}
    assert out["rule"] == "this_design_only"
    assert not out["ancestor"], "no ancestor is named, so none can be matched"


def test_a_cell_of_another_result_kind_is_refused_before_the_hash_is_read(rf):
    """v5 takes a result kind of its own, so ``load_cell_result`` refuses a v1-v4
    cell at load and the lineage check is a second gate rather than the only
    one."""
    assert rf.RESULT_KIND_V5 != rf.PILOT_SPEC_V4.result_kind
    assert rf.RESULT_KIND_V5 != rf.PILOT_SPEC_V3.result_kind
    assert rf.KIND_V5 not in {rf.PILOT_SPEC_V4.kind, rf.PILOT_SPEC_V3.kind}
    assert rf.SPEC_BY_PREREG_KIND[rf.PREREG_V5_KIND] is rf.PILOT_SPEC_V5
    prereg = {"design_sha256": "v" * 64, "cell_design_lineage": {}}
    cells = {"c": {"kind": rf.PILOT_SPEC_V4.result_kind,
                   "run_provenance": {"preregistration_design_sha256": "v" * 64}}}
    with pytest.raises(RuntimeError, match="whose rows another design specified"):
        rf.verify_cell_design_lineage(rf.PILOT_SPEC_V5, prereg, cells)


# ==========================================================================
# item 4 -- the development split
# ==========================================================================


@_needs_images
def test_the_split_is_disjoint_from_the_test_images_in_content(rf):
    """Disjoint in image BYTES, which is the only disjointness that matters here.

    A development split that partitioned filenames would partition nothing on a
    dataset whose held-out filenames are bytes that also appear in train, and a
    configuration selected on it would be selected partly on the measurement it is
    supposed to be blind to.  ``dev_split`` hashes the bytes and refuses, so the
    property is checked on the committed images rather than asserted about them.
    """
    man = rf.load_manifest(DS)
    split = rf.dev_split(man)
    for pair, rec in split[
            "disjoint_in_image_content_not_only_in_filename"].items():
        assert rec["n_shared_image_bytes"] == 0, f"{pair} shares image bytes"
    assert split["every_train_image_is_placed"] is True
    assert split["development_is_balanced_across_identities"] is True
    # the test images are the manifest's own, never named by the rule
    assert sorted(split["test_image_uris"]) == sorted(
        i["image_uri"] for i in man["items"] if i["split"] == "test")
    assert split["n_fit_images"] + split["n_development_images"] == \
        split["n_train_images"]

    # The rule is a COUNT, so which indices it landed on is read back off the
    # data rather than assumed -- and on SALMU it is 6 and 7 for every identity,
    # which is the split the design was sized against.
    train_idx, by_uri = {}, {}
    for it in man["items"]:
        by_uri[it["image_uri"]] = it
        if it["split"] == "train":
            train_idx.setdefault(it["identity_id"], []).append(
                rf.image_index_within_identity(it))
    n_dev = rf.N_DEVELOPMENT_IMAGES_PER_IDENTITY_V5
    for uri in split["development_image_uris"]:
        it = by_uri[uri]
        top = sorted(train_idx[it["identity_id"]])[-n_dev:]
        assert rf.image_index_within_identity(it) in top
    assert split["n_development_images"] == n_dev * len(train_idx)
    assert {rf.image_index_within_identity(by_uri[u])
            for u in split["development_image_uris"]} == {6, 7}


@_needs_images
def test_ppubench_cannot_host_a_disjoint_split_and_says_so(rf):
    """A finding, not a skip.

    PPUBench's train and test images share byte-values, so NO division of its
    train images is disjoint from its test images in content -- the property is
    absent from the dataset rather than from the rule.  ``dev_split`` refuses
    instead of returning a split that overlaps the measurement the configuration
    is supposed to be blind to, which is why v5 calibrates on SALMU and why
    "PPUBench later" is a construction problem and not a scheduling preference.
    """
    man = rf.load_manifest("ppubench")
    with pytest.raises(RuntimeError, match="share .* image byte-value"):
        rf.dev_split(man)
    # and the overlap is in BYTES: the filenames are disjoint, which is exactly
    # why a rule that compared names would have reported a clean split here
    train = {i["image_uri"] for i in man["items"] if i["split"] == "train"}
    test = {i["image_uri"] for i in man["items"] if i["split"] == "test"}
    assert train.isdisjoint(test)


def test_the_split_places_every_train_image_or_refuses(rf, tmp_path):
    """No image is left for a caller to place by hand, and none is placed twice.

    The alternative is a split whose size depends on what the data happened to
    contain, and a development set that silently shrank would still report a mean.
    """
    man = _disk_manifest(tmp_path)
    split = rf.dev_split(man)
    assert split["every_train_image_is_placed"] is True
    assert split["n_fit_images"] == len(IDENTITIES) * N_FIT
    assert split["n_development_images"] == len(IDENTITIES) * N_DEV
    assert split["n_test_images"] == len(IDENTITIES) * N_TEST
    assert set(split["development_image_uris"]).isdisjoint(
        split["test_image_uris"])
    assert set(split["fit_image_uris"]).isdisjoint(
        split["development_image_uris"])
    assert sorted(split["development_images_per_identity"].values()) == \
        [N_DEV] * len(IDENTITIES)
    assert len(set(split["fit_image_uris"])
               | set(split["development_image_uris"])) == \
        split["n_fit_images"] + split["n_development_images"]


def test_an_identity_with_too_few_train_images_is_refused(rf, tmp_path):
    """Holding out two of an identity's two train images would leave it nothing to
    fit on, and would select a configuration on images that identity was never
    trained on -- differently from every identity that had images left."""
    man = _disk_manifest(tmp_path, n_train=N_DEV)
    with pytest.raises(RuntimeError, match="would leave it nothing to fit on"):
        rf.dev_split(man)


def test_two_train_images_at_one_index_are_refused(rf, tmp_path):
    """'The highest-indexed two' names no particular images if an index repeats,
    and breaking the tie by list order would make the split depend on the order
    the manifest's rows happen to be in rather than on the images."""
    man = _disk_manifest(tmp_path)
    man["items"].append({"identity_id": IDENTITIES[0], "split": "train",
                         "image_uri": _write_image(
                             tmp_path / "img2" / f"SAL_{IDENTITIES[0]}_01.png",
                             "a-second-image-at-index-one")})
    with pytest.raises(RuntimeError,
                       match="more than one train image at image index"):
        rf.dev_split(man)


def test_an_unnumbered_filename_is_refused_rather_than_guessed(rf, tmp_path):
    """The index is READ out of the filename the generator wrote.  A manifest whose
    filenames do not carry it has no development split rather than one inferred by
    re-sorting -- re-sorting would agree today and be a second rule tomorrow."""
    man = _disk_manifest(tmp_path, unnumbered="SAL_001_extra.png")
    with pytest.raises(RuntimeError, match="cannot read an image index"):
        rf.dev_split(man)


def test_a_split_overlapping_the_test_bytes_is_refused(rf, tmp_path):
    """The non-vacuity of item 4: same filenames, different contents, and the
    refusal still fires -- because it compares digests and not names."""
    man = _disk_manifest(tmp_path, duplicate_dev_bytes=True)
    with pytest.raises(RuntimeError, match="share .* image byte-value"):
        rf.dev_split(man)


# ==========================================================================
# item 5 -- the grid and the rule that selects from it
# ==========================================================================


def test_c0_is_read_from_the_frozen_protocol_not_typed_beside_it(rf):
    """A table that retyped the incumbent could drift from the protocol it claims
    to reproduce and still call itself the incumbent, and then "no change" would be
    a fifth configuration nobody named."""
    base = {k: rf.frozen_route_protocol()[k]
            for k in rf.OVERRIDABLE_SCHEDULE_KEYS}
    assert dict(rf.direct_schedule_candidates_v5()["C0_incumbent"]) == base
    assert rf._incumbent_schedule() == OrderedDict(
        (k, base[k]) for k in rf.OVERRIDABLE_SCHEDULE_KEYS)


def test_only_the_schedule_can_vary(rf):
    """The LoRA shape is not overridable from here at all.

    ``RouteSessionV2.train`` forwards steps, warmup, lr and repeat and nothing
    else, and the adapter's rank, alpha and dropout are fixed inside the research
    module this runner is not permitted to edit.  So the grid varies the schedule
    and D_s's adapter stays the same class of adapter as the route's, which keeps
    the comparison about the pathway rather than about the recipe.  Asserted on
    the keys rather than on a note, because a candidate that grew an ``r`` would
    be a configuration no training call could honour.
    """
    grid = rf.direct_schedule_candidates_v5()
    assert list(grid) == ["C0_incumbent", "C1_half_lr",
                          "C2_half_as_many_steps_again", "C3_double_lr"]
    for name, sched in grid.items():
        assert set(sched) <= set(rf.OVERRIDABLE_SCHEDULE_KEYS), name
        assert not set(sched) & {"r", "lora_r", "alpha", "lora_alpha",
                                 "dropout", "lora_dropout"}, name
    assert rf.OVERRIDABLE_SCHEDULE_KEYS == ("steps", "warmup", "lr", "repeat")
    with pytest.raises(RuntimeError):
        rf._direct_schedule(grid["C0_incumbent"], {"r": 16})
    # an override of a key that IS overridable returns a copy, not a mutation
    out = rf._direct_schedule(grid["C0_incumbent"], {"lr": 1e-6})
    assert out["lr"] == 1e-6 and grid["C0_incumbent"]["lr"] != 1e-6


def test_the_floor_is_the_confirmatory_threshold(rf):
    """Read from the gate's own threshold rather than set beside it: a threshold
    typed twice is a threshold that can be changed in one place."""
    assert rf.calibration_floor_v5() == \
        rf.GATE_THRESHOLDS_V4["min_direct_image_accuracy"]
    assert rf.calibration_floor_v5() == \
        rf.PILOT_SPEC_V4.thresholds["min_direct_image_accuracy"]


def test_the_rule_selects_the_highest_mean_above_the_floor(rf):
    out = rf.select_direct_schedule(PASSING)
    assert out["selected"]["candidate"] == "C1_half_lr"
    assert out["selected"]["is_the_incumbent"] is False
    assert out["n_clearing_the_floor"] == 2, "0.89 and 0.65 are below the floor"
    assert out["candidates"]["C2_half_as_many_steps_again"][
        "clears_the_floor"] is False
    assert out["selection_refused"] is None
    assert out["rule"] is rf.CALIBRATION_SELECTION_RULE_V5


def test_the_rule_refuses_rather_than_selecting_the_best_of_a_failing_lot(rf):
    """A rule that always returns a winner can select a configuration the pilot
    then fails on, and the failure would read as a result about route-dependent
    forgetting rather than as a result about the direct adapter being unable to
    learn the associations at all."""
    out = rf.select_direct_schedule({
        "C0_incumbent": {17: 0.40, 42: 0.44},
        "C1_half_lr": {17: 0.50, 42: 0.52},
        "C2_half_as_many_steps_again": {17: 0.60, 42: 0.62},
        "C3_double_lr": {17: 0.30, 42: 0.34},
    })
    assert out["selected"] is None
    assert out["n_clearing_the_floor"] == 0
    refused = out["selection_refused"]
    assert refused is not None and "no candidate reached" in refused["reason"]
    assert "the thresholds were not lowered" in refused["what_was_not_done"]


def test_the_rule_refuses_a_candidate_nobody_pre_registered(rf):
    good = {k: v for k, v in PASSING.items() if k != "C3_double_lr"}
    with pytest.raises(RuntimeError, match="the frozen grid does not contain"):
        rf.select_direct_schedule({**good, "C9_invented": {17: 0.99, 42: 0.99}})
    with pytest.raises(RuntimeError, match="the calibration is missing"):
        rf.select_direct_schedule(good)
    with pytest.raises(RuntimeError, match="was measured under seeds"):
        rf.select_direct_schedule(
            {**good, "C3_double_lr": {17: 0.95, 43: 0.95}})


def test_ties_break_toward_stability_and_then_the_incumbent(rf):
    """Both tie-breaks, in the order the frozen rule states them."""
    floor = rf.calibration_floor_v5()
    ok = {k: {17: floor, 42: floor} for k in
          ("C2_half_as_many_steps_again", "C3_double_lr")}
    # same mean, different spread: the stable one wins
    by_spread = rf.select_direct_schedule({
        **ok, "C0_incumbent": {17: floor - 0.02, 42: floor + 0.02},
        "C1_half_lr": {17: floor, 42: floor}})
    assert by_spread["selected"]["candidate"] == "C1_half_lr"
    assert by_spread["selected"]["tie_broken_by"] is not None
    # same mean AND same spread: the earlier candidate in the frozen table order,
    # which is C0, so a tie resolves toward changing nothing
    tied = rf.select_direct_schedule({
        **ok, "C0_incumbent": {17: floor, 42: floor},
        "C1_half_lr": {17: floor, 42: floor}})
    assert tied["selected"]["candidate"] == "C0_incumbent"
    assert tied["selected"]["is_the_incumbent"] is True
    assert tied["selected"]["n_tied_on_the_metric"] == 4
    assert tied["selected"]["ranking"][0] == "C0_incumbent"


def test_the_selection_rule_was_frozen_before_any_candidate_was_trained(rf):
    """The rule is a module constant and the artifact says so, which is what makes
    "applied to whatever it was given" checkable rather than a description of the
    author's intentions."""
    rule = rf.CALIBRATION_SELECTION_RULE_V5
    assert rule["frozen_before_any_candidate_is_trained"] is True
    assert rf.CALIBRATION_SEEDS_V5 == (17, 42)
    assert rule["scored_over"] == "the mean of the calibration seeds"
    assert rule["metric"].startswith("accuracy on the DEVELOPMENT images only")
    assert "the confirmatory gates never read" in rule["metric"], \
        "the metric has to say which images it is blind to, not only which it uses"


# ==========================================================================
# item 6 -- the forget sets are derived
# ==========================================================================


def test_the_version_order_the_rule_reads(rf):
    assert rf.DESIGN_VERSION_ORDER == ("v1", "v2", "v3", "v4", "v5")
    with pytest.raises(RuntimeError, match="is not a design version"):
        rf.consumed_forget_sets(DS, "v6")


@_needs_tree
def test_the_sets_are_derived_and_none_was_consumed_earlier(rf):
    """The ids are what the rule returns on the committed tree, not a list written
    into the test: a set added to the matrix or a manifest added to the tree moves
    the answer visibly rather than silently."""
    consumed = rf.consumed_forget_sets(DS, "v5")
    assert consumed["versions_considered"] == ["v1", "v2", "v3", "v4"]
    assert consumed["consumed"], "an earlier design names at least one set"
    sel = rf.pilot_forget_sets_v5(DS, [17, 42, 123])
    chosen = [s["set_id"] for s in sel["sets"]]
    assert len(chosen) == len(set(chosen)) == sel["n_sets"]
    assert not set(chosen) & set(consumed["consumed"]), \
        "a chosen set is one an earlier design already names"
    seen_consumed = False
    for entry in sel["candidates"]:
        if entry["set_id"] in consumed["consumed"]:
            seen_consumed = True
            assert entry["considered"] is False
            assert "a design frozen before v5 already names it" in \
                entry["why_not"]
        elif entry["considered"]:
            assert entry["complete"] is True, entry["why_not"]
    assert seen_consumed, "the rule never saw a set to exclude, so it is untested"
    # every chosen set is single-target, the clause the pilot's own rule gives
    for s in sel["sets"]:
        assert len(s["targets"]) == 1


@_needs_tree
def test_the_rule_refuses_edit_seeds_the_matrix_did_not_train(rf):
    """v5 reuses the matrix's edited-h adapters, so it cannot reuse adapters
    trained under different seeds -- and asking for them is refused rather than
    silently reinterpreted."""
    declared = list(rf._matrix_manifest(DS).get("edit_seeds") or [])
    if not declared:
        pytest.skip(f"the {DS} matrix manifest records no edit_seeds")
    with pytest.raises(RuntimeError, match="are not the ones the frozen"):
        rf.pilot_forget_sets_v5(DS, [*declared, max(declared) + 1])


# ==========================================================================
# item 7 -- the design
# ==========================================================================


def test_the_design_qualifies_every_cell_and_every_row_together(rf, tmp_path,
                                                                monkeypatch):
    """The id, each row's ``cell_id`` and each ``row_id`` prefix move as one.

    ``aggregate_cells_for`` selects a cell's rows by ``r["cell_id"] == cell_id``,
    so renaming a cell and leaving its rows would match no cell and the gates
    would be computed over an empty list.  The assertion that matters is the last
    one: every cell has rows, which is the property a silent mismatch destroys.
    """
    design, _man, _sets = _v5_design(rf, tmp_path, monkeypatch)
    assert design["n_cells"] == len(design["cells"]) == 71
    assert {k: len(v) for k, v in design["cells_by_kind"].items()} == {
        "baseline": 5, "intervention": 15, "natural": 45, "direct": 3,
        "hybrid": 3}
    assert design["n_rows_total"] == sum(c["n_rows"] for c in design["cells"])
    for cell in design["cells"]:
        cid = cell["cell_id"]
        assert cid.split("__")[0] in rf.CELL_KIND_PHASE, \
            f"{cid!r} does not begin with its kind, so no phase claims it"
        assert cell["rows"], f"cell {cid!r} has no rows at all"
        assert len(cell["rows"]) == cell["n_rows"]
        for r in cell["rows"]:
            assert r["cell_id"] == cid
            assert r["row_id"].startswith(cid + "::"), r["row_id"]


def test_the_fifteen_intervention_cells_do_not_collapse(rf, tmp_path,
                                                        monkeypatch):
    """Keying by edit seed alone puts fifteen cells into three dictionary slots,
    each overwritten by the last -- and a natural cell would then compose against
    an h trained to suppress a different identity, while the factorization gate
    still returned a number."""
    design, _man, forget_sets = _v5_design(rf, tmp_path, monkeypatch)
    sets = sorted(s["set_id"] for s in forget_sets)
    inter = _cells(design, "intervention")
    pairs = [(c["forget_set_id"], c["edit_seed"]) for c in inter]
    assert len(set(pairs)) == len(pairs) == 15
    assert sorted({p[0] for p in pairs}) == sets
    assert sorted({p[1] for p in pairs}) == [17, 42, 123]
    # and the role each one loads is a different name, so a different adapter
    roles = {rf.edited_h_role_name(design, c["edit_seed"], c["forget_set_id"])
             for c in inter}
    assert len(roles) == 15
    nat = _cells(design, "natural")
    triples = {(c["forget_set_id"], c["router_seed"], c["edit_seed"])
               for c in nat}
    assert len(triples) == len(nat) == 45
    base = _cells(design, "baseline")
    assert sorted(c["forget_set_id"] for c in base) == sets


def test_direct_and_hybrid_are_filed_once_because_they_are_identical(
        rf, tmp_path, monkeypatch):
    """The comparison IS the derivation.  ``build_direct_rows_v2`` takes no forget
    set, and the design checks that claim across all five per-set builds instead of
    inheriting it -- filing five copies would report one measurement five times."""
    design, _man, _sets = _v5_design(rf, tmp_path, monkeypatch)
    for kind in ("direct", "hybrid"):
        cs = _cells(design, kind)
        assert len(cs) == 3
        assert all(c.get("forget_set_id") is None for c in cs)
        assert all(not c["cell_id"].startswith(f"{kind}__fs_test_")
                   for c in cs), "a forget-set-independent cell was qualified"
        # Namespaced with the adapter, because ``direct__d17`` is the id v3 filed
        # a cell under and v5's D_s is a DIFFERENTLY trained adapter -- fit on the
        # train split minus the development images, under the schedule RFC
        # selected.  At v3's path, a committed v3 report would end up describing a
        # measurement nobody can reproduce.
        assert sorted(c["cell_id"] for c in cs) == sorted(
            rf.direct_cell_id(kind, s, rf.DIRECT_NAMESPACE_V5)
            for s in (17, 42, 123))
        assert rf.direct_cell_id(kind, 17) not in {c["cell_id"] for c in cs}
        for c in cs:
            for r in c["rows"]:
                assert r["cell_id"] == c["cell_id"]
                assert r["row_id"].startswith(c["cell_id"] + "::")
    assert design["forget_set_dependent_kinds"] == \
        list(rf.FORGET_SET_DEPENDENT_KINDS)
    assert design["forget_set_independent_kinds"] == ["direct", "hybrid"]
    assert "compared across all 5 per-set designs" in \
        design["rows_are_not_duplicated_across_forget_sets"]


def test_rf1d_writes_the_ids_its_design_names(rf, tmp_path, monkeypatch):
    """One rule and two callers: the shared design constructor and RF1D.

    A phase that trained a v5 adapter and filed it where no v5 design looks would
    return success and leave the aggregate refusing with a list of missing cells,
    which reads as a run that did not finish rather than as two spellings of one
    id.  And with no namespace -- every design frozen before v5 -- the ids are the
    ones those designs have always used, which is what makes this a change to v5
    rather than a change to what v1-v4 filed.
    """
    design, _man, _sets = _v5_design(rf, tmp_path, monkeypatch, n_sets=1)
    ns = design["direct_training"]["namespace"]
    have = {c["cell_id"] for c in design["cells"]}
    for seed in design["direct_seeds"]:
        for kind in ("direct", "hybrid"):
            assert rf.direct_cell_id(kind, seed, ns) in have
    assert rf.direct_cell_id("direct", 17) == "direct__d17"
    assert rf.direct_cell_id("hybrid", 42) == "hybrid__d42"
    assert rf.direct_cell_id("direct", 17, None) == "direct__d17"


@_needs_tree
def test_the_v3_direct_cells_are_intact_beside_the_v5_ones(rf):
    """The non-vacuity of the namespacing, stated so it survives the run.

    ``direct__d17`` was filed by the v3 run and read by v4's analysis, and v5
    writes ``direct__v5__d17``.  This test used to assert that the second one did
    NOT exist, which was true when it was written and became false the moment the
    run it was written for succeeded -- a statement about a moment in this tree's
    history rather than about the code.

    What is true either way is the reason the namespace exists: the two ids
    resolve to different files, and v5 did not write over v3's cell.  That last
    part is checked by reading the v3 cell and asking which version's result it
    says it is, so an overwrite would be caught rather than merely made unlikely
    by the two paths differing.
    """
    old = rf.cell_result_path(DS, rf.direct_cell_id("direct", 17))
    new = rf.cell_result_path(
        DS, rf.direct_cell_id("direct", 17, rf.DIRECT_NAMESPACE_V5))
    assert old != new, "the namespace produced the path it exists to avoid"
    assert old.is_file(), f"{old} was expected from the v3 run"
    v3 = json.loads(old.read_text(encoding="utf-8"))
    assert v3["kind"] == rf.SPEC_BY_VERSION["v3"].result_kind, (
        "the cell at the v3 path is no longer a v3 result: v5 wrote over the run "
        "whose digests committed v4 reports still cite")
    assert v3["cell_id"] == rf.direct_cell_id("direct", 17)
    # and when the v5 cell exists it is a v5 cell under its own id, not a rename
    # of the v3 one -- the same claim from the other side
    if new.is_file():
        v5 = json.loads(new.read_text(encoding="utf-8"))
        assert v5["kind"] == rf.PILOT_SPEC_V5.result_kind
        assert v5["cell_id"] == rf.direct_cell_id(
            "direct", 17, rf.DIRECT_NAMESPACE_V5)
        assert v5["run_provenance"]["pilot_version"] == "v5"


def test_the_union_field_is_a_coverage_statement_and_not_an_input(rf, tmp_path,
                                                                  monkeypatch):
    design, man, forget_sets = _v5_design(rf, tmp_path, monkeypatch)
    assert design["forget_set_id"] is None
    assert design["n_forget_sets"] == 5
    assert design["forget_identity_ids"] == sorted(
        {t for s in forget_sets for t in s["targets"]})
    assert "no cell was built over the union" in \
        design["forget_identity_ids_are_the_union"]
    with pytest.raises(RuntimeError, match="are not the union of its forget"):
        rf.build_design_v5(DS, forget_sets, ["001"], [17], [17], [17], man=man,
                           images=rf.held_out_images(man))
    with pytest.raises(RuntimeError, match="covers nothing"):
        rf.build_design_v5(DS, [], [], [17], [17], [17], man=man,
                           images=rf.held_out_images(man))


def test_the_binding_table_is_the_same_object_for_v1_to_v4(rf):
    """A single-set document reaches ``CELL_INPUT_BINDINGS`` itself and not a
    copy, so v1-v4 take the identical code path and their reports keep their
    bytes.  Only a document that varies several sets gets the extended table."""
    assert rf._binding_table({}) is rf.CELL_INPUT_BINDINGS
    assert rf._binding_table({"forget_set_id": "fs_00036363"}) is \
        rf.CELL_INPUT_BINDINGS
    multi = rf._binding_table({"forget_sets": [{"set_id": "a"},
                                               {"set_id": "b"}]})
    assert multi is not rf.CELL_INPUT_BINDINGS
    assert multi["edited_h"] == (rf.EDITED_H_ROLE_MULTI, "adapter")
    for k, v in rf.CELL_INPUT_BINDINGS.items():
        if k != "edited_h":
            assert multi[k] == v
    # a single-set document formats the same role name with or without a set
    assert rf.edited_h_role_name({}, 17) == rf.edited_h_role_name({}, 17, "fs_x")
    assert rf.edited_h_role_name({"forget_sets": [1]}, 17, "fs_x") == \
        rf.EDITED_H_ROLE_MULTI.format(forget_set="fs_x", edit_seed=17)


def test_a_multi_set_phase_invocation_must_say_which_set(rf, tmp_path,
                                                         monkeypatch):
    """Defaulting to the first would run a fifth of the design and return success,
    and the aggregate would then refuse with a list of missing cells rather than
    naming the choice that was never made."""
    design, _man, forget_sets = _v5_design(rf, tmp_path, monkeypatch)
    sets = [s["set_id"] for s in forget_sets]
    with pytest.raises(RuntimeError, match="so pass --forget-set"):
        rf.select_forget_set(design, None, "RF1H", "intervention")
    assert rf.select_forget_set(design, sets[2], "RF1H", "intervention") == \
        sets[2]
    with pytest.raises(RuntimeError, match="which this design does not vary"):
        rf.select_forget_set(design, "fs_nope", "RF1H", "intervention")
    single = {"forget_set_id": "fs_00036363"}
    assert rf.select_forget_set(single, None, "RF1H", "intervention") is None
    assert rf.design_forget_sets(single) == []
    with pytest.raises(RuntimeError, match="varies one forget set"):
        rf.select_forget_set(single, "fs_x", "RF1H", "intervention")


def test_the_design_reads_the_selection_and_refuses_without_one(rf, tmp_path,
                                                                monkeypatch):
    """READ and never passed in: a design handed a configuration could be frozen
    with one nothing selected, and no downstream check could tell the
    difference."""
    man = _disk_manifest(tmp_path)
    forget_sets = [{"set_id": "fs_a", "targets": ["001"]}]
    monkeypatch.setattr(rf, "calibration_selection_path",
                        lambda ds, path=None: tmp_path / "absent.json")
    with pytest.raises(RuntimeError, match="no calibration selection is filed"):
        rf.build_design_v5(DS, forget_sets, ["001"], [17], [17], [17], man=man,
                           images=rf.held_out_images(man))
    # a filed selection that REFUSED is not a configuration either
    refused = rf.select_direct_schedule(
        {k: {17: 0.10, 42: 0.10} for k in rf.direct_schedule_candidates_v5()})
    p = tmp_path / "refused.json"
    p.write_text(json.dumps(refused, indent=2), encoding="utf-8")
    monkeypatch.setattr(rf, "calibration_selection_path",
                        lambda ds, path=None: p)
    with pytest.raises(RuntimeError, match="REFUSED to select"):
        rf.build_design_v5(DS, forget_sets, ["001"], [17], [17], [17], man=man,
                           images=rf.held_out_images(man))


def test_the_selected_configuration_is_bound_to_its_artifact(rf, tmp_path,
                                                             monkeypatch):
    """The design names the schedule AND the digest of the document it was read
    out of, so a reader can check the configuration against the rule rather than
    take the design's word for it."""
    design, _man, _sel = _v5_design(rf, tmp_path, monkeypatch)
    dt = design["direct_training"]
    assert dt["selected_candidate"] == "C1_half_lr"
    assert dict(dt["schedule"]) == dict(
        rf.direct_schedule_candidates_v5()["C1_half_lr"])
    assert dt["selected_by"]["floor"] == rf.calibration_floor_v5()
    recorded = dt["selected_by"]["selection_artifact"]
    assert dt["selected_by"]["selection_artifact_sha256"] == hashlib.sha256(
        Path(recorded).read_bytes()).hexdigest()
    assert dt["namespace"] == rf.DIRECT_NAMESPACE_V5
    assert dt["n_fit_images"] == len(IDENTITIES) * N_FIT
    assert len(dt["development_image_uris"]) == len(IDENTITIES) * N_DEV
    asym = dt["training_data_asymmetry"]
    assert "it handicaps the direct pathway" in asym["direction"]
    assert "g is the frozen route" in asym["why_it_is_not_removed"]


def test_the_direct_adapters_are_namespaced_away_from_the_v3_ones(rf):
    """``direct_seed_17/`` holds weights whose digest a committed v3 cell result
    records.  Overwriting them would leave a filed report describing an adapter
    nobody can inspect."""
    plain = rf.direct_checkpoint_path(DS, 17)
    named = rf.direct_checkpoint_path(DS, 17, rf.DIRECT_NAMESPACE_V5)
    assert plain != named
    assert "direct_seed_17" in str(plain)
    assert named.parent.parent.name == f"direct_seed_17__{rf.DIRECT_NAMESPACE_V5}"
    assert named.name == plain.name == "adapter_model.safetensors"


def test_the_checkpoint_roles_are_qualified_by_forget_set(rf, tmp_path,
                                                          monkeypatch):
    design, _man, forget_sets = _v5_design(rf, tmp_path, monkeypatch)
    req = design["checkpoint_requirements"]
    roles = req["roles"]
    edited = sorted(r for r in roles if r.startswith("edited_h__"))
    assert len(edited) == len(set(edited)) == 15 == len(forget_sets) * 3
    for r in edited:
        assert roles[r]["exists_already"] is True, "v5 trains no edited h"
        assert roles[r]["cell_results"], "the matrix's own result is an input"
    assert req["forget_set_id"] is None
    assert [s["set_id"] for s in req["forget_sets"]] == \
        [s["set_id"] for s in forget_sets]
    # Every role v5 reads rather than trains is a declared input: the baseline h
    # and the router at EVERY seed, because v5 trains no router at any of them.
    # The seeds the frozen route did not establish were trained by the run v2
    # pre-registered, so declaring them absent -- as v2 correctly did, before that
    # run existed -- names work no phase of this design performs.
    assert roles["baseline_h"]["exists_already"] is True
    for s in req["router_seeds"]:
        assert roles[f"router_g__seed{s}"]["exists_already"] is True, \
            f"v5 trains no router, so g at seed {s} is an input it reads"
    # The only training work is the direct adapters, and that is the same set the
    # design says it produces: the two declarations have to agree, because a role
    # declared absent and produced by nobody is one nothing would ever supply.
    trained = sorted(r for r, s in roles.items() if not s["exists_already"])
    produced = sorted(r for r, s in roles.items()
                      if s["produced_by_this_design"])
    expected = sorted(f"direct_d__seed{s}" for s in req["direct_seeds"])
    assert trained == produced == expected
    assert not [r for r, s in roles.items()
                if not s["exists_already"] and not s["produced_by_this_design"]]
    assert all(roles[r]["n_files_required"] for r in roles)
    # what the design PRODUCES is a separate statement from what is on disk, and
    # it names the direct adapters alone: v5 trains no router and no edited h
    assert sorted(r for r, s in roles.items() if s["produced_by_this_design"]) == \
        sorted(f"direct_d__seed{s}" for s in req["direct_seeds"])
    assert "v5 trains no " in req["what_this_design_produces"]


@_needs_tree
def test_the_router_adapters_exist_and_are_still_not_outputs(rf, tmp_path,
                                                             monkeypatch):
    """The non-vacuity of skipping by provenance rather than by presence.

    On a tree the route has already been run on, the router adapters are there.
    Inferring "is an output" from "was not there" listed them, and the
    freeze-ordering check would have refused every v5 freeze -- a check that fails
    so reliably it would have been disabled.  This asserts the inputs exist AND
    are still not counted.
    """
    design, _man, _sets = _v5_design(rf, tmp_path, monkeypatch, n_sets=1)
    roles = design["checkpoint_requirements"]["roles"]
    skipped = sorted(r for r, s in roles.items()
                     if not s["produced_by_this_design"])
    present = [r for r in skipped
               if rf.resolve_recorded_path(roles[r].get("adapter") or "").is_file()]
    assert present, "no skipped role exists on disk, so the skip is untested"
    block = {"cells": design["cells"],
             "checkpoint_requirements": design["checkpoint_requirements"]}
    # Not ``== []``, which is what this asserted before the confirmatory run and
    # which the run correctly made false: the three D_s adapters ARE outputs of
    # this design, and once trained they belong in the listing.  A listing that
    # hid them would let the same design be frozen again over cells that already
    # exist.  What must never appear is an input v5 was built over.
    listed = rf.confirmatory_outputs_present(DS, block)
    counted_inputs = [e for e in listed
                      if e["what"] == "checkpoint"
                      and not roles[e["role"]]["produced_by_this_design"]]
    assert not counted_inputs, (
        f"{len(counted_inputs)} input(s) this design was built over were counted "
        f"as its output, e.g. {counted_inputs[0]['role']!r}: the freeze-ordering "
        f"check would refuse every v5 freeze on a tree the route had been run on")
    assert all(e["what"] == "cell"
               or roles[e["role"]]["produced_by_this_design"]
               for e in listed), (
        "the listing named something that is neither a cell of this design nor "
        "a checkpoint role it produces")


def test_the_full_v5_block_builds_through_the_shared_pilot_builder(
        rf, tmp_path, monkeypatch):
    """The design is not the pre-registration, and only the second one is frozen.

    ``build_design_v5`` is the cells and their rows; ``build_pilot_preregistration_v5``
    wraps it with the audit, the gate applicability, the bootstrap plan, the
    promotion rule and the supersession record, and it is the shared
    ``build_pilot_preregistration_for`` that guards all of it.  Testing only the
    design leaves every one of those guards unexercised with several forget sets,
    which is how a check written for one set -- intervention cells must equal the
    number of edit seeds -- survived into a version that varies five.

    The counts here are the hermetic ones, not the committed tree's: five
    synthetic sets over a stub manifest.  The real tree's 71 cells and 2076 rows
    are checked by the probe in tmp_e2c_v5, which reads a selection nobody has
    filed yet and so cannot be a test.
    """
    design, man, forget_sets = _v5_design(rf, tmp_path, monkeypatch)
    selection = {"dataset": DS, "n_sets": len(forget_sets),
                 "sets": forget_sets,
                 "targets": sorted({t for s in forget_sets for t in s["targets"]}),
                 "selection_rule": "hermetic-fixture", "candidates": []}
    block = rf.build_pilot_preregistration_v5(
        DS, forget_sets, selection["targets"], [17, 42, 123], [17, 42, 123],
        [17, 42, 123], man=man, images=rf.held_out_images(man),
        selection=selection)
    assert block["kind"] == rf.PREREG_V5_KIND
    assert block["n_cells"] == design["n_cells"] == 71
    assert block["n_rows_total"] == design["n_rows_total"]
    assert {k: len(v) for k, v in block["cells_by_kind"].items()} == {
        "baseline": 5, "intervention": 15, "natural": 45, "direct": 3,
        "hybrid": 3}
    assert block["executed"] is False
    assert block["preregistered"] is True
    assert block["post_outcome_scope_amendment"] is False
    assert block["cell_design_lineage"]["ancestor"] in (None, {})
    assert block["pilot_forget_set_selection"]["n_sets"] == 5
    # every guard in the shared builder ran, and the block carries the parts only
    # it adds -- which is the point of building it rather than the design
    for key in ("gate_applicability", "cluster_bootstrap", "promotion_rule",
                "missing_data_policy", "supersession", "held_out_images",
                "frozen_gates", "verdicts_to_be_reported"):
        assert key in block, f"the shared builder did not add {key}"
    # all eight earlier pilots -- v1 to v4 on both datasets -- are marked
    # superseded and bound as hashed inputs, so "preserved byte-identical" is a
    # checked property of every later verification rather than an intention
    sup = rf.supersession_record(rf.PILOT_SPEC_V5)
    assert len(sup["superseded"]) == 8
    assert sup["authoritative_design"] == "v5"
    # recorded relative to the dataset root, so verifying the table is not tied to
    # one filesystem root -- and each entry carries the digest of the bytes it is
    # claiming were preserved
    assert sorted(e["path"] for e in sup["superseded"]) ==         sorted(f"{rf.MANIFEST_DIR}/{n}"
               for n in rf.PILOT_SPEC_V5.superseded_filenames)
    assert all(e["sha256"] and e["bytes"] for e in sup["superseded"])


def test_a_v5_design_is_not_a_relabelled_v4_design(rf, tmp_path, monkeypatch):
    """Same construction path, different document -- and the difference is the
    forget sets, not a version string."""
    design, man, forget_sets = _v5_design(rf, tmp_path, monkeypatch, n_sets=1)
    one = forget_sets[0]
    v4 = rf.build_design_v4(DS, one["set_id"], one["targets"], [17, 42, 123],
                            [17, 42, 123], [17, 42, 123], man=man,
                            images=rf.held_out_images(man))
    assert "forget_sets" not in v4 and design["forget_sets"]
    assert design["n_cells"] == v4["n_cells"], \
        "one set means one set's worth of cells under either version"
    assert rf.canonical_json(design) != rf.canonical_json(v4)
    assert rf.design_forget_sets(v4) == []
    assert [s["set_id"] for s in rf.design_forget_sets(design)] == \
        [one["set_id"]]
    # the difference is which ids the cells carry, not how many there are
    assert {c["cell_id"] for c in design["cells"]} != \
        {c["cell_id"] for c in v4["cells"]}


# ==========================================================================
# item 8 -- the ordering is checkable rather than promised
# ==========================================================================


def test_the_listing_is_filed_beside_the_manifest_and_not_inside_it(rf,
                                                                   tmp_path,
                                                                   monkeypatch):
    """Outside ``design_sha256`` deliberately: a listing inside the design would be
    covered by the hash and would stop reproducing the moment the run it
    constrains began."""
    design, _man, _sel = _v5_design(rf, tmp_path, monkeypatch, n_sets=1)
    assert "confirmatory_ordering_rule" in design
    assert "frozen before any cell it names is produced" in \
        design["confirmatory_ordering_rule"]
    listing = rf.freeze_ordering_report_path(DS)
    assert listing.parent == rf.DATASET_ROOT / rf.REPORTS_DIR
    assert listing.name == f"rf_freeze_ordering_{DS}_v5.json"
    assert rf.MANIFEST_DIR != rf.REPORTS_DIR
    # the design carries the RULE and not the outcome of applying it, so nothing
    # in it names a path a run would create
    assert not set(design) & {"outputs_that_already_existed",
                              "n_outputs_that_already_existed",
                              "the_freeze_was_therefore"}
    extra = [str(p) for p in rf._v5_extra_freeze_inputs(DS)]
    assert str(listing) not in extra, \
        "the sidecar must not be an input the manifest hashes"


def test_the_sidecar_counts_the_roles_the_design_produces(rf, tmp_path,
                                                          monkeypatch):
    """The listing's own number, from the real ``_freeze_v5``.

    It reported five roles as training work for a design that trains three,
    because it counted ``not exists_already`` while two routers were declared
    absent and produced by nobody.  Counting what the design PRODUCES is what the
    field name says, so this asserts the number against the roles themselves
    rather than against a literal -- a literal would keep passing if both were
    wrong in the same way.

    Both writes are captured, so ``_freeze_v5`` runs end to end and files nothing.
    """
    _v5_design(rf, tmp_path, monkeypatch)
    written = []

    def capture(path, obj):
        written.append((path, obj))
        return path, "0" * 64

    monkeypatch.setattr(rf, "atomic_write_json", capture)
    monkeypatch.setattr(rf, "_freeze_write_and_verify",
                        lambda spec, args, block: (None, "0" * 64))
    # The listing is stubbed empty and the stub is stated, because this test is
    # about the COUNT the sidecar files, which is computed from the role table and
    # is independent of what the listing finds.  On a tree where the confirmatory
    # run has happened the real listing names every cell and adapter the design
    # produced, and ``_freeze_v5`` refuses -- correctly, since a design cannot be
    # frozen after the fact -- which would test the refusal instead of the count.
    monkeypatch.setattr(rf, "confirmatory_outputs_present",
                        lambda ds, block, cells=None: [])
    rf._freeze_v5(rf.PILOT_SPEC_V5, _freeze_args())

    sidecar = [o for p, o in written
               if p == rf.freeze_ordering_report_path(DS)]
    assert len(sidecar) == 1, \
        f"expected exactly one sidecar write, got {[str(p) for p, _ in written]}"
    listing = sidecar[0]
    roles = rf.required_checkpoints_v5(
        DS, rf.pilot_forget_sets_v5(DS, [17, 42, 123])["sets"],
        [17, 42, 123], [17, 42, 123], [17, 42, 123],
        rf.DIRECT_NAMESPACE_V5)["roles"]
    produced = sorted(r for r, s in roles.items()
                      if s["produced_by_this_design"])
    got = listing["what_was_listed"]["checkpoint_roles_that_are_training_work"]
    assert got == len(produced) == 3, f"{got} vs {produced}"
    assert listing["what_was_listed"]["cells"] == 71
    # and the ordering claim it exists to make
    assert listing["n_outputs_that_already_existed"] == 0
    assert listing["the_freeze_was_therefore"] == "before its outputs"


def test_a_role_nobody_produces_and_nobody_supplies_refuses(rf, tmp_path,
                                                           monkeypatch):
    """The invariant, proved to fire on the table that violated it.

    Asserting only that v5's table is coherent would pass on a constructor that
    never checked anything, so the same check is handed a corrupted copy -- a
    router the design reads turned into work nobody performs, which is exactly
    what copying v2's declaration did.

    Called directly rather than by wrapping the constructor: the check runs inside
    ``required_checkpoints_v5`` before it returns, so a wrapper can only corrupt
    the table after the check has already passed it, and the test then reports
    DID NOT RAISE against a check that is present and working.
    """
    design, _man, _sets = _v5_design(rf, tmp_path, monkeypatch)
    roles = design["checkpoint_requirements"]["roles"]
    rf._require_checkpoint_roles_coherent(roles)

    label = next(r for r in roles
                 if r.startswith("router_g__seed")
                 and r != f"router_g__seed{rf.EXISTING_ROUTER_SEED}")
    broken = {k: dict(v) for k, v in roles.items()}
    broken[label]["exists_already"] = False
    with pytest.raises(RuntimeError,
                       match="declared as work this design does not produce"):
        rf._require_checkpoint_roles_coherent(broken)
    # the refusal names the role, because "2 roles are wrong" is not actionable
    with pytest.raises(RuntimeError, match=label):
        rf._require_checkpoint_roles_coherent(broken)
    # and the real table was not mutated by building the corrupted copy
    assert roles[label]["exists_already"] is True


def test_confirmatory_outputs_present_skips_the_inputs_it_was_built_over(
        rf, tmp_path, monkeypatch):
    """Roles declared ``exists_already`` are the frozen route and the matrix's
    edited-h adapters.  Counting them would report every v5 freeze as a freeze
    after the fact, which is a check that fails so reliably it would be
    disabled."""
    design, _man, _sel = _v5_design(rf, tmp_path, monkeypatch, n_sets=1)
    block = {"cells": design["cells"],
             "checkpoint_requirements": design["checkpoint_requirements"]}
    roles = design["checkpoint_requirements"]["roles"]
    # The inputs v5 reads -- the frozen route at every seed, the baseline h and
    # the fifteen edited-h adapters -- are never counted, whether or not the run
    # has since produced the outputs this design does make.
    listed = rf.confirmatory_outputs_present(DS, block)
    assert not [e for e in listed if e["what"] == "checkpoint"
                and not roles[e["role"]]["produced_by_this_design"]]
    # The direct and hybrid cells are NOT qualified by forget set -- their ids are
    # direct__v5__d17, because they depend on no forget set -- so on a tree where
    # v5 has run they are on disk even though this test's forget sets are
    # synthetic, and the listing is right to name them.  What must not appear is
    # any cell of the synthetic sets, whose ids do carry the set.
    synthetic = [s["set_id"] for s in design["forget_sets"]]
    assert not [e for e in listed if e["what"] == "cell"
                and any(fs in e["cell_id"] for fs in synthetic)]
    # and a cell that exists IS listed, with its digest
    cid = design["cells"][0]["cell_id"]
    p = tmp_path / f"{cid}.json"
    p.write_text(json.dumps({"kind": "x"}), encoding="utf-8")
    monkeypatch.setattr(
        rf, "cell_result_path",
        # One cell exists and the rest do not.  Returning the same path for every
        # cell would make all nineteen "exist" and the assertion would pass for a
        # reason that has nothing to do with the listing.
        lambda ds, c, out=None: p if c == cid
        else tmp_path / "absent" / f"{c}.json")
    present = rf.confirmatory_outputs_present(DS, block)
    # Filtered to cells rather than read off the whole listing: the listing may
    # also name the D_s adapters this design produces, and those carry a "role"
    # and no "cell_id", so indexing the first entry would be reading whichever
    # kind of entry happened to be appended first.
    cells = [e for e in present if e["what"] == "cell"]
    assert [e["cell_id"] for e in cells] == [cid]
    assert cells[0]["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()


@_needs_tree
def test_the_report_discloses_the_direct_adapter_it_is_verdicts_about(rf):
    """The disclosure the mediation verdict depends on, derived not restated.

    Read against the real frozen design, which is the document the run consumed.
    The divergence is the point: it is computed by comparing two schedules, so the
    assertion below checks WHICH field differs and against what, rather than
    trusting a sentence that says a divergence exists.
    """
    prereg = json.loads(
        (rf.DATASET_ROOT / "e2c_route_forgetting" / "manifests"
         / "rf_pilot_salmu_v5.json").read_text(encoding="utf-8"))
    out = rf._direct_training_disclosure(prereg)
    assert out["selected_candidate"] == "C3_double_lr"
    assert out["is_the_incumbent"] is False
    assert out["schedule_diverges_from_the_frozen_route"] is True
    assert sorted(out["schedule_fields_that_differ"]) == ["lr"], \
        "exactly one schedule field diverges, and the report should say which"
    lr = out["schedule_fields_that_differ"]["lr"]
    assert lr["the_frozen_route"] == rf._incumbent_schedule()["lr"]
    assert lr["the_direct_adapter"] == 2 * lr["the_frozen_route"]
    # the asymmetry block is the design's own, not a paraphrase of it
    assert out["training_data_asymmetry"] == \
        prereg["direct_training"]["training_data_asymmetry"]
    assert "72" in out["training_data_asymmetry"]["statement"]
    assert "96" in out["training_data_asymmetry"]["statement"]
    cal = out["selected_by_a_pre_registered_calibration"]
    assert cal["rule_frozen_before_any_candidate_was_trained"] is True
    assert cal["floor"] == rf.calibration_floor_v5()
    assert cal["selection_artifact_sha256"] == hashlib.sha256(
        rf.calibration_selection_path(DS).read_bytes()).hexdigest()
    # and the block says what it does not do, because a disclosure that read as an
    # excuse would be worse than no disclosure at all
    assert "applied exactly as frozen" in out["this_block_waives_nothing"]
    assert "not as a general claim" in out["this_block_waives_nothing"]


@_needs_tree
def test_the_disclosure_reports_no_divergence_when_there_is_none(rf):
    """Non-vacuity: a disclosure that always claimed divergence would be a
    sentence, not a comparison.  The incumbent schedule has to come back clean."""
    prereg = json.loads(
        (rf.DATASET_ROOT / "e2c_route_forgetting" / "manifests"
         / "rf_pilot_salmu_v5.json").read_text(encoding="utf-8"))
    incumbent = dict(prereg["direct_training"])
    incumbent["schedule"] = OrderedDict(rf._incumbent_schedule())
    incumbent["selected_candidate"] = "C0_incumbent"
    out = rf._direct_training_disclosure({"direct_training": incumbent})
    assert out["schedule_diverges_from_the_frozen_route"] is False
    assert out["schedule_fields_that_differ"] == {}
    # the asymmetry is a property of the split and not of the schedule, so it
    # survives a change of candidate unchanged
    assert out["training_data_asymmetry"] == \
        prereg["direct_training"]["training_data_asymmetry"]


@_needs_tree
def test_versions_that_trained_d_s_under_the_route_schedule_report_nothing(rf):
    """The gate on ``direct_training`` is what keeps v1-v4's reports identical.

    Asserted over every earlier manifest rather than over one, because the claim
    is that no version before v5 carries the key -- and a v6 that added it would
    start emitting a disclosure about a design that never made one.
    """
    for name in ("rf_pilot_salmu.json", "rf_pilot_salmu_v2.json",
                 "rf_pilot_salmu_v3.json", "rf_pilot_salmu_v4.json",
                 "rf_pilot_ppubench.json", "rf_pilot_ppubench_v2.json",
                 "rf_pilot_ppubench_v3.json", "rf_pilot_ppubench_v4.json"):
        p = (rf.DATASET_ROOT / "e2c_route_forgetting" / "manifests" / name)
        if not p.is_file():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        assert "direct_training" not in doc, name


@_needs_tree
def test_the_disclosure_reaches_the_filed_report(rf):
    """The end of the chain: the report a reader opens, not the helper.

    Reads the committed RF2 and RF2P reports and requires the block in both, with
    the same content -- RF2P re-parses stored raw text on CPU and has to reproduce
    RF2, and a disclosure present in only one of them would mean the two reports
    disagree about what the experiment was.
    """
    reports = {}
    for kind in ("RF2", "RF2P"):
        p = rf.report_path(DS, kind, spec=rf.PILOT_SPEC_V5)
        if not p.is_file():
            pytest.skip(f"{kind} has filed no report at {rf._rel(p)}")
        reports[kind] = json.loads(p.read_text(encoding="utf-8"))
    for kind, rep in reports.items():
        got = rep.get("direct_training_disclosure")
        assert got, f"the {kind} report carries no direct_training_disclosure"
        assert got["selected_candidate"] == "C3_double_lr"
        assert got["schedule_diverges_from_the_frozen_route"] is True
    assert reports["RF2"]["direct_training_disclosure"] == \
        reports["RF2P"]["direct_training_disclosure"]


def test_the_freeze_checks_the_selection_before_it_does_anything_else(rf,
                                                                     tmp_path,
                                                                     monkeypatch):
    """The ordering v5 exists to enforce is that the configuration was chosen
    before the confirmatory design was frozen.  A freeze that failed later, for
    some unrelated reason, would not have demonstrated that ordering at all -- it
    would only have failed.  So the refusal is the FIRST thing, and it is the one
    a bare clone can observe."""
    monkeypatch.setattr(rf, "calibration_selection_path",
                        lambda ds, path=None: tmp_path / "absent.json")
    calls = []
    for name in ("load_manifest", "held_out_images", "pilot_seed_policy_v2",
                 "pilot_forget_sets_v5", "build_pilot_preregistration_v5",
                 "confirmatory_outputs_present", "_freeze_write_and_verify"):
        monkeypatch.setattr(rf, name,
                            lambda *a, _n=name, **k: calls.append(_n))
    args = rf._build_parser().parse_args(["--dataset", DS, "--preregister"])
    with pytest.raises(RuntimeError, match="no calibration selection is filed"):
        rf._freeze_v5(rf.PILOT_SPEC_V5, args)
    assert calls == [], f"the freeze did work before checking: {calls}"


def test_the_freeze_refuses_hand_passed_forget_ids(rf, tmp_path, monkeypatch):
    """v5's targets come from the rule.  A design whose targets were passed in is a
    design whose targets somebody chose, and choosing them after seeing which
    checkpoints exist is the selection the rule exists to make once."""
    _filed_selection(rf, monkeypatch, tmp_path / "sel")
    monkeypatch.setattr(rf, "load_manifest",
                        lambda ds: _disk_manifest(tmp_path / "m"))
    monkeypatch.setattr(rf, "pilot_seed_policy_v2",
                        lambda ds, man=None: {"router_seeds": [17, 42, 123],
                                              "edit_seeds": [17, 42, 123],
                                              "direct_seeds": [17, 42, 123]})
    args = rf._build_parser().parse_args(
        ["--dataset", DS, "--preregister", "--forget-ids", "001"])
    with pytest.raises(RuntimeError, match="come from the selection rule"):
        rf._freeze_v5(rf.PILOT_SPEC_V5, args)


def test_the_freeze_dispatches_by_version(rf):
    """``FREEZE_BY_VERSION`` is empty except for v5, so the shared freeze IS the
    default and declaring v5 did not change how v2, v3 or v4 are frozen."""
    assert rf.FREEZE_BY_VERSION == {"v5": rf._freeze_v5}
    for version in ("v2", "v3", "v4"):
        assert rf.FREEZE_BY_VERSION.get(version, rf._freeze_for) is \
            rf._freeze_for


def test_the_v5_freeze_binds_both_calibration_artifacts(rf):
    """The design's ``direct_training`` is read out of the selection, and the
    selection was made under the pre-registration.  Binding neither would leave a
    design naming a choice nothing could re-check."""
    names = sorted(Path(p).name for p in rf._v5_extra_freeze_inputs(DS))
    assert names == [f"rf_calibration_{DS}_v5.json",
                     f"rf_calibration_selection_{DS}_v5.json"]
    v4 = [Path(p).name
          for p in rf.EXTRA_FREEZE_INPUTS_BY_VERSION["v4"](DS)]
    assert v4 and not any(n.startswith("rf_calibration_") for n in v4), \
        "v4 binds the v3 reports its amendment cites and no calibration"


# ==========================================================================
# the calibration itself
# ==========================================================================


def _hermetic_root(rf, monkeypatch, tmp_path, man, dataset="hermetic"):
    """Point the runner at a temporary dataset root holding only a manifest.

    Freeze #1 binds ``DATASET_ROOT / MANIFEST_PATHS[dataset]`` as an input and
    writes under ``DATASET_ROOT``, so a hermetic freeze needs a hermetic root --
    and verifying it for real afterwards is what makes the test a test of the
    artifact rather than of the stub.

    ``dataset`` is a name because the runner indexes ``MANIFEST_PATHS``,
    ``CELLS_DIR_V2`` and ``MATRIX_CELLS`` by it, and a test that invented a name
    in none of those tables would be testing a KeyError rather than a freeze.
    """
    root = tmp_path / "root"
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    mpath = root / "manifests" / "hermetic_manifest.json"
    mpath.write_text(json.dumps(man, indent=2), encoding="utf-8")
    monkeypatch.setattr(rf, "DATASET_ROOT", root)
    monkeypatch.setitem(rf.MANIFEST_PATHS, dataset,
                        Path("manifests/hermetic_manifest.json"))
    monkeypatch.setattr(rf, "load_manifest", lambda ds: man)
    return root


def _freeze_args(dataset=DS, router_seeds=None, edit_seeds=None,
                 direct_seeds=None):
    """The fields ``_freeze_v5`` reads.

    ``dataset`` is a real name here and not a hermetic one because ``_freeze_v5``
    derives the forget sets from the committed matrix, and that derivation is part
    of what is under test.  The seeds default to None so the frozen seed policy
    supplies them exactly as the CLI does when they are not passed.
    """
    return argparse.Namespace(
        dataset=dataset, out=None, cells=None, forget_ids=None,
        router_seeds=router_seeds, edit_seeds=edit_seeds,
        direct_seeds=direct_seeds)


def _cal_args(dataset="hermetic", out=None):
    """The two fields ``freeze_calibration`` reads, for a stub dataset.

    Built directly rather than through ``_build_parser``, because ``--dataset``
    has choices and a hermetic dataset is not one of them.  What these tests are
    about is the freeze and its verification, not argparse's validation -- which
    the CLI tests cover on a dataset name the runner accepts.
    """
    return argparse.Namespace(dataset=dataset, out=out)


def test_freeze_1_is_freezable_without_a_gpu_and_verifies(rf, tmp_path,
                                                          monkeypatch):
    """Freeze #1 has to be possible before any training exists, or the grid could
    only ever be frozen after the candidates had been run -- which is a grid that
    agrees with its own outcome.

    ``verify_manifest`` is left unstubbed on purpose: it is the step that rebuilds
    the document from its recorded inputs, and a freeze whose own verifier cannot
    read back what it wrote is a freeze that filed an artifact nobody can check.
    """
    man = _disk_manifest(tmp_path)
    root = _hermetic_root(rf, monkeypatch, tmp_path, man)
    args = _cal_args()
    assert rf.freeze_calibration(args) == 0
    p = root / rf.MANIFEST_DIR / "rf_calibration_hermetic_v5.json"
    assert p.is_file()
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["kind"] == rf.CALIBRATION_PREREG_KIND_V5
    assert doc["frozen"] is True and doc["executed"] is False
    assert list(doc["grid"]["candidates"]) == \
        list(rf.direct_schedule_candidates_v5())
    assert doc["grid"]["seeds"] == list(rf.CALIBRATION_SEEDS_V5)
    assert doc["grid"]["n_trainings"] == \
        doc["grid"]["n_candidates"] * len(rf.CALIBRATION_SEEDS_V5)
    assert doc["grid"]["incumbent_read_from"] == "frozen_route_protocol()"
    assert doc["floor"] == rf.calibration_floor_v5()
    assert doc["development_split"]["n_development_images"] == \
        len(IDENTITIES) * N_DEV
    assert doc["development_split"]["n_fit_images"] == len(IDENTITIES) * N_FIT
    assert "D_s, the direct X -> Y adapter" == \
        doc["what_is_being_calibrated"]["component"]
    assert "g and h" in doc["what_is_being_calibrated"][
        "what_is_NOT_being_calibrated"]
    assert rf.verify_manifest(p)["valid"] is True


def test_freeze_1_refuses_if_any_calibration_output_exists(rf, tmp_path,
                                                           monkeypatch):
    """The pre-registration has to precede the training it constrains, so freezing
    after a candidate has been run would produce a grid that agrees with
    measurements already taken."""
    man = _disk_manifest(tmp_path)
    root = _hermetic_root(rf, monkeypatch, tmp_path, man)
    outdir = root / rf.CELLS_DIR_V2 / "hermetic" / rf.CALIBRATION_SUBDIR_V5
    (outdir / "C0_incumbent__seed17").mkdir(parents=True)
    (outdir / "C0_incumbent__seed17" / "calibration_result.json").write_text(
        "{}", encoding="utf-8")
    args = _cal_args()
    with pytest.raises(RuntimeError, match="calibration output\\(s\\) already"):
        rf.freeze_calibration(args)
    assert not (root / rf.MANIFEST_DIR /
                "rf_calibration_hermetic_v5.json").exists(), \
        "a refused freeze filed a document anyway"


def test_freeze_1_happens_once(rf, tmp_path, monkeypatch):
    """Re-freezing after the candidates have been trained would produce a document
    that agrees with the outcome, which is the thing it exists to make
    impossible."""
    man = _disk_manifest(tmp_path)
    _hermetic_root(rf, monkeypatch, tmp_path, man)
    args = _cal_args()
    assert rf.freeze_calibration(args) == 0
    with pytest.raises(RuntimeError, match="frozen once"):
        rf.freeze_calibration(args)


def test_rfc_verifies_the_preregistration_before_it_trains(rf, tmp_path,
                                                           monkeypatch):
    """Every other phase verifies before it loads, through ``load_prereg_for``.
    RFC reads the grid it is about to train, so a document edited after freezing
    has to be refused rather than trained against -- the edit would otherwise be
    invisible in the selection it produced."""
    man = _disk_manifest(tmp_path)
    root = _hermetic_root(rf, monkeypatch, tmp_path, man)
    with pytest.raises(RuntimeError, match="no frozen calibration"):
        rf.load_calibration_prereg("hermetic")
    assert rf.freeze_calibration(_cal_args()) == 0
    p = root / rf.MANIFEST_DIR / "rf_calibration_hermetic_v5.json"
    assert rf.load_calibration_prereg("hermetic")["kind"] == \
        rf.CALIBRATION_PREREG_KIND_V5
    doc = json.loads(p.read_text(encoding="utf-8"))
    # an edited floor: the document still parses and still names the right kind,
    # so only the rebuild catches it
    doc["floor"] = 0.01
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not verify"):
        rf.load_calibration_prereg("hermetic")
    # verify=False is the escape hatch every loader here has, and using it is
    # visible in what comes back rather than silently blessed
    assert rf.load_calibration_prereg("hermetic", verify=False)["floor"] == 0.01

    doc["kind"] = rf.PREREG_V5_KIND
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError, match="is not a v5 calibration"):
        rf.load_calibration_prereg("hermetic", verify=False)


def test_the_selection_step_files_json_and_not_a_path(rf, tmp_path,
                                                     monkeypatch):
    """The step that CONSUMES the grid, exercised end to end.

    ``select_direct_schedule`` is the rule and was already well covered.
    ``_rfc_select`` is the step around it: it reads eight filed measurements,
    checks each one was produced against the frozen pre-registration, refuses if
    any is missing, applies the rule, and FILES the outcome.  Only the filing was
    broken -- ``measurements_consumed`` held a ``Path`` and ``canonical_json``
    raised ``TypeError`` -- and it broke after nine and a half hours of training
    had already been paid for, because nothing tested the step.

    Every candidate is stubbed at the same accuracy, which also makes this a second
    test of the tie-break: all four clear the floor with zero spread, so the frozen
    table order has to pick the incumbent.
    """
    man = _disk_manifest(tmp_path)
    root = _hermetic_root(rf, monkeypatch, tmp_path, man, dataset=DS)
    assert rf.freeze_calibration(_cal_args(dataset=DS)) == 0
    cal = rf.load_calibration_prereg(DS)
    prereg = rf.calibration_prereg_path(DS)
    prereg_sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
    cal_dir = root / rf.CELLS_DIR_V2 / DS / rf.CALIBRATION_SUBDIR_V5
    for cand in rf.direct_schedule_candidates_v5():
        for seed in rf.CALIBRATION_SEEDS_V5:
            (cal_dir / f"{cand}__seed{seed}").mkdir(parents=True,
                                                    exist_ok=True)
            rf.atomic_write_json(rf.calibration_result_path(DS, cand, seed), {
                "development_accuracy": 0.95,
                "calibration_preregistration_sha256": prereg_sha})

    names = list(rf.direct_schedule_candidates_v5())
    out = rf._rfc_select(DS, cal, names)
    p = rf.calibration_selection_path(DS)
    assert p.is_file(), "the step returned without filing anything"
    filed = json.loads(p.read_text(encoding="utf-8"))
    assert filed["selected"]["candidate"] == names[0]
    assert filed["selected"]["is_the_incumbent"] is True
    assert filed["selected"]["n_tied_on_the_metric"] == len(names)
    assert filed["calibration_preregistration_sha256"] == prereg_sha
    # the property the TypeError was about: every path in the filed document is a
    # relative string that resolves to a file that exists
    assert sorted(filed["measurements_consumed"]) == sorted(names)
    for cand, per_seed in filed["measurements_consumed"].items():
        assert sorted(per_seed) == [str(s) for s in rf.CALIBRATION_SEEDS_V5]
        for seed, rel in per_seed.items():
            assert isinstance(rel, str), f"{cand}/{seed} filed a {type(rel)}"
            assert not Path(rel).is_absolute(), rel
            assert (root / rel).is_file(), rel
    # What was filed is exactly the canonical JSON of what the step returned --
    # the property the TypeError broke.  Compared through a round trip rather than
    # by ``out == filed``, because the rule it embeds holds TUPLES (its tie_breaks)
    # and JSON has no tuple: ``out == filed`` is false for a correctly filed
    # document, and an assertion that fails on success is worse than none.
    assert json.loads(rf.canonical_json(out)) == filed


def test_the_selection_step_refuses_a_measurement_from_another_grid(rf,
                                                                   tmp_path,
                                                                   monkeypatch):
    """The binding check inside the step, which is the one that makes "these eight
    measurements answer the frozen grid" a fact rather than an assumption."""
    man = _disk_manifest(tmp_path)
    root = _hermetic_root(rf, monkeypatch, tmp_path, man, dataset=DS)
    assert rf.freeze_calibration(_cal_args(dataset=DS)) == 0
    cal = rf.load_calibration_prereg(DS)
    names = list(rf.direct_schedule_candidates_v5())
    cal_dir = root / rf.CELLS_DIR_V2 / DS / rf.CALIBRATION_SUBDIR_V5
    for cand in names:
        for seed in rf.CALIBRATION_SEEDS_V5:
            d = cal_dir / f"{cand}__seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            rf.atomic_write_json(rf.calibration_result_path(DS, cand, seed), {
                "development_accuracy": 0.95,
                "calibration_preregistration_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="a different calibration"):
        rf._rfc_select(DS, cal, names)
    # and a grid with a hole in it is refused rather than ranked over what arrived
    good = hashlib.sha256(
        rf.calibration_prereg_path(DS).read_bytes()).hexdigest()
    for cand in names:
        for seed in rf.CALIBRATION_SEEDS_V5:
            rf.atomic_write_json(rf.calibration_result_path(DS, cand, seed), {
                "development_accuracy": 0.95,
                "calibration_preregistration_sha256": good})
    (cal_dir / f"{names[-1]}__seed{rf.CALIBRATION_SEEDS_V5[-1]}"
     / "calibration_result.json").unlink()
    with pytest.raises(RuntimeError, match="have no filed measurement"):
        rf._rfc_select(DS, cal, names)


def test_the_calibration_writes_beside_the_cells_not_among_them(rf):
    """No design consumes a calibration file: it is evidence about a choice and not
    an input to a verdict, and a reader who found it inside the cell table would
    reasonably assume an aggregate had read it."""
    d = rf.calibration_dir(DS, "C1_half_lr", 42)
    assert rf.CALIBRATION_SUBDIR_V5 == "calibration"
    assert d.parent.name == "calibration"
    assert d.parent.parent.name == DS
    assert d.name == "C1_half_lr__seed42"
    assert rf.calibration_result_path(DS, "C1_half_lr", 42) == \
        d / "calibration_result.json"
    assert rf.calibration_selection_path(DS).parent == \
        rf.DATASET_ROOT / rf.REPORTS_DIR
