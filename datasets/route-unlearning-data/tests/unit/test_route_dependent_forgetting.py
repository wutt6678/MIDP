"""Route-dependent forgetting: the design must be checkable with no model.

The claim this runner exists to test is that after editing ``h`` the output
follows the INTERVENED route rather than the image.  That claim is only worth
anything if the design that produces it is complete: a partial cross leaves a
cell of the image-by-route table unmeasured, and an empty one passes every
``all(...)`` gate while measuring nothing.

So the tests below exercise the CPU layers -- design construction, the
expectation each intervention requires, coverage, strict scoring, Delta_route
and its cluster bootstrap, the seven pre-registered gates, checkpoint
requirements and manifest freezing -- and never load a model.  RF1 is the only
phase that needs one, and the module says so rather than failing obscurely.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"

#: The five checkpoints ``verify_checkpoints(req, "ppubench", "fs_001")`` hashes,
#: laid out as ``required_checkpoints`` lays them out.
#:
#: Adapter weights are gitignored and live in the HF revision-pinned archive, so
#: a fresh clone -- which is what CI checks out -- has the tracked dataset and
#: matrix manifests and the tracked ``adapter_config.json`` beside each adapter,
#: but none of the adapter bytes.  Enumerated as exactly these five rather than
#: globbed: ``e2c_v3_real/outputs/realdata`` also holds adapters this test never
#: looks at, and a guard that counted them would skip on their account.  The
#: guarded test asserts the count against what the code hashed, so the two cannot
#: drift apart.
_PPU_ADAPTER = ("adapter_final", "adapter_model.safetensors")
_PPU_CHECKPOINTS = (
    [_ROOT / "e2c_v3_real" / "outputs" / "realdata" / sub / Path(*_PPU_ADAPTER)
     for sub in ("g_X_to_C", "h_C_to_Y")]
    + [_ROOT / "e2c_matrix" / "outputs" / "ppubench" / "cells" / "fs_001"
       / f"seed_{s}" / "edited_h" / Path(*_PPU_ADAPTER)
       for s in (17, 42, 123)])
_ABSENT_PPU_ADAPTERS = [str(p.relative_to(_ROOT)) for p in _PPU_CHECKPOINTS
                        if not p.is_file()]

_needs_ppu_weights = pytest.mark.skipif(
    bool(_ABSENT_PPU_ADAPTERS),
    reason=("asserts on the actual adapter bytes, which are gitignored and live "
            "in the HF revision-pinned archive, so a fresh clone has the tracked "
            "manifests but not the weights"
            + (f" -- {len(_ABSENT_PPU_ADAPTERS)} of {len(_PPU_CHECKPOINTS)} "
               f"absent, e.g. {_ABSENT_PPU_ADAPTERS[0]}"
               if _ABSENT_PPU_ADAPTERS else "")
            + ". The refusal half of this behaviour needs no weights and is "
              "asserted separately, so it still runs everywhere"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "route_forgetting_under_test",
        _SCRIPTS / "e2c_v3_route_dependent_forgetting.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rf():
    return _load()


# --------------------------------------------------------------------------
# a hermetic four-identity manifest, shaped like the PPUBench one
# --------------------------------------------------------------------------

ALIASES = {"001": "Oden", "002": "Seri", "003": "Kael", "004": "Dax"}


def _manifest(n_test=3, n_train=2, identity_ids=("001", "002", "003", "004")):
    items = []
    for iid in identity_ids:
        for i in range(n_train):
            items.append({"identity_id": iid, "split": "train",
                          "image_uri": f"/img/{iid}/train_{i}.png"})
        for i in range(n_test):
            items.append({"identity_id": iid, "split": "test",
                          "image_uri": f"/img/{iid}/test_{i}.png"})
    return {
        "dataset": "ppubench",
        "identity_ids": list(identity_ids),
        "alias_of": {i: ALIASES[i] for i in identity_ids},
        "code_of": {i: f"RID_{i}" for i in identity_ids},
        "deleted_label": "Unknown",
        "forget_identity_ids": ["001"],
        "images_per_identity": {i: n_train + n_test for i in identity_ids},
        "items": items,
        "seed": 17,
    }


def _design(rf, man=None, **kw):
    man = man or _manifest()
    images = rf.held_out_images(man)
    args = {"dataset": "ppubench", "forget_set_id": "fs_001",
            "forget_ids": ["001"], "router_seeds": [17, 42, 123],
            "edit_seeds": [17, 42, 123], "man": man, "images": images}
    args.update(kw)
    return rf.build_design(args.pop("dataset"), args.pop("forget_set_id"),
                           args.pop("forget_ids"), args.pop("router_seeds"),
                           args.pop("edit_seeds"), **args)


def _rows(design, cell=0):
    return design["cells"][cell]["rows"]


def _simulate(rows, man, behaves="design", noise=None):
    """Attach a stored generation to every row, as RF1 would.

    Three constant behaviours, which the design has to tell apart:

      ``design``  what the experiment expects: the output follows the FORCED
                  route where a route is forced, and follows the image on the
                  direct path -- i.e. exactly ``expected_label``.
      ``image``   the null behaviour: the code was edited but the model still
                  answers from the image.  A design that cannot distinguish this
                  from ``design`` is not measuring mediation.
      ``code``    the degenerate behaviour: the code dominates everywhere,
                  including the direct path that was never trained on.
    """
    noise = noise or {}
    out = []
    for r in rows:
        if behaves == "design":
            label = r["expected_label"]
        elif behaves == "image":
            label = r["image_label_of_record"]
        elif behaves == "code":
            label = r["forced_label"] or r["image_label_of_record"]
        else:
            raise AssertionError(f"unknown behaviour {behaves!r}")
        out.append({**r, "raw_text": noise.get(r["row_id"], label)})
    return out


def _scored(rf, design, behaves="design", noise=None, cell=0):
    man = _manifest()
    scored, missing = rf.score_rows(_simulate(_rows(design, cell), man,
                                              behaves, noise),
                                    rf.label_vocab(man))
    assert not missing
    return scored


# --------------------------------------------------------------------------
# 1. exact intervention coverage
# --------------------------------------------------------------------------

def test_the_cross_is_complete_and_its_denominator_is_stated(rf):
    """Every cell of the image-by-route table must exist.

    The decisive claim is "the output follows the route regardless of image
    identity", which is a claim about ALL combinations.  A sample of them would
    leave the unmeasured ones free to disagree.
    """
    man = _manifest()
    d = _design(rf, man)
    cov = d["cells"][0]["coverage"]

    assert cov["exact"] is True and cov["defects"] == []
    # 3 held-out images for the forgotten identity, 3 each for the 3 retained
    assert cov["per_condition"] == {
        "natural_mediated": 3,                       # forgotten images
        "retained_route_intervention": 9,            # 3 forgotten x 3 retained
        "forgotten_route_intervention": 9,           # 3 retained x 1 forgotten
        "retained_control": 9,                       # 3 retained images
        "direct_path": 12,                           # all 4 identities
    }
    assert cov["per_condition"] == cov["per_condition_required"]
    assert cov["n_rows"] == 42
    assert cov["decisive_rows"] == 18
    assert d["n_cells"] == 3 and d["n_rows_total"] == 126

    # every (image, forced code) pair the cross implies is present
    pairs = {(r["condition"], r["image_identity_id"], r["forced_identity_id"])
             for r in _rows(d)}
    for f in ("001",):
        for ret in ("002", "003", "004"):
            assert ("retained_route_intervention", f, ret) in pairs
            assert ("forgotten_route_intervention", ret, f) in pairs
            assert ("retained_control", ret, ret) in pairs


def test_a_partial_cross_is_a_defect_not_a_smaller_result(rf):
    """Dropping one arm of the cross must be named, not silently absorbed."""
    man = _manifest()
    d = _design(rf, man)
    rows = [r for r in _rows(d)
            if not (r["condition"] == "retained_route_intervention"
                    and r["forced_identity_id"] == "003")]
    cov = rf.coverage_audit(rows, man, ["001"], rf.held_out_images(man))

    assert cov["exact"] is False
    assert any("retained_route_intervention" in x for x in cov["defects"])
    assert any("missing cell of the cross" in x for x in cov["defects"])


# --------------------------------------------------------------------------
# 2. no vacuous pass over zero rows
# --------------------------------------------------------------------------

def test_zero_rows_never_reads_as_a_pass(rf):
    """``all([])`` is True and a mean over nothing serializes as null.

    This is the failure that let a granularity pilot report two green gates
    after evaluating no cells at all, so every layer here refuses rather than
    returning its neutral value.
    """
    man = _manifest()
    cov = rf.coverage_audit([], man, ["001"], rf.held_out_images(man))
    assert cov["exact"] is False and cov["n_rows"] == 0
    assert any("ZERO intervention rows" in x for x in cov["defects"])
    # and the empty audit still carries every key, so a caller cannot KeyError
    # into thinking the counts were merely absent
    assert set(cov["per_condition"]) == set(rf.CONDITIONS)

    with pytest.raises(RuntimeError, match="no gate has a denominator"):
        rf.evaluate_gates([], {"17": 1.0}, None)

    with pytest.raises(RuntimeError, match="Delta_route has no denominator"):
        rf.delta_route([], ["001"])


def test_a_gate_whose_condition_was_never_run_fails_and_says_so(rf):
    """A condition that was never run must fail loudly, not vanish.

    Only the conditions that are a gate's SOLE source can zero its denominator,
    so each is checked against the gate it feeds -- and natural_mediated is
    checked to feed none, because a gate that silently depended on context rows
    would be measuring the wrong thing.
    """
    d = _design(rf)
    scored = _scored(rf, d)
    e2e = {"predicted": 1.0, "observed": 1.0, "n_images": 12}
    full = rf.evaluate_gates(scored, {"17": 1.0, "42": 1.0, "123": 1.0}, e2e)
    assert full["passed"] is True, full["failed_gates"]

    for dropped, gate in (("forgotten_route_intervention",
                           "conditional_forgotten_route_suppression"),
                          ("direct_path",
                           "direct_path_code_following_rate")):
        subset = [r for r in scored if r["condition"] != dropped]
        got = rf.evaluate_gates(subset, {"17": 1.0}, e2e)
        g = got["gates"][gate]
        assert g["n"] == 0 and g["passed"] is False
        assert "zero rows" in g["failed_because"]
        assert got["passed"] is False

    # both sources of the retained-route gate
    subset = [r for r in scored
              if r["condition"] not in ("retained_route_intervention",
                                        "retained_control")]
    got = rf.evaluate_gates(subset, {"17": 1.0}, e2e)
    g3 = got["gates"]["conditional_retained_route_accuracy"]
    assert g3["n"] == 0 and g3["passed"] is False

    # natural_mediated is context, not a gate: dropping it changes no
    # denominator, which is itself the property being pinned
    subset = [r for r in scored if r["condition"] != "natural_mediated"]
    got = rf.evaluate_gates(subset, {"17": 1.0}, e2e)
    assert got["passed"] is True
    assert all(g["n"] for g in got["gates"].values() if g["name"] !=
               "router_held_out_accuracy")


# --------------------------------------------------------------------------
# 3. target/retain separation
# --------------------------------------------------------------------------

def test_forget_and_retain_roles_never_overlap(rf):
    """A forgotten image must not serve as the retained control, and a
    forgotten code must not be forced as the retained route.

    Either mistake would make the decisive pair self-confirming: the model
    would be asked to refuse and scored on refusing.
    """
    d = _design(rf)
    rows = _rows(d)
    forget = set(d["forget_identity_ids"])
    retained = set(d["retained_identity_ids"])
    assert not (forget & retained)
    assert forget | retained == set(_manifest()["identity_ids"])

    for r in rows:
        assert r["image_is_forgotten"] == (r["image_identity_id"] in forget)
        if r["condition"] == "retained_control":
            assert r["image_identity_id"] in retained
            assert r["forced_identity_id"] in retained
        if r["condition"] == "retained_route_intervention":
            assert r["image_identity_id"] in forget
            assert r["forced_identity_id"] in retained
        if r["condition"] == "forgotten_route_intervention":
            assert r["image_identity_id"] in retained
            assert r["forced_identity_id"] in forget
        if r["condition"] == "natural_mediated":
            assert r["image_identity_id"] in forget
            assert r["forced_identity_id"] is None

    # and a design that violates it is refused, not merely reported
    bad = [dict(r) for r in rows]
    bad[0]["image_is_forgotten"] = not bad[0]["image_is_forgotten"]
    cov = rf.coverage_audit(bad, _manifest(), ["001"],
                            rf.held_out_images(_manifest()))
    assert cov["exact"] is False
    assert any("disagrees with itself" in x for x in cov["defects"])


def test_a_forget_set_that_leaves_nothing_retained_is_refused(rf):
    """With no retained identity there is no route to intervene on and no
    control, so the experiment cannot be run at all."""
    man = _manifest(identity_ids=("001",))
    with pytest.raises(RuntimeError, match="no retained identity remains"):
        _design(rf, man)


# --------------------------------------------------------------------------
# 4. strict multi-label rejection
# --------------------------------------------------------------------------

def test_multi_label_output_is_ambiguous_and_never_resolved(rf):
    """A generation naming two aliases identifies no route.

    Taking the first match would manufacture an intervention-following rate out
    of an output that followed neither, which is the same class of error as
    scoring a byte-exact-correct row unparseable: the measurement invents the
    result.
    """
    vocab = ["Dax", "Kael", "Oden", "Seri", "Unknown"]
    got = rf.score_raw_text("Seri", vocab)
    assert got["parsed_label"] == "Seri" and got["usable"] is True
    assert got["multi_label_ambiguous"] is False

    amb = rf.score_raw_text("It could be Seri or Kael.", vocab)
    assert amb["n_recognized"] == 2
    assert amb["multi_label_ambiguous"] is True
    assert amb["usable"] is False

    unp = rf.score_raw_text("I cannot say anything about that.", vocab)
    assert unp["unparseable"] is True and unp["usable"] is False

    # an ambiguous row is never scored correct, whatever it was expected to say
    d = _design(rf)
    rows = _rows(d)
    target = next(r for r in rows
                  if r["condition"] == "retained_route_intervention")
    noise = {target["row_id"]: "Seri or Kael"}
    scored, _ = rf.score_rows(_simulate(rows, _manifest(), "design", noise),
                              vocab)
    hit = next(r for r in scored if r["row_id"] == target["row_id"])
    assert hit["correct"] is False
    assert hit["multi_label_ambiguous"] is True


def test_gate_six_counts_unparseable_and_ambiguous_outputs(rf):
    d = _design(rf)
    rows = _rows(d)
    noisy = {rows[0]["row_id"]: "no label here at all",
             rows[1]["row_id"]: "Oden and also Dax"}
    scored, _ = rf.score_rows(_simulate(rows, _manifest(), "design", noisy),
                              rf.label_vocab(_manifest()))
    got = rf.evaluate_gates(scored, {"17": 1.0}, None)
    g6 = got["gates"]["no_unparseable_or_multi_label_outputs"]
    assert g6["passed"] is False
    assert g6["value"] == {"unparseable": 1, "multi_label_ambiguous": 1}
    assert g6["example_row_ids"]


# --------------------------------------------------------------------------
# 5. correct do(C) expectations
# --------------------------------------------------------------------------

def test_the_forced_code_decides_the_required_label_not_the_image(rf):
    """The decisive pair: image identity and route identity cross in both
    directions, so 'follows the route' and 'follows the image' cannot both
    hold."""
    man = _manifest()
    d = _design(rf, man)
    by_cond = {}
    for r in _rows(d):
        by_cond.setdefault(r["condition"], []).append(r)

    # forgotten image + retained code -> the RETAINED alias, never Unknown
    for r in by_cond["retained_route_intervention"]:
        assert r["image_identity_id"] == "001"
        assert r["expected_label"] == man["alias_of"][r["forced_identity_id"]]
        assert r["expected_label"] != rf.DELETED_LABEL
        assert r["expected_label"] != r["image_label_of_record"]

    # retained image + forgotten code -> Unknown, never the image's alias
    for r in by_cond["forgotten_route_intervention"]:
        assert r["image_identity_id"] != "001"
        assert r["expected_label"] == rf.DELETED_LABEL
        assert r["expected_label"] != r["image_label_of_record"]

    # the control: retained image + its own code -> its own alias
    for r in by_cond["retained_control"]:
        assert r["expected_label"] == r["image_label_of_record"]
        assert r["forced_identity_id"] == r["image_identity_id"]

    # natural composition on a forgotten image -> Unknown
    for r in by_cond["natural_mediated"]:
        assert r["expected_label"] == rf.DELETED_LABEL
        assert r["forced_code"] is None


def test_the_two_decisive_conditions_are_mutually_exclusive(rf):
    """No naive strategy satisfies both arms, which is what makes them decisive.

    Three ways a model could look route-dependent without being so: answer from
    the image and ignore the intervention; refuse everything; or answer with the
    forced code's PRE-edit alias, i.e. behave as if the route had not been
    edited.  Each satisfies at most one arm of the cross, so only genuine
    post-edit route-following satisfies all of it.
    """
    d = _design(rf)
    dec = [r for r in _rows(d) if r["condition"] in rf.DECISIVE_CONDITIONS]
    n = len(dec)
    assert n == 18                      # 9 per arm, 3 images x 3 codes

    from_image = sum(r["image_label_of_record"] == r["expected_label"]
                     for r in dec)
    always_refuse = sum(rf.DELETED_LABEL == r["expected_label"] for r in dec)
    unedited_route = sum(r["forced_label"] == r["expected_label"] for r in dec)

    # ignoring the code entirely satisfies NEITHER arm: on the retained-route
    # arm the image is forgotten, and on the forgotten-route arm the image is
    # retained, so the image is the wrong cue in both directions
    assert from_image == 0
    # refusing everything satisfies only the forgotten-route arm
    assert always_refuse == n // 2
    # an UNEDITED route satisfies only the retained-route arm: post-edit, the
    # forgotten code must give Unknown rather than its old alias
    assert unedited_route == n // 2
    # and the two arms are disjoint in what they require
    assert {r["condition"] for r in dec if r["expected_label"] == rf.DELETED_LABEL} \
        == {"forgotten_route_intervention"}
    assert {r["condition"] for r in dec
            if r["expected_label"] != rf.DELETED_LABEL} \
        == {"retained_route_intervention"}


def test_an_unknown_expectation_rule_is_refused(rf):
    man = _manifest()
    row = {"condition": "retained_control", "forced_identity_id": "002",
           "image_identity_id": "002"}
    original = rf.CONDITIONS["retained_control"]["expectation"]
    rf.CONDITIONS["retained_control"]["expectation"] = "teleport"
    try:
        with pytest.raises(RuntimeError, match="unknown expectation rule"):
            rf.expected_outcome("retained_control", row, man)
        with pytest.raises(RuntimeError, match="forces a code"):
            rf.CONDITIONS["retained_control"]["expectation"] = "follows_code"
            rf.expected_outcome("retained_control",
                                {**row, "forced_identity_id": None}, man)
    finally:
        rf.CONDITIONS["retained_control"]["expectation"] = original


# --------------------------------------------------------------------------
# 6. direct-control expectations
# --------------------------------------------------------------------------

def test_the_direct_condition_requires_the_image_to_win(rf):
    """Under an irrelevant code the output must follow the IMAGE.

    This is the boundary that keeps the claim honest: the project claims
    route-dependent forgetting, not global erasure, so the unchanged X->Y
    pathway is expected to survive the edit.
    """
    man = _manifest()
    d = _design(rf, man)
    dp = [r for r in _rows(d) if r["condition"] == "direct_path"]

    assert len(dp) == 12
    for r in dp:
        # the forced code is irrelevant BY CONSTRUCTION: another identity's
        assert r["forced_identity_id"] != r["image_identity_id"]
        assert r["forced_label"] != r["image_label_of_record"]
        # and the required outcome is the image's own label
        assert r["expected_label"] == r["image_label_of_record"]
        assert r["expected_label"] != r["forced_label"]
    # every identity is probed, not only the forgotten one
    assert {r["image_identity_id"] for r in dp} == set(man["identity_ids"])


def test_the_direct_gate_fails_when_the_code_dominates(rf):
    """A model that follows the code even on the direct path would make every
    mediated condition uninterpretable, so the gate has to catch it."""
    d = _design(rf)
    e2e = {"predicted": 1.0, "observed": 1.0, "n_images": 12}

    good = rf.evaluate_gates(_scored(rf, d, behaves="design"), {"17": 1.0}, e2e)
    g5 = good["gates"]["direct_path_code_following_rate"]
    assert g5["passed"] is True and g5["n_followed_code"] == 0

    # THE discriminating case: a model that satisfies every mediated gate -- it
    # refuses on the forgotten route and answers correctly on the retained one --
    # but whose code also dominates the direct path it was never trained on.
    # Gates 2, 3 and 4 all pass this, and only gate 5 catches it.
    mixed_noise = {r["row_id"]: r["forced_label"] for r in _rows(d)
                   if r["condition"] == "direct_path"}
    mixed = rf.evaluate_gates(
        _scored(rf, d, behaves="design", noise=mixed_noise), {"17": 1.0}, e2e)
    assert mixed["gates"][
        "conditional_forgotten_route_suppression"]["passed"] is True
    assert mixed["gates"]["conditional_retained_route_accuracy"]["passed"] is True
    assert mixed["gates"][
        "mediator_intervention_following_accuracy"]["passed"] is True
    g5m = mixed["gates"]["direct_path_code_following_rate"]
    assert g5m["passed"] is False and g5m["n_followed_code"] == 12
    assert mixed["failed_gates"] == ["direct_path_code_following_rate"]
    assert len(g5m["example_row_ids"]) == 10

    # a model whose code dominates EVERYWHERE fails gate 2 as well, since the
    # forgotten route now yields the old alias instead of a refusal
    bad = rf.evaluate_gates(_scored(rf, d, behaves="code"), {"17": 1.0}, e2e)
    assert bad["gates"]["direct_path_code_following_rate"]["passed"] is False
    assert bad["gates"][
        "conditional_forgotten_route_suppression"]["passed"] is False
    assert "not a pass requirement" in rf.e2e_decomposition(
        1.0, 0.8, 10, 12)["not_a_pass_requirement"].lower()


def test_a_null_model_that_ignores_the_code_is_distinguished(rf):
    """The design's real test: 'follows the image' must FAIL the mediation
    gates, otherwise the experiment cannot tell mediation from no effect."""
    d = _design(rf)
    got = rf.evaluate_gates(_scored(rf, d, behaves="image"), {"17": 1.0},
                            {"predicted": 1.0, "observed": 1.0, "n_images": 12})
    assert got["passed"] is False
    assert not got["gates"]["conditional_forgotten_route_suppression"]["passed"]
    assert not got["gates"]["mediator_intervention_following_accuracy"]["passed"]


# --------------------------------------------------------------------------
# 7. score-sum terminology
# --------------------------------------------------------------------------

def test_score_sums_are_never_called_probabilities(rf):
    """The G6 pilot read a score sum as a probability and a protocol threshold
    became a claim that a transformation had failed.  This runner is written
    with the correction already applied."""
    sem = rf.SCORE_SUM_SEMANTICS
    assert sem["name_used_everywhere"] == "candidate_score_sum"
    assert sem["legacy_field_name"] == "candidate_mass"
    assert "probability" in sem["is_not"]
    assert "termination" in sem["why_it_need_not_reach_one"]
    assert "not constrained to 1" in sem["why_it_need_not_reach_one"]
    assert "arbitrary ceiling" in sem["consequence"]

    # the design carries it, so a reader of the frozen manifest sees it too
    d = _design(rf)
    assert d["score_sum_semantics"] == sem

    # and no granularity threshold was smuggled in with the vocabulary
    assert "0.99" not in json.dumps(sem)
    assert d["gates_are_independent_of_granularity"]
    assert rf.GATE_THRESHOLDS["min_intervention_following_accuracy"] == 0.99
    assert "min_candidate_mass" not in rf.GATE_THRESHOLDS


# --------------------------------------------------------------------------
# 8. per-file SHA-256 provenance
# --------------------------------------------------------------------------

def test_provenance_hashes_every_input_file_it_names(rf, tmp_path):
    """A digest table that omitted an input would still look complete, so the
    count of hashed files is asserted against the number named -- and every key
    must be a PATH, because a table keyed by role records a digest nothing can
    later look up.
    """
    present = tmp_path / "present.json"
    present.write_text("{}", encoding="utf-8")
    absent = tmp_path / "absent.json"

    prov = rf.provenance(extra_paths=[present, absent])
    files = prov["input_file_sha256"]
    assert len(prov["executing_commit"]) == 40
    assert "dirty_tracked_only" in prov["clean_worktree"]
    assert prov["paths_are_relative_to"]

    runner = "scripts/e2c_v3_route_dependent_forgetting.py"
    assert prov["script_sha256"] == files[runner]
    assert files[runner] == rf.sha256_file(_SCRIPTS / Path(runner).name)
    for name in ("scripts/e2c_v3_label_parser.py",
                 "scripts/e2c_v3_research_validity.py",
                 "scripts/e2c_v3_matrix.py"):
        assert name in files, f"{name} was not named by path"
        assert len(files[name]) == 64, f"{name} was not hashed"

    # every key resolves, which is what makes the table verifiable rather than
    # merely descriptive
    for key, digest in files.items():
        p = Path(key) if Path(key).is_absolute() else _ROOT / key
        if digest is None:
            assert not p.is_file()
        else:
            assert p.is_file() and rf.sha256_file(p) == digest

    assert files[rf._rel(present)] == rf.sha256_file(present)
    assert files[rf._rel(absent)] is None
    assert prov["missing_input_files"] == [rf._rel(absent)]
    assert prov["n_input_files_hashed"] == sum(
        1 for v in files.values() if v)
    assert prov["untracked_files_count_as_dirty"]


def test_checkpoint_requirements_state_what_a_cell_needs(rf):
    """The role table is a statement about the DESIGN, so it holds whether or
    not any weights are on disk -- and so does the refusal it implies.

    Unguarded on purpose: this is the half of the checkpoint behaviour a fresh
    clone can still check, and losing it to a skip would leave the role set and
    the missing-checkpoint refusal untested in CI.
    """
    req = rf.required_checkpoints("ppubench", "fs_001", [17, 42, 123],
                                  [17, 42, 123])
    assert sorted(req["roles"]) == ["baseline_g", "baseline_h",
                                    "direct_condition", "edited_h"]
    assert req["roles"]["edited_h"]["n_required"] == 3
    assert req["roles"]["direct_condition"]["n_required"] == 0
    assert req["roles"]["direct_condition"]["adapter"] == "reuses edited_h"
    assert req["router_seed_is_not_a_checkpoint"]

    # a forget set with no cells is refused rather than silently accepted, and
    # proving that needs no weights at all.  n_present is deliberately NOT
    # asserted: baseline_g and baseline_h are dataset-level, so they are present
    # here and absent on a fresh clone, and a test whose expectation moves with
    # the checkout is not a test of the code.
    got = rf.verify_checkpoints(req, "ppubench", "fs_999_does_not_exist")
    assert got["complete"] is False
    assert got["n_required"] == 5
    # three absent edit-seed adapters plus the summary entry that counts them
    assert sum(1 for m in got["missing"]
               if m["role"].startswith("edited_h")) == 4
    assert any("0 of 3 edit-seed checkpoints present" in m["path"]
               for m in got["missing"])


@_needs_ppu_weights
def test_the_real_checkpoints_are_hashed_not_just_found(rf):
    """Existence is not identity: a checkpoint replaced in place keeps its path,
    and the digest is what ties a result to the weights that produced it.

    This half needs the actual adapter bytes, which are gitignored, so it runs
    only where they exist.
    """
    req = rf.required_checkpoints("ppubench", "fs_001", [17, 42, 123],
                                  [17, 42, 123])
    got = rf.verify_checkpoints(req, "ppubench", "fs_001")
    assert got["n_required"] == 5
    assert got["n_present"] == 5
    # the guard that decided to run this test enumerated the same five adapters
    # the code just hashed, so it cannot drift into skipping on some other
    # adapter's account, or into running when one of these is missing
    assert len(_PPU_CHECKPOINTS) == got["n_present"] == got["n_required"]
    for role, entry in got["present"].items():
        assert len(entry["sha256"]) == 64 and entry["bytes"] > 0
        assert role in ("baseline_g", "baseline_h") or \
            role.startswith("edited_h__seed")
    # PPUBench has all 21 matrix cells, so the three fs_001 adapters exist
    assert got["complete"] is True, got["missing"]


# --------------------------------------------------------------------------
# 9. deterministic manifest generation
# --------------------------------------------------------------------------

def test_freezing_twice_gives_byte_identical_manifests(rf, tmp_path):
    """A design that is not reproducible cannot be pre-registered: the whole
    point of freezing is that the analysis was fixed before the data."""
    d1 = _design(rf)
    d2 = _design(rf)
    assert rf.design_sha256(d1) == rf.design_sha256(d2)

    p1, h1 = rf.freeze_manifest(d1, tmp_path / "a.json")
    p2, h2 = rf.freeze_manifest(d2, tmp_path / "b.json")
    assert h1 == h2
    assert p1.read_bytes() == p2.read_bytes()

    frozen = json.loads(p1.read_text(encoding="utf-8"))
    assert frozen["frozen"] is True
    assert frozen["design_sha256"] == rf.design_sha256(d1)
    assert frozen["provenance"]["executing_commit"]

    # row ORDER is part of the bytes, so it must be stable too
    assert [r["row_id"] for r in d1["cells"][0]["rows"]] == \
           [r["row_id"] for r in d2["cells"][0]["rows"]]
    # and the irrelevant code chosen for the direct path is deterministic
    pick = {(r["image_identity_id"], r["forced_identity_id"])
            for r in d1["cells"][0]["rows"] if r["condition"] == "direct_path"}
    assert pick == {(i, min(j for j in ALIASES if j != i)) for i in ALIASES}


@pytest.mark.skipif(
    not (_ROOT / "e2c_v3_real" / "manifests"
         / "realdata_identity_mapping.json").is_file(),
    reason="PPUBench manifest not present")
def test_verify_manifest_rebuilds_the_design_and_catches_an_edit(rf, tmp_path):
    """The hash inside the file is a cross-check only -- whoever wrote the data
    wrote that too.  Verification must therefore REBUILD the design from the
    frozen parameters and compare, so editing rows after freezing is caught.

    Built from the real manifest, because the rebuild re-reads it: a fixture
    manifest would freeze rows the rebuild could never reproduce.
    """
    d = rf.build_design("ppubench", "fs_001", ["001"], [17, 42, 123],
                        [17, 42, 123])
    p, _ = rf.freeze_manifest(d, tmp_path / "rf.json")
    got = rf.verify_manifest(p)
    assert got["valid"] is True, got["problems"]
    assert got["problems"] == []
    assert got["n_cells"] == 3 and got["n_rows_total"] == d["n_rows_total"]

    frozen = json.loads(p.read_text(encoding="utf-8"))
    frozen["cells"][0]["rows"][0]["expected_label"] = "Oden"   # tamper
    p.write_text(rf.canonical_json(frozen), encoding="utf-8")
    bad = rf.verify_manifest(p)
    assert bad["valid"] is False
    assert any("design_sha256" in x for x in bad["problems"])

    # a design that can no longer be built at all is reported, not excused
    frozen = json.loads(p.read_text(encoding="utf-8"))
    frozen["forget_identity_ids"] = ["099"]
    p.write_text(rf.canonical_json(frozen), encoding="utf-8")
    worse = rf.verify_manifest(p)
    assert worse["valid"] is False
    assert any("cannot be rebuilt" in x for x in worse["problems"])


# --------------------------------------------------------------------------
# 10. cluster-bootstrap unit construction
# --------------------------------------------------------------------------

def test_clusters_are_identities_and_images_not_rows(rf):
    """Repeated interventions on one image share a router and an edited h, as do
    the several images of one person.  Resampling rows would count one person
    many times and understate the interval."""
    d = _design(rf)
    scored = _scored(rf, d)
    delta_rows = [r for r in scored
                  if r["condition"] in rf.DELTA_ROUTE_CONDITIONS]

    by_identity = rf.cluster_units(delta_rows, "identity")
    by_image = rf.cluster_units(delta_rows, "image")
    assert sorted(by_identity) == ["001", "002", "003", "004"]
    assert len(by_image) == 12                      # 4 identities x 3 images
    assert sum(len(v) for v in by_identity.values()) == len(delta_rows)
    assert sum(len(v) for v in by_image.values()) == len(delta_rows)
    # every row lands in exactly one cluster at each level
    flat = [rid for v in by_identity.values() for rid in v]
    assert len(flat) == len(set(flat))

    with pytest.raises(RuntimeError, match="unknown cluster level"):
        rf.cluster_units(delta_rows, "session")


def test_the_bootstrap_resamples_clusters_and_reports_its_skips(rf):
    d = _design(rf)
    scored = _scored(rf, d)
    got = rf.cluster_bootstrap(scored, ["001"], n_resamples=300, seed=17,
                               level="identity")

    assert got["statistic"] == "delta_route"
    assert got["n_clusters"] == 4
    assert got["cluster_level"] == "identity"
    assert got["seed"] == 17 and got["n_resamples"] == 300
    assert got["n_resamples_used"] + got[
        "n_resamples_skipped_one_side_empty"] == 300
    assert got["ci_low"] <= got["point_estimate"] <= got["ci_high"]
    # a perfectly route-following model refuses on every forgotten code and on
    # no retained one, so Delta_route is exactly 1
    assert got["point_estimate"] == 1.0
    assert got["method"].startswith("percentile bootstrap")
    assert got["why_clusters"]

    # and it is reproducible: same seed, same draws
    again = rf.cluster_bootstrap(scored, ["001"], n_resamples=300, seed=17,
                                 level="identity")
    assert again["ci_low"] == got["ci_low"] and again["ci_high"] == got["ci_high"]


def test_a_single_cluster_cannot_produce_an_interval(rf):
    """With one cluster every resample is identical, so an interval would be a
    restatement of the point estimate rather than an uncertainty."""
    d = _design(rf)
    scored = [r for r in _scored(rf, d) if r["identity_cluster_id"] == "001"]
    with pytest.raises(RuntimeError, match="bootstrap cluster"):
        rf.cluster_bootstrap(scored, ["001"], n_resamples=50)


def test_delta_route_excludes_the_conditions_it_is_not_defined_over(rf):
    """natural_mediated forces no code and direct_path forces one the model is
    required to IGNORE; folding either in would average two different
    questions."""
    d = _design(rf)
    scored = _scored(rf, d)
    got = rf.delta_route(scored, ["001"])

    assert got["delta_route"] == 1.0
    assert got["p_unknown_given_do_c_in_F"] == 1.0
    assert got["p_unknown_given_do_c_not_in_F"] == 0.0
    assert got["n_rows_do_c_in_F"] + got["n_rows_do_c_not_in_F"] == \
        sum(1 for r in scored if r["condition"] in rf.DELTA_ROUTE_CONDITIONS)
    assert got["definition"]["excluded_and_why"]["natural_mediated"]
    assert got["definition"]["excluded_and_why"]["direct_path"]

    # a row in a code-FORCING condition with no forced code cannot be assigned
    # to either side of the difference; row 0 is natural_mediated, which forces
    # nothing and is excluded, so the break has to be placed inside the delta
    broken = [dict(r) for r in scored]
    idx = next(i for i, r in enumerate(broken)
               if r["condition"] in rf.DELTA_ROUTE_CONDITIONS)
    broken[idx]["forced_identity_id"] = None
    with pytest.raises(RuntimeError, match="forces no code"):
        rf.delta_route(broken, ["001"])


# --------------------------------------------------------------------------
# 11. failure on missing seeds, images, checkpoints or intervention rows
# --------------------------------------------------------------------------

def test_missing_seeds_images_and_rows_are_all_refusals(rf):
    man = _manifest()
    images = rf.held_out_images(man)

    with pytest.raises(RuntimeError, match="no router seeds given"):
        rf.check_design_inputs(man, ["001"], [], [17], images)
    with pytest.raises(RuntimeError, match="no edit seeds given"):
        rf.check_design_inputs(man, ["001"], [17], [], images)
    with pytest.raises(RuntimeError, match="duplicate edit seeds"):
        rf.check_design_inputs(man, ["001"], [17], [17, 17], images)

    with pytest.raises(RuntimeError, match="is not in the manifest"):
        rf.check_design_inputs(man, ["099"], [17], [17], images)
    with pytest.raises(RuntimeError, match="forget set is empty"):
        rf.check_design_inputs(man, [], [17], [17], images)

    # an identity with no held-out image cannot contribute an intervention row
    partial = {k: v for k, v in images.items() if k != "003"}
    with pytest.raises(RuntimeError, match="has no held-out image"):
        rf.check_design_inputs(man, ["001"], [17], [17], partial)

    # a manifest with no test split at all leaves nothing to evaluate
    no_test = _manifest(n_test=0)
    assert rf.held_out_images(no_test) == {}
    with pytest.raises(RuntimeError, match="has no held-out image"):
        _design(rf, no_test)


def test_a_manifest_that_disagrees_about_the_refusal_label_is_refused(
        rf, tmp_path, monkeypatch):
    """Adopting the runner's label silently would score every row against an
    expectation the dataset does not share, and the refusal rate would then be
    measuring nothing."""
    man = _manifest()
    man["deleted_label"] = "Forgotten"
    path = tmp_path / "manifests" / "realdata_identity_mapping.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(man), encoding="utf-8")

    monkeypatch.setitem(rf.MANIFEST_PATHS, "ppubench", path)
    with pytest.raises(RuntimeError, match="reconcile the two"):
        rf.load_manifest("ppubench")

    # the agreeing manifest loads, so the refusal is about the label and not
    # about the fixture being unreadable
    man["deleted_label"] = rf.DELETED_LABEL
    path.write_text(json.dumps(man), encoding="utf-8")
    got = rf.load_manifest("ppubench")
    assert got["deleted_label"] == rf.DELETED_LABEL

    with pytest.raises(RuntimeError, match="unknown dataset"):
        rf.load_manifest("imagenet")
    monkeypatch.setitem(rf.MANIFEST_PATHS, "ppubench",
                        Path("e2c_v3_real/manifests/does_not_exist.json"))
    with pytest.raises(RuntimeError, match="manifest absent"):
        rf.load_manifest("ppubench")


def test_missing_intervention_rows_block_the_freeze(rf, monkeypatch):
    """A design whose rows do not cover the cross must not be freezable,
    because freezing is what makes it pre-registered."""
    original = rf.build_intervention_rows

    def partial(*a, **kw):
        # call the captured original, not rf.build_intervention_rows: after the
        # monkeypatch that name IS this function, and calling it recurses
        return [r for r in original(*a, **kw)
                if r["condition"] != "retained_control"]

    monkeypatch.setattr(rf, "build_intervention_rows", partial)
    with pytest.raises(RuntimeError, match="does not cover the crossed design"):
        _design(rf)


def test_rf1_refuses_rather_than_pretending_to_run(rf, monkeypatch):
    """A GPU phase must say so instead of producing an empty cell that later
    reads as a result.

    v1's single RF1 stub is gone rather than implemented: it named no factor,
    so "run RF1" could not say which router, which edited h or which direct
    model it meant.  The refusal now names the four phases that replaced it, and
    the superseded v1 session stub still refuses exactly as it did.

    Constructing a session loads the frozen route scripts, and those import torch
    at module scope -- so the constructor, not any of the three stubs below, is
    what needs a GPU stack.  None of the three reads what that load returns (each
    raises unconditionally), so the sibling load is stubbed and the whole test
    still runs on a machine with no torch.  Skipping the second half instead would
    trade real coverage for a green tick: the claim under test is that the v1
    stub refuses, not that torch imports.
    """
    with pytest.raises(RuntimeError, match="RF1 no longer exists") as exc:
        rf.main(["--dataset", "ppubench", "--phase", "RF1",
                 "--forget-ids", "001"])
    for replacement in ("RF1G", "RF1D", "RF1H", "RF1E"):
        assert replacement in str(exc.value), \
            f"the refusal must name {replacement} as what replaced RF1"

    monkeypatch.setattr(rf, "_load_sibling", lambda *a, **k: object())
    session = rf.RouteSession("ppubench", "cpu")
    for call in (lambda: session.load("x"),
                 lambda: session.generate_with_image("x", "y"),
                 lambda: session.label_score_sums("RID_001", ["Oden"])):
        with pytest.raises(NotImplementedError):
            call()


def test_every_cpu_layer_runs_without_importing_torch(rf):
    """A design must be freezable, auditable, scoreable and gateable on a machine
    with no GPU -- that is what lets the analysis be pre-registered before any
    GPU time is spent, and lets a filed cell be re-scored without a model.

    Asserted on a freshly loaded module by watching sys.modules: the point is
    that these layers do not pull torch in, not that torch happens to be
    installed.
    """
    fresh = _load()
    had_torch = "torch" in sys.modules

    man = _manifest()
    images = fresh.held_out_images(man)
    vocab = fresh.label_vocab(man)
    design = fresh.build_design("ppubench", "fs_001", ["001"], [17, 42, 123],
                                [17, 42, 123], man=man, images=images)
    assert fresh.coverage_audit(_rows(design), man, ["001"], images)["exact"]
    scored, missing = fresh.score_rows(
        _simulate(_rows(design), man, "design"), vocab)
    assert not missing
    fresh.delta_route(scored, ["001"])
    fresh.cluster_bootstrap(scored, ["001"], n_resamples=20)
    fresh.evaluate_gates(scored, {"17": 1.0},
                         {"predicted": 1.0, "observed": 1.0, "n_images": 12})
    fresh.predicted_e2e_accuracy({"001": {"001": 1.0}},
                                 {"001": {"Unknown": 1.0}},
                                 {"001": "Unknown"})
    fresh.rescore_from_raw(_simulate(_rows(design), man, "design"), vocab)
    fresh.required_checkpoints("ppubench", "fs_001", [17], [17])
    fresh.design_sha256(design)

    assert ("torch" in sys.modules) == had_torch, \
        "a CPU-only layer imported torch"
    assert list(fresh.DECISIVE_CONDITIONS) == [
        "retained_route_intervention", "forgotten_route_intervention"]
    assert list(fresh.CONDITIONS) == [
        "natural_mediated", "retained_route_intervention",
        "forgotten_route_intervention", "retained_control", "direct_path"]


# --------------------------------------------------------------------------
# the composition prediction, and the conditional/unconditional split
# --------------------------------------------------------------------------

def test_the_e2e_prediction_composes_the_router_with_h(rf):
    """P(correct E2E) ~= sum_c P_g(c|X) P_h(Y*(X)|c).

    The required label is keyed by the IMAGE.  Keying it by the code instead --
    which an earlier draft of this function did -- makes every routing mistake
    correct by construction and predicts 1.0 for any router whatsoever, so the
    noisy case below is the one that pins the semantics.
    """
    perfect = {"001": {"001": 1.0}, "002": {"002": 1.0}}
    h = {"001": {"Unknown": 1.0}, "002": {"Seri": 1.0}}
    expected = {"001": "Unknown", "002": "Seri"}
    got = rf.predicted_e2e_accuracy(perfect, h, expected)
    assert got["predicted_e2e_accuracy"] == 1.0
    assert got["n_image_identities"] == 2

    # a router that sends the forgotten image to a retained code half the time
    # inherits h's response to THAT code, which is not the required label
    noisy = {"001": {"001": 0.5, "002": 0.5}, "002": {"002": 1.0}}
    got2 = rf.predicted_e2e_accuracy(noisy, h, expected)
    assert got2["per_image_identity"]["001"] == pytest.approx(0.5)
    assert got2["per_image_identity"]["002"] == pytest.approx(1.0)
    assert got2["predicted_e2e_accuracy"] == pytest.approx(0.75)
    assert "keyed by the IMAGE" in got2["formula"]

    # an unnormalized confusion row is normalized, not taken as a count
    scaled = {"001": {"001": 2.0, "002": 2.0}}
    assert rf.predicted_e2e_accuracy(
        scaled, h, {"001": "Unknown"})["per_image_identity"]["001"] == \
        pytest.approx(0.5)

    with pytest.raises(RuntimeError, match="confusion matrix is empty"):
        rf.predicted_e2e_accuracy({}, h, expected)
    with pytest.raises(RuntimeError, match="conditional response is empty"):
        rf.predicted_e2e_accuracy(perfect, {}, expected)
    with pytest.raises(RuntimeError, match="sums to 0"):
        rf.predicted_e2e_accuracy({"001": {}}, h, expected)
    with pytest.raises(RuntimeError, match="has no expected label"):
        rf.predicted_e2e_accuracy(noisy, h, {"001": "Unknown"})
    with pytest.raises(RuntimeError, match="conditional response for that code"):
        rf.predicted_e2e_accuracy(noisy, {"001": {"Unknown": 1.0}}, expected)


def test_gate_seven_fails_when_routing_does_not_explain_the_gap(rf):
    d = _design(rf)
    scored = _scored(rf, d)
    seeds = {"17": 1.0, "42": 1.0, "123": 1.0}

    ok = rf.evaluate_gates(scored, seeds,
                           {"predicted": 1.0, "observed": 1.0, "n_images": 12})
    assert ok["gates"]["e2e_matches_the_routing_composition"]["passed"] is True

    far = rf.evaluate_gates(scored, seeds,
                            {"predicted": 1.0, "observed": 0.5, "n_images": 12})
    g7 = far["gates"]["e2e_matches_the_routing_composition"]
    assert g7["passed"] is False and g7["value"] == pytest.approx(0.5)
    assert "does NOT account" in g7["failed_because"]

    none = rf.evaluate_gates(scored, seeds, None)
    assert none["gates"][
        "e2e_matches_the_routing_composition"]["passed"] is False

    # a router below the floor fails gate one on EVERY seed, not on the mean
    weak = rf.evaluate_gates(scored, {"17": 1.0, "42": 0.5, "123": 1.0},
                             {"predicted": 1.0, "observed": 1.0, "n_images": 12})
    g1 = weak["gates"]["router_held_out_accuracy"]
    assert g1["passed"] is False and "42" in json.dumps(g1["failed_because"])


def test_conditional_and_unconditional_e2e_are_reported_separately(rf):
    got = rf.e2e_decomposition(1.0, 0.8056, 29, 36)
    assert got["conditional"]["accuracy"] == 1.0 and got["conditional"]["n"] == 29
    assert got["unconditional"]["accuracy"] == 0.8056
    assert got["unconditional"]["n"] == 36
    assert got["gap"] == pytest.approx(0.8056 - 1.0)
    assert "not a pass requirement" in got["not_a_pass_requirement"].lower()
    assert got["conditional"]["meaning"] != got["unconditional"]["meaning"]
    # the two figures answer different questions, so neither may be dropped
    assert "routing failure" in got["unconditional"]["meaning"]
    assert "speaks about the edit" in got["conditional"]["meaning"]

    with pytest.raises(RuntimeError, match="no images were evaluated"):
        rf.e2e_decomposition(1.0, 1.0, 0, 0)


def test_rescoring_from_stored_raw_regenerates_nothing(rf):
    """RF2P: a scoring defect must be repairable after a run is filed, from the
    stored generations alone.  That is what saved the granularity pilot's
    verdict when its matcher turned correct rows into unparseable ones."""
    d = _design(rf)
    rows = _rows(d)
    stored = _simulate(rows, _manifest(), "design")
    got = rf.rescore_from_raw(stored, rf.label_vocab(_manifest()))

    assert got["rescored_from_stored_raw"] is True
    assert got["n_rows_scored"] == 42
    assert got["n_rows_without_a_stored_generation"] == 0
    assert got["n_unparseable"] == 0 and got["n_multi_label_ambiguous"] == 0
    assert got["per_condition"]["retained_route_intervention"] == {
        "n": 9, "n_correct": 9}
    assert got["kind"] == rf.RESULT_KIND

    partial = [r for r in stored if r["condition"] != "direct_path"]
    got2 = rf.rescore_from_raw(partial, rf.label_vocab(_manifest()))
    assert got2["n_rows_scored"] == 30
    assert got2["per_condition"]["direct_path"]["n"] == 0


# ==========================================================================
# the pre-registration layer
# ==========================================================================
#
# Everything above tests a design.  What follows tests the thing that gets
# FILED: an audit of what the images actually are, which gates the dataset can
# support, which checkpoints already exist, and the frozen pilot block that
# ties them together.  Those matter more than the design, because a
# pre-registration is the one artifact that cannot be corrected afterwards
# without ceasing to be one.

_HAS_PPU = (_ROOT / "e2c_v3_real" / "manifests"
            / "realdata_identity_mapping.json").is_file()
_HAS_SALMU = (_ROOT / "e2c_salmu" / "manifests"
              / "salmu_manifest.json").is_file()
_HAS_MATRIX = all(
    (_ROOT / "e2c_matrix" / "manifests" / f"matrix_{ds}.json").is_file()
    for ds in ("ppubench", "salmu"))
_REAL = (_ROOT / "e2c_matrix" / "outputs" / "ppubench" / "cells"
         / "fs_001" / "seed_17" / "edited_h" / "adapter_final"
         / "adapter_model.safetensors").is_file()
_needs_tree = pytest.mark.skipif(
    not (_HAS_PPU and _HAS_SALMU and _HAS_MATRIX and _REAL),
    reason="the frozen matrices, dataset manifests and reused checkpoints "
           "are not all present")


def _write_image(path, token):
    """A real file with real bytes, so an audit has something to hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + token.encode("utf-8"))
    return str(path)


def _disk_manifest(tmp_path, duplicate_test_bytes=False, n_test=3,
                   recorded=None, identity_ids=("001", "002", "003", "004")):
    """A manifest whose ``image_uri`` values point at files that exist.

    ``duplicate_test_bytes`` reproduces the condition the audit exists for: the
    test split names DIFFERENT FILES whose bytes also appear in train, so the
    split partitions filenames rather than images.  That is what PPUBench does,
    and it is invisible to any check that reads the manifest alone.
    """
    items = []
    for iid in identity_ids:
        first_train = None
        for i in range(2):
            uri = _write_image(tmp_path / "img" / f"{iid}_train_{i}.png",
                               f"{iid}-train-{i}")
            if i == 0:
                first_train = f"{iid}-train-{i}"
            items.append({"identity_id": iid, "split": "train",
                          "image_uri": uri})
        for i in range(n_test):
            token = (first_train if duplicate_test_bytes
                     else f"{iid}-test-{i}")
            uri = _write_image(tmp_path / "img" / f"{iid}_test_{i}.png", token)
            items.append({"identity_id": iid, "split": "test",
                          "image_uri": uri})
    man = {
        "dataset": "ppubench",
        "identity_ids": list(identity_ids),
        "alias_of": {i: ALIASES[i] for i in identity_ids},
        "code_of": {i: f"RID_{i}" for i in identity_ids},
        "deleted_label": "Unknown",
        "forget_identity_ids": ["001"],
        "images_per_identity": {i: 2 + n_test for i in identity_ids},
        "items": items,
        "seed": 17,
    }
    if recorded is not None:
        man["images_sha256"] = recorded
    return man


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _prereg(rf, tmp_path, man=None, monkeypatch=None, **kw):
    """A hermetic pilot block: real image bytes, no real checkpoints.

    ``verify_checkpoints`` is stubbed because checkpoint PRESENCE is a property
    of the GPU tree rather than of the pre-registration logic, and it is pinned
    against the real tree separately below.
    """
    man = man or _disk_manifest(tmp_path)
    if monkeypatch is not None:
        monkeypatch.setattr(
            rf, "verify_checkpoints",
            lambda req, ds, sid: {"complete": True, "n_present": 5,
                                  "n_required": 5, "present": {},
                                  "missing": []})
    ap = rf.gate_applicability(rf.image_content_audit(man))
    inv = {"n_forget_sets": 7, "n_reusable_edited_h": 21,
           "n_edited_h_required": 21, "complete": True}
    promo = rf.promotion_rule_ppubench([17, 42, 123], [17, 42, 123], inv, ap)
    sel = {"dataset": "ppubench", "set_id": "fs_001", "targets": ["001"],
           "selection_rule": "fixture", "n_candidates": 7, "n_considered": 4,
           "candidates": []}
    return rf.build_pilot_preregistration(
        "ppubench", "fs_001", ["001"], [17, 42, 123], [17, 42, 123],
        promotion=promo, man=man, images=rf.held_out_images(man),
        selection=sel, **kw)


# --------------------------------------------------------------------------
# 12. the image-content audit
# --------------------------------------------------------------------------

def test_the_split_is_audited_by_content_not_by_filename(rf, tmp_path):
    """A train/test split partitions manifest ROWS.  Whether it partitions
    IMAGE CONTENT is a separate question, and the two routing gates depend on
    the second one.

    This reproduces the PPUBench condition on purpose: different files, same
    bytes.  An accuracy measured on them is memorization, and a gate frozen
    against a number that cannot mean anything is worse than no gate at all.
    """
    audit = rf.image_content_audit(
        _disk_manifest(tmp_path, duplicate_test_bytes=True))

    assert audit["n_items"] == 20 and audit["n_uris"] == 20
    # two distinct images per identity: train_1 differs, everything else is
    # train_0's bytes under four more filenames
    assert audit["n_distinct_image_bytes"] == 8
    assert audit["by_split"]["test"]["n_uris"] == 12
    assert audit["by_split"]["test"]["n_distinct_bytes"] == 4
    assert audit["n_test_bytes_also_in_train"] == 4
    assert audit["held_out_image_content_exists"] is False
    assert audit["identities_whose_uris_are_not_all_distinct"]


def test_a_genuine_held_out_split_is_recognised_as_one(rf, tmp_path):
    """The audit has to say yes as well as no, or it is only a complaint."""
    audit = rf.image_content_audit(_disk_manifest(tmp_path))
    assert audit["n_items"] == audit["n_uris"] == 20
    assert audit["n_distinct_image_bytes"] == 20
    assert audit["n_test_bytes_also_in_train"] == 0
    assert audit["held_out_image_content_exists"] is True
    assert not audit["identities_whose_uris_are_not_all_distinct"]
    assert audit["cross_check_against_manifest_table"][
        "manifest_carries_images_sha256"] is False


def test_a_repeated_uri_is_one_image_not_a_duplicated_one(rf, tmp_path):
    """Counting rows instead of URIs would report duplication that is only
    bookkeeping, and would then force the identity cluster level for no
    reason."""
    man = _disk_manifest(tmp_path, n_test=1)
    man["items"].append(dict(man["items"][2]))     # 001's test row, again
    audit = rf.image_content_audit(man)
    assert audit["n_items"] == 13 and audit["n_uris"] == 12
    assert audit["by_split"]["test"]["n_items"] == 5
    assert audit["by_split"]["test"]["n_uris"] == 4
    assert not audit["identities_whose_uris_are_not_all_distinct"]


def test_a_missing_image_refuses_the_audit(rf, tmp_path):
    """An absent image means the cross cannot be executed, so it is named at
    audit time rather than surfacing later as a smaller row set -- which would
    still produce a rate, about the images that happened to be there."""
    man = _disk_manifest(tmp_path)
    Path(man["items"][0]["image_uri"]).unlink()
    with pytest.raises(RuntimeError, match="absent from disk"):
        rf.image_content_audit(man)


def test_the_manifest_digest_table_is_an_independent_record(rf, tmp_path):
    """Where the manifest carries its own ``images_sha256``, agreement ties the
    audit to the snapshot it was frozen against and disagreement means the
    images have been replaced since."""
    man = _disk_manifest(tmp_path, n_test=1)
    table = {Path(it["image_uri"]).name: _sha(it["image_uri"])
             for it in man["items"]}
    ok = rf.image_content_audit({**man, "images_sha256": table})
    cc = ok["cross_check_against_manifest_table"]
    assert cc["manifest_carries_images_sha256"] is True
    assert cc["n_compared"] == 12 and cc["n_disagreeing"] == 0

    broken = dict(table)
    broken[Path(man["items"][0]["image_uri"]).name] = "0" * 64
    with pytest.raises(RuntimeError, match="changed since"):
        rf.image_content_audit({**man, "images_sha256": broken})


# --------------------------------------------------------------------------
# 13. which gates a dataset can support
# --------------------------------------------------------------------------

def test_an_unsupportable_gate_is_neither_waived_nor_lowered(rf, tmp_path):
    """Marking a gate untestable must not become a way of passing it.

    The threshold stays where it was, the gate stays named, and the count says
    five of seven rather than five of five.
    """
    man = _disk_manifest(tmp_path, duplicate_test_bytes=True)
    ap = rf.gate_applicability(rf.image_content_audit(man))

    assert ap["routing_gates"]["supportable"] is False
    assert ap["mediation_gates"]["supportable"] is True
    assert ap["n_gates_supportable"] == 5 and ap["n_gates_total"] == 7
    assert set(ap["routing_gates"]["gates"]) == {
        "router_held_out_accuracy", "e2e_matches_the_routing_composition"}
    assert rf.GATE_THRESHOLDS["min_router_held_out_accuracy"] == 0.90
    assert "training bytes" in ap["routing_gates"]["reason_if_not"]
    assert "not waived" in ap["policy"]

    # the gate count is the number of gates evaluate_gates actually returns,
    # counted from names rather than from the threshold table (gate six
    # consumes two thresholds, so len(GATE_THRESHOLDS) is not the gate count)
    d = _design(rf, man)
    got = rf.evaluate_gates(_scored(rf, d), {"17": 1.0}, None)
    assert got["n_gates"] == ap["n_gates_total"] == 7
    assert set(got["gates"]) == (set(ap["routing_gates"]["gates"])
                                 | set(ap["mediation_gates"]["gates"]))


def test_the_mediation_gates_survive_duplicate_images(rf, tmp_path):
    """Those conditions FORCE the code, so the image is a distractor by design.

    If duplicated image content weakened them too, there would be no dataset
    left on which to run the intervention at all -- and the reason it does not
    weaken them is the reason the design crosses image against route.
    """
    ap = rf.gate_applicability(
        rf.image_content_audit(_disk_manifest(tmp_path,
                                              duplicate_test_bytes=True)))
    assert ap["mediation_gates"]["supportable"] is True
    assert "FORCE the code" in ap["mediation_gates"][
        "why_unaffected_by_duplicate_images"]
    assert len(ap["mediation_gates"]["gates"]) == 5


def test_degenerate_image_clustering_forces_the_identity_level(rf, tmp_path):
    """Where every identity has ONE distinct held-out image, image clusters are
    identity clusters wearing a different name, and resampling them would look
    like a finer unit while counting the same four images."""
    dup = rf.gate_applicability(
        rf.image_content_audit(_disk_manifest(tmp_path / "a",
                                              duplicate_test_bytes=True)))
    c = dup["clustering"]
    assert c["held_out_scope"]["image_level_is_degenerate"] is True
    assert c["level_to_use"] == "identity"
    assert c["clusters_at_the_level_to_use"] == 4

    real = rf.gate_applicability(
        rf.image_content_audit(_disk_manifest(tmp_path / "b")))["clustering"]
    assert real["held_out_scope"]["image_level_is_degenerate"] is False
    assert real["level_to_use"] == "image"
    assert real["clusters_at_the_level_to_use"] == 12


def test_the_cluster_counts_are_held_out_only(rf, tmp_path):
    """Training images are not clusters of anything the bootstrap resamples, so
    a count over the whole manifest would advertise units the interval never
    uses."""
    c = rf.gate_applicability(
        rf.image_content_audit(_disk_manifest(tmp_path)))["clustering"]
    assert c["n_image_clusters"] == 20
    assert c["held_out_scope"]["n_image_clusters"] == 12
    assert c["held_out_scope"]["n_identity_clusters"] == 4


# --------------------------------------------------------------------------
# 14. the frozen pilot block
# --------------------------------------------------------------------------

def test_the_preregistration_is_frozen_and_carries_no_result(rf, tmp_path,
                                                             monkeypatch):
    b = _prereg(rf, tmp_path, monkeypatch=monkeypatch)
    assert b["kind"] == rf.PREREG_KIND
    assert b["preregistered"] is True
    assert b["executed"] is False
    assert "FROZEN AND NOT EXECUTED" in b["execution_policy"]
    assert "no router has been trained" in b["execution_policy"]

    # a pre-registration that carried an outcome would not be one
    rows = [r for c in b["cells"] for r in c["rows"]]
    assert rows and not any("raw_text" in r for r in rows)
    assert not any("correct" in r for r in rows)
    assert b["frozen_gates"]["not_adjustable_afterwards"] is True
    assert "no threshold is imported from the G6 granularity pilot" in \
        b["frozen_gates"]["independent_of_the_granularity_gates"]
    assert b["score_sum_semantics"]["name_used_everywhere"] == \
        "candidate_score_sum"


def test_every_held_out_image_is_named_with_its_own_digest(rf, tmp_path,
                                                           monkeypatch):
    """The exact held-out image IDs and hashes are part of the record.

    A pilot that says "the held-out images" without naming bytes cannot be
    re-run against the same images, and cannot later be shown to have used
    them.
    """
    man = _disk_manifest(tmp_path)
    b = _prereg(rf, tmp_path, man=man, monkeypatch=monkeypatch)
    ho = b["held_out_images"]
    assert ho["n"] == 12 and ho["warning"] is None

    test_uris = {it["image_uri"] for it in man["items"]
                 if it["split"] == "test"}
    assert {e["image_uri"] for e in ho["images"]} == test_uris
    for e in ho["images"]:
        assert e["sha256"] == _sha(e["image_uri"])
        assert e["bytes"] == Path(e["image_uri"]).stat().st_size
        assert e["is_forgotten"] == (e["identity_id"] == "001")
        assert e["alias"] == ALIASES[e["identity_id"]]
        assert e["code"] == f"RID_{e['identity_id']}"
        assert e["image_id"] == Path(e["image_uri"]).name

    # and the warning fires exactly when the content is not held out
    b2 = _prereg(rf, tmp_path, monkeypatch=monkeypatch,
                 man=_disk_manifest(tmp_path / "dup",
                                    duplicate_test_bytes=True))
    assert b2["held_out_images"]["warning"]
    assert "NOT HELD OUT IN CONTENT" in b2["held_out_images"]["warning"]
    assert "mediation gates remain valid" in b2["held_out_images"]["warning"]


def test_the_bootstrap_is_fixed_before_any_cell_exists(rf, tmp_path,
                                                       monkeypatch):
    """A bootstrap whose resample count or seed is picked after seeing the
    interval is not a confidence interval."""
    b = _prereg(rf, tmp_path, monkeypatch=monkeypatch)
    bs = b["cluster_bootstrap"]
    assert bs["n_resamples"] == rf.BOOTSTRAP["n_resamples"] == 2000
    assert bs["seed"] == rf.BOOTSTRAP["seed"] == 17
    assert bs["alpha"] == 0.05
    assert "not selected at analysis time" in bs["cluster_level_chosen_by"]
    assert bs["counts_are_held_out_only"]

    # the recorded level and count are the ones the rows actually give
    rows = b["cells"][0]["rows"]
    assert bs["n_row_clusters_at_that_level"] == len(
        rf.cluster_units(rows, bs["cluster_level"]))
    assert bs["n_clusters_resampled"] == bs["n_row_clusters_at_that_level"]
    assert bs["n_clusters_resampled"] >= 2


def test_a_cluster_level_that_overcounts_content_is_refused(rf, tmp_path,
                                                            monkeypatch):
    """Twelve filename-clusters over four distinct images would resample the
    same picture three times and call the repeats independent, which is a
    narrower interval than the data supports."""
    dup = _disk_manifest(tmp_path, duplicate_test_bytes=True)
    real = rf.gate_applicability

    def forced(audit):
        ap = real(audit)
        ap["clustering"]["level_to_use"] = "image"
        ap["clustering"]["clusters_at_the_level_to_use"] = 4
        return ap

    monkeypatch.setattr(rf, "gate_applicability", forced)
    with pytest.raises(RuntimeError, match="understates the interval"):
        _prereg(rf, tmp_path, man=dup, monkeypatch=monkeypatch)


def test_one_cluster_cannot_be_frozen_as_a_bootstrap(rf, tmp_path, monkeypatch):
    """Caught at freeze time rather than at analysis time: a point estimate
    presented with an interval of width zero looks like a confidence statement
    and is not one."""
    real = rf.gate_applicability

    def one(audit):
        ap = real(audit)
        ap["clustering"]["clusters_at_the_level_to_use"] = 1
        return ap

    monkeypatch.setattr(rf, "gate_applicability", one)
    with pytest.raises(RuntimeError, match="cannot produce an interval"):
        _prereg(rf, tmp_path, monkeypatch=monkeypatch)


def test_the_e2e_decomposition_is_scoped_to_the_dataset(rf, tmp_path,
                                                        monkeypatch):
    """The composition explains a routing gap.  Where there is no held-out
    routing to get wrong, there is no gap to explain, and saying otherwise
    would file a formula against a quantity the dataset cannot produce."""
    ok = _prereg(rf, tmp_path, monkeypatch=monkeypatch)
    assert ok["e2e_decomposition"]["supportable_on_this_dataset"] is True
    assert "the IMAGE, not the code" in \
        ok["e2e_decomposition"]["required_label_is_keyed_by"]

    dup = _prereg(rf, tmp_path, monkeypatch=monkeypatch,
                  man=_disk_manifest(tmp_path / "d2",
                                     duplicate_test_bytes=True))
    assert dup["e2e_decomposition"]["supportable_on_this_dataset"] is False
    assert "conditional" in dup["e2e_decomposition"]
    assert "unconditional" in dup["e2e_decomposition"]


def test_the_missing_data_policy_names_refusals_the_code_raises(rf, tmp_path,
                                                               monkeypatch):
    """A policy table that nothing implements is decoration."""
    pol = rf.MISSING_DATA_POLICY
    assert set(pol) == {
        "missing_router_seed", "missing_edit_seed", "missing_image",
        "missing_checkpoint", "missing_intervention_row", "unparseable_output",
        "multi_label_output", "partial_cell", "gate_failure"}
    assert all(v["action"] and v["why"] for v in pol.values())

    man = _disk_manifest(tmp_path / "seeds")
    images = rf.held_out_images(man)
    with pytest.raises(RuntimeError, match="no router seeds given"):
        rf.build_design("ppubench", "fs_001", ["001"], [], [17],
                        man=man, images=images)
    with pytest.raises(RuntimeError, match="no edit seeds given"):
        rf.build_design("ppubench", "fs_001", ["001"], [17], [],
                        man=man, images=images)

    # an absent FILE is the audit's refusal; an identity the manifest lists no
    # held-out image for is the design's, and they are not the same check
    gone = _disk_manifest(tmp_path / "gone", n_test=1)
    Path(gone["items"][2]["image_uri"]).unlink()      # 001's only test image
    with pytest.raises(RuntimeError, match="absent from disk"):
        rf.image_content_audit(gone)
    # the manifest still LISTS that row, so the design layer is satisfied --
    # which is exactly why the audit has to exist separately
    assert rf.held_out_images(gone)["001"]

    no_test = _disk_manifest(tmp_path / "notest")
    no_test["items"] = [it for it in no_test["items"]
                        if not (it["split"] == "test"
                                and it["identity_id"] == "001")]
    with pytest.raises(RuntimeError, match="no held-out image"):
        rf.build_design("ppubench", "fs_001", ["001"], [17], [17],
                        man=no_test, images=rf.held_out_images(no_test))

    # the checkpoint refusal names every absent role rather than reporting a
    # smaller requirement as complete
    req = rf.required_checkpoints("ppubench", "fs_001", [17, 42, 123],
                                  [17, 42, 123])
    monkeypatch.setattr(rf, "ADAPTER_RELPATH",
                        ("adapter_final", "does_not_exist.safetensors"))
    got = rf.verify_checkpoints(req, "ppubench", "fs_001")
    assert got["complete"] is False and got["n_present"] == 0
    roles = {m["role"] for m in got["missing"]}
    assert "baseline_g" in roles and "baseline_h" in roles
    assert sum(1 for m in got["missing"]
               if m["role"].startswith("edited_h")) == 4



# --------------------------------------------------------------------------
# 15. the two frozen pilots, read off the real tree
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_pilots(rf):
    out = {}
    for ds in ("ppubench", "salmu"):
        sel = rf.pilot_forget_set(ds, [17, 42, 123])
        out[ds] = (sel, rf.build_pilot_preregistration(
            ds, sel["set_id"], sel["targets"], [17, 42, 123], [17, 42, 123],
            selection=sel))
    return out


@_needs_tree
def test_the_pilot_forget_set_is_chosen_by_a_stated_rule(rf, real_pilots):
    """Not hand-picked: whatever the rule selects from the frozen matrix, with
    the sets passed over recorded and the reason they were passed over."""
    sel, b = real_pilots["ppubench"]
    assert sel["set_id"] == "fs_001" and sel["targets"] == ["001"]
    assert b["forget_set_id"] == "fs_001"
    assert b["forget_identity_ids"] == ["001"]
    assert sel["n_candidates"] == 7 and sel["n_considered"] == 4
    passed_over = [c for c in sel["candidates"] if not c["considered"]]
    assert len(passed_over) == 3
    assert all("single-target" in c["why_not"] for c in passed_over)
    assert all(c["complete"] for c in sel["candidates"] if c["considered"])
    assert "first single-target set" in sel["selection_rule"]
    assert "selection effect" in sel["rule_is_derived_not_preferred"]

    # SALMU's dataset manifest declares no forget_identity_ids at all, so the
    # rule is the only way its target is not typed in by hand
    s_sel, s = real_pilots["salmu"]
    assert len(s_sel["targets"]) == 1
    assert s_sel["n_considered"] == 6
    assert s["forget_identity_ids"] == s_sel["targets"]
    assert s_sel["set_id"] == "fs_" + s_sel["targets"][0]


@_needs_tree
def test_edit_seeds_the_matrix_never_trained_are_refused(rf):
    """Reusing checkpoints trained under other seeds is not reuse, it is a
    request for a retraining -- and this turn trains nothing."""
    with pytest.raises(RuntimeError, match="not the ones the frozen"):
        rf.pilot_forget_set("ppubench", [17, 42])
    with pytest.raises(RuntimeError, match="not the ones the frozen"):
        rf.pilot_forget_set("salmu", [7])


@_needs_tree
def test_the_promotion_counts_are_multiplied_not_written_down(rf, real_pilots):
    _, p = real_pilots["ppubench"]
    pr = p["promotion_rule"]
    assert (pr["n_router_seeds"], pr["n_forget_sets"],
            pr["n_edit_seeds"]) == (3, 7, 3)
    assert pr["n_e2e_cells"] == 63 == 3 * pr["n_forget_sets"] * 3
    assert pr["retrained_per_cell"] is False
    assert pr["reuses_existing_edited_h"] == pr["edited_h_required"] == 21
    assert pr["reuse_inventory_is_complete"] is True

    # the inventory names real bytes, so "reuse" is a checkable claim
    inv = pr["reuse_inventory"]
    assert inv["n_forget_sets"] == 7 and inv["complete"] is True
    for sid, e in inv["sets"].items():
        assert e["n_reusable"] == e["n_required"] == 3, sid
        for seed, v in e["per_edit_seed"].items():
            assert v["adapter_present"] and v["cell_results_present"], (sid, seed)
            assert v["adapter_sha256"] and len(v["adapter_sha256"]) == 64
    assert all(s["present"] and s["sha256"]
               for s in inv["shared_route_checkpoints"].values())
    assert set(inv["shared_route_checkpoints"]) == {"baseline_g", "baseline_h"}

    _, s = real_pilots["salmu"]
    spr = s["promotion_rule"]
    assert spr["n_e2e_cells"] == 3 * spr["n_forget_sets"] * 3
    assert spr["blocked_by"] is None
    assert spr["held_out_routing_is_testable_here"] is True


@_needs_tree
def test_ppubench_routing_gates_are_recorded_untestable_not_passed(rf,
                                                                  real_pilots):
    """The headline PPUBench fact, derived here rather than asserted: its test
    images are byte-identical to its training images, so the 1.0 held-out
    routing accuracy on record was measured on training bytes."""
    _, b = real_pilots["ppubench"]
    a = b["held_out_images"]["content_audit"]
    assert a["n_uris"] == 74 and a["n_distinct_image_bytes"] == 4
    assert a["by_split"]["test"]["n_uris"] == 12
    assert a["by_split"]["test"]["n_distinct_bytes"] == 4
    assert a["n_test_bytes_also_in_train"] == 4
    assert a["held_out_image_content_exists"] is False

    ap = b["gate_applicability"]
    assert ap["n_gates_supportable"] == 5 and ap["n_gates_total"] == 7
    assert ap["routing_gates"]["supportable"] is False
    assert "UNSUPPORTABLE" in b["promotion_rule"]["blocked_by"]
    assert b["held_out_images"]["warning"]

    # untestable is not lowered, and not waived
    th = b["frozen_gates"]["thresholds"]
    assert th == dict(rf.GATE_THRESHOLDS)
    assert th["min_router_held_out_accuracy"] == 0.90
    assert b["frozen_gates"]["n_gates"] == 7
    assert ap["clustering"]["level_to_use"] == "identity"
    assert b["cluster_bootstrap"]["n_clusters_resampled"] == 4


@_needs_tree
def test_salmu_is_the_pilot_that_can_test_routing(rf, real_pilots):
    _, b = real_pilots["salmu"]
    a = b["held_out_images"]["content_audit"]
    assert a["n_uris"] == a["n_distinct_image_bytes"] == 132
    assert a["n_test_bytes_also_in_train"] == 0
    assert a["held_out_image_content_exists"] is True
    assert b["held_out_images"]["warning"] is None

    # the manifest's own digest table agrees, which ties the audit to the
    # snapshot the manifest was frozen against
    cc = a["cross_check_against_manifest_table"]
    assert cc["manifest_carries_images_sha256"] is True
    assert cc["n_compared"] == 132 and cc["n_disagreeing"] == 0

    assert b["gate_applicability"]["n_gates_supportable"] == 7
    assert b["cluster_bootstrap"]["cluster_level"] == "image"
    assert b["cluster_bootstrap"]["n_clusters_resampled"] == 36
    assert b["checkpoint_requirements"]["complete"] is True


@_needs_tree
def test_gate_one_expected_outcome_is_computed_not_asserted(rf, real_pilots,
                                                            monkeypatch):
    """0.8056 against a 0.90 floor is a FAIL, and the artifact says so before
    the run.  That it flips when the input flips is what makes it computed."""
    _, s = real_pilots["salmu"]
    g1 = s["promotion_rule"]["expected_gate_one_outcome"]
    assert g1["recorded_held_out_accuracy"] == pytest.approx(0.8056)
    assert g1["frozen_threshold"] == rf.GATE_THRESHOLDS[
        "min_router_held_out_accuracy"] == 0.90
    assert g1["expected_to_pass"] is False
    assert "expected to FAIL" in g1["reading"]
    assert "not to be tuned away" in g1["reading"]
    assert g1["recorded_source"] == rf.HELD_OUT_G["salmu"]["source"]

    monkeypatch.setitem(rf.HELD_OUT_G["salmu"], "accuracy", 0.95)
    inv = {"n_forget_sets": 9, "n_reusable_edited_h": 27,
           "n_edited_h_required": 27, "complete": True}
    got = rf.promotion_rule_salmu([17, 42, 123], [17, 42, 123], inv,
                                  s["gate_applicability"])
    assert got["expected_gate_one_outcome"]["expected_to_pass"] is True
    assert "expected to PASS" in got["expected_gate_one_outcome"]["reading"]


@_needs_tree
def test_the_promotion_block_follows_the_audit(rf, real_pilots):
    """Whether promotion is blocked is a consequence of the bytes, not a
    sentence written about them."""
    _, p = real_pilots["ppubench"]
    _, s = real_pilots["salmu"]
    assert p["promotion_rule"]["held_out_routing_is_testable_here"] is False
    assert s["promotion_rule"]["held_out_routing_is_testable_here"] is True

    inv = {"n_forget_sets": 7, "n_reusable_edited_h": 21,
           "n_edited_h_required": 21, "complete": True}
    supportable = {"routing_gates": {"supportable": True, "gates": [],
                                     "reason_if_not": "n/a"}}
    got = rf.promotion_rule_ppubench([17, 42, 123], [17, 42, 123], inv,
                                     supportable)
    assert got["blocked_by"] is None
    assert got["held_out_routing_is_testable_here"] is True

    # the two rules are read side by side, so the fields a reader compares must
    # exist in both: "blocked_by" absent on one and None on the other would be
    # indistinguishable from a field that was never computed
    common = set(rf._promotion_common("ppubench", [17, 42, 123], [17, 42, 123],
                                      inv))
    for name, rule in (("ppubench", p["promotion_rule"]),
                       ("salmu", s["promotion_rule"])):
        assert common <= set(rule), name
        assert "blocked_by" in rule, name
        assert "held_out_routing_is_testable_here" in rule, name
        assert "reuse_inventory" in rule, name
    assert "expected_gate_one_outcome" in s["promotion_rule"]


def test_mllmu_gets_no_promotion_rule_rather_than_another_datasets(rf):
    """MLLMU has ONE image per identity, so held-out routing does not exist
    there.  A default branch would hand it SALMU's prose and freeze a claim the
    dataset cannot support."""
    assert "mllmu" not in rf.PROMOTION_RULES
    assert set(rf.PROMOTION_RULES) == {"ppubench", "salmu"}
    inv = {"n_forget_sets": 1, "n_reusable_edited_h": 3,
           "n_edited_h_required": 3, "complete": True}
    ap = {"routing_gates": {"supportable": False, "gates": [],
                            "reason_if_not": "one image per identity"}}
    with pytest.raises(RuntimeError, match="no promotion rule for 'mllmu'"):
        rf.promotion_rule("mllmu", [17], [17, 42, 123], inv, ap)


# --------------------------------------------------------------------------
# 16. freezing and verifying a pilot
# --------------------------------------------------------------------------

@_needs_tree
def test_a_frozen_pilot_verifies_through_the_constructor_that_made_it(
        rf, real_pilots, tmp_path):
    """The bug this pins: verify_manifest rebuilt EVERY manifest with
    build_design, so a pilot -- which also carries the audit, the gates and the
    promotion rule -- could never reproduce its own digest and would have
    reported invalid the moment it was frozen."""
    _, block = real_pilots["ppubench"]
    path, digest = rf.freeze_manifest(block, tmp_path / "rf_pilot.json")
    got = rf.verify_manifest(path)
    assert got["valid"] is True, got["problems"]
    assert got["kind"] == rf.PREREG_KIND
    assert got["executed"] is False
    assert got["n_cells"] == 3 and got["n_rows_total"] == 126
    assert got["n_checkpoints_rehashed"] == 5

    frozen = json.loads(path.read_text(encoding="utf-8"))
    assert frozen["frozen"] is True
    assert got["design_sha256"] == frozen["design_sha256"]
    assert digest == _sha(path)
    # every input the manifest names is resolvable, so the digest table binds
    for name, want in frozen["provenance"]["input_file_sha256"].items():
        p = rf.DATASET_ROOT / name
        assert p.is_file(), name
        assert _sha(p) == want, name
    assert frozen["provenance"]["missing_input_files"] == []

    # a bare design is still verified as a bare design, and is not mistaken
    # for a pilot
    d = rf.build_design("ppubench", "fs_001", ["001"], [17, 42, 123],
                        [17, 42, 123])
    p2, _ = rf.freeze_manifest(d, tmp_path / "rf_manifest.json")
    got2 = rf.verify_manifest(p2)
    assert got2["valid"] is True, got2["problems"]
    assert got2["kind"] == rf.KIND
    assert got2["executed"] is None
    assert got2["n_checkpoints_rehashed"] == 0


@_needs_tree
def test_the_pilot_manifest_is_deterministic(rf, real_pilots):
    """Deriving the promotion rule and the selection explicitly must give the
    same bytes as letting the constructor derive both."""
    sel, block = real_pilots["ppubench"]
    again = rf.build_pilot_preregistration(
        "ppubench", sel["set_id"], sel["targets"], [17, 42, 123], [17, 42, 123])
    assert rf.design_sha256(again) == rf.design_sha256(block)
    assert again["pilot_forget_set_selection"] == block[
        "pilot_forget_set_selection"]
    assert again["promotion_rule"] == block["promotion_rule"]


@_needs_tree
def test_editing_a_frozen_pilot_is_detected(rf, real_pilots, tmp_path):
    """A pre-registration is only one if it cannot be quietly edited
    afterwards; the digest plus the rebuild is what makes that checkable."""
    _, block = real_pilots["ppubench"]
    path, _ = rf.freeze_manifest(block, tmp_path / "p.json")
    assert rf.verify_manifest(path)["valid"] is True

    frozen = json.loads(path.read_text(encoding="utf-8"))
    frozen["frozen_gates"]["thresholds"][
        "min_router_held_out_accuracy"] = 0.50     # lower a gate afterwards
    path.write_text(rf.canonical_json(frozen), encoding="utf-8")
    got = rf.verify_manifest(path)
    assert got["valid"] is False
    assert any("design_sha256 does not match" in p for p in got["problems"])

    # claiming the pilot had been executed is caught the same way
    frozen = json.loads(path.read_text(encoding="utf-8"))
    frozen["executed"] = True
    path.write_text(rf.canonical_json(frozen), encoding="utf-8")
    assert rf.verify_manifest(path)["valid"] is False


@_needs_tree
def test_a_manifest_of_an_unknown_kind_is_not_verified_as_a_design(
        rf, real_pilots, tmp_path):
    """Rebuilding through the wrong constructor is how a verifier comes to
    fail everything, so an unrecognised kind is refused rather than guessed
    at."""
    _, block = real_pilots["ppubench"]
    path, _ = rf.freeze_manifest(block, tmp_path / "q.json")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    frozen["kind"] = "something_else_v1"
    frozen["design_sha256"] = rf.design_sha256(
        {k: v for k, v in frozen.items()
         if k not in ("frozen", "design_sha256", "provenance")})
    path.write_text(rf.canonical_json(frozen), encoding="utf-8")
    got = rf.verify_manifest(path)
    assert got["valid"] is False
    assert any("cannot be rebuilt at all" in p for p in got["problems"])
    assert "unknown manifest kind" in json.dumps(got["problems"])



