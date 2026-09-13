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

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"


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


def test_checkpoints_are_hashed_and_a_missing_one_is_a_refusal(rf):
    """Existence is not identity: a checkpoint replaced in place keeps its path,
    and the digest is what ties a result to the weights that produced it."""
    req = rf.required_checkpoints("ppubench", "fs_001", [17, 42, 123],
                                  [17, 42, 123])
    assert sorted(req["roles"]) == ["baseline_g", "baseline_h",
                                    "direct_condition", "edited_h"]
    assert req["roles"]["edited_h"]["n_required"] == 3
    assert req["roles"]["direct_condition"]["n_required"] == 0
    assert req["roles"]["direct_condition"]["adapter"] == "reuses edited_h"
    assert req["router_seed_is_not_a_checkpoint"]

    got = rf.verify_checkpoints(req, "ppubench", "fs_001")
    assert got["n_required"] == 5
    for role, entry in got["present"].items():
        assert len(entry["sha256"]) == 64 and entry["bytes"] > 0
        assert role in ("baseline_g", "baseline_h") or \
            role.startswith("edited_h__seed")
    # PPUBench has all 21 matrix cells, so the three fs_001 adapters exist
    assert got["complete"] is True, got["missing"]

    # a forget set with no cells cannot be silently accepted
    got2 = rf.verify_checkpoints(req, "ppubench", "fs_999_does_not_exist")
    assert got2["complete"] is False
    assert any(m["role"].startswith("edited_h") for m in got2["missing"])


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


def test_rf1_refuses_rather_than_pretending_to_run(rf):
    """The GPU phase is the one thing this iteration does not implement, and it
    must say so instead of producing an empty cell that later reads as a
    result."""
    with pytest.raises(RuntimeError, match="needs a GPU session"):
        rf.main(["--dataset", "ppubench", "--phase", "RF1",
                 "--forget-ids", "001"])

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
