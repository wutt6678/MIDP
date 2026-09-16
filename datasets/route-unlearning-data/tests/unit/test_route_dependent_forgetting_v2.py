"""Route-dependent forgetting, v2: the repairs, pinned one per test.

v1 was frozen and never executed, and eight things about it were wrong in ways
that a passing run would not have revealed:

  1. the forced-code rows presented an image to ``h`` as well as a code, so the
     system under test was f(X, C) and not the frozen Y = h(C);
  2. the direct control "reused edited_h", an adapter trained on code-to-label
     pairs, so it was not a direct image model at all -- and because the only
     gate on it bounded code-following from ABOVE, a model emitting arbitrary
     wrong labels would have passed;
  3. three router seeds were declared over one checkpoint and one cached
     prediction file, so the seed was a label rather than a factor;
  4. the runner had one RF1 stub that raised, so no phase could say which factor
     it varied;
  5. one omnibus ``passed`` boolean let a routing failure cancel a mediation
     success;
  6. SALMU's bootstrap was image-level, over images that share an identity, a
     router and an edited h;
  7. the factorization took P_h from candidate score sums, which are not
     probabilities;
  8. the manifests recorded all of the above.

Each test below names the repair it pins.  They exercise the CPU layers only --
no test loads a model -- and the ones that need the real dataset bytes or the
gitignored adapter weights say so and skip with the absent path named.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location(
        "route_forgetting_v2_under_test",
        _SCRIPTS / "e2c_v3_route_dependent_forgetting.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rf():
    return _load()


# --------------------------------------------------------------------------
# what the real tree has, checked rather than assumed
# --------------------------------------------------------------------------

_MANIFESTS = {"ppubench": _ROOT / "e2c_v3_real" / "manifests"
              / "realdata_identity_mapping.json",
              "salmu": _ROOT / "e2c_salmu" / "manifests"
              / "salmu_manifest.json"}


def _absent_images(dataset):
    """The first image the dataset manifest names that is not on disk.

    PPUBench's images live OUTSIDE the repository and SALMU's are gitignored, so
    a fresh clone has both manifests and neither set of bytes.  A test that
    hashes images must say so, and must skip on the path that is actually
    missing rather than on a directory that happens to exist.
    """
    path = _MANIFESTS.get(dataset)
    if not path or not path.is_file():
        return f"manifest absent: {path}"
    man = json.loads(path.read_text(encoding="utf-8"))
    for it in man["items"]:
        p = Path(it["image_uri"])
        p = p if p.is_absolute() else _ROOT / p
        if not p.is_file():
            return str(p)
    return None


_ABSENT = {ds: _absent_images(ds) for ds in ("ppubench", "salmu")}
_needs_real_images = pytest.mark.skipif(
    any(_ABSENT.values()),
    reason=("asserts on the real dataset images, which are out of tree "
            "(PPUBench) or gitignored (SALMU), so a fresh clone has the "
            "manifests but not the bytes -- "
            + "; ".join(f"{ds}: {p}" for ds, p in sorted(_ABSENT.items()) if p)))

_V1_MANIFESTS = [_ROOT / "e2c_route_forgetting" / "manifests" / n
                 for n in ("rf_pilot_ppubench.json", "rf_pilot_salmu.json")]
_V1_PRESENT = [p for p in _V1_MANIFESTS if p.is_file()]
#: "Preserved byte-identical" is checked against the files' own recorded digests
#: and against the commit that froze them, never against a digest copied into
#: this test file -- a number written down here would be a claim nothing verifies.
_needs_v1_manifests = pytest.mark.skipif(
    len(_V1_PRESENT) != len(_V1_MANIFESTS),
    reason=("the superseded v1 pilots are not in this checkout: "
            + ", ".join(str(p) for p in _V1_MANIFESTS if not p.is_file())))


# --------------------------------------------------------------------------
# a hermetic manifest with real image bytes on disk
# --------------------------------------------------------------------------
#
# Mirrors the v1 module's helper rather than importing it: two test modules that
# share a fixture by import order are two modules that break together, and the
# hermetic manifest is twenty lines.

ALIASES = {"001": "Oden", "002": "Seri", "003": "Kael", "004": "Dax"}


def _write_image(path, token):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + token.encode("utf-8"))
    return str(path)


def _disk_manifest(tmp_path, duplicate_test_bytes=False, n_test=3,
                   identity_ids=("001", "002", "003", "004")):
    """A manifest whose ``image_uri`` values point at files that exist.

    ``duplicate_test_bytes`` reproduces PPUBench's real condition: the test
    split names DIFFERENT FILES whose bytes also appear in train, so the split
    partitions filenames rather than images.
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
            token = first_train if duplicate_test_bytes else f"{iid}-test-{i}"
            uri = _write_image(tmp_path / "img" / f"{iid}_test_{i}.png", token)
            items.append({"identity_id": iid, "split": "test",
                          "image_uri": uri})
    return {
        "dataset": "hermetic",
        "identity_ids": list(identity_ids),
        "alias_of": {i: ALIASES[i] for i in identity_ids},
        "code_of": {i: f"RID_{i}" for i in identity_ids},
        "deleted_label": "Unknown",
        "forget_identity_ids": ["001"],
        "items": items,
        "seed": 17,
    }


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SELECTION = {"dataset": "ppubench", "set_id": "fs_001", "targets": ["001"],
             "selection_rule": "fixture", "n_candidates": 7, "n_considered": 4,
             "candidates": []}

FAKE_INVENTORY = {"dataset": "ppubench", "n_forget_sets": 7, "sets": {},
                  "n_reusable_edited_h": 21, "n_edited_h_required": 21,
                  "shared_route_checkpoints": {}, "complete": True}


def _stub_tree(rf, monkeypatch):
    """Replace the three things a hermetic build cannot have.

    Checkpoint PRESENCE is a property of the GPU tree, the forget sets are a
    property of the frozen matrix, and the reuse inventory is a property of both.
    All three are pinned against the real tree separately, below.
    """
    monkeypatch.setattr(rf, "verify_checkpoints_v2", lambda req: {
        "complete": False, "n_present": 0, "n_required": len(req["roles"]),
        "n_files_required": sum(s["n_files_required"]
                                for s in req["roles"].values()),
        "present": {}, "absent": {role: {} for role in req["roles"]},
        "must_be_trained": sorted(req["roles"]),
        "n_must_be_trained": len(req["roles"]),
        "unexpectedly_absent": [], "note": "stub",
        "hashed_not_just_existence_checked": "stub"})
    monkeypatch.setattr(rf, "_promotion_forget_sets",
                        lambda ds: [f"fs_{i:03d}" for i in range(1, 8)])
    monkeypatch.setattr(rf, "reuse_inventory",
                        lambda ds, sets, seeds: dict(FAKE_INVENTORY))


def _prereg_v2(rf, tmp_path, monkeypatch, man=None, dataset="ppubench",
               router_seeds=None, edit_seeds=None, direct_seeds=None):
    """A hermetic v2 pilot: real image bytes, no real checkpoints or matrices."""
    man = man or _disk_manifest(tmp_path)
    _stub_tree(rf, monkeypatch)
    if router_seeds is None or direct_seeds is None:
        policy = rf.pilot_seed_policy_v2(dataset, man=man)
        router_seeds = policy["router_seeds"] if router_seeds is None \
            else router_seeds
        direct_seeds = policy["direct_seeds"] if direct_seeds is None \
            else direct_seeds
    edit_seeds = [17, 42, 123] if edit_seeds is None else edit_seeds
    return rf.build_pilot_preregistration_v2(
        dataset, "fs_001", ["001"], router_seeds, edit_seeds, direct_seeds,
        man=man, images=rf.held_out_images(man), selection=dict(SELECTION)), man


def _design_v2(rf, tmp_path, monkeypatch=None, man=None, **kw):
    man = man or _disk_manifest(tmp_path)
    args = {"dataset": "ppubench", "forget_set_id": "fs_001",
            "forget_ids": ["001"], "router_seeds": [17, 42, 123],
            "edit_seeds": [17, 42, 123], "direct_seeds": [17],
            "man": man, "images": rf.held_out_images(man)}
    args.update(kw)
    return rf.build_design_v2(
        args.pop("dataset"), args.pop("forget_set_id"), args.pop("forget_ids"),
        args.pop("router_seeds"), args.pop("edit_seeds"),
        args.pop("direct_seeds"), **args), man


def _cells(design, kind):
    return [c for c in design["cells"] if c["kind"] == kind]


def _rows(design, kind, i=0):
    return _cells(design, kind)[i]["rows"]


# --------------------------------------------------------------------------
# a hermetic stand-in for the RF1 phases
# --------------------------------------------------------------------------
#
# The GPU phases generate text; everything downstream of that is pure.  So the
# tests fabricate the generations a well-behaved -- or a specifically misbehaving
# -- model would have produced, and run the real scorer and the real aggregate on
# them.  What is fabricated is the model, never the logic under test.

def _identity_of_code(code, man):
    for iid in man["identity_ids"]:
        if man["code_of"][iid] == code:
            return iid
    return None


def _fill_router(rows, man, wrong_routes=()):
    """Give natural rows a routed code: perfect, except for the images named.

    RF1E fills these from the router seed's own prediction file; the design is
    frozen with them pending, because pre-registering an observation is not
    pre-registration.
    """
    out = []
    for r in rows:
        own = man["code_of"][r["image_identity_id"]]
        code = own
        if r["image_uri"] in wrong_routes:
            other = next(i for i in man["identity_ids"]
                         if i != r["image_identity_id"])
            code = man["code_of"][other]
        out.append({**r, "routed_code": code, "routed_code_correct": code == own,
                    "routable": True, "observation_pending": False,
                    "filled_by": "fixture"})
    return out


def _wrong_label(man, iid, not_this=None):
    """A label that is in the vocabulary and is neither the right one nor, where
    a code was presented, the one that code names.

    Without ``not_this`` the "arbitrary wrong label" fixture accidentally
    reproduces the forced code's own alias on the hybrid rows, because the design
    picks the irrelevant code as the FIRST identity that is not the image's and a
    naive "first wrong alias" picks the same one.  A model that follows the code
    is not the model item 2 is about.
    """
    for i, alias in sorted(man["alias_of"].items()):
        if i != iid and alias != not_this:
            return alias
    return DELETED


def _fake_generation(row, man, forget, behaves, forced):
    """The text a model with this behaviour would have emitted for one row.

    ``forced`` is the module's own FORCED_CODE_CONDITIONS_V2 rather than a list
    written down here, so a condition added to the design is simulated by
    whatever rule the design gives it instead of silently falling through.
    """
    cond = row["condition"]
    if cond in forced:
        iid = row["forced_identity_id"]
        if behaves == "arbitrary":
            return _wrong_label(man, iid)
        return DELETED if iid in forget else man["alias_of"][iid]
    if cond == "natural_mediated":
        code = row["routed_code"]
        if code is None:
            return None
        iid = _identity_of_code(code, man)
        if behaves == "leaks":
            # h answered from the IMAGE rather than from the code it was given:
            # the mediator is not hard, and the factorization must catch it
            iid = row["image_identity_id"]
        return DELETED if iid in forget else man["alias_of"][iid]
    iid = row["image_identity_id"]
    if behaves == "arbitrary":
        return _wrong_label(man, iid, row.get("forced_label"))
    if behaves == "captured" and row.get("forced_label"):
        return row["forced_label"]
    return man["alias_of"][iid]


DELETED = "Unknown"


def _scored(rf, design, man, kind, behaves="design", wrong_routes=(), cell=0):
    """One cell's rows, generated by the fake model and scored by the real one."""
    rows = _cells(design, kind)[cell]["rows"]
    forget = set(design["forget_identity_ids"])
    forced = rf.FORCED_CODE_CONDITIONS_V2
    if kind == "natural":
        rows = _fill_router(rows, man, wrong_routes)
    filled = []
    for r in rows:
        field = rf.generation_field(r["condition"])
        filled.append({**r,
                       field: _fake_generation(r, man, forget, behaves, forced)})
    scored, missing = rf.score_rows_v2(filled, design["vocab"])
    assert not missing, f"the fixture left rows ungenerated: {missing}"
    return scored


def _all_scored(rf, design, man, behaves="design", wrong_routes=()):
    return {kind: _scored(rf, design, man, kind, behaves, wrong_routes)
            for kind in ("intervention", "natural", "direct", "hybrid")}


E2E_OK = {"observed": 1.0, "predicted": 1.0, "n_images": 12}


# ==========================================================================
# item 1 -- the mediator stays hard: Y = h(do(C=c)), no image to h
# ==========================================================================

def test_no_v2_condition_sends_an_image_to_h(rf):
    """The frozen architecture is Y = h(C).  The moment h also receives the
    image the system is f(X, C) and an intervention on C no longer speaks about
    the route, so this is the invariant everything else rests on."""
    for name, cond in rf.CONDITIONS_V2.items():
        assert cond["image_to_h"] is False, \
            f"{name} sends an image to h, which makes the system f(X, C)"
    assert set(rf.CONDITIONS_V2) == {
        "natural_mediated", "forgotten_route_intervention",
        "retained_route_intervention", "retained_control", "direct_control",
        "hybrid_conflict_probe"}


def test_no_row_of_any_cell_kind_sends_an_image_to_h(rf, tmp_path):
    design, _ = _design_v2(rf, tmp_path)
    rows = [r for c in design["cells"] for r in c["rows"]]
    assert rows, "a design with no rows would pass this vacuously"
    assert not any(r["image_to_h"] for r in rows)
    for cell in design["cells"]:
        assert cell["coverage"]["mediator_is_hard"] is True


def test_a_row_that_claimed_to_send_an_image_to_h_is_a_defect(rf, tmp_path):
    design, man = _design_v2(rf, tmp_path)
    rows = [dict(r) for r in _rows(design, "intervention")]
    rows[0]["image_to_h"] = True
    audit = rf.coverage_audit_v2(rows, man, ["001"], "intervention")
    assert audit["exact"] is False
    assert audit["mediator_is_hard"] is False
    assert any("f(X, C)" in d for d in audit["defects"])


def test_a_forced_code_row_may_not_carry_an_image_at_all(rf, tmp_path):
    """The image identity is allowed on a forced-code row as METADATA -- as the
    list of that identity's held-out images -- but never as an image the row
    presents.  A row carrying ``image_uri`` in a do(C=c) condition has nowhere
    for that image to go except into h."""
    design, man = _design_v2(rf, tmp_path)
    rows = [dict(r) for r in _rows(design, "intervention")]
    rows[0]["image_uri"] = "/img/001/test_0.png"
    audit = rf.coverage_audit_v2(rows, man, ["001"], "intervention")
    assert audit["exact"] is False
    assert any("forces a code but carries an image_uri" in d
               for d in audit["defects"])


def test_the_image_is_metadata_on_intervention_rows_and_never_an_input(
        rf, tmp_path):
    design, man = _design_v2(rf, tmp_path)
    for row in _rows(design, "intervention"):
        assert row["image_uri"] is None
        assert row["image_identity_id"] is None
        # the identity's held-out images are recorded so a reader can see what
        # v1 presented and why presenting it broke the architecture
        assert row["held_out_images_of_this_identity"] == [
            it["image_uri"] for it in man["items"]
            if it["identity_id"] == row["forced_identity_id"]
            and it["split"] == "test"]


def test_natural_mediated_may_carry_an_image_because_it_goes_to_g(rf, tmp_path):
    """The distinction the check has to get right: ``natural_mediated`` is the
    one condition where X reaches the system, and it reaches g.  Forbidding an
    image there would forbid the composition the whole experiment is about."""
    design, man = _design_v2(rf, tmp_path)
    rows = _rows(design, "natural")
    assert all(r["image_uri"] for r in rows)
    assert all(r["image_to_h"] is False for r in rows)
    assert rf.coverage_audit_v2(rows, man, ["001"], "natural")["exact"]
    assert "natural_mediated" not in rf.FORCED_CODE_CONDITIONS_V2
    assert "natural_mediated" in rf.MEDIATED_CONDITIONS_V2


def test_the_forced_code_conditions_are_exactly_the_ones_h_sees_alone(rf):
    assert set(rf.FORCED_CODE_CONDITIONS_V2) == {
        "forgotten_route_intervention", "retained_route_intervention",
        "retained_control"}
    assert set(rf.DECISIVE_CONDITIONS_V2) == {
        "forgotten_route_intervention", "retained_route_intervention"}


def test_the_image_plus_code_condition_is_renamed_and_auxiliary(rf):
    """Item 1's fallback: image-plus-code evaluation may stay, but as an
    auxiliary bypass/conflict probe and never as the causal intervention."""
    probe = rf.CONDITIONS_V2["hybrid_conflict_probe"]
    assert probe["auxiliary"] is True
    assert "not_the_mediator_intervention" in probe
    assert rf.AUXILIARY_CONDITIONS_V2 == ("hybrid_conflict_probe",)
    assert "hybrid_conflict_probe" not in rf.DECISIVE_CONDITIONS_V2
    assert "hybrid_conflict_probe" not in rf.MEDIATED_CONDITIONS_V2
    assert "direct_path" not in rf.CONDITIONS_V2, \
        "v1's name for the image-plus-code condition must not survive as a " \
        "condition, only as the renamed auxiliary probe"
    # and it runs on D_s, not on h
    assert probe["prompt_key"] == "hybrid_image_and_code"
    assert probe["depends_on"] == ("direct_seed",)


def test_the_retained_control_is_the_same_call_not_a_second_one(rf, tmp_path):
    """With a hard mediator no image distinguishes the retained control from the
    retained-route intervention, so evaluating both would run one call twice and
    report it as two observations."""
    assert rf.CONDITIONS_V2["retained_control"]["same_execution_as"] == \
        "retained_route_intervention"
    design, man = _design_v2(rf, tmp_path)
    rows = _rows(design, "intervention")
    assert len(rows) == len(man["identity_ids"]), \
        "one row per route code, not one per condition-name per code"
    retained = [r for r in rows if r["arm"] == "retained"]
    assert all(r["also_reported_as"] == ("retained_control",) for r in retained)
    assert all(r["also_reported_as"] == () for r in rows
               if r["arm"] == "forgotten")
    assert sum(1 for r in rows if r["condition"] == "retained_control") == 0, \
        "retained_control is a NAME the retained rows are also reported under, " \
        "not a second set of rows"


def test_a_code_evaluated_twice_in_one_cell_is_a_defect(rf, tmp_path):
    design, man = _design_v2(rf, tmp_path)
    rows = [dict(r) for r in _rows(design, "intervention")]
    rows.append({**rows[0], "row_id": rows[0]["row_id"] + "::again"})
    audit = rf.coverage_audit_v2(rows, man, ["001"], "intervention")
    assert audit["exact"] is False
    assert any("more than once" in d for d in audit["defects"])


def test_the_h_prompt_is_read_from_the_frozen_scripts(rf):
    """v1 retyped the h prompt as 'Identity code: {code}. What is the alias?'.
    The route was trained with 'Generate the alias.', so v1 evaluated an input
    the model had never seen -- a second off-protocol deviation, independent of
    the image leak.  v2 reads the constant out of the frozen scripts by AST."""
    prompts = rf.frozen_route_prompts()
    assert prompts["h_code_to_alias"] == \
        "Identity code: {code}. Generate the alias."
    assert prompts["h_code_to_alias"] != rf.CODE_PROMPT, \
        "v1's paraphrase must not be what v2 uses"
    assert prompts["g_image_to_code"] == \
        "What is the identity code for this person?"
    assert prompts["agreement_checked"] is True
    assert prompts["read_from"] == list(rf.FROZEN_ROUTE_SCRIPTS)


def test_the_prompts_are_read_by_ast_and_not_by_importing(rf):
    """The frozen route scripts import torch at module scope, so reading them by
    import would drag a GPU dependency into every CPU layer."""
    had = "torch" in sys.modules
    rf.frozen_route_prompts()
    rf.frozen_route_protocol()
    assert ("torch" in sys.modules) == had


def test_scripts_that_disagree_on_a_prompt_are_refused(rf, tmp_path,
                                                       monkeypatch):
    """If the three route scripts ever diverged, "the same prompt" would be an
    assumption rather than a fact, and this runner refuses to pick one."""
    original = (_SCRIPTS / rf.FROZEN_ROUTE_SCRIPTS[0]).read_text(
        encoding="utf-8")
    divergent = original.replace(
        'CODE_TO_ALIAS_PROMPT = "Identity code: {code}. Generate the alias."',
        'CODE_TO_ALIAS_PROMPT = "Identity code: {code}. Name the person."')
    assert divergent != original, "the frozen prompt moved; update this test"
    fake = tmp_path / rf.FROZEN_ROUTE_SCRIPTS[0]
    fake.write_text(divergent, encoding="utf-8")
    for name in rf.FROZEN_ROUTE_SCRIPTS[1:]:
        (tmp_path / name).write_text((_SCRIPTS / name).read_text(
            encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(rf, "SCRIPT_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="disagree on CODE_TO_ALIAS_PROMPT"):
        rf.frozen_route_prompts()


def test_a_script_that_lacks_the_frozen_constant_is_refused(rf, tmp_path,
                                                            monkeypatch):
    fake = tmp_path / rf.FROZEN_ROUTE_SCRIPTS[0]
    fake.write_text("SOMETHING_ELSE = 1\n", encoding="utf-8")
    monkeypatch.setattr(rf, "SCRIPT_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="does not define"):
        rf.frozen_route_prompts()


def test_the_training_protocol_is_read_and_required_to_agree(rf):
    """g_42 and g_123 must be trained under the SAME recipe as g_17, or
    "router seed" confounds the seed with the protocol."""
    protocol = rf.frozen_route_protocol()
    assert (protocol["steps"], protocol["warmup"], protocol["lr"],
            protocol["repeat"]) == (3000, 200, 2e-5, 50)
    assert protocol["agreement_checked"] is True
    assert protocol["read_from"] == list(rf.PROTOCOL_SCRIPTS)
    assert protocol["lora"]["rank"] == 8
    assert "fresh_reinit_lora" in protocol["fresh_lora_init"]


def test_scripts_that_disagree_on_the_protocol_are_refused(rf, tmp_path,
                                                           monkeypatch):
    original = (_SCRIPTS / rf.PROTOCOL_SCRIPTS[0]).read_text(encoding="utf-8")
    divergent = original.replace("ROUTE_STEPS = 3000", "ROUTE_STEPS = 1000")
    assert divergent != original, "the frozen protocol moved; update this test"
    (tmp_path / rf.PROTOCOL_SCRIPTS[0]).write_text(divergent,
                                                   encoding="utf-8")
    for name in rf.PROTOCOL_SCRIPTS[1:]:
        (tmp_path / name).write_text((_SCRIPTS / name).read_text(
            encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(rf, "SCRIPT_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="disagree on ROUTE_STEPS"):
        rf.frozen_route_protocol()


# ==========================================================================
# item 2 -- the direct control is its own model, and needs two gates
# ==========================================================================

def test_no_role_reuses_edited_h_as_the_direct_pathway(rf, tmp_path):
    """v1's ``direct_condition.adapter`` was the string "reuses edited_h".  An
    adapter trained on code-to-label pairs is not an image model, so evaluating
    it on an image measured whatever a code-trained adapter happens to do with
    pixels."""
    design, _ = _design_v2(rf, tmp_path)
    roles = design["checkpoint_requirements"]["roles"]
    assert "direct_condition" not in roles
    direct = {k: v for k, v in roles.items() if k.startswith("direct_d__")}
    assert direct, "there must be a direct role or there is no direct control"
    for name, role in direct.items():
        assert role["adapter"] != "reuses edited_h"
        assert role["adapter"].endswith("adapter_model.safetensors")
        assert f"direct_seed_{name.rsplit('seed', 1)[1]}" in role["adapter"]
        assert role["exists_already"] is False, \
            "D_s does not exist yet and must not be claimed as present"
    assert design["checkpoint_requirements"][
        "no_direct_condition_reuses_edited_h"]
    blob = json.dumps(roles, sort_keys=True)
    assert "reuses edited_h" not in blob


def test_the_direct_adapter_paths_are_independent_per_seed(rf, tmp_path):
    design, _ = _design_v2(rf, tmp_path, direct_seeds=[17, 42, 123])
    paths = [r["adapter"] for k, r in
             design["checkpoint_requirements"]["roles"].items()
             if k.startswith("direct_d__")]
    assert len(paths) == 3 == len(set(paths))
    edited = [r["adapter"] for k, r in
              design["checkpoint_requirements"]["roles"].items()
              if k.startswith("edited_h__")]
    assert not set(paths) & set(edited), \
        "a direct adapter must not share a path with an edited h"


def test_the_direct_conditions_present_no_code_and_the_probe_presents_one(
        rf, tmp_path):
    design, man = _design_v2(rf, tmp_path)
    for row in _rows(design, "direct"):
        assert row["condition"] == "direct_control"
        assert row["forced_code"] is None
        assert row["forced_identity_id"] is None
        assert row["prompt_key"] == "d_image_to_alias"
        assert row["expected_label"] == man["alias_of"][row["image_identity_id"]]
    for row in _cells(design, "hybrid")[0]["rows"]:
        assert row["condition"] == "hybrid_conflict_probe"
        assert row["forced_code"] is not None
        assert row["forced_identity_id"] != row["image_identity_id"], \
            "the probe's code must be IRRELEVANT to the image"
        assert row["auxiliary"] is True


def test_a_direct_model_that_emits_arbitrary_wrong_labels_fails(rf, tmp_path):
    """THE hole item 2 names: bounding code-following from above is satisfied by
    a model that ignores the code and answers wrongly.  Low code-following alone
    is not evidence that the direct pathway survived."""
    design, man = _design_v2(rf, tmp_path)
    wrong = _scored(rf, design, man, "direct", behaves="arbitrary")
    assert all(r["parsed_label"] in design["vocab"] for r in wrong), \
        "the model must emit a real label, not noise, or the wrong gate catches it"
    assert not any(r["correct"] for r in wrong)
    hyb = _scored(rf, design, man, "hybrid", behaves="arbitrary")
    assert not any(r["names_forced_code_label"] for r in hyb), \
        "the fixture must not accidentally follow the code it is meant to ignore"
    gates = rf.evaluate_gates_v2(
        _scored(rf, design, man, "intervention"),
        _scored(rf, design, man, "natural"), wrong, hyb, {"17": 1.0}, E2E_OK)
    assert gates["direct_code_following_rate"]["passed"] is True, \
        "the wrong-label model follows no code, so the v1 gate would pass it"
    assert gates["direct_code_following_rate"]["value"] == 0.0
    assert gates["direct_image_accuracy"]["passed"] is False
    assert gates["direct_image_accuracy"]["value"] == 0.0
    assert "v1" in gates["direct_image_accuracy"]["description"]


def test_both_direct_gates_are_needed_together(rf, tmp_path):
    """A model that identifies the image perfectly and a model that never
    follows a code are different failures, and each gate catches one of them."""
    design, man = _design_v2(rf, tmp_path)
    captured = _scored(rf, design, man, "hybrid", behaves="captured")
    assert all(r["names_forced_code_label"] for r in captured), \
        "the fixture must actually produce a pathway captured by the code"
    gates = rf.evaluate_gates_v2(
        _scored(rf, design, man, "intervention"),
        _scored(rf, design, man, "natural"),
        _scored(rf, design, man, "direct"), captured, {"17": 1.0}, E2E_OK)
    assert gates["direct_image_accuracy"]["passed"] is True
    assert gates["direct_code_following_rate"]["passed"] is False, \
        "a direct pathway captured by an irrelevant code must fail"


def test_the_direct_gate_states_its_resolution(rf, tmp_path):
    """A 0.90 floor over twelve images is not "one failure allowed": 11/12 is
    0.917 and 10/12 is 0.833, so the achievable rates are coarse and the
    threshold reads as more permissive than it is unless the step is stated."""
    design, man = _design_v2(rf, tmp_path)
    gates = rf.evaluate_gates_v2(
        _scored(rf, design, man, "intervention"),
        _scored(rf, design, man, "natural"),
        _scored(rf, design, man, "direct"),
        _scored(rf, design, man, "hybrid"), {"17": 1.0}, E2E_OK)
    res = gates["direct_image_accuracy"]["resolution"]
    assert res["n"] == 12
    assert res["minimum_count_that_clears"] == 11
    assert gates["direct_image_accuracy"]["measured_on"].startswith("D_s")


def test_the_direct_gate_refuses_a_zero_denominator(rf, tmp_path):
    """``all([])`` is True and a mean over nothing serializes as null, which is
    how a run that evaluated nothing once reported green gates."""
    design, man = _design_v2(rf, tmp_path)
    gates = rf.evaluate_gates_v2(
        _scored(rf, design, man, "intervention"),
        _scored(rf, design, man, "natural"), [], [], {"17": 1.0}, E2E_OK)
    assert gates["direct_image_accuracy"]["passed"] is False
    assert gates["direct_image_accuracy"]["value"] is None
    assert "zero rows" in gates["direct_image_accuracy"]["failed_because"]


def test_a_threshold_not_supplied_is_refused(rf, tmp_path):
    design, man = _design_v2(rf, tmp_path)
    partial = dict(rf.GATE_THRESHOLDS_V2)
    del partial["min_direct_image_accuracy"]
    s = _all_scored(rf, design, man)
    with pytest.raises(RuntimeError, match="min_direct_image_accuracy"):
        rf.evaluate_gates_v2(s["intervention"], s["natural"], s["direct"],
                             s["hybrid"], {"17": 1.0}, E2E_OK,
                             thresholds=partial)



# ==========================================================================
# item 3 -- the router seed is a factor, not a label
# ==========================================================================

def test_a_router_seed_needs_a_checkpoint_and_its_own_predictions(rf, tmp_path):
    """v1 declared three router seeds over one baseline_g and one cached
    prediction file, so "router seed 42" and "router seed 123" described the
    same predictions three times."""
    design, _ = _design_v2(rf, tmp_path)
    roles = design["checkpoint_requirements"]["roles"]
    for seed in design["router_seeds"]:
        role = roles[f"router_g__seed{seed}"]
        assert role["n_files_required"] == 2, \
            "an adapter and a prediction file, or the seed is not a factor"
        assert role["adapter"].endswith("adapter_model.safetensors")
        assert role["held_out_predictions"].endswith(".json")
        assert role["exists_already"] is (seed == rf.EXISTING_ROUTER_SEED)
    # the frozen router replays the route's own committed cache; a trained one
    # files predictions of its own beside its adapter
    frozen = roles[f"router_g__seed{rf.EXISTING_ROUTER_SEED}"]
    assert rf.resolve_recorded_path(frozen["held_out_predictions"]) == \
        rf.DATASET_ROOT / rf.G_CACHE_PATHS["ppubench"]
    for seed in design["router_seeds"]:
        if seed == rf.EXISTING_ROUTER_SEED:
            continue
        assert roles[f"router_g__seed{seed}"]["held_out_predictions"].endswith(
            f"g_seed_{seed}/held_out_predictions.json")


def test_a_frozen_pilot_records_paths_another_clone_can_resolve(rf):
    """v1's verification table recorded the checkpoint paths it hashed as
    ABSOLUTE, so a frozen v1 manifest reproduces its design_sha256 only under the
    filesystem root it was frozen at; from any other clone every role reads as
    absent and the rebuild disagrees.  A pre-registration nobody else can verify
    is a claim rather than a record.

    v2 records each path relative to the dataset root and resolves it back
    through ``resolve_recorded_path``, which still accepts the absolute shape --
    so the superseded v1 manifests keep verifying where they were frozen.
    """
    # The real requirement table and the real verifier, not the hermetic
    # pilot's: _stub_tree replaces verify_checkpoints_v2, and the recording
    # convention is exactly what is under test here.  Every role is checked
    # whether or not the weights exist, because an ABSENT path is recorded too
    # and is the one a later clone has to resolve.
    req = rf.required_checkpoints_v2("ppubench", "fs_001", [17, 42, 123],
                                     [17, 42, 123], [17])
    verification = rf.verify_checkpoints_v2(req)
    seen = 0
    for table in ("present", "absent"):
        for label, recorded, _digest in rf._frozen_checkpoint_digests(
                verification[table]):
            seen += 1
            assert not Path(recorded).is_absolute(), (label, recorded)
            assert rf.resolve_recorded_path(recorded).is_absolute()
    assert seen == verification["n_files_required"] == \
        sum(spec["n_files_required"] for spec in req["roles"].values()), \
        "a required file that is recorded nowhere is a file nothing verifies"
    for role, spec in req["roles"].items():
        for key in rf.CHECKPOINT_FILE_KEYS_V2:
            if spec.get(key):
                assert not Path(spec[key]).is_absolute(), (role, key)

    # The same resolver still reads v1's absolute shape, which is what lets the
    # superseded manifests be verified at all rather than written off as stale.
    outside = str(rf.DATASET_ROOT.parent / "elsewhere" / "adapter.safetensors")
    assert rf.resolve_recorded_path(outside) == Path(outside)
    assert rf._rel(outside) == outside, \
        "a path outside the dataset root has no relative form, so _rel must " \
        "leave it absolute rather than inventing one"


def test_the_existing_router_is_read_and_the_others_are_training_work(
        rf, tmp_path):
    design, _ = _design_v2(rf, tmp_path)
    roles = design["checkpoint_requirements"]["roles"]
    assert roles[f"router_g__seed{rf.EXISTING_ROUTER_SEED}"][
        "exists_already"] is True
    assert roles[f"router_g__seed{rf.EXISTING_ROUTER_SEED}"]["role"].startswith(
        "the frozen router, read not retrained")
    for seed in design["router_seeds"]:
        if seed == rf.EXISTING_ROUTER_SEED:
            continue
        assert f"g_seed_{seed}" in roles[f"router_g__seed{seed}"]["adapter"]
        assert "identical frozen protocol" in roles[f"router_g__seed{seed}"]["role"]


def test_the_seed_prediction_paths_differ_per_seed(rf):
    """One file per router seed is the whole repair: a shared cache replayed
    under three names is one router measured once."""
    paths = {s: str(rf.router_prediction_path("salmu", s))
             for s in (17, 42, 123)}
    assert len(set(paths.values())) == 3
    assert "e2e_post" in paths[17], \
        "the frozen router's predictions are the route's own committed cache"
    for seed in (42, 123):
        assert f"g_seed_{seed}" in paths[seed]


def test_a_router_seed_without_a_prediction_file_is_refused(rf, tmp_path,
                                                            monkeypatch):
    _, man = _design_v2(rf, tmp_path)
    monkeypatch.setattr(rf, "router_prediction_path",
                        lambda ds, s: tmp_path / "nope" / f"g_seed_{s}.json")
    with pytest.raises(RuntimeError, match="a label, not a factor"):
        rf.load_router_predictions("ppubench", 42, rf.held_out_uris(man))


def test_a_prediction_file_missing_a_held_out_image_is_refused(rf, tmp_path,
                                                               monkeypatch):
    """An image silently missing from the confusion matrix would make the
    factorization agree with itself over the rows that happened to exist."""
    _, man = _design_v2(rf, tmp_path)
    rows = [{"identity_id": it["identity_id"], "image_uri": it["image_uri"],
             "split": it["split"],
             "g_raw_text": man["code_of"][it["identity_id"]],
             "pred_code": man["code_of"][it["identity_id"]],
             "code_correct": True}
            for it in man["items"] if it["split"] == "test"]
    dropped = rows[:-1]
    path = tmp_path / "g_seed_42.json"
    rf.atomic_write_json(path, {"rows": dropped})
    monkeypatch.setattr(rf, "router_prediction_path", lambda ds, s: path)
    held_out = rf.held_out_uris(man)
    missing = rows[-1]["image_uri"]
    assert missing in held_out
    # the refusal has to name the IMAGE that is missing: a coverage check that
    # only counted rows would pass on a file covering a different image set
    with pytest.raises(RuntimeError, match=re.escape(missing)):
        rf.load_router_predictions("ppubench", 42, held_out)


def test_a_prediction_row_without_an_image_cannot_be_attached(rf, tmp_path,
                                                              monkeypatch):
    _, man = _design_v2(rf, tmp_path)
    path = tmp_path / "g_seed_42.json"
    rf.atomic_write_json(path, {"rows": [{"pred_code": "RID_002"}]})
    monkeypatch.setattr(rf, "router_prediction_path", lambda ds, s: path)
    with pytest.raises(RuntimeError, match="no image_uri"):
        rf.load_router_predictions("ppubench", 42, rf.held_out_uris(man))


def test_the_natural_cells_are_the_real_factorial(rf, tmp_path):
    design, _ = _design_v2(rf, tmp_path)
    natural = _cells(design, "natural")
    assert len(natural) == len(design["router_seeds"]) * len(
        design["edit_seeds"])
    assert {(c["router_seed"], c["edit_seed"]) for c in natural} == {
        (g, h) for g in design["router_seeds"] for h in design["edit_seeds"]}
    assert {c["cell_id"] for c in natural} == {
        f"natural__g{g}__h{h}" for g in design["router_seeds"]
        for h in design["edit_seeds"]}


def test_intervention_cells_are_not_multiplied_by_router_seed(rf, tmp_path):
    """Forced-code interventions do not involve g, so evaluating them per router
    seed would run the identical call three times and file it as three
    observations -- which is the defect item 3 names."""
    design, _ = _design_v2(rf, tmp_path)
    assert len(design["router_seeds"]) == 3
    assert len(_cells(design, "intervention")) == len(design["edit_seeds"]) == 3
    for cell in _cells(design, "intervention"):
        assert "router_seed" not in cell
        assert cell["phase"] == "RF1H"
    assert design["rows_are_not_duplicated_across_router_seeds"]
    # and the counts follow from what each kind depends on, not from one product
    assert design["n_cells"] == 3 + 9 + 1 + 1


def test_the_cell_counts_are_checked_against_the_factors(rf, tmp_path,
                                                         monkeypatch):
    """A cell count that does not follow from what the measurement depends on is
    refused at freeze time rather than filed as a smaller design."""
    man = _disk_manifest(tmp_path)
    _stub_tree(rf, monkeypatch)
    real = rf.build_design_v2

    def broken(*a, **kw):
        d = real(*a, **kw)
        d["cells"] = [c for c in d["cells"] if c["kind"] != "natural"]
        d["cells_by_kind"] = {"intervention": ["intervention__h17"],
                              "direct": ["direct__d17"], "hybrid": ["hybrid__d17"]}
        return d

    monkeypatch.setattr(rf, "build_design_v2", broken)
    with pytest.raises(RuntimeError, match="the factors imply"):
        rf.build_pilot_preregistration_v2(
            "ppubench", "fs_001", ["001"], [17, 42, 123], [17, 42, 123], [17],
            man=man, images=rf.held_out_images(man), selection=dict(SELECTION))


def test_the_router_budget_goes_where_it_can_be_used(rf, tmp_path):
    """Item 3: PPUBench's test bytes are its train bytes, so an extra router
    there could only measure seed stability on repeated content.  The budget is
    spent on the dataset that has genuinely held-out images."""
    duplicated = _disk_manifest(tmp_path, duplicate_test_bytes=True)
    policy = rf.pilot_seed_policy_v2("ppubench", man=duplicated)
    assert policy["router_seeds"] == [rf.EXISTING_ROUTER_SEED]
    assert policy["direct_seeds"] == [rf.EXISTING_ROUTER_SEED]
    assert policy["edit_seeds"] == [17, 42, 123], \
        "the mediation verdict is what every dataset can support"
    assert "cannot establish held-out routing" in policy["router_seed_budget"]

    held_out = _disk_manifest(tmp_path / "b")
    assert rf.pilot_seed_policy_v2("salmu", man=held_out)["router_seeds"] == \
        [17, 42, 123]


def test_training_a_router_a_dataset_cannot_use_is_refused(rf, tmp_path):
    duplicated = _disk_manifest(tmp_path, duplicate_test_bytes=True)
    with pytest.raises(RuntimeError, match="router-training budget"):
        rf.check_seed_budget("ppubench", [17, 42], [17], man=duplicated)
    with pytest.raises(RuntimeError, match="direct seeds"):
        rf.check_seed_budget("ppubench", [17], [17, 42], man=duplicated)
    # the same request on a dataset with held-out content is fine
    rf.check_seed_budget("salmu", [17, 42, 123], [17, 42, 123],
                         man=_disk_manifest(tmp_path / "b"))


def test_router_accuracy_is_gated_per_seed_not_on_the_mean(rf, tmp_path):
    """Averaging a bad router with two good ones is how a factor stops being a
    factor: the mean clears the floor while one real router does not."""
    design, man = _design_v2(rf, tmp_path)
    s = _all_scored(rf, design, man)
    accs = {17: 1.0, 42: 0.75, 123: 1.0}
    gates = rf.evaluate_gates_v2(s["intervention"], s["natural"], s["direct"],
                                 s["hybrid"], accs, E2E_OK)
    gate = gates["router_held_out_accuracy"]
    assert gate["passed"] is False
    assert gate["aggregation"] == "every seed, not the mean"
    assert sum(accs.values()) / len(accs) > gate["threshold"], \
        "the mean clears the floor, so a mean-based gate would have passed"
    assert "42" in gate["failed_because"]


def test_a_router_seed_that_reported_nothing_fails_the_gate(rf, tmp_path):
    design, man = _design_v2(rf, tmp_path)
    s = _all_scored(rf, design, man)
    gates = rf.evaluate_gates_v2(s["intervention"], s["natural"], s["direct"],
                                 s["hybrid"], {}, E2E_OK)
    assert gates["router_held_out_accuracy"]["passed"] is False
    assert gates["router_held_out_accuracy"]["value"] is None
    assert "no denominator" in gates["router_held_out_accuracy"]["failed_because"]


def test_rf1g_refuses_to_retrain_the_frozen_router(rf):
    """Retraining g_17 would replace the router the route was established with,
    and the pilot would then measure a route nobody froze."""
    with pytest.raises(RuntimeError, match="never retrained"):
        rf.phase_rf1g("ppubench", rf.EXISTING_ROUTER_SEED, device="cpu")


# ==========================================================================
# item 7 -- P_h is the empirical hard response, not a score sum
# ==========================================================================

def test_p_h_is_the_empirical_hard_response_matrix(rf, tmp_path):
    design, man = _design_v2(rf, tmp_path)
    rows = _scored(rf, design, man, "intervention")
    H = rf.hard_response_matrix(rows, design["vocab"])
    assert H["definition"] == "H_e(c, y) = 1{parse(h_e(c)) = y}"
    assert H["n_codes"] == len(man["identity_ids"])
    forgotten = man["code_of"]["001"]
    assert H["matrix"][forgotten][rf.DELETED_LABEL] == 1
    assert sum(H["matrix"][forgotten].values()) == 1
    retained = man["code_of"]["002"]
    assert H["matrix"][retained][man["alias_of"]["002"]] == 1
    assert "not probabilities" in H["why_not_score_sums"]


def test_a_score_sum_is_never_substituted_for_p_h(rf, tmp_path):
    """Item 7.  A score sum over teacher-forced full sequences has no termination
    event and no cross-candidate normalization, so its terms overlap and it is
    not the probability of anything."""
    design, man = _design_v2(rf, tmp_path)
    assert design["p_h_is_not_a_score_sum"]
    assert design["score_sum_semantics"]["is_not"].startswith("a probability")
    assert design["score_sum_semantics"]["name_used_everywhere"] == \
        "candidate_score_sum"
    H = rf.hard_response_matrix(_scored(rf, design, man, "intervention"),
                                design["vocab"])
    for code, row in H["matrix"].items():
        assert sorted(set(row.values())) == [0, 1], \
            f"H_e({code}) is not a hard 0/1 response"


def test_a_duplicate_code_in_h_e_is_a_contradiction_not_a_replicate(rf,
                                                                    tmp_path):
    """h(c) is one call, so a second row for the same code is not a repeat
    measurement -- it is two answers to one question."""
    design, man = _design_v2(rf, tmp_path)
    rows = _scored(rf, design, man, "intervention")
    with pytest.raises(RuntimeError, match="one call"):
        rf.hard_response_matrix(rows + [dict(rows[0])], design["vocab"])


def test_h_e_from_zero_rows_is_refused(rf, tmp_path):
    """An empty matrix would predict 0 for every image and read as a finding.

    ``hard_response_matrix`` builds the post-edit H_e and the pre-edit H_base, so
    the refusal is stated over rows rather than over intervention rows: the
    baseline cell has no intervention rows at all and is still a matrix.
    """
    design, _ = _design_v2(rf, tmp_path)
    with pytest.raises(RuntimeError, match="zero rows"):
        rf.hard_response_matrix([], design["vocab"])


def test_rows_that_are_no_forced_code_measurement_build_no_matrix(rf, tmp_path):
    """Rows present but none of them a forced-code row is refused by name.

    Filtering on a hardcoded pair of condition names would have dropped every
    baseline row and then reported an empty matrix as a finding about the edit,
    so the refusal names the conditions it was asked for.
    """
    design, man = _design_v2(rf, tmp_path)
    rows = _scored(rf, design, man, "natural")
    with pytest.raises(RuntimeError, match="forced-code conditions"):
        rf.hard_response_matrix(rows, design["vocab"])


def test_an_unparseable_h_response_gets_its_own_column(rf, tmp_path):
    """An unparseable response means the composition produces NO label, which is
    different from producing a wrong one; folding it into the wrong-label columns
    would hide a parse failure inside an accuracy figure."""
    design, man = _design_v2(rf, tmp_path)
    rows = _scored(rf, design, man, "intervention")
    field = rf.generation_field(rows[0]["condition"])
    rows[0] = {**rows[0], field: "!! nothing recognizable !!"}
    rescored, _ = rf.score_rows_v2(rows, design["vocab"])
    assert rescored[0]["unparseable"] is True
    H = rf.hard_response_matrix(rescored, design["vocab"])
    code = rows[0]["forced_code"]
    assert H["matrix"][code][rf.UNPARSEABLE] == 1
    assert H["unparseable_key"] == rf.UNPARSEABLE


def test_a_label_outside_the_vocabulary_is_refused(rf, tmp_path):
    design, man = _design_v2(rf, tmp_path)
    rows = _scored(rf, design, man, "intervention")
    rows[0]["parsed_label"] = "NotACandidate"
    rows[0]["usable"] = True
    rows[0]["unparseable"] = False
    with pytest.raises(RuntimeError, match="not in the candidate vocabulary"):
        rf.hard_response_matrix(rows, design["vocab"])


def test_the_required_label_is_keyed_by_the_image_not_the_code(rf, tmp_path):
    """Keying it by the code would make every routing mistake correct by
    construction and predict 1.0 for any router whatsoever -- the quantity the
    decomposition exists to explain would disappear."""
    design, man = _design_v2(rf, tmp_path)
    H = rf.hard_response_matrix(_scored(rf, design, man, "intervention"),
                                design["vocab"])
    rows = _scored(rf, design, man, "natural")
    confusion = rf.router_confusion(rows)
    pred = rf.predicted_e2e_from_empirical(rows, H, confusion)
    assert pred["required_label_is_keyed_by"] == "the IMAGE, not the code"
    assert pred["formula"] == "sum_c P_g(c|X) * H_e(c, Y*(X))"
    for entry in pred["per_image"]:
        assert entry["required_label"] == \
            man["alias_of"][entry["identity_id"]] or \
            entry["required_label"] == rf.DELETED_LABEL


def test_a_misrouted_image_costs_the_composition(rf, tmp_path):
    """The point of the noisy-router condition: a routing error has to show up in
    the prediction, or the factorization explains nothing."""
    design, man = _design_v2(rf, tmp_path)
    H = rf.hard_response_matrix(_scored(rf, design, man, "intervention"),
                                design["vocab"])
    perfect = _scored(rf, design, man, "natural")
    one = [r["image_uri"] for r in perfect
           if r["image_identity_id"] == "002"][:1]
    misrouted = _scored(rf, design, man, "natural", wrong_routes=one)
    p_perfect = rf.predicted_e2e_from_empirical(
        perfect, H, rf.router_confusion(perfect))
    p_wrong = rf.predicted_e2e_from_empirical(
        misrouted, H, rf.router_confusion(misrouted))
    assert p_perfect["predicted"] == 1.0
    assert p_wrong["predicted"] < p_perfect["predicted"]
    assert rf.router_confusion(misrouted)["n_unroutable"] == 0


def test_a_router_that_emitted_nothing_is_counted_not_dropped(rf, tmp_path):
    design, man = _design_v2(rf, tmp_path)
    rows = _scored(rf, design, man, "natural")
    rows[0] = {**rows[0], "routed_code": None, "routable": False,
               "h_raw_text": None}
    rescored, missing = rf.score_rows_v2(rows, design["vocab"])
    assert not missing, "an unroutable image is a routing failure, not a gap"
    assert rescored[0]["correct"] is False
    assert "unscored_because" in rescored[0]
    confusion = rf.router_confusion(rescored)
    assert confusion["n_unroutable"] == 1
    assert confusion["n_images"] == len(rescored)
    assert confusion["unroutable_counted_not_dropped"]


def test_a_router_code_h_e_has_no_row_for_is_refused(rf, tmp_path):
    """H_e must cover every code the router can emit, or the prediction silently
    omits one and looks better than the system is."""
    design, man = _design_v2(rf, tmp_path)
    H = rf.hard_response_matrix(_scored(rf, design, man, "intervention"),
                                design["vocab"])
    del H["matrix"][man["code_of"]["003"]]
    rows = _scored(rf, design, man, "natural")
    with pytest.raises(RuntimeError, match="H_e has no row for"):
        rf.predicted_e2e_from_empirical(rows, H, rf.router_confusion(rows))


def test_a_leaky_mediator_breaks_the_factorization(rf, tmp_path):
    """The factorization is a TEST OF THE HARD MEDIATOR, not a tautology.

    Under greedy decoding an h that is a function of the code alone makes the
    composed system and sum_c P_g(c|X) H_e(c, Y*(X)) coincide exactly.  An h
    that also answers from the image does not: on a MISROUTED image the code says
    one thing and the image says another, so the observation and the prediction
    come apart by exactly the number of images where they disagree.  With a
    perfect router a leak is invisible, which is why the noisy-router condition
    is the informative one.
    """
    design, man = _design_v2(rf, tmp_path)
    H = rf.hard_response_matrix(_scored(rf, design, man, "intervention"),
                                design["vocab"])
    misrouted = [r["image_uri"] for r in _scored(rf, design, man, "natural")
                 if r["image_identity_id"] == "002"][:1]
    assert misrouted
    honest = _scored(rf, design, man, "natural", wrong_routes=misrouted)
    leaky = _scored(rf, design, man, "natural", wrong_routes=misrouted,
                    behaves="leaks")

    def observed(rows):
        return sum(1 for r in rows if r["correct"]) / len(rows)

    def predicted(rows):
        return rf.predicted_e2e_from_empirical(
            rows, H, rf.router_confusion(rows))["predicted"]

    assert observed(honest) == predicted(honest) == 11 / 12, \
        "a hard mediator factorizes exactly"
    assert observed(leaky) == 1.0, "a leaky h gets the misrouted image right"
    assert predicted(leaky) == 11 / 12, \
        "the prediction still follows the CODE, because H_e is keyed by code"
    assert abs(observed(leaky) - predicted(leaky)) > \
        rf.GATE_THRESHOLDS_V2["max_abs_e2e_prediction_error"], \
        "the gap must exceed the frozen tolerance or the gate proves nothing"

    gates = rf.evaluate_gates_v2(
        _scored(rf, design, man, "intervention"), leaky,
        _scored(rf, design, man, "direct"),
        _scored(rf, design, man, "hybrid"), {17: 1.0},
        {"observed": observed(leaky), "predicted": predicted(leaky),
         "n_images": len(leaky)})
    assert gates["e2e_matches_the_routing_composition"]["passed"] is False


# ==========================================================================
# item 5 -- three verdicts, and no omnibus boolean
# ==========================================================================

def test_every_gate_feeds_exactly_one_verdict(rf, tmp_path):
    """And the gates produced are exactly the gates mapped: a gate outside the
    map would be computed and then ignored, and a mapped gate that was never
    computed would make its verdict refuse."""
    design, man = _design_v2(rf, tmp_path)
    s = _all_scored(rf, design, man)
    gates = rf.evaluate_gates_v2(s["intervention"], s["natural"], s["direct"],
                                 s["hybrid"], {17: 1.0}, E2E_OK)
    assert set(gates) == set(rf.GATE_TO_VERDICT)
    for verdict in rf.GATE_TO_VERDICT.values():
        assert verdict in rf.VERDICT_NAMES
    for name in rf.VERDICT_NAMES:
        assert [g for g, v in rf.GATE_TO_VERDICT.items() if v == name], \
            f"verdict {name} has no gate, so it could never be anything but " \
            f"vacuously true"
    applicability = rf.gate_applicability_v2(rf.image_content_audit(man),
                                             "ppubench")
    assert applicability["n_gates_total"] == len(rf.GATE_TO_VERDICT) == len(gates)
    assert rf.GATE_TO_VERDICT["direct_image_accuracy"] == "mediation"
    assert rf.GATE_TO_VERDICT["router_held_out_accuracy"] == \
        "routing_reliability"
    assert rf.GATE_TO_VERDICT["e2e_matches_the_routing_composition"] == \
        "routing_factorization"


def test_every_threshold_is_consumed_by_some_gate(rf, tmp_path):
    """A threshold nothing reads is not a gate: it is decoration that survives a
    change to the measurement while still describing the old one.  Each one is
    set to a value nothing can satisfy and some gate must flip."""
    design, man = _design_v2(rf, tmp_path)
    s = _all_scored(rf, design, man)

    def gates_with(**over):
        th = dict(rf.GATE_THRESHOLDS_V2)
        th.update(over)
        return rf.evaluate_gates_v2(s["intervention"], s["natural"],
                                    s["direct"], s["hybrid"], {17: 1.0}, E2E_OK,
                                    thresholds=th)

    base = gates_with()
    assert all(g["passed"] for g in base.values()), \
        "the fixture must produce an all-pass baseline or nothing can flip"
    for key, value in rf.GATE_THRESHOLDS_V2.items():
        absurd = -1.0 if key.startswith("max_") else value + 10
        moved = [n for n, g in gates_with(**{key: absurd}).items()
                 if g["passed"] != base[n]["passed"]]
        assert moved, f"threshold {key} is read by no gate"


def test_there_is_no_omnibus_verdict(rf, tmp_path):
    design, man = _design_v2(rf, tmp_path)
    s = _all_scored(rf, design, man)
    gates = rf.evaluate_gates_v2(s["intervention"], s["natural"], s["direct"],
                                 s["hybrid"], {17: 1.0}, E2E_OK)
    verdicts = rf.verdicts_from_gates(
        gates, rf.gate_applicability_v2(rf.image_content_audit(man),
                                        "ppubench"))
    assert "passed" not in verdicts
    assert verdicts["no_omnibus_verdict"]
    assert {k for k in verdicts if k.endswith("_pass")} == {
        f"{n}_pass" for n in rf.VERDICT_NAMES}
    for name in rf.VERDICT_NAMES:
        assert verdicts[name]["state"] in ("pass", "fail", rf.NOT_ESTABLISHED)
        assert verdicts[name]["thresholds_unchanged"] == \
            dict(rf.GATE_THRESHOLDS_V2)


def test_a_routing_failure_does_not_cancel_a_mediation_success(rf, tmp_path):
    """The whole reason for item 5: SALMU's router misses the floor and its
    mediation is nonetheless supported, and one boolean would have reported only
    the failure."""
    design, man = _design_v2(rf, tmp_path)
    s = _all_scored(rf, design, man)
    gates = rf.evaluate_gates_v2(s["intervention"], s["natural"], s["direct"],
                                 s["hybrid"], {17: 0.8056}, E2E_OK)
    verdicts = rf.verdicts_from_gates(
        gates, rf.gate_applicability_v2(rf.image_content_audit(man),
                                        "ppubench"))
    assert gates["router_held_out_accuracy"]["passed"] is False
    assert verdicts["routing_reliability_pass"] is False
    assert verdicts["mediation_pass"] is True
    assert verdicts["routing_factorization_pass"] is True
    assert verdicts["routing_reliability"]["failed_gates"] == \
        ["router_held_out_accuracy"]
    assert verdicts["mediation"]["failed_gates"] == []


def test_an_unsupportable_verdict_is_not_established(rf, tmp_path):
    """Neither True (an unmeasured success) nor False (an unobserved failure):
    a question the dataset cannot answer is recorded as unanswerable, with the
    threshold unchanged and the gate values still reported."""
    man = _disk_manifest(tmp_path, duplicate_test_bytes=True)
    applicability = rf.gate_applicability_v2(rf.image_content_audit(man),
                                             "ppubench")
    assert applicability["held_out_image_content_exists"] is False
    assert applicability["verdict_support"]["routing_reliability"][
        "supportable"] is False
    design, _ = _design_v2(rf, tmp_path, man=man, router_seeds=[17],
                           direct_seeds=[17])
    s = _all_scored(rf, design, man)
    gates = rf.evaluate_gates_v2(s["intervention"], s["natural"], s["direct"],
                                 s["hybrid"], {17: 1.0}, E2E_OK)
    verdicts = rf.verdicts_from_gates(gates, applicability)
    assert verdicts["routing_reliability_pass"] is None
    assert verdicts["routing_factorization_pass"] is None
    assert verdicts["mediation_pass"] is True
    assert verdicts["routing_reliability"]["state"] == rf.NOT_ESTABLISHED
    assert verdicts["routing_reliability"]["thresholds_unchanged"][
        "min_router_held_out_accuracy"] == 0.90, \
        "not established must not mean the threshold was lowered or waived"
    assert gates["router_held_out_accuracy"]["passed"] is True, \
        "the gate is still computed and reported; only the VERDICT is withheld"


def test_the_expected_verdicts_match_the_preregistered_table(rf, tmp_path):
    """Item 5's table, and the fact that it is derived: no dataset is named in a
    branch, so replacing a dataset's images changes the expectation instead of
    leaving a stale one behind."""
    held_out = _disk_manifest(tmp_path / "a")
    duplicated = _disk_manifest(tmp_path / "b", duplicate_test_bytes=True)
    good = rf.gate_applicability_v2(rf.image_content_audit(held_out),
                                    "ppubench")
    bad = rf.gate_applicability_v2(rf.image_content_audit(duplicated),
                                   "ppubench")
    assert good["expected_verdicts"] == {
        "mediation": "supported", "routing_reliability": "measured",
        "routing_factorization": "measured"}
    assert bad["expected_verdicts"] == {
        "mediation": "supported",
        "routing_reliability": rf.NOT_ESTABLISHED,
        "routing_factorization": f"{rf.NOT_ESTABLISHED} on unseen content"}
    assert good["expected_verdicts_are_derived"]
    # a recorded router below the floor is expected to fail, whatever the dataset
    noisy = rf.gate_applicability_v2(rf.image_content_audit(held_out), "salmu")
    assert noisy["expected_verdicts"]["routing_reliability"].startswith(
        "measured, expected to FAIL")
    assert "0.8056" in noisy["expected_verdicts"]["routing_reliability"]
    assert "scientifically informative" in \
        noisy["expected_verdicts"]["routing_factorization"]


def test_a_verdict_computed_over_part_of_its_gates_is_refused(rf, tmp_path):
    design, man = _design_v2(rf, tmp_path)
    s = _all_scored(rf, design, man)
    gates = rf.evaluate_gates_v2(s["intervention"], s["natural"], s["direct"],
                                 s["hybrid"], {17: 1.0}, E2E_OK)
    applicability = rf.gate_applicability_v2(rf.image_content_audit(man),
                                             "ppubench")
    del gates["direct_image_accuracy"]
    with pytest.raises(RuntimeError, match="missing gates"):
        rf.verdicts_from_gates(gates, applicability)


def test_the_e2e_prediction_is_missing_means_the_gate_fails(rf, tmp_path):
    design, man = _design_v2(rf, tmp_path)
    s = _all_scored(rf, design, man)
    gates = rf.evaluate_gates_v2(s["intervention"], s["natural"], s["direct"],
                                 s["hybrid"], {17: 1.0}, None)
    gate = gates["e2e_matches_the_routing_composition"]
    assert gate["passed"] is False
    assert "never evaluated" in gate["failed_because"]



# ==========================================================================
# item 6 -- the unit of inference is the identity, not the image
# ==========================================================================

def test_the_primary_bootstrap_is_identity_level_and_image_is_sensitivity(rf,
                                                                          tmp_path):
    """Three images of one person share an identity, the router that was trained
    on that person and the edited h, so they are not independent.  v1 selected
    image level on SALMU as "a real refinement"; it is not a refinement, it is a
    narrower interval than the data supports."""
    man = _disk_manifest(tmp_path)
    plan = rf.bootstrap_plan_for(rf.image_content_audit(man), "salmu")
    assert plan["primary"]["level"] == "identity"
    assert plan["sensitivity"]["level"] == "image"
    assert plan["primary_n_clusters"] == 4
    assert plan["sensitivity_n_clusters"] == 12
    assert plan["repeated_intervention_rows_are_never_independent"]
    assert plan["n_resamples"] == 2000 and plan["seed"] == 17
    assert plan["alpha"] == 0.05
    assert "not independent" in plan["primary"]["why"]
    assert "never instead of it" in plan["sensitivity"]["why"]


def test_a_degenerate_image_level_says_it_restates_the_primary(rf, tmp_path):
    man = _disk_manifest(tmp_path, n_test=1)
    plan = rf.bootstrap_plan_for(rf.image_content_audit(man), "salmu")
    assert plan["image_level_is_degenerate"] is True
    assert plan["primary_n_clusters"] == plan["sensitivity_n_clusters"] == 4
    assert "restates the primary one" in \
        plan["sensitivity_is_not_a_refinement_here"]


def test_intervals_are_reported_at_both_levels_side_by_side(rf, tmp_path):
    design, man = _design_v2(rf, tmp_path)
    plan = rf.bootstrap_plan_for(rf.image_content_audit(man), "salmu")
    rows = _scored(rf, design, man, "natural")
    both = rf.bootstrap_both_levels(rows, rf._mean_correct, "e2e_accuracy", plan)
    assert both["primary"]["n_clusters"] == 4
    assert both["sensitivity"]["n_clusters"] == 12
    assert both["primary"]["point_estimate"] == \
        both["sensitivity"]["point_estimate"] == 1.0
    assert both["which_to_believe"].startswith("the primary identity-level")
    # and the wider interval is the identity-level one, because its units are the
    # independent ones: the sensitivity interval is never the safer choice
    assert (both["primary"]["ci_high"] - both["primary"]["ci_low"]) >= \
        (both["sensitivity"]["ci_high"] - both["sensitivity"]["ci_low"])


def test_the_image_cluster_is_the_bytes_not_the_filename(rf, tmp_path):
    """PPUBench has twelve held-out filenames over four distinct byte-values.
    Clustering by name would resample one picture three times and call the copies
    independent, which narrows every interval and makes a noisy result look
    precise."""
    man = _disk_manifest(tmp_path, duplicate_test_bytes=True)
    audit = rf.image_content_audit(man)
    assert audit["by_split"]["test"]["n_items"] == 12
    assert audit["by_split"]["test"]["n_distinct_bytes"] == 4
    design = rf.build_design_v2(
        "ppubench", "fs_001", ["001"], [17], [17, 42, 123], [17], man=man,
        images=rf.held_out_images(man),
        image_sha_by_uri=audit["images_by_uri_sha256"])
    rows = _rows(design, "natural")
    assert len({r["image_uri"] for r in rows}) == 12
    assert len({r["image_cluster_id"] for r in rows}) == 4, \
        "the image-level cluster must be the content digest"
    assert len(rf.cluster_units_v2(rows, "identity")) == 4
    assert len(rf.cluster_units_v2(rows, "image")) == 4


def test_a_row_cluster_count_that_disagrees_with_the_audit_is_refused(
        rf, tmp_path, monkeypatch):
    """The audit counts clusters by image CONTENT and the design counts them by
    the key each row carries.  At a level the bootstrap will use, the two have to
    agree or the interval is computed over more units than exist.

    The disagreement needs duplicate bytes: with twelve distinct images the
    twelve URIs and the twelve digests are twelve clusters either way, and a
    design that dropped the digest table would look correct.  Duplicate held-out
    bytes also trip the seed budget, which refuses first, so the pilot is frozen
    with the single router seed that budget allows.
    """
    man = _disk_manifest(tmp_path, duplicate_test_bytes=True)
    _stub_tree(rf, monkeypatch)
    real = rf.build_design_v2
    monkeypatch.setattr(rf, "build_design_v2",
                        lambda *a, **kw: real(*a, **{**kw,
                                                     "image_sha_by_uri": None}))
    with pytest.raises(RuntimeError, match="cluster"):
        rf.build_pilot_preregistration_v2(
            "ppubench", "fs_001", ["001"], [17], [17, 42, 123], [17],
            man=man, images=rf.held_out_images(man), selection=dict(SELECTION))


def test_one_cluster_at_either_level_is_refused_not_reported(rf, tmp_path):
    """A bootstrap over a single cluster restates the point estimate as an
    interval of width zero, which reads as a confidence statement and is not
    one."""
    man = _disk_manifest(tmp_path, identity_ids=("001", "002"))
    plan_rows = _rows(rf.build_design_v2(
        "ppubench", "fs_001", ["001"], [17], [17], [17], man=man,
        images=rf.held_out_images(man)), "natural")
    with pytest.raises(RuntimeError, match="cluster"):
        rf.cluster_bootstrap_v2([r for r in plan_rows
                                 if r["identity_cluster_id"] == "001"],
                                rf._mean_correct, "identity", "e2e",
                                n_resamples=5)


def test_a_row_that_belongs_to_no_cluster_is_refused(rf, tmp_path):
    """A row with no cluster key is silently dropped from every resample, which
    narrows the interval without saying so."""
    design, man = _design_v2(rf, tmp_path)
    rows = [dict(r) for r in _scored(rf, design, man, "natural")]
    rows[0]["image_cluster_id"] = None
    with pytest.raises(RuntimeError, match="cannot be assigned"):
        rf.cluster_bootstrap_v2(rows, rf._mean_correct, "image", "e2e",
                                n_resamples=5)


def test_delta_route_is_over_codes_and_says_when_no_interval_is_possible(
        rf, tmp_path):
    """With a hard mediator a forced-code row has no image, so the resampling
    unit is the CODE.  A single-target forget set puts exactly one code on the
    forgotten arm, and the design says so up front rather than letting the
    bootstrap discover it."""
    design, man = _design_v2(rf, tmp_path)
    rows = _scored(rf, design, man, "intervention")
    delta = rf.delta_route_v2(rows, design["forget_identity_ids"])
    assert delta["unit_of_analysis"].startswith("the CODE")
    assert delta["delta_route"] == 1.0
    assert delta["arms"]["forgotten"]["n_codes"] == 1
    assert delta["arms"]["retained"]["n_codes"] == 3
    assert delta["interval_possible_on_forgotten_arm"] is False
    assert delta["interval_possible_on_retained_arm"] is True
    assert "single-target forget set" in delta["single_target_limitation"]
    assert delta["rows_are_codes_not_images"]
    for name in ("natural_mediated", "direct_control", "hybrid_conflict_probe",
                 "retained_control"):
        assert name in delta["excluded_and_why"]
    assert "count one measurement twice" in \
        delta["excluded_and_why"]["retained_control"]


def test_delta_route_refuses_an_arm_with_no_codes(rf, tmp_path):
    design, man = _design_v2(rf, tmp_path)
    rows = [r for r in _scored(rf, design, man, "intervention")
            if r["arm"] == "retained"]
    with pytest.raises(RuntimeError, match="both arms"):
        rf.delta_route_v2(rows, design["forget_identity_ids"])
    with pytest.raises(RuntimeError, match="no denominator"):
        rf.delta_route_v2([], design["forget_identity_ids"])


def test_a_simultaneous_forget_set_would_support_an_interval(rf, tmp_path):
    """The limitation is a property of the SINGLE-TARGET pilot, not of the
    statistic: two forgotten codes give the forgotten arm a denominator."""
    man = _disk_manifest(tmp_path)
    design = rf.build_design_v2(
        "ppubench", "fs_001-002", ["001", "002"], [17], [17], [17], man=man,
        images=rf.held_out_images(man))
    rows = _scored(rf, design, man, "intervention")
    delta = rf.delta_route_v2(rows, ["001", "002"])
    assert delta["arms"]["forgotten"]["n_codes"] == 2
    assert delta["interval_possible_on_forgotten_arm"] is True
    assert delta["single_target_limitation"] is None


# ==========================================================================
# item 4 -- the phases, and the machinery that makes them resumable
# ==========================================================================

def _write_routers(rf, prereg, man, tmp, monkeypatch, misroutes=None):
    """A per-seed prediction file on disk, and a patched path that points at it.

    Written as real files rather than patched into the loader, so
    ``load_router_predictions`` -- which refuses a seed that has no file of its
    own -- is exercised too.

    ``misroutes[k]`` is the set of held-out positions seed ``k`` gets wrong.
    The default misroutes a DIFFERENT single image per seed rather than a
    different NUMBER of images: at twelve held-out images the only accuracies
    that clear the 0.90 floor are 12/12 and 11/12, so three seeds cannot have
    three distinct accuracies and still be three passable routers.  Distinct
    mistakes are the stronger witness anyway -- two routers that agree on every
    image are the same router whatever their accuracy says.
    """
    root = tmp / "routers"
    made = {}
    for k, seed in enumerate(prereg["router_seeds"]):
        bad = set(misroutes[k]) if misroutes else {k}
        rows, j = [], 0
        for it in man["items"]:
            own = man["code_of"][it["identity_id"]]
            code, raw = own, own
            if it["split"] == "test":
                if j in bad:
                    other = next(i for i in man["identity_ids"]
                                 if i != it["identity_id"])
                    code, raw = man["code_of"][other], "not sure"
                j += 1
            rows.append({"identity_id": it["identity_id"],
                         "image_uri": str(it["image_uri"]),
                         "split": it["split"],
                         "image_sha256": _sha(it["image_uri"]),
                         "g_raw_text": raw, "pred_code": code,
                         "code_correct": code == own})
        path = root / f"g_seed_{seed}" / "held_out_predictions.json"
        rf.atomic_write_json(path, {"rows": rows, "router_seed": seed})
        made[seed] = path
    monkeypatch.setattr(rf, "router_prediction_path", lambda ds, s: made[s])
    return made


def _materialize_roles(rf, prereg, tmp, routers=None):
    """Real bytes for every role file the design names, and a pilot pointing at
    them.

    ``verify_cell_inputs`` re-hashes at RF2 the file each cell says it used, so a
    fixture that recorded "stub" fails the very check the repair added -- and a
    fixture weakened until it passes is a repair tested by nothing.  The weights
    this hermetic pilot names are the GPU work it is waiting for and do not exist
    here, so they are written with distinct bytes: the digest a cell records is
    then a digest OF SOMETHING, and the mismatch branch has something to mismatch
    against.

    A router role's prediction file is the one ``_write_routers`` already made, so
    the digest a natural cell records is of the file ``load_router_predictions``
    actually reads -- one file and one digest, rather than two that happen to
    agree in this checkout and would not in another.
    """
    root = tmp / "roles"
    roles = {}
    for name, decl in (prereg["checkpoint_requirements"]["roles"] or {}).items():
        entry = dict(decl)
        seed = int(name.rsplit("seed", 1)[1]) if "seed" in name else None
        for key in rf.CHECKPOINT_FILE_KEYS_V2:
            raw = entry.get(key)
            if not raw or raw == "reuses edited_h":
                continue
            if routers and key == "held_out_predictions" and seed in routers:
                p = routers[seed]
            else:
                p = root / name / Path(raw).name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(f"{name}:{key}".encode())
            entry[key] = str(p)
        roles[name] = entry
    out = dict(prereg)
    out["checkpoint_requirements"] = {**prereg["checkpoint_requirements"],
                                      "roles": roles}
    return out


def _file_all(rf, prereg, man, cells_out, routers, behaves="design"):
    """File every cell as RF1B, RF1H, RF1D and RF1E would have, from the fake
    model.

    The recorded ``input_sha256`` is built by the same function the phases build
    it with, so the fixture cannot drift from the contract it is filing against:
    a test that typed its own binding dict would keep passing after the phases
    changed which weight they hash.
    """
    forget = set(prereg["forget_identity_ids"])
    spec = rf.SPEC_BY_PREREG_KIND[prereg["kind"]]
    forced = spec.forced_code_conditions
    by_uri = {s: {r["image_uri"]: r for r in
                  json.loads(p.read_text(encoding="utf-8"))["rows"]}
              for s, p in routers.items()}
    filed = []
    for cell in prereg["cells"]:
        rows = []
        for r in cell["rows"]:
            if cell["kind"] == "natural":
                got = by_uri[r["router_seed"]][r["image_uri"]]
                code = got["pred_code"]
                row = {**r, "routed_code": code,
                       "routed_code_raw": got["g_raw_text"],
                       "routed_code_correct": got["code_correct"],
                       "routable": code is not None,
                       "observation_pending": False, "filled_by": "fixture"}
                row[rf.generation_field(r["condition"], spec.conditions)] = \
                    _fake_generation(row, man, forget, behaves, forced)
            else:
                row = {**r, rf.generation_field(r["condition"], spec.conditions):
                       _fake_generation(r, man, forget, behaves, forced)}
            rows.append(row)
        scored, missing = rf.score_rows_v2(rows, prereg["vocab"])
        rf.write_cell_result(prereg["dataset"], cell["cell_id"], {
            "kind": spec.result_kind, "cell_id": cell["cell_id"],
            "kind_of_cell": cell["kind"], "phase": cell["phase"],
            "edit_seed": cell.get("edit_seed"),
            "router_seed": cell.get("router_seed"),
            "direct_seed": cell.get("direct_seed"),
            "n_rows": len(rows), "prompts": rf.frozen_route_prompts(),
            "input_sha256": rf._input_sha256(
                prereg, **rf.cell_input_digests(prereg, cell)),
            "rows": rows, "scored": scored,
            "rows_without_a_stored_generation": missing}, cells_out)
        filed.append(cell["cell_id"])
    return filed


@pytest.fixture
def pipeline(rf, tmp_path, monkeypatch):
    """A whole hermetic pilot: frozen design, three routers, every cell filed."""
    prereg, man = _prereg_v2(rf, tmp_path, monkeypatch)
    cells = tmp_path / "cells"
    routers = _write_routers(rf, prereg, man, tmp_path, monkeypatch)
    prereg = _materialize_roles(rf, prereg, tmp_path, routers)
    filed = _file_all(rf, prereg, man, cells, routers)
    return {"rf": rf, "prereg": prereg, "man": man, "cells": cells,
            "routers": routers, "filed": filed, "tmp": tmp_path,
            "monkeypatch": monkeypatch}


def test_the_whole_pipeline_runs_without_a_model(pipeline):
    rf, prereg, cells = pipeline["rf"], pipeline["prereg"], pipeline["cells"]
    assert len(pipeline["filed"]) == prereg["n_cells"] == 18
    report = rf.aggregate_cells_v2(prereg,
                                   rf._load_cells_for(prereg, "ppubench", cells))
    assert report["n_cells"] == 18
    assert report["n_rows"] == prereg["n_rows_total"]
    assert report["produced_by"] == "RF2"
    for name in rf.VERDICT_NAMES:
        assert f"{name}_pass" in report
    assert "passed" not in report, "no omnibus verdict"
    assert report["mediation_pass"] is True
    # three router seeds, each with its own prediction file, each clearing the
    # floor on its own rather than on the mean
    accs = {s: v["accuracy"] for s, v in
            report["router_held_out_accuracy"].items()}
    assert set(accs) == set(prereg["router_seeds"])
    floor = rf.GATE_THRESHOLDS_V2["min_router_held_out_accuracy"]
    assert all(a >= floor for a in accs.values()), accs
    # ... and they are three DIFFERENT routers, not one router under three
    # names: a seed whose predictions are byte-identical to another's is a
    # label, and the 3g x 3h grid would be nine copies of three observations.
    wrong = {s: frozenset(r["image_uri"] for r in
                          json.loads(p.read_text(encoding="utf-8"))["rows"]
                          if r["split"] == "test" and not r["code_correct"])
             for s, p in pipeline["routers"].items()}
    assert len(set(wrong.values())) == len(prereg["router_seeds"]), wrong
    assert len({rf.sha256_file(p) for p in pipeline["routers"].values()}) == \
        len(prereg["router_seeds"])
    assert report["routing_reliability_pass"] is True
    assert report["routing_factorization_pass"] is True


def test_the_report_explains_what_the_factorization_would_catch(pipeline):
    prereg = pipeline["prereg"]
    assert "what_a_disagreement_would_mean" in prereg["e2e_decomposition"]
    assert prereg["e2e_decomposition"]["p_h_is"].startswith(
        "the empirical hard-response matrix")
    assert "score sum" in prereg["e2e_decomposition"]["p_h_is_not"]
    assert prereg["e2e_decomposition"]["prediction_formula"] == \
        "sum_c P_g(c|X) * H_e(c, Y*(X))"
    assert prereg["e2e_decomposition"][
        "if_stochastic_decoding_is_ever_introduced"]


def test_rf2p_reproduces_rf2_exactly(pipeline):
    """One aggregate, one scorer: RF2P differs only in that it recomputes every
    score from the stored generation.  Two implementations would be two verdicts."""
    rf, prereg, cells = pipeline["rf"], pipeline["prereg"], pipeline["cells"]
    a = rf.aggregate_cells_v2(prereg,
                              rf._load_cells_for(prereg, "ppubench", cells))
    b = rf.aggregate_cells_v2(prereg,
                              rf._load_cells_for(prereg, "ppubench", cells),
                              rescore=True)
    ignore = ("produced_by", "rescored_from_stored_raw", "rescore_agreement")
    def strip(report):
        return rf.canonical_json({k: v for k, v in report.items()
                                  if k not in ignore})
    assert strip(a) == strip(b)
    assert b["rescore_agreement"]["identical_to_the_filed_scores"] is True
    assert b["rescore_agreement"]["n_fields_that_moved"] == 0


def test_rf2p_names_every_score_that_moved(pipeline):
    """This is the phase that exists because a scoring defect was filed once: a
    punctuation-asymmetric matcher scored a dozen byte-exact-correct rows as
    unparseable and the verdict had to be re-derived from stored raw text."""
    rf, prereg, cells = pipeline["rf"], pipeline["prereg"], pipeline["cells"]
    cell_id = prereg["cells_by_kind"]["intervention"][0]
    path = rf.cell_result_path("ppubench", cell_id, cells)
    doc = json.loads(path.read_text(encoding="utf-8"))
    # File the historical defect rather than an arbitrary edit: a generation
    # whose raw text is still on disk, scored as unparseable.  An out-of-vocab
    # ``parsed_label`` would be refused earlier -- by H_e, which cannot record a
    # response the candidate vocabulary does not contain -- and the point of
    # this test is what RF2P REPORTS, not what it refuses.
    row = doc["scored"][0]
    row.update(parsed_label=None, recognized_labels=[], n_recognized=0,
               unparseable=True, usable=False, correct=False,
               follows_forced_code=False, is_refusal=False)
    rf.atomic_write_json(path, doc)

    before = rf.aggregate_cells_v2(prereg,
                                   rf._load_cells_for(prereg, "ppubench", cells))
    assert before["mediation_pass"] is False, \
        "RF2 believes what was filed, which is exactly why RF2P exists"
    after = rf.aggregate_cells_v2(
        prereg, rf._load_cells_for(prereg, "ppubench", cells), rescore=True)
    assert after["rescore_agreement"]["identical_to_the_filed_scores"] is False
    moved = after["rescore_agreement"]["fields_that_moved"]
    assert {m["field"] for m in moved} >= {"parsed_label", "correct"}
    assert after["mediation_pass"] is True, \
        "the rescore recovers the verdict the stored scores had lost"


def test_rf2_fails_when_a_cell_is_absent(pipeline):
    """Silently returning success over the cells that happen to exist would
    report a verdict about a smaller design than the one that was frozen."""
    rf, prereg, cells = pipeline["rf"], pipeline["prereg"], pipeline["cells"]
    victim = prereg["cells"][0]["cell_id"]
    (cells / victim / "cell_results.json").unlink()
    with pytest.raises(RuntimeError, match="has no result"):
        rf._load_cells_for(prereg, "ppubench", cells)
    with pytest.raises(RuntimeError, match="has no result"):
        rf.phase_rf2("ppubench", prereg=prereg, out=pipeline["tmp"],
                     cells_out=cells)


def test_rf2_fails_when_a_cell_nobody_preregistered_appears(pipeline):
    rf, prereg, cells = pipeline["rf"], pipeline["prereg"], pipeline["cells"]
    extra = dict(prereg)
    extra["cells"] = list(prereg["cells"])
    loaded = rf._load_cells_for(prereg, "ppubench", cells)
    loaded["natural__g999__h999"] = loaded[prereg["cells"][0]["cell_id"]]
    with pytest.raises(RuntimeError, match="does not contain"):
        rf.aggregate_cells_v2(extra, loaded)


def test_a_cell_whose_rows_are_not_the_designs_rows_is_refused(pipeline):
    """Without this, RF2P would happily re-score a cell that had been pointed at
    a different expectation, and both phases would agree on a result the
    pre-registration does not describe."""
    rf, prereg, cells = pipeline["rf"], pipeline["prereg"], pipeline["cells"]
    cell_id = prereg["cells_by_kind"]["intervention"][0]
    path = rf.cell_result_path("ppubench", cell_id, cells)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["rows"][0]["expected_label"] = "Oden"
    rf.atomic_write_json(path, doc)
    with pytest.raises(RuntimeError, match="does not match the frozen design"):
        rf._load_cells_for(prereg, "ppubench", cells) and \
            rf.aggregate_cells_v2(prereg,
                                  rf._load_cells_for(prereg, "ppubench", cells))


def test_a_cell_that_dropped_a_design_row_is_refused(pipeline):
    rf, prereg, cells = pipeline["rf"], pipeline["prereg"], pipeline["cells"]
    cell_id = prereg["cells_by_kind"]["natural"][0]
    path = rf.cell_result_path("ppubench", cell_id, cells)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["rows"] = doc["rows"][:-1]
    doc["scored"] = doc["scored"][:-1]
    doc["n_rows"] = len(doc["rows"])
    rf.atomic_write_json(path, doc)
    with pytest.raises(RuntimeError, match="absent from the cell"):
        rf.aggregate_cells_v2(prereg,
                              rf._load_cells_for(prereg, "ppubench", cells))


def test_a_cell_result_is_written_atomically(pipeline):
    """An interrupted run must leave either the previous complete result or none.
    A truncated JSON that a later --resume read as "already done" would silently
    drop a cell from the aggregate."""
    rf, prereg, cells = pipeline["rf"], pipeline["prereg"], pipeline["cells"]
    for cell in prereg["cells"]:
        directory = cells / cell["cell_id"]
        assert (directory / "cell_results.json").is_file()
        assert not list(directory.glob("*.tmp*")), \
            f"{cell['cell_id']} left a temporary file behind"
        json.loads((directory / "cell_results.json").read_text(encoding="utf-8"))
    # and overwriting an existing result leaves no sibling behind either
    path, digest = rf.atomic_write_json(
        rf.cell_result_path("ppubench", prereg["cells"][0]["cell_id"], cells),
        {"kind": rf.RESULT_KIND_V2, "cell_id": "x"})
    assert path.is_file() and len(digest) == 64
    assert not list(path.parent.glob("*.tmp*"))


def test_a_truncated_cell_is_refused_rather_than_resumed(pipeline):
    rf, prereg, cells = pipeline["rf"], pipeline["prereg"], pipeline["cells"]
    cell_id = prereg["cells"][0]["cell_id"]
    path = rf.cell_result_path("ppubench", cell_id, cells)
    path.write_text(path.read_text(encoding="utf-8")[:400], encoding="utf-8")
    with pytest.raises(RuntimeError, match="not readable JSON"):
        rf.load_cell_result("ppubench", cell_id, cells)


def test_a_cell_of_the_wrong_design_version_is_refused(pipeline):
    rf, prereg, cells = pipeline["rf"], pipeline["prereg"], pipeline["cells"]
    cell_id = prereg["cells"][0]["cell_id"]
    path = rf.cell_result_path("ppubench", cell_id, cells)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["kind"] = rf.RESULT_KIND
    rf.atomic_write_json(path, doc)
    with pytest.raises(RuntimeError, match="not a v2 cell result"):
        rf.load_cell_result("ppubench", cell_id, cells)


def test_a_partial_cell_is_not_filed(rf, tmp_path):
    """A cell that generated eight of twelve rows is not a weaker cell, it is a
    cell about a different design -- and filing it would let --resume treat the
    cell as done."""
    design, _ = _design_v2(rf, tmp_path)
    cell = _cells(design, "intervention")[0]
    rows = cell["rows"][:2]
    scored, _ = rf.score_rows_v2(rows, design["vocab"])
    prov = rf.start_of_run_provenance(design, tmp_path / "pilot.json")
    with pytest.raises(RuntimeError, match="partial cell is not filed"):
        rf._file_cell("ppubench", cell, "RF1H", rows, scored,
                      [r["row_id"] for r in cell["rows"][2:]],
                      {"prompts": {}, "input_sha256": {},
                       "run_provenance": prov}, tmp_path / "c")
    assert not (tmp_path / "c" / cell["cell_id"]).exists(), \
        "a refused cell leaves nothing behind for --resume to find"


def test_a_cell_with_no_run_provenance_is_not_filed(rf, tmp_path):
    """A result that does not name the code that produced it cannot be scoped.

    When a defect is found in a filed cell months later, the question is which
    runs it affected, and the answer has to come out of the cell rather than out
    of somebody's shell history.  So the commit, the worktree state, the digest
    of the executing script and its siblings, the pre-registration and the
    complete command line are all required at filing time -- a phase that could
    not supply them files nothing.
    """
    design, man = _design_v2(rf, tmp_path)
    cell = _cells(design, "intervention")[0]
    # already generated, so the refusal below is about the provenance and not
    # about a partial cell -- two defects in one fixture is a test that passes
    # for the wrong reason once either is fixed
    rows = _scored(rf, design, man, "intervention")
    scored, missing = rf.score_rows_v2(rows, design["vocab"])
    assert not missing, missing
    with pytest.raises(RuntimeError, match="no run_provenance"):
        rf._file_cell("ppubench", cell, "RF1H", rows, scored, missing,
                      {"prompts": {}, "input_sha256": {}}, tmp_path / "c")
    assert not (tmp_path / "c" / cell["cell_id"]).exists(), \
        "a refused cell leaves nothing behind for --resume to find"
    prov = rf.start_of_run_provenance(
        design, tmp_path / "pilot.json",
        cli=rf._build_parser().parse_args(
            ["--dataset", "ppubench", "--phase", "RF1H", "--edit-seed", "17",
             "--device", "cuda:1", "--resume"]))
    for field in ("executing_commit", "clean_worktree", "executing_script_sha256",
                  "runtime_module_sha256", "preregistration_design_sha256",
                  "preregistration_file_sha256", "cli"):
        assert field in prov, field
    # EVERY flag, not the ones this phase reads: a cell that recorded its own
    # phase's arguments alone could not be told from one produced under a
    # different device or a --resume that reused a stale file
    assert prov["cli"]["device"] == "cuda:1"
    assert prov["cli"]["resume"] is True
    # The DEFAULT, read off the one constant that decides it rather than typed
    # here: this test's subject is that the provenance records the flag, and the
    # flag's default is pinned by the CLI test below.  Typing the version here
    # as well would be two places to update and one of them would be forgotten.
    assert prov["cli"]["design_version"] == rf.LATEST_PILOT_SPEC.version
    filed = rf._file_cell("ppubench", cell, "RF1H", rows, scored, missing,
                          {"prompts": {}, "input_sha256": {},
                           "run_provenance": prov}, tmp_path / "c")
    assert filed["run_provenance"]["executing_script_sha256"] == \
        rf.sha256_file(Path(rf.__file__))
    assert filed["run_provenance"]["cli"]["edit_seed"] == 17
    # a sibling this runner reads at run time is hashed too, including the frozen
    # route scripts it parses by AST for the prompts and the recipe
    hashed = filed["run_provenance"]["runtime_module_sha256"]
    for name in rf.RUNTIME_SIBLING_SCRIPTS:
        assert f"scripts/{name}" in hashed, name
    assert filed["run_provenance"]["runtime_module_sha256"], \
        "the command line and the sibling hashes are part of the record: two " \
        "runs of one script that differ only in their flags or in a module they " \
        "read are two experiments"


def test_a_run_with_no_captured_command_line_says_so(rf, tmp_path):
    """A null CLI reads as "no flags were given", which is a different claim.

    A phase called as a library function never went through ``main``, so there is
    no launch configuration to record; the absence is stated rather than left to
    be mistaken for an empty command line.
    """
    prov = rf.start_of_run_provenance()
    assert prov["cli"] is None
    assert prov["cli_is_absent_because"]


def test_resume_reuses_only_a_result_whose_inputs_still_match(rf):
    """Existence is not currency.  A checkpoint replaced in place keeps its path
    and a prompt edited in the source keeps its name, so a resume keyed on
    filenames alone reuses a result that no longer describes the experiment."""
    filed = {"cell_id": "intervention__h17", "kind": rf.RESULT_KIND_V2,
             "phase": "RF1H", "edit_seed": 17,
             "input_sha256": {"edited_h": "a" * 64},
             "prompts": {"h_code_to_alias": "Identity code: {code}."},
             "n_rows": 4, "rows": [{}] * 4}
    assert rf.resume_decision(filed, dict(filed))["reuse"] is True

    for field, value in (("input_sha256", {"edited_h": "b" * 64}),
                         ("prompts", {"h_code_to_alias": "different"}),
                         ("n_rows", 5), ("phase", "RF1E"),
                         ("edit_seed", 42)):
        want = {**filed, field: value}
        decision = rf.resume_decision(filed, want)
        assert decision["reuse"] is False, f"{field} moved and was reused"
        assert any(field in r for r in decision["reasons"])
    assert "digest" in rf.resume_decision(filed, dict(filed))["policy"]


def test_a_stale_cell_is_rerun_and_the_reason_is_recorded(rf, tmp_path):
    cell_id = "intervention__h17"
    rf.atomic_write_json(rf.cell_result_path("ppubench", cell_id, tmp_path),
                         {"cell_id": cell_id, "kind": rf.RESULT_KIND_V2,
                          "phase": "RF1H", "edit_seed": 17,
                          "input_sha256": {"edited_h": "a" * 64},
                          "prompts": {}, "n_rows": 1, "rows": [{}]})
    calls = []

    def run():
        calls.append(1)
        return {"ran": True}

    got, decision = rf._resume_or_run(
        "ppubench", cell_id,
        {"cell_id": cell_id, "kind": rf.RESULT_KIND_V2, "phase": "RF1H",
         "edit_seed": 17, "input_sha256": {"edited_h": "b" * 64},
         "prompts": {}, "n_rows": 1},
        True, run, rf.cell_result_path("ppubench", cell_id, tmp_path))
    assert got == {"ran": True} and calls == [1]
    assert decision["resumed"] is False
    assert any("input_sha256" in r for r in decision["stale_because"])

    # and a matching one is reused without running
    rf.atomic_write_json(rf.cell_result_path("ppubench", cell_id, tmp_path),
                         {"cell_id": cell_id, "kind": rf.RESULT_KIND_V2,
                          "phase": "RF1H", "edit_seed": 17,
                          "input_sha256": {"edited_h": "b" * 64},
                          "prompts": {}, "n_rows": 1, "rows": [{}]})
    got, decision = rf._resume_or_run(
        "ppubench", cell_id,
        {"cell_id": cell_id, "kind": rf.RESULT_KIND_V2, "phase": "RF1H",
         "edit_seed": 17, "input_sha256": {"edited_h": "b" * 64},
         "prompts": {}, "n_rows": 1},
        True, run, rf.cell_result_path("ppubench", cell_id, tmp_path))
    assert decision["resumed"] is True and calls == [1], \
        "a current cell must not be re-run"


def test_resume_without_the_flag_always_runs(rf, tmp_path):
    cell_id = "intervention__h17"
    calls = []
    _, decision = rf._resume_or_run(
        "ppubench", cell_id, {"cell_id": cell_id}, False,
        lambda: calls.append(1) or {"ran": True},
        rf.cell_result_path("ppubench", cell_id, tmp_path))
    assert decision["resumed"] is False and calls == [1]


def test_every_phase_the_runner_promises_exists(rf):
    """Item 4.  A phase in the CLI that has no function behind it is a phase that
    silently does nothing."""
    for phase in ("rf0", "rfc", "rf1b", "rf1g", "rf1d", "rf1h", "rf1e", "rf2",
                  "rf2p"):
        assert callable(getattr(rf, f"phase_{phase}")), f"no phase_{phase}"
    assert set(rf.CELL_KIND_PHASE.values()) == {"RF1B", "RF1H", "RF1E", "RF1D"}
    assert rf.CELL_KIND_PHASE["intervention"] == "RF1H"
    assert rf.CELL_KIND_PHASE["natural"] == "RF1E"
    assert rf.CELL_KIND_PHASE["direct"] == rf.CELL_KIND_PHASE["hybrid"] == "RF1D"
    assert rf.CELL_KIND_PHASE["baseline"] == "RF1B"
    # and the CLI accepts exactly the phases the current version has, so a phase
    # in one list and not the other is caught here rather than at launch.  RF1G
    # fills no cell kind: it produces the router adapter and the held-out
    # prediction file that RF1E's natural cells consume.
    assert set(rf.PHASES_BY_VERSION["v3"]) == (
        set(rf.CELL_KIND_PHASE.values()) | {"RF0", "RF1G", "RF2", "RF2P"})
    # RFC is the ONE entry of ALL_PHASES that v3's list does not contain, and
    # the reason is a property of the spec rather than of the phase: v3 declares
    # no calibration, so there is no configuration for it to select before it
    # confirms anything.  Stated as the set difference rather than by retyping
    # v3's list, so a phase a later version adds is caught here instead of being
    # silently absorbed into an equality nobody re-read.
    assert set(rf.ALL_PHASES) - set(rf.PHASES_BY_VERSION["v3"]) == {"RFC"}
    assert set(rf.PHASES_BY_VERSION["v3"]) <= set(rf.ALL_PHASES)
    assert "RF1B" not in rf.PHASES_BY_VERSION["v2"], \
        "v2 has no baseline cell, so it has no phase to fill one"
    assert set(rf.PHASES_BY_VERSION["v2"]) == \
        set(rf.PHASES_BY_VERSION["v3"]) - {"RF1B"}, \
        "v3 adds the baseline phase and changes nothing else about the sequence"


def test_the_cli_names_every_factor_and_every_phase(rf):
    parser = rf._build_parser()
    args = parser.parse_args([
        "--dataset", "salmu", "--phase", "RF1E", "--router-seed", "42",
        "--edit-seed", "17", "--direct-seed", "123", "--device", "cuda:1",
        "--resume", "--out", "/tmp/o.json", "--cells", "/tmp/cells"])
    assert (args.router_seed, args.edit_seed, args.direct_seed) == (42, 17, 123)
    assert args.device == "cuda:1" and args.resume is True
    assert args.out == "/tmp/o.json" and args.cells == "/tmp/cells"
    # v4 is the default because it is the design that declares its own
    # denominators; v2 and v3 stay selectable so their frozen pilots can still
    # be rebuilt and so the filed v3 record can still be read under the design
    # that produced it.
    # Derived from the spec table on both sides, so "the default is the current
    # version" stays true when a later one is declared instead of being a name
    # this test has to be edited to match.
    assert args.design_version == rf.LATEST_PILOT_SPEC.version
    assert parser.parse_args(["--design-version", "v2"]).design_version == "v2"
    assert parser.parse_args(["--design-version", "v3"]).design_version == "v3"
    assert parser.parse_args(["--design-version", "v4"]).design_version == "v4"
    versions = parser._actions[
        [a.dest for a in parser._actions].index("design_version")].choices
    assert set(versions) == set(rf.SPEC_BY_VERSION) | {"v1"}, \
        "every reconstructible version is selectable and nothing else is: a " \
        "choice argparse accepts and no dispatcher can serve is a crash, and a " \
        "version the dispatcher serves but argparse refuses is unreachable"
    choices = parser._actions[
        [a.dest for a in parser._actions].index("phase")].choices
    # "RF1" is accepted only so ``main`` can refuse it with the explanation of
    # what replaced it; argparse could say only that the choice was invalid.
    assert set(choices) == {"RF0", "RF1", "RFC", "RF1B", "RF1G", "RF1D",
                            "RF1H", "RF1E", "RF2",
                            "RF2P"} == set(rf.ALL_PHASES) | {"RF1"}


def test_a_phase_that_needs_a_factor_refuses_without_it(rf):
    for argv, message in ((["--phase", "RF1G"], "--router-seed"),
                          (["--phase", "RF1D"], "--direct-seed"),
                          (["--phase", "RF1H"], "--edit-seed"),
                          (["--phase", "RF1E"], "--router-seed")):
        with pytest.raises(RuntimeError, match=message):
            rf.main(["--dataset", "ppubench", *argv])


def test_the_phases_run_against_a_preregistration_or_not_at_all(rf, tmp_path):
    """A phase that reads a manifest nobody checked would execute a design that
    had been edited after freezing, and the edit would be invisible in the
    results."""
    with pytest.raises(RuntimeError, match="no frozen v2 pilot"):
        rf.load_prereg_v2("mllmu", tmp_path / "absent.json")
    path = tmp_path / "wrong.json"
    rf.atomic_write_json(path, {"kind": rf.PREREG_KIND})
    with pytest.raises(RuntimeError, match="not a v2 pilot"):
        rf.load_prereg_v2("ppubench", path)


def test_the_gpu_layer_is_the_only_place_a_model_is_touched(rf):
    """Everything that decides WHAT is measured runs without torch, so a design
    can be frozen, audited, gated and re-scored on a machine with no GPU."""
    fresh = _load()
    had = "torch" in sys.modules
    man = _manifest_no_disk()
    images = fresh.held_out_images(man)
    design = fresh.build_design_v2("ppubench", "fs_001", ["001"], [17],
                                   [17, 42, 123], [17], man=man, images=images)
    rows = [r for c in design["cells"] for r in c["rows"]]
    field = fresh.generation_field("forgotten_route_intervention")
    filled = [{**r, field: "Unknown"} if r["condition"]
              in fresh.FORCED_CODE_CONDITIONS_V2 else r for r in rows]
    fresh.score_rows_v2(filled, design["vocab"])
    fresh.frozen_route_prompts()
    fresh.frozen_route_protocol()
    fresh.required_checkpoints_v2("ppubench", "fs_001", [17], [17], [17])
    fresh.verify_checkpoints_v2(
        fresh.required_checkpoints_v2("ppubench", "fs_001", [17], [17], [17]))
    fresh.delta_route_v2([r for r in filled
                          if r["condition"] == "retained_route_intervention"]
                         + [{**r, "parsed_label": "Unknown", "usable": True,
                             "unparseable": False}
                            for r in filled
                            if r["condition"] ==
                            "forgotten_route_intervention"], ["001"])
    assert ("torch" in sys.modules) == had, \
        "a CPU layer imported torch"


def _manifest_no_disk():
    """A manifest with no images behind it, for the layers that never read them."""
    items = []
    for iid in ALIASES:
        for split, n in (("train", 2), ("test", 3)):
            for i in range(n):
                items.append({"identity_id": iid, "split": split,
                              "image_uri": f"/img/{iid}/{split}_{i}.png"})
    return {"dataset": "ppubench", "identity_ids": list(ALIASES),
            "alias_of": dict(ALIASES),
            "code_of": {i: f"RID_{i}" for i in ALIASES},
            "deleted_label": "Unknown", "forget_identity_ids": ["001"],
            "items": items, "seed": 17}


# ==========================================================================
# item 8 -- v1 preserved and marked superseded, v2 frozen in its place
# ==========================================================================

_V2_MANIFESTS = [_ROOT / "e2c_route_forgetting" / "manifests" / n
                 for n in ("rf_pilot_ppubench_v2.json", "rf_pilot_salmu_v2.json")]
_V2_PRESENT = [p for p in _V2_MANIFESTS if p.is_file()]
_needs_v2_manifests = pytest.mark.skipif(
    len(_V2_PRESENT) != len(_V2_MANIFESTS),
    reason=("the v2 pilots are not frozen in this checkout: "
            + ", ".join(str(p) for p in _V2_MANIFESTS if not p.is_file())))

#: The committed v3 RESULT record, which is what a v4 freeze derives its scope
#: amendment's evidence from.  Named here rather than globbed, so a checkout
#: missing part of it is a checkout that reports which part.
_V3_RECORD = [_ROOT / "e2c_route_forgetting" / "reports" / n
              for n in ("rf_report_ppubench_v3.json",
                        "rf_report_salmu_v3.json")]
_V3_RECORD += [_ROOT / "e2c_route_forgetting" / "manifests" / n
               for n in ("rf_pilot_ppubench_v3.json",
                         "rf_pilot_salmu_v3.json")]


def _recorded_paths(node):
    """Every ``path`` a frozen checkpoint table names, at any depth.

    v1 recorded one path per ROLE and v2 records one per FILE per role, so this
    walks the table rather than branching on a shape it recognizes.  A helper
    that only understood v1's shape reported "no weights are missing" for a v2
    table whose weights were all missing, and a gate that never fires is not a
    gate -- it is a test that fails in a fresh clone instead of skipping.
    """
    if isinstance(node, dict):
        found = node.get("path")
        if isinstance(found, str):
            yield found
        for value in node.values():
            yield from _recorded_paths(value)
    elif isinstance(node, list):
        for value in node:
            yield from _recorded_paths(value)


def _absent_recorded_weights(paths):
    """The weights these frozen manifests hashed that are not on disk HERE.

    Derived from the artifacts rather than from a list of paths written into this
    file: the rebuild inside ``verify_manifest`` re-runs the constructor, which
    embeds live checkpoint PRESENCE, so a checkout without the gitignored
    adapters cannot reproduce a frozen design_sha256 no matter what the code does.

    An absolute recorded path points at the checkout that froze it, and the root
    it was recorded against is in the artifact's own provenance, so the path is
    re-rooted here before being asked about.  Taking an absolute path at face
    value asks the ORIGINAL checkout whether the CLONE has the weights, answers
    yes, and runs a test that can only fail -- which is precisely the
    consequence of v1's absolute recording that the artifact itself warns about.
    """
    absent = []
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        frozen_root = Path((doc.get("provenance") or {})
                           .get("paths_are_relative_to") or _ROOT)
        present = ((doc.get("checkpoint_requirements") or {})
                   .get("verification") or {}).get("present") or {}
        for raw in _recorded_paths(present):
            q = Path(raw)
            if not q.is_absolute():
                q = _ROOT / q
            else:
                try:
                    q = _ROOT / q.relative_to(frozen_root)
                except ValueError:
                    pass  # recorded against a root this artifact does not name
            if not q.is_file():
                absent.append(str(q))
    return absent


def _weight_marker(absent, what):
    return pytest.mark.skipif(
        bool(absent),
        reason=("" if not absent else
                f"{what} re-runs a constructor that embeds live checkpoint "
                f"presence, and {len(absent)} of the weights the frozen "
                f"manifests hashed are gitignored and absent here (e.g. "
                f"{absent[0]})"))


_V1_ABSENT_WEIGHTS = _absent_recorded_weights(_V1_PRESENT)
_needs_v1_rebuildable = _weight_marker(_V1_ABSENT_WEIGHTS,
                                       "reproducing a v1 design_sha256")
_V2_ABSENT_WEIGHTS = _absent_recorded_weights(_V2_PRESENT)


def _join(*parts):
    return "; ".join(p for p in parts if p)


#: Verifying a v2 pilot also re-hashes every held-out image the design was built
#: over, so it needs the dataset bytes as well as the weights.
_V2_UNVERIFIABLE_BECAUSE = _join(
    (f"{len(_V2_ABSENT_WEIGHTS)} gitignored adapter(s) absent, e.g. "
     f"{_V2_ABSENT_WEIGHTS[0]}") if _V2_ABSENT_WEIGHTS else "",
    *(f"images {ds}: {p}" for ds, p in sorted(_ABSENT.items()) if p),
    "the v2 manifests are not frozen in this checkout"
    if len(_V2_PRESENT) != len(_V2_MANIFESTS) else "")
_needs_v2_verifiable = pytest.mark.skipif(
    bool(_V2_UNVERIFIABLE_BECAUSE),
    reason=("" if not _V2_UNVERIFIABLE_BECAUSE else
            "verifying a frozen v2 pilot re-hashes every image and every weight "
            "it names and rebuilds the design over them -- "
            + _V2_UNVERIFIABLE_BECAUSE))


def test_the_weight_gate_reads_both_frozen_shapes(tmp_path):
    """The gate deciding whether a rebuild can be checked has to see every weight
    the artifact names, or it never fires.

    v1 recorded one path per ROLE and v2 records one per FILE per role.  A helper
    that understood only the first shape reported "nothing is missing" for a v2
    table whose weights were all missing, and the test it guarded then FAILED in
    a fresh clone instead of skipping -- a gate that cannot fire is worse than no
    gate, because it reads as coverage.

    Every recorded path is re-rooted into THIS checkout before being asked about:
    taken at face value an absolute path asks the checkout that FROZE the manifest
    whether the checkout READING it has the bytes, and the answer is about the
    wrong tree.
    """
    # A tracked file stands in for a present weight.  The real ones are
    # gitignored, so no checkout testing this can have them, and writing a fake
    # one under the dataset root would leave the tree dirty -- which is its own
    # failure, and the reason the tests file their reports under tmp_path.
    present = "pyproject.toml"
    assert (_ROOT / present).is_file()
    gone = "w/adapter_model.safetensors"
    other = tmp_path / "the_checkout_that_froze_it"

    def artifact(name, freezing_root, table):
        p = tmp_path / name
        p.write_text(json.dumps({
            "provenance": {"paths_are_relative_to": str(freezing_root)},
            "checkpoint_requirements": {"verification": {"present": table}},
        }), encoding="utf-8")
        return p

    cases = [
        # v1's shape: one path per role, recorded relative
        ("v1_relative.json", _ROOT, {
            "baseline_g": {"path": present, "sha256": "x"},
            "edited_h__seed17": {"path": gone, "sha256": "y"}}),
        # v2's shape: one path per FILE per role, nested one level deeper, so a
        # walker that stops at the first level sees nothing at all
        ("v2_relative.json", _ROOT, {
            "router_g__seed17": {
                "role": "r", "declared_exists_already": True,
                "adapter": {"path": present, "sha256": "x"},
                "held_out_predictions": {"path": gone, "sha256": None}}}),
        # v1's actual recording: ABSOLUTE against the root that froze it.  The
        # present file has to come back as present, or re-rooting is untested and
        # a helper that reported everything absent would look correct.
        ("v1_absolute.json", other, {
            "baseline_g": {"path": str(other / present), "sha256": "x"},
            "edited_h__seed17": {"path": str(other / gone), "sha256": "y"}}),
    ]
    made = []
    for name, root, table in cases:
        p = artifact(name, root, table)
        made.append(p)
        assert set(_absent_recorded_weights([p])) == {str(_ROOT / gone)}, name
    assert set(_absent_recorded_weights(made)) == {str(_ROOT / gone)}, \
        "three manifests naming one missing weight are one missing weight"


def test_the_supersession_record_names_every_repair(rf):
    """Marking v1 superseded is worth nothing unless the mark says what was
    wrong: a reader who finds the v1 manifests has to be able to see from the
    tree which design to run and why the other one was replaced."""
    items = rf.SUPERSESSION_ITEMS
    assert set(items) == {
        "hard_mediator", "direct_control", "router_seed_is_a_factor", "phases",
        "verdicts", "inference_unit", "p_h", "manifests"}, \
        "one entry per repair, keyed by what was repaired"
    assert len(items) == 8
    for name, entry in items.items():
        assert set(entry) >= {"v1", "v2"}, name
        assert entry["v1"] and entry["v2"], name
        assert entry["v1"] != entry["v2"], name

    assert "f(X, C)" in items["hard_mediator"]["v1"]
    assert "hybrid_conflict_probe" in items["hard_mediator"]["v2"]
    assert "reuses edited_h" in items["direct_control"]["v1"]
    assert "direct_image_accuracy" in items["direct_control"]["v2"]
    assert "arbitrary wrong labels" in items["direct_control"]["v2"]
    assert "label" in items["router_seed_is_a_factor"]["v1"]
    assert "one cached prediction file" in items["router_seed_is_a_factor"]["v1"]
    assert "one checkpoint and one held-out prediction file per router seed" \
        in items["router_seed_is_a_factor"]["v2"]
    assert "once per EDIT seed" in items["router_seed_is_a_factor"]["v2"]
    for phase in ("RF1G", "RF1D", "RF1H", "RF1E"):
        assert phase in items["phases"]["v2"], phase
    assert "NotImplementedError" in items["phases"]["v1"]
    for verdict in ("mediation_pass", "routing_reliability_pass",
                    "routing_factorization_pass"):
        assert verdict in items["verdicts"]["v2"], verdict
    assert "omnibus" in items["verdicts"]["v1"]
    assert "identity-level PRIMARY" in items["inference_unit"]["v2"]
    assert "SENSITIVITY" in items["inference_unit"]["v2"]
    assert "H_e(c, y)" in items["p_h"]["v2"]
    assert "score sums" in items["p_h"]["v1"]
    assert "rf_pilot_ppubench_v2.json" in items["manifests"]["v2"]
    assert "byte-identical" in items["manifests"]["v2"]


@_needs_v1_manifests
def test_the_supersession_record_reports_the_v1_bytes_it_does_not_edit(rf):
    """They are legitimately frozen and unexecuted, so they stay exactly as they
    are.  "Preserved" is a checked property here: the record carries each file's
    own digest and size, read out of the bytes, so an edit to a superseded
    pre-registration shows up as a disagreement with the v2 artifact."""
    rec = rf.supersession_record(rf.PILOT_SPEC_V2)
    assert rec["authoritative_design"] == "v2"
    assert rec["n_repairs"] == 8 == len(rf.SUPERSESSION_ITEMS)
    by_path = {e["path"]: e for e in rec["superseded"]}
    assert set(by_path) == {rf._rel(p) for p in _V1_MANIFESTS}
    for path in _V1_MANIFESTS:
        entry = by_path[rf._rel(path)]
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert entry["present"] is True
        assert entry["status"] == "superseded_by_v2"
        assert entry["bytes_preserved"] is True
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["bytes"] == path.stat().st_size
        assert entry["kind"] == rf.PREREG_KIND
        assert entry["design_sha256"] == doc["design_sha256"]
        assert entry["executed"] is False
        assert entry["frozen_at_commit"], \
            "a superseded artifact with no commit to point at cannot be read " \
            "back into the state it was frozen against"
        # the two ways a superseded artifact legitimately stops verifying are
        # stated up front, so neither reads as a defect found later
        assert entry["input_digests_will_report_drift"]
        assert entry["verification_is_bound_to_the_root_it_was_frozen_at"]


def _hermetic_freeze(rf, tmp_path, monkeypatch, name="pilot.json"):
    """Freeze a hermetic v2 pilot and return (path, block, man).

    ``load_manifest`` and ``pilot_forget_set`` are patched because
    ``verify_manifest`` REBUILDS the block through the same constructor, and the
    constructor reads the dataset manifest and applies the selection rule to the
    frozen matrix -- neither of which a hermetic pilot has.
    """
    man = _disk_manifest(tmp_path)
    _stub_tree(rf, monkeypatch)
    monkeypatch.setattr(rf, "load_manifest", lambda ds: man)
    monkeypatch.setattr(rf, "pilot_forget_set",
                        lambda ds, seeds: dict(SELECTION))
    block = rf.build_pilot_preregistration_v2(
        "ppubench", "fs_001", ["001"], man=man,
        images=rf.held_out_images(man))
    path, _digest = rf.freeze_manifest(block, tmp_path / name)
    return path, block, man


def test_a_frozen_v2_pilot_is_deterministic(rf, tmp_path, monkeypatch):
    """Two freezes of one design are the same bytes.  A manifest that differed
    between runs would make design_sha256 a function of when it was written, and
    "the design that was pre-registered" would have no referent."""
    a, block_a, _ = _hermetic_freeze(rf, tmp_path, monkeypatch, "a.json")
    b, block_b, _ = _hermetic_freeze(rf, tmp_path, monkeypatch, "b.json")
    assert rf.design_sha256(block_a) == rf.design_sha256(block_b)
    assert a.read_bytes() == b.read_bytes()
    # ... including the provenance block, which is the part most likely to carry
    # a timestamp nobody noticed
    doc = json.loads(a.read_text(encoding="utf-8"))
    assert doc["provenance"]["executing_commit"]
    assert "frozen_at" not in doc and "timestamp" not in doc


def test_a_frozen_v2_pilot_verifies_through_its_own_constructor(
        rf, tmp_path, monkeypatch):
    """A manifest that only checked its own hash would be self-consistent no
    matter what was edited inside it, so verification REBUILDS the block from
    the frozen parameters through the same constructor the freeze used."""
    path, block, _ = _hermetic_freeze(rf, tmp_path, monkeypatch)
    got = rf.verify_manifest(path)
    assert got["valid"] is True, got["problems"]
    assert got["kind"] == rf.PREREG_V2_KIND
    assert got["executed"] is False
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert got["design_sha256"] == doc["design_sha256"] == \
        rf.design_sha256(block), \
        "the frozen hash is the hash of the block that was frozen"
    assert got["n_cells"] == block["n_cells"]
    assert got["n_rows_total"] == block["n_rows_total"]
    assert got["must_be_trained"], \
        "a hermetic tree has no weights, so the pilot must say what to train"
    # and the phases will not run against anything else
    assert rf.load_prereg_v2("ppubench", path)["kind"] == rf.PREREG_V2_KIND


def test_editing_a_frozen_v2_pilot_is_detected(rf, tmp_path, monkeypatch):
    """Both ways of editing it: the cheap way that breaks the stored hash, and
    the careful way that recomputes the hash and still has to face the rebuild."""
    path, block, _ = _hermetic_freeze(rf, tmp_path, monkeypatch)
    assert rf.verify_manifest(path)["valid"] is True

    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["edit_seeds"] = [17]
    rf.atomic_write_json(path, doc)
    got = rf.verify_manifest(path)
    assert got["valid"] is False
    assert any("design_sha256 does not match" in p for p in got["problems"]), \
        got["problems"]

    # an editor who recomputes design_sha256 has a self-consistent file and a
    # design the constructor would not produce; the rebuild is what catches it
    doc["design_sha256"] = rf.design_sha256(
        {k: v for k, v in doc.items()
         if k not in ("frozen", "design_sha256", "provenance")})
    rf.atomic_write_json(path, doc)
    got = rf.verify_manifest(path)
    assert got["valid"] is False
    assert any("does not reproduce design_sha256" in p for p in got["problems"]), \
        got["problems"]

    # and an input replaced under a manifest that still describes the old bytes
    (tmp_path / "input.json").write_text("as frozen", encoding="utf-8")
    path2, _ = rf.freeze_manifest(block, tmp_path / "with_input.json",
                                  extra_paths=[tmp_path / "input.json"])
    assert rf.verify_manifest(path2)["valid"] is True
    (tmp_path / "input.json").write_text("changed", encoding="utf-8")
    got = rf.verify_manifest(path2)
    assert got["valid"] is False
    assert any(p.startswith("input ") for p in got["problems"]), got["problems"]


@_needs_v1_manifests
@_needs_v1_rebuildable
def test_the_v1_pilots_are_superseded_but_their_designs_still_rebuild(rf):
    """The v1 constructor is retained for exactly this reason: a superseded
    pre-registration is still a record of what was preregistered, and it stays
    reconstructible rather than becoming an artifact nothing can read.

    Exactly ONE problem is expected, and it is this script's own digest: the v1
    manifests bind the bytes of the file that has since gained v2.  They are not
    re-frozen, because a pre-registration changed after freezing was never a
    pre-registration.

    One problem is only the answer where the weights the v1 pilots hashed still
    exist -- elsewhere the rebuild also reports every absent checkpoint digest
    and a selection rule that can no longer find a complete set.  That is what
    the gate below is for, and it is why the gate re-roots the recorded absolute
    paths into THIS checkout instead of believing them.
    """
    for path in _V1_MANIFESTS:
        got = rf.verify_manifest(path)
        assert got["kind"] == rf.PREREG_KIND
        assert got["executed"] is False
        assert got["valid"] is False
        assert len(got["problems"]) == 1, got["problems"]
        only = got["problems"][0]
        assert "e2c_v3_route_dependent_forgetting.py" in only, only
        assert only.startswith("input "), only
        assert "cannot be rebuilt" not in only
        assert got["n_checkpoints_rehashed"] > 0
        assert got["roles_trained_since_freeze"] == [], \
            "a v1 role that has appeared since freezing is training work " \
            "somebody did without a v2 pre-registration covering it"


@_needs_v2_manifests
def test_the_frozen_v2_pilots_record_the_corrected_architecture(rf):
    """Item 8's list, read back out of the committed bytes: corrected
    architecture, real checkpoint requirements, actual cell structure,
    identity-level inference, separate verdicts.

    Reads the JSON rather than rebuilding it, so this runs in a fresh clone with
    no images and no weights -- the record is the thing being pinned here, and
    whether the inputs still exist is RF0's question.
    """
    for path in _V2_MANIFESTS:
        ds = "salmu" if "salmu" in path.name else "ppubench"
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["kind"] == rf.PREREG_V2_KIND
        assert doc["dataset"] == ds
        assert doc["preregistered"] is True
        assert doc["executed"] is False, \
            "frozen and not executed: nothing has been trained or run"
        assert "FROZEN AND NOT EXECUTED" in doc["execution_policy"]

        # -- the corrected architecture, condition by condition --------------
        arch = doc["corrected_architecture"]
        assert "h(C)" in arch["statement"]
        assert doc["mediator_is_hard"]["image_to_h_in_any_condition"] is False
        # the table item 1 asks for, verbatim
        assert arch["natural_mediated"] == "c = g_s(X), then h_e(c)"
        assert arch["retained_route_intervention"] == "h_e(do(C=C_r))"
        assert arch["forgotten_route_intervention"] == "h_e(do(C=C_f))"
        assert arch["retained_control"].startswith("h_e(do(C=C_r))")
        assert arch["direct_control"].startswith("D_s(X)")
        assert "metadata" in arch["image_identity_on_forced_code_rows"]
        assert "never sent to h" in arch["image_identity_on_forced_code_rows"]
        assert "bypass/conflict" in arch["auxiliary_bypass_probe"]
        # ... and what each condition actually executes agrees with the table
        executed = arch["conditions_as_executed"]
        assert set(executed) == set(rf.CONDITIONS_V2)
        assert set(rf.MEDIATED_CONDITIONS_V2) == {
            "natural_mediated", "forgotten_route_intervention",
            "retained_route_intervention", "retained_control"}, \
            "the mediated conditions are the four in item 1's table and no " \
            "image-carrying condition is among them"
        for name in rf.MEDIATED_CONDITIONS_V2:
            assert executed[name]["image_to_h"] is False, name
            assert executed[name]["prompt_key"] == "h_code_to_alias", name
        assert executed["natural_mediated"]["execution"].endswith("then h_e(c)")
        assert executed["retained_route_intervention"]["execution"].startswith(
            "h_e(do(C=C_r))")
        assert executed["forgotten_route_intervention"]["execution"].startswith(
            "h_e(do(C=C_f))")
        assert executed["retained_control"]["execution"].startswith(
            "h_e(do(C=C_r))")
        assert executed["direct_control"]["execution"].startswith("D_s(X)")
        assert executed["direct_control"]["prompt_key"] == "d_image_to_alias"
        assert executed["hybrid_conflict_probe"]["auxiliary"] is True
        assert executed["hybrid_conflict_probe"]["prompt_key"] == \
            "hybrid_image_and_code"

        # -- the cell structure follows from what each kind depends on -------
        n_g, n_h = len(doc["router_seeds"]), len(doc["edit_seeds"])
        n_d = len(doc["direct_seeds"])
        kinds = {k: len(v) for k, v in doc["cells_by_kind"].items()}
        assert kinds["intervention"] == n_h, \
            "forced-code interventions do not depend on g, so they are " \
            "evaluated once per EDIT seed and never multiplied by router seed"
        assert kinds["natural"] == n_g * n_h
        assert kinds["direct"] == kinds["hybrid"] == n_d
        assert doc["n_cells"] == sum(kinds.values())

        # -- real checkpoint requirements ------------------------------------
        roles = doc["checkpoint_requirements"]["roles"]
        assert "direct_condition" not in roles, \
            "v1's role whose adapter was 'reuses edited_h'"
        for seed in doc["router_seeds"]:
            role = roles[f"router_g__seed{seed}"]
            assert role["adapter"] and role["held_out_predictions"]
            assert role["n_files_required"] == 2
            assert role["exists_already"] is (seed == rf.EXISTING_ROUTER_SEED)
        for seed in doc["direct_seeds"]:
            role = roles[f"direct_d__seed{seed}"]
            assert role["exists_already"] is False
            assert "never edited when h is edited" in role["role"]
        cr = doc["checkpoint_requirements"]
        assert cr["must_be_trained_before_this_pilot_can_run"], \
            "a pilot frozen before its training work exists must name it"
        assert cr["complete"] is False

        # -- identity-level inference ----------------------------------------
        boot = doc["cluster_bootstrap"]
        assert boot["primary"]["level"] == "identity"
        assert boot["sensitivity"]["level"] == "image"
        assert boot["repeated_intervention_rows_are_never_independent"]
        if ds == "salmu":
            # 12 people, three held-out images each: the counts item 6 asks for
            assert boot["primary_n_clusters"] == 12
            assert boot["sensitivity_n_clusters"] == 36
            assert boot["exploratory"] is False
        else:
            assert boot["primary_n_clusters"] == 4
            assert boot["exploratory"] is True
            assert "byte" in boot["exploratory_because"] or \
                "train split" in boot["exploratory_because"]

        # -- separate verdicts -----------------------------------------------
        vtbr = doc["verdicts_to_be_reported"]
        assert vtbr["names"] == list(rf.VERDICT_NAMES)
        assert vtbr["no_omnibus_boolean"]
        assert "passed" not in doc, "no omnibus boolean anywhere in the record"
        assert set(doc["frozen_gates"]["gate_to_verdict"]) == \
            {"router_held_out_accuracy",
             "conditional_forgotten_route_suppression",
             "conditional_retained_route_accuracy",
             "mediator_intervention_following_accuracy",
             "direct_image_accuracy", "direct_code_following_rate",
             "no_unparseable_or_multi_label_outputs",
             "e2e_matches_the_routing_composition"}
        th = doc["frozen_gates"]["thresholds"]
        assert th["min_direct_image_accuracy"] == 0.90
        assert th["max_direct_code_following_rate"] == 0.10

        # -- P_h is the hard response, not a score sum ------------------------
        e2e = doc["e2e_decomposition"]
        assert e2e["prediction_formula"] == "sum_c P_g(c|X) * H_e(c, Y*(X))"
        assert e2e["p_h_is"].startswith("the empirical hard-response matrix")
        assert "score sum" in e2e["p_h_is_not"]
        assert e2e["if_stochastic_decoding_is_ever_introduced"]

        # -- and it supersedes the v1 files by digest, not by mention ---------
        sup = {e["path"]: e for e in doc["supersession"]["superseded"]}
        assert set(sup) == {rf._rel(p) for p in _V1_MANIFESTS}
        for p in _V1_MANIFESTS:
            e = sup[rf._rel(p)]
            assert e["status"] == "superseded_by_v2"
            assert e["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest(), \
                "the v2 record and the v1 bytes disagree: a superseded " \
                "pre-registration was edited"
            assert e["bytes"] == p.stat().st_size


@_needs_v2_manifests
def test_the_frozen_v2_pilots_spend_the_router_budget_where_it_buys_something(
        rf):
    """Item 3: SALMU gets the real 3g x 3h grid because it has held-out image
    content; PPUBench stays at one replayed router and three edited-h cells,
    because a second router trained on repeated bytes could only measure seed
    stability and no verdict would use it."""
    docs = {("salmu" if "salmu" in p.name else "ppubench"):
            json.loads(p.read_text(encoding="utf-8")) for p in _V2_MANIFESTS}
    assert docs["salmu"]["router_seeds"] == [17, 42, 123]
    assert docs["salmu"]["direct_seeds"] == [17, 42, 123]
    assert docs["ppubench"]["router_seeds"] == [rf.EXISTING_ROUTER_SEED]
    assert docs["ppubench"]["direct_seeds"] == [rf.EXISTING_ROUTER_SEED]
    for ds, doc in docs.items():
        assert doc["edit_seeds"] == [17, 42, 123], ds
        policy = doc["seed_policy"]
        assert policy["router_seeds"] == doc["router_seeds"], ds
        assert policy["held_out_image_content_exists"] is (ds == "salmu"), ds
    must = {ds: set(doc["checkpoint_requirements"]
                    ["must_be_trained_before_this_pilot_can_run"])
            for ds, doc in docs.items()}
    assert must["salmu"] == {"router_g__seed42", "router_g__seed123",
                             "direct_d__seed17", "direct_d__seed42",
                             "direct_d__seed123"}
    assert must["ppubench"] == {"direct_d__seed17"}
    assert rf.EXISTING_ROUTER_SEED not in {
        int(r.split("seed")[1]) for r in must["salmu"]
        if r.startswith("router_g")}, \
        "the frozen g the route was established with is read, never retrained"


@_needs_v2_manifests
def test_rf0_reports_the_frozen_v2_pilots_against_the_tree(rf, tmp_path):
    """RF0 is the phase that says what is missing.

    The manifest is named explicitly: RF0 without a path reads the LATEST
    version's canonical name, and a test that meant the v2 pilots but got the v4
    one would report on a design it did not load.

    WHAT MUST BE TRAINED is computed live rather than read out of the frozen
    record, so it is the same answer in a fresh clone with no weights and on the
    machine that has them -- which is the repair, because reading it from a block
    frozen before the training existed could never say the training happened.
    Every quantity below is derived from the tree for the same reason: this pilot
    has since had its training done by v3, and a test pinned at "still missing"
    would be pinning a date rather than a rule.
    """
    for path in _V2_MANIFESTS:
        ds = "salmu" if "salmu" in path.name else "ppubench"
        doc = json.loads(path.read_text(encoding="utf-8"))
        rep = rf.phase_rf0(ds, path, tmp_path)
        assert rep["phase"] == "RF0" and rep["dataset"] == ds
        assert rep["kind"] == rf.PREREG_V2_KIND
        assert rep["design_version"] == "v2", \
            "RF0 reports the version of the manifest it was handed, not the " \
            "module default, or a v2 report would be filed under a v4 name"
        assert rep["manifest"] == rf._rel(path)
        assert rep["executed"] is False
        assert rep["manifest_valid"] == (rep["problems"] == []), \
            "invalid without naming a problem is a refusal with no reason"
        assert rep["runnable_now"] is False
        assert rep["why_not_runnable"]
        assert "does not verify" in rep["why_not_runnable"], \
            "a superseded pilot binds an older runner, and that alone is enough " \
            "to keep it out of reach of the phases"
        # the live answer, computed the way the frozen block could not have been
        # after it was written: the declared training work minus whatever has
        # since arrived
        assert rep["readiness_is_computed_live"] is True
        assert set(rep["must_be_trained_before_this_pilot_can_run"]) == (
            set(doc["checkpoint_requirements"]
                ["must_be_trained_before_this_pilot_can_run"])
            - set(rep["roles_present"])), (ds, rep["roles_present"])
        assert rep["n_must_be_trained"] == len(
            rep["must_be_trained_before_this_pilot_can_run"])
        assert rep["n_cells_filed"] + rep["n_cells_missing"] == \
            rep["n_cells"] == doc["n_cells"]
        assert rep["held_out_image_drift_since_freeze"] == [], \
            "the dataset manifests are tracked; a drift here means the pilot " \
            "was frozen over images this tree no longer names"
        frozen_router = rep["routers"][str(rf.EXISTING_ROUTER_SEED)]
        assert frozen_router["predictions_present"] is True, \
            "the frozen g's predictions are the route's own committed cache"
        acc = frozen_router["held_out_accuracy"]
        assert acc and not frozen_router["accuracy_error"]
        # Recomputed from the prediction file rather than read out of
        # HELD_OUT_G: the table carries the published rounding and RF0 carries
        # the fraction, and a report that echoed the table would still say
        # 0.8056 after the cache it describes had been replaced.
        assert acc["n_correct"] / acc["n"] == acc["accuracy"]
        assert round(acc["accuracy"], 4) == rf.HELD_OUT_G[ds]["accuracy"], \
            (ds, acc)
        assert acc["n"] == rep["n_held_out_images"], \
            "an accuracy over fewer images than the design names is an accuracy " \
            "over a subset"
        for seed in doc["router_seeds"]:
            entry = rep["routers"][str(seed)]
            # Derived from the file rather than pinned at "RF1G has not run":
            # it has since run for every seed v3 declared, and the predictions
            # are committed.
            assert entry["predictions_present"] is \
                rf.router_prediction_path(ds, seed).is_file()
            if not entry["predictions_present"]:
                continue
            got = entry["held_out_accuracy"]
            assert got and not entry["accuracy_error"], entry
            assert got["n_correct"] / got["n"] == got["accuracy"]
            assert got["n"] == rep["n_held_out_images"]


@_needs_v2_manifests
def test_the_phases_read_the_frozen_v2_pilots_and_nothing_else(rf):
    for path in _V2_MANIFESTS:
        ds = "salmu" if "salmu" in path.name else "ppubench"
        assert rf.prereg_path_v2(ds) == path
        doc = rf.load_prereg_v2(ds, verify=False)
        assert doc["dataset"] == ds
        assert doc["kind"] == rf.PREREG_V2_KIND
        # verify=False is the only reading that works without the gitignored
        # weights; the verified reading is RF0's job and is asserted above


@_needs_v2_verifiable
def test_the_frozen_v2_pilots_show_the_defect_they_were_superseded_for(
        rf, tmp_path):
    """Each v2 pilot is still readable, still names its own design, and still
    refuses to load -- and the reason it refuses is now measurable.

    When this test was written the v2 pilots were waiting for training that had
    not happened, so "v2's ``design_sha256`` covered which adapters were on
    disk" was a reading of the constructor.  v3 then trained the same roles, and
    the reading became an observation: the work arrived and the artifact stopped
    reproducing at that moment, which is the exact sequence the supersession
    record predicts and the reason a v2 pilot was unrunnable by construction.

    Both states are asserted, because which one a checkout is in depends on
    whether the gitignored adapters are there, and a test that only knew one of
    them would fail in a fresh clone instead of describing it.

    Gated on the images and the weights because the rebuild re-hashes both; the
    record-reading tests above run in a fresh clone.

    RF0 files its report under ``tmp_path``: a test that writes into the
    repository leaves an untracked artifact behind, and CI fails on a tree that
    is not clean after the suite has run.
    """
    for path in _V2_MANIFESTS:
        ds = "salmu" if "salmu" in path.name else "ppubench"
        got = rf.verify_manifest(path)
        assert got["kind"] == rf.PREREG_V2_KIND
        assert got["executed"] is False
        assert got["valid"] is False, (path.name, got["problems"])
        drift = [p for p in got["problems"] if p.startswith("input ")]
        rebuilt = [p for p in got["problems"]
                   if "does not reproduce design_sha256" in p]
        trained = got["roles_trained_since_freeze"]
        assert drift, "a superseded pilot always binds an older runner"
        assert all("e2c_v3_route_dependent_forgetting.py" in p for p in drift), \
            drift
        assert set(got["problems"]) == set(drift) | set(rebuilt), \
            "a problem that is neither the runner's own drift nor the rebuild " \
            "failing is one this test cannot explain: " \
            f"{got['problems']}"
        assert got["n_checkpoints_rehashed"] > 0, \
            "a pilot that hashed nothing binds no weights"
        assert got["checkpoint_readiness"] is not None
        assert got["checkpoint_readiness"]["computed_live_not_read_from_the_"
                                          "frozen_design"] is True
        ready = got["checkpoint_readiness"]
        if trained:
            assert rebuilt, (
                f"{len(trained)} role(s) this pilot was waiting for have "
                f"arrived ({sorted(t['role'] for t in trained)}) and the design "
                "still reproduces, which would mean its hash never covered live "
                "checkpoint status after all")
            assert ready["runnable_now"] is True and ready["complete"] is True
            assert ready["must_be_trained"] == []
            assert got["must_be_trained"] == []
            assert got["checkpoints_complete"] is True
        else:
            # Nothing has arrived here, so the constructor embeds the same
            # absence it froze and only the runner's bytes have moved.
            assert not rebuilt, got["problems"]
            assert got["must_be_trained"], \
                "frozen, not executed, and nothing trained in this checkout"
            assert got["checkpoints_complete"] is False
            assert ready["runnable_now"] is False
        # the verified reading refuses a drifted manifest, so the design hash is
        # compared through the unverified one; refusing is the point, not a bug
        with pytest.raises(RuntimeError, match="does not verify"):
            rf.load_prereg_v2(ds)
        assert rf.load_prereg_v2(ds, verify=False)["design_sha256"] == \
            got["design_sha256"]
        rep = rf.phase_rf0(ds, path, tmp_path)
        assert rep["manifest_valid"] is False, rep["problems"]
        assert rep["runnable_now"] is False
        assert "does not verify" in rep["why_not_runnable"], \
            "every reason, not the first one: an operator who fixed the drift " \
            "and re-ran would otherwise meet the next reason as a surprise"
        if not trained:
            assert "must be trained first" in rep["why_not_runnable"]


def test_out_is_a_directory_in_every_v2_phase_that_writes(rf, tmp_path):
    """One flag, one meaning.

    RF0 used to take ``--out`` as a FILE while RF2 and RF2P took it as a
    directory, so the same command line filed a report in one phase and died
    inside ``os.replace`` with an IsADirectoryError in another -- an error about
    a filesystem call, where the actual subject is what the flag means.

    Every name carries its version, so a v3 report cannot land on a v2 one: the
    two designs do not have the same cells, and a report whose name said v2 while
    its contents were v3 would be read against the wrong design.
    """
    for spec, tag in ((rf.PILOT_SPEC_V2, "v2"), (rf.PILOT_SPEC_V3, "v3")):
        assert rf.report_path("salmu", "RF0", tmp_path, spec) == \
            tmp_path / f"rf0_salmu_{tag}.json"
        assert rf.report_path("salmu", "RF2", tmp_path, spec) == \
            tmp_path / f"rf_report_salmu_{tag}.json"
        assert rf.report_path("salmu", "RF2P", tmp_path, spec) == \
            tmp_path / f"rf_report_salmu_{tag}_rescored.json"
    # and the default is the current version, one constant rather than a default
    # repeated per phase
    assert rf.report_path("salmu", "RF0", tmp_path) == \
        rf.report_path("salmu", "RF0", tmp_path, rf.LATEST_PILOT_SPEC)
    names = [rf.report_path("salmu", k, tmp_path).name
             for k in rf.REPORT_FILENAMES]
    assert len(set(names)) == len(names), \
        "RF2P reproduces RF2; a reproduction that overwrote the original would " \
        "leave no way to see whether the two agreed"
    # a phase that files no report says so instead of raising a KeyError from a
    # dict lookup, which is a stack trace rather than a refusal
    with pytest.raises(RuntimeError, match="files no report"):
        rf.report_path("salmu", "RF1G", tmp_path)


def test_the_cli_freezes_into_out_under_the_canonical_name(rf, tmp_path,
                                                           monkeypatch):
    """``--out`` is the directory this run writes into, and the manifest keeps
    the name the phases look for inside it: a staged artifact written under some
    other name is one that has to be renamed by hand, and renaming by hand is
    how a manifest gets placed at a path nothing reads."""
    man = _disk_manifest(tmp_path)
    _stub_tree(rf, monkeypatch)
    monkeypatch.setattr(rf, "load_manifest", lambda ds: man)
    monkeypatch.setattr(rf, "pilot_forget_set",
                        lambda ds, seeds: dict(SELECTION))
    cases = [(["--design-version", "v2"], "rf_pilot_ppubench_v2.json",
              rf.prereg_path_v2),
             (["--design-version", "v3"], "rf_pilot_ppubench_v3.json",
              rf.prereg_path_v3)]
    # Freezing v4 reads the committed v3 record its scope amendment cites -- both
    # datasets' RF2 reports and both v3 manifests -- and refuses without it,
    # because an amendment whose evidence cannot be read is an assertion.  So
    # that case runs where the record is present rather than pretending the
    # dependency is not there.
    if all(p.is_file() for p in _V3_RECORD):
        cases.append((["--design-version", "v4"], "rf_pilot_ppubench_v4.json",
                      rf.prereg_path_v4))
    else:
        with pytest.raises(RuntimeError, match="scope amendment cites"):
            rf.main(["--dataset", "ppubench", "--design-version", "v4",
                     "--preregister", "--out", str(tmp_path / "nov4")])
    # No flag means the authoritative version, which is v5 -- and v5 refuses to
    # freeze before RFC has selected a configuration.  That refusal IS the
    # ordering the version exists to enforce, so it is asserted here rather than
    # stubbed away: a test that supplied a fake selection to reach the write
    # would be testing the write and calling it a test of the order.
    with pytest.raises(RuntimeError, match="no calibration selection is filed"):
        rf.main(["--dataset", "ppubench", "--preregister",
                 "--out", str(tmp_path / "nov5")])
    for argv, expected, getter in cases:
        stage = tmp_path / (expected.split("_")[-1].split(".")[0])
        assert rf.main(["--dataset", "ppubench", "--preregister",
                        "--out", str(stage), *argv]) == 0
        placed = stage / expected
        assert placed.is_file(), sorted(p.name for p in stage.iterdir())
        assert placed == getter("ppubench", placed), \
            "the canonical name is the one the phases resolve, not a name this " \
            "test happens to expect"
        assert rf.verify_manifest(placed)["valid"] is True, \
            rf.verify_manifest(placed)["problems"]
        assert sorted(p.name for p in stage.iterdir()) == [placed.name], \
            "a staging directory that also accumulated something else is a " \
            "directory whose bytes are not the bytes that get placed"
