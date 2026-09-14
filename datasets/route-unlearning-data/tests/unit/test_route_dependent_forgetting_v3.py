"""Route-dependent forgetting, v3: the launch blockers, pinned one per test.

v2 was frozen and never executed, and four things about it made it unrunnable
rather than merely imperfect -- plus three scientific repairs worth making before
spending GPU time:

  1. the frozen design embedded the LIVE ``checkpoint_requirements.verification``
     block, so ``design_sha256`` covered which adapters had been trained: the
     training that made a pilot runnable was the training that made it
     unverifiable and unloadable, and RF0 read ``complete`` out of the frozen
     document so ``runnable_now`` could never become True;
  2. training opened images with ``Image.open(uri)`` while evaluation resolved
     through ``resolve_recorded_path``, so SALMU -- which records relative URIs --
     trained only when launched from exactly the dataset root;
  3. RF1G never loaded the pre-registration, so ``--router-seed 999`` could
     consume a full GPU run that no design named and no verdict could use;
  4. a filed cell bound its prompts and the weights it selected but not the code
     that executed it, and RF2 hashed none of the cells it aggregated, so an
     adapter replaced after a cell was filed left a valid-looking aggregate;
  5. ``baseline_h`` was declared as "the before in before/after" and never
     evaluated, so the suppression gates were satisfiable by an h that answered
     Unknown to every code whatever;
  6. the direct-model gates pooled all three seeds, so one good adapter could
     carry two that failed the floor and the pooled rate still read as a
     measurement of "the direct pathway";
  7. the top-level comment still claimed PPUBench has genuinely held-out
     same-person images, which its own content audit contradicts.

Each test below names the repair it pins.  They exercise the CPU layers only --
no test loads a model -- and the ones that need the real dataset bytes or the
gitignored adapter weights say so and skip with the absent path named.
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


def _load():
    spec = importlib.util.spec_from_file_location(
        "route_forgetting_v3_under_test",
        _SCRIPTS / "e2c_v3_route_dependent_forgetting.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rf():
    return _load()


_MANIFESTS = {"ppubench": _ROOT / "e2c_v3_real" / "manifests"
              / "realdata_identity_mapping.json",
              "salmu": _ROOT / "e2c_salmu" / "manifests"
              / "salmu_manifest.json"}
_V2_MANIFESTS = [_ROOT / "e2c_route_forgetting" / "manifests" / n
                 for n in ("rf_pilot_ppubench_v2.json", "rf_pilot_salmu_v2.json")]
_V1_MANIFESTS = [_ROOT / "e2c_route_forgetting" / "manifests" / n
                 for n in ("rf_pilot_ppubench.json", "rf_pilot_salmu.json")]
_needs_superseded = pytest.mark.skipif(
    not all(p.is_file() for p in _V1_MANIFESTS + _V2_MANIFESTS),
    reason=("the superseded pilots are not all in this checkout: "
            + ", ".join(str(p) for p in _V1_MANIFESTS + _V2_MANIFESTS
                        if not p.is_file())))

#: The authoritative pilots.  Tracked, so a fresh clone has them -- and the tests
#: that read them are deliberately NOT gated on the gitignored adapter weights
#: being present, because "this manifest verifies in a clone that has no weights"
#: is the property item 1 exists for.  A gate on the weights would skip exactly
#: the checkout the property is about.
_V3_MANIFESTS = [_ROOT / "e2c_route_forgetting" / "manifests" / n
                 for n in ("rf_pilot_ppubench_v3.json", "rf_pilot_salmu_v3.json")]
_BY_DATASET = {("salmu" if "salmu" in p.name else "ppubench"): p
               for p in _V3_MANIFESTS}
_needs_v3_manifests = pytest.mark.skipif(
    not all(p.is_file() for p in _V3_MANIFESTS),
    reason=("the v3 pilots are not frozen in this checkout: "
            + ", ".join(str(p) for p in _V3_MANIFESTS if not p.is_file())))


def _absent_images(dataset):
    """The first image the dataset manifest names that is not on disk.

    PPUBench's images live OUTSIDE the repository and SALMU's are gitignored, so
    a fresh clone has both manifests and neither set of bytes.
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
            "(PPUBench) or gitignored (SALMU) -- "
            + "; ".join(f"{ds}: {p}" for ds, p in sorted(_ABSENT.items()) if p)))


# --------------------------------------------------------------------------
# a hermetic pilot whose weights are files this test controls
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


def _write_png(path, identity_index, salt=0):
    """A real, decodable 4x4 PNG whose first pixel names its identity.

    Real bytes rather than a PNG signature with text after it: the training path
    under test OPENS these with PIL, and a file that only looks like an image
    would fail for a reason that has nothing to do with where it was looked for.

    The identity in the first pixel is what lets a stubbed session answer from
    the image it was actually handed, so a test can tell that the RIGHT bytes
    reached the model rather than merely that some bytes did.

    ``salt`` keeps two images of one identity from being byte-identical.  Without
    it every test image duplicates its own train image, which is PPUBench's
    condition rather than SALMU's, and the content audit would then report no
    held-out content and the seed policy would spend no router budget -- so the
    fixture would silently be testing a different dataset's design.
    """
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (identity_index, salt % 251, (salt * 7) % 251)).save(path)
    return str(path)


def _disk_manifest(tmp_path, relative_uris=False, n_test=3,
                   identity_ids=("001", "002", "003", "004")):
    """A manifest whose ``image_uri`` values point at files that exist.

    ``relative_uris`` reproduces SALMU's shape: the URI is recorded relative to
    the dataset root rather than absolute, which is the condition item 2 turns
    on.  The bytes are written at that same relative location under ``tmp_path``,
    and the caller points ``DATASET_ROOT`` at it so the recorded form resolves
    there and nowhere near the repository.
    """
    items = []
    for k, iid in enumerate(identity_ids):
        for split, n in (("train", 2), ("test", n_test)):
            for i in range(n):
                name = f"{iid}_{split}_{i}.png"
                rel = Path("e2c_salmu") / "images" / name
                _write_png(tmp_path / rel, k, salt=k * 31 + i * 7 + len(split))
                items.append({"identity_id": iid, "split": split,
                              "image_uri": (str(rel) if relative_uris
                                            else str(tmp_path / rel))})
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


def _stub_tree_except_the_probe(rf, monkeypatch):
    """Everything a hermetic build needs stubbed EXCEPT the checkpoint prober.

    The prober is the subject: item 1 is about where its answer is recorded, so a
    test that stubbed it would stub the thing it is testing.  The forget sets and
    the reuse inventory are properties of the frozen matrix and stay stubbed.
    """
    monkeypatch.setattr(rf, "_promotion_forget_sets",
                        lambda ds: [f"fs_{i:03d}" for i in range(1, 8)])
    monkeypatch.setattr(rf, "reuse_inventory",
                        lambda ds, sets, seeds: dict(FAKE_INVENTORY))


def _roles_under_tmp(rf, tmp, dataset, router_seeds, edit_seeds, direct_seeds):
    """A role table whose every file lives under ``tmp``, and where each is.

    Recorded ABSOLUTE, because ``resolve_recorded_path`` accepts that shape and a
    relative one would resolve against the real dataset root -- which is how a
    test that meant to create a checkpoint ends up creating one in the
    repository.

    The recorded RELATIVE SHAPE is kept under the temporary root rather than
    flattened, because the phases derive directories from it: RF1G trains into
    ``ckpt.parents[1]`` and requires the prediction file to be in that same
    directory, so a flattened layout would fail a path check that has nothing to
    do with what the test is about.
    """
    real = rf.required_checkpoints_v2(dataset, "fs_001", router_seeds,
                                      edit_seeds, direct_seeds)
    roles, files = OrderedDict(), {}
    for name, decl in real["roles"].items():
        entry = dict(decl)
        for key in rf.CHECKPOINT_FILE_KEYS_V2:
            raw = entry.get(key)
            if not raw or raw == "reuses edited_h":
                continue
            rel = Path(raw)
            if rel.is_absolute():
                rel = rel.relative_to(rel.anchor)
            p = tmp / "weights" / rel
            entry[key] = str(p)
            files[(name, key)] = p
        roles[name] = entry
    return {**real, "roles": roles}, files


def _arrive(files, role):
    """Make one role's declared files appear, as a finished training run would.

    Written with distinct bytes per role and per file key, so the digests that
    appear are the digests of something and a later re-hash has a real value to
    compare against.
    """
    made = []
    for (name, key), p in sorted(files.items()):
        if name != role:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"{name}:{key}:trained".encode())
        made.append(p)
    assert made, f"role {role} declares no file to arrive"
    return made


def _hermetic_pilot(rf, tmp_path, monkeypatch, version="v3", arrived=(),
                    dataset="salmu", man=None):
    """One hermetic pilot whose weights are files this test controls.

    The roles the design declares as ALREADY on disk are materialized here,
    because that is what the declaration means: they are inputs the design was
    built over, not training work.  What is left absent is the pending work, so
    the lifecycle starts where a real freeze starts.

    ``arrived`` names additional roles whose files exist WHEN THE DESIGN IS
    BUILT.  The returned ``files`` map lets a test make more of them arrive
    afterwards, which is the whole point: item 1 is about whether a design that
    has already been built and frozen survives its own training completing.

    SALMU is the default dataset because its seed policy varies all three
    factors, so there are five pending roles to arrive one at a time; PPUBench's
    policy spends no router budget and leaves only one.
    """
    man = man if man is not None else _disk_manifest(tmp_path)
    _stub_tree_except_the_probe(rf, monkeypatch)
    monkeypatch.setattr(rf, "load_manifest", lambda ds: man)
    monkeypatch.setattr(rf, "pilot_forget_set",
                        lambda ds, seeds: dict(SELECTION))
    policy = rf.pilot_seed_policy_v2(dataset, man=man)
    router_seeds = policy["router_seeds"]
    direct_seeds = policy["direct_seeds"]
    edit_seeds = [17, 42, 123]
    table, files = _roles_under_tmp(rf, tmp_path, dataset, router_seeds,
                                    edit_seeds, direct_seeds)
    monkeypatch.setattr(rf, "required_checkpoints_v2",
                        lambda *a, **kw: dict(table))
    declared_present = sorted(r for r, d in table["roles"].items()
                              if d.get("exists_already"))
    for role in [*declared_present, *arrived]:
        _arrive(files, role)
    build = (rf.build_pilot_preregistration_v3 if version == "v3"
             else rf.build_pilot_preregistration_v2)
    block = build(dataset, "fs_001", ["001"], router_seeds, edit_seeds,
                  direct_seeds, man=man, images=rf.held_out_images(man),
                  selection=dict(SELECTION))
    pending = sorted(r for r, d in table["roles"].items()
                     if not d.get("exists_already"))
    return {"rf": rf, "block": block, "man": man, "files": files,
            "table": table, "dataset": dataset, "version": version,
            "router_seeds": router_seeds, "edit_seeds": edit_seeds,
            "direct_seeds": direct_seeds, "pending": pending,
            "declared_present": declared_present, "tmp": tmp_path}


def _design_hash(rf, block):
    """The hash a verifier recomputes: the design minus its own sealing fields."""
    return rf.design_sha256({k: v for k, v in block.items()
                             if k not in ("frozen", "design_sha256",
                                          "provenance")})


def _freeze(rf, block, tmp_path, name):
    """Freeze a hermetic pilot where the phases will look for it."""
    path = tmp_path / "manifests" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    out, _digest = rf.freeze_manifest(block, path)
    return Path(out)


# --------------------------------------------------------------------------
# a stub session: the model is fabricated, the logic under test is not
# --------------------------------------------------------------------------

class _StubAdapter:
    """Records every supervised example the training loop built."""

    def __init__(self, seen):
        self.seen = seen

    def build_supervised_example(self, processor, image=None, prompt=None,
                                 answer_text=None):
        example = {"image": image, "prompt": prompt, "answer": answer_text}
        self.seen.append(example)
        return example


class _StubRv:
    """The two research-validity entry points a phase touches."""

    def __init__(self, adapter_relpath):
        # read off the module rather than retyped here: the phases refuse to
        # continue unless the adapter exists at the path the design declared, so
        # a stub that wrote it anywhere else would test the refusal instead of
        # the phase
        self.adapter_relpath = tuple(adapter_relpath)

    def train_supervised(self, condition, adapter, model, processor, examples,
                         output_dir, device, steps=None, warmup=None, lr=None):
        p = Path(output_dir).joinpath(*self.adapter_relpath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"adapter:{condition}:{steps}:{lr}".encode())
        return p

    def _extract_code(self, raw, codes):
        return raw if raw in codes else None


class _StubMx:
    def seed_everything(self, seed):
        self.seeded = seed


class _StubHandle:
    def __init__(self, seen):
        self.adapter = _StubAdapter(seen)
        self.processor = object()
        self.model = object()

    def release(self):
        pass


def _fake_session(rf, man, forget=(), behaves="design", role="h", trained=None):
    """A stand-in for ``RouteSessionV2`` that behaves like a model with stated
    properties.

    ``role`` selects which part of the route the stub stands in for, because the
    three take different inputs and give different outputs:

      "g"  sees an image and emits a code
      "h"  sees a code and NO image, and emits an alias
      "d"  sees an image and emits an alias, with no code presented

    ``behaves="baseline"`` is the UNEDITED h: every code produces its own alias,
    forgotten ones included.  ``behaves="design"`` is the post-edit h: a
    forgotten code produces the refusal label and every other code its alias.

    ``train`` opens each image through the module's own ``_open_image`` and
    records the resulting object, so a test can see WHICH BYTES training reached
    rather than which path it asked for.
    """
    trained = [] if trained is None else trained
    ids = list(man["identity_ids"])
    aliases = man["alias_of"]
    codes = man["code_of"]

    def _identity_of_image(image):
        assert image is not None, f"role {role} is presented an image"
        return ids[image.getpixel((0, 0))[0]]

    class _Session:
        def __init__(self, device, adapter_name, seed):
            self.device = device
            self.adapter_name = adapter_name
            self.seed = seed
            self.session = _StubHandle(trained)
            self.mx = _StubMx()
            self.rv = _StubRv(rf.ADAPTER_RELPATH)
            self._backend = None

        def reset_to(self, ckpt):
            self.ckpt = Path(ckpt)

        def reset_fresh(self, seed):
            self.ckpt = None

        def train(self, condition, pairs, output_dir, protocol):
            for uri, prompt, answer in pairs:
                self.session.adapter.build_supervised_example(
                    self.session.processor, image=rf._open_image(uri),
                    prompt=prompt, answer_text=answer)
            self.rv.train_supervised(condition, self.session.adapter,
                                     self.session.model,
                                     self.session.processor, trained,
                                     output_dir, self.device,
                                     steps=protocol["steps"],
                                     warmup=protocol["warmup"],
                                     lr=protocol["lr"])
            self._backend = None
            return Path(output_dir)

        def generate(self, image, prompt, max_new_tokens=8):
            if role == "g":
                return codes[_identity_of_image(image)]
            if role == "d":
                return aliases[_identity_of_image(image)]
            # role "h": the mediator is hard, so h is never presented an image
            assert image is None, "the mediator is hard: h sees no image"
            # found by which code the prompt names rather than by parsing the
            # template's punctuation, so a change to the frozen prompt breaks the
            # stub visibly instead of silently producing the wrong alias
            named = [codes[i] for i in ids if codes[i] in prompt]
            assert len(named) == 1, \
                f"the prompt names no single code: {prompt!r}"
            iid = next(i for i in ids if codes[i] == named[0])
            if behaves == "baseline":
                return aliases[iid]
            return DELETED if iid in forget else aliases[iid]

        def release(self):
            self._backend = None
            self.session.release()

    return _Session


def _forbid_the_model(rf, monkeypatch):
    """Make any attempt to construct a real session fail immediately and loudly.

    These tests exercise the layers AROUND the model.  A test that reached
    ``RouteSessionV2`` would fetch and load a 9B checkpoint: slow, GPU-shaped,
    and it would turn an assertion about logic into one about the environment --
    and it would do so only for whichever test forgot, which is how a suite ends
    up passing locally and hanging in CI.
    """
    def _no(*args, **kwargs):
        raise AssertionError(
            "this test reached the real model session; stub RouteSessionV2 with "
            "_fake_session, or the assertion belongs in a GPU test")
    monkeypatch.setattr(rf, "RouteSessionV2", _no)


@pytest.fixture(autouse=True)
def _no_real_model(rf, monkeypatch):
    """Autouse, so the guarantee does not rest on each test remembering to ask.

    A test that needs a session replaces ``RouteSessionV2`` itself, and its own
    ``monkeypatch.setattr`` runs after this fixture and therefore wins.
    """
    _forbid_the_model(rf, monkeypatch)


# ==========================================================================
# item 1 -- a pre-registration survives its own training
# ==========================================================================

def test_the_v3_design_hash_does_not_move_when_the_training_arrives(
        rf, tmp_path, monkeypatch):
    """The repair, stated as an invariance.

    A design may bind the bytes it is a statement ABOUT -- the images, the
    dataset manifest, the prompts, the thresholds -- because those are inputs and
    a change in them means the design now describes something else.  Which of the
    adapters it asks to be trained have arrived is not an input, it is PROGRESS,
    and hashing progress makes a pre-registration invalidate itself as its own
    work completes.
    """
    pilot = _hermetic_pilot(rf, tmp_path, monkeypatch)
    assert pilot["pending"], "a pilot with no training work cannot show the bug"
    assert len(pilot["pending"]) > 1, \
        "one pending role cannot show the middle of the lifecycle, which is the " \
        "stage v2 broke at"
    block, files = pilot["block"], pilot["files"]
    before = _design_hash(rf, block)
    ready0 = rf.checkpoint_readiness(block)
    assert ready0["complete"] is False and ready0["runnable_now"] is False
    assert sorted(ready0["must_be_trained"]) == pilot["pending"]
    assert ready0["unexpectedly_absent"] == [], \
        "every role declared as already present IS present, so the only absent " \
        "ones are training work and not a broken assumption"
    assert ready0["trained_since_declared"] == []

    # one pending role arrives
    first = pilot["pending"][0]
    _arrive(files, first)
    ready1 = rf.checkpoint_readiness(block)
    assert _design_hash(rf, block) == before, \
        "training one role changed the design, so the design hashed progress"
    assert ready1["complete"] is False and ready1["runnable_now"] is False
    assert [t["role"] for t in ready1["trained_since_declared"]] == [first]
    assert ready1["n_must_be_trained"] == len(pilot["pending"]) - 1

    # ... and then all of them
    for role in pilot["pending"][1:]:
        _arrive(files, role)
    ready2 = rf.checkpoint_readiness(block)
    assert _design_hash(rf, block) == before, \
        "the design hash moved when the last role arrived"
    assert ready2["complete"] is True
    assert ready2["runnable_now"] is True
    assert ready2["must_be_trained"] == []
    assert ready2["n_files_present"] >= len(pilot["pending"])
    assert sorted(t["role"] for t in ready2["trained_since_declared"]) == \
        pilot["pending"]
    # and the block itself says where the answer comes from
    req = block["checkpoint_requirements"]
    assert "verification" not in req, \
        "a v3 design freezes no status block, so there is nothing in it that " \
        "could go stale"
    assert req["readiness_is_computed_live_not_frozen"]
    assert req["what_the_design_does_bind"]


def test_the_v2_design_hash_does_move_which_is_why_it_is_superseded(
        rf, tmp_path, monkeypatch):
    """The defect, reproduced rather than described.

    Rebuilt after one role arrives, because v2's constructor runs the live probe
    and hashes its result: the SAME design over a tree that has done some of the
    work produces different bytes.  A pre-registration that changes when the work
    it pre-registered is completed cannot be verified by the run that satisfies
    it, so v2 is superseded for a structural reason and not a scientific one.
    """
    before = _hermetic_pilot(rf, tmp_path, monkeypatch, version="v2")
    empty = _design_hash(rf, before["block"])
    frozen0 = before["block"]["checkpoint_requirements"]["verification"]
    assert frozen0, "v2 froze the live probe's answer inside the design"
    assert frozen0["complete"] is False
    assert frozen0["n_present"] == len(before["declared_present"])
    assert frozen0["must_be_trained"] == before["pending"]

    arrived = tmp_path / "second"
    arrived.mkdir()
    after = _hermetic_pilot(rf, arrived, monkeypatch, version="v2",
                            arrived=(before["pending"][0],),
                            man=before["man"])
    assert _design_hash(rf, after["block"]) != empty, \
        "v2's design did not move when a role arrived, so this test is not " \
        "reproducing the defect it names"
    frozen1 = after["block"]["checkpoint_requirements"]["verification"]
    assert frozen1["n_present"] == frozen0["n_present"] + 1
    assert frozen1["must_be_trained"] == before["pending"][1:]
    # and the same tree under v3 leaves the design where it was
    v3 = _hermetic_pilot(rf, tmp_path / "third", monkeypatch, version="v3",
                         arrived=(before["pending"][0],), man=before["man"])
    assert "verification" not in v3["block"]["checkpoint_requirements"]


def test_a_frozen_v3_pilot_verifies_before_during_and_after_its_own_training(
        rf, tmp_path, monkeypatch):
    """The lifecycle, end to end through ``verify_manifest``.

    Valid before training, still valid after one pending role appears, still
    valid after all of them appear -- and at every one of those stages the design
    it reproduces is the SAME design, which is what makes the artifact a
    pre-registration rather than a description of the tree.
    """
    pilot = _hermetic_pilot(rf, tmp_path, monkeypatch)
    path = _freeze(rf, pilot["block"], tmp_path, "rf_pilot_salmu_v3.json")
    files, pending = pilot["files"], pilot["pending"]

    got0 = rf.verify_manifest(path)
    assert got0["valid"] is True, got0["problems"]
    assert got0["kind"] == rf.PREREG_V3_KIND
    assert got0["executed"] is False
    assert got0["checkpoints_complete"] is False
    assert got0["unexpectedly_absent"] == [], \
        "the roles declared as already present are present, so the only absent " \
        "ones are training work"
    assert got0["checkpoint_readiness"]["runnable_now"] is False
    assert got0["readiness_is_computed_live"]
    frozen_design = got0["design_sha256"]

    _arrive(files, pending[0])
    got1 = rf.verify_manifest(path)
    assert got1["valid"] is True, \
        ("a role arriving invalidated the pre-registration that named it",
         got1["problems"])
    assert got1["design_sha256"] == frozen_design
    assert [t["role"] for t in got1["roles_trained_since_freeze"]] == \
        [pending[0]]
    assert got1["checkpoints_complete"] is False

    for role in pending[1:]:
        _arrive(files, role)
    got2 = rf.verify_manifest(path)
    assert got2["valid"] is True, got2["problems"]
    assert got2["design_sha256"] == frozen_design
    assert got2["checkpoints_complete"] is True
    assert got2["unexpectedly_absent"] == []
    assert got2["must_be_trained"] == []
    assert got2["checkpoint_readiness"]["runnable_now"] is True
    assert got2["n_checkpoints_rehashed"] >= len(pending)


def test_rf0_transitions_to_runnable_now_when_the_last_role_arrives(
        rf, tmp_path, monkeypatch):
    """v2's RF0 read ``complete`` out of the frozen document, so ``runnable_now``
    could never become True however much training was done -- and worse, the
    training that would have made it True drove ``manifest_valid`` False instead,
    so RF0 answered "the manifest does not verify" to a pilot that was ready."""
    pilot = _hermetic_pilot(rf, tmp_path, monkeypatch)
    path = _freeze(rf, pilot["block"], tmp_path, "rf_pilot_salmu_v3.json")
    files, pending, ds = pilot["files"], pilot["pending"], pilot["dataset"]
    out, cells = tmp_path / "reports", tmp_path / "cells"

    rep0 = rf.phase_rf0(ds, path, out, cells)
    assert rep0["manifest_valid"] is True, rep0["problems"]
    assert rep0["design_version"] == "v3"
    assert rep0["runnable_now"] is False
    assert rep0["readiness_is_computed_live"] is True
    assert "must be trained first" in rep0["why_not_runnable"]
    assert set(rep0["must_be_trained_before_this_pilot_can_run"]) == set(pending)
    assert rep0["unexpectedly_absent"] == []
    assert rep0["roles_present"] == pilot["declared_present"]

    _arrive(files, pending[0])
    rep1 = rf.phase_rf0(ds, path, out, cells)
    assert rep1["manifest_valid"] is True, rep1["problems"]
    assert rep1["runnable_now"] is False, "one role of many is not ready"
    assert rep1["roles_present"] == sorted([*pilot["declared_present"],
                                            pending[0]])
    assert set(rep1["must_be_trained_before_this_pilot_can_run"]) == \
        set(pending[1:])

    for role in pending[1:]:
        _arrive(files, role)
    rep2 = rf.phase_rf0(ds, path, out, cells)
    assert rep2["manifest_valid"] is True, rep2["problems"]
    assert rep2["runnable_now"] is True, rep2["why_not_runnable"]
    assert rep2["why_not_runnable"] is None
    assert rep2["checkpoint_requirements_complete"] is True
    assert set(rep2["roles_present"]) == set(pilot["table"]["roles"])
    assert rep2["n_cells_missing"] == rep2["n_cells"], \
        "runnable_now is about INPUTS: it says every weight is present, not that " \
        "any cell has been run or that any gate would pass"
    assert rep2["runnable_now_is_about_inputs_not_about_results"]
    assert rep2["verification_and_readiness_are_separate_questions"]
    # the report is filed under the version of the manifest it read, so a v3 RF0
    # cannot land on a v2 one
    assert (out / "rf0_salmu_v3.json").is_file()
    assert rf.report_path(ds, "RF0", out, rf.PILOT_SPEC_V2).name == \
        "rf0_salmu_v2.json"


def test_the_later_phases_load_the_manifest_the_training_just_satisfied(
        rf, tmp_path, monkeypatch):
    """The consequence item 1 names: RF1H, RF1E, RF2 and RF2P all verify before
    they load, so a pilot invalidated by its own training was a pilot no later
    phase could run at all -- the deadlock was total, not partial."""
    pilot = _hermetic_pilot(rf, tmp_path, monkeypatch)
    path = _freeze(rf, pilot["block"], tmp_path, "rf_pilot_salmu_v3.json")
    ds = pilot["dataset"]
    frozen_design = json.loads(path.read_text(encoding="utf-8"))["design_sha256"]

    # the verified reading refuses nothing at any stage of the lifecycle
    for role in pilot["pending"]:
        doc = rf.load_prereg_v3(ds, path)
        assert doc["design_sha256"] == frozen_design
        assert rf.prereg_path_v3(ds, path) == path
        _arrive(pilot["files"], role)
    doc = rf.load_prereg_v3(ds, path)
    assert doc["design_sha256"] == frozen_design
    assert rf.checkpoint_readiness(doc)["runnable_now"] is True
    # and the phase-level loader agrees, since that is what RF1H/RF1E/RF2 call
    prereg, prov, spec = rf._phase_prereg(ds, None, rf.PILOT_SPEC_V3, path)
    assert spec.version == "v3"
    assert prereg["kind"] == rf.PREREG_V3_KIND
    assert prov["preregistration_file_sha256"] == _sha(path)
    assert prov["preregistration_design_sha256"] == frozen_design


def test_rf1b_files_the_baseline_cell_through_a_stubbed_session(
        rf, tmp_path, monkeypatch):
    """RF1B needs no trained adapter, so it is the one phase runnable end to end
    without a GPU: the session is stubbed and everything around it -- the
    manifest load, the role check, the prompt, the scorer, the filing, the input
    binding -- is the real code."""
    pilot = _hermetic_pilot(rf, tmp_path, monkeypatch, arrived=("baseline_h",))
    ds, man, cells = pilot["dataset"], pilot["man"], tmp_path / "cells"
    path = _freeze(rf, pilot["block"], tmp_path, "rf_pilot_salmu_v3.json")
    prereg = json.loads(path.read_text(encoding="utf-8"))
    monkeypatch.setattr(rf, "RouteSessionV2",
                        _fake_session(rf, man, behaves="baseline"))

    doc = rf.phase_rf1b(ds, prereg=prereg, device="cpu", cells_out=cells)
    assert doc["cell_id"] == "baseline__h_base"
    assert doc["kind_of_cell"] == "baseline"
    assert doc["kind"] == rf.RESULT_KIND_V3
    assert doc["phase"] == "RF1B"
    assert doc["n_rows"] == len(man["identity_ids"]) == doc["n_scored"]
    # every code, a forgotten one included, produced ITS OWN alias: that is the
    # whole content of the before
    for row in doc["scored"]:
        assert row["condition"] == rf.BASELINE_CONDITION_V3
        assert row["parsed_label"] == man["alias_of"][row["forced_identity_id"]]
        assert row["correct"] is True
        assert row["image_to_h"] is False
    # bound to the weights it used, through the table RF2 re-checks against
    assert doc["input_sha256"]["baseline_h"] == _sha(
        rf.resolve_recorded_path(
            pilot["table"]["roles"]["baseline_h"]["adapter"]))
    assert doc["input_sha256"]["preregistration_design"] == \
        prereg["design_sha256"]
    assert doc["run_provenance"]["executing_script_sha256"] == _sha(
        Path(rf.__file__))
    assert rf.cell_result_path(ds, "baseline__h_base", cells).is_file()
    # and it is filed exactly once: the baseline depends on no factor, so a
    # second copy would be one measurement reported twice
    assert [c["cell_id"] for c in prereg["cells"]
            if c["kind"] == "baseline"] == ["baseline__h_base"]


def test_rf1b_refuses_a_pilot_with_no_baseline_cell(rf, tmp_path, monkeypatch):
    """A v2 manifest has no baseline cell, and running RF1B against one would
    measure nothing rather than refusing -- so the refusal names the repair."""
    pilot = _hermetic_pilot(rf, tmp_path, monkeypatch, version="v2")
    with pytest.raises(RuntimeError, match="no cell 'baseline__h_base'"):
        rf.phase_rf1b(pilot["dataset"], prereg=pilot["block"], device="cpu",
                      cells_out=tmp_path / "cells")


def test_rf1b_refuses_an_absent_unedited_h(rf, tmp_path, monkeypatch):
    """The baseline is the route's own h, declared as already present, so its
    absence is a broken assumption about the tree rather than training work to
    wait for -- and the refusal says which of the two it is."""
    pilot = _hermetic_pilot(rf, tmp_path, monkeypatch, arrived=("baseline_h",))
    assert rf.checkpoint_readiness(pilot["block"])["unexpectedly_absent"] == []
    for (name, _key), p in sorted(pilot["files"].items()):
        if name == "baseline_h" and p.is_file():
            p.unlink()
    gone = rf.checkpoint_readiness(pilot["block"])
    assert gone["unexpectedly_absent"] == ["baseline_h"], \
        "a role declared as already present and found absent is not training work"
    assert "baseline_h" not in gone["must_be_trained"]
    with pytest.raises(RuntimeError, match="the unedited h is absent"):
        rf.phase_rf1b(pilot["dataset"], prereg=pilot["block"], device="cpu",
                      cells_out=tmp_path / "cells")


# --------------------------------------------------------------------------
# a whole hermetic v3 pilot: every router written, every cell filed
# --------------------------------------------------------------------------

def _use_this_manifest(rf, monkeypatch, path):
    """Point the phases at a frozen manifest that is not in the repository.

    Patching ``prereg_path_for`` rather than passing the document in, so the
    phase performs its own verified load: a test that handed the document over
    directly would skip the gate item 3 is about.
    """
    monkeypatch.setattr(rf, "prereg_path_for",
                        lambda spec, dataset, p=None: Path(p) if p else path)


def _wrong_label(man, iid, not_this=None):
    """A label in the vocabulary that is neither the right one nor, where a code
    was presented, the one that code names.

    Without ``not_this`` an "arbitrary wrong label" accidentally reproduces the
    forced code's own alias on the hybrid rows, and a model that FOLLOWS the code
    is not the model item 6's ceiling is about.
    """
    for i, alias in sorted(man["alias_of"].items()):
        if i != iid and alias != not_this:
            return alias
    return DELETED


def _fake_output(rf, row, man, forget, behaves, baseline, forced):
    """The text a model with these stated properties would have emitted.

    ``forced`` is the module's own table for the version under test rather than a
    list written down here, so a condition added to a design is simulated by
    whatever rule the design gives it instead of silently falling through.

    ``baseline`` controls the UNEDITED h separately from the edited one, because
    the whole of item 5 is that they can disagree: ``refuses_too`` is an h that
    answered Unknown to a forgotten code before anything edited it, which is the
    model v2's suppression gates could not tell from a real edit.
    """
    cond = row["condition"]
    aliases, codes = man["alias_of"], man["code_of"]
    if cond in forced:
        iid = row["forced_identity_id"]
        if cond == rf.BASELINE_CONDITION_V3:
            if baseline == "refuses_too" and iid in forget:
                return DELETED
            return aliases[iid]
        if behaves == "arbitrary":
            return _wrong_label(man, iid)
        return DELETED if iid in forget else aliases[iid]
    if cond == "natural_mediated":
        code = row["routed_code"]
        if code is None:
            return None
        iid = next(i for i in man["identity_ids"] if codes[i] == code)
        if behaves == "leaks":
            # h answered from the IMAGE rather than the code it was given: the
            # mediator is not hard and the factorization has to catch it
            iid = row["image_identity_id"]
        return DELETED if iid in forget else aliases[iid]
    iid = row["image_identity_id"]
    if behaves == "arbitrary":
        return _wrong_label(man, iid, row.get("forced_label"))
    if behaves == "captured" and row.get("forced_label"):
        return row["forced_label"]
    return aliases[iid]


def _write_routers(rf, prereg, man, monkeypatch, misroutes=None):
    """A per-seed prediction file written WHERE THE DESIGN DECLARED IT.

    Written at the role's own declared path rather than somewhere a patched
    loader is told to look, so the file RF2 re-hashes against a natural cell's
    binding is the file the composition actually read -- one file and one digest
    rather than two that happen to agree.

    ``misroutes[k]`` is the set of held-out positions seed ``k`` gets wrong.  The
    default misroutes a DIFFERENT single image per seed rather than a different
    NUMBER of images: at twelve held-out images the only accuracies clearing the
    0.90 floor are 12/12 and 11/12, so three seeds cannot have three distinct
    accuracies and all pass.  Distinct mistakes are the stronger witness anyway.
    """
    roles = prereg["checkpoint_requirements"]["roles"]
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
            p = rf.resolve_recorded_path(it["image_uri"])
            rows.append({"identity_id": it["identity_id"],
                         "image_uri": str(it["image_uri"]),
                         "split": it["split"],
                         "image_sha256": _sha(p) if p.is_file() else None,
                         "g_raw_text": raw, "pred_code": code,
                         "code_correct": code == own})
        path = rf.resolve_recorded_path(
            roles[f"router_g__seed{seed}"]["held_out_predictions"])
        rf.atomic_write_json(path, {"rows": rows, "router_seed": seed})
        made[seed] = path
    monkeypatch.setattr(rf, "router_prediction_path", lambda ds, s: made[s])
    return made


def _file_all(rf, prereg, man, cells_out, routers, prov, behaves="design",
              baseline="follows_every_code"):
    """File every cell as RF1B, RF1H, RF1D and RF1E would have.

    The recorded ``input_sha256`` is built by the same function the phases build
    it with, so this fixture cannot drift from the contract it files against: a
    test that typed its own binding dict would keep passing after the phases
    changed which weight they hash.
    """
    spec = rf.SPEC_BY_PREREG_KIND[prereg["kind"]]
    forced = spec.forced_code_conditions
    forget = set(prereg["forget_identity_ids"])
    by_uri = {s: {r["image_uri"]: r for r in
                  json.loads(p.read_text(encoding="utf-8"))["rows"]}
              for s, p in routers.items()}
    filed = []
    for cell in prereg["cells"]:
        rows = []
        for r in cell["rows"]:
            row = dict(r)
            if cell["kind"] == "natural":
                got = by_uri[r["router_seed"]][r["image_uri"]]
                row.update({
                    "routed_code": got["pred_code"],
                    "routed_code_raw": got["g_raw_text"],
                    "routed_code_correct": got["code_correct"],
                    "routable": got["pred_code"] is not None,
                    "observation_pending": False, "filled_by": "fixture"})
            row[rf.generation_field(r["condition"], spec.conditions)] = \
                _fake_output(rf, row, man, forget, behaves, baseline, forced)
            rows.append(row)
        scored, missing = rf.score_rows_v2(rows, prereg["vocab"],
                                           spec.conditions)
        assert not missing, f"the fixture left rows ungenerated: {missing}"
        rf.write_cell_result(prereg["dataset"], cell["cell_id"], {
            "kind": spec.result_kind, "cell_id": cell["cell_id"],
            "kind_of_cell": cell["kind"], "phase": cell["phase"],
            "edit_seed": cell.get("edit_seed"),
            "router_seed": cell.get("router_seed"),
            "direct_seed": cell.get("direct_seed"),
            "n_rows": len(rows), "prompts": rf.frozen_route_prompts(),
            "input_sha256": rf._input_sha256(
                prereg, **rf.cell_input_digests(prereg, cell)),
            "run_provenance": prov,
            "rows": rows, "scored": scored,
            "rows_without_a_stored_generation": missing}, cells_out)
        filed.append(cell["cell_id"])
    return filed


@pytest.fixture
def pilot_v3(rf, tmp_path, monkeypatch):
    """A frozen v3 pilot whose every role has arrived, ready to be run."""
    pilot = _hermetic_pilot(rf, tmp_path, monkeypatch)
    assert pilot["router_seeds"] == [17, 42, 123], \
        "the hermetic manifest has genuinely held-out image CONTENT, so the " \
        "router budget is spent and there is a factorial to run"
    assert len(pilot["pending"]) == 5, pilot["pending"]
    for role in pilot["pending"]:
        _arrive(pilot["files"], role)
    path = _freeze(rf, pilot["block"], tmp_path, "rf_pilot_salmu_v3.json")
    prereg = json.loads(path.read_text(encoding="utf-8"))
    assert rf.verify_manifest(path)["valid"] is True, \
        rf.verify_manifest(path)["problems"]
    pilot.update({"path": path, "prereg": prereg})
    return pilot


@pytest.fixture
def run_v3(rf, pilot_v3, monkeypatch, tmp_path):
    """The whole pilot executed: three routers, every cell filed, no aggregate."""
    man, prereg = pilot_v3["man"], pilot_v3["prereg"]
    cells = tmp_path / "cells"
    routers = _write_routers(rf, prereg, man, monkeypatch)
    prov = rf.start_of_run_provenance(
        prereg, pilot_v3["path"], rf.PILOT_SPEC_V3,
        cli=rf._build_parser().parse_args(["--dataset", pilot_v3["dataset"]]))
    filed = _file_all(rf, prereg, man, cells, routers, prov)
    return {"rf": rf, **pilot_v3, "cells": cells, "routers": routers,
            "filed": filed, "prov": prov,
            "loaded": rf._load_cells_and_paths_for(prereg,
                                                   pilot_v3["dataset"], cells)}


# ==========================================================================
# item 2 -- training resolves a recorded path like evaluation does
# ==========================================================================

def test_training_resolves_a_relative_image_uri_from_another_working_directory(
        rf, tmp_path, monkeypatch):
    """SALMU records its image URIs RELATIVE to the dataset root, and training
    opened them with ``Image.open(uri)`` while evaluation resolved them through
    ``resolve_recorded_path``.  So RF1G and RF1D trained only when launched from
    exactly the dataset root, and everywhere else failed as a missing file --
    which reads as broken data rather than as what it is, a launch directory.
    """
    from PIL import Image

    root = tmp_path / "root"
    man = _disk_manifest(root, relative_uris=True)
    assert all(Path(root / it["image_uri"]).is_file() for it in man["items"]), \
        "the fixture has to write the bytes where the recorded URI says they are"
    monkeypatch.setattr(rf, "DATASET_ROOT", root)
    prompts = rf.frozen_route_prompts()
    train = [it for it in man["items"] if it["split"] == "train"]
    pairs = [(it["image_uri"], prompts["g_image_to_code"],
              man["code_of"][it["identity_id"]]) for it in train]
    trained = []
    sess = _fake_session(rf, man, role="g", trained=trained)(
        "cpu", "e2c_rf_g42", 42)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    # the recorded URI is not a path from HERE, which is the whole defect
    assert all(not Path(uri).is_file() for uri, _p, _a in pairs)
    with pytest.raises(FileNotFoundError):
        Image.open(pairs[0][0])

    sess.train("g_seed42", pairs, tmp_path / "out", rf.frozen_route_protocol())
    assert len(trained) == len(pairs) == len(train)
    for example, it in zip(trained, train):
        assert example["prompt"] == prompts["g_image_to_code"]
        assert example["answer"] == man["code_of"][it["identity_id"]]
        # the bytes that reached the model are the ones the URI names, not a
        # placeholder: the identity is in the first pixel
        assert example["image"].getpixel((0, 0))[0] == \
            man["identity_ids"].index(it["identity_id"])
        assert example["image"].mode == "RGB"


def test_no_training_path_opens_an_image_except_through_the_resolver(rf):
    """The asymmetry item 2 names, pinned in the source rather than in one run.

    ``Image.open`` on a recorded URI is correct in exactly one place -- inside
    the resolver itself -- and wrong everywhere else, because everywhere else the
    URI came from a manifest that may have recorded it relative to the dataset
    root.  One call site is also one place to fix the next dataset that records
    its paths differently again.
    Read as syntax rather than as text, because the repair is described in prose
    in the same file and a grep for ``Image.open`` counts the sentence that
    explains it alongside the call it explains.
    """
    import ast

    tree = ast.parse(Path(rf.__file__).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "open"
             and getattr(n.func.value, "id", None) == "Image"]
    assert len(calls) == 1, \
        f"{len(calls)} places open an image; every recorded URI has to go " \
        f"through the resolver or the open depends on the launch directory"
    arg = calls[0].args[0]
    assert isinstance(arg, ast.Call) and \
        getattr(arg.func, "id", None) == "resolve_recorded_path", \
        "the one open is not resolving the path it was given"
    # and it is the resolver's own call site, so the resolver is the single door
    enclosing = next(
        f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
        if any(n is calls[0] for n in ast.walk(f)))
    assert enclosing == "_open_image", enclosing


def test_rf1d_trains_and_files_both_cells_from_another_working_directory(
        rf, pilot_v3, monkeypatch, tmp_path):
    """The same resolution through the phase that actually trains, from a
    working directory that is not the dataset root.

    RF1D is the phase item 2 names alongside RF1G, and it is the one that files
    cells -- so this also shows the two cells it files binding the adapter it
    just wrote.
    """
    man, prereg, ds = pilot_v3["man"], pilot_v3["prereg"], pilot_v3["dataset"]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    trained = []
    monkeypatch.setattr(rf, "RouteSessionV2",
                        _fake_session(rf, man, role="d", trained=trained))
    seed = pilot_v3["direct_seeds"][0]
    doc = rf.phase_rf1d(ds, seed, prereg=prereg, device="cpu",
                        cells_out=tmp_path / "cells")
    assert doc["cells_filed"] == [f"direct__d{seed}", f"hybrid__d{seed}"]
    assert len(trained) == sum(1 for it in man["items"]
                               if it["split"] == "train")
    for example, it in zip(trained, [i for i in man["items"]
                                     if i["split"] == "train"]):
        assert example["image"].getpixel((0, 0))[0] == \
            man["identity_ids"].index(it["identity_id"])
    assert doc["adapter_sha256"] == _sha(
        rf.resolve_recorded_path(prereg["checkpoint_requirements"]["roles"]
                                 [f"direct_d__seed{seed}"]["adapter"]))
    assert doc["direct_pre_edit_held_out_accuracy"] == 1.0
    for cid in doc["cells_filed"]:
        cell = json.loads(rf.cell_result_path(ds, cid,
                                              tmp_path / "cells")
                          .read_text(encoding="utf-8"))
        assert cell["input_sha256"]["direct_adapter"] == doc["adapter_sha256"]
        assert cell["run_provenance"]["pilot_version"] == "v3"


# ==========================================================================
# item 3 -- RF1G runs against the frozen pre-registration
# ==========================================================================

def test_rf1g_refuses_a_seed_the_preregistration_does_not_name(
        rf, pilot_v3, monkeypatch):
    """``check_seed_budget`` permits any seed wherever held-out image content
    exists, so on SALMU ``--router-seed 999`` was a full GPU training run that no
    design named, no cell could consume and no verdict could use.  A factor has
    levels; a level nobody pre-registered is not a level.

    The model is forbidden module-wide here, so reaching the training step would
    fail the test rather than load a checkpoint: a RuntimeError naming the seed
    is proof the refusal came before any work.
    """
    prereg = pilot_v3["prereg"]
    assert 999 not in prereg["router_seeds"]
    with pytest.raises(RuntimeError, match="not pre-registered"):
        rf.phase_rf1g(pilot_v3["dataset"], 999, prereg=prereg, device="cpu")
    assert "router_g__seed999" not in \
        prereg["checkpoint_requirements"]["roles"], \
        "the design that refused the seed is the design that never declared it"
    made = [p for p in (pilot_v3["tmp"] / "weights").rglob("*")
            if "999" in p.name]
    assert not made, \
        f"the refusal has to come before the training, and these exist: {made}"


def test_rf1g_refuses_a_preregistered_seed_with_no_declared_role(
        rf, pilot_v3):
    """The role check is separate from the seed check: a seed the design names
    but declares nowhere to put has no path, so RF0 would never see the router
    however successfully it trained."""
    prereg = pilot_v3["prereg"]
    doctored = {**prereg, "router_seeds": [*prereg["router_seeds"], 999]}
    with pytest.raises(RuntimeError, match="declares no checkpoint role"):
        rf.phase_rf1g(pilot_v3["dataset"], 999, prereg=doctored, device="cpu")


def test_rf1g_refuses_to_retrain_the_frozen_router_without_a_manifest(rf):
    """Refused before any manifest is loaded, because this one is a fact about
    the route rather than about a design: g_17 is the router the route was
    established with under ANY pre-registration.  Answering "no frozen pilot"
    here would answer a question nobody asked."""
    with pytest.raises(RuntimeError, match="never retrained"):
        rf.phase_rf1g("mllmu", rf.EXISTING_ROUTER_SEED, device="cpu")


def test_rf1g_records_the_design_and_the_manifest_it_ran_against(
        rf, pilot_v3, monkeypatch, tmp_path):
    """Item 3's fourth requirement, driven end to end with a stubbed session.

    The manifest is reached through the phase's own verified load rather than
    handed in, so the gate is exercised and not bypassed; what is stubbed is the
    model, never the check.
    """
    ds, man, path = pilot_v3["dataset"], pilot_v3["man"], pilot_v3["path"]
    prereg = pilot_v3["prereg"]
    _use_this_manifest(rf, monkeypatch, path)
    trained = []
    monkeypatch.setattr(rf, "RouteSessionV2",
                        _fake_session(rf, man, role="g", trained=trained))
    seed = next(s for s in pilot_v3["router_seeds"]
                if s != rf.EXISTING_ROUTER_SEED)

    doc = rf.phase_rf1g(ds, seed, device="cpu", spec=rf.PILOT_SPEC_V3)
    assert doc["kind"] == "router_training_result_v3"
    assert doc["phase"] == "RF1G" and doc["router_seed"] == seed
    # the pre-registration it ran against, both ways round
    assert doc["input_sha256"]["preregistration_design"] == \
        prereg["design_sha256"]
    assert doc["run_provenance"]["preregistration_design_sha256"] == \
        prereg["design_sha256"]
    assert doc["run_provenance"]["preregistration_file_sha256"] == _sha(path)
    assert doc["run_provenance"]["preregistration_kind"] == rf.PREREG_V3_KIND
    # and the weights it produced, at the paths the design declared
    role = prereg["checkpoint_requirements"]["roles"][f"router_g__seed{seed}"]
    ckpt = rf.resolve_recorded_path(role["adapter"])
    preds = rf.resolve_recorded_path(role["held_out_predictions"])
    assert doc["adapter_sha256"] == _sha(ckpt)
    assert doc["predictions_sha256"] == _sha(preds)
    assert json.loads(preds.read_text(encoding="utf-8"))["kind"] == \
        "router_held_out_predictions_v3"
    assert len(trained) == sum(1 for it in man["items"] if it["split"] == "train")
    # closing the loop with item 1: the router that just arrived is visible to
    # the readiness probe, and the design did not move because of it
    assert rf.checkpoint_readiness(prereg)["trained_since_declared"], \
        "a trained router is progress the readiness probe reports"
    assert rf.verify_manifest(path)["valid"] is True


def test_rf1g_refuses_a_manifest_that_does_not_verify(rf, pilot_v3,
                                                      monkeypatch, tmp_path):
    """Loading is verifying: a design edited after freezing would otherwise be
    executed, and the edit would be invisible in the results."""
    path = pilot_v3["path"]
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["router_seeds"] = [*doc["router_seeds"], 999]
    path.write_text(rf.canonical_json(doc), encoding="utf-8")
    _use_this_manifest(rf, monkeypatch, path)
    with pytest.raises(RuntimeError, match="does not verify"):
        rf.phase_rf1g(pilot_v3["dataset"], 999, device="cpu",
                      spec=rf.PILOT_SPEC_V3)


# ==========================================================================
# item 4 -- the result names the code that produced it, and RF2 re-hashes it
# ==========================================================================

def test_rf2_re_hashes_the_weights_behind_every_cell(rf, run_v3):
    """A cell binds the digest of the weights that produced its rows, and nothing
    checked that binding at aggregation time: an adapter replaced AFTER a cell
    was filed left an aggregate that looked valid while describing weights that
    no longer existed -- and the aggregate is the artifact the verdicts are read
    off."""
    report = rf.aggregate_cells_v3(run_v3["prereg"], run_v3["loaded"][0],
                                   cell_paths=run_v3["loaded"][1],
                                   run_provenance=run_v3["prov"])
    check = report["input_verification"]
    assert check["n_digests_compared"] == sum(
        len(rf.CELL_KIND_INPUT_KEYS[c["kind"]]) for c in
        run_v3["prereg"]["cells"]), check["bindings_not_recheckable"]
    assert check["bindings_not_recheckable"] == []
    for cell_id, row in check["per_cell"].items():
        for key, entry in row.items():
            if entry["bound_to_a_role_file"]:
                assert entry["state"] == "matches", (cell_id, key, entry)
    # every cell file the aggregate consumed is hashed in the report, so a cell
    # edited after filing is visible in the artifact that was built from it
    assert report["n_consumed_cell_files"] == len(run_v3["filed"]) == \
        run_v3["prereg"]["n_cells"]
    assert set(report["consumed_cell_files"]) == {
        rf._rel(p) for p in run_v3["loaded"][1].values()}
    for p, digest in report["consumed_cell_files"].items():
        assert digest == _sha(rf.resolve_recorded_path(p)), p
    assert report["run_provenance"]["preregistration_design_sha256"] == \
        run_v3["prereg"]["design_sha256"]


def test_an_adapter_replaced_after_a_cell_was_filed_is_refused(rf, run_v3):
    """The swap the check exists for.  Refused rather than reported, because a
    mismatch means the filed rows describe an adapter nobody can inspect, and the
    aggregate would go on to report verdicts about it."""
    prereg, cells = run_v3["prereg"], run_v3["loaded"][0]
    role = "edited_h__seed17"
    p = rf.resolve_recorded_path(
        prereg["checkpoint_requirements"]["roles"][role]["adapter"])
    p.write_bytes(b"an adapter replaced after the cell was filed")
    with pytest.raises(RuntimeError, match="was produced by"):
        rf.aggregate_cells_v3(prereg, cells)


def test_an_absent_adapter_is_reported_and_not_refused(rf, run_v3):
    """Absence is not a mismatch, and treating it as one would break RF2P.

    RF2P exists to reproduce a run from stored raw generations with no GPU, and
    in a fresh clone the adapters are gitignored and legitimately not there.  The
    rows a cell needs are stored IN the cell, so what absence means is that the
    binding could not be re-checked -- and saying so is the honest report.
    """
    prereg, cells = run_v3["prereg"], run_v3["loaded"][0]
    p = rf.resolve_recorded_path(
        prereg["checkpoint_requirements"]["roles"]["edited_h__seed17"]["adapter"])
    p.unlink()
    check = rf.verify_cell_inputs(prereg, cells)
    assert any(k.startswith("intervention__h17:")
               for k in check["bindings_not_recheckable"]), check
    assert check["n_digests_compared"] < sum(
        len(rf.CELL_KIND_INPUT_KEYS[c["kind"]]) for c in prereg["cells"])
    report = rf.aggregate_cells_v3(prereg, cells)
    assert report["input_verification"]["bindings_not_recheckable"]


def test_a_cell_that_recorded_no_weight_binding_is_refused(rf, run_v3):
    """A cell filed by a phase that did not know what it used.  Checked at
    aggregation rather than only at filing, because the cells on disk may predate
    the check."""
    prereg = run_v3["prereg"]
    cells = dict(run_v3["loaded"][0])
    cid = "intervention__h17"
    stripped = dict(cells[cid])
    stripped["input_sha256"] = {
        k: v for k, v in stripped["input_sha256"].items() if k != "edited_h"}
    cells[cid] = stripped
    with pytest.raises(RuntimeError, match="recorded no edited_h digest"):
        rf.verify_cell_inputs(prereg, cells)


def test_a_cell_bound_to_a_role_the_design_does_not_declare_is_refused(
        rf, run_v3):
    """A direct cell claiming it consumed an edited h is a contradiction, and the
    role template resolves it to a name the design never declared."""
    prereg = run_v3["prereg"]
    cells = dict(run_v3["loaded"][0])
    cid = next(c["cell_id"] for c in prereg["cells"] if c["kind"] == "direct")
    doctored = dict(cells[cid])
    doctored["input_sha256"] = {**doctored["input_sha256"], "edited_h": "a" * 64}
    cells[cid] = doctored
    with pytest.raises(RuntimeError, match="does not declare"):
        rf.verify_cell_inputs(prereg, cells)


def test_the_resume_key_carries_the_design_and_not_the_manifest_file(rf):
    """``resume_decision`` compares ``input_sha256`` as a whole dict, so what goes
    in it decides what forces a re-train.

    The DESIGN hash belongs there, because a changed design is a different
    experiment.  The manifest's FILE hash does not, because re-freezing an
    identical design at a new commit is the same experiment recorded in new bytes
    -- and keying on the file would re-train every adapter on every commit that
    touched the runner.
    """
    prereg = {"design_sha256": "d" * 64}
    got = rf._input_sha256(prereg, edited_h="a" * 64)
    assert got == {"edited_h": "a" * 64, "preregistration_design": "d" * 64}
    assert "preregistration_file" not in got
    assert rf._input_sha256(prereg, edited_h="a" * 64) == got, \
        "the key and the filed record are built by one function, so they cannot " \
        "disagree and make every --resume refuse"


# --------------------------------------------------------------------------
# reading rows back out of a run, for the gate arithmetic
# --------------------------------------------------------------------------

E2E_OK = {"observed": 1.0, "predicted": 1.0, "n_images": 12}


def _rows_of(run, kind):
    return [dict(r) for _cid, doc in run["loaded"][0].items()
            if doc["kind_of_cell"] == kind for r in doc["scored"]]


def _all_rows(run):
    return {k: _rows_of(run, k) for k in
            ("baseline", "intervention", "natural", "direct", "hybrid")}


def _accuracies(report):
    return {s: v["accuracy"]
            for s, v in report["router_held_out_accuracy"].items()}


def _gates(rf, rows, accuracies, version="v3"):
    evaluate = rf.evaluate_gates_v3 if version == "v3" else rf.evaluate_gates_v2
    extra = {"baseline_rows": rows["baseline"]} if version == "v3" else {}
    return evaluate(rows["intervention"], rows["natural"], rows["direct"],
                    rows["hybrid"], accuracies, E2E_OK, **extra)


def _refile_baseline(rf, run, refuses_too=False):
    """The same run with the baseline cell filed by a different h.

    ``refuses_too`` is the model item 5 exists for: an h that answered Unknown to
    a forgotten code BEFORE anything edited it.  Only the baseline cell changes,
    so every difference in the aggregate is attributable to the before.
    """
    prereg, man = run["prereg"], run["man"]
    forget = set(prereg["forget_identity_ids"])
    cell = next(c for c in prereg["cells"] if c["kind"] == "baseline")
    rows = []
    for r in cell["rows"]:
        row = dict(r)
        iid = r["forced_identity_id"]
        row[rf.generation_field(r["condition"], rf.CONDITIONS_V3)] = (
            DELETED if (refuses_too and iid in forget) else man["alias_of"][iid])
        rows.append(row)
    scored, missing = rf.score_rows_v2(rows, prereg["vocab"], rf.CONDITIONS_V3)
    assert not missing, missing
    cells = dict(run["loaded"][0])
    cells[cell["cell_id"]] = {**cells[cell["cell_id"]], "rows": rows,
                              "scored": scored}
    return cells


# ==========================================================================
# item 5 -- baseline_h is evaluated: the before in before/after
# ==========================================================================

def test_v3_adds_exactly_one_baseline_cell_and_changes_no_v2_row(
        rf, tmp_path, monkeypatch):
    """v3's condition table is a strict superset of v2's, so every v2 row builder
    produces identical rows under v3 and the only structural difference is the
    one cell that evaluates the unedited h.

    Evaluated ONCE: it depends on no edit seed, no router seed and no direct
    seed, so filing it per seed would be one measurement reported three times --
    the same duplication item 3 of the v2 repairs removed for interventions
    across router seeds.
    """
    man = _disk_manifest(tmp_path)
    args = ("salmu", "fs_001", ["001"], [17, 42, 123], [17, 42, 123],
            [17, 42, 123])
    kw = {"man": man, "images": rf.held_out_images(man)}
    v2 = rf.build_design_v2(*args, **kw)
    v3 = rf.build_design_v3(*args, **kw)

    assert set(rf.CONDITIONS_V2) < set(rf.CONDITIONS_V3)
    assert set(rf.CONDITIONS_V3) - set(rf.CONDITIONS_V2) == \
        {rf.BASELINE_CONDITION_V3}
    for name, entry in rf.CONDITIONS_V2.items():
        assert rf.CONDITIONS_V3[name] is entry, name

    base = [c for c in v3["cells"] if c["kind"] == "baseline"]
    assert len(base) == 1 and base[0]["cell_id"] == "baseline__h_base"
    assert base[0]["phase"] == "RF1B"
    assert base[0]["n_rows"] == len(man["identity_ids"])
    assert v3["n_cells"] == v2["n_cells"] + 1
    assert v3["n_rows_total"] == v2["n_rows_total"] + base[0]["n_rows"]
    assert v3["cells_by_kind"]["baseline"] == ["baseline__h_base"]
    # every v2 cell survives unchanged, rows and all
    v2_by_id = {c["cell_id"]: c for c in v2["cells"]}
    assert len(v2_by_id) == v2["n_cells"]
    for c in v3["cells"]:
        if c["kind"] == "baseline":
            continue
        assert c == v2_by_id[c["cell_id"]], c["cell_id"]
    # and the baseline condition is not decisive: it establishes the change, it
    # is not itself a claim about the edited route
    assert rf.CONDITIONS_V3[rf.BASELINE_CONDITION_V3]["decisive"] is False
    assert rf.BASELINE_CONDITION_V3 not in rf.DECISIVE_CONDITIONS_V3
    assert rf.BASELINE_CONDITION_V3 in rf.FORCED_CODE_CONDITIONS_V3
    assert rf.CONDITIONS_V3[rf.BASELINE_CONDITION_V3]["image_to_h"] is False


def test_the_baseline_requires_the_forgotten_codes_own_alias(rf, tmp_path):
    """Every code produces ITS OWN alias here, the forgotten ones included: that
    is the whole content of the measurement.  A baseline that required the refusal
    label would be a second copy of the post-edit condition and would establish
    nothing about what the edit changed."""
    man = _disk_manifest(tmp_path)
    rows = rf.build_baseline_rows_v3(man, ["001"], "baseline__h_base")
    forget = set(man["forget_identity_ids"])
    assert [r["forced_identity_id"] for r in rows] == sorted(man["identity_ids"])
    for r in rows:
        assert r["condition"] == rf.BASELINE_CONDITION_V3
        assert r["edit_seed"] is None, "there is no edit in the baseline"
        assert r["image_uri"] is None and r["image_to_h"] is False
        assert r["arm"] == ("forgotten" if r["forced_identity_id"] in forget
                            else "retained")
        assert r["expected_label"] == man["alias_of"][r["forced_identity_id"]], \
            "the before is the code's own alias for every code, forgotten or not"
        assert r["also_reported_as"] == ()
    assert any(r["arm"] == "forgotten" for r in rows)
    # scored by the one scorer, through the one rule table
    filled = [{**r, rf.generation_field(r["condition"], rf.CONDITIONS_V3):
               man["alias_of"][r["forced_identity_id"]]} for r in rows]
    scored, missing = rf.score_rows_v2(filled, rf.label_vocab(man),
                                       rf.CONDITIONS_V3)
    assert not missing
    assert all(r["correct"] and r["follows_forced_code"] for r in scored)


def test_the_clean_run_reports_the_before_beside_the_after(rf, run_v3):
    report = rf.aggregate_cells_v3(run_v3["prereg"], run_v3["loaded"][0],
                                   cell_paths=run_v3["loaded"][1],
                                   run_provenance=run_v3["prov"])
    assert report["kind"] == rf.RESULT_KIND_V3
    h_base = report["baseline_hard_response_matrix"]
    assert h_base["cell_id"] == "baseline__h_base"
    assert h_base["n_codes"] == len(run_v3["man"]["identity_ids"])
    # the BEFORE: every code, a forgotten one included, produced its own alias
    for code, row in h_base["matrix"].items():
        iid = next(i for i in run_v3["man"]["identity_ids"]
                   if run_v3["man"]["code_of"][i] == code)
        assert row[run_v3["man"]["alias_of"][iid]] == 1, code
        assert row[rf.DELETED_LABEL] == 0, code
    moved = report["what_the_edit_moved"]
    assert set(moved) == set(run_v3["edit_seeds"])
    for es, entry in moved.items():
        assert entry["n_codes"] == h_base["n_codes"]
        assert entry["n_codes_whose_response_moved"] == \
            len(run_v3["prereg"]["forget_identity_ids"])
        for diff in entry["codes"]:
            iid = next(i for i in run_v3["man"]["identity_ids"]
                       if run_v3["man"]["code_of"][i] == diff["code"])
            assert iid in run_v3["prereg"]["forget_identity_ids"], es
            assert diff["h_base"][run_v3["man"]["alias_of"][iid]] == 1
            assert diff["h_edited"][rf.DELETED_LABEL] == 1
    assert report["baseline_is_the_before_of_the_same_map"]
    gates = report["gates"]
    assert gates["baseline_forgotten_route_following"]["passed"] is True
    assert gates["baseline_forgotten_route_following"]["value"] == 1.0
    assert gates["baseline_forgotten_route_following"]["paired_with"] == \
        "conditional_forgotten_route_suppression"
    assert gates["conditional_forgotten_route_suppression"]["passed"] is True
    assert report["mediation_pass"] is True


def test_an_h_that_refused_the_forgotten_codes_all_along_is_caught(rf, run_v3):
    """THE hole item 5 names.

    v2 computed ``h_e(do(C=C_f)) == Unknown`` and never asked what ``h_base`` did
    with the same code, so an h that answered Unknown to a forgotten code before
    anything edited it satisfied every suppression gate v2 had.  The run below is
    exactly that model: the post-edit gates all still pass, and only the baseline
    gate and the before/after difference say that nothing was established.
    """
    cells = _refile_baseline(rf, run_v3, refuses_too=True)
    report = rf.aggregate_cells_v3(run_v3["prereg"], cells)
    gates = report["gates"]
    # the post-edit picture is unchanged and still looks like a success
    assert gates["conditional_forgotten_route_suppression"]["passed"] is True
    assert gates["conditional_retained_route_accuracy"]["passed"] is True
    assert gates["mediator_intervention_following_accuracy"]["passed"] is True
    # ... and the baseline says the route was never clean to begin with
    assert gates["baseline_forgotten_route_following"]["passed"] is False
    assert gates["baseline_forgotten_route_following"]["value"] == 0.0
    assert "before" in gates["baseline_forgotten_route_following"]["description"]
    assert report["mediation_pass"] is False, \
        "the baseline gate feeds the mediation verdict, so an unestablished " \
        "before cannot leave the verdict supported"
    # the sharpest witness: the edit moved NOTHING, because the before and the
    # after are the same map
    for es, entry in report["what_the_edit_moved"].items():
        assert entry["n_codes_whose_response_moved"] == 0, es
    assert report["verdicts"]["mediation"]["state"] == "fail"
    assert report["verdicts"]["mediation"]["failed_gates"] == \
        ["baseline_forgotten_route_following"]


def test_the_baseline_gate_is_not_computed_by_v2(rf, run_v3):
    """v2 has no baseline cell, so it has no baseline gate.  The two versions
    share one implementation and differ only through the spec, which is what
    makes "v2 computes exactly the gates it was frozen with" a property rather
    than a hope."""
    rows = _all_rows(run_v3)
    accs = _accuracies(rf.aggregate_cells_v3(run_v3["prereg"],
                                             run_v3["loaded"][0]))
    v3 = _gates(rf, rows, accs, "v3")
    v2 = _gates(rf, rows, accs, "v2")
    assert "baseline_forgotten_route_following" in v3
    assert "baseline_forgotten_route_following" not in v2
    # the two direct gates differ as well, and are supposed to: item 6 is that
    # v3 requires them of every seed where v2 pooled them.  Everything else is
    # the same arithmetic over the same rows.
    assert set(v3) - set(v2) == {"baseline_forgotten_route_following"}
    changed = {"direct_image_accuracy", "direct_code_following_rate"}
    for name, gate in v2.items():
        if name in changed:
            assert gate != v3[name], name
            continue
        if name == "no_unparseable_or_multi_label_outputs":
            # The one gate that counts every row the design produced rather than
            # one condition's, so the baseline cell legitimately joins its
            # denominator.  What it is measuring does not change: the same
            # outputs are unparseable in both readings, and the extra rows are
            # exactly the baseline cell's.
            assert v3[name]["n"] - gate["n"] == len(rows["baseline"])
            assert v3[name]["value"] == gate["value"]
            assert v3[name]["passed"] is gate["passed"] is True
            continue
        assert gate == v3[name], name
    assert rf.GATE_TO_VERDICT_V3["baseline_forgotten_route_following"] == \
        "mediation"
    assert set(rf.GATE_THRESHOLDS_V3) - set(rf.GATE_THRESHOLDS_V2) == \
        {"min_baseline_forgotten_route_following"}
    assert rf.GATE_THRESHOLDS_V3["min_baseline_forgotten_route_following"] == \
        rf.GATE_THRESHOLDS_V2["forgotten_route_suppression"], \
        "the two are the paired halves of one claim, so a floor looser on the " \
        "before than on the after would let a partial pre-edit failure count as " \
        "an edit-induced change"


def test_the_baseline_gate_refuses_a_zero_denominator(rf, run_v3):
    """``all([])`` is True, so a baseline that was never evaluated would
    otherwise pass the gate that exists to say it was."""
    rows = _all_rows(run_v3)
    accs = _accuracies(rf.aggregate_cells_v3(run_v3["prereg"],
                                             run_v3["loaded"][0]))
    gates = _gates(rf, {**rows, "baseline": []}, accs)
    gate = gates["baseline_forgotten_route_following"]
    assert gate["passed"] is False
    assert gate["value"] is None
    assert "zero rows" in gate["failed_because"]


# ==========================================================================
# item 6 -- the direct gates are required of every seed, not of the pool
# ==========================================================================

def test_the_direct_gates_are_reported_per_seed(rf, run_v3):
    """Three direct seeds are three measurements of the direct pathway, each on
    its own separately trained and separately hashed adapter."""
    report = rf.aggregate_cells_v3(run_v3["prereg"], run_v3["loaded"][0],
                                   cell_paths=run_v3["loaded"][1],
                                   run_provenance=run_v3["prov"])
    for name in ("direct_image_accuracy", "direct_code_following_rate"):
        gate = report["gates"][name]
        assert gate["passed"] is True, (name, gate["failed_because"])
        assert gate["n_seeds"] == len(run_v3["direct_seeds"]) == 3
        assert set(gate["value"]) == set(run_v3["direct_seeds"])
        assert set(gate["per_seed_n"]) == set(gate["value"])
        assert "every direct_seed" in gate["aggregation"]
        assert "pooled" in gate["aggregation"], gate["aggregation"]
    acc = report["gates"]["direct_image_accuracy"]
    assert acc["resolution_is_per_seed"]
    assert set(acc["resolution"]) == set(acc["per_seed_n"])
    for seed, res in acc["resolution"].items():
        assert res["n"] == acc["per_seed_n"][seed]
    # and the pooled reading is not what the gate was computed from
    assert acc["value"] != acc["n"] / sum(acc["per_seed_n"].values())
    # the seed keys are the seeds the design names, not strings of them: a gate
    # keyed differently from the design's seed list cannot be joined to it
    assert set(acc["ok"]) == set(run_v3["direct_seeds"])


def test_one_direct_seed_below_the_floor_fails_even_when_the_pool_passes(
        rf, run_v3):
    """The witness.  Pooling let one good adapter carry two that failed the
    floor, and the pooled rate still read as a measurement of "the direct
    pathway" rather than of the seed that worked.

    Ten of twelve on one seed and twelve of twelve on the other two is 0.833 and
    0.944: the pool clears the 0.90 floor and the factor does not.
    """
    rows = _all_rows(run_v3)
    accs = _accuracies(rf.aggregate_cells_v3(run_v3["prereg"],
                                             run_v3["loaded"][0]))
    weak = run_v3["direct_seeds"][0]
    seen = 0
    for r in rows["direct"]:
        if r["direct_seed"] == weak and seen < 2:
            r["correct"] = False
            seen += 1
    assert seen == 2
    per_seed = _gates(rf, rows, accs, "v3")["direct_image_accuracy"]
    pooled = _gates(rf, rows, accs, "v2")["direct_image_accuracy"]
    assert per_seed["value"][weak] < rf.GATE_THRESHOLDS_V3[
        "min_direct_image_accuracy"], per_seed["value"]
    assert per_seed["passed"] is False
    assert per_seed["levels_that_failed"] == [weak]
    assert per_seed["rate_by_failed_level"] == {weak: per_seed["value"][weak]}
    assert str(weak) in per_seed["failed_because"], per_seed["failed_because"]
    assert pooled["passed"] is True, \
        "the pooled gate has to pass for this to be a witness about pooling " \
        "rather than about a floor nobody meets"
    assert pooled["value"] > per_seed["value"][weak]


def test_one_direct_seed_captured_by_a_code_fails_even_when_the_pool_passes(
        rf, run_v3):
    """The same arithmetic on the ceiling: three captured rows out of twelve on
    one seed is 0.25, and three out of thirty-six pooled is 0.083."""
    rows = _all_rows(run_v3)
    accs = _accuracies(rf.aggregate_cells_v3(run_v3["prereg"],
                                             run_v3["loaded"][0]))
    weak = run_v3["direct_seeds"][0]
    captured = 0
    for r in rows["hybrid"]:
        if r["direct_seed"] == weak and captured < 3:
            r["parsed_label"] = r["forced_label"]
            captured += 1
    assert captured == 3
    per_seed = _gates(rf, rows, accs, "v3")["direct_code_following_rate"]
    pooled = _gates(rf, rows, accs, "v2")["direct_code_following_rate"]
    assert per_seed["value"][weak] > rf.GATE_THRESHOLDS_V3[
        "max_direct_code_following_rate"], per_seed["value"]
    assert per_seed["passed"] is False
    assert per_seed["levels_that_failed"] == [weak]
    assert str(weak) in per_seed["failed_because"], per_seed["failed_because"]
    assert pooled["passed"] is True
    assert pooled["value"] < per_seed["value"][weak]


def test_a_direct_row_with_no_seed_is_refused_not_pooled(rf, run_v3):
    """A per-seed gate has to know which seed a row belongs to.  Pooling a row
    with no seed would silently add it to every denominator or to none, and
    either way the gate would report a rate over rows it cannot attribute."""
    rows = _all_rows(run_v3)
    accs = _accuracies(rf.aggregate_cells_v3(run_v3["prereg"],
                                             run_v3["loaded"][0]))
    rows["direct"][0]["direct_seed"] = None
    with pytest.raises(RuntimeError, match="direct_seed"):
        _gates(rf, rows, accs, "v3")


def test_all_three_seeds_passing_is_what_passes_the_gate(rf, run_v3):
    """The per-seed gate is not stricter than the design: three seeds each at
    twelve of twelve pass, and the gate says so per seed rather than in aggregate
    only."""
    rows = _all_rows(run_v3)
    accs = _accuracies(rf.aggregate_cells_v3(run_v3["prereg"],
                                             run_v3["loaded"][0]))
    gate = _gates(rf, rows, accs, "v3")["direct_image_accuracy"]
    assert gate["passed"] is True and gate["failed_because"] is None
    assert all(v == 1.0 for v in gate["value"].values())
    assert gate["n"] == sum(gate["per_seed_n"].values()) == 36


# ==========================================================================
# item 7 -- the held-out claim about PPUBench says what the audit found
# ==========================================================================

def test_the_held_out_comment_does_not_claim_images_ppubench_does_not_have(rf):
    """PPUBench routes its twelve held-out FILENAMES perfectly, but every one of
    those images' bytes also appears in its train split: the split partitions
    filenames rather than images.  A comment claiming genuinely held-out
    same-person images contradicted the runner's own audit, and a reader who
    believed it would read the 1.0 as a routing result."""
    src = Path(rf.__file__).read_text(encoding="utf-8")
    block = src.split("#: Held-out routing accuracy", 1)[1]
    comment = block.split("HELD_OUT_G = {", 1)[0]
    assert "genuine" not in comment.lower(), comment
    assert "filenames rather than images" in comment
    assert "BYTES also appears in its train split" in comment
    assert "held-out ROUTING" in comment, \
        "the comment has to say what PPUBench cannot claim, not only what it can"
    assert "SALMU is the dataset whose held-out images are held out in content" \
        in comment


def test_the_module_docstring_names_the_conditions_and_phases_the_code_has(rf):
    """Item 7 is about prose that outlived the fact it stated, and the front
    door of this module had the same defect: a table of "the five conditions"
    naming a ``direct_path`` no design ever had, in a file whose current design
    has seven conditions and eight phases.  A reader who believed that table
    looked for a condition that does not exist and did not look for the baseline
    that does.

    Pinned rather than merely corrected: prose nothing checks goes stale again at
    the next repair, and the tables are the part of this module a reader meets
    before any of the code.
    """
    doc = rf.__doc__

    def _table(after, before):
        block = doc.split(after, 1)[1].split(before, 1)[0]
        return [ln.split()[0] for ln in block.splitlines()
                if ln.startswith("  ") and set(ln.split()[0]) != {"="}
                and ln.split()[0] != "condition"]

    conditions = _table("The conditions\n==============",
                        "Relationship to the existing components")
    assert set(conditions) == set(rf.CONDITIONS_V3), conditions
    assert conditions[0] == "baseline_forced_code", \
        "the table is ordered before-to-after, so the BEFORE comes first"
    assert "before" in doc.split(
        "``baseline_forced_code`` is the", 1)[1][:80].lower(), \
        "the prose has to say what the condition IS, not only list its name"

    phases = _table("Phases\n======", "RF1 no longer exists.")
    assert phases == list(rf.ALL_PHASES), phases
    assert "RF1" not in phases and "RF1 no longer exists" in doc
    assert rf.PHASES_BY_VERSION["v2"] == tuple(
        p for p in rf.ALL_PHASES if p != "RF1B"), \
        "v2 has no baseline cell, so it has no phase that would fill one"

    versions = doc.split("Design versions\n===============", 1)[1].split(
        "Independence from the granularity gates", 1)[0]
    flat = " ".join(versions.split())
    assert f"``{rf.LATEST_PILOT_SPEC.version}`` is the current one" in flat
    assert "--design-version" in flat and "SUPERSEDED" in flat
    for spec in rf.SPEC_BY_VERSION.values():
        assert spec.version in flat, spec.version
    assert "PilotSpec" in flat, \
        "the section has to say the versions share one construction path, " \
        "because that is what makes a repair unable to reach one and not another"


@_needs_real_images
def test_the_ppubench_audit_agrees_with_the_comment(rf):
    """The comment is a claim about the bytes, so it is checked against them
    where the bytes are.  PPUBench's images live outside the repository, and the
    gate above says so rather than failing."""
    man = rf.load_manifest("ppubench")
    audit = rf.image_content_audit(man)
    assert audit["held_out_image_content_exists"] is False
    assert rf.pilot_seed_policy_v2("ppubench", man=man)[
        "held_out_image_content_exists"] is False
    assert rf.pilot_seed_policy_v2("ppubench", man=man)["router_seeds"] == \
        [rf.EXISTING_ROUTER_SEED], \
        "no router budget is spent on a dataset that cannot answer a held-out " \
        "routing question"
    salmu = rf.load_manifest("salmu")
    assert rf.image_content_audit(salmu)["held_out_image_content_exists"] is True


# ==========================================================================
# the aggregate is reproducible, and v3 supersedes v2 without editing it
# ==========================================================================

def test_rf2p_reproduces_the_v3_aggregate(rf, run_v3):
    """Same cells, same gates, same verdicts, every score recomputed from the
    stored generation.  A reproduction that silently disagreed with the original
    would be worse than no reproduction."""
    loaded, paths = run_v3["loaded"]
    first = rf.aggregate_cells_v3(run_v3["prereg"], loaded, cell_paths=paths,
                                  run_provenance=run_v3["prov"])
    again = rf.aggregate_cells_v3(run_v3["prereg"], loaded, rescore=True,
                                  cell_paths=paths,
                                  run_provenance=run_v3["prov"])
    assert again["rescore_agreement"]["identical_to_the_filed_scores"] is True
    assert again["rescore_agreement"]["n_fields_that_moved"] == 0
    assert again["gates"] == first["gates"]
    assert again["verdicts"] == first["verdicts"]
    assert again["baseline_hard_response_matrix"] == \
        first["baseline_hard_response_matrix"]
    assert again["what_the_edit_moved"] == first["what_the_edit_moved"]
    assert again["consumed_cell_files"] == first["consumed_cell_files"]


def test_v3_supersedes_v1_and_v2_without_editing_either(rf):
    """Both earlier versions stay exactly as they are: deleting or rewriting a
    frozen pre-registration would destroy the record of what was registered and
    why it was replaced."""
    rec = rf.supersession_record(rf.PILOT_SPEC_V3)
    assert rec["authoritative_design"] == "v3"
    assert rec["kind"] == rf.SUPERSESSION_KIND_V3
    assert rec["n_repairs"] == len(rf.SUPERSESSION_ITEMS) + \
        len(rf.SUPERSESSION_ITEMS_V3)
    assert set(rf.SUPERSESSION_ITEMS_V3) == {
        "preregistration_survives_its_own_training", "training_image_paths",
        "rf1g_runs_against_the_preregistration", "run_provenance",
        "rf2_hashes_its_inputs", "baseline_h_is_evaluated",
        "direct_gates_are_per_seed"}
    # v2's eight repairs are carried forward verbatim, so a reader who knows the
    # v2 record can find every entry of it in the v3 one
    for name, entry in rf.SUPERSESSION_ITEMS.items():
        assert rec["repairs"][name] == entry, name
    for name, entry in rf.SUPERSESSION_ITEMS_V3.items():
        assert set(entry) >= {"v2", "v3"}, name
        assert entry["v2"] and entry["v3"] and entry["v2"] != entry["v3"], name
    assert "design_sha256" in \
        rec["repairs"]["preregistration_survives_its_own_training"]["v2"]
    assert "_open_image" in rec["repairs"]["training_image_paths"]["v3"]
    assert "working directory" in \
        rec["repairs"]["training_image_paths"]["v2"].lower() or \
        "Image.open" in rec["repairs"]["training_image_paths"]["v2"]
    assert "router_seeds" in \
        rec["repairs"]["rf1g_runs_against_the_preregistration"]["v3"]


@_needs_superseded
def test_the_supersession_record_reports_the_bytes_of_all_four(rf):
    """"Preserved byte-identical" is a checked property: the record carries each
    superseded file's own digest and size read out of its bytes, so an edit to a
    superseded pre-registration shows up as a disagreement with the v3 artifact."""
    rec = rf.supersession_record(rf.PILOT_SPEC_V3)
    by_path = {e["path"]: e for e in rec["superseded"]}
    assert set(by_path) == {rf._rel(p) for p in _V1_MANIFESTS + _V2_MANIFESTS}
    for path in _V1_MANIFESTS + _V2_MANIFESTS:
        entry = by_path[rf._rel(path)]
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert entry["present"] is True
        assert entry["status"] == "superseded_by_v3"
        assert entry["bytes_preserved"] is True
        assert entry["sha256"] == _sha(path)
        assert entry["bytes"] == path.stat().st_size
        assert entry["design_sha256"] == doc["design_sha256"]
        assert entry["executed"] is False
        assert entry["frozen_at_commit"]
    v1 = {rf._rel(p) for p in _V1_MANIFESTS}
    for path, entry in by_path.items():
        assert entry["kind"] == (rf.PREREG_KIND if path in v1
                                 else rf.PREREG_V2_KIND)
    # the v2 record still describes only v1: superseding is per version, so a
    # reader of the v2 artifact is not told about a version that did not exist
    v2rec = rf.supersession_record(rf.PILOT_SPEC_V2)
    assert {e["path"] for e in v2rec["superseded"]} == v1
    assert v2rec["authoritative_design"] == "v2"


@_needs_superseded
def test_the_v2_notes_say_why_v2_is_superseded_structurally(rf):
    """A reader who finds a v2 manifest has to be able to see from the tree that
    it was unrunnable by construction, not merely out of date."""
    notes = rf.SUPERSEDED_NOTES[rf.PREREG_V2_KIND]
    assert notes["design_is_still_reconstructible"]
    assert "does not reproduce design_sha256" in \
        notes["design_sha256_covered_live_checkpoint_status"]
    assert "load_prereg" in notes["design_sha256_covered_live_checkpoint_status"]
    assert "unrunnable by construction" in \
        notes["design_sha256_covered_live_checkpoint_status"]
    assert notes["paths_are_relative_so_they_still_resolve"]
    for path in _V2_MANIFESTS:
        doc = json.loads(path.read_text(encoding="utf-8"))
        got = rf.verify_manifest(path)
        assert got["kind"] == rf.PREREG_V2_KIND
        # the design still rebuilds; only the runner's own bytes have moved, and
        # that is stated up front rather than discovered by a reader
        assert all("does not reproduce design_sha256" not in p
                   for p in got["problems"]), got["problems"]
        assert doc["checkpoint_requirements"]["verification"] is not None
        assert rf.checkpoint_readiness(doc)[
            "frozen_status_block_present_in_this_manifest"] is True


# ==========================================================================
# the frozen v3 pilots themselves
# ==========================================================================
#
# Everything above pins the repairs on hermetic pilots this file builds.  What
# follows pins the two artifacts that were actually frozen and committed, because
# a repair that holds on a fixture and not on the artifact is a repair to the
# fixture.

@_needs_v3_manifests
def test_the_frozen_v3_pilots_verify_with_no_problems_at_all(rf):
    """Zero problems, not one.

    The superseded pilots each report exactly one problem -- this runner's own
    digest, which moved when v3 was added -- and that is what superseding means.
    The v3 pilots were frozen from the commit that implements v3, so every input
    they name is still on disk with the bytes they recorded, and a manifest that
    reported drift here would be reporting that the implementation moved after
    the artifact was frozen against it.

    NOT gated on the adapter weights being present.  That is the property, not an
    omission: the design binds no gitignored file as an input, so it verifies in a
    fresh clone exactly as it verifies on the machine that has every weight.  The
    weights are bound by the RESULTS at RF2, where an absent one is reported and a
    mismatched one is refused.
    """
    for path in _V3_MANIFESTS:
        doc = json.loads(path.read_text(encoding="utf-8"))
        got = rf.verify_manifest(path)
        assert got["valid"] is True, (path.name, got["problems"])
        assert got["problems"] == [], (path.name, got["problems"])
        assert got["kind"] == rf.PREREG_V3_KIND
        assert got["design_sha256"] == doc["design_sha256"]
        # Not a fixed number: this is what the live probe hashed on THIS disk, so
        # it is 9 here and 0 in a clone with no gitignored weights.  Asserting 9
        # would make this a test that passes only on the machine that froze it --
        # the same shape as v1's absolute paths.
        ready = got["checkpoint_readiness"]
        assert got["n_checkpoints_rehashed"] == ready["n_files_present"]
        n_declared = sum(e["n_files_required"]
                         for e in doc["checkpoint_requirements"]["roles"].values())
        assert 0 <= ready["n_files_present"] <= n_declared
        assert ready["n_files_required"] == n_declared
        assert got["executed"] is False
        # readiness is live, and says so in the block rather than only in prose
        assert got["readiness_is_computed_live"], \
            "the report has to say the readiness beside it was probed and not " \
            "read out of the frozen design"
        assert ready["computed_live_not_read_from_the_frozen_design"] is True
        assert ready["frozen_status_block_present_in_this_manifest"] is False, \
            "a v3 design carries no frozen status block to be tempted by"
        assert ready["why_this_is_not_in_the_design"]
        assert ready["runnable_now_is_about_inputs_not_about_results"]
        assert got["checkpoints_complete"] is ready["complete"]
        assert got["must_be_trained"] == ready["must_be_trained"]
        assert got["n_must_be_trained"] == len(ready["must_be_trained"])
        cr = doc["checkpoint_requirements"]
        for absent in ("verification", "complete", "unexpectedly_absent",
                       "must_be_trained_before_this_pilot_can_run",
                       "n_must_be_trained"):
            assert absent not in cr, \
                f"{absent} is live status, and live status inside the design is " \
                f"the defect that made v2 unverifiable by its own training"
        assert cr["readiness_is_computed_live_not_frozen"]
        assert cr["what_the_design_does_bind"]
        assert cr["paths_are_recorded_relative_not_absolute"]
        # v1 recorded the paths it hashed as ABSOLUTE, against the checkout that
        # froze them, so its design_sha256 reproduced only at that root and every
        # role read as absent from anywhere else
        recorded = [entry[key] for entry in cr["roles"].values()
                    for key in rf.CHECKPOINT_FILE_KEYS_V2
                    if entry.get(key) and entry[key] != "reuses edited_h"]
        assert recorded
        assert all(not Path(raw).is_absolute() for raw in recorded), recorded
        # The claim about where an existing input's digest lives is checked
        # against the artifact rather than read: it says the gitignored adapters
        # are NOT in provenance.input_file_sha256, because a digest recorded
        # there at freeze time would make verify_manifest report drift in every
        # fresh clone.  Prose that names a location is a claim about that
        # location, and item 7 is what happens to one nobody checks.
        bound = cr["where_an_input_that_already_existed_is_bound"]
        assert "NOT listed in provenance.input_file_sha256" in bound
        assert "gitignored" in bound
        inputs = doc["provenance"]["input_file_sha256"]
        adapters = [rel for rel in inputs if rel.endswith(".safetensors")]
        assert adapters == [], adapters
        # A role file MAY be a freeze-time input where it is tracked evidence a
        # clone also has -- PPUBench's frozen router predictions are exactly
        # that, which is why the design binds them.  What may not be one is a
        # weight, and what may not be one at all is a role the design declared as
        # work still to do: binding pending work at freeze time is binding
        # progress, which is the defect this version exists for.
        exists_already_files = {
            e[key] for e in cr["roles"].values()
            if e.get("exists_already")
            for key in rf.CHECKPOINT_FILE_KEYS_V2
            if e.get(key) and e[key] != "reuses edited_h"}
        role_inputs = sorted(raw for raw in recorded if raw in inputs)
        assert set(role_inputs) <= exists_already_files, \
            f"a pending role bound at freeze time: {role_inputs}"
        assert all(not raw.endswith(".safetensors") for raw in role_inputs), \
            "a gitignored adapter recorded as a freeze-time input would make " \
            "every fresh clone report drift while telling the truth about its " \
            "own disk"
        assert all((rf.DATASET_ROOT / raw).is_file() for raw in role_inputs)
        # ... and the live prober can read this table, so readiness is a real
        # answer here rather than a None that quietly means "wrong shape"
        assert rf.roles_are_v2_shaped(cr)
        assert got["roles_trained_since_freeze"] == []


@_needs_v3_manifests
def test_the_frozen_v3_pilots_name_the_commit_and_tree_they_came_from(rf):
    """Item 4's list, on the artifact rather than on a cell: commit, worktree
    state, script digest, every input hashed, and no input missing."""
    for path in _V3_MANIFESTS:
        doc = json.loads(path.read_text(encoding="utf-8"))
        prov = doc["provenance"]
        ws = prov["clean_worktree"]
        assert prov["executing_commit"] and len(prov["executing_commit"]) == 40
        assert ws["git_commit"] == prov["executing_commit"]
        assert ws["dirty_tracked_only"] is False, ws["dirty_tracked_only_lines"]
        assert ws["dirty_including_untracked"] is False
        assert ws["untracked_outside_exclusions"] == []
        assert ws["excluded_prefixes"], \
            "the exclusion of this stage's own output prefix is reported, so a " \
            "reader can see what was not counted rather than infer it"
        assert prov["script_sha256"] == _sha(
            _SCRIPTS / "e2c_v3_route_dependent_forgetting.py"), \
            "the frozen script digest and the script that just verified it " \
            "disagree, so this manifest was not frozen from this implementation"
        assert prov["missing_input_files"] == []
        assert prov["n_input_files_hashed"] == len(prov["input_file_sha256"])
        inputs = prov["input_file_sha256"]
        assert "scripts/e2c_v3_route_dependent_forgetting.py" in inputs
        for sup in _V1_MANIFESTS + _V2_MANIFESTS:
            assert rf._rel(sup) in inputs, \
                "a superseded pre-registration is an input to the design that " \
                "supersedes it, so its bytes are named and hashed"
        for rel, digest in inputs.items():
            p = Path(prov["paths_are_relative_to"]) / rel
            assert p.is_file(), rel
            assert _sha(p) == digest, rel


@_needs_v3_manifests
def test_the_frozen_v3_pilots_declare_the_baseline_cell_and_its_gate(rf):
    """Item 5, on the artifact: the before is a cell with rows, a phase to fill
    it, and a gate that feeds a verdict."""
    for path in _V3_MANIFESTS:
        doc = json.loads(path.read_text(encoding="utf-8"))
        baseline = [c for c in doc["cells"] if c["kind"] == "baseline"]
        assert len(baseline) == 1, doc["cells_by_kind"]
        cell = baseline[0]
        assert cell["phase"] == "RF1B"
        assert "RF1B" in rf.PHASES_BY_VERSION["v3"]
        assert "RF1B" not in rf.PHASES_BY_VERSION["v2"]
        n_codes = len(doc["retained_identity_ids"]) + \
            len(doc["forget_identity_ids"])
        assert cell["n_rows"] == n_codes == len(cell["rows"]), \
            "one row per route code and no more, because a hard mediator's " \
            "response to a code does not depend on anything else -- which is " \
            "also why the baseline is not multiplied by router or edit seed"
        assert len(doc["cells_by_kind"]["baseline"]) == 1
        assert all(r["condition"] == "baseline_forced_code"
                   for r in cell["rows"])
        assert all(r["execution"].startswith("h_base(") for r in cell["rows"])
        assert all(r["image_to_h"] is False for r in cell["rows"]), \
            "the baseline is the intervention call with the unedited h, so it " \
            "cannot be the one place an image reaches the mediator"
        assert all(r["image_uri"] is None for r in cell["rows"])
        forgotten = [r for r in cell["rows"]
                     if r["forced_identity_id"] in doc["forget_identity_ids"]]
        assert forgotten, "a baseline with no forgotten code measures no before"
        assert len(forgotten) == len(doc["forget_identity_ids"])
        assert all(r["arm"] == "forgotten" for r in forgotten)
        # the whole of item 5 in two lines: BEFORE the edit, a forgotten code is
        # required to produce its OWN alias, and the refusal label is not an
        # acceptable answer -- an h that had already refused it would make every
        # post-edit suppression gate pass without the edit having moved anything
        assert all(r["expected_label"] == r["forced_label"] for r in forgotten)
        assert all(r["expected_label"] != doc["deleted_label"]
                   for r in forgotten), doc["deleted_label"]
        assert all(r["expected_label"] != doc["deleted_label"]
                   for r in cell["rows"])
        # and the gate that makes it a requirement rather than a description
        fg = doc["frozen_gates"]
        assert "baseline_forgotten_route_following" in fg["gate_to_verdict"]
        assert fg["gate_to_verdict"]["baseline_forgotten_route_following"] == \
            "mediation"
        assert fg["thresholds"]["min_baseline_forgotten_route_following"] == \
            fg["thresholds"]["forgotten_route_suppression"], \
            "the two are the paired halves of one claim, so a floor looser on " \
            "the before than on the after would let a partial pre-edit failure " \
            "count as an edit-induced change"
        assert fg["added_in_this_version"]["gates"] == \
            ["baseline_forgotten_route_following"]
        assert fg["added_in_this_version"]["thresholds"] == \
            ["min_baseline_forgotten_route_following"]
        assert doc["conditions"]["baseline_forced_code"]["prompt_key"] == \
            doc["conditions"]["forgotten_route_intervention"]["prompt_key"], \
            "the before and the after differ in which h answers, not in what is " \
            "asked"


@_needs_v3_manifests
def test_the_declared_gate_aggregation_is_the_one_the_gates_use(rf, run_v3):
    """Item 6, on the artifact -- and the declaration is checked against the
    behaviour rather than merely present.

    A threshold and an aggregation are two halves of one rule, and only the
    threshold was recorded before: "a 0.90 floor" and "a 0.90 floor on every
    direct seed separately" are different analyses of the same rows that reach
    different verdicts, so a pre-registration carrying only the number had not
    recorded the rule it was frozen with.  Prose nothing checks goes stale, which
    is item 7, so this reads the declaration and then computes the gates.
    """
    for path in _V3_MANIFESTS:
        doc = json.loads(path.read_text(encoding="utf-8"))
        decl = doc["frozen_gates"]["direct_gate_aggregation"]
        spec = rf.SPEC_BY_PREREG_KIND[doc["kind"]]
        assert decl["rule"] == spec.direct_gate_aggregation == "every_seed"
        assert set(decl["gates"]) == {"direct_image_accuracy",
                                      "direct_code_following_rate"}
        assert decl["applies_to_every_level_of"] == \
            decl["level_field_on_a_row"] == "direct_seed"
        assert "pooled" in decl["why"] and "three measurements" in decl["why"]
        # v2's block has no such field, and cannot gain one: its bytes are frozen
        for old in _V2_MANIFESTS:
            if old.is_file():
                v2doc = json.loads(old.read_text(encoding="utf-8"))
                assert "direct_gate_aggregation" not in v2doc["frozen_gates"]
                assert rf.SPEC_BY_PREREG_KIND[
                    v2doc["kind"]].declared_gate_aggregation == {}

    # ... and the declaration is true of what the gates do
    rows = _all_rows(run_v3)
    accs = _accuracies(rf.aggregate_cells_v3(run_v3["prereg"],
                                             run_v3["loaded"][0]))
    per_seed = _gates(rf, rows, accs, "v3")
    pooled = _gates(rf, rows, accs, "v2")
    for name in rf.PER_SEED_GATE_AGGREGATION["gates"]:
        assert per_seed[name]["aggregation"] == \
            "every direct_seed, not the pooled mean"
        assert per_seed[name]["n_seeds"] == len(run_v3["direct_seeds"])
        assert "levels_that_failed" in per_seed[name]
        assert "levels_that_failed" not in pooled[name], \
            "the pooled gate has no levels to name, which is the defect"
        assert pooled[name]["n"] == per_seed[name]["n"], \
            "the same rows, aggregated differently -- not different rows"


@_needs_v3_manifests
def test_load_prereg_dispatches_on_version_and_only_v3_loads(rf):
    """The phases load the authoritative design and refuse the superseded ones.

    A superseded pre-registration that still loaded would be a design two
    versions of the runner could disagree about, and the disagreement would show
    up as two different answers from the same command line.
    """
    for ds in ("ppubench", "salmu"):
        doc = rf.load_prereg_v3(ds)
        assert doc["kind"] == rf.PREREG_V3_KIND
        assert doc["dataset"] == ds
        assert doc["preregistered"] and doc["frozen"] and not doc["executed"]
        # A phase that names no version gets the authoritative design, so the
        # default and the explicit call cannot disagree about which pilot ran.
        assert rf.load_prereg(ds)["design_sha256"] == doc["design_sha256"]
        assert rf.load_prereg_for(rf.LATEST_PILOT_SPEC, ds)[
            "design_sha256"] == doc["design_sha256"]
        with pytest.raises(RuntimeError, match="does not verify"):
            rf.load_prereg_v2(ds)
        with pytest.raises(RuntimeError, match="does not verify"):
            rf.load_prereg_for(rf.PILOT_SPEC_V2, ds)
        # ... and the path a version reads is the path that version froze
        assert rf.prereg_path_for(rf.PILOT_SPEC_V3, ds).name == \
            f"rf_pilot_{ds}_v3.json"
        assert rf.prereg_path_for(rf.PILOT_SPEC_V2, ds).name == \
            f"rf_pilot_{ds}_v2.json"
        assert rf.prereg_path_for(rf.PILOT_SPEC_V3, ds) == _BY_DATASET[ds], \
            "the version's canonical name has to be the file that is tracked, " \
            "or a phase run with no --manifest reads something else"


@_needs_v3_manifests
def test_rf0_on_the_frozen_v3_pilots_reports_what_is_still_missing(
        rf, tmp_path):
    """RF0 is the phase that says what is missing, and it now says it live.

    The report goes under ``tmp_path``: a test that files into the repository
    leaves an untracked artifact behind, and CI's post-preflight step fails on a
    tree that is not clean -- a red build whose subject is the test run rather
    than the code.
    """
    for path in _V3_MANIFESTS:
        ds = "salmu" if "salmu" in path.name else "ppubench"
        doc = json.loads(path.read_text(encoding="utf-8"))
        out = tmp_path / ds
        out.mkdir()
        rep = rf.phase_rf0(ds, path=path, out=out)
        assert rep["manifest_valid"] is True, rep["problems"]
        assert rep["problems"] == []
        assert rep["design_version"] == "v3"
        assert rep["kind"] == rf.PREREG_V3_KIND
        assert rep["design_sha256"] == doc["design_sha256"]
        assert rep["readiness_is_computed_live"] is True
        assert rep["roles_that_changed_since_freeze"] == []
        assert rep["held_out_image_drift_since_freeze"] == [], \
            "the drift check compares recorded URIs against the dataset " \
            "manifest, both of them tracked, so it answers the same thing in a " \
            "clone with no image bytes as it does here"
        assert rep["n_cells_filed"] == 0
        assert rep["n_cells_missing"] == rep["n_cells"] == doc["n_cells"]
        assert rep["executed"] is False
        # Verification and readiness are separate questions, and the report says
        # so rather than leaving a reader to infer it from two booleans.
        assert rep["verification_and_readiness_are_separate_questions"]
        assert rep["runnable_now_is_about_inputs_not_about_results"]
        # What is still missing is derived live from the declared roles and what
        # is on disk, never read out of a block frozen before the training
        # existed -- which is why RF0 can ever say the work arrived.
        roles = doc["checkpoint_requirements"]["roles"]
        work = {r for r, e in roles.items() if not e.get("exists_already")}
        inputs = set(roles) - work
        assert work and inputs, roles
        present = set(rep["roles_present"])
        must = set(rep["must_be_trained_before_this_pilot_can_run"])
        # Readiness keeps PENDING WORK and MISSING INPUTS apart, and both answers
        # are about this disk.  A fresh clone has neither the gitignored adapters
        # the design was built over nor the ones it asks to be trained, so both
        # lists are non-empty there while only the first is non-empty here --
        # which is why neither is asserted empty.
        assert must == work - present
        assert set(rep["unexpectedly_absent"]) == inputs - present
        assert must <= work, \
            "readiness never demands a role the design declared was an input " \
            "that already existed, and never invents one it did not declare"
        assert rep["n_must_be_trained"] == len(must)
        assert rep["checkpoint_requirements_complete"] is (
            not (set(roles) - present))
        assert rep["runnable_now"] is not (set(roles) - present)
        if rep["runnable_now"]:
            assert rep["why_not_runnable"] is None
        else:
            assert rep["why_not_runnable"]
            for role in sorted(must | set(rep["unexpectedly_absent"])):
                assert role in rep["why_not_runnable"], \
                    f"{role} is what is missing and the report does not name it"
        assert sorted(p.name for p in out.iterdir()) == [f"rf0_{ds}_v3.json"]
        assert json.loads((out / f"rf0_{ds}_v3.json").read_text(
            encoding="utf-8"))["design_sha256"] == doc["design_sha256"]

