#!/usr/bin/env python3
"""E2C-v3 G6.1 -- the MLLMU five-set coarsening pilot.

G6.0 (``e2c_v3_mllmu_hierarchy.py``) audited and froze the MLLMU -> SOC
mapping.  This module turns that frozen hierarchy into the pilot the audit
exists to license: five edit sets, one per pilot target, each replacing an
entity-specific profession with a deliberately coarser but genuinely valid
SOC ancestor, and four gates that decide whether the full 12-target matrix
may proceed.

WHAT IS FROZEN HERE, AND WHAT IS NOT
------------------------------------
G6.0's two artifacts -- ``mllmu_hierarchy.json`` and
``mllmu_target_selection.json`` -- are the audited evidence.  This module
NEVER writes them.  It reads them, verifies them (including their committed
git state), and writes two NEW artifacts of its own:

    e2c_mllmu/manifests/mllmu_g6_manifest.json   (pilot identities)
    e2c_mllmu/manifests/matrix_mllmu.json        (the five sets)

Both are verify-not-rewrite, exactly like the SALMU and CelebA matrices: a
rebuild that differs from the committed bytes is a hard error, because a
matrix that can drift is not a frozen experimental design.

TWO DESIGN DECISIONS THAT THE DATA FORCED
-----------------------------------------
1. THE CHAIN IS [raw label, broad occupation, minor group].  SOC has four
   levels; ``gx.hierarchy_of`` carries exactly three, so one is elided.  The
   major group is elided, and is recorded as metadata rather than used as a
   coarsening target.  This is a deliberate choice against the alternative
   that would have looked better on paper: with level2 = major group the five
   pilot sets have 26 cousin slots, against 3 with level2 = minor group.
   Picking the taxonomy level that maximises control counts is choosing the
   metric to suit the design.  Both retained depths are real ancestors and
   both are meaningful coarsenings, whereas "Life, Physical, and Social
   Science Occupations" is so broad that emitting it is nearly a non-answer.
   The sparse cousin coverage is reported as a structural fact, not hidden.

2. SIBLING COVERAGE IS STRUCTURALLY PARTIAL, AND SAYS SO.  Only 2 of the 5
   pilot sets have any sibling at all (Marine Biologist -> Biotechnologist,
   Environmental Scientist -> Hydrologist), and both siblings are professions
   with exactly ONE identity in the whole 500-identity benchmark.  Software
   Engineer, Cultural Anthropologist and Museum Curator have zero siblings:
   their broad SOC group contains no other retained detailed occupation.  The
   sibling-coverage gate therefore passes on "every sibling that exists was
   evaluated and retained its own label", records null -- never a vacuous
   1.0 -- where none exists, and fails if an existing sibling was skipped.

THE SAME-LEAF CONTROL
---------------------
All five pilot targets are exactly the five raw labels that share a SOC leaf
with another retained raw label (Software Engineer / Software Developer both
15-1252; Marine Biologist / Wildlife Biologist both 19-1023; Environmental
Scientist / Ecologist both 19-2041; Cultural Anthropologist / Archaeologist
both 19-3091; Museum Curator / Art Curator both 25-4012).  That is not a
coincidence -- it is why they were adjudicated -- and it makes this the
sharpest pilot available: the same-leaf partner is the SAME occupation under
a different surface label, so an edit that generalises by occupation rather
than by label string will drag it along.  ``gx.sibling_controls`` used to file
such a partner under "sibling", asserting a taxonomic distance that does not
exist; it now takes ``leaf_of`` and reports it as ``same_leaf``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import subprocess
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2c_v3_mllmu_matrix")

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent

# ====================================================================== #
# Frozen design constants.  Changing any of these changes the experiment,
# so a rebuild will no longer match the committed matrix and GX0-style
# verification will refuse.  That is the point.
# ====================================================================== #
IDS_PER_PROFESSION = 2
EDIT_SEEDS = [17, 42, 123]
ORACLE_SEED = 17
DELETED_LABEL = "Unknown"

# Unrelated controls: the four highest-frequency retained professions that
# are not a target, same-leaf partner, sibling or cousin of any pilot target.
# Frozen as an explicit count so the roster is reproducible from the artifact
# alone and cannot drift with the benchmark's label frequencies.
N_UNRELATED_PROFESSIONS = 4

# The coarsening depth every pilot set uses: chain index 1 = broad SOC
# occupation.  One depth per set keeps the pilot at exactly five sets.
PILOT_TARGET_DEPTH = 1

# The set mode MUST come from the runner's fixed vocabulary
# (e2c_v3_granularity_matrix.SINGLE_MODES | SAME_DEPTH_MODES | MIXED_MODES):
# run_cells skips any entry whose mode is not in the phase's set, so an
# invented name does not fail loudly -- it silently trains every oracle and
# then evaluates zero cells.  That is not hypothetical; it is what the first
# 16.4-hour execution of this pilot did.  "single_level1" is the existing name
# for exactly this set shape: one target coarsened to chain depth 1.
PILOT_MODE = "single_level1"

CHAIN_LEVELS = ("raw_label", "broad_occupation", "minor_group")
ELIDED_LEVEL = "major_group"


def _load_sibling(module_name, filename):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gx = _load_sibling("e2c_gx_shared", "e2c_v3_granularity.py")
mh = _load_sibling("e2c_mh_shared", "e2c_v3_mllmu_hierarchy.py")

_RV = None


def parser_module():
    """The module that owns the strict parser, loaded on demand and cached.

    Lazy rather than module-level: ``e2c_v3_research_validity`` imports torch at
    module scope, while this module is otherwise torch-free so its design tests
    run in seconds.  The parser is needed only by ``validate_g6``, which is the
    one place that must not reimplement label recognition -- a second copy of
    the normalization rules is exactly how the two sides of the matcher drifted
    apart in the first place.
    """
    global _RV
    if _RV is None:
        _RV = _load_sibling("e2c_rv_g6", "e2c_v3_research_validity.py")
    return _RV

MANIFEST_DIR = DATASET_ROOT / "e2c_mllmu" / "manifests"
G6_MANIFEST_PATH = MANIFEST_DIR / "mllmu_g6_manifest.json"
G6_MATRIX_PATH = MANIFEST_DIR / "matrix_mllmu.json"
HIERARCHY_PATH = MANIFEST_DIR / "mllmu_hierarchy.json"
SELECTION_PATH = MANIFEST_DIR / "mllmu_target_selection.json"


def sha256_file(path, chunk_bytes=1 << 20):
    """Streamed SHA-256.  Local copy on purpose: importing the research
    validity module for this would pull torch into a CPU-only freeze."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def content_sha(obj):
    """Digest of a JSON-able object, excluding its own digest field."""
    payload = {k: v for k, v in obj.items() if k != "content_sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")).hexdigest()


# ====================================================================== #
# G6.0 evidence: read and verify, never write
# ====================================================================== #
def load_hierarchy(path=HIERARCHY_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_selection(path=SELECTION_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def retained_records(art):
    """The 30 records that are in the G6 matrix at all.

    ``excluded_out_of_scope`` labels are out of the matrix entirely: they
    carry no SOC chain, so they cannot be transformed, cannot count as
    taxonomic coverage, and -- because ``gx.sibling_controls`` unpacks a
    3-element chain for every identity -- cannot even serve as "unrelated"
    controls.  They are absent from ``hierarchy_of``, not present-and-ignored.
    """
    return [r for r in art["records"]
            if r["matrix_role"] != "excluded_out_of_scope"]


def frozen_hierarchy_evidence(verify=True):
    """The G6.0 provenance this pilot is licensed by.

    Refuses unless ``mh.verify_hierarchy`` passes: the gates, the committed
    git state of all four files, the reproducibility of the target selection
    from the frozen identity counts, and the selection manifest's own digest
    are all re-checked here rather than trusted.  A pilot that ran against an
    edited-but-uncommitted hierarchy would be unreproducible from its own
    manifest, which is the failure mode G6.0's G7 exists to prevent.
    """
    art = load_hierarchy()
    sel = load_selection()
    ev = {
        "hierarchy_artifact": mh.record_path(HIERARCHY_PATH),
        "hierarchy_content_sha256": art["content_sha256"],
        "selection_manifest": mh.record_path(SELECTION_PATH),
        "selection_sha256": sel["selection_sha256"],
        "source_table_path": art["source_table"]["path"],
        "source_table_sha256": art["source_table"]["sha256"],
        "source_dataset": art.get("source_dataset"),
        "g7_status": art.get("g7_status"),
        "n_records": len(art["records"]),
        "n_retained": len(retained_records(art)),
        "n_selected_targets": sel["n_selected_targets"],
    }
    if verify:
        rep = mh.verify_hierarchy(HIERARCHY_PATH)
        ev["verified"] = {
            "ok": rep["ok"],
            "gates_passed": rep["gates_passed"],
            "selection_reproduced_from_artifact":
                rep["selection_reproduced_from_artifact"],
            "selection_manifest_checks": rep["selection_manifest"]["checks"],
            "committed_state": {
                role: {"tracked": st["tracked"],
                       "matches_index": st["matches_index"],
                       "matches_head": st["matches_head"],
                       "clean": st["clean"]}
                for role, st in rep["committed_state"].items()},
            "n_warnings": len(rep["warnings"]),
        }
        if not rep["ok"]:
            raise RuntimeError("G6.0 verification failed; refusing to build "
                               "the pilot on unverified evidence")
    return ev


# ====================================================================== #
# Pilot target and roster derivation (pure functions of the frozen artifact)
# ====================================================================== #
def pilot_targets(art, sel):
    """The five pilot targets: every adjudicated primary target, then the
    highest-frequency confirmatory targets to make five.

    Derived, not hardcoded.  Recomputing this from the committed artifacts is
    what makes "the pilot ran the five sets it says it ran" checkable after
    the fact, and it means promoting a target is a decision recorded in G6.0,
    not an edit to this file.
    """
    counts = art["identity_counts"]
    primary = sorted(sel["primary_targets"])
    rest = [l for l in sel["selected_targets"] if l not in set(primary)]
    # frequency descending, then label: the same order G6.0's confirmatory
    # selection used, so the two selections cannot disagree about rank.
    rest.sort(key=lambda l: (-counts.get(l, 0), l))
    n_confirmatory = max(0, 5 - len(primary))
    out = primary + rest[:n_confirmatory]
    if len(out) != 5:
        raise RuntimeError(
            f"expected 5 pilot targets, derived {len(out)}: {out}.  The "
            f"frozen selection has {len(primary)} primary and {len(rest)} "
            f"confirmatory targets; a five-set pilot needs at least 5 "
            f"selected targets in total.")
    return out


def _broad(rec):
    return rec["level2_parent"]["code"]


def _minor(rec):
    return rec["minor_group_parent"]["code"]


def pilot_roster(art, targets):
    """Ordered {profession: role} for every profession the pilot needs.

    Roles are assigned by taxonomic relation to the pilot targets, in a fixed
    precedence: target > same_leaf > sibling > cousin > unrelated.  Precedence
    matters because a profession can stand in more than one relation to
    different targets, and the sharpest relation is the one worth reporting --
    an identity that shares a target's leaf is a same-leaf control even if it
    is also some other target's cousin.
    """
    ret = retained_records(art)
    by = {r["original_label"]: r for r in ret}
    counts = art["identity_counts"]
    missing = [t for t in targets if t not in by]
    if missing:
        raise RuntimeError(
            f"pilot targets absent from the retained hierarchy: {missing}.  "
            f"A target must be a retained record to have a SOC chain.")
    roster = OrderedDict()
    for t in sorted(targets):
        roster[t] = "target"

    def add(pred, role):
        for t in sorted(targets):
            x = by[t]
            for label in sorted(by):
                if label in roster:
                    continue
                if pred(by[label], x):
                    roster[label] = role

    add(lambda y, x: y["external_occupation_id"] == x["external_occupation_id"],
        "same_leaf")
    add(lambda y, x: _broad(y) == _broad(x), "sibling")
    add(lambda y, x: _minor(y) == _minor(x), "cousin")

    rest = [l for l in by if l not in roster]
    rest.sort(key=lambda l: (-counts.get(l, 0), l))
    for label in rest[:N_UNRELATED_PROFESSIONS]:
        roster[label] = "unrelated"
    return roster


def roster_report(art, roster):
    """Per-role availability, including the identities actually obtainable.

    Recorded because two roster professions (Biotechnologist, Hydrologist)
    have exactly ONE identity in the entire benchmark, and one same-leaf
    partner (Ecologist) also has one.  A control resting on a single identity
    is still a control, but the manifest has to say so rather than let a
    reader assume the usual two.
    """
    counts = art["identity_counts"]
    out = OrderedDict()
    for label, role in roster.items():
        n = counts.get(label, 0)
        out[label] = {"role": role, "n_identities_in_benchmark": n,
                      "n_selected": min(n, IDS_PER_PROFESSION),
                      "single_identity_control": n < IDS_PER_PROFESSION}
    return out


# ====================================================================== #
# G6 pilot manifest: deterministic identity selection from Full_Set.jsonl
# ====================================================================== #
def read_source_rows(path=None):
    """The benchmark rows, or a RuntimeError naming what needs them.

    Only ``--freeze`` reads the source dataset.  ``--verify``, the gates and
    every test work from the committed manifest, so a checkout without the
    benchmark can still audit the pilot design.
    """
    p = Path(path) if path else mh.FULL_SET
    if not p.exists():
        raise RuntimeError(
            f"MLLMU source dataset not found at {p}.  Only --freeze needs it; "
            f"--verify and the gates read the committed manifest instead.")
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def profession_of(row):
    """The profession is ``biography.Employment`` -- a JSON string nested
    inside the row, not a top-level key.  Mirrors G6.0's counting semantics so
    the identity counts in the frozen hierarchy and the identities selected
    here cannot disagree about what a profession is."""
    return json.loads(row["biography"])["Employment"]


def build_g6_manifest(art, roster, rows, source_path=None):
    """Select pilot identities: the lowest ``IDS_PER_PROFESSION`` identity IDs
    per roster profession, capped by availability.

    Lowest-ID selection (rather than a seeded sample) is what the existing
    MLLMU route manifest already does, and it is reproducible without a
    generator: given the same benchmark bytes, the same identities come back.
    The seed still matters -- it governs the edit, not the selection -- and is
    recorded per set.
    """
    by_prof = {}
    for row in rows:
        by_prof.setdefault(profession_of(row), []).append(row)
    for prof in roster:
        if prof not in by_prof:
            raise RuntimeError(
                f"roster profession {prof!r} has no identities in the source "
                f"dataset, but the frozen hierarchy counts "
                f"{art['identity_counts'].get(prof)} for it.  The benchmark "
                f"and the artifact disagree about what is in the data.")

    selected, alias_of, items = [], {}, []
    for prof in roster:                       # roster order is deterministic
        ids = sorted(r["ID"] for r in by_prof[prof])[:IDS_PER_PROFESSION]
        selected.extend(ids)
    row_by_id = {r["ID"]: r for r in rows}
    for iid in selected:
        row = row_by_id[iid]
        alias_of[iid] = profession_of(row)
        items.append({"identity_id": iid,
                      "image_uri": str(mh.FULL_SET.parent.parent
                                       / row["image"]),
                      "split": "train"})
    code_of = {iid: f"MID_{iid}" for iid in selected}
    qa_probes = {}
    for iid in selected:
        tasks = row_by_id[iid].get("Mask_Task", [])[:3]
        qa_probes[iid] = [{"question": t["Question"],
                           "ground_truth": t["Ground_Truth"]} for t in tasks]

    return {
        "kind": "mllmu_g6_pilot_manifest",
        "dataset": "mllmu-bench (Full_Set)",
        "source_sha256": sha256_file(source_path or mh.FULL_SET),
        "identity_ids": selected,
        "code_of": code_of,
        "alias_of": alias_of,
        "items": items,
        "qa_probes": qa_probes,
        "professions": list(roster),
        "roster": roster_report(art, roster),
        "ids_per_profession": IDS_PER_PROFESSION,
        "deleted_label": DELETED_LABEL,
        "chain_levels": list(CHAIN_LEVELS),
        "elided_level": ELIDED_LEVEL,
        "images_per_identity": {i: 1 for i in selected},
        "held_out_note": (
            "MLLMU-Bench has ONE image per identity, so image-level held-out g "
            "evaluation is impossible and is stated as a limitation.  "
            "Held-out evidence is at the association level: strict parsing, "
            "oracle distance and the native Mask_Task probes."),
    }


# ====================================================================== #
# Chains, leaves and the matrix
# ====================================================================== #
def hierarchy_of_g6(manifest, art):
    """{identity: [raw label, broad occupation title, minor group title]}.

    ``chain[0]`` must equal ``alias_of[iid]`` because ``gx.validate_set``
    checks every assignment's ``source`` against the baseline label -- the
    model is trained on the benchmark's own wording, and the SOC titles are
    the coarsening TARGETS, never the starting point.
    """
    by = {r["original_label"]: r for r in retained_records(art)}
    out = {}
    for iid in manifest["identity_ids"]:
        label = manifest["alias_of"][iid]
        rec = by.get(label)
        if rec is None:
            raise RuntimeError(
                f"identity {iid} has profession {label!r}, which is not a "
                f"retained hierarchy record; it cannot enter the G6 matrix.")
        out[iid] = [label,
                    rec["level2_parent"]["title"],
                    rec["minor_group_parent"]["title"]]
    return out


def leaf_of_g6(manifest, art):
    """{identity: canonical SOC leaf code}.

    The leaf is the DETAILED OCCUPATION CODE, not the label text.  Two raw
    labels that adjudicated onto one code are the same occupation however
    differently they are spelled, which is exactly the distinction
    ``gx.sibling_controls(leaf_of=...)`` needs and cannot recover from the
    chain, whose first element is the raw label.
    """
    by = {r["original_label"]: r for r in retained_records(art)}
    return {iid: by[manifest["alias_of"][iid]]["external_occupation_id"]
            for iid in manifest["identity_ids"]}


def major_group_of(art):
    """{raw label: major group title} -- recorded metadata, never a target."""
    return {r["original_label"]: r["level1_parent"]["title"]
            for r in retained_records(art)}


def g6_ctx(manifest, art, matrix):
    """The evaluation context, in the shape ``gx.validate_set`` and the GX
    runner's ``ctx``-driven code paths expect.  ``leaf_of`` is the one key
    SALMU and CelebA do not carry, and it is what makes same-leaf partners
    report as ``same_leaf`` instead of ``sibling``."""
    return {
        "kind": "taxonomic",
        "identity_ids": manifest["identity_ids"],
        "code_of": manifest["code_of"],
        "baseline_alias_of": manifest["alias_of"],
        "hierarchy_of": hierarchy_of_g6(manifest, art),
        "leaf_of": leaf_of_g6(manifest, art),
        "dag": {k: v for k, v in matrix["dag"].items()},
        "vocab": matrix["vocab"],
    }


def build_g6_matrix(manifest, art, sel, seeds=None, evidence=None):
    """The frozen five-set pilot matrix.

    One set per pilot target.  Every identity bearing that profession is
    transformed to the broad SOC occupation title; every other identity is
    retained and named in ``retain_ids``, so a set that silently dropped a
    control fails validation rather than quietly measuring less.

    ``art`` and ``sel`` are parameters rather than file reads so the builder is
    testable against a synthetic hierarchy; ``evidence`` defaults to the real
    G6.0 digests, which is what a freeze records.
    """
    seeds = seeds or EDIT_SEEDS
    ev = evidence if evidence is not None else frozen_hierarchy_evidence(
        verify=False)
    targets = pilot_targets(art, sel)
    hierarchy_of = hierarchy_of_g6(manifest, art)
    dag = gx.build_label_dag(hierarchy_of)
    alias_of = manifest["alias_of"]
    ids = manifest["identity_ids"]

    vocab = sorted(set(alias_of.values())
                   | {ch[1] for ch in hierarchy_of.values()}
                   | {ch[2] for ch in hierarchy_of.values()}
                   | {DELETED_LABEL})

    sets = []
    for target_label in targets:
        chain = next(ch for iid, ch in hierarchy_of.items()
                     if ch[0] == target_label)
        assigns = {}
        for iid in sorted(ids):
            if alias_of[iid] != target_label:
                continue
            assigns[iid] = {"operation": "taxonomic",
                            "source": chain[0],
                            "target": chain[PILOT_TARGET_DEPTH],
                            "source_depth": 0,
                            "target_depth": PILOT_TARGET_DEPTH}
        if not assigns:
            raise RuntimeError(
                f"pilot target {target_label!r} has no identities in the "
                f"manifest; a set with no targets is not a set.")
        soc = next(r["external_occupation_id"] for r in retained_records(art)
                   if r["original_label"] == target_label)
        sets.append({
            "set_id": f"gx_mll_{soc.replace('-', '')}",
            "mode": PILOT_MODE,
            "target_label": target_label,
            "target_soc": soc,
            "target_depth": PILOT_TARGET_DEPTH,
            "coarser_label": chain[PILOT_TARGET_DEPTH],
            "assignments": assigns,
            "retain_ids": [i for i in ids if i not in assigns],
            "seeds": seeds,
        })

    matrix = {
        "dataset": "mllmu",
        "kind": "taxonomic",
        "edit_seeds": seeds,
        "vocab": vocab,
        "dag": dag,
        "pilot_targets": targets,
        "chain_levels": list(CHAIN_LEVELS),
        "elided_level": ELIDED_LEVEL,
        "target_depth": PILOT_TARGET_DEPTH,
        "sets": sets,
        "n_sets": len(sets),
        "n_cells": len(sets) * len(seeds),
        "g6_0_evidence": {
            "hierarchy_content_sha256": ev["hierarchy_content_sha256"],
            "selection_sha256": ev["selection_sha256"],
            "source_table_sha256": ev["source_table_sha256"],
            "source_dataset_sha256":
                (ev["source_dataset"] or {}).get("sha256"),
        },
    }
    ctx = g6_ctx(manifest, art, matrix)
    return matrix, ctx


def validate_g6(matrix, ctx, art):
    """GX0-equivalent validation, plus the checks that are specific to a
    pilot built on someone else's frozen audit."""
    issues, notes = [], {}
    for entry in matrix["sets"]:
        iss = gx.validate_set(entry, ctx)
        if iss:
            issues.extend(f"{entry['set_id']}: {i}" for i in iss)
        if entry.get("control_notes"):
            notes[entry['set_id']] = entry["control_notes"]
    vocab_issues, collisions = gx.validate_vocab(ctx["vocab"])
    issues.extend(vocab_issues)
    # Every label this design expects the model to produce must round-trip
    # through the strict parser, or the set is unpassable BY CONSTRUCTION and
    # the failure presents as an "unparseable" model output -- pointing at the
    # model instead of at the measurement.  Two of the five pilot targets are
    # comma-bearing broad SOC titles; while recognized_labels_in stripped
    # punctuation from the output text but not from the labels, the parser could
    # not represent them, and the pilot's first real run reported 0.6 strict
    # accuracy over 30 target rows that were every one byte-exact correct.  That
    # cost 16 GPU-hours to discover; here it costs microseconds at freeze time.
    issues.extend(
        f"expected label the strict parser cannot recognize when emitted "
        f"verbatim: {lab!r}"
        for lab in parser_module().check_vocab_parseable(ctx["vocab"]))
    # gx.check_vocab_collisions returns TUPLES.  json.dump writes a tuple as an
    # array and json.load reads it back as a list, so storing the tuples
    # directly makes the freeze its own enemy: the next rebuild compares
    # ("a","b") against ["a","b"] and reports drift that is not there.  Every
    # value stored in a frozen artifact has to survive a JSON round trip
    # unchanged, so normalize here rather than at each call site.
    collisions = {"hard_collisions": [list(p) for p in
                                      collisions["hard_collisions"]],
                  "nested_longest_match_wins":
                      [list(p) for p in collisions["nested_longest_match_wins"]]}

    # Every same-leaf partner that EXISTS in the retained hierarchy must be
    # present in the manifest.  This is the invariant that matters: the
    # same-leaf partner is the sharpest retention control in the benchmark, and
    # a manifest that quietly lost one would let the sibling-coverage and
    # retention gates report a pass over a control that was never measured.
    #
    # A target with NO same-leaf partner is recorded, not failed.  That all
    # five current pilot targets have one is a fact about the adjudicated
    # selection (the primaries are the duplicate-leaf representatives), not a
    # logical necessity, so asserting it would make this validator break the
    # moment G6.0 promotes a target that happens to be unique.
    ret = {r["original_label"]: r for r in retained_records(art)}
    present = set(ctx["baseline_alias_of"].values())
    same_leaf_coverage = OrderedDict()
    for entry in matrix["sets"]:
        soc = entry["target_soc"]
        partners = sorted(l for l, r in ret.items()
                          if r["external_occupation_id"] == soc
                          and l != entry["target_label"])
        missing = [p for p in partners if p not in present]
        if missing:
            issues.append(
                f"{entry['set_id']}: same-leaf partner(s) {missing} exist in "
                f"the retained hierarchy for SOC {soc} but are absent from the "
                f"pilot manifest, so the sharpest retention control for this "
                f"target would go unmeasured")
        same_leaf_coverage[entry["set_id"]] = {
            "target_label": entry["target_label"],
            "target_soc": soc,
            "same_leaf_partners_in_hierarchy": partners,
            "all_present_in_manifest": not missing,
            "note": None if partners else (
                "no other retained raw label adjudicated onto this SOC leaf; "
                "this target has no same-leaf control available"),
        }
    if issues:
        for i in issues:
            logger.error(f"G6X0 ISSUE: {i}")
        # The issues go in the EXCEPTION, not only the log: a launcher that
        # catches this needs to know what to fix, and a count on its own turns
        # a specific design defect into a debugging session.
        raise RuntimeError(
            f"G6X0 validation failed with {len(issues)} issue(s): "
            + "; ".join(issues))
    return {"issues": issues, "control_notes": notes,
            "vocab_collisions": collisions,
            "same_leaf_coverage": same_leaf_coverage}


def sibling_availability(matrix):
    """Which sets have a sibling at all, and which are structurally null.

    Reported separately from the gate verdict so a reader can see that a null
    is a fact about the benchmark's retained occupations, not a skipped check.
    """
    out = OrderedDict()
    for entry in matrix["sets"]:
        sibs = sorted({o for ctl in entry.get("controls", {}).values()
                       for o in ctl.get("sibling", [])})
        same = sorted({o for ctl in entry.get("controls", {}).values()
                       for o in ctl.get("same_leaf", [])})
        out[entry["set_id"]] = {
            "target_label": entry["target_label"],
            "target_soc": entry["target_soc"],
            "n_targets": len(entry["assignments"]),
            "siblings": sibs,
            "same_leaf": same,
            "sibling_available": bool(sibs),
            "sibling_metric": "measured" if sibs else None,
            "sibling_null_reason": None if sibs else (
                "no other retained detailed occupation shares broad SOC group "
                "with this target; sibling metric null, never 1.0"),
        }
    return out



# ====================================================================== #
# The four pilot gates.
#
# Pure functions over per-cell results, so they are testable on CPU with
# fabricated outcomes and cannot be influenced by whatever produced them.
# Thresholds come from gx.PASS_CRITERIA -- the same numbers the SALMU and
# CelebA matrices are held to -- rather than a pilot-specific relaxation.
# ====================================================================== #
def _iter_cells(cells):
    for cell in cells:
        for iid, row in sorted(cell["per_identity"].items()):
            yield cell, iid, row


def behavioral_gate(cells, matrix):
    """Did the target identities actually move to the coarser label?

    Strict parsing only: the emitted label must equal the intended SOC
    ancestor exactly.  A multi-label or unparseable output is a failure, not a
    partial credit, and a wrong-branch answer (a valid occupation that is not
    an ancestor of the target's own) is counted separately because it is the
    specific way this experiment can fail while still looking fluent.
    """
    crit = gx.PASS_CRITERIA
    target_rows, failures = [], []
    for cell, iid, row in _iter_cells(cells):
        if row["control_group"] != "target":
            continue
        target_rows.append((cell["set_id"], cell["seed"], iid, row))
        if row["strict_parsed"] != row["expected"]:
            failures.append(
                f"{cell['set_id']}/seed{cell['seed']}/{iid}: expected "
                f"{row['expected']!r}, strict-parsed "
                f"{row['strict_parsed']!r} (class={row['taxonomic_class']})")
        if row["p_desired"] < crit["min_target_p_desired"]:
            failures.append(
                f"{cell['set_id']}/seed{cell['seed']}/{iid}: p_desired "
                f"{row['p_desired']:.4f} < "
                f"{crit['min_target_p_desired']}")
        if row["p_source"] > crit["max_target_p_source"]:
            failures.append(
                f"{cell['set_id']}/seed{cell['seed']}/{iid}: p_source "
                f"{row['p_source']:.4f} > {crit['max_target_p_source']} "
                f"-- the exact label still carries mass")
        if row["candidate_mass"] < crit["min_candidate_mass"]:
            failures.append(
                f"{cell['set_id']}/seed{cell['seed']}/{iid}: candidate_mass "
                f"{row['candidate_mass']:.4f} < "
                f"{crit['min_candidate_mass']}")
        if row["multi_label"]:
            failures.append(f"{cell['set_id']}/seed{cell['seed']}/{iid}: "
                            f"multi-label output")
        if row["unparseable"]:
            failures.append(f"{cell['set_id']}/seed{cell['seed']}/{iid}: "
                            f"unparseable output")
    wrong = [r for _, _, _, r in target_rows
             if r["taxonomic_class"] == "wrong_branch"]
    if wrong and crit["max_wrong_branch_rate"] == 0.0:
        failures.append(f"{len(wrong)} target row(s) classified wrong_branch")
    if not target_rows:
        # A gate over zero rows is not a passed gate.  Without this the first
        # pilot run reported behavioral=True with n_target_rows=0, because
        # every per-row check was vacuously satisfied -- the classic guard that
        # cannot fire.  Zero rows means the cells were never produced, which is
        # the most serious possible outcome and must read as a failure.
        failures.append(
            "no target rows were evaluated at all: the behavioral gate has "
            "nothing to score, which means no cell produced predictions for "
            "this matrix's targets")
    return {
        "name": "behavioral",
        "passed": not failures,
        "n_target_rows": len(target_rows),
        "strict_expected_accuracy": (
            sum(1 for _, _, _, r in target_rows
                if r["strict_parsed"] == r["expected"]) / len(target_rows)
            if target_rows else None),
        "n_wrong_branch": len(wrong),
        "criteria": {k: crit[k] for k in
                     ("min_target_p_desired", "max_target_p_source",
                      "min_candidate_mass", "max_wrong_branch_rate",
                      "max_unparseable_outputs", "max_multi_label_outputs")},
        "failures": failures,
    }


def retention_gate(cells, matrix):
    """Did every retained identity keep its OWN label?

    Reported per control group, because the groups are not equally
    informative.  ``same_leaf`` is the hardest case available anywhere in this
    benchmark: the partner is the same occupation under a different surface
    label, so an edit that generalises on occupation rather than label will
    move it too.  A single accuracy number averaged across groups would hide
    exactly that failure, so the groups are never pooled into one verdict
    without also being reported separately.
    """
    crit = gx.PASS_CRITERIA
    groups = OrderedDict()
    failures = []
    for cell, iid, row in _iter_cells(cells):
        if row["control_group"] == "target":
            continue
        g = groups.setdefault(row["control_group"],
                              {"n_rows": 0, "n_correct": 0, "wrong": []})
        g["n_rows"] += 1
        if row["strict_parsed"] == row["expected"]:
            g["n_correct"] += 1
        else:
            g["wrong"].append(
                f"{cell['set_id']}/seed{cell['seed']}/{iid}: expected "
                f"{row['expected']!r}, got {row['strict_parsed']!r}")
    for name, g in sorted(groups.items()):
        g["strict_accuracy"] = g["n_correct"] / g["n_rows"] if g["n_rows"] \
            else None
        if g["strict_accuracy"] is not None and \
                g["strict_accuracy"] < crit["retained_strict_accuracy"]:
            failures.extend(g["wrong"])
    if not groups:
        # Same defect as the behavioral gate: the first pilot run reported
        # retention=True over an EMPTY per_control_group map.  "No retained
        # identity was ever evaluated" is the opposite of "every retained
        # identity kept its label", and reporting it as the latter would have
        # let a run that produced nothing look like a clean pass.
        failures.append(
            "no retained rows were evaluated at all: retention has nothing to "
            "score, so no claim about preserved associations is supported")
    return {
        "name": "retention",
        "passed": not failures,
        "per_control_group": groups,
        "criteria": {"retained_strict_accuracy":
                     crit["retained_strict_accuracy"]},
        "note": ("groups are reported separately and never pooled: same_leaf "
                 "is the sharpest control in this benchmark and an average "
                 "over all retained identities would conceal its failure"),
        "failures": failures,
    }


def sibling_coverage_gate(cells, matrix):
    """Was every sibling that EXISTS actually evaluated, and retained?

    Three distinct outcomes, deliberately not collapsed into one number:
      measured + all correct -> pass
      measured + any wrong   -> fail, naming the identity
      no sibling exists      -> null with the structural reason

    The third is the case that makes this gate honest rather than decorative:
    3 of the 5 pilot sets have no sibling at all, because their broad SOC
    group contains no other retained detailed occupation.  Reporting 1.0 for
    those would be a vacuous pass over an empty set; reporting a failure would
    blame the pilot for a fact about the benchmark.  Null, with the reason, is
    the only correct answer -- and the gate still FAILS if a sibling is
    present in the manifest but missing from the evaluated rows, which is the
    real risk this gate exists to catch.
    """
    crit = gx.PASS_CRITERIA
    avail = sibling_availability(matrix)
    per_set, failures = OrderedDict(), []
    for entry in matrix["sets"]:
        sid = entry["set_id"]
        info = avail[sid]
        expected_sibs = set(info["siblings"])
        seen = {}
        for cell in cells:
            if cell["set_id"] != sid:
                continue
            for iid, row in cell["per_identity"].items():
                if row["control_group"] == "sibling":
                    seen[iid] = row
        if not expected_sibs:
            per_set[sid] = {"sibling_metric": None,
                            "n_expected": 0, "n_evaluated": 0,
                            "strict_accuracy": None,
                            "reason": info["sibling_null_reason"]}
            continue
        missing = expected_sibs - set(seen)
        if missing:
            failures.append(
                f"{sid}: sibling(s) {sorted(missing)} exist in the manifest "
                f"but were not evaluated -- sibling coverage is incomplete")
        n_ok = sum(1 for r in seen.values()
                   if r["strict_parsed"] == r["expected"])
        acc = n_ok / len(seen) if seen else None
        per_set[sid] = {"sibling_metric": "measured",
                        "n_expected": len(expected_sibs),
                        "n_evaluated": len(seen),
                        "strict_accuracy": acc,
                        "reason": None}
        if acc is not None and acc < crit["sibling_strict_accuracy"]:
            failures.extend(
                f"{sid}: sibling {iid} expected {r['expected']!r}, got "
                f"{r['strict_parsed']!r}"
                for iid, r in sorted(seen.items())
                if r["strict_parsed"] != r["expected"])
    n_measured = sum(1 for v in per_set.values() if v["sibling_metric"])
    return {
        "name": "sibling_coverage",
        "passed": not failures,
        "per_set": per_set,
        "n_sets_with_sibling": n_measured,
        "n_sets_sibling_null": len(per_set) - n_measured,
        "criteria": {"sibling_strict_accuracy":
                     crit["sibling_strict_accuracy"]},
        "note": ("null is recorded where the benchmark has no sibling for "
                 "that target; it is never reported as 1.0"),
        "failures": failures,
    }


def matched_oracle_gate(cells, matrix):
    """Is the transformation-MATCHED oracle the right reference?

    delta_oracle = D(edit, loo_finetune) - D(edit, matched_finetune).  A
    positive delta means the edited model sits closer to the oracle that was
    trained on the SAME coarsened mapping than to the one trained on the
    retained mapping only -- i.e. the edit did what the transformation asked
    rather than merely deleting the association.  Aggregated per set across
    seeds, because a single seed can invert by noise and the pilot's whole
    purpose is to decide whether the design is sound before 12 targets and
    three seeds are spent on it.
    """
    per_set, failures = OrderedDict(), []
    by_set = {}
    for cell in cells:
        by_set.setdefault(cell["set_id"], []).append(cell)
    for entry in matrix["sets"]:
        sid = entry["set_id"]
        got = by_set.get(sid, [])
        if not got:
            failures.append(f"{sid}: no cells evaluated")
            per_set[sid] = {"mean_delta_oracle": None, "n_seeds": 0}
            continue
        deltas = [c["oracle_distances"]["delta_oracle"] for c in got]
        missing = [c for c, d in zip(got, deltas) if d is None]
        if missing:
            # An oracle distance that could not be computed is not a pass.  It
            # means the matched or LOO reference is absent or fell below the
            # candidate-mass floor, and silently averaging over the rows that
            # did resolve would hide a broken oracle behind a plausible number.
            failures.append(
                f"{sid}: no reliable delta_oracle for seed(s) "
                f"{sorted(c['seed'] for c in missing)} -- the matched or "
                f"leave-one-out oracle is missing or unreliable for those "
                f"cells")
        usable = [d for d in deltas if d is not None]
        if not usable:
            per_set[sid] = {"mean_delta_oracle": None, "n_seeds": 0,
                            "n_seeds_unreliable": len(missing)}
            continue
        mean = sum(usable) / len(usable)
        per_set[sid] = {"mean_delta_oracle": mean,
                        "per_seed": {str(c["seed"]):
                                     c["oracle_distances"]["delta_oracle"]
                                     for c in sorted(got,
                                                     key=lambda c: c["seed"])},
                        "n_seeds": len(usable),
                        "n_seeds_unreliable": len(missing),
                        "distance_to_matched":
                            [c["oracle_distances"]["matched"] for c in got],
                        "distance_to_loo":
                            [c["oracle_distances"]["loo"] for c in got]}
        if mean <= 0.0:
            failures.append(
                f"{sid}: mean delta_oracle {mean:+.4f} <= 0 -- the edit is no "
                f"closer to the transformation-matched oracle than to the "
                f"leave-one-out oracle, so it is not evidence of controlled "
                f"coarsening")
    return {
        "name": "matched_oracle",
        "passed": not failures,
        "per_set": per_set,
        "criteria": {"min_mean_delta_oracle": 0.0,
                     "aggregation": "mean over edit seeds, per set"},
        "failures": failures,
    }


def cells_for_gates(runner_cells):
    """Adapt the GX runner's ``cell_results.json`` records to the gate schema.

    The runner stores per-identity hard predictions, soft probabilities and
    PER-IDENTITY oracle distances; the gates want one flat row per identity and
    one cell-level delta.  Adapting here rather than inside the gates keeps the
    gates pure functions of a documented schema, so they stay testable on CPU
    without a runner, a GPU or a model.

    ``p_source`` is the runner's ``p_baseline_alias``: for a target identity the
    baseline alias IS the exact label the edit is supposed to remove, so mass
    sitting there is precisely the residual the behavioral gate rejects.

    delta_oracle is averaged over the TARGET identities only, dropping rows the
    runner could not compute a gated distance for: a distance below the
    candidate-mass floor is not evidence about anything, and averaging it in
    would let an unreliable row cancel a real one.  How many rows were dropped
    is recorded, because "mean over 2 of 2" and "mean over 1 of 2" are not the
    same claim.
    """
    out = []
    for cell in runner_cells:
        per = {}
        for p in cell["hard_preds"]:
            iid = p["identity_id"]
            soft = cell.get("soft", {}).get(iid, {})
            per[iid] = {
                "expected": p["expected_post_edit"],
                "strict_parsed": p["parsed_label"],
                "control_group": p["group"],
                "p_desired": soft.get("p_expected", 0.0),
                "p_source": soft.get("p_baseline_alias", 0.0),
                "candidate_mass": soft.get("candidate_mass", 0.0),
                "taxonomic_class": p.get("classification"),
                "multi_label": bool(p.get("multi_label_ambiguous")),
                "unparseable": p["parsed_label"] is None,
            }
        fam = cell.get("oracle_families") or {}
        targets = sorted(cell.get("assignments") or {})

        def _l2(iid, family, _fam=fam):
            # bound as a default argument: a closure over the loop variable
            # would resolve to the LAST cell's oracle block if it were ever
            # called after the iteration ended
            rec = _fam.get(iid) or {}
            d = (rec.get(family) or {}).get("distance")
            return d.get("l2") if isinstance(d, dict) else None

        matched = [x for x in (_l2(i, "matched_finetune") for i in targets)
                   if x is not None]
        loo = [x for x in (_l2(i, "loo_finetune") for i in targets)
               if x is not None]
        deltas = [x for x in ((fam.get(i) or {}).get("delta_ft_l2")
                              for i in targets) if x is not None]
        out.append({
            "set_id": cell["set_id"], "seed": cell["seed"],
            "per_identity": per,
            "oracle_distances": {
                "matched": sum(matched) / len(matched) if matched else None,
                "loo": sum(loo) / len(loo) if loo else None,
                "delta_oracle": sum(deltas) / len(deltas) if deltas else None,
                "n_rows_used": len(deltas),
                "n_target_rows": len(targets),
            },
        })
    return out


GATE_FUNCS = (behavioral_gate, retention_gate, sibling_coverage_gate,
              matched_oracle_gate)


def evaluate_gates(cells, matrix):
    """All four gates.  ``passed`` is the conjunction, and the pilot may not
    proceed to the 12-target matrix unless it is True."""
    gates = OrderedDict()
    for fn in GATE_FUNCS:
        g = fn(cells, matrix)
        gates[g["name"]] = g
    # Coverage is checked at the top level as well as inside each gate, because
    # "which sets were never run" is a property of the RUN rather than of any
    # one metric, and it is the first thing a reader needs to know.
    covered = {c["set_id"] for c in cells}
    uncovered = [e["set_id"] for e in matrix["sets"]
                 if e["set_id"] not in covered]
    coverage = {
        "n_cells_evaluated": len(cells),
        "n_cells_expected_when_complete": matrix["n_cells"],
        "n_sets_covered": len(covered & {e["set_id"] for e in matrix["sets"]}),
        "n_sets": len(matrix["sets"]),
        "sets_with_no_cells": uncovered,
        "complete": not uncovered,
    }
    failed = [k for k, g in gates.items() if not g["passed"]]
    return {
        "passed": not failed,
        "failed_gates": failed,
        "gates": gates,
        "coverage": coverage,
        "n_cells_evaluated": len(cells),
        "proceed_to_full_matrix": not failed,
        "proceed_note": ("the full 12-target matrix may proceed only when all "
                         "four gates pass; a null sibling metric on a set with "
                         "no structural sibling does not block, and is not a "
                         "pass either"),
    }


# ====================================================================== #
# Provenance: everything the run manifest must record
# ====================================================================== #
def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd),
                          capture_output=True, text=True, check=False)


def worktree_state(exclude_prefixes=()):
    """Clean-worktree state, reported BOTH ways, with exclusions named.

    The tracked-only determination (``--untracked-files=no``) is what the
    existing runners gate on, and it has to stay that way: a run writes its own
    outputs into the repository tree, and those outputs are committed as
    evidence afterwards, so an untracked-inclusive check would report dirty on
    every run that had already started.

    But the untracked-inclusive determination is recorded too, because a
    tracked-only "clean" can coexist with an untracked ``conftest.py`` or a
    shadowing top-level module that changes what actually executes -- in which
    case the recorded commit cannot reconstruct the run.  Excluding the
    stage's own output prefix is legitimate; excluding it SILENTLY is not, so
    both the exclusions and the unfiltered count are recorded.
    """
    root = mh.REPO_ROOT
    tracked = _git(["status", "--porcelain", "--untracked-files=no"], root)
    allst = _git(["status", "--porcelain", "--untracked-files=all"], root)
    tracked_lines = [l for l in tracked.stdout.splitlines() if l.strip()]
    all_lines = [l for l in allst.stdout.splitlines() if l.strip()]

    def _path(line):
        return line[3:].strip().strip('"')

    excluded = [l for l in all_lines
                if any(_path(l).startswith(p) for p in exclude_prefixes)]
    untracked_other = [l for l in all_lines
                       if l.startswith("??")
                       and not any(_path(l).startswith(p)
                                   for p in exclude_prefixes)]
    return {
        "repo_root": str(root),
        "git_commit": _git(["rev-parse", "HEAD"], root).stdout.strip(),
        "dirty_tracked_only": bool(tracked_lines),
        "n_dirty_tracked_only": len(tracked_lines),
        "dirty_tracked_only_lines": tracked_lines,
        "dirty_including_untracked": bool(all_lines),
        "n_dirty_including_untracked": len(all_lines),
        "untracked_outside_exclusions": untracked_other,
        "excluded_prefixes": list(exclude_prefixes),
        "n_excluded": len(excluded),
        "exclusion_note": ("only this stage's own output prefix is excluded, "
                           "and the exclusion is reported rather than applied "
                           "silently; any other untracked file counts"),
    }


def g6_provenance(out_base, args=None):
    """The provenance block every G6 run manifest carries."""
    ev = frozen_hierarchy_evidence(verify=True)
    art = load_hierarchy()
    sel = load_selection()
    # Computed once: the git queries are subprocesses, and calling them twice
    # could in principle straddle a state change and record a commit that does
    # not match the worktree determination beside it.
    ws = worktree_state(exclude_prefixes=(mh.record_path(out_base) + "/",))
    return {
        "executing_commit": ws["git_commit"],
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "hierarchy": ev,
        "source_dataset_sha256": (ev["source_dataset"] or {}).get("sha256"),
        "source_dataset_matches_frozen": bool(
            ev["source_dataset"]
            and ev["source_dataset"].get("sha256") == (
                art.get("source_dataset") or {}).get("sha256")),
        "clean_worktree": ws,
        "identity_counts_sha256": content_sha(
            {"identity_counts": art.get("identity_counts", {})}),
        "selection_criteria": sel["confirmatory_selection"]["criteria"],
        "config": {"seed": getattr(args, "seed", None),
                   "device": getattr(args, "device", None),
                   "edit_seeds": EDIT_SEEDS,
                   "ids_per_profession": IDS_PER_PROFESSION,
                   "target_depth": PILOT_TARGET_DEPTH},
    }


def per_set_provenance(matrix, cells):
    """Selected target / set / seed, per cell -- recorded separately from the
    run-level provenance so a single cell's manifest is self-describing."""
    out = []
    for cell in sorted(cells, key=lambda c: (c["set_id"], c["seed"])):
        entry = next(e for e in matrix["sets"] if e["set_id"] == cell["set_id"])
        out.append({
            "set_id": cell["set_id"],
            "seed": cell["seed"],
            "target_label": entry["target_label"],
            "target_soc": entry["target_soc"],
            "exact_label": entry["target_label"],
            "coarser_label": entry["coarser_label"],
            "target_depth": entry["target_depth"],
            "n_target_identities": len(entry["assignments"]),
            "target_identity_ids": sorted(entry["assignments"]),
            "n_retained_identities": len(entry["retain_ids"]),
        })
    return out


def hash_outputs(out_base):
    """SHA-256 of every file under the run directory, and of every checkpoint.

    Checkpoints are hashed separately from the general output sweep because
    they are the only files a later phase reloads: if an adapter's bytes move,
    every result derived from it is unverifiable even though the JSON that
    describes it still parses.
    """
    out_base = Path(out_base)
    outputs, checkpoints = {}, {}
    if not out_base.exists():
        return {"outputs_sha256": outputs, "checkpoints_sha256": checkpoints}
    for p in sorted(out_base.rglob("*")):
        if not p.is_file() or p.name == "run_manifest.json":
            continue
        rel = str(p.relative_to(out_base))
        outputs[rel] = sha256_file(p)
        if p.name == "adapter_model.safetensors":
            checkpoints[rel] = outputs[rel]
    return {"outputs_sha256": outputs, "checkpoints_sha256": checkpoints}


def run_manifest_g6(out_base, matrix, cells, gates, args=None, t_start=None):
    """Write the run manifest.  Every field the launch sequence requires is
    here: executing commit, hierarchy and selection digests, source-dataset
    SHA-256, clean-worktree state, selected target/set/seed, and checkpoint
    and output hashes."""
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    prov = g6_provenance(out_base, args)
    hashes = hash_outputs(out_base)
    manifest = {
        "kind": "mllmu_g6_pilot_run_manifest",
        "executing_commit": prov["executing_commit"],
        "cli_invocation": sys.argv,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_sec": round(time.time() - t_start, 1) if t_start else None,
        "provenance": prov,
        "frozen_inputs": {
            "hierarchy_content_sha256":
                prov["hierarchy"]["hierarchy_content_sha256"],
            "selection_sha256": prov["hierarchy"]["selection_sha256"],
            "source_table_sha256": prov["hierarchy"]["source_table_sha256"],
            "source_dataset_sha256": prov["source_dataset_sha256"],
            "g6_manifest_sha256": sha256_file(G6_MANIFEST_PATH)
            if G6_MANIFEST_PATH.exists() else None,
            "g6_matrix_sha256": sha256_file(G6_MATRIX_PATH)
            if G6_MATRIX_PATH.exists() else None,
        },
        "clean_worktree_state": prov["clean_worktree"],
        "sets": per_set_provenance(matrix, cells),
        "n_sets": matrix["n_sets"],
        "n_cells": matrix["n_cells"],
        "gates": gates,
        **hashes,
    }
    with open(out_base / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"G6 run manifest written: commit={manifest['executing_commit']}"
                f" outputs_hashed={len(hashes['outputs_sha256'])} "
                f"gates_passed={gates['passed']}")
    return manifest


# ====================================================================== #
# Freeze / verify (verify-not-rewrite, like every other matrix here)
# ====================================================================== #
def _canonical(obj):
    """JSON-canonical form of a value, for comparing frozen artifacts.

    Comparing the Python objects directly is what let a tuple through:
    ``json.dump`` writes ``("a", "b")`` as an array and ``json.load`` reads it
    back as a list, so the rebuild never equals the file it just wrote and the
    freeze is not a fixed point.  Serializing both sides first makes the
    comparison ask the question that actually matters -- would these bytes be
    the same on disk -- and ignores dict key order, which carries no meaning
    in a committed artifact.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str)


def _write_frozen(path, obj, label):
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _canonical(existing) != _canonical(obj):
            raise RuntimeError(
                f"committed {label} {path} differs from the rebuild.  Frozen "
                f"experimental designs must never drift: review the diff and "
                f"re-commit deliberately, or restore the file with git "
                f"checkout if the rebuild was unintended.")
        logger.info(f"G6X1: committed {label} verified (not rewritten): {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    logger.info(f"G6X1: {label} frozen to {path}")
    return True


def _g6x0_block(matrix, validation):
    """The validation block attached to the frozen matrix.

    Factored out because two paths must produce it identically: ``--freeze``,
    which writes the file, and the GX runner's ``build_or_verify``, which
    re-derives it to compare against the committed bytes.  If they diverged by
    even one key the runner would report drift in a design nothing had changed.
    """
    return {
        "control_notes": validation["control_notes"],
        "hard_collisions": validation["vocab_collisions"]["hard_collisions"],
        "nested_labels_longest_match_wins":
            validation["vocab_collisions"]["nested_longest_match_wins"],
        "sibling_availability": sibling_availability(matrix),
        "same_leaf_coverage": validation["same_leaf_coverage"],
        "pass_criteria": gx.PASS_CRITERIA,
    }


def rebuild_frozen_matrix(verify=True):
    """Rebuild the committed pilot design from COMMITTED inputs only.

    Reads no benchmark: identity selection is already frozen in the manifest,
    so this runs on a bare checkout with no torch and no out-of-repo data.  The
    matrix is rebuilt because its correctness is a property of the code and the
    G6.0 artifact, and a rebuild that disagrees with the committed file means
    one of them moved.
    """
    manifest, _committed = load_frozen_g6()
    art = load_hierarchy()
    sel = load_selection()
    ev = frozen_hierarchy_evidence(verify=verify)
    matrix, ctx = build_g6_matrix(manifest, art, sel, evidence=ev)
    validation = validate_g6(matrix, ctx, art)
    matrix["g6x0_validation"] = _g6x0_block(matrix, validation)
    return {"manifest": manifest, "art": art, "sel": sel, "evidence": ev,
            "matrix": matrix, "ctx": ctx, "validation": validation}


def freeze(art=None, sel=None, rows=None):
    """Build and freeze the pilot manifest and matrix.

    Deterministic and timestamp-free: both files are committed evidence, so a
    field that differed on every run would dirty the tracked worktree and
    abort any concurrently launching run.
    """
    art = art or load_hierarchy()
    sel = sel or load_selection()
    ev = frozen_hierarchy_evidence(verify=True)
    logger.info(f"G6X0: G6.0 evidence verified -- hierarchy "
                f"{ev['hierarchy_content_sha256'][:12]}, selection "
                f"{ev['selection_sha256'][:12]}, g7_status="
                f"{ev['g7_status']}")
    targets = pilot_targets(art, sel)
    logger.info(f"G6X0: pilot targets {targets}")
    roster = pilot_roster(art, targets)
    logger.info(f"G6X0: roster {len(roster)} professions "
                f"({Counter(roster.values())})")

    rows = rows if rows is not None else read_source_rows()
    manifest = build_g6_manifest(art, roster, rows)
    matrix, ctx = build_g6_matrix(manifest, art, sel, evidence=ev)
    validation = validate_g6(matrix, ctx, art)
    logger.info(f"G6X0: validation PASSED ({matrix['n_sets']} sets, "
                f"{matrix['n_cells']} cells, vocab {len(ctx['vocab'])})")

    matrix["g6x0_validation"] = _g6x0_block(matrix, validation)
    wrote_m = _write_frozen(G6_MANIFEST_PATH, manifest, "G6 pilot manifest")
    wrote_x = _write_frozen(G6_MATRIX_PATH, matrix, "G6 pilot matrix")
    return {"manifest": manifest, "matrix": matrix, "ctx": ctx,
            "evidence": ev, "wrote": {"manifest": wrote_m, "matrix": wrote_x}}


def load_frozen_g6():
    """Load the committed pilot manifest and matrix, refusing if either is
    absent -- a pilot cannot be verified against a design that was never
    frozen."""
    for p, what in ((G6_MANIFEST_PATH, "G6 pilot manifest"),
                    (G6_MATRIX_PATH, "G6 pilot matrix")):
        if not p.exists():
            raise RuntimeError(
                f"{what} not found at {p}.  Run --freeze first; --verify "
                f"checks a frozen design and cannot invent one.")
    manifest = json.loads(G6_MANIFEST_PATH.read_text(encoding="utf-8"))
    matrix = json.loads(G6_MATRIX_PATH.read_text(encoding="utf-8"))
    return manifest, matrix


def verify(recount_source=False):
    """Re-derive the pilot design from committed state and compare.

    Reads no source dataset unless ``--recount-source`` is passed, so it runs
    on a bare checkout.  The rebuild is compared against the committed bytes
    for both artifacts, and the G6.0 evidence is re-verified rather than
    trusted from the recorded digests.
    """
    manifest, matrix = load_frozen_g6()
    art = load_hierarchy()
    sel = load_selection()
    ev = frozen_hierarchy_evidence(verify=True)

    problems = []
    for key, expected in (
            ("hierarchy_content_sha256", ev["hierarchy_content_sha256"]),
            ("selection_sha256", ev["selection_sha256"]),
            ("source_table_sha256", ev["source_table_sha256"])):
        got = matrix["g6_0_evidence"].get(key)
        if got != expected:
            problems.append(
                f"matrix records {key}={got} but the verified G6.0 artifact "
                f"now has {expected}: the pilot was frozen against a "
                f"different hierarchy than the one committed now")

    targets = pilot_targets(art, sel)
    if matrix["pilot_targets"] != targets:
        problems.append(
            f"frozen pilot_targets {matrix['pilot_targets']} != re-derived "
            f"{targets}")
    roster = pilot_roster(art, targets)
    if manifest["professions"] != list(roster):
        problems.append(
            f"frozen roster {manifest['professions']} != re-derived "
            f"{list(roster)}")
    if manifest["source_sha256"] != (ev["source_dataset"] or {}).get("sha256"):
        problems.append(
            f"manifest source_sha256 {manifest['source_sha256']} != frozen "
            f"hierarchy source_dataset sha256 "
            f"{(ev['source_dataset'] or {}).get('sha256')}: the pilot "
            f"identities were selected from different benchmark bytes than "
            f"the ones the hierarchy counts describe")

    rebuilt, ctx = build_g6_matrix(manifest, art, sel, evidence=ev)
    for key in ("vocab", "dag", "edit_seeds", "n_sets", "n_cells"):
        if rebuilt[key] != matrix[key]:
            problems.append(f"rebuilt matrix {key} differs from committed")
    for got, exp in zip(rebuilt["sets"], matrix["sets"]):
        for key in ("set_id", "target_label", "target_soc", "coarser_label",
                    "assignments", "retain_ids"):
            if got[key] != exp[key]:
                problems.append(
                    f"set {exp['set_id']}: rebuilt {key} differs from "
                    f"committed")
    validation = validate_g6(rebuilt, ctx, art)

    recount = None
    if recount_source:
        rows = read_source_rows()
        fresh = build_g6_manifest(art, roster, rows)
        drift = {iid: (manifest["alias_of"][iid], fresh["alias_of"][iid])
                 for iid in manifest["identity_ids"]
                 if iid not in fresh["alias_of"]
                 or fresh["alias_of"][iid] != manifest["alias_of"][iid]}
        recount = {
            "source_sha256": fresh["source_sha256"],
            "matches_frozen": fresh["source_sha256"] == manifest["source_sha256"],
            "n_identities_redrawn": len(fresh["identity_ids"]),
            "n_drifted_identities": len(drift),
            "drift": drift,
        }
        if drift or not recount["matches_frozen"]:
            problems.append(
                f"--recount-source found the benchmark has moved: "
                f"{len(drift)} drifted identities, hash match "
                f"{recount['matches_frozen']}")

    if problems:
        for p in problems:
            logger.error(f"G6 VERIFY: {p}")
        raise RuntimeError(
            f"G6 pilot verification failed: {len(problems)} problem(s): "
            + "; ".join(problems))
    return {
        "ok": True,
        "n_sets": matrix["n_sets"],
        "n_cells": matrix["n_cells"],
        "n_identities": len(manifest["identity_ids"]),
        "n_professions": len(manifest["professions"]),
        "pilot_targets": matrix["pilot_targets"],
        "vocab_size": len(matrix["vocab"]),
        "g6_0_evidence": ev,
        "sibling_availability": matrix["g6x0_validation"]
        ["sibling_availability"],
        "control_notes": validation["control_notes"],
        "source_recount": recount,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--freeze", action="store_true",
                   help="build and freeze the pilot manifest and matrix "
                        "(needs the MLLMU source dataset)")
    p.add_argument("--verify", action="store_true",
                   help="re-derive the frozen design and compare (no dataset, "
                        "no torch)")
    p.add_argument("--recount-source", action="store_true",
                   help="with --verify: re-read the benchmark and confirm the "
                        "selected identities still carry the same professions")
    p.add_argument("--report", action="store_true",
                   help="print the frozen design: targets, roster, sibling "
                        "availability")
    args = p.parse_args(argv)

    if args.freeze:
        out = freeze()
        print(json.dumps({
            "froze": out["wrote"],
            "n_sets": out["matrix"]["n_sets"],
            "n_cells": out["matrix"]["n_cells"],
            "n_identities": len(out["manifest"]["identity_ids"]),
            "pilot_targets": out["matrix"]["pilot_targets"],
            "hierarchy_content_sha256":
                out["evidence"]["hierarchy_content_sha256"],
            "selection_sha256": out["evidence"]["selection_sha256"],
        }, indent=2))
        return 0
    if args.verify:
        rep = verify(recount_source=args.recount_source)
        print(json.dumps(rep, indent=2, default=str))
        return 0
    if args.report:
        manifest, matrix = load_frozen_g6()
        print(json.dumps({
            "pilot_targets": matrix["pilot_targets"],
            "sets": [{k: e[k] for k in
                      ("set_id", "target_label", "target_soc",
                       "coarser_label", "mode")}
                     | {"n_targets": len(e["assignments"])}
                     for e in matrix["sets"]],
            "n_identities": len(manifest["identity_ids"]),
            "roster": manifest["roster"],
            "sibling_availability":
                matrix["g6x0_validation"]["sibling_availability"],
            "edit_seeds": matrix["edit_seeds"],
        }, indent=2))
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
