#!/usr/bin/env python3
"""E2C-v3 rule generalization (RG0-RG3): entity-specific edit or branch rule?

THE QUESTION
============
Every granularity cell so far trains the transformation on a member AND
trains every other identity -- including same-branch members -- to keep its
baseline label.  Retention of a sibling under that recipe is enforced by the
training data, so it says nothing about whether the model abstracted the
edit as an ENTITY-SPECIFIC replacement ("only this code changes") or as a
BRANCH RULE ("members of this parent class now map to the parent label").
This runner holds branch members OUT of the edit's training data and
measures which of the two outcomes occurs.  It changes no threshold, gate,
promotion criterion or training recipe anywhere in the project, and nothing
reads its output.

TWO OUTCOMES, NEVER CONFLATED
=============================
entity_specific_abstraction  every held-out member still emits its OWN
                             baseline label under strict parsing: only the
                             edited code changed.
rule_generalization          held-out members of the branch follow the
                             abstraction rule: same-branch members emit the
                             parent/bin label the edit installed, and (for
                             numeric bins) values outside the bin do NOT --
                             the rule is applied with its boundary.
Anything else (a member that drifted to a third label, refused, became
unparseable, or a mix of outcomes) is reported as mixed/indeterminate with
its exact composition.  A mixed composition is NEVER collapsed into either
pure label, and an edit that failed on its own trained member makes the set
an edit_failure precondition -- not a generalization measurement.

SALMU (taxonomic branch rules)
==============================
Train ONE profession -> its parent class; the same-branch member(s) are
ABSENT from the edit's training data (neither transformed nor retention-
trained).  Frozen sets: both directions of each level-1 sibling pair
(therapy {00060576, 00102677}, design {00082704, 00106148}) and one level-2
set training the first media member by sorted id (00041286) with the other
three media members held out -- they sit under two further level-1 parents
(design, information) besides the trained member's (media professional), so
the level-2 rule spans three level-1 branches rather than one sibling pair.

NUMERIC (withheld interior and boundary values)
===============================================
Train 15 -> "15-19 years" and test WITHHELD values 16, 18, 19; train
20 -> "20-29 years" and test withheld 21, 25, 29; and separately test the
19|20 and 29|30 boundary discontinuities: does a generalized bin rule STOP
at the bin edge?  The frozen 24-profile benchmark does not contain 16, 18,
21, 25, 29 or 30, and a value the route was never established for cannot
be interpreted, so this runner FROZEN-EXTENDS the profile set with exactly
those six values (GRN_24..GRN_29) and trains its OWN baseline route h over
the 30 codes with the ORIGINAL route recipe (3000/200/2e-5, repeat 50,
seed 17).  The frozen granularity benchmark (24 profiles, its baseline,
its cells) is untouched: extension profiles live only in this tree.

HELD-OUT MEMBER SEMANTICS (numeric)
interior / interior_upper_edge  value strictly inside the trained bin:
    rule_generalization expects the BIN label; entity_specific expects the
    member's own exact-value label.
boundary_outside  value exactly one step outside the bin edge (20 for
    15-19; 19 and 30 for 20-29): following the rule means KEEPING its own
    label; emitting the bin label is overgeneralized_boundary_violation --
    a third outcome that is neither of the two headline labels.

MEASUREMENT
===========
Canonical prompt only (rd.CODE_TO_ALIAS_PROMPT); prompt-robustness is the
prompt panel's question, not this one.  Strict parsing (multi-label =
invalid, never first-match) decides every outcome; full-sequence candidate
probabilities are recorded beside each decision as soft evidence
(p_rule_label, p_baseline_alias, candidate score sums) and never override
the strict classification.  Association level only: no route-g involvement.

CACHED RECORDS ARE VALIDATED BEFORE REUSE
=========================================
A readable JSON file is not evidence that it measures THIS experiment: every
reused record (the baseline reference, each per-cell result) is checked
against the frozen RG manifest hash, the checkpoint bytes on disk, the
dataset/set/seed, the held-out roster and the declared scorer before it is
trusted (RG_CACHE_VALIDATION).  A mismatch is quarantined beside itself as
``<stem>.stale-<sha8(reasons)>.json`` and the work is redone; aggregation
rejects -- never silently trusts -- any record it cannot validate.

Phases: RG0 build+verify frozen manifests (CPU) | RG1 numeric extended
        baseline route + baseline reference evals | RG2 cells (edit +
        classify, cached per set x seed) | RG3 aggregate + report + archive

Usage:
    python scripts/e2c_v3_rule_generalization.py --dataset salmu --phase all
    python scripts/e2c_v3_rule_generalization.py --dataset celeba_numeric \
        --phase all --device cuda:0
GPU: one device via CUDA_VISIBLE_DEVICES with --device cuda:0 inside the
masked namespace.  Qwen3.5-9B bf16 needs ~19-20 GB resident.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("e2c_v3_rg")

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rv = _load_sibling("e2c_rv_rg", "e2c_v3_research_validity.py")
rd = _load_sibling("e2c_rd_rg", "e2c_v3_realdata.py")
mx = _load_sibling("e2c_mx_rg", "e2c_v3_matrix.py")
gx = _load_sibling("e2c_gx_rg", "e2c_v3_granularity.py")
gxm = _load_sibling("e2c_gxm_rg", "e2c_v3_granularity_matrix.py")

RG_ROOT = Path("e2c_rule_generalization")
RG_MANIFEST_DIR = RG_ROOT / "manifests"
RG_OUT_ROOT = RG_ROOT / "outputs"
RG_REPORT_DIR = RG_ROOT / "reports"

RG_SEEDS = list(gx.SEEDS_DEFAULT)          # 17, 42, 123
ROUTE_SEED = 17                            # original route-h seed (GX1R)

#: Captured in main BEFORE any artifact is written (the 9e488a3 rule).
EXECUTING_COMMIT = None

# ====================================================================== #
# Outcome taxonomy -- exact strings; reports and claims may only use these
# ====================================================================== #
ENTITY_SPECIFIC = "entity_specific"
RULE_GENERALIZED = "rule_generalization"
RULE_CONSISTENT = "rule_consistent_boundary_respected"
OVERGENERALIZED = "overgeneralized_boundary_violation"
INDETERMINATE = "indeterminate"

VERDICT_ENTITY = "entity_specific_abstraction"
VERDICT_RULE = "rule_generalization"
VERDICT_MIXED = "mixed_partial_generalization"
VERDICT_INDETERMINATE = "indeterminate_members_present"
VERDICT_EDIT_FAILURE = "edit_failure_precondition"

BOUNDARY_SHARP = "sharp_boundary"
BOUNDARY_VIOLATED = "boundary_violated"
BOUNDARY_NO_GEN = "no_generalization"
BOUNDARY_INDETERMINATE = "indeterminate"

TAXONOMY = {
    "member_outcomes": {
        ENTITY_SPECIFIC: ("held-out member emits its OWN baseline label "
                          "under strict parsing: only the edited code "
                          "changed"),
        RULE_GENERALIZED: ("held-out member emits the rule label the edit "
                           "installed for the branch/bin it belongs to"),
        RULE_CONSISTENT: ("a boundary-OUTSIDE member keeps its own label: "
                          "the generalized rule, if any, stops at the bin "
                          "edge -- this is the rule-following outcome for "
                          "an outside member, NOT entity-specific evidence "
                          "about the rule itself"),
        OVERGENERALIZED: ("a boundary-OUTSIDE member emits the bin label: "
                          "the abstraction leaked across the boundary -- a "
                          "third outcome, never counted as either headline "
                          "label"),
        INDETERMINATE: ("parsed output is neither the baseline label nor "
                        "the rule label (wrong branch, refusal, "
                        "unparseable, multi-label): reported with its raw "
                        "evidence, never coerced into a headline label"),
    },
    "set_verdicts": {
        VERDICT_ENTITY: "edit succeeded and EVERY held-out member is "
                        "entity_specific / rule_consistent",
        VERDICT_RULE: ("edit succeeded, every interior/branch member is "
                       "rule_generalization and every boundary-outside "
                       "member is rule_consistent"),
        VERDICT_MIXED: ("edit succeeded and members split across outcomes "
                        "-- the composition is reported, never collapsed"),
        VERDICT_INDETERMINATE: "at least one held-out member is "
                               "indeterminate",
        VERDICT_EDIT_FAILURE: ("the trained member itself did not reach the "
                               "rule label under strict parsing: the set "
                               "measures no generalization at all"),
    },
    "mislabeling_guards": [
        "a mixed composition is never reported as either pure outcome",
        ("an outside member keeping its label is rule_consistent, not "
         "entity_specific: the two are counted separately everywhere"),
        ("soft probabilities are recorded beside every strict decision and "
         "never override it"),
        "edit failure is a precondition failure, not an outcome",
    ],
}

# ====================================================================== #
# FROZEN set definitions (chosen from the committed taxonomies and the
# frozen numeric schema BEFORE any GPU run; never adjusted afterwards)
# ====================================================================== #
#: SALMU: both directions of every level-1 sibling pair, plus one level-2
#: media set training the first member by sorted id (frozen selection rule,
#: arbitrary but fixed before results exist).
RG_SALMU_SETS = [
    {"set_id": "rg_sal_L1_therapy_from_00060576", "trained": "00060576",
     "rule_depth": 1, "holdouts": ["00102677"]},
    {"set_id": "rg_sal_L1_therapy_from_00102677", "trained": "00102677",
     "rule_depth": 1, "holdouts": ["00060576"]},
    {"set_id": "rg_sal_L1_design_from_00082704", "trained": "00082704",
     "rule_depth": 1, "holdouts": ["00106148"]},
    {"set_id": "rg_sal_L1_design_from_00106148", "trained": "00106148",
     "rule_depth": 1, "holdouts": ["00082704"]},
    {"set_id": "rg_sal_L2_media_from_00041286", "trained": "00041286",
     "rule_depth": 2, "holdouts": ["00082704", "00106148", "00110390"]},
]

#: Numeric extension: the exact withheld values the experiment requires.
#: The frozen 24 profiles do not contain 16/18/21/25/29/30, and a value
#: with no established baseline association cannot be interpreted, so the
#: RG route is established over 24+6 codes in its own tree.
RG_NUMERIC_NEW_PROFILES = [
    ("RY16", "years_experience", 16, "exact",
     "RG withheld interior of the 15-19 narrow bin"),
    ("RY18", "years_experience", 18, "exact",
     "RG withheld interior of the 15-19 narrow bin"),
    ("RY21", "years_experience", 21, "exact",
     "RG withheld interior of the 20-29 broad bin"),
    ("RY25", "years_experience", 25, "exact",
     "RG withheld interior of the 20-29 broad bin"),
    ("RY29", "years_experience", 29, "exact",
     "RG withheld upper edge of 20-29; INSIDE of the 29|30 boundary pair"),
    ("RY30", "years_experience", 30, "exact",
     "RG withheld one step above 20-29; OUTSIDE of the 29|30 boundary pair"),
]

#: Numeric RG sets.  Trained member per the frozen schema operations;
#: interior members are withheld from training; boundary_outside members are
#: withheld too, so their behavior is spontaneous on both sides of an edge.
RG_NUMERIC_SETS = [
    {"set_id": "rg_num_narrow_15", "trained_pid": "Y02",
     "operation": "exact_to_narrow",
     "holdouts": [{"pid": "RY16", "role": "interior"},
                  {"pid": "RY18", "role": "interior"},
                  {"pid": "Y03", "role": "interior_upper_edge"},
                  {"pid": "Y04", "role": "boundary_outside"}],
     "boundary_pairs": [{"inside_pid": "Y03", "outside_pid": "Y04",
                         "edge": "19|20"}]},
    {"set_id": "rg_num_broad_20", "trained_pid": "Y04",
     "operation": "exact_to_broad",
     "holdouts": [{"pid": "RY21", "role": "interior"},
                  {"pid": "RY25", "role": "interior"},
                  {"pid": "RY29", "role": "interior_upper_edge"},
                  {"pid": "Y03", "role": "boundary_outside"},
                  {"pid": "RY30", "role": "boundary_outside"}],
     "boundary_pairs": [{"inside_pid": "RY29", "outside_pid": "RY30",
                         "edge": "29|30"},
                        {"inside_pid": "Y04", "outside_pid": "Y03",
                         "edge": "19|20", "inside_is_trained_member": True}]},
]

#: The requested coverage, checked at RG0 rather than trusted: every value
#: the experiment was specified with must appear with the specified role.
RG_USER_SPEC = {
    "rg_num_narrow_15": {"trained_value": 15, "interior": [16, 18, 19],
                         "boundary_edges": ["19|20"]},
    "rg_num_broad_20": {"trained_value": 20, "interior": [21, 25, 29],
                        "boundary_edges": ["29|30", "19|20"]},
}


# ====================================================================== #
# RG0: frozen manifest builders + validation (CPU, deterministic)
# ====================================================================== #
def _salmu_source():
    with open(gxm.SALMU_MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def build_rg_salmu_manifest():
    """Resolve the frozen SALMU RG sets against the committed taxonomy."""
    sm = _salmu_source()
    matrix = gxm.load_frozen("salmu")
    hierarchy_of = {i: list(v) for i, v in sm["job_levels"].items()}
    ids = list(sm["identity_ids"])
    sets = []
    for spec in RG_SALMU_SETS:
        trained = spec["trained"]
        depth = spec["rule_depth"]
        rule_label = hierarchy_of[trained][depth]
        holdouts = []
        for h in spec["holdouts"]:
            holdouts.append({
                "identity_id": h,
                "role": "branch_member",
                "code": sm["code_of"][h],
                "baseline_alias": sm["alias_of"][h],
                "shares_rule_at_depth": hierarchy_of[h][depth] == rule_label,
                "level1": hierarchy_of[h][1],
                "same_level1_as_trained": hierarchy_of[h][1]
                == hierarchy_of[trained][1],
            })
        sets.append({
            "set_id": spec["set_id"], "kind": "taxonomic",
            "rule_depth": depth,
            "trained": {"identity_id": trained, "code": sm["code_of"][trained],
                        "source_alias": sm["alias_of"][trained],
                        "rule_label": rule_label},
            "holdouts": holdouts,
            "retain_ids": [i for i in ids
                           if i != trained
                           and i not in spec["holdouts"]],
            "seeds": RG_SEEDS,
        })
    return {
        "kind": "e2c_v3_rule_generalization_manifest_v1",
        "dataset": "salmu",
        "produced_by": "scripts/e2c_v3_rule_generalization.py",
        "inputs": {
            "salmu_manifest_sha256": rv.sha256_file(gxm.SALMU_MANIFEST),
            "matrix_salmu_sha256": rv.sha256_file(
                gxm.MANIFEST_DIR / "matrix_salmu.json"),
        },
        "vocab": matrix["vocab"],
        "dag": matrix["dag"],
        "identity_ids": ids,
        "code_of": sm["code_of"],
        "baseline_alias_of": sm["alias_of"],
        "hierarchy_of": hierarchy_of,
        "sets": sets,
        "n_sets": len(sets),
        "n_cells": len(sets) * len(RG_SEEDS),
        "edit_seeds": RG_SEEDS,
        "selection_note": ("both directions of every level-1 sibling pair, "
                           "plus one level-2 media set training the first "
                           "member by sorted id -- frozen before any run"),
        "taxonomy": TAXONOMY,
    }


def build_rg_numeric_manifest():
    """The frozen 24 profiles PLUS the six withheld RG values.

    The frozen part must equal the committed numeric manifest exactly: the
    extension adds route members, it never moves the benchmark.
    """
    base = gx.build_numeric_manifest()
    schema = base["schema"]
    profiles = list(gx.NUMERIC_PROFILES) + RG_NUMERIC_NEW_PROFILES
    identities, code_of, alias_of, prof_out = [], {}, {}, {}
    all_values = {p[1]: {q[2] for q in profiles if q[1] == p[1]}
                  for p in profiles}
    for i, prof in enumerate(profiles):
        pid, field, value, kind, rationale = prof
        iid = f"{i:02d}"
        lab = gx.baseline_label(prof, schema)
        identities.append(iid)
        code_of[iid] = f"GRN_{i:02d}"
        alias_of[iid] = lab
        prof_out[iid] = {
            "profile_id": pid, "field": field, "exact_value": value,
            "baseline_kind": kind, "rationale": rationale,
            "unit": schema[field]["unit"],
            "boundary_tags": gx.boundary_tags(schema[field], value,
                                              all_values[field]),
            "narrow_bin": list(gx.narrow_interval(schema[field], value)),
            "broad_bin": list(gx.broad_interval(schema[field], value)),
            "rounded": gx.round_value(schema[field], value),
            "category": gx.category_of(schema[field], value),
            "rg_extension": pid.startswith("RY"),
        }
    if len(set(alias_of.values())) != len(alias_of):
        raise ValueError("extended baseline labels are not unique")
    # frozen part must equal the committed benchmark part exactly
    for iid in base["identity_ids"]:
        if (code_of[iid] != base["code_of"][iid]
                or alias_of[iid] != base["alias_of"][iid]):
            raise ValueError(f"extension moved frozen profile {iid}")
    prof_by_pid = {p[0]: p for p in profiles}
    iid_by_pid = {prof_out[i]["profile_id"]: i for i in identities}
    sets = []
    for spec in RG_NUMERIC_SETS:
        t_pid = spec["trained_pid"]
        t_iid = iid_by_pid[t_pid]
        rule_label = gx.target_label(spec["operation"], prof_by_pid[t_pid],
                                     schema)
        hold_ids = [iid_by_pid[h["pid"]] for h in spec["holdouts"]]
        holdouts = [{
            "identity_id": iid_by_pid[h["pid"]], "role": h["role"],
            "profile_id": h["pid"],
            "code": code_of[iid_by_pid[h["pid"]]],
            "baseline_alias": alias_of[iid_by_pid[h["pid"]]],
            "exact_value": prof_by_pid[h["pid"]][2],
        } for h in spec["holdouts"]]
        pairs = [{
            "edge": p["edge"],
            "inside_identity_id": iid_by_pid[p["inside_pid"]],
            "outside_identity_id": iid_by_pid[p["outside_pid"]],
            "inside_value": prof_by_pid[p["inside_pid"]][2],
            "outside_value": prof_by_pid[p["outside_pid"]][2],
            "inside_is_trained_member": p.get(
                "inside_is_trained_member", False),
        } for p in spec["boundary_pairs"]]
        sets.append({
            "set_id": spec["set_id"], "kind": "numeric",
            "operation": spec["operation"],
            "trained": {"identity_id": t_iid, "profile_id": t_pid,
                        "code": code_of[t_iid],
                        "source_alias": alias_of[t_iid],
                        "exact_value": prof_by_pid[t_pid][2],
                        "rule_label": rule_label,
                        "field": prof_by_pid[t_pid][1]},
            "holdouts": holdouts,
            "boundary_pairs": pairs,
            "retain_ids": [i for i in identities
                           if i != t_iid and i not in hold_ids],
            "seeds": RG_SEEDS,
        })
    vocab = sorted(set(alias_of.values())
                   | {s["trained"]["rule_label"] for s in sets}
                   | {gx.DELETED_LABEL})
    return {
        "kind": "e2c_v3_rule_generalization_manifest_v1",
        "dataset": "celeba_numeric",
        "produced_by": "scripts/e2c_v3_rule_generalization.py",
        "inputs": {
            "numeric_manifest_sha256": rv.sha256_file(
                gxm.MANIFEST_DIR / "numeric_manifest.json"),
        },
        "schema": schema,
        "extension": {
            "n_new_profiles": len(RG_NUMERIC_NEW_PROFILES),
            "new_values": sorted(p[2] for p in RG_NUMERIC_NEW_PROFILES),
            "reason": ("the frozen 24 profiles do not contain the withheld "
                       "interior/boundary values the experiment requires "
                       "(16, 18, 21, 25, 29, 30); a value with no "
                       "established baseline association cannot be "
                       "interpreted, so the RG route is established over "
                       "30 codes in its own tree and the frozen benchmark "
                       "is untouched"),
            "frozen_part_verified_equal_to_committed_manifest": True,
        },
        "vocab": vocab,
        "identity_ids": identities,
        "code_of": code_of,
        "alias_of": alias_of,
        "profiles": prof_out,
        "sets": sets,
        "n_sets": len(sets),
        "n_cells": len(sets) * len(RG_SEEDS),
        "edit_seeds": RG_SEEDS,
        "taxonomy": TAXONOMY,
    }


def validate_rg_manifest(ds, man):
    """Hard RG0 gate: the frozen sets must mean what they claim."""
    issues = []
    if ds == "salmu":
        vocab = set(man["vocab"])
        hier = man["hierarchy_of"]
        ids = set(man["identity_ids"])
        for s in man["sets"]:
            sid = s["set_id"]
            t = s["trained"]
            if t["identity_id"] not in ids:
                issues.append(f"{sid}: trained id not in taxonomy")
            if t["rule_label"] not in vocab:
                issues.append(f"{sid}: rule label not in vocab")
            if t["rule_label"] == t["source_alias"]:
                issues.append(f"{sid}: rule label equals the source alias")
            if hier[t["identity_id"]][s["rule_depth"]] != t["rule_label"]:
                issues.append(f"{sid}: rule label is not the trained "
                              f"member's own parent at the rule depth")
            seen = set()
            for h in s["holdouts"]:
                hid = h["identity_id"]
                if hid == t["identity_id"] or hid in seen:
                    issues.append(f"{sid}: holdout {hid} duplicates the "
                                  f"trained member or another holdout")
                seen.add(hid)
                if not h["shares_rule_at_depth"]:
                    issues.append(f"{sid}: holdout {hid} does not share the "
                                  f"rule label at depth {s['rule_depth']}")
                if h["baseline_alias"] == t["rule_label"]:
                    issues.append(f"{sid}: holdout {hid}'s baseline alias "
                                  f"IS the rule label (degenerate)")
                if h["baseline_alias"] not in vocab:
                    issues.append(f"{sid}: holdout {hid} alias not in vocab")
            if set(s["retain_ids"]) & (seen | {t["identity_id"]}):
                issues.append(f"{sid}: retain overlaps trained/holdouts")
            if not s["retain_ids"]:
                issues.append(f"{sid}: no retention-trained members left")
    else:
        schema = man["schema"]
        vocab = set(man["vocab"])
        frozen_values = {p[2] for p in gx.NUMERIC_PROFILES}
        new_values = [p[2] for p in RG_NUMERIC_NEW_PROFILES]
        if len(set(new_values)) != len(new_values):
            issues.append("extension values are not unique")
        if set(new_values) & frozen_values:
            issues.append("extension duplicates a frozen profile value")
        for s in man["sets"]:
            sid = s["set_id"]
            t = s["trained"]
            fs = schema[t["field"]]
            kind = ("narrow" if s["operation"] == "exact_to_narrow"
                    else "broad")
            lo, hi = (gx.narrow_interval(fs, t["exact_value"])
                      if kind == "narrow"
                      else gx.broad_interval(fs, t["exact_value"]))
            if t["rule_label"] != gx.fmt_label(fs, kind, interval=(lo, hi)):
                issues.append(f"{sid}: rule label is not the frozen-schema "
                              f"bin of the trained value")
            if t["rule_label"] not in vocab:
                issues.append(f"{sid}: rule label not in vocab")
            hold_ids = set()
            for h in s["holdouts"]:
                hid, v = h["identity_id"], h["exact_value"]
                hold_ids.add(hid)
                if hid == t["identity_id"]:
                    issues.append(f"{sid}: trained member held out")
                inside = lo <= v <= hi
                if h["role"] in ("interior", "interior_upper_edge"):
                    if not inside:
                        issues.append(f"{sid}: interior holdout {v} is not "
                                      f"inside [{lo},{hi}]")
                    if h["role"] == "interior_upper_edge" and v != hi:
                        issues.append(f"{sid}: upper-edge holdout {v} != {hi}")
                elif h["role"] == "boundary_outside":
                    if inside:
                        issues.append(f"{sid}: boundary_outside {v} is "
                                      f"INSIDE [{lo},{hi}]")
                    if v not in (lo - 1, hi + 1):
                        issues.append(f"{sid}: boundary_outside {v} is not "
                                      f"adjacent to [{lo},{hi}]")
                else:
                    issues.append(f"{sid}: unknown holdout role {h['role']}")
            for p in s["boundary_pairs"]:
                if p["outside_identity_id"] not in hold_ids:
                    issues.append(f"{sid}: pair {p['edge']} outside member "
                                  f"is not held out")
                if (not p["inside_is_trained_member"]
                        and p["inside_identity_id"] not in hold_ids):
                    issues.append(f"{sid}: pair {p['edge']} inside member "
                                  f"is neither trained nor held out")
                if abs(p["inside_value"] - p["outside_value"]) != 1:
                    issues.append(f"{sid}: pair {p['edge']} is not adjacent")
            if set(s["retain_ids"]) & (hold_ids | {t["identity_id"]}):
                issues.append(f"{sid}: retain overlaps trained/holdouts")
        # user-spec coverage: the requested values must be where requested
        by_set = {s["set_id"]: s for s in man["sets"]}
        for sid, want in RG_USER_SPEC.items():
            s = by_set.get(sid)
            if s is None:
                issues.append(f"user spec: set {sid} missing")
                continue
            if s["trained"]["exact_value"] != want["trained_value"]:
                issues.append(f"{sid}: trained value != "
                              f"{want['trained_value']}")
            interior = {h["exact_value"] for h in s["holdouts"]
                        if h["role"].startswith("interior")}
            if not set(want["interior"]) <= interior:
                issues.append(f"{sid}: missing interior values "
                              f"{sorted(set(want['interior']) - interior)}")
            edges = {p["edge"] for p in s["boundary_pairs"]}
            if not set(want["boundary_edges"]) <= edges:
                issues.append(f"{sid}: missing boundary edges "
                              f"{sorted(set(want['boundary_edges']) - edges)}")
        v_issues, _collisions = gx.validate_vocab(man["vocab"])
        issues.extend(v_issues)
    return issues


def rg_manifest_path(ds):
    return RG_MANIFEST_DIR / f"rg_manifest_{ds}.json"


def build_or_verify(ds):
    """RG0: verify-not-rewrite the committed manifest (CPU, deterministic)."""
    RG_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    man = (build_rg_salmu_manifest() if ds == "salmu"
           else build_rg_numeric_manifest())
    issues = validate_rg_manifest(ds, man)
    if issues:
        for i in issues:
            logger.error("RG0 ISSUE: %s", i)
        raise RuntimeError(f"RG0 validation failed with {len(issues)} "
                           f"issue(s)")
    path = rg_manifest_path(ds)
    if path.exists():
        committed = json.loads(path.read_text(encoding="utf-8"))
        if committed != man:
            raise RuntimeError(
                f"committed {path} differs from the frozen builder; RG "
                "manifests are committed inputs and must not drift")
        logger.info("RG0: committed manifest verified (not rewritten): %s",
                    path)
    else:
        path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
        logger.info("RG0: manifest frozen to %s (%d sets, %d cells)", path,
                    man["n_sets"], man["n_cells"])
    # byte-deterministic validation record (NO timestamp: committed file)
    validation = {
        "dataset": ds, "n_sets": man["n_sets"], "n_cells": man["n_cells"],
        "n_identities": len(man["identity_ids"]),
        "vocab_size": len(man["vocab"]),
        "sets": [{"set_id": s["set_id"],
                  "trained": s["trained"]["identity_id"],
                  "rule_label": s["trained"]["rule_label"],
                  "holdouts": [(h["identity_id"], h["role"])
                               for h in s["holdouts"]],
                  "boundary_pairs": [p["edge"] for p in
                                     s.get("boundary_pairs", [])]}
                 for s in man["sets"]],
        "issues": issues,
    }
    out_base = RG_OUT_ROOT / ds
    out_base.mkdir(parents=True, exist_ok=True)
    (out_base / "rg0_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    return man


def load_frozen_rg(ds):
    return json.loads(rg_manifest_path(ds).read_text(encoding="utf-8"))


def rg_dataset_ctx(ds, man):
    """Evaluation context; salmu dag/vocab come from the frozen matrix."""
    if ds == "salmu":
        return {"kind": "taxonomic",
                "identity_ids": man["identity_ids"],
                "code_of": man["code_of"],
                "baseline_alias_of": man["baseline_alias_of"],
                "hierarchy_of": man["hierarchy_of"],
                "dag": man["dag"],
                "vocab": man["vocab"]}
    return {"kind": "numeric",
            "identity_ids": man["identity_ids"],
            "code_of": man["code_of"],
            "baseline_alias_of": man["alias_of"],
            "profiles": man["profiles"],
            "schema": man["schema"],
            "vocab": man["vocab"]}


# ====================================================================== #
# Classification (pure functions -- unit-tested, no GPU)
# ====================================================================== #
def classify_holdout_outcome(parsed, baseline_alias, rule_label, role):
    """The strict outcome for ONE held-out member. Never coerced."""
    outside = role == "boundary_outside"
    if parsed is None:
        return INDETERMINATE
    if outside:
        if parsed == rule_label:
            return OVERGENERALIZED
        if parsed == baseline_alias:
            return RULE_CONSISTENT
        return INDETERMINATE
    if parsed == rule_label:
        return RULE_GENERALIZED
    if parsed == baseline_alias:
        return ENTITY_SPECIFIC
    return INDETERMINATE


def set_verdict(edit_success, outcomes):
    """The set-level verdict from member outcomes; mixed never collapses."""
    if not edit_success:
        return VERDICT_EDIT_FAILURE
    vals = set(outcomes.values())
    if not vals:
        return VERDICT_INDETERMINATE
    if INDETERMINATE in vals:
        return VERDICT_INDETERMINATE
    entity_side = {ENTITY_SPECIFIC, RULE_CONSISTENT}
    if vals <= entity_side:
        return VERDICT_ENTITY
    branch_side = {RULE_GENERALIZED, RULE_CONSISTENT}
    if vals <= branch_side and RULE_GENERALIZED in vals:
        return VERDICT_RULE
    return VERDICT_MIXED


def boundary_pair_verdict(inside_emits_rule, outside_outcome):
    """Discontinuity at one bin edge, from the two members' outcomes."""
    if inside_emits_rule and outside_outcome == RULE_CONSISTENT:
        return BOUNDARY_SHARP
    if inside_emits_rule and outside_outcome == OVERGENERALIZED:
        return BOUNDARY_VIOLATED
    if not inside_emits_rule and outside_outcome == RULE_CONSISTENT:
        return BOUNDARY_NO_GEN
    return BOUNDARY_INDETERMINATE


# ====================================================================== #
# Evaluation primitives (strict parsing + soft evidence, canonical prompt)
# ====================================================================== #
def _prompt_for(ctx, iid):
    return rd.CODE_TO_ALIAS_PROMPT.format(code=ctx["code_of"][iid])


def rg_hard_eval(session, ctx, args):
    """Strict hard generation over every identity, canonical prompt only."""
    backend = session.backend()
    session.model.eval()
    out = {}
    with torch.no_grad():
        for iid in ctx["identity_ids"]:
            gen = backend.generate(None, _prompt_for(ctx, iid),
                                   max_new_tokens=args.max_gen_tokens)
            raw = gen.text.strip()
            labels = rv.recognized_labels_in(raw, ctx["vocab"])
            out[iid] = {
                "raw": raw,
                "parsed_label": rv.parse_recognized_label(raw, ctx["vocab"]),
                "recognized_labels": labels,
                "multi_label_invalid": len(labels) > 1,
                "unparseable": rv.parse_recognized_label(raw,
                                                         ctx["vocab"]) is None,
            }
    return out


def rg_soft_eval(session, ctx, args):
    """Full-sequence candidate scores per identity (same scorer as every
    other artifact in this project; see the prompt panel's
    SCORING_LIMITATION for the score-sum semantics of candidate_mass)."""
    out = {}
    for iid in ctx["identity_ids"]:
        probs = rv.full_sequence_label_probs(
            session.adapter, session.model, session.processor,
            ctx["code_of"][iid], ctx["vocab"], args.device)
        prob_by_label = {l: probs.get(l, {}).get("prob", 0.0)
                         for l in ctx["vocab"]}
        out[iid] = rv.build_candidate_summary(prob_by_label, ctx["vocab"],
                                              gx.DELETED_LABEL)
    return out


def _baseline_ckpt_rg(ds, out_base):
    if ds == "salmu":
        return gxm.SALMU_ROUTE_H
    return out_base / "route_h" / "adapter_final" / "adapter_model.safetensors"


def _strict_acc(hard, expected_of):
    ok = sum(1 for i, e in expected_of.items()
             if hard[i]["parsed_label"] == e)
    return ok / max(len(expected_of), 1)


# ====================================================================== #
# Cached-record validation (the prompt panel's blocking issue 2, applied
# here from the start): a readable JSON file is NOT evidence that it
# measures the current experiment.  Every reused record is checked against
# the frozen manifest, the checkpoint bytes on disk and the declared
# scorer; a mismatch is quarantined (never deleted, never trusted) and the
# work is redone.
# ====================================================================== #
CELL_KIND = "rule_generalization_cell_v1"
REF_KIND = "rg_baseline_reference"

#: Declared policy: what a cached record must prove before it is reused.
#: ``runner_script_sha256`` is RECORDED but deliberately not required to
#: match -- a runner fix that cannot change measurement semantics must not
#: invalidate results; the shared scorer hash can.
RG_CACHE_VALIDATION = {
    "required_to_match": [
        "kind", "dataset", "set_id", "seed",
        "checkpoint_sha256", "provenance.rg_manifest_sha256",
        "provenance.shared_scoring_script_sha256", "holdout_roster",
    ],
    "recorded_not_required": ["provenance.runner_script_sha256",
                              "provenance.git_commit"],
    "on_mismatch": ("quarantine the file as <stem>.stale-<sha8(reasons)>.json "
                    "and rerun the cell; a stale record is never aggregated"),
}


def _quarantine(path, reasons):
    """Rename a stale record beside itself; the evidence is kept."""
    joined = "|".join(sorted(reasons)).encode()
    digest = hashlib.sha256(joined).hexdigest()[:8]
    dest = path.with_name(f"{path.stem}.stale-{digest}{path.suffix}")
    path.rename(dest)
    return dest


def validate_cached_cell(rec, ds, sid, seed, holdout_ids, ckpt):
    """Reasons this cached cell must NOT be reused (empty = reusable)."""
    reasons = []
    if rec.get("kind") != CELL_KIND:
        reasons.append("kind")
    if rec.get("dataset") != ds:
        reasons.append("dataset")
    if rec.get("set_id") != sid:
        reasons.append("set_id")
    if rec.get("seed") != seed:
        reasons.append("seed")
    if sorted(rec.get("outcomes") or {}) != sorted(holdout_ids):
        reasons.append("holdout_roster")
    prov = rec.get("provenance") or {}
    man_path = rg_manifest_path(ds)
    if (not man_path.exists()
            or prov.get("rg_manifest_sha256") != rv.sha256_file(man_path)):
        reasons.append("provenance.rg_manifest_sha256")
    if prov.get("shared_scoring_script_sha256") != rv.script_sha256():
        reasons.append("provenance.shared_scoring_script_sha256")
    if (not ckpt.exists()
            or rec.get("checkpoint_sha256") != rv.sha256_file(ckpt)):
        reasons.append("checkpoint_sha256")
    return reasons


def validate_cached_reference(rec, ds, ctx, ckpt):
    """Same discipline for the baseline reference record."""
    reasons = []
    if rec.get("kind") != REF_KIND:
        reasons.append("kind")
    if rec.get("dataset") != ds:
        reasons.append("dataset")
    if sorted(rec.get("per_identity") or {}) != sorted(ctx["identity_ids"]):
        reasons.append("identity_roster")
    if (not ckpt.exists()
            or rec.get("checkpoint_sha256") != rv.sha256_file(ckpt)):
        reasons.append("checkpoint_sha256")
    return reasons


# ====================================================================== #
# RG1: numeric extended baseline route + baseline reference evals
# ====================================================================== #
def ensure_rg_numeric_route(args, ctx, out_base):
    """Train the 30-code baseline route h with the ORIGINAL route recipe.

    The frozen granularity benchmark keeps its own 24-code route; this is
    the RG tree's own establishment over the extended profile set, same
    protocol as GX1R (3000/200/2e-5, repeat 50, seed 17).
    """
    ckpt = _baseline_ckpt_rg("celeba_numeric", out_base)
    if ckpt.exists():
        logger.info("RG1: extended numeric baseline route already present")
        return ckpt
    logger.info("RG1: TRAIN extended numeric baseline route h (%d codes, "
                "%d/%d/%g, repeat %d, seed %d)", len(ctx["identity_ids"]),
                args.route_steps, args.route_warmup, args.route_lr,
                args.route_repeat, ROUTE_SEED)
    session = mx.ModelSession(args, "e2c_rg_num_h")
    try:
        mx.seed_everything(ROUTE_SEED)
        pairs = [{"prompt": _prompt_for(ctx, i),
                  "answer": ctx["baseline_alias_of"][i]}
                 for i in ctx["identity_ids"]]
        items = rv.build_supervised_items(session.adapter, session.processor,
                                          pairs, repeat=args.route_repeat)
        rv.train_supervised("rg_num_route_h", session.adapter, session.model,
                            session.processor, items, out_base / "route_h",
                            args.device, steps=args.route_steps,
                            warmup=args.route_warmup, lr=args.route_lr)
        hard = rg_hard_eval(session, ctx, args)
        acc = _strict_acc(hard, {i: ctx["baseline_alias_of"][i]
                                 for i in ctx["identity_ids"]})
        logger.info("RG1: extended baseline strict accuracy = %.4f", acc)
        est = {
            "dataset": "celeba_numeric", "kind": "rg_route_establishment",
            "checkpoint": str(ckpt), "checkpoint_sha256":
            rv.sha256_file(ckpt) if ckpt.exists() else None,
            "strict_accuracy": acc, "n": len(ctx["identity_ids"]),
            "recipe": {"steps": args.route_steps,
                       "warmup": args.route_warmup, "lr": args.route_lr,
                       "repeat": args.route_repeat, "seed": ROUTE_SEED},
            "per_identity": {i: {**hard[i],
                                 "expected": ctx["baseline_alias_of"][i]}
                             for i in ctx["identity_ids"]},
        }
        # NOT baseline_reference.json: that record is written by
        # baseline_reference() with the soft evidence beside every parse, so
        # the two never compete for one path.
        (out_base / "route_h_establishment.json").write_text(
            json.dumps(est, indent=2) + "\n", encoding="utf-8")
        if acc < 1.0 and not args.smoke:
            raise RuntimeError(
                f"RG1: the extended baseline route did not reach strict "
                f"accuracy 1.0 ({acc:.4f}).  A held-out value with no "
                f"established baseline association cannot be interpreted, "
                f"so the RG probe has no ground to stand on; refusing to "
                f"continue.  (smoke runs only warn.)")
    finally:
        session.release()
    return ckpt


def baseline_reference(args, ds, ctx, out_base):
    """Baseline behavior on every identity BEFORE any edit, both datasets.

    Establishes that each held-out member maps to its own label pre-edit;
    without this reference an entity-specific post-edit outcome could not be
    distinguished from a member the route never carried.  Returns
    ``(record, stale_reasons)``; a cached record is reused only after it
    proves it describes THIS checkpoint and THIS identity roster.
    """
    path = out_base / "baseline_reference.json"
    ckpt = _baseline_ckpt_rg(ds, out_base)
    stale = []
    if path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        reasons = validate_cached_reference(rec, ds, ctx, ckpt)
        if not reasons:
            logger.info("RG1: baseline reference cached + validated "
                        "(strict %.4f)", rec["strict_accuracy"])
            return rec, stale
        dest = _quarantine(path, reasons)
        stale.append({"record": "baseline_reference", "dataset": ds,
                      "reasons": reasons, "quarantined_as": str(dest)})
        logger.warning("RG1: cached baseline reference is STALE (%s); "
                       "quarantined as %s and recomputing",
                       ", ".join(reasons), dest.name)
    session = mx.ModelSession(args, f"e2c_rg_{ds}_ref")
    try:
        session.reset_to(ckpt)
        mx.seed_everything(ROUTE_SEED)
        hard = rg_hard_eval(session, ctx, args)
        soft = rg_soft_eval(session, ctx, args)
    finally:
        session.release()
    acc = _strict_acc(hard, {i: ctx["baseline_alias_of"][i]
                             for i in ctx["identity_ids"]})
    ref = {
        "dataset": ds, "kind": REF_KIND,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": rv.sha256_file(ckpt) if ckpt.exists() else None,
        "strict_accuracy": acc, "n": len(ctx["identity_ids"]),
        "per_identity": {
            i: {**hard[i], "expected": ctx["baseline_alias_of"][i],
                "p_expected": soft[i]["probs"].get(
                    ctx["baseline_alias_of"][i], 0.0),
                "candidate_mass": soft[i]["candidate_mass"],
                "other_mass": soft[i]["other_mass"]}
            for i in ctx["identity_ids"]},
    }
    path.write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    logger.info("RG1: baseline reference (%s) strict accuracy = %.4f", ds,
                acc)
    if acc < 1.0 and not args.smoke:
        raise RuntimeError(
            f"RG1: baseline route strict accuracy {acc:.4f} < 1.0 on {ds}; "
            f"held-out members must map to their own labels pre-edit or "
            f"their post-edit behavior cannot be attributed")
    return ref, stale


# ====================================================================== #
# RG2: cells -- withhold-and-train edits, strict classification
# ====================================================================== #
def _detail_classification(ds, ctx, iid, parsed, expected):
    """Depth/containment detail for an INDETERMINATE or confirming parse."""
    if parsed is None:
        return {"classification": "unparseable"}
    if ds == "salmu":
        return {"classification": gx.classify_taxonomic(
            parsed, expected, ctx["dag"])}
    prof = ctx["profiles"][iid]
    return gx.classify_numeric(parsed, expected,
                               ctx["schema"][prof["field"]],
                               prof["exact_value"], ctx["schema"])


def run_rg_cells(args, ds, ctx, man, out_base):
    """Train every (set, seed) edit and classify its held-out members.

    Returns ``(cells_root, stale_records)``: ``stale_records`` lists the
    cached cells that failed validation and were quarantined + rerun, so
    RG3 can report that a reuse was refused rather than silently taken.
    """
    cells_root = out_base / "cells"
    cells_root.mkdir(parents=True, exist_ok=True)
    baseline = _baseline_ckpt_rg(ds, out_base)
    only = set(args.only_sets) if args.only_sets else None
    only_seeds = set(args.only_seeds) if args.only_seeds else None
    n_sets = len([e for e in man["sets"] if not only
                  or e["set_id"] in only])
    eff_seeds = sorted({s for e in man["sets"] for s in e["seeds"]
                        if not only_seeds or s in only_seeds})
    logger.info("=" * 60)
    logger.info("RG2: CELLS (%s) -- %d/%d sets x EXECUTING seeds %s "
                "(the frozen manifest always declares %s)", ds, n_sets,
                man["n_sets"], eff_seeds, man["edit_seeds"])
    logger.info("=" * 60)
    stale = []
    session = mx.ModelSession(args, f"e2c_rg_{ds}_cell")
    try:
        for entry in man["sets"]:
            sid = entry["set_id"]
            if only and sid not in only:
                continue
            t = entry["trained"]
            t_iid = t["identity_id"]
            rule_label = t["rule_label"]
            hold = {h["identity_id"]: h for h in entry["holdouts"]}
            for seed in entry["seeds"]:
                if only_seeds and seed not in only_seeds:
                    continue
                cell_id = f"{sid}__seed{seed}"
                cell_dir = cells_root / sid / f"seed_{seed}"
                cell_dir.mkdir(parents=True, exist_ok=True)
                result_path = cell_dir / "rg_cell_results.json"
                ckpt = (cell_dir / "edited_h" / "adapter_final"
                        / "adapter_model.safetensors")
                if result_path.exists() and ckpt.exists():
                    rec = json.loads(result_path.read_text(encoding="utf-8"))
                    reasons = validate_cached_cell(
                        rec, ds, sid, seed, list(hold), ckpt)
                    if not reasons:
                        logger.info("[%s] cached + validated, skipping",
                                    cell_id)
                        continue
                    dest = _quarantine(result_path, reasons)
                    stale.append({"record": "rg_cell", "cell_id": cell_id,
                                  "reasons": reasons,
                                  "quarantined_as": str(dest)})
                    logger.warning("[%s] cached record is STALE (%s); "
                                   "quarantined as %s and rerunning",
                                   cell_id, ", ".join(reasons), dest.name)
                logger.info("-" * 56)
                logger.info("[%s] trained=%s rule=%r holdouts=%s retain=%d",
                            cell_id, t_iid, rule_label, sorted(hold),
                            len(entry["retain_ids"]))
                session.reset_to(baseline)
                mx.seed_everything(seed)
                # The edit's training data: the trained member's rule
                # mapping plus retention for every OTHER member -- with the
                # held-out members ABSENT.  That absence is the experiment:
                # whatever they do afterwards, they were not trained to do.
                pairs = [{"prompt": _prompt_for(ctx, t_iid),
                          "answer": rule_label}]
                retain_pairs = [
                    {"prompt": _prompt_for(ctx, i),
                     "answer": ctx["baseline_alias_of"][i]}
                    for i in entry["retain_ids"]]
                items = (
                    rv.build_supervised_items(session.adapter,
                                              session.processor, pairs,
                                              repeat=args.ul_repeat
                                              * gxm.TARGET_BOOST)
                    + rv.build_supervised_items(session.adapter,
                                                session.processor,
                                                retain_pairs,
                                                repeat=args.ul_repeat
                                                * gxm.RETAIN_REPEAT))
                rv.train_supervised(f"rg_{cell_id}", session.adapter,
                                    session.model, session.processor, items,
                                    cell_dir / "edited_h", args.device,
                                    steps=args.ul_steps,
                                    warmup=args.ul_warmup, lr=args.ul_lr)
                mx.seed_everything(seed)
                hard = rg_hard_eval(session, ctx, args)
                soft = rg_soft_eval(session, ctx, args)
                # ---- trained member: the precondition, not an outcome ----
                t_parsed = hard[t_iid]["parsed_label"]
                edit_success = t_parsed == rule_label
                trained_block = {
                    "identity_id": t_iid, "code": t["code"],
                    "source_alias": t["source_alias"],
                    "rule_label": rule_label,
                    "raw": hard[t_iid]["raw"], "parsed_label": t_parsed,
                    "edit_success": edit_success,
                    "p_rule_label": soft[t_iid]["probs"].get(rule_label, 0.0),
                    "p_source_alias": soft[t_iid]["probs"].get(
                        t["source_alias"], 0.0),
                    "detail": _detail_classification(
                        ds, ctx, t_iid, t_parsed, rule_label),
                }
                # ---- held-out members: the measurement ----
                holdout_blocks = {}
                outcomes = {}
                for hid, h in sorted(hold.items()):
                    parsed = hard[hid]["parsed_label"]
                    baseline_alias = h["baseline_alias"]
                    outcome = classify_holdout_outcome(
                        parsed, baseline_alias, rule_label, h["role"])
                    outcomes[hid] = outcome
                    holdout_blocks[hid] = {
                        "role": h["role"], "code": h["code"],
                        "baseline_alias": baseline_alias,
                        "rule_label": rule_label,
                        "raw": hard[hid]["raw"], "parsed_label": parsed,
                        "recognized_labels": hard[hid]["recognized_labels"],
                        "multi_label_invalid":
                            hard[hid]["multi_label_invalid"],
                        "outcome": outcome,
                        "detail": _detail_classification(
                            ds, ctx, hid, parsed, baseline_alias),
                        "soft": {
                            "p_rule_label": soft[hid]["probs"].get(
                                rule_label, 0.0),
                            "p_baseline_alias": soft[hid]["probs"].get(
                                baseline_alias, 0.0),
                            "candidate_mass": soft[hid]["candidate_mass"],
                            "other_mass": soft[hid]["other_mass"],
                        },
                        **({"exact_value": h["exact_value"]}
                           if "exact_value" in h else {}),
                    }
                # ---- boundary discontinuities (numeric) ----
                boundary = []
                for p in entry.get("boundary_pairs", []):
                    out_iid = p["outside_identity_id"]
                    out_outcome = outcomes.get(out_iid)
                    if p.get("inside_is_trained_member"):
                        inside_emits_rule = edit_success
                    else:
                        inside_emits_rule = (
                            outcomes.get(p["inside_identity_id"])
                            == RULE_GENERALIZED)
                    boundary.append({
                        "edge": p["edge"],
                        "inside_identity_id": p["inside_identity_id"],
                        "outside_identity_id": out_iid,
                        "inside_emits_rule": inside_emits_rule,
                        "outside_outcome": out_outcome,
                        "verdict": boundary_pair_verdict(inside_emits_rule,
                                                         out_outcome),
                    })
                # ---- retention-trained controls ----
                drift = [i for i in entry["retain_ids"]
                         if hard[i]["parsed_label"]
                         != ctx["baseline_alias_of"][i]]
                retain_acc = ((len(entry["retain_ids"]) - len(drift))
                              / len(entry["retain_ids"])
                              if entry["retain_ids"] else None)
                verdict = set_verdict(edit_success, outcomes)
                cell = {
                    "cell_id": cell_id, "dataset": ds, "set_id": sid,
                    "seed": seed, "kind": CELL_KIND,
                    "trained": trained_block,
                    "edit_success": edit_success,
                    "holdouts": holdout_blocks,
                    "outcomes": outcomes,
                    "verdict": verdict,
                    "boundary_pairs": boundary,
                    "retain_controls": {"n": len(entry["retain_ids"]),
                                        "strict_accuracy": retain_acc,
                                        "drifted_ids": drift},
                    "min_candidate_mass": min(
                        v["candidate_mass"] for v in soft.values()),
                    "hard_preds": hard,
                    "soft_probs_full": {i: soft[i]["probs"]
                                        for i in ctx["identity_ids"]},
                    "checkpoint_sha256": rv.sha256_file(ckpt),
                    "provenance": {
                        "git_commit": EXECUTING_COMMIT,
                        "runner_script_sha256": rv.sha256_file(
                            Path(__file__).resolve()),
                        "shared_scoring_script_sha256": rv.script_sha256(),
                        "rg_manifest_sha256": rv.sha256_file(
                            rg_manifest_path(ds)),
                        "device": args.device,
                    },
                }
                result_path.write_text(json.dumps(cell, indent=2) + "\n",
                                       encoding="utf-8")
                comp = {}
                for o in outcomes.values():
                    comp[o] = comp.get(o, 0) + 1
                logger.info("[%s] edit_success=%s verdict=%s outcomes=%s "
                            "retain=%.3f", cell_id, edit_success, verdict,
                            comp, retain_acc if retain_acc is not None
                            else float("nan"))
    finally:
        session.release()
    return cells_root, stale


# ====================================================================== #
# RG3: aggregation, report, archive
# ====================================================================== #
def _load_cells(out_base, man, ds):
    """Load the ACCEPTED cells, validating each one before it is trusted.

    Quarantined ``*.stale-*.json`` siblings are never read, and a record
    that is readable but does not match the current frozen manifest, the
    checkpoint bytes on disk or the declared scorer is REJECTED here (an
    RG3-only rerun must not aggregate what an RG2 run refused).  Returns
    ``(cells, rejected)``.
    """
    cells, rejected = {}, []
    for entry in man["sets"]:
        hold_ids = [h["identity_id"] for h in entry["holdouts"]]
        for seed in entry["seeds"]:
            cell_dir = out_base / "cells" / entry["set_id"] / f"seed_{seed}"
            path = cell_dir / "rg_cell_results.json"
            if not path.exists():
                continue
            rec = json.loads(path.read_text(encoding="utf-8"))
            ckpt = (cell_dir / "edited_h" / "adapter_final"
                    / "adapter_model.safetensors")
            reasons = validate_cached_cell(rec, ds, entry["set_id"], seed,
                                           hold_ids, ckpt)
            if reasons:
                rejected.append({
                    "cell_id": f"{entry['set_id']}__seed{seed}",
                    "reasons": reasons, "path": str(path)})
                logger.warning("RG3: rejecting cached cell %s (%s)",
                               f"{entry['set_id']}__seed{seed}",
                               ", ".join(reasons))
                continue
            cells[f"{entry['set_id']}__seed{seed}"] = rec
    return cells, rejected


def aggregate_rg(ds, man, out_base):
    cells, rejected = _load_cells(out_base, man, ds)
    sets_out = {}
    for entry in man["sets"]:
        sid = entry["set_id"]
        per_seed = {}
        verdicts = []
        holdout_comp = {h["identity_id"]: {} for h in entry["holdouts"]}
        boundary_rows = []
        for seed in entry["seeds"]:
            c = cells.get(f"{sid}__seed{seed}")
            if c is None:
                continue
            verdicts.append(c["verdict"])
            for hid, outcome in c["outcomes"].items():
                holdout_comp[hid][outcome] = holdout_comp[hid].get(
                    outcome, 0) + 1
            boundary_rows.extend(
                [dict(b, seed=seed) for b in c["boundary_pairs"]])
            per_seed[str(seed)] = {
                "edit_success": c["edit_success"], "verdict": c["verdict"],
                "outcomes": c["outcomes"],
                "retain_strict_accuracy":
                    c["retain_controls"]["strict_accuracy"],
                "retain_drifted_ids": c["retain_controls"]["drifted_ids"],
                "min_candidate_mass": c["min_candidate_mass"],
                "p_rule_on_trained": c["trained"]["p_rule_label"],
                "p_source_on_trained": c["trained"]["p_source_alias"],
            }
        comp = {}
        for v in verdicts:
            comp[v] = comp.get(v, 0) + 1
        sets_out[sid] = {
            "trained": entry["trained"],
            "holdouts_declared": entry["holdouts"],
            "n_seeds_expected": len(entry["seeds"]),
            "n_seeds_evaluated": len(per_seed),
            "executed_seeds": sorted(int(k) for k in per_seed),
            "per_seed": per_seed,
            "verdict_composition": comp,
            "consistent_across_seeds": len(set(verdicts)) == 1
            if verdicts else None,
            "holdout_outcome_composition": holdout_comp,
            "boundary_pairs": boundary_rows,
        }
    all_verdicts = [c["verdict"] for c in cells.values()]
    comp_all = {}
    for v in all_verdicts:
        comp_all[v] = comp_all.get(v, 0) + 1
    expected_cells = man["n_cells"]
    return {
        "dataset": ds,
        "n_cells_expected": expected_cells,
        "n_cells_evaluated": len(cells),
        "complete": len(cells) >= expected_cells,
        "rejected_cached_cells": rejected,
        "verdict_composition_all_set_seeds": comp_all,
        "sets": sets_out,
    }


def build_rg_claims(ds, agg, man):
    """Claims in the exact taxonomy; compositions are never collapsed."""
    comp = agg["verdict_composition_all_set_seeds"]
    n = agg["n_cells_evaluated"]
    comp_text = ", ".join(f"{k}={v}" for k, v in sorted(comp.items())) \
        or "none evaluated"
    executed = sorted({seed for row in agg.get("sets", {}).values()
                       for seed in row.get("executed_seeds", [])})
    scope = (
        f"every outcome in this report was measured at the association level "
        f"(code -> label) on the canonical prompt only, under strict parsing "
        f"(multi-label = invalid, never first-match), with held-out members "
        f"ABSENT from the edit's training data; dataset {ds}, "
        f"{agg['n_cells_evaluated']}/{agg['n_cells_expected']} set-seed "
        f"cells evaluated, EXECUTED edit seeds {executed} (the frozen "
        f"manifest declares {man['edit_seeds']}); soft candidate "
        f"scores are recorded beside every strict decision and never "
        f"override it")
    headline = (
        f"rule generalization vs entity-specific abstraction ({ds}): across "
        f"{n} evaluated set-seed cells the strict verdicts are [{comp_text}]"
        f".  A cell is {VERDICT_ENTITY} only when the edit succeeded on its "
        f"trained member AND every held-out member kept its own baseline "
        f"label (boundary-outside members keeping their label count as "
        f"{RULE_CONSISTENT}); it is {VERDICT_RULE} only when the edit "
        f"succeeded AND every held-out branch/interior member followed the "
        f"rule AND no boundary-outside member crossed into the bin; mixed "
        f"compositions are reported as {VERDICT_MIXED} and never collapsed "
        f"into either pure label; a failed edit is {VERDICT_EDIT_FAILURE} "
        f"and measures no generalization")
    not_claimed = [
        ("no threshold, gate, promotion criterion or training recipe in this "
         "repository changes as a result of this probe, and nothing reads "
         "its output"),
        ("entity_specific here means the held-out member kept its baseline "
         "label under THIS edit recipe on the canonical prompt; it is not "
         "evidence that no recipe or prompt would generalize -- prompt "
         "robustness is the prompt panel's question"),
        ("rule_generalization here is a behavioral observation about "
         "withheld branch members; it is not a claim that the rule is "
         "represented symbolically or would transfer beyond the evaluated "
         "members"),
        ("the numeric extension profiles (GRN_24..GRN_29) exist only in "
         "this tree; the frozen 24-profile granularity benchmark, its "
         "baseline route and its cells are untouched"),
        ("boundary_outside members keeping their own label are counted as "
         "rule_consistent, never as entity_specific: the two outcomes "
         "answer different questions and are reported separately "
         "everywhere"),
    ]
    return {"scope": scope, "headline": headline, "not_claimed": not_claimed}


def archive_rg(ds, out_base, man, commit):
    """Checkpoint + report archive, revision-pinned (GX2S pattern)."""
    rel = Path("releases") / f"e2c_rulegen_{ds}_{commit[:7]}"
    rel.mkdir(parents=True, exist_ok=True)
    entries = []
    base_ckpt = _baseline_ckpt_rg(ds, out_base)
    if ds == "celeba_numeric" and base_ckpt.exists():
        dest = rel / "rg_route_h_extended.safetensors"
        shutil.copy2(base_ckpt, dest)
        entries.append({"kind": "rg_baseline_route", "key": "route_h_30",
                        "file": dest.name,
                        "sha256": rv.sha256_file(dest),
                        "bytes": dest.stat().st_size})
    for entry in man["sets"]:
        for seed in entry["seeds"]:
            src = (out_base / "cells" / entry["set_id"] / f"seed_{seed}"
                   / "edited_h" / "adapter_final"
                   / "adapter_model.safetensors")
            if not src.exists():
                continue
            name = f"rg_{entry['set_id']}__seed{seed}.safetensors"
            dest = rel / name
            shutil.copy2(src, dest)
            entries.append({"kind": "rg_edited_cell",
                            "key": f"{entry['set_id']}__seed{seed}",
                            "file": name, "sha256": rv.sha256_file(dest),
                            "bytes": dest.stat().st_size})
    report_src = RG_REPORT_DIR / f"rule_generalization_{ds}.json"
    if report_src.exists():
        shutil.copy2(report_src, rel / report_src.name)
    hf_ok, hf_revision = False, None
    try:
        from huggingface_hub import HfApi, whoami
        whoami()
        api = HfApi()
        api.create_repo(gxm.HF_ARCHIVE_REPO, repo_type="model",
                        exist_ok=True)
        url = api.upload_folder(
            folder_path=str(rel), repo_id=gxm.HF_ARCHIVE_REPO,
            path_in_repo=f"rule_generalization_{ds}_{commit[:7]}",
            repo_type="model",
            commit_message=f"E2C-v3 rule generalization ({ds}) @ "
                           f"{commit[:7]}")
        hf_ok = True
        hf_revision = url.rstrip("/").rsplit("/", 1)[-1] if url else None
        for e in entries:
            e["hf_revision"] = hf_revision
            e["hf_uri"] = (f"https://huggingface.co/{gxm.HF_ARCHIVE_REPO}/"
                           f"resolve/{hf_revision}/rule_generalization_"
                           f"{ds}_{commit[:7]}/{e['file']}")
        logger.info("RG3: uploaded %d checkpoints, revision %s",
                    len(entries), hf_revision)
    except Exception as exc:
        logger.warning("RG3: HF upload unavailable (%s)", str(exc)[:120])
    with open(rel / "CHECKSUMS.txt", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(f"{e['sha256']}  {e['file']}\n")
    manifest = {"kind": "rule_generalization_archive", "dataset": ds,
                "git_commit": commit, "release_dir": str(rel),
                "hf_repo": gxm.HF_ARCHIVE_REPO if hf_ok else None,
                "hf_upload_ok": hf_ok, "hf_revision": hf_revision,
                "n_files": len(entries),
                "uri_immutability_note": "resolve/<hf_commit_sha> pinned",
                "entries": entries}
    (rel / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_rg3(args, ds, ctx, man, out_base, provenance, commit, t_start,
            stale=None):
    agg = aggregate_rg(ds, man, out_base)
    if not agg["n_cells_evaluated"]:
        rejected = agg["rejected_cached_cells"]
        detail = ("; every cached record on disk was REJECTED: "
                  + json.dumps(rejected)) if rejected else ""
        raise RuntimeError("RG3: no validated cells to aggregate; run RG2 "
                           f"first{detail}")
    ref_path = out_base / "baseline_reference.json"
    baseline_ref, ref_rejected = None, []
    if ref_path.exists():
        rec = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_rejected = validate_cached_reference(
            rec, ds, ctx, _baseline_ckpt_rg(ds, out_base))
        if ref_rejected:
            logger.warning("RG3: baseline reference is STALE (%s); it is "
                           "reported as rejected, not as evidence",
                           ", ".join(ref_rejected))
        else:
            baseline_ref = rec
    cache_block = {
        "policy": RG_CACHE_VALIDATION,
        "stale_records_quarantined": stale or [],
        "cached_cells_rejected_at_aggregation":
            agg["rejected_cached_cells"],
        "baseline_reference_rejected": ref_rejected,
    }
    archive = {"release_dir": None, "hf_upload_ok": False,
               "hf_revision": None, "n_files": 0}
    if not args.smoke:
        archive = archive_rg(ds, out_base, man, commit)
    report = {
        "kind": "e2c_v3_rule_generalization_report_v1",
        "dataset": ds,
        "produced_by": "scripts/e2c_v3_rule_generalization.py",
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "provenance": provenance,
        "taxonomy": TAXONOMY,
        "cache_validation": cache_block,
        "baseline_reference": baseline_ref,
        "aggregate": agg,
        "claims": build_rg_claims(ds, agg, man),
        "archive": {"release_dir": archive.get("release_dir"),
                    "hf_upload_ok": archive.get("hf_upload_ok"),
                    "hf_revision": archive.get("hf_revision"),
                    "n_files": archive.get("n_files", 0)},
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    RG_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = RG_REPORT_DIR / f"rule_generalization_{ds}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("RG3: report written to %s", path)
    logger.info("RG3 CLAIM scope: %s", report["claims"]["scope"])
    logger.info("RG3 CLAIM headline: %s", report["claims"]["headline"])
    # run manifest beside the outputs
    checkpoints = {cid: c["checkpoint_sha256"] for cid, c in
                   _load_cells(out_base, man, ds)[0].items()}
    run_manifest = {
        "experiment": f"e2c_v3_rule_generalization_{ds}",
        "produced_by": "scripts/e2c_v3_rule_generalization.py",
        "provenance": provenance,
        "inputs_sha256": man["inputs"],
        "rg_manifest_sha256": rv.sha256_file(rg_manifest_path(ds)),
        "baseline_checkpoint_sha256": (baseline_ref or {}).get(
            "checkpoint_sha256"),
        "checkpoints_sha256": checkpoints,
        "results": {"n_cells_evaluated": agg["n_cells_evaluated"],
                    "n_cells_expected": agg["n_cells_expected"],
                    "verdict_composition":
                        agg["verdict_composition_all_set_seeds"]},
        "cache_validation": cache_block,
        "archive": report["archive"],
        "cumulative_elapsed_sec": round(time.time() - t_start, 1),
        "updated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
    }
    (out_base / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    return report


# ====================================================================== #
# Provenance and entry point
# ====================================================================== #
RG_CODE = ["scripts/e2c_v3_rule_generalization.py",
           "scripts/e2c_v3_research_validity.py",
           "scripts/e2c_v3_granularity.py",
           "scripts/e2c_v3_granularity_matrix.py",
           "scripts/e2c_v3_matrix.py",
           "scripts/e2c_v3_realdata.py"]


def _dirty_tracked_code():
    """Tracked-file changes among the EXECUTED code (not result outputs).

    A declared script that is not on disk is reported as such: dropping it from
    the pathspec would silently widen ``git status`` to the WHOLE worktree and
    blame this run for edits made by the parallel ones.
    """
    missing = [p for p in RG_CODE if not Path(p).exists()]
    if missing:
        return [f"<declared executed code missing: {p}>" for p in missing]
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no", *RG_CODE],
            text=True)
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return ["<git status failed>"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   choices=["salmu", "celeba_numeric"])
    p.add_argument("--phase", default="all",
                   choices=["all", "RG0", "RG1", "RG2", "RG3"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", type=int, nargs="+", default=RG_SEEDS)
    p.add_argument("--only-sets", nargs="*", default=None)
    p.add_argument("--only-seeds", type=int, nargs="*", default=None)
    p.add_argument("--route-steps", type=int, default=mx.ROUTE_STEPS)
    p.add_argument("--route-warmup", type=int, default=mx.ROUTE_WARMUP)
    p.add_argument("--route-lr", type=float, default=mx.ROUTE_LR)
    p.add_argument("--route-repeat", type=int, default=mx.ROUTE_REPEAT)
    p.add_argument("--ul-steps", type=int, default=mx.UL_STEPS)
    p.add_argument("--ul-warmup", type=int, default=mx.UL_WARMUP)
    p.add_argument("--ul-lr", type=float, default=mx.UL_LR)
    p.add_argument("--ul-repeat", type=int, default=mx.UL_REPEAT)
    p.add_argument("--max-gen-tokens", type=int, default=12)
    p.add_argument("--smoke", action="store_true",
                   help="tiny steps/repeats, seed 17 only, *_smoke tree, "
                        "gates warn instead of raising, no archive")
    return p.parse_args()


def main():
    global EXECUTING_COMMIT
    args = parse_args()
    t_start = time.time()
    if args.smoke:
        args.route_steps, args.route_warmup = 20, 2
        args.ul_steps, args.ul_warmup = 20, 2
        args.route_repeat = 2
        args.only_seeds = [17]
    args.seed = (args.only_seeds or args.seeds)[0]
    ds = args.dataset
    suffix = "_smoke" if args.smoke else ""
    out_base = RG_OUT_ROOT / f"{ds}{suffix}"
    out_base.mkdir(parents=True, exist_ok=True)

    # Captured BEFORE any artifact is written (the 9e488a3 rule).
    EXECUTING_COMMIT = rv.git_commit_sha()
    dirty_code = _dirty_tracked_code()
    provenance = {
        "commit": EXECUTING_COMMIT,
        "runner_script_sha256": rv.sha256_file(Path(__file__).resolve()),
        "shared_scoring_script_sha256": rv.script_sha256(),
        "granularity_lib_sha256": rv.sha256_file(
            SCRIPT_DIR / "e2c_v3_granularity.py"),
        "granularity_runner_sha256": rv.sha256_file(
            SCRIPT_DIR / "e2c_v3_granularity_matrix.py"),
        "dirty": rv.git_worktree_dirty(),
        "dirty_executed_code": dirty_code,
        "clean_code_required": not args.smoke,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": args.device,
    }
    logger.info("Provenance: commit=%s runner=%s rv=%s dirty=%s",
                EXECUTING_COMMIT[:12],
                provenance["runner_script_sha256"][:12],
                provenance["shared_scoring_script_sha256"][:12],
                provenance["dirty"])
    if dirty_code and not args.smoke:
        raise RuntimeError(
            f"{args.phase}: executed CODE not committed: {dirty_code}.  "
            f"Result files may be dirty (parallel granularity/panel runs "
            f"own them), but the code that produces this probe may not be.")

    man = build_or_verify(ds)
    ctx = rg_dataset_ctx(ds, man)
    if args.phase == "RG0":
        logger.info("RG0 COMPLETE -- commit the manifest before evaluating")
        return 0
    if args.smoke:
        man = json.loads(json.dumps(man))
        for s in man["sets"]:
            s["seeds"] = [17]
        man["n_cells"] = man["n_sets"]

    stale_records = []
    if args.phase in ("all", "RG1"):
        if ds == "celeba_numeric":
            ensure_rg_numeric_route(args, ctx, out_base)
        _ref, ref_stale = baseline_reference(args, ds, ctx, out_base)
        stale_records.extend(ref_stale)
    if args.phase in ("all", "RG2"):
        _cells_root, cell_stale = run_rg_cells(args, ds, ctx, man, out_base)
        stale_records.extend(cell_stale)
    if args.phase in ("all", "RG3"):
        run_rg3(args, ds, ctx, man, out_base, provenance, EXECUTING_COMMIT,
                t_start, stale=stale_records)
    logger.info("=" * 60)
    logger.info("RULE GENERALIZATION (%s) PHASE %s COMPLETE (%.1fs)", ds,
                args.phase, time.time() - t_start)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
