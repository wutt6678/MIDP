#!/usr/bin/env python3
"""E2C-v3 granularity METHOD baselines (MB0-MB2): which edit method works?

THE QUESTION
============
Every granularity result so far was produced by ONE method: teacher-forced
supervised fine-tuning on the coarser label plus weighted retention (the
frozen "edited cell E" recipe).  The 147-cell matrix (SALMU 63 + numeric 84)
therefore measures what ONE editing objective can do.  Before any method is
multiplied across that matrix, this runner puts the candidate objectives side
by side on the SAME transformation target -- never the refusal target -- over
a frozen representative subset, with identical target/retain data and
identical training budgets wherever that is methodologically meaningful.

METHODS (frozen table, MB_METHODS)
==================================
sft_target          direct supervised fine-tuning on the coarser label plus
                    weighted retention.  This IS the frozen granularity cell
                    E: the existing seed-17 checkpoint is REUSED (validated by
                    hash), never retrained, and re-scored with this runner's
                    metric set so every method is measured identically.
cf_relabel          counterfactual supervised relabeling.  For a
                    TRANSFORMATION target the counterfactual label IS the
                    requested coarser label, so CF and direct SFT are the same
                    objective on the same data -- they differ only in the
                    deletion setting, where the counterfactual is the refusal
                    label.  Reported as an explicit ALIAS of sft_target with
                    the reason recorded; no duplicate training, no pretending
                    the two are independent evidence.
gd_distribution     distribution matching toward the coarser label: minimize
                    KL(one-hot(target) || normalized full-sequence candidate
                    scores) plus supervised descent on the retain set.  This
                    is the transformation analog of rv.train_gd_refusal with
                    one deliberate repair: the refusal version matches FIRST
                    TOKENS, and the numeric bin labels share their first token
                    with the exact labels ("15 years" and "15-19 years" both
                    start with "15"), so first-token matching cannot even
                    EXPRESS a granularity transformation.  GD here matches
                    full-sequence label scores over the frozen candidate
                    vocabulary, batched into one forward per step.
ga_retain_descent   gradient ASCENT on the source association (code -> the
                    specific label being replaced) plus supervised DESCENT on
                    the retain set AND the target mapping (rv.train_ga, the
                    corrected version: ascent on the forget side only).
npo                 Negative Preference Optimization against the pre-edit
                    reference on the source association, descent on
                    retain+target (rv.train_npo, beta=1.0 frozen from RV3).
kl_anchored_edit    KL-anchored EDITING: supervised descent on target+retain
                    with a beta_kl * KL(pi_theta || pi_ref) anchor measured on
                    the retain prompts (beta_kl=0.5, the frozen RV value).
kl_ascent_anchor    rv.train_kl exactly as shipped for DELETION (ascent on the
                    source + KL anchor, and NO descent term anywhere).  Kept as
                    a separate row because it is what "KL" meant in RV3, and it
                    is structurally unable to install a new label: running it
                    shows that empirically instead of asserting it.
prompt_only         no weight change at all: the frozen baseline route under
                    instruction policies that state the abstraction rule.
matched_retrain     fresh matched retraining: the EXISTING fresh-LoRA
                    matched_retrain oracle for the set (3000/200/2e-5, target
                    x5, retain x50, seed 17).  Its budget is NOT the edit
                    budget and is flagged as such -- it is the reference, not a
                    competitor for the same compute.
Reference rows (not methods): baseline_route (the pre-edit floor) and
loo_retrain (the fresh deletion reference the Delta is measured against).

DATA AND BUDGET DISCIPLINE
==========================
One frozen data spec per set: target pairs (each transformed member -> its
coarser label, x ul_repeat*TARGET_BOOST), source pairs (each transformed
member -> the specific label being replaced, same weight: the ascent side for
the ascent-family methods), retain pairs (every other member -> its own
baseline label, x ul_repeat*RETAIN_REPEAT).  Every trainable method consumes
exactly these pairs and exactly the frozen edit budget (steps/warmup/lr/seed
of the granularity cell recipe); methods differ ONLY in the objective and in
which side of it each pair set feeds.  Each row records its own recipe, wall
clock and trainable-parameter count, so a budget difference is never implicit.

METRICS (all of them, per method per set)
=========================================
transformation success (strict, per target and all-target rate) | source-label
suppression (p_source full-sequence + hard leak) | retention (strict accuracy,
worst retained member, drifted ids) | sibling preservation (strict accuracy
with sibling COVERAGE, null when a set has no sibling control -- never a
vacuous 1.0) | wrong-branch rate (frozen depth classification) | candidate
validity (multi-label invalid, unparseable, candidate score sums and whether
they clear the frozen 0.99 interpretability threshold) | matched-vs-LOO
oracle distance (gated L2 to matched_retrain and loo_retrain plus
Delta_retrain = D(loo) - D(matched), per transformation target) | training
time and trainable parameters.  The per-cell criteria block is computed by the
SAME frozen function the matrix uses (gxm._pass_criteria), so the yardstick is
identical to the cells, and the reused sft_target row is cross-checked against
the committed cell's own criteria block.

PROMPT-ONLY HONESTY
===================
prompt_only changes no weights, so its retention and sibling numbers are 1.0
BY CONSTRUCTION -- that is not evidence of a better editing method and the
report says so.  Its policies are frozen text that state the abstraction rule
without naming the answer: MB0 hard-fails if a policy prompt contains the
target label or the source label of the set it is used on, and few-shot
demonstrations are drawn from a DIFFERENT branch/bin than the target so a
demonstration can never hand over the answer.  Primary mode applies the policy
to the transformed members only (comparable to the weight-editing rows); a
hard-generation over-application probe additionally reports what the same
instruction does when it is applied to every member.

CACHED RECORDS ARE VALIDATED BEFORE REUSE
=========================================
Same policy as the prompt panel and the RG probe: a readable JSON file is not
evidence that it measured THIS method on THIS set.  Every reused record must
match the frozen design manifest hash, the checkpoint bytes on disk, the
dataset/set/seed, the method id and recipe, and the declared scorer; a
mismatch is quarantined as <stem>.stale-<sha8(reasons)>.json and redone, and
aggregation rejects records it cannot validate.

Phases: MB0 build+verify the frozen design manifest (CPU) | MB1 train/eval
        every method row (cached per set x row) | MB2 aggregate + report +
        archive.  ``--phase all`` runs MB0 then MB1 then MB2.

Usage:
    python scripts/e2c_v3_method_baselines.py --dataset salmu --phase all \
        --device cuda:0
    python scripts/e2c_v3_method_baselines.py --dataset celeba_numeric \
        --phase all --device cuda:0
GPU: one device via CUDA_VISIBLE_DEVICES with --device cuda:0 inside the
masked namespace.  Qwen3.5-9B bf16 needs ~19-20 GB resident; this runner
keeps ONE session alive per dataset and resets it between methods.
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
logger = logging.getLogger("e2c_v3_mb")

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rv = _load_sibling("e2c_rv_mb", "e2c_v3_research_validity.py")
rd = _load_sibling("e2c_rd_mb", "e2c_v3_realdata.py")
mx = _load_sibling("e2c_mx_mb", "e2c_v3_matrix.py")
gx = _load_sibling("e2c_gx_mb", "e2c_v3_granularity.py")
gxm = _load_sibling("e2c_gxm_mb", "e2c_v3_granularity_matrix.py")

MB_ROOT = Path("e2c_method_baselines")
MB_MANIFEST_DIR = MB_ROOT / "manifests"
MB_OUT_ROOT = MB_ROOT / "outputs"
MB_REPORT_DIR = MB_ROOT / "reports"

#: The edit seed the representative cells and the retrain oracles exist at.
MB_SEED = gxm.ORACLE_SEED                     # 17
#: Frozen representative subset: one set per transformation TYPE (the same
#: representatives GX2B/GX2S used), never the full 147-cell matrix.
MB_REP_SETS = {ds: list(sids) for ds, sids in gxm.BALANCED_REP_SETS.items()}

#: Captured in main BEFORE any artifact is written (the 9e488a3 rule).
EXECUTING_COMMIT = None

# ====================================================================== #
# Frozen method table
# ====================================================================== #
#: trainer keys: reuse (read an existing checkpoint), rv.* (shipped trainer),
#: mb.* (implemented here), prompt (no weights).
MB_METHODS = {
    "sft_target": {
        "label": "direct supervised fine-tuning (frozen cell E)",
        "objective": ("teacher-forced CE descent on target+retain pairs"),
        "trainer": "reuse_granularity_cell",
        "ascent_set": None, "descent_set": ["target", "retain"],
        "budget": "edit", "trains": False,
        "note": ("the existing seed-17 edited cell checkpoint is reused after "
                 "hash validation and re-scored by this runner"),
    },
    "cf_relabel": {
        "label": "counterfactual supervised relabeling",
        "objective": ("teacher-forced CE descent toward the counterfactual "
                      "label"),
        "trainer": "alias", "aliased_to": "sft_target",
        "ascent_set": None, "descent_set": ["target", "retain"],
        "budget": "edit", "trains": False,
        "note": ("for a transformation target the counterfactual label IS the "
                 "requested coarser label, so CF and direct SFT coincide by "
                 "construction; recorded as an alias, never as independent "
                 "evidence"),
    },
    "gd_distribution": {
        "label": "GD distribution matching toward the coarser label",
        "objective": ("KL(one-hot(target) || normalized full-sequence "
                      "candidate scores) + retain descent"),
        "trainer": "mb.train_gd_target",
        "ascent_set": None, "descent_set": ["retain"],
        "distribution_set": "target", "anchor_set": None,
        "budget": "edit", "trains": True,
        "note": ("full-sequence label scores, batched; the refusal variant's "
                 "first-token matching cannot express a bin transformation "
                 "because '15 years' and the 15-19 bin share a first token"),
    },
    "ga_retain_descent": {
        "label": "gradient ascent on the source + retain descent",
        "objective": ("ascent on the source association, descent on "
                      "retain+target"),
        "trainer": "rv.train_ga",
        "ascent_set": "source", "descent_set": ["retain", "target"],
        "anchor_set": None,
        "budget": "edit", "trains": True,
        "note": "rv.train_ga as corrected in RV: ascent on the forget side "
                "only, so retention is genuinely trained",
    },
    "npo": {
        "label": "Negative Preference Optimization vs the pre-edit reference",
        "objective": ("-log sigmoid(beta * (loss_theta - loss_ref)) on the "
                      "source association + 0.1 * retain/target descent"),
        "trainer": "rv.train_npo", "beta": 1.0,
        "ascent_set": "source", "descent_set": ["retain", "target"],
        "anchor_set": None,
        "budget": "edit", "trains": True,
        "note": "beta frozen at the RV3 value; reference = pre-edit route",
    },
    "kl_anchored_edit": {
        "label": "KL-anchored editing (descent + anchor)",
        "objective": ("CE descent on target+retain + beta_kl * "
                      "KL(pi_theta || pi_ref) on retain prompts"),
        "trainer": "mb.train_kl_anchored_edit", "beta_kl": 0.5,
        "ascent_set": None, "descent_set": ["target", "retain"],
        "anchor_set": ["retain"],
        "budget": "edit", "trains": True,
        "note": ("the transformation reading of 'KL-anchored editing': the "
                 "anchor bounds drift while the descent installs the label"),
    },
    "kl_ascent_anchor": {
        "label": "rv.train_kl as shipped for deletion (ascent + anchor)",
        "objective": ("ascent on the source association + beta_kl * "
                      "KL(pi_theta || pi_ref) on the retain set, and no CE "
                      "descent anywhere"),
        "trainer": "rv.train_kl", "beta_kl": 0.5,
        "ascent_set": "source", "descent_set": [],
        "anchor_set": ["retain"],
        "budget": "edit", "trains": True,
        "note": ("kept as its own row: it has NO descent term, so it cannot "
                 "install a coarser label -- measured, not asserted"),
    },
    "prompt_only": {
        "label": "prompting-only policy (no weight change)",
        "objective": "none: frozen baseline route under instruction policies",
        "trainer": "prompt", "ascent_set": None, "descent_set": [],
        "budget": "none", "trains": False,
        "note": ("retention and sibling numbers are 1.0 BY CONSTRUCTION "
                 "(weights untouched) and are never presented as a method "
                 "advantage"),
    },
    "matched_retrain": {
        "label": "fresh matched retraining (oracle)",
        "objective": ("fresh-LoRA supervised training on target+retain with "
                      "the original route protocol"),
        "trainer": "reuse_matched_retrain_oracle",
        "ascent_set": None, "descent_set": ["target", "retain"],
        "budget": "oracle", "trains": False,
        "note": ("the existing matched_retrain oracle checkpoint is reused; "
                 "3000/200/2e-5 is NOT the edit budget and is flagged.  This "
                 "is the WEIGHTED matched reference (target x5) every "
                 "existing Delta_retrain number was measured against; "
                 "whether the reading survives WITHOUT the boost is GX2B's "
                 "question and is not re-litigated here"),
    },
}
#: Reference rows: measured with the same metrics, never called methods.
MB_REFERENCES = {
    "baseline_route": {
        "label": "pre-edit baseline route h (floor)",
        "trainer": "reuse_baseline_route", "budget": "none", "trains": False,
        "note": "the association route before any edit; no target installed",
    },
    "loo_retrain": {
        "label": "fresh leave-one-out retraining (deletion reference)",
        "trainer": "reuse_loo_retrain_oracle", "budget": "oracle",
        "trains": False,
        "note": "the reference Delta_retrain is measured against",
    },
}
MB_TRAINABLE = [m for m, s in MB_METHODS.items() if s["trains"]]
MB_ALL_ROWS = [*MB_METHODS, *MB_REFERENCES]

#: The metrics the task requires; MB0 fails if the report cannot emit them.
MB_REQUIRED_METRICS = [
    "transformation_success", "source_label_suppression", "retention",
    "sibling_preservation", "wrong_branch_rate", "candidate_validity",
    "matched_vs_loo_oracle_distance", "training_time", "trainable_parameters",
]

# ====================================================================== #
# Frozen prompt-only policies (text is part of the design manifest)
# ====================================================================== #
PROMPT_POLICY_ROLES = ("canonical_control", "coarsen_instruction",
                       "rule_statement", "fewshot_other_branch")

POLICY_TEMPLATES = {
    "taxonomic": {
        "coarsen_instruction": (
            "Identity code: {code}. Generate the alias for this identity at "
            "the broad professional class level - the level-{depth} ancestor "
            "of their specific job title in the profession taxonomy - instead "
            "of the specific job title."),
        "rule_statement": (
            "Identity code: {code}. Policy: report this identity at "
            "abstraction level {depth} of the profession taxonomy (the "
            "level-{depth} parent class), not at level 0. Generate the "
            "alias."),
        "fewshot_other_branch": (
            "Map each identity code to its level-{depth} professional "
            "class.\n"
            "{demos}\n"
            "Identity code: {code}. Generate the alias."),
    },
    "numeric": {
        "coarsen_instruction": (
            "Identity code: {code}. Generate the alias for this identity as "
            "the {width}-year experience band that contains their exact years "
            "of experience, instead of the exact number of years."),
        "rule_statement": (
            "Identity code: {code}. Policy: years of experience are reported "
            "as the frozen {width}-year band containing the exact value "
            "(bin edges are multiples of {width} from the schema anchor), not "
            "as the exact value. Generate the alias."),
        "fewshot_other_branch": (
            "Map each identity code to its {width}-year experience band.\n"
            "{demos}\n"
            "Identity code: {code}. Generate the alias."),
    },
}
#: How many demonstrations the few-shot policy may use, and the frozen
#: selection rule: the first N identities BY SORTED CODE whose own coarse
#: label at the requested depth differs from the target label, so a
#: demonstration can never hand over the answer.
N_DEMOS = 2


#: The shared scorer has no termination event, so candidate_mass is a SUM of
#: normalized label-prefix scores, not a literal probability mass (the prompt
#: panel's SCORING_LIMITATION, restated here so this report is readable on its
#: own).  Every threshold on it -- including the frozen 0.99 interpretability
#: criterion -- is a threshold on a score sum.
MB_SCORING_LIMITATION = {
    "missing_termination_event": True,
    "candidate_mass_is": ("sum of normalized full-sequence label scores "
                          "WITHOUT a termination event; overlapping candidate "
                          "strings can sum above 1"),
    "consequence": ("normalized candidate-score comparisons and gated "
                    "distances remain comparable across methods and to every "
                    "existing artifact, but min_candidate_mass>=0.99 is a "
                    "threshold on a SCORE SUM, not on a probability"),
    "same_scorer_as": ("scripts/e2c_v3_research_validity.py "
                       "full_sequence_label_probs + build_candidate_summary, "
                       "the frozen scorer every other artifact uses"),
}

#: Frozen budget policy: what must be identical, and the one sanctioned
#: exception (the retraining oracle, which is a reference, not a competitor).
MB_BUDGET_POLICY = {
    "identical_across_trainable_methods": [
        "target pairs and their repeat", "source pairs and their repeat",
        "retain pairs and their repeat", "steps", "warmup", "lr",
        "edit seed", "LoRA configuration",
        ("initialization (the frozen baseline route h, loaded fail-closed "
         "before every method)"),
    ],
    "may_differ": ["the objective and which pair set feeds which side of it"],
    "sanctioned_exception": {
        "rows": ["matched_retrain", "loo_retrain"],
        "reason": ("fresh-retrain oracles keep their own frozen protocol "
                   "(3000/200/2e-5, target x5, retain x50, fresh LoRA init, "
                   "seed 17); they are references for the distance metric, "
                   "not budget-matched competitors, and every row records its "
                   "own recipe so the difference is never implicit"),
    },
}


# ====================================================================== #
# MB0: the frozen design manifest (CPU, deterministic, verify-not-rewrite)
# ====================================================================== #
def _prompt(ctx, iid):
    return rd.CODE_TO_ALIAS_PROMPT.format(code=ctx["code_of"][iid])


def data_spec(ctx, entry):
    """The ONE frozen data spec every trainable method consumes.

    ``source_pairs`` are the associations the ascent-family methods climb:
    the transformed member mapped to the SPECIFIC label being replaced.  No
    method sees a pair that is not in this spec.  The budget recorded here is
    the FROZEN design budget (never the smoke budget), so the committed
    manifest is identical whether MB0 runs under --smoke or not; what a row
    actually executed is recorded beside the row.
    """
    targets = sorted(entry["assignments"])
    return {
        "targets": targets,
        "target_pairs": [{"prompt": _prompt(ctx, t),
                          "answer": entry["assignments"][t]["target"]}
                         for t in targets],
        "source_pairs": [{"prompt": _prompt(ctx, t),
                          "answer": entry["assignments"][t]["source"]}
                         for t in targets],
        "retain_pairs": [{"prompt": _prompt(ctx, i),
                          "answer": ctx["baseline_alias_of"][i]}
                         for i in entry["retain_ids"]],
        "repeats": {
            "target": mx.UL_REPEAT * gxm.TARGET_BOOST,
            "source": mx.UL_REPEAT * gxm.TARGET_BOOST,
            "retain": mx.UL_REPEAT * gxm.RETAIN_REPEAT,
        },
        "budget": {"steps": mx.UL_STEPS, "warmup": mx.UL_WARMUP,
                   "lr": mx.UL_LR, "seed": MB_SEED},
    }


def _coarse_label(ctx, op, iid):
    """The label this identity WOULD get under the set's operation."""
    if ctx["kind"] == "taxonomic":
        depth = int(op["target_depth"])
        return ctx["hierarchy_of"][iid][depth]
    prof = ctx["profiles"][iid]
    tup = (prof["profile_id"], prof["field"], prof["exact_value"],
           prof["baseline_kind"], prof["rationale"])
    return gx.target_label(op["operation"], tup, ctx["schema"])


def transformation_targets(entry):
    """Transformed members only; refusal controls are out of scope here."""
    return sorted(t for t, a in entry["assignments"].items()
                  if a.get("operation") != "refusal")


def refusal_controls(entry):
    return sorted(t for t, a in entry["assignments"].items()
                  if a.get("operation") == "refusal")


def _reference_assignment(entry):
    """The assignment whose requested resolution names the policy's depth /
    band width (the first transformed member by sorted id -- frozen rule;
    refusal controls carry no depth and are never the reference)."""
    return entry["assignments"][transformation_targets(entry)[0]]


def _demo_pool(ctx, entry, a, target_labels):
    """Identities usable as few-shot demonstrations, sorted by code.

    A demonstration must teach the RULE without handing over the answer, so
    the pool excludes every transformed member and every identity whose own
    coarse label equals one of the set's target labels -- for taxonomic sets
    that means a different branch, for numeric sets a different bin.

    The pool is resolved AT THE REQUESTED MEMBER'S OWN RESOLUTION ``a``: a
    mixed-depth set asks one member for a level-1 class and another for a
    level-2 class, and a demonstration at the wrong resolution would teach a
    different rule than the one the prompt states.
    """
    pool = []
    for iid in sorted(ctx["identity_ids"], key=lambda i: ctx["code_of"][i]):
        if iid in entry["assignments"]:
            continue
        lab = _coarse_label(ctx, a, iid)
        if lab in target_labels or lab == ctx["baseline_alias_of"][iid]:
            # a demo whose coarse label equals its own specific label teaches
            # nothing about abstraction
            continue
        pool.append({"identity_id": iid, "code": ctx["code_of"][iid],
                     "coarse_label": lab,
                     "own_label": ctx["baseline_alias_of"][iid]})
    return pool


def resolve_policies(ds, ctx, entry):
    """Frozen prompt-only policy text for one set, per transformed member."""
    kind = "taxonomic" if ctx["kind"] == "taxonomic" else "numeric"
    tmpl = POLICY_TEMPLATES[kind]
    targets = transformation_targets(entry)
    target_labels = {entry["assignments"][t]["target"] for t in targets}
    by_target, pool_sizes = {}, {}
    for t in targets:
        a = entry["assignments"][t]
        code = ctx["code_of"][t]
        pool = _demo_pool(ctx, entry, a, target_labels)
        pool_sizes[t] = len(pool)
        demos = pool[:N_DEMOS]
        fields = {"code": code,
                  "demos": "\n".join(f"{d['code']} -> {d['coarse_label']}"
                                     for d in demos)}
        if kind == "taxonomic":
            fields["depth"] = a["target_depth"]
        else:
            fs = ctx["schema"][ctx["profiles"][t]["field"]]
            fields["width"] = fs["narrow" if a["operation"]
                                  == "exact_to_narrow" else "broad"]["width"]
        by_target[t] = {
            "canonical_control": rd.CODE_TO_ALIAS_PROMPT.format(code=code),
            "coarsen_instruction": tmpl["coarsen_instruction"].format(
                **fields),
            "rule_statement": tmpl["rule_statement"].format(**fields),
            "fewshot_other_branch": tmpl["fewshot_other_branch"].format(
                **fields),
            "demonstrations": demos,
            "n_demo_pool": len(pool),
        }
    return {"kind": kind, "roles": list(PROMPT_POLICY_ROLES),
            "n_demo_pool_by_target": pool_sizes, "by_target": by_target,
            "refusal_controls_excluded": refusal_controls(entry),
            "canonical_control_is": ("the baseline_route reference row: same "
                                     "weights, same canonical prompt, so it "
                                     "is not re-run as a policy")}


def _artifact_availability(ds, out_base_gran, sid):
    """What already exists for this set (reuse is validated, not assumed)."""
    cell_dir = out_base_gran / "cells" / sid / f"seed_{MB_SEED}"
    oracle_root = out_base_gran / "oracles"
    return {
        "edited_cell_record": (cell_dir / "cell_results.json").exists(),
        "edited_cell_checkpoint": (cell_dir / "edited_h" / "adapter_final"
                                   / "adapter_model.safetensors").exists(),
        "matched_retrain_soft": (oracle_root / f"matched_retrain_{sid}"
                                 / "oracle_soft.json").exists(),
        "matched_retrain_checkpoint": (
            oracle_root / f"matched_retrain_{sid}" / "adapter_final"
            / "adapter_model.safetensors").exists(),
        "loo_retrain_soft": (oracle_root / f"loo_retrain_{sid}"
                             / "oracle_soft.json").exists(),
        "loo_retrain_checkpoint": (oracle_root / f"loo_retrain_{sid}"
                                   / "adapter_final"
                                   / "adapter_model.safetensors").exists(),
    }


def build_mb_manifest(ds, matrix, ctx):
    """The frozen design manifest: sets, data, methods, policies, metrics."""
    rep = MB_REP_SETS[ds]
    gran_out = gxm.OUT_ROOT / ds
    sets = []
    for sid in rep:
        entry = next(e for e in matrix["sets"] if e["set_id"] == sid)
        spec = data_spec(ctx, entry)
        sets.append({
            "set_id": sid, "mode": entry["mode"],
            "assignments": entry["assignments"],
            "controls": entry.get("controls", {}),
            "control_notes": entry.get("control_notes", {}),
            "retain_ids": entry["retain_ids"],
            "data_spec": spec,
            "prompt_policies": resolve_policies(ds, ctx, entry),
            "available_artifacts": _artifact_availability(ds, gran_out, sid),
            "transformation_targets": sorted(
                t for t, a in entry["assignments"].items()
                if a.get("operation") != "refusal"),
            "refusal_controls": sorted(
                t for t, a in entry["assignments"].items()
                if a.get("operation") == "refusal"),
        })
    methods = {}
    for mid, spec in MB_METHODS.items():
        m = dict(spec)
        m["installs_target"] = bool(
            "target" in (m.get("descent_set") or [])
            or m.get("distribution_set") == "target")
        m["budget_mismatch_ok"] = (m["budget"] == "oracle")
        methods[mid] = m
    refs = {rid: dict(r, installs_target=False, budget_mismatch_ok=(
        r["budget"] == "oracle")) for rid, r in MB_REFERENCES.items()}
    return {
        "kind": "e2c_v3_method_baselines_manifest_v1",
        "dataset": ds,
        "produced_by": "scripts/e2c_v3_method_baselines.py",
        "edit_seed": MB_SEED,
        "n_sets": len(sets),
        "n_method_rows": len(methods),
        "n_reference_rows": len(refs),
        "n_trained_rows": len([m for m in methods.values() if m["trains"]]),
        "n_rows_per_set": len(ROW_ORDER),
        "n_runs_planned": len(sets) * len(ROW_ORDER),
        "inputs": {
            "matrix_sha256": rv.sha256_file(
                gxm.MANIFEST_DIR / ("matrix_salmu.json" if ds == "salmu"
                                    else "matrix_celeba_numeric.json")),
            "granularity_runner_sha256": rv.sha256_file(
                SCRIPT_DIR / "e2c_v3_granularity_matrix.py"),
            "granularity_lib_sha256": rv.sha256_file(
                SCRIPT_DIR / "e2c_v3_granularity.py"),
            "shared_scoring_script_sha256": rv.script_sha256(),
        },
        "rep_set_policy": ("one representative per transformation TYPE (the "
                           "same representatives GX2B/GX2S used); the full "
                           "147-cell matrix is deliberately NOT multiplied "
                           "out before viable methods are identified"),
        "sets": sets,
        "methods": methods,
        "references": refs,
        "budget_policy": MB_BUDGET_POLICY,
        "required_metrics": MB_REQUIRED_METRICS,
        "scoring_limitation": MB_SCORING_LIMITATION,
        "pass_criteria_reference": gx.PASS_CRITERIA,
    }


def validate_mb_manifest(ds, man, ctx):
    """Hard MB0 gate: the design must mean what it claims."""
    issues = []
    methods, sets = man["methods"], man["sets"]
    # --- budget discipline ------------------------------------------------
    budgets = {m["budget"] for m in methods.values()}
    if not budgets <= {"edit", "oracle", "none"}:
        issues.append(f"unknown budget labels {sorted(budgets)}")
    for mid, m in methods.items():
        if m["budget"] != "edit" or not m["trains"]:
            continue
        for side in ("descent_set", "ascent_set", "anchor_set"):
            v = m.get(side)
            names = v if isinstance(v, list) else ([v] if v else [])
            if not set(names) <= {"target", "source", "retain"}:
                issues.append(f"{mid}: {side} names a pair set that is not "
                              f"in the frozen data spec: {names}")
    if methods["cf_relabel"].get("aliased_to") != "sft_target":
        issues.append("cf_relabel must declare its alias to sft_target")
    if methods["kl_ascent_anchor"]["installs_target"]:
        issues.append("kl_ascent_anchor gained a descent term: it is only "
                      "kept as a row because it has none")
    for mid in ("sft_target", "gd_distribution", "ga_retain_descent", "npo",
                "kl_anchored_edit"):
        if not methods[mid]["installs_target"]:
            issues.append(f"{mid} does not install the target label: the "
                          f"comparison would not be about the same estimand")
    # --- data spec --------------------------------------------------------
    for s in sets:
        sid, spec = s["set_id"], s["data_spec"]
        if not s["transformation_targets"]:
            issues.append(f"{sid}: no transformation target (refusal-only "
                          f"sets are out of scope for this comparison)")
        if not spec["retain_pairs"]:
            issues.append(f"{sid}: empty retain set")
        if len(spec["target_pairs"]) != len(s["transformation_targets"]) + \
                len(s["refusal_controls"]):
            issues.append(f"{sid}: target pairs do not cover every assignment")
        for p in spec["source_pairs"]:
            if p["answer"] == gx.DELETED_LABEL:
                issues.append(f"{sid}: a source pair is the refusal label; "
                              f"this comparison is about transformation")
        if spec["budget"]["seed"] != MB_SEED:
            issues.append(f"{sid}: data spec seed != the frozen edit seed")
        # --- prompt-policy leak guards --------------------------------
        pol = s["prompt_policies"]
        labels = {a["target"] for a in s["assignments"].values()} | \
            {a["source"] for a in s["assignments"].values()}
        for t, roles in pol["by_target"].items():
            for role in PROMPT_POLICY_ROLES:
                text = roles[role].lower()
                for lab in labels:
                    if lab.lower() in text:
                        issues.append(
                            f"{sid}/{t}/{role}: the policy prompt CONTAINS "
                            f"the label {lab!r} -- a prompting baseline may "
                            f"state the rule, never the answer")
            demos = roles["demonstrations"]
            if len(demos) < N_DEMOS:
                issues.append(f"{sid}/{t}: only {len(demos)} demonstrations "
                              f"available; the frozen rule needs {N_DEMOS}")
            for d in demos:
                if d["identity_id"] in s["assignments"]:
                    issues.append(f"{sid}/{t}: demonstration {d['code']} is a "
                                  f"transformed member")
                if d["coarse_label"] in labels:
                    issues.append(f"{sid}/{t}: demonstration answer "
                                  f"{d['coarse_label']!r} is one of this "
                                  f"set's labels")
                # a demonstration must be resolved at THIS member's own
                # requested resolution, or it teaches a different rule
                try:
                    own = _coarse_label(ctx, s["assignments"][t],
                                        d["identity_id"])
                except Exception as exc:
                    issues.append(
                        f"{sid}/{t}: demonstration {d['code']} cannot be "
                        f"resolved at this member's requested resolution "
                        f"({exc})")
                    continue
                if own != d["coarse_label"]:
                    issues.append(
                        f"{sid}/{t}: demonstration {d['code']} is labelled "
                        f"{d['coarse_label']!r} but resolves to {own!r} at "
                        f"this member's requested resolution")
        # --- reuse availability ---------------------------------------
        av = s["available_artifacts"]
        if not (av["edited_cell_record"] and av["edited_cell_checkpoint"]):
            issues.append(f"{sid}: no frozen edited cell at seed {MB_SEED} to "
                          f"reuse as sft_target")
        for key in ("matched_retrain_soft", "loo_retrain_soft",
                    "matched_retrain_checkpoint", "loo_retrain_checkpoint"):
            if not av[key]:
                issues.append(f"{sid}: missing {key}; matched-vs-LOO oracle "
                              f"distance cannot be computed")
    # --- metric coverage --------------------------------------------------
    emitted = set(MB_REPORTED_METRICS)
    missing = [m for m in man["required_metrics"] if m not in emitted]
    if missing:
        issues.append(f"the report cannot emit required metrics {missing}")
    return issues


#: The metric keys the report emits per row (checked at MB0, not trusted).
MB_REPORTED_METRICS = [
    "transformation_success", "source_label_suppression", "retention",
    "sibling_preservation", "wrong_branch_rate", "candidate_validity",
    "matched_vs_loo_oracle_distance", "training_time", "trainable_parameters",
    "frozen_cell_criteria",
]


def mb_manifest_path(ds):
    return MB_MANIFEST_DIR / f"mb_manifest_{ds}.json"


def build_or_verify(ds, matrix, ctx):
    """MB0: verify-not-rewrite the committed design manifest."""
    MB_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    man = build_mb_manifest(ds, matrix, ctx)
    issues = validate_mb_manifest(ds, man, ctx)
    if issues:
        for i in issues:
            logger.error("MB0 ISSUE: %s", i)
        raise RuntimeError(f"MB0 validation failed with {len(issues)} "
                           f"issue(s)")
    path = mb_manifest_path(ds)
    if path.exists():
        committed = json.loads(path.read_text(encoding="utf-8"))
        if committed != man:
            raise RuntimeError(
                f"committed {path} differs from the frozen builder; the "
                f"method-baseline design is a committed input and must not "
                f"drift (if a method or policy changed, that is a new design "
                f"and needs a deliberate re-commit)")
        logger.info("MB0: committed manifest verified (not rewritten): %s",
                    path)
    else:
        path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
        logger.info("MB0: design frozen to %s (%d sets x %d rows = %d runs; "
                    "%d methods trained per set, %d rows reused, %d prompt "
                    "policies)", path, man["n_sets"], man["n_rows_per_set"],
                    man["n_runs_planned"], man["n_trained_rows"],
                    len([m for m in man["methods"].values()
                         if not m["trains"] and m["budget"] != "none"]),
                    len(PROMPT_ROW_ROLES))
    validation = {
        "dataset": ds, "edit_seed": MB_SEED, "n_sets": man["n_sets"],
        "n_rows_per_set": man["n_rows_per_set"],
        "n_runs_planned": man["n_runs_planned"],
        "n_trained_rows_per_set": man["n_trained_rows"],
        "methods": sorted(man["methods"]), "references": sorted(
            man["references"]),
        "trained_methods": sorted(m for m, s in man["methods"].items()
                                  if s["trains"]),
        "prompt_policy_roles": list(PROMPT_POLICY_ROLES),
        "sets": [{"set_id": s["set_id"], "mode": s["mode"],
                  "transformation_targets": s["transformation_targets"],
                  "refusal_controls": s["refusal_controls"],
                  "n_retain": len(s["data_spec"]["retain_pairs"]),
                  "repeats": s["data_spec"]["repeats"],
                  "budget": s["data_spec"]["budget"],
                  "demo_pool": s["prompt_policies"]["n_demo_pool_by_target"],
                  "available_artifacts": s["available_artifacts"]}
                 for s in man["sets"]],
        "required_metrics": man["required_metrics"],
        "issues": issues,
        # deterministic bytes: NO timestamp -- committed file
    }
    out_base = MB_OUT_ROOT / ds
    out_base.mkdir(parents=True, exist_ok=True)
    (out_base / "mb0_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    return man


def load_frozen_mb(ds):
    return json.loads(mb_manifest_path(ds).read_text(encoding="utf-8"))


# ====================================================================== #
# Cached-record validation (a readable file is not evidence)
# ====================================================================== #
ROW_KIND = "method_baseline_row_v1"

MB_CACHE_VALIDATION = {
    "required_to_match": [
        "kind", "dataset", "set_id", "seed", "row_id", "recipe",
        "checkpoint_sha256", "provenance.mb_manifest_sha256",
        "provenance.shared_scoring_script_sha256",
    ],
    "recorded_not_required": ["provenance.runner_script_sha256",
                              "provenance.git_commit", "train_sec"],
    "on_mismatch": ("quarantine as <stem>.stale-<sha8(reasons)>.json and "
                    "rerun the row; a stale row is never aggregated"),
}


def _quarantine(path, reasons):
    """Rename a stale record beside itself; the evidence is kept."""
    joined = "|".join(sorted(str(r) for r in reasons)).encode()
    digest = hashlib.sha256(joined).hexdigest()[:8]
    dest = path.with_name(f"{path.stem}.stale-{digest}{path.suffix}")
    path.rename(dest)
    return dest


def validate_cached_row(rec, ds, sid, row_id, recipe, ckpt_sha):
    """Reasons this cached row must NOT be reused (empty = reusable)."""
    reasons = []
    if rec.get("kind") != ROW_KIND:
        reasons.append("kind")
    if rec.get("dataset") != ds:
        reasons.append("dataset")
    if rec.get("set_id") != sid:
        reasons.append("set_id")
    if rec.get("row_id") != row_id:
        reasons.append("row_id")
    if rec.get("seed") != MB_SEED:
        reasons.append("seed")
    if rec.get("recipe") != recipe:
        reasons.append("recipe")
    man_path = mb_manifest_path(ds)
    prov = rec.get("provenance") or {}
    if (not man_path.exists()
            or prov.get("mb_manifest_sha256") != rv.sha256_file(man_path)):
        reasons.append("provenance.mb_manifest_sha256")
    if prov.get("shared_scoring_script_sha256") != rv.script_sha256():
        reasons.append("provenance.shared_scoring_script_sha256")
    if ckpt_sha is not None and rec.get("checkpoint_sha256") != ckpt_sha:
        reasons.append("checkpoint_sha256")
    return reasons


# ====================================================================== #
# Objective implementations that do not exist for a TRANSFORMATION target
# (rv's shipped trainers are deletion-shaped; rv itself is NOT modified --
# its SHA-256 is pinned by every cached artifact in this project)
# ====================================================================== #
def _label_scores(model, processor, prompt_text, labels, device):
    """Full-sequence log-score of every candidate label, WITH gradient.

    The differentiable counterpart of rv.full_sequence_label_probs: one
    right-padded batched forward over (prompt + label) per candidate, then the
    summed log-probability at each label-token position.  Right padding is safe
    under causal attention (padding sits after the scored tokens), and the
    prompt ids come from rv._build_prompt_ids so the scoring matches the frozen
    scorer exactly.
    """
    tok = processor.tokenizer
    prompt_ids = rv._build_prompt_ids(processor, None,
                                      prompt_text=prompt_text).to(device)
    plen = int(prompt_ids.shape[0])
    rows, spans = [], []
    for lab in labels:
        lids = tok.encode(lab, add_special_tokens=False)
        rows.append(torch.cat([prompt_ids,
                               torch.tensor(lids, device=device,
                                            dtype=prompt_ids.dtype)]))
        spans.append(len(lids))
    maxlen = max(int(r.shape[0]) for r in rows)
    pad = tok.pad_token_id
    if pad is None:
        pad = tok.eos_token_id
    ids = torch.full((len(rows), maxlen), int(pad), dtype=torch.long,
                     device=device)
    mask = torch.zeros((len(rows), maxlen), dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        ids[i, :r.shape[0]] = r
        mask[i, :r.shape[0]] = 1
    logits = model(ids, attention_mask=mask, use_cache=False).logits.float()
    lp = torch.log_softmax(logits, dim=-1)
    scores = []
    for i, n in enumerate(spans):
        if n == 0:
            scores.append(torch.zeros((), device=device, dtype=lp.dtype))
            continue
        s = torch.zeros((), device=device, dtype=lp.dtype)
        for j in range(n):
            s = s + lp[i, plen - 1 + j, ids[i, plen + j]]
        scores.append(s)
    return torch.stack(scores)


def train_gd_target(condition, adapter, model, processor, gd_targets, vocab,
                    retain_items, output_dir, device, steps, warmup, lr):
    """GD distribution matching toward the coarser label + retain descent.

    Per step, each transformed member contributes
    ``logsumexp(scores) - score(target)`` -- KL(one-hot(target) || normalized
    full-sequence candidate scores) over the frozen candidate vocabulary --
    and one retain batch contributes supervised CE descent, the same
    ascent/descent pairing rv.train_ga and rv.train_npo use.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    logger.info("[%s] GD-target targets=%d vocab=%d retain=%d steps=%d lr=%g",
                condition, len(gd_targets), len(vocab), len(retain_items),
                steps, lr)
    retain_loader = rv.make_loader(adapter, retain_items)
    retain_iter = iter(retain_loader)
    optimizer, scheduler = rv.make_optimizer_scheduler(params, steps, warmup,
                                                       lr)
    model.train()
    trace, step = [], 0
    t_start = time.time()
    for _ in range(100000):
        for prompt_text, target_label in gd_targets:
            scores = _label_scores(model, processor, prompt_text, vocab,
                                   device)
            idx = vocab.index(target_label)
            gd_loss = torch.logsumexp(scores, dim=0) - scores[idx]
            gd_loss.backward()
            try:
                rbatch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                rbatch = next(retain_iter)
            rbd = rv.move_batch(rbatch, device)
            rloss = rv.forward_loss(model, rbd)
            rloss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            step += 1
            if step % 50 == 0 or step <= 5:
                trace.append({"step": step, "gd_loss": float(gd_loss.item()),
                              "retain_loss": float(rloss.item()),
                              "target_p": float(
                                  torch.softmax(scores, dim=0)[idx].item())})
                logger.info("[%s] step %d/%d gd=%.6f retain=%.6f "
                            "normalized_target_p=%.4f", condition, step,
                            steps, gd_loss.item(), rloss.item(),
                            trace[-1]["target_p"])
            if step >= steps:
                break
        if step >= steps:
            break
    adapter.save_unlearning_adapter(model, output_dir / "adapter_final")
    with open(output_dir / "training_trace.jsonl", "w",
              encoding="utf-8") as f:
        f.writelines(json.dumps(e) + "\n" for e in trace)
    return {"trace": trace, "train_sec": round(time.time() - t_start, 1),
            "steps_executed": step}


def train_kl_anchored_edit(condition, adapter, model, processor,
                           descent_items, anchor_items, output_dir, device,
                           steps, warmup, lr, beta_kl):
    """KL-anchored EDITING: CE descent on target+retain, KL anchor on retain.

    rv.train_kl ascends on the forget set and anchors on the retain set; it has
    no descent term, so it cannot install a new label.  This is the
    transformation reading of "KL-anchored editing": the descent installs the
    coarser label and keeps retention, the anchor bounds drift from the
    pre-edit reference measured at the token-distribution level on the retain
    prompts.  Reference handling follows rv.train_kl exactly (snapshot, swap in
    the reference for its forward only, restore before the optimizer step).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    logger.info("[%s] KL-anchored edit descent=%d anchor=%d steps=%d "
                "beta_kl=%g", condition, len(descent_items),
                len(anchor_items), steps, beta_kl)
    ref_weights = rv.snapshot_lora(model)
    descent_loader = rv.make_loader(adapter, descent_items)
    anchor_loader = rv.make_loader(adapter, anchor_items)
    anchor_iter = iter(anchor_loader)
    optimizer, scheduler = rv.make_optimizer_scheduler(params, steps, warmup,
                                                       lr)
    model.train()
    trace, step = [], 0
    t_start = time.time()
    for _ in range(100000):
        for dbatch in descent_loader:
            dbd = rv.move_batch(dbatch, device)
            ce = rv.forward_loss(model, dbd)
            ce.backward()
            try:
                abatch = next(anchor_iter)
            except StopIteration:
                anchor_iter = iter(anchor_loader)
                abatch = next(anchor_iter)
            abd = rv.move_batch(abatch, device)
            logits_theta = rv.forward_logits(model, abd)
            current = rv.snapshot_lora(model)
            rv.restore_lora(model, ref_weights)
            with torch.no_grad():
                logits_ref = rv.forward_logits(model, abd).detach()
            rv.restore_lora(model, current)
            lp_theta = torch.log_softmax(logits_theta.float(), dim=-1)
            lp_ref = torch.log_softmax(logits_ref.float(), dim=-1)
            mask = (abd["labels"] != -100).unsqueeze(-1)
            kl = torch.nn.functional.kl_div(lp_ref, lp_theta,
                                            reduction="none", log_target=True)
            kl = (kl * mask).sum() / mask.sum().clamp(min=1)
            (beta_kl * kl).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            step += 1
            if step % 50 == 0 or step <= 5:
                trace.append({"step": step, "ce_loss": float(ce.item()),
                              "kl": float(kl.item())})
                logger.info("[%s] step %d/%d ce=%.6f kl=%.6f", condition,
                            step, steps, ce.item(), kl.item())
            if step >= steps:
                break
        if step >= steps:
            break
    adapter.save_unlearning_adapter(model, output_dir / "adapter_final")
    with open(output_dir / "training_trace.jsonl", "w",
              encoding="utf-8") as f:
        f.writelines(json.dumps(e) + "\n" for e in trace)
    return {"trace": trace, "train_sec": round(time.time() - t_start, 1),
            "steps_executed": step}


def count_parameters(model):
    """Trainable vs total parameters, plus the LoRA configuration."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    # this project loads its adapter under its own name ("unlearning"), not
    # "default", so the configuration is found by name rather than assumed
    configs = getattr(model, "peft_config", None) or {}
    name = ("default" if "default" in configs
            else (min(configs) if configs else None))
    cfg = configs.get(name) if name is not None else None
    if cfg is None:
        lora = {"unavailable": "the model exposes no peft_config"}
    else:
        lora = {"adapter_name": name,
                "peft_type": str(getattr(cfg, "peft_type", None)),
                "r": getattr(cfg, "r", None),
                "lora_alpha": getattr(cfg, "lora_alpha", None),
                "lora_dropout": getattr(cfg, "lora_dropout", None),
                "target_modules": sorted(getattr(cfg, "target_modules", [])
                                         or [])}
    return {"trainable_parameters": trainable, "total_parameters": total,
            "trainable_fraction": (trainable / total) if total else None,
            "lora": lora}


# ====================================================================== #
# Evaluation: the frozen engine, with an optional prompt override
# ====================================================================== #
def mb_hard_eval(session, ctx, entry, args, prompt_fn=None):
    """Strict hard generation over every identity.

    Byte-for-byte the frozen cell engine (gxm._hard_eval) when ``prompt_fn``
    is None; the override exists only for the prompt-only policy rows, and a
    unit test pins the two paths to the same output.
    """
    backend = session.backend()
    session.model.eval()
    preds = []
    with torch.no_grad():
        for iid in ctx["identity_ids"]:
            prompt = (prompt_fn(iid) if prompt_fn is not None
                      else rd.CODE_TO_ALIAS_PROMPT.format(
                          code=ctx["code_of"][iid]))
            gen = backend.generate(None, prompt,
                                   max_new_tokens=args.max_gen_tokens)
            raw = gen.text.strip()
            labels = rv.recognized_labels_in(raw, ctx["vocab"])
            parsed = rv.parse_recognized_label(raw, ctx["vocab"])
            exp = gxm.expected_label(ctx, entry, iid)
            preds.append({
                "identity_id": iid, "raw": raw, "parsed_label": parsed,
                "recognized_labels": labels,
                "multi_label_ambiguous": len(labels) > 1,
                "group": ("target" if iid in entry["assignments"]
                          else gxm.control_group(ctx, entry, iid)),
                "expected_post_edit": exp,
                "correct_post_edit": parsed == exp,
                "source_leaked": (iid in entry["assignments"]
                                  and entry["assignments"][iid]["source"]
                                  in labels),
                **gxm._classify(ctx, entry, iid, parsed),
            })
    return preds


def mb_soft_eval(session, ctx, args, prompt_fn=None, only=None):
    """Full-sequence candidate scores per identity (frozen scorer).

    ``prompt_fn`` overrides the canonical prompt for the prompt-only rows via
    the keyword-only ``prompt_text`` of rv.full_sequence_label_probs; the
    default path is the frozen one every artifact on disk was produced with.
    ``only`` restricts the pass to a subset of identities (the prompt-only
    rows score the transformed members and inherit the baseline row's
    distributions for everybody else, since no weight changed).
    """
    out = {}
    ids = ctx["identity_ids"] if only is None else list(only)
    for iid in ids:
        prompt_text = prompt_fn(iid) if prompt_fn is not None else None
        probs = rv.full_sequence_label_probs(
            session.adapter, session.model, session.processor,
            ctx["code_of"][iid], ctx["vocab"], args.device,
            prompt_text=prompt_text)
        prob_by_label = {l: probs.get(l, {}).get("prob", 0.0)
                         for l in ctx["vocab"]}
        summary = rv.build_candidate_summary(prob_by_label, ctx["vocab"],
                                             gx.DELETED_LABEL)
        summary["log_probs"] = {l: probs.get(l, {}).get("log_prob", -1e9)
                                for l in ctx["vocab"]}
        out[iid] = summary
    return out


def method_metrics(hard, soft, entry, ctx):
    """The requested metric set, on top of the frozen per-cell criteria."""
    criteria = gxm._pass_criteria(hard, soft, entry, ctx)
    by_id = {p["identity_id"]: p for p in hard}
    targets = [t for t, a in entry["assignments"].items()
               if a.get("operation") != "refusal"]
    refusals = [t for t, a in entry["assignments"].items()
                if a.get("operation") == "refusal"]
    retained = [i for i in ctx["identity_ids"] if i not in entry["assignments"]]
    tsucc, suppress = {}, {}
    for t in sorted(entry["assignments"]):
        a = entry["assignments"][t]
        p = by_id[t]
        tsucc[t] = {"expected": a["target"], "parsed": p["parsed_label"],
                    "strict_ok": bool(p["correct_post_edit"]),
                    "operation": a.get("operation"),
                    "classification": p.get("classification"),
                    "p_target": soft[t]["probs"].get(a["target"], 0.0)}
        suppress[t] = {"source": a["source"],
                       "p_source": soft[t]["probs"].get(a["source"], 0.0),
                       "hard_leak": bool(p["source_leaked"]),
                       "recognized_labels": p["recognized_labels"]}
    trans_ok = [tsucc[t]["strict_ok"] for t in targets]
    drifted = [i for i in retained if not by_id[i]["correct_post_edit"]]
    worst = min(retained, key=lambda i: soft[i]["probs"].get(
        gxm.expected_label(ctx, entry, i), 0.0)) if retained else None
    cls_counts = {}
    for p in hard:
        key = p.get("classification") or "unclassified"
        cls_counts[key] = cls_counts.get(key, 0) + 1
    masses = [v["candidate_mass"] for v in soft.values()]
    return {
        "transformation_success": {
            "scope": ("transformation targets only; refusal controls are "
                      "reported beside them and never inside the headline"),
            "per_target": {t: tsucc[t] for t in targets},
            "per_refusal_control": {t: tsucc[t] for t in refusals},
            "all_targets_strict": (all(trans_ok) if trans_ok else None),
            "strict_rate": (sum(trans_ok) / len(trans_ok)) if trans_ok
            else None,
        },
        "source_label_suppression": {
            "per_target": {t: suppress[t] for t in targets},
            "max_p_source": max([suppress[t]["p_source"] for t in targets],
                                default=None),
            "any_hard_leak": any(suppress[t]["hard_leak"] for t in targets),
        },
        "retention": {
            "n_retained": len(retained),
            "strict_accuracy": criteria["retain_acc"],
            "drifted_ids": drifted,
            "worst_retained_identity": worst,
            "worst_retained_p_expected": (
                soft[worst]["probs"].get(
                    gxm.expected_label(ctx, entry, worst), 0.0)
                if worst else None),
        },
        "sibling_preservation": {
            "sibling_ids": criteria["sibling_ids"],
            "coverage": {"n_sibling_controls": len(criteria["sibling_ids"]),
                         "available": bool(criteria["sibling_ids"]),
                         "note": ("null, never a vacuous 1.0, when the set has "
                                  "no sibling control (unique-branch "
                                  "identities)")},
            "strict_accuracy": criteria["sibling_acc"],
        },
        "wrong_branch_rate": {
            "rate": (len(criteria["wrong_branch_ids"]) / len(hard))
            if hard else None,
            "ids": criteria["wrong_branch_ids"],
            "classification_counts": cls_counts,
        },
        "candidate_validity": {
            "multi_label_ids": criteria["multi_label_ids"],
            "unparseable_ids": criteria["unparseable_ids"],
            "multi_label_rate": len(criteria["multi_label_ids"]) / len(hard)
            if hard else None,
            "unparseable_rate": len(criteria["unparseable_ids"]) / len(hard)
            if hard else None,
            "min_candidate_score_sum": min(masses) if masses else None,
            "mean_candidate_score_sum": (sum(masses) / len(masses))
            if masses else None,
            "clears_frozen_interpretability_threshold": (
                (min(masses) >= gx.PASS_CRITERIA["min_candidate_mass"])
                if masses else None),
            "semantics": ("score SUM without a termination event, not a "
                          "literal probability mass; see scoring_limitation"),
        },
        "frozen_cell_criteria": criteria,
    }


def oracle_distance_summary(soft, entry, ctx, oracle_root, sid):
    """Matched-vs-LOO retraining distance per transformation target."""
    fam = gxm.oracle_family_distances(soft, entry, ctx, oracle_root, sid)
    targets = sorted(t for t, a in entry["assignments"].items()
                     if a.get("operation") != "refusal")

    def _l2(iid, family):
        d = fam[iid][family]["distance"]
        return d["l2"] if d else None

    per_target = {}
    for t in targets:
        per_target[t] = {
            "l2_matched_retrain": _l2(t, "matched_retrain"),
            "l2_loo_retrain": _l2(t, "loo_retrain"),
            "l2_matched_finetune": _l2(t, "matched_finetune"),
            "l2_loo_finetune": _l2(t, "loo_finetune"),
            "delta_retrain_l2": fam[t]["delta_retrain_l2"],
            "delta_ft_l2": fam[t]["delta_ft_l2"],
            "delta_retrain_reliable": fam[t]["delta_retrain_reliable"],
            "reason_matched_retrain": fam[t]["matched_retrain"]["reason"],
            "reason_loo_retrain": fam[t]["loo_retrain"]["reason"],
        }
    deltas = [v["delta_retrain_l2"] for v in per_target.values()
              if v["delta_retrain_l2"] is not None]
    return {
        "metric": ("gated L2 over the frozen candidate vocabulary; positive "
                   "Delta_retrain = closer to the fresh MATCHED retraining "
                   "reference than to the fresh leave-one-out reference"),
        "gate": {"min_candidate_score_sum": gxm.MIN_CANDIDATE_MASS,
                 "below_gate_is": "None ('not established'), never 0.0"},
        "scope": ("transformation targets only; refusal controls never "
                  "contribute to this distance"),
        "per_target": per_target,
        "n_targets": len(targets),
        "n_targets_reliable": len(deltas),
        "mean_delta_retrain_l2": (sum(deltas) / len(deltas)) if deltas
        else None,
        "worst_delta_retrain_l2": min(deltas) if deltas else None,
        "closer_to_matched_than_loo": (all(d > 0 for d in deltas)
                                       if deltas else None),
    }


# ====================================================================== #
# MB1: rows -- one (set, method) row per frozen design entry
# ====================================================================== #
#: prompt-only rows: one per frozen instruction policy.  The canonical-prompt
#: control is NOT re-run -- it is the baseline_route reference row (identical
#: weights, identical prompt), and the report says so.
PROMPT_ROW_ROLES = ("coarsen_instruction", "rule_statement",
                    "fewshot_other_branch")


def gran_out_base(ds):
    return gxm.OUT_ROOT / ds


def row_dir(out_base, sid, row_id):
    return out_base / "rows" / sid / row_id


def row_recipe(man, sid, row_id, args):
    """The recipe a cached row must match to be reused.

    Objective fields come from the frozen design; the budget fields are what
    THIS invocation executes, so a smoke row (20 steps) can never be reused as
    a full-budget row (500 steps) and vice versa.
    """
    base = row_id.split("__")[0]
    m = man["methods"].get(base) or man["references"].get(base) or {}
    s = next(x for x in man["sets"] if x["set_id"] == sid)
    return {"objective": m.get("objective"), "trainer": m.get("trainer"),
            "budget": m.get("budget"), "ascent_set": m.get("ascent_set"),
            "descent_set": m.get("descent_set"),
            "anchor_set": m.get("anchor_set"),
            "distribution_set": m.get("distribution_set"),
            "beta": m.get("beta"), "beta_kl": m.get("beta_kl"),
            "prompt_role": (row_id.split("__", 1)[1]
                            if "__" in row_id else None),
            "repeats": executed_repeats(args),
            "steps": args.ul_steps, "warmup": args.ul_warmup,
            "lr": args.ul_lr, "seed": MB_SEED,
            "design_repeats": s["data_spec"]["repeats"],
            "design_budget": s["data_spec"]["budget"]}


def executed_repeats(args):
    return {"target": args.ul_repeat * gxm.TARGET_BOOST,
            "source": args.ul_repeat * gxm.TARGET_BOOST,
            "retain": args.ul_repeat * gxm.RETAIN_REPEAT}


def check_budget_matches_design(args, spec, smoke):
    """Outside smoke, the executed budget MUST be the frozen design budget."""
    if smoke:
        return []
    diffs = []
    if executed_repeats(args) != spec["repeats"]:
        diffs.append(f"repeats {executed_repeats(args)} != design "
                     f"{spec['repeats']}")
    for key, got in (("steps", args.ul_steps), ("warmup", args.ul_warmup),
                     ("lr", args.ul_lr)):
        if got != spec["budget"][key]:
            diffs.append(f"{key} {got} != design {spec['budget'][key]}")
    if diffs:
        raise RuntimeError(
            "MB1: the executed budget differs from the frozen design "
            f"({'; '.join(diffs)}).  Methods are only comparable at the same "
            "budget; use --smoke for a cheap pass, or restore the defaults.")
    return diffs


def plan_rows(ds, sid, s, man, gran_out, out_base):
    """Ordered row plans.  Order matters: alias and prompt rows read the
    rows they depend on, so those come first."""
    cell_dir = gran_out / "cells" / sid / f"seed_{MB_SEED}"
    oracle_root = gran_out / "oracles"
    baseline = gxm._baseline_ckpt(ds, gran_out)
    plans = [{
        "row_id": "baseline_route", "row_type": "reference",
        "spec": man["references"]["baseline_route"],
        "ckpt": baseline, "weights_source": "baseline_route",
        "train": None, "prompt_role": None,
    }, {
        "row_id": "sft_target", "row_type": "method",
        "spec": man["methods"]["sft_target"],
        "ckpt": cell_dir / "edited_h" / "adapter_final"
        / "adapter_model.safetensors",
        "weights_source": "reused_granularity_cell",
        "train": None, "prompt_role": None,
        "frozen_cell_record": cell_dir / "cell_results.json",
    }, {
        "row_id": "cf_relabel", "row_type": "alias",
        "spec": man["methods"]["cf_relabel"], "ckpt": None,
        "weights_source": "alias_of_sft_target", "train": None,
        "prompt_role": None, "aliased_to": "sft_target",
    }]
    for mid in ("gd_distribution", "ga_retain_descent", "npo",
                "kl_anchored_edit", "kl_ascent_anchor"):
        plans.append({
            "row_id": mid, "row_type": "method", "spec": man["methods"][mid],
            "ckpt": row_dir(out_base, sid, mid) / "adapter_final"
            / "adapter_model.safetensors",
            "weights_source": "trained_here", "train": mid,
            "prompt_role": None,
        })
    plans.append({
        "row_id": "matched_retrain", "row_type": "method",
        "spec": man["methods"]["matched_retrain"],
        "ckpt": oracle_root / f"matched_retrain_{sid}" / "adapter_final"
        / "adapter_model.safetensors",
        "weights_source": "reused_matched_retrain_oracle", "train": None,
        "prompt_role": None,
        "oracle_record": oracle_root / f"matched_retrain_{sid}"
        / "oracle_results.json",
    })
    plans.append({
        "row_id": "loo_retrain", "row_type": "reference",
        "spec": man["references"]["loo_retrain"],
        "ckpt": oracle_root / f"loo_retrain_{sid}" / "adapter_final"
        / "adapter_model.safetensors",
        "weights_source": "reused_loo_retrain_oracle", "train": None,
        "prompt_role": None,
        "oracle_record": oracle_root / f"loo_retrain_{sid}"
        / "oracle_results.json",
    })
    for role in PROMPT_ROW_ROLES:
        plans.append({
            "row_id": f"prompt_only__{role}", "row_type": "prompt",
            "spec": man["methods"]["prompt_only"], "ckpt": baseline,
            "weights_source": "baseline_route_unchanged", "train": None,
            "prompt_role": role,
        })
    return plans


def _items(session, pairs, repeat):
    return rv.build_supervised_items(session.adapter, session.processor,
                                     pairs, repeat=repeat)


def _train_row(session, mid, spec, ctx, out_dir, args, t_iids):
    """Train one method on the frozen data spec; returns a timing block.

    The pairs are the design's; the REPEATS and budget are what this
    invocation executes (identical to the design outside --smoke, enforced by
    check_budget_matches_design), and both are recorded.
    """
    reps = executed_repeats(args)
    budget = {"steps": args.ul_steps, "warmup": args.ul_warmup,
              "lr": args.ul_lr, "seed": MB_SEED}
    target_items = _items(session, spec["target_pairs"], reps["target"])
    source_items = _items(session, spec["source_pairs"], reps["source"])
    retain_items = _items(session, spec["retain_pairs"], reps["retain"])
    common = {"steps": budget["steps"], "warmup": budget["warmup"],
              "lr": budget["lr"]}
    tag = f"mb_{mid}_{out_dir.parent.name}"
    t0 = time.time()
    if mid == "gd_distribution":
        gd_targets = [(p["prompt"], p["answer"])
                      for p in spec["target_pairs"]]
        info = train_gd_target(tag, session.adapter, session.model,
                               session.processor, gd_targets, ctx["vocab"],
                               retain_items, out_dir, args.device,
                               **common)
    elif mid == "ga_retain_descent":
        rv.train_ga(tag, session.adapter, session.model, session.processor,
                    source_items, retain_items + target_items, out_dir,
                    args.device, **common)
        info = {}
    elif mid == "npo":
        rv.train_npo(tag, session.adapter, session.model, session.processor,
                     source_items, retain_items + target_items, out_dir,
                     args.device, beta=MB_METHODS["npo"]["beta"], **common)
        info = {}
    elif mid == "kl_anchored_edit":
        info = train_kl_anchored_edit(
            tag, session.adapter, session.model, session.processor,
            target_items + retain_items, retain_items, out_dir, args.device,
            beta_kl=MB_METHODS["kl_anchored_edit"]["beta_kl"], **common)
    elif mid == "kl_ascent_anchor":
        # rv.train_kl's second item set is the ANCHOR set, not a descent set:
        # as shipped for deletion it is the retain set, and it stays that way
        # here so the only difference from the deletion recipe is what the
        # ascent side is pointed at.
        rv.train_kl(tag, session.adapter, session.model, session.processor,
                    source_items, retain_items, out_dir,
                    args.device,
                    beta_kl=MB_METHODS["kl_ascent_anchor"]["beta_kl"],
                    **common)
        info = {}
    else:
        raise RuntimeError(f"MB1: no trainer for method {mid!r}")
    return {
        "trained": True, "method": mid,
        "train_sec": round(info.get("train_sec", time.time() - t0), 1),
        "steps_executed": info.get("steps_executed"),
        "steps_requested": budget["steps"], "warmup": budget["warmup"],
        "lr": budget["lr"], "seed": budget["seed"],
        "design_budget": spec["budget"], "design_repeats": spec["repeats"],
        "items": {"target_pairs": len(spec["target_pairs"]),
                  "source_pairs": len(spec["source_pairs"]),
                  "retain_pairs": len(spec["retain_pairs"]),
                  "target_item_count": len(target_items),
                  "source_item_count": len(source_items),
                  "retain_item_count": len(retain_items),
                  "repeats": reps},
        "n_transformed_members": len(t_iids),
    }


def _policy_prompt_for(ctx, s, entry, role, iid):
    """The frozen policy text for ANY identity under ``role``.

    Transformed members get their own resolved prompt (their own requested
    depth / band width).  Everybody else gets the same instruction with the
    set's requested depth / width taken from the first transformed member --
    that is the over-application probe (mode b), and it is why the probe is
    reported separately from the primary mode-(a) row.
    """
    by_target = s["prompt_policies"]["by_target"]
    if iid in by_target:
        return by_target[iid][role]
    kind = s["prompt_policies"]["kind"]
    tmpl = POLICY_TEMPLATES[kind][role]
    a = _reference_assignment(entry)
    first = transformation_targets(entry)[0]
    demos = by_target[first]["demonstrations"]
    fields = {"code": ctx["code_of"][iid],
              "demos": "\n".join(f"{d['code']} -> {d['coarse_label']}"
                                 for d in demos)}
    if kind == "taxonomic":
        fields["depth"] = a["target_depth"]
    else:
        fs = ctx["schema"][ctx["profiles"][first]["field"]]
        fields["width"] = fs["narrow" if a["operation"] == "exact_to_narrow"
                             else "broad"]["width"]
    return tmpl.format(**fields)


def _mode_a_prompt_fn(ctx, s, role):
    """Primary prompt-only mode: the policy where it is aimed, the canonical
    prompt everywhere else (weights are untouched, so the other members are
    the baseline route's own behavior)."""
    by_target = s["prompt_policies"]["by_target"]

    def fn(iid):
        if iid in by_target:
            return by_target[iid][role]
        return rd.CODE_TO_ALIAS_PROMPT.format(code=ctx["code_of"][iid])
    return fn


def run_rows(args, ds, ctx, man, matrix, out_base):
    """MB1: train/evaluate every row of the frozen design (cached per row)."""
    logger.info("=" * 60)
    logger.info("MB1: METHOD ROWS (%s) -- %d sets x %d rows", ds,
                man["n_sets"], len(ROW_ORDER))
    logger.info("=" * 60)
    gran_out = gran_out_base(ds)
    oracle_root = gran_out / "oracles"
    only = set(args.only_sets) if args.only_sets else None
    only_rows = set(args.only_rows) if args.only_rows else None
    stale, session, params = [], None, None
    try:
        for s in man["sets"]:
            sid = s["set_id"]
            if only and sid not in only:
                continue
            entry = next(e for e in matrix["sets"] if e["set_id"] == sid)
            spec = s["data_spec"]
            check_budget_matches_design(args, spec, args.smoke)
            t_iids = sorted(entry["assignments"])
            redo = set()
            for plan in plan_rows(ds, sid, s, man, gran_out, out_base):
                row_id = plan["row_id"]
                if only_rows and row_id not in only_rows:
                    continue
                rdir = row_dir(out_base, sid, row_id)
                rdir.mkdir(parents=True, exist_ok=True)
                rpath = rdir / "mb_row.json"
                recipe = row_recipe(man, sid, row_id, args)
                ckpt = plan["ckpt"]
                ckpt_sha = (rv.sha256_file(ckpt)
                            if ckpt is not None and Path(ckpt).exists()
                            else None)
                if rpath.exists():
                    rec = json.loads(rpath.read_text(encoding="utf-8"))
                    reasons = validate_cached_row(rec, ds, sid, row_id,
                                                  recipe, ckpt_sha)
                    if not reasons:
                        logger.info("[%s/%s] cached + validated, skipping",
                                    sid, row_id)
                        continue
                    dest = _quarantine(rpath, reasons)
                    redo.add(row_id)
                    stale.append({"record": "mb_row", "set_id": sid,
                                  "row_id": row_id, "reasons": reasons,
                                  "quarantined_as": str(dest)})
                    logger.warning("[%s/%s] cached row STALE (%s); "
                                   "quarantined as %s", sid, row_id,
                                   ", ".join(reasons), dest.name)
                logger.info("-" * 56)
                logger.info("[%s/%s] %s (%s)", sid, row_id,
                            plan["spec"].get("label"), plan["weights_source"])
                # ---- alias row: no weights, no evaluation ---------------
                if plan["row_type"] == "alias":
                    src = (row_dir(out_base, sid, plan["aliased_to"])
                           / "mb_row.json")
                    if not src.exists():
                        raise RuntimeError(
                            f"MB1: {row_id} aliases {plan['aliased_to']} but "
                            f"that row has no record at {src}")
                    src_rec = json.loads(src.read_text(encoding="utf-8"))
                    rec = _alias_record(ds, sid, row_id, plan, recipe,
                                        src_rec)
                    rpath.write_text(json.dumps(rec, indent=2) + "\n",
                                     encoding="utf-8")
                    continue
                if session is None:
                    session = mx.ModelSession(args, f"e2c_mb_{ds}")
                    params = count_parameters(session.model)
                    logger.info("MB1: trainable parameters %s of %s (%s)",
                                f"{params['trainable_parameters']:,}",
                                f"{params['total_parameters']:,}",
                                params["lora"])
                rec = _run_one_row(args, ds, ctx, entry, s, sid, plan,
                                   recipe, spec, t_iids, session, params,
                                   rdir, oracle_root, out_base,
                                   force_train=row_id in redo)
                rpath.write_text(json.dumps(rec, indent=2) + "\n",
                                 encoding="utf-8")
                m = rec["metrics"]
                logger.info(
                    "[%s/%s] transformation=%s max_p_source=%s retain=%s "
                    "sib=%s wrong_branch=%s min_score_sum=%s "
                    "Delta_retrain=%s",
                    sid, row_id,
                    m["transformation_success"]["strict_rate"],
                    _fmt(m["source_label_suppression"]["max_p_source"]),
                    _fmt(m["retention"]["strict_accuracy"]),
                    _fmt(m["sibling_preservation"]["strict_accuracy"]),
                    _fmt(m["wrong_branch_rate"]["rate"]),
                    _fmt(m["candidate_validity"]["min_candidate_score_sum"]),
                    _fmt(rec["oracle_distances"]["worst_delta_retrain_l2"]))
    finally:
        if session is not None:
            session.release()
    return stale


def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


def _over_application_probe(session, ctx, entry, s, role, args,
                            baseline_hard=None):
    """Mode (b): the same instruction applied to EVERY member, hard only.

    Descriptive: it shows what a global prompting policy does to the members
    it was not aimed at.  It never feeds the primary comparison, whose
    retention numbers come from mode (a).

    ``baseline_hard`` (identity -> label parsed under the canonical prompt at
    the SAME untouched weights) turns the raw counts into the only number that
    really measures over-application: how many members CHANGED.
    """
    op = _reference_assignment(entry)

    def probe_fn(iid):
        return _policy_prompt_for(ctx, s, entry, role, iid)

    hard = mb_hard_eval(session, ctx, entry, args, prompt_fn=probe_fn)
    rows = []
    for p in hard:
        iid = p["identity_id"]
        before = (baseline_hard or {}).get(iid)
        coarse = _coarse_label(ctx, op, iid)
        rows.append({
            "identity_id": iid,
            "is_transformed_member": iid in entry["assignments"],
            "parsed_label": p["parsed_label"],
            "own_label": ctx["baseline_alias_of"][iid],
            "canonical_prompt_label": before,
            "changed_vs_canonical_prompt": (None if before is None
                                            else p["parsed_label"] != before),
            "coarse_label_at_requested_resolution": coarse,
            # a member can FOLLOW an over-applied rule and still be
            # unparseable: the requested resolution has no frozen candidate
            # label for every value, and strict parsing only recognizes the
            # frozen vocabulary
            "coarse_label_in_vocabulary": coarse in ctx["vocab"],
            "emits_own_label": p["parsed_label"]
            == ctx["baseline_alias_of"][iid],
            "emits_coarse_label": p["parsed_label"] == coarse,
            "classification": p.get("classification"),
        })
    non = [r for r in rows if not r["is_transformed_member"]]
    return {
        "mode": ("over-application probe: the policy applied to every member, "
                 "hard generation only; descriptive, never part of the "
                 "primary comparison"),
        "count_semantics": (
            "emits_own_label and emits_coarse_label are NOT exclusive: a "
            "member whose baseline label already sits at the requested "
            "resolution satisfies both without changing at all, so "
            "over-application is read from changed_vs_canonical_prompt, never "
            "from a raw count"),
        "compared_against": ("the baseline_route row's strict parse of the "
                             "same identity under the canonical prompt, at "
                             "identical weights"
                             if baseline_hard else
                             "nothing: the baseline_route row was not "
                             "available, so no change count is reported"),
        "n_members": len(rows), "n_non_target_members": len(non),
        "non_target_emitting_own_label":
            sum(r["emits_own_label"] for r in non),
        "non_target_emitting_their_own_coarse_label":
            sum(r["emits_coarse_label"] for r in non),
        "non_target_other_or_unparseable":
            sum(1 for r in non
                if not r["emits_own_label"] and not r["emits_coarse_label"]),
        "non_target_coarse_label_outside_vocabulary":
            sum(1 for r in non if not r["coarse_label_in_vocabulary"]),
        "non_target_changed_vs_canonical_prompt": (
            sum(1 for r in non if r["changed_vs_canonical_prompt"])
            if baseline_hard else None),
        "rows": rows,
    }


def _alias_record(ds, sid, row_id, plan, recipe, src):
    """The CF row: the sft_target measurement, labelled as an alias."""
    return {
        "kind": ROW_KIND, "dataset": ds, "set_id": sid, "row_id": row_id,
        "row_type": "alias", "seed": MB_SEED, "recipe": recipe,
        "label": plan["spec"]["label"],
        "objective": plan["spec"]["objective"],
        "note": plan["spec"]["note"],
        "aliased_to": plan["aliased_to"],
        "alias_reason": (
            "for a TRANSFORMATION target the counterfactual label IS the "
            "requested coarser label, so counterfactual relabeling and direct "
            "supervised fine-tuning are the same objective on the same data "
            "with the same budget; they differ only in the deletion setting, "
            "where the counterfactual is the refusal label.  This row is the "
            "sft_target measurement and is never counted as independent "
            "evidence or as a second trained method"),
        "weights": dict(src["weights"]),
        "checkpoint_sha256": src["checkpoint_sha256"],
        "training": dict(src["training"], trained=False,
                         train_sec=0.0,
                         note="no training: aliased row"),
        "parameters": src["parameters"],
        "metrics": src["metrics"],
        "oracle_distances": src["oracle_distances"],
        "provenance": dict(src["provenance"], aliased_row=True),
    }


def _run_one_row(args, ds, ctx, entry, s, sid, plan, recipe, spec,
                 t_iids, session, params, rdir, oracle_root, out_base,
                 force_train=False):
    """Train (if this row trains), evaluate, and assemble the row record.

    ``force_train`` is set when this row's cached record was just quarantined:
    an adapter left behind by a run that was stale (a different budget, a
    different design) is NOT reused, because the new record would then claim a
    recipe those weights were never trained with.
    """
    row_id, role = plan["row_id"], plan["prompt_role"]
    ckpt = Path(plan["ckpt"]) if plan["ckpt"] is not None else None
    baseline = gxm._baseline_ckpt(ds, gran_out_base(ds))
    training = {"trained": False, "train_sec": 0.0,
                "steps_requested": spec["budget"]["steps"],
                "warmup": spec["budget"]["warmup"],
                "lr": spec["budget"]["lr"], "seed": MB_SEED,
                "items": {"target_pairs": len(spec["target_pairs"]),
                          "source_pairs": len(spec["source_pairs"]),
                          "retain_pairs": len(spec["retain_pairs"]),
                          "repeats": spec["repeats"]},
                "n_transformed_members": len(t_iids)}
    if plan["train"] is not None:
        if ckpt.exists() and not args.retrain and not force_train:
            logger.info("[%s/%s] adapter present, reusing weights", sid,
                        row_id)
            training["reused_existing_adapter"] = True
            training["budget_of_those_weights"] = (
                "unknown: no validated record described these weights, so the "
                "budget below is what THIS row would have executed, not what "
                "produced them")
        else:
            if ckpt.exists() and force_train:
                logger.warning("[%s/%s] the cached record was stale, so the "
                               "adapter on disk is NOT reused", sid, row_id)
            session.reset_to(baseline)
            mx.seed_everything(MB_SEED)
            training = _train_row(session, plan["train"], spec, ctx, rdir,
                                  args, t_iids)
    # fail-closed: the weights evaluated are exactly the bytes on disk
    session.reset_to(ckpt if ckpt is not None else baseline)
    mx.seed_everything(MB_SEED)
    prompt_policy, probe = None, None
    if role is not None:
        fn = _mode_a_prompt_fn(ctx, s, role)
        hard = mb_hard_eval(session, ctx, entry, args, prompt_fn=fn)
        base_row = (row_dir(out_base, sid, "baseline_route") / "mb_row.json")
        if not base_row.exists():
            raise RuntimeError(
                f"MB1: {row_id} needs the baseline_route row to inherit the "
                f"untouched members' distributions, but {base_row} is "
                f"missing")
        base_rec = json.loads(base_row.read_text(encoding="utf-8"))
        base_probs = base_rec["soft_probs_full"]
        baseline_hard = {p["identity_id"]: p["parsed_label"]
                         for p in base_rec["hard_preds"]}
        policy_soft = mb_soft_eval(session, ctx, args, prompt_fn=fn,
                                   only=t_iids)
        probs = {i: base_probs[i] for i in ctx["identity_ids"]}
        for t in t_iids:
            probs[t] = policy_soft[t]["probs"]
        soft = gxm.summaries_from_probs(probs, ctx)
        prompt_policy = {
            "role": role, "mode": ("(a) primary: the policy is applied to the "
                                   "transformed members only; every other "
                                   "member keeps the canonical prompt"),
            "texts": {t: s["prompt_policies"]["by_target"][t][role]
                      for t in t_iids
                      if t in s["prompt_policies"]["by_target"]},
            "demonstrations": {
                t: s["prompt_policies"]["by_target"][t]["demonstrations"]
                for t in t_iids
                if t in s["prompt_policies"]["by_target"]},
            "non_target_distributions_from": (
                "the baseline_route row: no weight changed, so the untouched "
                "members' candidate scores are that row's, not a second "
                "measurement"),
            "leak_guard": ("MB0 verified that no policy text contains this "
                           "set's target or source labels"),
        }
        probe = _over_application_probe(session, ctx, entry, s, role, args,
                                        baseline_hard=baseline_hard)
    else:
        hard = mb_hard_eval(session, ctx, entry, args)
        soft = mb_soft_eval(session, ctx, args)
    metrics = method_metrics(hard, soft, entry, ctx)
    dist = oracle_distance_summary(soft, entry, ctx, oracle_root, sid)
    ckpt_used = ckpt if ckpt is not None else Path(baseline)
    rec = {
        "kind": ROW_KIND, "dataset": ds, "set_id": sid, "row_id": row_id,
        "row_type": plan["row_type"], "seed": MB_SEED, "recipe": recipe,
        "label": plan["spec"].get("label"),
        "objective": plan["spec"].get("objective"),
        "note": plan["spec"].get("note"),
        "weights": {
            "source": plan["weights_source"],
            "checkpoint": str(ckpt_used),
            "checkpoint_sha256": (rv.sha256_file(ckpt_used)
                                  if ckpt_used.exists() else None),
            "initialized_from": str(baseline),
            "initialized_from_sha256": (rv.sha256_file(baseline)
                                        if Path(baseline).exists() else None),
        },
        "checkpoint_sha256": (rv.sha256_file(ckpt_used)
                              if ckpt_used.exists() else None),
        "training": training,
        "parameters": params,
        "metrics": metrics,
        "oracle_distances": dist,
        "hard_preds": hard,
        "soft_probs_full": {i: soft[i]["probs"] for i in ctx["identity_ids"]},
        "provenance": {
            "git_commit": EXECUTING_COMMIT,
            "runner_script_sha256": rv.sha256_file(
                Path(__file__).resolve()),
            "shared_scoring_script_sha256": rv.script_sha256(),
            "mb_manifest_sha256": rv.sha256_file(mb_manifest_path(ds)),
            "device": args.device,
        },
    }
    if prompt_policy is not None:
        rec["prompt_policy"] = prompt_policy
        rec["over_application_probe"] = probe
    frozen_cell = plan.get("frozen_cell_record")
    if frozen_cell is not None and Path(frozen_cell).exists():
        cell = json.loads(Path(frozen_cell).read_text(encoding="utf-8"))
        rec["frozen_cell_cross_check"] = {
            "source": str(frozen_cell),
            "cell_id": cell.get("cell_id"),
            "cell_checkpoint_sha256": cell.get("checkpoint_sha256"),
            "checkpoint_sha256_matches": (cell.get("checkpoint_sha256")
                                          == rec["checkpoint_sha256"]),
            "cell_criteria": cell.get("criteria"),
            "criteria_identical": (cell.get("criteria")
                                   == metrics["frozen_cell_criteria"]),
            "note": ("this runner re-scored the frozen cell checkpoint with "
                     "its own evaluation path; identical criteria prove the "
                     "method comparison uses the same yardstick as the "
                     "matrix cells"),
        }
    oracle_rec = plan.get("oracle_record")
    if oracle_rec is not None and Path(oracle_rec).exists():
        rec["oracle_fit"] = json.loads(
            Path(oracle_rec).read_text(encoding="utf-8"))
    return rec


# ====================================================================== #
# MB2: comparison table, descriptive screening, report, archive
# ====================================================================== #
ROW_ORDER = ("baseline_route", "sft_target", "cf_relabel",
             "gd_distribution", "ga_retain_descent", "npo",
             "kl_anchored_edit", "kl_ascent_anchor", "matched_retrain",
             "loo_retrain", *[f"prompt_only__{r}" for r in PROMPT_ROW_ROLES])
REFERENCE_ROWS = ("baseline_route", "loo_retrain")


def _load_rows(args, out_base, man, ds):
    """Load the ACCEPTED rows, validating each before it is trusted."""
    rows, rejected = {}, []
    for s in man["sets"]:
        sid = s["set_id"]
        for row_id in ROW_ORDER:
            rdir = row_dir(out_base, sid, row_id)
            path = rdir / "mb_row.json"
            if not path.exists():
                continue
            rec = json.loads(path.read_text(encoding="utf-8"))
            ckpt = rec.get("weights", {}).get("checkpoint")
            ckpt_sha = (rv.sha256_file(Path(ckpt))
                        if ckpt and Path(ckpt).exists() else None)
            reasons = validate_cached_row(rec, ds, sid, row_id,
                                          row_recipe(man, sid, row_id, args),
                                          ckpt_sha)
            if reasons:
                rejected.append({"set_id": sid, "row_id": row_id,
                                 "reasons": reasons, "path": str(path)})
                logger.warning("MB2: rejecting cached row %s/%s (%s)", sid,
                               row_id, ", ".join(reasons))
                continue
            rows[(sid, row_id)] = rec
    return rows, rejected


def _worst(values):
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _table_row(rec):
    m, d = rec["metrics"], rec["oracle_distances"]
    crit = m["frozen_cell_criteria"]
    per_t = d["per_target"]
    return {
        "row_id": rec["row_id"], "row_type": rec["row_type"],
        "label": rec["label"], "objective": rec["objective"],
        "weights_source": rec["weights"]["source"],
        "budget": rec["recipe"]["budget"],
        "trains": bool(rec["training"].get("trained")),
        "aliased_to": rec.get("aliased_to"),
        "transformation_success": {
            "strict_rate": m["transformation_success"]["strict_rate"],
            "all_targets_strict":
                m["transformation_success"]["all_targets_strict"],
            "per_target_strict": {
                t: v["strict_ok"] for t, v in
                m["transformation_success"]["per_target"].items()},
            "per_refusal_control_strict": {
                t: v["strict_ok"] for t, v in
                m["transformation_success"]["per_refusal_control"].items()},
        },
        "source_label_suppression": {
            "max_p_source": m["source_label_suppression"]["max_p_source"],
            "any_hard_leak": m["source_label_suppression"]["any_hard_leak"],
            "per_target_p_source": {
                t: v["p_source"] for t, v in
                m["source_label_suppression"]["per_target"].items()},
        },
        "retention": m["retention"],
        "sibling_preservation": m["sibling_preservation"],
        "wrong_branch_rate": m["wrong_branch_rate"],
        "candidate_validity": m["candidate_validity"],
        "matched_vs_loo_oracle_distance": {
            "worst_l2_matched_retrain": _worst(
                [v["l2_matched_retrain"] for v in per_t.values()]),
            "worst_l2_loo_retrain": _worst(
                [v["l2_loo_retrain"] for v in per_t.values()]),
            "mean_delta_retrain_l2": d["mean_delta_retrain_l2"],
            "worst_delta_retrain_l2": d["worst_delta_retrain_l2"],
            "closer_to_matched_than_loo": d["closer_to_matched_than_loo"],
            "n_targets": d["n_targets"],
            "n_targets_reliable": d["n_targets_reliable"],
            "per_target": per_t,
            "metric": d["metric"], "gate": d["gate"], "scope": d["scope"],
        },
        "training_time": {"train_sec": rec["training"].get("train_sec"),
                          "steps_requested":
                              rec["training"].get("steps_requested"),
                          "steps_executed":
                              rec["training"].get("steps_executed"),
                          "items": rec["training"].get("items")},
        "trainable_parameters": rec.get("parameters"),
        "frozen_cell_criteria": {"cell_pass": crit["cell_pass"],
                                 "failed_criteria": crit["failed_criteria"],
                                 "checks": crit["checks"]},
        "prompt_policy": ({"role": rec["prompt_policy"]["role"],
                           "mode": rec["prompt_policy"]["mode"],
                           "texts": rec["prompt_policy"]["texts"]}
                          if rec.get("prompt_policy") else None),
        "over_application_probe": (
            {k: v for k, v in rec["over_application_probe"].items()
             if k != "rows"} if rec.get("over_application_probe") else None),
    }


def aggregate_mb(args, ds, man, out_base):
    rows, rejected = _load_rows(args, out_base, man, ds)
    comparison, cross_checks = {}, []
    for s in man["sets"]:
        sid = s["set_id"]
        table = [_table_row(rows[(sid, r)]) for r in ROW_ORDER
                 if (sid, r) in rows]
        comparison[sid] = {"set_id": sid, "mode": s["mode"],
                           "transformation_targets":
                               s["transformation_targets"],
                           "refusal_controls": s["refusal_controls"],
                           "n_rows_expected": len(ROW_ORDER),
                           "n_rows_evaluated": len(table),
                           "table": table}
        sft = rows.get((sid, "sft_target"))
        if sft is not None and sft.get("frozen_cell_cross_check"):
            cc = sft["frozen_cell_cross_check"]
            cross_checks.append({
                "set_id": sid, "check": "sft_target re-scored against the "
                                        "committed frozen cell",
                "checkpoint_sha256_matches": cc[
                    "checkpoint_sha256_matches"],
                "criteria_identical": cc["criteria_identical"],
                "failed_criteria_here":
                    sft["metrics"]["frozen_cell_criteria"][
                        "failed_criteria"],
            })
    screening = screen_rows(comparison, man)
    return {
        "dataset": ds, "edit_seed": MB_SEED,
        "n_sets": len(comparison),
        "n_rows_evaluated": sum(c["n_rows_evaluated"]
                                for c in comparison.values()),
        "n_rows_expected": len(ROW_ORDER) * len(comparison),
        "rejected_cached_rows": rejected,
        "comparison": comparison,
        "cross_checks": cross_checks,
        "screening": screening,
    }


def screen_rows(comparison, man):
    """DESCRIPTIVE screening on the representative subset.

    This is not a promotion gate and changes no threshold anywhere: it uses
    the frozen per-cell criteria (gx.PASS_CRITERIA) that the matrix already
    applies, and reports how many representative sets each row passes.
    """
    out = {}
    for row_id in ROW_ORDER:
        rows = [(sid, next(r for r in c["table"] if r["row_id"] == row_id))
                for sid, c in comparison.items()
                if any(r["row_id"] == row_id for r in c["table"])]
        if not rows:
            out[row_id] = {"n_sets_evaluated": 0,
                           "screening_verdict": "no_rows"}
            continue
        if row_id in REFERENCE_ROWS:
            out[row_id] = {
                "n_sets_evaluated": len(rows),
                "screening_verdict": "reference_row_not_screened",
                "note": man["references"][row_id]["note"],
            }
            continue
        passing = [sid for sid, r in rows
                   if r["frozen_cell_criteria"]["cell_pass"]]
        transforming = [sid for sid, r in rows
                        if r["transformation_success"]["all_targets_strict"]]
        failed = sorted({c for _sid, r in rows for c in
                         r["frozen_cell_criteria"]["failed_criteria"]})
        deltas = [r["matched_vs_loo_oracle_distance"][
            "worst_delta_retrain_l2"] for _sid, r in rows]
        verdict = ("viable_on_representative_subset"
                   if len(passing) == len(rows)
                   else "partially_viable" if passing else "not_viable")
        if row_id == "cf_relabel":
            verdict = "alias_of_sft_target_not_screened_separately"
        out[row_id] = {
            "label": rows[0][1]["label"],
            "row_type": rows[0][1]["row_type"],
            "aliased_to": rows[0][1]["aliased_to"],
            "n_sets_evaluated": len(rows),
            "n_sets_passing_frozen_criteria": len(passing),
            "n_sets_transforming_all_targets": len(transforming),
            "sets_passing": passing,
            "sets_failing": [sid for sid, _r in rows
                             if sid not in set(passing)],
            "failed_criteria_union": failed,
            "worst_delta_retrain_l2": min(
                [d for d in deltas if d is not None], default=None),
            "total_train_sec": round(sum(
                (r["training_time"]["train_sec"] or 0.0)
                for _sid, r in rows), 1),
            "trainable_parameters": rows[0][1]["trainable_parameters"],
            "screening_verdict": verdict,
            "screening_is_descriptive": True,
        }
    return out


def build_mb_claims(ds, agg, man):
    """Claims generated FROM the table: no number is written by hand."""
    scr = agg["screening"]
    viable = sorted(k for k, v in scr.items()
                    if v.get("screening_verdict")
                    == "viable_on_representative_subset")
    partial = sorted(k for k, v in scr.items()
                     if v.get("screening_verdict") == "partially_viable")
    failed = sorted(k for k, v in scr.items()
                    if v.get("screening_verdict") == "not_viable")
    missing = sorted(k for k, v in scr.items()
                     if v.get("screening_verdict") == "no_rows")
    deltas = {k: v.get("worst_delta_retrain_l2") for k, v in scr.items()
              if v.get("worst_delta_retrain_l2") is not None}
    delta_text = ", ".join(f"{k}={v:+.3f}" for k, v in sorted(
        deltas.items(), key=lambda kv: -kv[1])) or "none established"
    times = {k: round(v, 1) for k, v in sorted(
        ((k, v.get("total_train_sec")) for k, v in scr.items()),
        key=lambda kv: kv[0]) if v}
    time_text = (", ".join(f"{k}={v}s" for k, v in times.items())
                 or "none (no row trained)")
    scope = (
        f"granularity METHOD screening on a frozen representative subset "
        f"({agg['n_sets']} sets, one per transformation type, edit seed "
        f"{MB_SEED} only, {agg['n_rows_evaluated']}/"
        f"{agg['n_rows_expected']} rows evaluated); every row was measured at "
        f"the association level (code -> label) under strict parsing "
        f"(multi-label = invalid, never first-match) with the SAME frozen "
        f"per-cell criteria the matrix cells use; every trainable row consumed "
        f"the identical target/source/retain pairs at the identical edit "
        f"budget and only the objective differed; the fresh-retrain oracle "
        f"rows keep their own frozen protocol and are references, not "
        f"budget-matched competitors; candidate quantities are SCORE SUMS "
        f"without a termination event, so the 0.99 interpretability threshold "
        f"is a threshold on a score sum")
    headline = (
        f"method screening ({ds}): rows passing the frozen criteria on every "
        f"representative set are {viable or 'NONE'}; partially viable "
        f"{partial or 'none'}; not viable {failed or 'none'}; not evaluated "
        f"{missing or 'none'}.  Worst-case Delta_retrain = D(loo_retrain) - "
        f"D(matched_retrain) per row: [{delta_text}] (positive = closer to "
        f"the fresh matched retraining reference than to the fresh deletion "
        f"reference).  Total training time per row over the subset: "
        f"[{time_text}], trainable parameters identical across weight-editing "
        f"rows (same frozen LoRA configuration)")
    not_claimed = [
        ("this is a SCREENING pass on a representative subset at one edit "
         "seed: no method is promoted, no threshold, gate, promotion "
         "criterion or training recipe anywhere in this repository changes, "
         "and nothing reads this output"),
        ("the full 147-cell matrix (SALMU 63 + numeric 84) is deliberately "
         "NOT multiplied out across these baselines; that happens only for "
         "methods this pass identifies as viable"),
        ("cf_relabel is an ALIAS of sft_target, not a second method: for a "
         "transformation target the counterfactual label is the requested "
         "coarser label, so the two are the same objective on the same data"),
        ("prompt_only changes no weights, so its retention and sibling "
         "numbers are 1.0 BY CONSTRUCTION and are not evidence of a better "
         "editing method; its policies state the abstraction rule and MB0 "
         "fails if any policy text contains a target or source label"),
        ("matched_retrain and loo_retrain are evaluated at their own frozen "
         "3000/200/2e-5 fresh-LoRA protocol; their training time is not "
         "comparable to the 500-step edit budget and is labelled as such"),
        ("matched_retrain is the WEIGHTED (target x5) reference every existing "
         "Delta_retrain number used; the balanced target-x1 variant is GX2B's "
         "ablation and is not part of this method comparison"),
        ("kl_ascent_anchor is rv.train_kl as shipped for DELETION: it has no "
         "descent term, so a near-zero transformation success there is a "
         "property of the objective, not a tuning failure"),
        ("oracle distances are gated: below the frozen minimum candidate "
         "score sum a distance is None ('not established'), never 0.0, and "
         "a row whose distances are not established makes no proximity "
         "claim"),
    ]
    return {"scope": scope, "headline": headline, "not_claimed": not_claimed}


def archive_mb(ds, out_base, man, commit):
    """Trained adapters + report, revision-pinned (the GX2S/RG pattern)."""
    rel = Path("releases") / f"e2c_methodbaselines_{ds}_{commit[:7]}"
    rel.mkdir(parents=True, exist_ok=True)
    entries = []
    for s in man["sets"]:
        sid = s["set_id"]
        for row_id in ROW_ORDER:
            src = (row_dir(out_base, sid, row_id) / "adapter_final"
                   / "adapter_model.safetensors")
            if not src.exists():
                continue
            name = f"mb_{sid}__{row_id}.safetensors"
            dest = rel / name
            shutil.copy2(src, dest)
            entries.append({"kind": "method_baseline_adapter",
                            "key": f"{sid}__{row_id}", "file": name,
                            "sha256": rv.sha256_file(dest),
                            "bytes": dest.stat().st_size})
    report_src = MB_REPORT_DIR / f"method_baselines_{ds}.json"
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
            path_in_repo=f"method_baselines_{ds}_{commit[:7]}",
            repo_type="model",
            commit_message=f"E2C-v3 method baselines ({ds}) @ {commit[:7]}")
        hf_ok = True
        hf_revision = url.rstrip("/").rsplit("/", 1)[-1] if url else None
        for e in entries:
            e["hf_revision"] = hf_revision
            e["hf_uri"] = (f"https://huggingface.co/{gxm.HF_ARCHIVE_REPO}/"
                           f"resolve/{hf_revision}/method_baselines_"
                           f"{ds}_{commit[:7]}/{e['file']}")
        logger.info("MB2: uploaded %d adapters, revision %s", len(entries),
                    hf_revision)
    except Exception as exc:
        logger.warning("MB2: HF upload unavailable (%s)", str(exc)[:120])
    with open(rel / "CHECKSUMS.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{e['sha256']}  {e['file']}\n" for e in entries)
    manifest = {"kind": "method_baselines_archive", "dataset": ds,
                "git_commit": commit, "release_dir": str(rel),
                "hf_repo": gxm.HF_ARCHIVE_REPO if hf_ok else None,
                "hf_upload_ok": hf_ok, "hf_revision": hf_revision,
                "n_files": len(entries),
                "uri_immutability_note": "resolve/<hf_commit_sha> pinned",
                "entries": entries}
    (rel / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_mb2(args, ds, man, out_base, provenance, commit, t_start,
            stale=None):
    agg = aggregate_mb(args, ds, man, out_base)
    if not agg["n_rows_evaluated"]:
        rejected = agg["rejected_cached_rows"]
        detail = ("; every cached row on disk was REJECTED: "
                  + json.dumps(rejected)) if rejected else ""
        raise RuntimeError("MB2: no validated rows to compare; run MB1 "
                           f"first{detail}")
    cache_block = {
        "policy": MB_CACHE_VALIDATION,
        "stale_records_quarantined": stale or [],
        "cached_rows_rejected_at_aggregation": agg["rejected_cached_rows"],
    }
    archive = {"release_dir": None, "hf_upload_ok": False,
               "hf_revision": None, "n_files": 0}
    if not args.smoke:
        archive = archive_mb(ds, out_base, man, commit)
    report = {
        "kind": "e2c_v3_method_baselines_report_v1",
        "dataset": ds,
        "produced_by": "scripts/e2c_v3_method_baselines.py",
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "provenance": provenance,
        "design": {
            "mb_manifest_sha256": rv.sha256_file(mb_manifest_path(ds)),
            "edit_seed": MB_SEED,
            "rep_set_policy": man["rep_set_policy"],
            "budget_policy": man["budget_policy"],
            "methods": man["methods"], "references": man["references"],
            "required_metrics": man["required_metrics"],
            "reported_metrics": MB_REPORTED_METRICS,
            "scoring_limitation": man["scoring_limitation"],
            "prompt_policy_roles": list(PROMPT_POLICY_ROLES),
            "pass_criteria_reference": man["pass_criteria_reference"],
        },
        "cache_validation": cache_block,
        "comparison": agg["comparison"],
        "cross_checks": agg["cross_checks"],
        "screening": agg["screening"],
        "aggregate": {k: v for k, v in agg.items()
                      if k not in ("comparison", "screening",
                                   "cross_checks")},
        "claims": build_mb_claims(ds, agg, man),
        "archive": {"release_dir": archive.get("release_dir"),
                    "hf_upload_ok": archive.get("hf_upload_ok"),
                    "hf_revision": archive.get("hf_revision"),
                    "n_files": archive.get("n_files", 0)},
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    MB_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = MB_REPORT_DIR / f"method_baselines_{ds}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("MB2: report written to %s", path)
    logger.info("MB2 CLAIM scope: %s", report["claims"]["scope"])
    logger.info("MB2 CLAIM headline: %s", report["claims"]["headline"])
    for row_id, sc in sorted(agg["screening"].items()):
        logger.info("MB2 SCREEN %-34s %s (%s/%s sets pass)", row_id,
                    sc.get("screening_verdict"),
                    sc.get("n_sets_passing_frozen_criteria", 0),
                    sc.get("n_sets_evaluated", 0))
    run_manifest = {
        "experiment": f"e2c_v3_method_baselines_{ds}",
        "produced_by": "scripts/e2c_v3_method_baselines.py",
        "provenance": provenance,
        "inputs_sha256": man["inputs"],
        "mb_manifest_sha256": rv.sha256_file(mb_manifest_path(ds)),
        "rows_evaluated": agg["n_rows_evaluated"],
        "rows_expected": agg["n_rows_expected"],
        "screening_verdicts": {k: v.get("screening_verdict")
                               for k, v in agg["screening"].items()},
        "cross_checks": agg["cross_checks"],
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
MB_CODE = ["scripts/e2c_v3_method_baselines.py",
           "scripts/e2c_v3_research_validity.py",
           "scripts/e2c_v3_granularity.py",
           "scripts/e2c_v3_granularity_matrix.py",
           "scripts/e2c_v3_matrix.py",
           "scripts/e2c_v3_realdata.py"]


def _dirty_tracked_code():
    """Tracked-file changes among the EXECUTED code (not result outputs).

    A declared script that is not on disk is reported as such: dropping it from
    the pathspec would silently widen ``git status`` to the WHOLE worktree and
    blame this comparison for edits made by the parallel runs.
    """
    missing = [p for p in MB_CODE if not Path(p).exists()]
    if missing:
        return [f"<declared executed code missing: {p}>" for p in missing]
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no", *MB_CODE],
            text=True)
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return ["<git status failed>"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   choices=["salmu", "celeba_numeric"])
    p.add_argument("--phase", default="all",
                   choices=["all", "MB0", "MB1", "MB2"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--only-sets", nargs="*", default=None)
    p.add_argument("--only-rows", nargs="*", default=None,
                   help="restrict EXECUTION to these row ids (the frozen "
                        "design always declares all of them)")
    p.add_argument("--retrain", action="store_true",
                   help="retrain a row even if its adapter exists (the cached "
                        "record is still validated, never trusted blindly)")
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
                   help="tiny budget, one set, a few rows, *_smoke tree, no "
                        "archive")
    return p.parse_args()


SMOKE_ROWS = ["baseline_route", "sft_target", "cf_relabel",
              "gd_distribution", "kl_anchored_edit",
              "prompt_only__rule_statement"]


def main():
    global EXECUTING_COMMIT
    args = parse_args()
    t_start = time.time()
    if args.smoke:
        args.ul_steps, args.ul_warmup = 20, 2
        args.route_steps, args.route_warmup = 20, 2
        args.route_repeat = 2
        args.only_rows = list(SMOKE_ROWS)
    args.seed = MB_SEED
    ds = args.dataset
    suffix = "_smoke" if args.smoke else ""
    out_base = MB_OUT_ROOT / f"{ds}{suffix}"
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
            f"Result files may be dirty (parallel granularity/panel/RG runs "
            f"own them), but the code that produces this comparison may not "
            f"be.")

    # the frozen granularity matrix is a committed INPUT: read-only here
    matrix = gxm.load_frozen(ds)
    ctx = gxm.dataset_ctx(ds, matrix)
    man = build_or_verify(ds, matrix, ctx)
    if args.phase == "MB0":
        logger.info("MB0 COMPLETE -- commit the design before evaluating")
        return 0
    if args.smoke:
        man = json.loads(json.dumps(man))
        man["sets"] = man["sets"][:1]
        man["n_sets"] = 1
        if args.only_sets:
            man["sets"] = [s for s in man["sets"]
                           if s["set_id"] in set(args.only_sets)]
            man["n_sets"] = len(man["sets"])
    stale = []
    if args.phase in ("all", "MB1"):
        stale = run_rows(args, ds, ctx, man, matrix, out_base)
    if args.phase in ("all", "MB2"):
        run_mb2(args, ds, man, out_base, provenance, EXECUTING_COMMIT,
                t_start, stale=stale)
    logger.info("=" * 60)
    logger.info("METHOD BASELINES (%s) PHASE %s COMPLETE (%.1fs)", ds,
                args.phase, time.time() - t_start)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
