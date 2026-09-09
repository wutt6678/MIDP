#!/usr/bin/env python3
"""E2C-v3 granularity matrix runner (GX0-GX7): controlled coarsening edits.

Answers: can one edited h reliably replace an entity-specific association
with a VALID, deliberately COARSER association at a controlled taxonomic or
numeric depth, while preserving unrelated associations and sibling
specificity?  (Plan sections 3-12; suppression-specific pass logic from
e2c_v3_matrix.py is deliberately NOT reused -- only low-level facilities:
model session, exact checkpoint reset, strict parsing, full-sequence
scoring, candidate/OTHER accounting, archival, provenance.)

Datasets
========
- salmu            : taxonomic (specific->level1, specific->level2,
                     mixed-depth simultaneous with a refusal control);
                     21 sets x 3 seeds = 63 cells; cached-g E2E replay.
- celeba_numeric   : frozen synthetic numeric profiles (24 identities,
                     years_experience + activity_count; exact->narrow,
                     exact->broad, exact->rounded, narrow->broad);
                     28 sets x 3 seeds = 84 cells; ASSOCIATION LEVEL ONLY
                     (no image router; CelebA g redesign is separate).
- mllmu            : G6.1 five-set coarsening pilot over the externally
                     audited MLLMU -> SOC hierarchy (see
                     e2c_v3_mllmu_matrix.py).  5 sets x 3 seeds = 15 cells;
                     ASSOCIATION LEVEL ONLY -- MLLMU-Bench has one image per
                     identity, so there is no image-level held-out g and no
                     cached-g replay.  The design is frozen by that module and
                     re-derived here for comparison; this runner never rebuilds
                     it from the benchmark, so a run needs no out-of-repo data.
                     Adds a same_leaf control category: two raw labels
                     adjudicated onto one SOC leaf are the SAME occupation, and
                     reporting one as the other's sibling would put the sharpest
                     retention control in the benchmark into a metric that
                     assumes taxonomic distance.

Oracles (plan section 9, CORRECTED in G3.1): FOUR named families per set,
distinguishing initialization AND data --
  matched_finetune : trained-baseline-h init; transformed full mapping
                     (continued fine-tuning reference);
  loo_finetune     : trained-baseline-h init; retained mapping only
                     (deletion-as-fine-tuning reference; salmu reuses
                     suppression-matrix LOO oracles on exact set match);
  matched_retrain  : FRESH base + fresh LoRA; transformed full mapping;
                     route h's own protocol (3000/200/2e-5, uniform repeat
                     50, seed 17) PLUS the EDIT recipe's target
                     oversampling (mx.TARGET_BOOST=5).  Reported as
                     matched_retrain_weighted, because route h itself is
                     trained by rd.train_h at a uniform repeat with NO
                     target boost -- so the x5 weighting is not "the
                     ordinary route-h recipe" and must not be described as
                     one;
  loo_retrain      : FRESH base + fresh LoRA; retained mapping only;
                     same protocol, and no transformed target to weight.
Delta_FT     = D(edit, loo_finetune) - D(edit, matched_finetune)
Delta_retrain= D(edit, loo_retrain)  - D(edit, matched_retrain)
Positive delta = the edit is closer to the transformation-matched
reference than to the deletion reference of the SAME family.  G3's
"matched/loo" oracles were finetune-family references; the
"policy-matched-retraining equivalent" wording of commit a1df9be was
therefore too strong and is superseded: only the retrain families
support retraining claims, and even then proximity is scoped to the
evaluated code prompts and candidate-label space (a tiny L2 between
nearly one-hot distributions does not establish global functional or
parameter equivalence).

GX2B balanced matched-retrain ablation: the main matched_retrain reference
oversamples the transformed target x5.  That x5 is mx.TARGET_BOOST -- the
multiplier the EDIT/suppression recipe applies to its suppression pairs
(ul_repeat*5 against retained ul_repeat*3) -- and NOT a property of route h,
which rd.train_h builds at a uniform repeat with no target boost at all.  The
x5 reference is therefore named matched_retrain_weighted, and GX2B trains
matched_retrain_balanced with target_boost=1 (the transformed mapping appears
once per epoch, exactly like each retained mapping, which is how route h itself
is built), holding steps/warmup/lr/repeat/LoRA-config/seed IDENTICAL, on one
representative set per transformation type (SALMU L1 single, L2 single,
mixed-depth; numeric narrow, broad).  It reuses the SAME edited cells E and
compares D(E, matched_retrain_balanced), D(E, matched_retrain_weighted) and
D(E, loo_retrain).  Promotion requires: the balanced oracle FITS
transformed+retained mappings, D(E, balanced) < D(E, loo_retrain),
Delta_balanced = D(E,loo_retrain)-D(E,balanced) >= 0.5, and the conclusion
agrees across ALL representative transformation types (refusal controls
reported separately).

GX2S oracle-seed sensitivity: the full matrices vary the EDIT seed but hold
the fresh-retrain references at seed 17.  GX2S trains matched_retrain +
loo_retrain at ORACLE seeds 42 and 123 too (same protocol, seed 17 dirs
reused) for the same representative sets, and reports the paired
(edit_seed x oracle_seed) table Set | Edit seed | Oracle seed | D_matched |
D_LOO | Delta (worst-case transformation target per row).  The robust claim
rests on the SIGN/MARGIN of Delta, not exact distance equality: gate A =
every edit seed closer to every matched-oracle seed than to the
corresponding LOO oracle; gate B = worst-case Delta stays positive
(preferably >= 0.5); passed = A OR B, passed_strong = A AND worst >= 0.5.
Missing edit seeds (e.g. numeric 42/123 before G5 completes cells) are
reported as pending and the CPU comparison can be re-run to fill the table.

Phases: GX0 validate schemas/matrices | GX1 freeze matrix files
        GX1R numeric baseline route h | GX2 oracle families
        GX2R CPU re-evaluation of stored cells vs all families
        GX2B balanced matched-retrain ablation (target_boost=1) + report
        GX2S oracle-seed sensitivity (matched/LOO retrain @ seeds 42,123)
        GX3 single cells | GX4 same-depth simultaneous | GX5 mixed
        GX6 (inside cells: conditional/unconditional E2E)
        GX7 aggregate (+G3.1 gate, scoped claims) + archive
            (revision-pinned HF) + manifest
"""
import argparse
import importlib.util
import json
import logging
import shutil
import statistics
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("e2c_v3_gx")

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rv = _load_sibling("e2c_rv_gx", "e2c_v3_research_validity.py")
rd = _load_sibling("e2c_rd_gx", "e2c_v3_realdata.py")
mx = _load_sibling("e2c_mx_gx", "e2c_v3_matrix.py")
gx = _load_sibling("e2c_gx_lib", "e2c_v3_granularity.py")
# G6.1 owns the frozen MLLMU pilot design.  Loaded as a sibling so this runner
# re-derives the design with the SAME builder that wrote it, instead of keeping
# a second copy of the rules that could drift from the committed artifact.
g6m = _load_sibling("e2c_g6_mllmu", "e2c_v3_mllmu_matrix.py")

GRAN_ROOT = Path("e2c_granularity")
MANIFEST_DIR = GRAN_ROOT / "manifests"
OUT_ROOT = GRAN_ROOT / "outputs"

SALMU_MANIFEST = Path("e2c_salmu/manifests/salmu_manifest.json")
SALMU_ROUTE_H = Path("e2c_salmu/outputs/salmu/h_C_to_Y/adapter_final/"
                     "adapter_model.safetensors")
SALMU_G_CACHE = Path("e2c_salmu/outputs/salmu/e2e_post/e2e_results.json")
# leave-one-out oracles from the suppression matrix (exact-set reuse only)
SUPPRESSION_LOO_DIR = Path("e2c_matrix/outputs/salmu/oracles")

ORACLE_SEED = 17
HF_ARCHIVE_REPO = mx.HF_ARCHIVE_REPO
TARGET_BOOST = 5
RETAIN_REPEAT = 3
MIN_CANDIDATE_MASS = 0.01
# G3.1 correction: the families below distinguish initialization AND data.
# matched/loo *_finetune = trained-baseline-h init (continued fine-tuning);
# matched/loo *_retrain  = FRESH base + fresh LoRA, trained with route h's
# own protocol (3000/200/2e-5, uniform repeat 50, seed 17).
# Only the retrain families support retraining claims.
RETRAIN_STEPS = 3000
RETRAIN_WARMUP = 200
RETRAIN_LR = 2e-5
RETRAIN_REPEAT = 50             # rd.ROUTE_REPEAT: every mapping, uniform
# The transformed target's oversampling.  This is mx.TARGET_BOOST -- the
# multiplier the EDIT/suppression recipe applies to its suppression pairs --
# and it is NOT part of route h: rd.train_h builds every code->alias pair at
# repeat=ROUTE_REPEAT with no boost anywhere in the module.  Describing x5 as
# "the original route-h recipe" misattributes the edit recipe's weighting to
# the route and makes the weighted reference read as the neutral one, which is
# why the two references carry the names below.
RETRAIN_TARGET_BOOST = 5
DELTA_RETRAIN_MIN_MARGIN = 0.5  # G3.1 materiality margin (L2)
ORACLE_FAMILIES = ("matched_finetune", "loo_finetune",
                   "matched_retrain", "loo_retrain")

# GX2B balanced matched-retrain ablation.  matched_retrain_balanced uses
# target_boost=1 (the transformed mapping appears once per epoch, exactly like
# each retained mapping -- which is how route h itself is built), with
# steps/warmup/lr/repeat/LoRA-config/seed held IDENTICAL to the weighted
# reference.  Comparing D(E, balanced), D(E, weighted) and D(E, loo_retrain)
# tests whether the 'edit ~= policy-matched retraining' reading survives
# WITHOUT the edit recipe's target oversampling, using the SAME edited cells E
# (no retraining of E).
RETRAIN_TARGET_BOOST_BALANCED = 1
BALANCED_FAMILY = "matched_retrain_balanced"
WEIGHTED_FAMILY = "matched_retrain"          # existing x5 dir name
#: Reporting names.  Reports and prose name the reference they mean; "x5"/"x1"
#: survive only as stored numeric keys, declared in REFERENCE_NAMING below.
WEIGHTED_LABEL = "matched_retrain_weighted"  # target_boost=5
BALANCED_LABEL = "matched_retrain_balanced"  # target_boost=1
#: Shipped in the reports so a reader can map every stored key to the
#: reference it holds, and see why the keys were not renamed.  Same pattern as
#: the prompt panel's report_layer_naming: name the quantity in prose, keep the
#: stored field stable, and say so.
REFERENCE_NAMING = {
    "matched_retrain_weighted": (
        "target_boost=5: the transformed mapping is oversampled 5x.  That x5 "
        "is mx.TARGET_BOOST from the EDIT/suppression recipe, NOT route h's "
        "recipe -- rd.train_h builds every mapping at a uniform "
        "repeat=ROUTE_REPEAT with no target boost"),
    "matched_retrain_balanced": (
        "target_boost=1: the transformed mapping appears once per epoch, "
        "exactly like each retained mapping, which is how route h itself is "
        "trained"),
    "stored_numeric_keys_keep_the_shorthand": (
        "d_E_Mx1 / d_E_Mx5 / delta_x1 / delta_x5 / Mx1_vs_Mx5 / "
        "Mx1_closer_than_L are UNCHANGED: renaming a stored key would break "
        "comparability with the committed ablation reports and with the cells "
        "already measured under those keys"),
    "key_to_name": {"Mx1": "matched_retrain_balanced",
                    "Mx5": "matched_retrain_weighted",
                    "x1": "matched_retrain_balanced",
                    "x5": "matched_retrain_weighted",
                    "L": "loo_retrain",
                    "E": "edited cell E (edit seed 17)"},
}
# One representative per transformation type; all already have the edited
# cell E (seed 17), matched_retrain_weighted and loo_retrain on disk from the
# G3.1 pilot / main matrix.
BALANCED_REP_SETS = {
    "salmu": ["gx_sal_s_L1_00060576",   # specific->level1 (sibling ctl)
              "gx_sal_s_L2_00039880",   # specific->level2
              "gx_sal_mix_0"],          # mixed-depth + refusal control
    "celeba_numeric": ["gx_num_s_exact_to_narrow_Y02",  # exact->narrow
                       "gx_num_s_exact_to_broad_Y04"],  # exact->broad
}


def _device_name():
    return (torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu")


def _cumulative_elapsed(artifact_path, t_start, device_now=None):
    """Cost of every pass that produced the artifact at ``artifact_path``.

    ``elapsed_sec`` used to be ``time.time() - t_start`` for the invoking pass
    alone, so re-aggregating a finished matrix on CPU overwrote the record of a
    244,305s GPU run with the seconds the re-derivation took, and the cost of
    the training that produced the numbers was gone from the manifest.  A field
    describing what producing this state cost has to accumulate, and a pass that
    cannot read its predecessor says so rather than silently restarting the
    total from zero.  Same fix as the method-baseline runner's
    ``_cumulative_elapsed``.
    """
    this_pass = round(time.time() - t_start, 1)
    path = Path(artifact_path)
    prior, devices, carried = 0.0, [], None
    carried_gap = None
    prev_named_a_device = False
    note = "no earlier artifact at this path: this pass is the whole total"
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
            prior = float(prev.get("elapsed_sec") or 0.0)
            devices = list(prev.get("devices_used")
                           or ([prev["gpu"]] if prev.get("gpu") else []))
            prev_named_a_device = bool(devices)
            carried = prev.get("elapsed_restoration")
            carried_gap = prev.get("elapsed_device_coverage")
            note = ("prior elapsed_sec read from the artifact this pass "
                    "overwrites")
        except Exception as exc:
            note = (f"the earlier artifact was UNREADABLE ({exc!r}), so the "
                    f"total restarts at this pass and UNDERSTATES every run "
                    f"before it")
    if device_now and device_now not in devices:
        devices.append(device_now)
    out = {
        "elapsed_sec": round(prior + this_pass, 1),
        "elapsed_this_pass_sec": this_pass,
        "elapsed_prior_passes_sec": prior,
        "elapsed_covers": (
            "every pass whose artifact this one overwrites, including the GPU "
            "passes that trained the cells and the oracles; a CPU "
            "re-aggregation adds its own seconds and erases nothing"),
        "elapsed_accumulation": note,
    }
    if devices:
        out["devices_used"] = devices
    if carried_gap:
        # Once recorded, never dropped.  Recomputing this from the predecessor
        # cannot work: the first pass to inherit the gap writes a devices_used
        # of its own, and every later pass then reads that list as coverage of
        # a cost it does not cover.
        out["elapsed_device_coverage"] = carried_gap
    elif prior > 0.0 and not prev_named_a_device:
        # The cost of the earlier passes is carried, but they recorded no
        # device, so devices_used cannot vouch for what produced it.  Saying
        # "cpu" beside a 5,654s prior would describe GPU training as CPU work.
        out["elapsed_device_coverage"] = (
            f"devices_used lists only the devices recorded from this pass "
            f"onward; the {prior}s in elapsed_prior_passes_sec came from "
            f"earlier passes that recorded a cost but no device, so their "
            f"device is NOT established here and devices_used UNDERSTATES "
            f"them")
    if carried:
        out["elapsed_restoration"] = carried
    return out


# GX2S oracle-seed sensitivity.  The full matrices vary the EDIT seed but
# hold the fresh-retrain references at ORACLE_SEED=17.  GX2S additionally
# trains matched_retrain + loo_retrain at these oracle seeds for the same
# representative sets and reports the paired (edit_seed x oracle_seed)
# Delta table.  The robust claim rests on the SIGN and MARGIN of Delta, not
# exact distance equality across seeds.
ORACLE_SEEDS_SENSITIVITY = (17, 42, 123)


def retrain_oracle_dir(family, sid, oracle_seed):
    """Directory name for a fresh-retrain oracle at a given oracle seed.

    Seed 17 (ORACLE_SEED) keeps the EXISTING unsuffixed name so the main
    matrix / G3.1 artifacts are reused; other seeds get an explicit
    ``__oseed<N>`` suffix (matched_retrain_<sid>__oseed42, etc.).
    """
    base = f"{family}_{sid}"
    return base if oracle_seed == ORACLE_SEED else f"{base}__oseed{oracle_seed}"

SINGLE_MODES = {"single_level1", "single_level2", "single_exact_to_narrow",
                "single_exact_to_broad", "single_exact_to_rounded"}
SAME_DEPTH_MODES = {"simultaneous_same_depth_l1", "simultaneous_same_depth_l2",
                    "simultaneous_same_resolution"}
MIXED_MODES = {"simultaneous_mixed_depth", "simultaneous_mixed_resolution"}


def dataset_ctx(ds, matrix):
    """Build the per-dataset evaluation context (frozen inputs only)."""
    if ds == "mllmu":
        # leaf_of is the one key SALMU and CelebA do not carry, and it is what
        # makes a same-leaf partner report as same_leaf instead of sibling.
        manifest = json.loads(g6m.G6_MANIFEST_PATH.read_text(encoding="utf-8"))
        return g6m.g6_ctx(manifest, g6m.load_hierarchy(), matrix)
    if ds == "salmu":
        with open(SALMU_MANIFEST) as f:
            sm = json.load(f)
        hierarchy_of = {i: list(v) for i, v in sm["job_levels"].items()}
        return {
            "kind": "taxonomic",
            "identity_ids": sm["identity_ids"],
            "code_of": sm["code_of"],
            "baseline_alias_of": sm["alias_of"],
            "hierarchy_of": hierarchy_of,
            "dag": {k: v for k, v in matrix["dag"].items()},
            "vocab": matrix["vocab"],
        }
    # celeba_numeric
    with open(MANIFEST_DIR / "numeric_manifest.json") as f:
        nm = json.load(f)
    vocab = sorted(set(nm["alias_of"].values())
                   | {a["target"] for e in matrix["sets"]
                      for a in e["assignments"].values()}
                   | {gx.DELETED_LABEL})
    return {
        "kind": "numeric",
        "identity_ids": nm["identity_ids"],
        "code_of": nm["code_of"],
        "baseline_alias_of": nm["alias_of"],
        "profiles": nm["profiles"],
        "schema": nm["schema"],
        "vocab": vocab,
    }


def expected_label(ctx, entry, iid):
    a = entry["assignments"].get(iid)
    return a["target"] if a else ctx["baseline_alias_of"][iid]


def control_group(ctx, entry, iid):
    """target | same_leaf | sibling | cousin | unrelated | retain (report)."""
    if iid in entry["assignments"]:
        return "target"
    # same_leaf BEFORE sibling: an identity sharing the target's canonical
    # leaf is the same occupation at a different raw label, and reporting it
    # as a sibling would put the sharpest retention control available into a
    # metric that assumes taxonomic distance.  Absent for SALMU/CelebA, whose
    # entries carry no same_leaf key, so their reports are unchanged.
    for ctl in entry.get("controls", {}).values():
        if iid in ctl.get("same_leaf", []):
            return "same_leaf"
    for ctl in entry.get("controls", {}).values():
        if iid in ctl.get("sibling", []):
            return "sibling"
    for ctl in entry.get("controls", {}).values():
        if iid in ctl.get("cousin", []):
            return "cousin"
    return "retain"


# ====================================================================== #
# GX0 / GX1: validation + frozen matrix files (verify-not-rewrite)
# ====================================================================== #
def build_or_verify(ds, args):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    # matrices are FROZEN with the default seed set; runs restrict
    # execution via --only-seeds/--only-sets, never by rebuilding
    extra_validation = {}
    if ds == "mllmu":
        # Re-derived with the SAME builder that froze it, from committed inputs
        # only: the pilot manifest already holds the identity selection, so no
        # benchmark read is needed and a run works on a bare checkout.  The G6.0
        # hierarchy is re-verified rather than trusted from its recorded digests.
        rebuilt = g6m.rebuild_frozen_matrix(verify=True)
        matrix = rebuilt["matrix"]
        path = g6m.G6_MATRIX_PATH
        extra_validation = {
            "g6_0_evidence": rebuilt["evidence"],
            "same_leaf_coverage":
                rebuilt["validation"]["same_leaf_coverage"],
            "sibling_availability": g6m.sibling_availability(matrix),
            "n_identities": len(rebuilt["manifest"]["identity_ids"]),
            "roster": rebuilt["manifest"]["roster"],
        }
    elif ds == "salmu":
        with open(SALMU_MANIFEST) as f:
            sm = json.load(f)
        matrix, _builder_ctx = gx.build_salmu_matrix(sm)
        path = MANIFEST_DIR / "matrix_salmu.json"
    else:
        nm = gx.build_numeric_manifest()
        nm_path = MANIFEST_DIR / "numeric_manifest.json"
        if nm_path.exists():
            with open(nm_path) as f:
                committed = json.load(f)
            if committed != nm:
                raise RuntimeError(
                    f"committed {nm_path} differs from the frozen schema in "
                    f"code; bins/profiles must never change post-hoc")
        else:
            with open(nm_path, "w") as f:
                json.dump(nm, f, indent=2)
        matrix = gx.build_numeric_matrix(nm)
        path = MANIFEST_DIR / "matrix_celeba_numeric.json"
    ctx = dataset_ctx(ds, matrix)

    # GX0 validation (hard gate before anything else)
    issues, notes = [], {}
    for entry in matrix["sets"]:
        iss = gx.validate_set(entry, ctx)
        if iss:
            issues.extend(f"{entry['set_id']}: {i}" for i in iss)
        if entry.get("control_notes"):
            notes[entry["set_id"]] = entry["control_notes"]
    vocab_issues, collisions = gx.validate_vocab(ctx["vocab"])
    issues.extend(vocab_issues)
    if ds == "celeba_numeric":
        nm = json.loads((MANIFEST_DIR / "numeric_manifest.json").read_text())
        issues.extend(gx.validate_numeric_boundary_coverage(nm))
    if issues:
        for i in issues:
            logger.error(f"GX0 ISSUE: {i}")
        raise RuntimeError(f"GX0 validation failed with {len(issues)} issues")
    logger.info(f"GX0: {ds} validation PASSED "
                f"({matrix['n_sets']} sets, {matrix['n_cells']} cells)")
    logger.info(f"GX0: vocab size {len(ctx['vocab'])}; nested labels "
                f"(longest-match-wins, audited): "
                f"{collisions['nested_longest_match_wins']}")

    # GX1 freeze (verify-not-rewrite: matrices are committed inputs)
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
        if existing != matrix:
            raise RuntimeError(
                f"committed matrix {path} differs from the rebuild; review "
                f"and re-commit deliberately -- matrices must never drift")
        logger.info(f"GX1: committed matrix verified (not rewritten): {path}")
    else:
        with open(path, "w") as f:
            json.dump(matrix, f, indent=2)
        logger.info(f"GX1: matrix frozen to {path}")
    validation = {
        "dataset": ds, "n_sets": matrix["n_sets"],
        "n_cells": matrix["n_cells"],
        "vocab_size": len(ctx["vocab"]),
        "nested_labels_longest_match_wins":
            collisions["nested_longest_match_wins"],
        "hard_collisions": collisions["hard_collisions"],
        "control_notes": notes,
        "pass_criteria": gx.PASS_CRITERIA,
        # deterministic bytes: NO timestamp -- this file is committed and
        # must not dirty the tracked worktree on re-runs
    }
    validation.update(extra_validation)
    return matrix, ctx, validation


def load_frozen(ds):
    if ds == "mllmu":
        # lives beside the G6.0 artifacts it is derived from, not in the
        # granularity manifest directory
        with open(g6m.G6_MATRIX_PATH) as f:
            return json.load(f)
    name = ("matrix_salmu.json" if ds == "salmu"
            else "matrix_celeba_numeric.json")
    with open(MANIFEST_DIR / name) as f:
        return json.load(f)


# ====================================================================== #
# GX1R: numeric baseline route h (association route for the 24 codes)
# ====================================================================== #
def ensure_numeric_route(args, ctx, out_base, ds="celeba_numeric"):
    """Train the baseline association route h (code -> own label).

    Shared by celeba_numeric and mllmu: both are association-level datasets
    with no image router, and the training pairs are built from ctx alone
    (code_of + baseline_alias_of), so nothing here is dataset-specific except
    the names written into the logs and the checkpoint directory.
    """
    tag = "num" if ds == "celeba_numeric" else "mll"
    ckpt = out_base / "route_h" / "adapter_final" / "adapter_model.safetensors"
    if ckpt.exists():
        logger.info(f"GX1R: {ds} baseline route h already present")
        return ckpt
    logger.info(f"GX1R: TRAIN {ds} baseline route h (3000/200/2e-5, "
                f"repeat 50, seed 17) over {len(ctx['identity_ids'])} codes")
    session = mx.ModelSession(args, f"e2c_gx_{tag}_h")
    try:
        mx.seed_everything(17)
        pairs = [{"prompt": rd.CODE_TO_ALIAS_PROMPT.format(
                      code=ctx["code_of"][i]),
                  "answer": ctx["baseline_alias_of"][i]}
                 for i in ctx["identity_ids"]]
        items = rv.build_supervised_items(session.adapter, session.processor,
                                          pairs, repeat=args.route_repeat)
        rv.train_supervised(f"gx_{tag}_route_h", session.adapter,
                            session.model,
                            session.processor, items, out_base / "route_h",
                            args.device, steps=args.route_steps,
                            warmup=args.route_warmup, lr=args.route_lr)
        # strict baseline check: every code must yield its own label
        acc = _strict_accuracy(session, ctx,
                               {i: ctx["baseline_alias_of"][i]
                                for i in ctx["identity_ids"]}, args)
        logger.info(f"GX1R: baseline route strict accuracy = {acc:.4f}")
        if acc < 1.0:
            logger.warning("GX1R: baseline route did NOT reach 1.0 -- "
                           "granularity cells require a perfect route")
    finally:
        session.release()
    return ckpt


def _strict_accuracy(session, ctx, expected_of, args):
    backend = session.backend()
    session.model.eval()
    ok = 0
    with torch.no_grad():
        for iid, exp in expected_of.items():
            prompt = rd.CODE_TO_ALIAS_PROMPT.format(code=ctx["code_of"][iid])
            gen = backend.generate(None, prompt,
                                   max_new_tokens=args.max_gen_tokens)
            parsed = rv.parse_recognized_label(gen.text.strip(),
                                               ctx["vocab"])
            ok += int(parsed == exp)
    return ok / max(len(expected_of), 1)


# ====================================================================== #
# GX2: oracle FAMILIES (G3.1 correction)
#   matched_finetune / loo_finetune: trained-baseline-h init (continued
#     fine-tuning references -- what G3 mistakenly called "retraining");
#   matched_retrain / loo_retrain: FRESH LoRA init (lora_A kaiming,
#     lora_B zeros) + route h's own protocol (3000/200/2e-5, uniform
#     repeat 50, seed 17); the matched reference additionally applies the
#     EDIT recipe's target oversampling (x5) and is reported as
#     matched_retrain_weighted.
# ====================================================================== #
def matched_pairs(ctx, entry):
    pairs = []
    for iid in ctx["identity_ids"]:
        pairs.append({"prompt": rd.CODE_TO_ALIAS_PROMPT.format(
                          code=ctx["code_of"][iid]),
                      "answer": expected_label(ctx, entry, iid)})
    return pairs


def retained_pairs(ctx, entry):
    return [{"prompt": rd.CODE_TO_ALIAS_PROMPT.format(
                 code=ctx["code_of"][i]),
             "answer": ctx["baseline_alias_of"][i]}
            for i in ctx["identity_ids"] if i not in entry["assignments"]]


def fresh_reinit_lora(named_params, seed):
    """In-place FRESH LoRA initialization replicating peft
    get_peft_model(init_lora_weights=True): lora_A ~ kaiming_uniform(a=sqrt5),
    lora_B = zeros, under the given seed.  Never reads any checkpoint --
    this is the property that distinguishes *_retrain from *_finetune.
    Fail-closed on layout and on the zero-B fresh-init signature."""
    import math as _math
    mx.seed_everything(seed)
    params = list(named_params)   # materialize ONCE (may be a generator)
    a_params = [(n, p) for n, p in params if "lora_A" in n]
    b_params = [(n, p) for n, p in params if "lora_B" in n]
    if not a_params or len(a_params) != len(b_params):
        raise RuntimeError(
            f"fresh_reinit_lora: unexpected LoRA layout "
            f"({len(a_params)} A / {len(b_params)} B)")
    with torch.no_grad():
        for _, p in a_params:
            torch.nn.init.kaiming_uniform_(p, a=_math.sqrt(5))
        for _, p in b_params:
            p.zero_()
    if any(torch.count_nonzero(p).item() != 0 for _, p in b_params):
        raise RuntimeError("fresh_reinit_lora: lora_B nonzero after re-init")
    return len(a_params), len(b_params)


def session_reset_fresh(session, seed=ORACLE_SEED):
    # mx.ModelSession exposes the live PeftModel as .model.  ``seed`` sets
    # the fresh-init RNG (GX2S oracle-seed sensitivity varies it; the main
    # matrix and GX2B use the default ORACLE_SEED).
    n_a, n_b = fresh_reinit_lora(
        list(session.model.named_parameters()), seed)
    logger.info("reset_fresh(): LoRA re-initialized fresh "
                "(A kaiming / B zeros, seed %d; %d/%d tensors)",
                seed, n_a, n_b)


def _fit_metrics_from_soft(soft, ctx, entry):
    """CPU fit proxy for cached oracles: argmax over the full vocab per
    identity vs the expected (transformed / retained) label.  ``soft`` is
    the _soft_all output keyed by IDENTITY ID."""
    ok = n = 0
    mass_min = 1.0
    for iid in ctx["identity_ids"]:
        summ = soft.get(iid)
        if not summ:
            continue
        dist = summ.get("probs", summ)
        exp = expected_label(ctx, entry, iid)
        ok += int(max(dist, key=dist.get) == exp)
        n += 1
        if iid in entry["assignments"]:
            mass_min = min(mass_min, summ.get(
                "candidate_mass",
                sum(dist.get(v, 0.0) for v in ctx["vocab"])))
    return {"fit_proxy_argmax": (ok / n) if n else None,
            "fit_proxy_n": n, "min_candidate_mass_proxy": mass_min}


def _write_oracle_results(out_dir, family, init, protocol, extra=None):
    res = {"family": family, "init": init, "protocol": protocol}
    if extra:
        res.update(extra)
    with open(out_dir / "oracle_results.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


def train_oracle_retrain(session, ds, ctx, entry, family, out_dir, args,
                         target_boost=None, oracle_seed=None):
    """Fresh-init retraining oracle under route h's own protocol.

    ``target_boost`` oversamples the transformed targets.  None ->
    RETRAIN_TARGET_BOOST (5, the matched_retrain_weighted reference).  The
    boost is the EDIT recipe's (mx.TARGET_BOOST), not route h's: rd.train_h
    applies a uniform repeat with no boost, so target_boost=1 -- what the
    GX2B balanced ablation passes, giving matched_retrain_balanced -- is the
    variant constructed like the route, the transformed mapping appearing
    exactly once per epoch like each retained mapping.
    ``oracle_seed`` sets the fresh-init + training RNG.  None ->
    ORACLE_SEED (17, the main matrix).  GX2S oracle-seed sensitivity
    passes 42 / 123.  steps/warmup/lr/repeat/LoRA-config are IDENTICAL.
    """
    is_matched = family.startswith("matched_retrain")
    boost = RETRAIN_TARGET_BOOST if target_boost is None else target_boost
    oseed = ORACLE_SEED if oracle_seed is None else oracle_seed
    out_dir.mkdir(parents=True, exist_ok=True)
    session_reset_fresh(session, oseed)
    mx.seed_everything(oseed)
    items = []
    if is_matched:
        assign = [{"prompt": rd.CODE_TO_ALIAS_PROMPT.format(
                       code=ctx["code_of"][i]),
                   "answer": expected_label(ctx, entry, i)}
                  for i in sorted(entry["assignments"])] * boost
        items += rv.build_supervised_items(
            session.adapter, session.processor, assign,
            repeat=RETRAIN_REPEAT)
    items += rv.build_supervised_items(
        session.adapter, session.processor, retained_pairs(ctx, entry),
        repeat=RETRAIN_REPEAT)
    rv.train_supervised(f"{family}_{entry['set_id']}", session.adapter,
                        session.model, session.processor, items, out_dir,
                        args.device, steps=RETRAIN_STEPS,
                        warmup=RETRAIN_WARMUP, lr=RETRAIN_LR)
    soft = _soft_all(session, ctx, args)
    with open(out_dir / "oracle_soft.json", "w") as f:
        json.dump(soft, f, indent=2)
    # G3.1 gate inputs.  matched_retrain(_balanced) must fit ALL transformed
    # AND retained mappings; loo_retrain never saw the targets, so its fit
    # check covers the RETAINED mapping only (targets are expected to
    # drift -- that is the deletion reference).
    if is_matched:
        fit_ids = list(ctx["identity_ids"])
    else:
        fit_ids = [i for i in ctx["identity_ids"]
                   if i not in entry["assignments"]]
    strict = _strict_accuracy(session, ctx,
                              {i: expected_label(ctx, entry, i)
                               for i in fit_ids}, args)
    masses = []
    for iid in entry["assignments"]:
        dist = soft[iid].get("probs", {})
        masses.append(sum(dist.get(l, 0.0) for l in ctx["vocab"]))
    mass = min(masses) if masses else None
    if is_matched:
        fit_ok = bool(strict == 1.0 and (mass is None or mass >= 0.99))
        scope = "all_transformed_and_retained"
    else:
        fit_ok = bool(strict == 1.0)
        scope = "retained_only(targets_never_seen)"
    _write_oracle_results(
        out_dir, family, "fresh_lora",
        {"steps": RETRAIN_STEPS, "warmup": RETRAIN_WARMUP,
         "lr": RETRAIN_LR, "repeat": RETRAIN_REPEAT,
         "target_boost": boost if is_matched else 0, "seed": oseed},
        {"set_id": entry["set_id"], "strict_fit_scope": scope,
         "strict_all_expected": strict,
         "min_candidate_mass": mass, "fit_ok": fit_ok})
    logger.info("GX2[%s]: %s trained (fresh init; seed=%d boost=%d "
                "strict=%.4f mass=%s fit_ok=%s)", entry["set_id"], family,
                oseed, boost, strict,
                f"{mass:.4f}" if mass is not None else "n/a", fit_ok)
    return {"mode": "trained_fresh", "target_boost": boost,
            "oracle_seed": oseed, "strict_all_expected": strict,
            "min_candidate_mass": mass, "fit_ok": fit_ok}


def loo_reuse_path(ds, entry):
    """Exact-match reuse of a suppression-matrix LOO oracle (salmu only).
    These were trained from the trained baseline h (finetune family)."""
    if ds != "salmu":
        return None
    targets = sorted(entry["assignments"])
    cand = SUPPRESSION_LOO_DIR / ("fs_" + "-".join(targets))
    ck = cand / "adapter_final" / "adapter_model.safetensors"
    soft = cand / "oracle_soft.json"
    if ck.exists() and soft.exists():
        return cand
    return None


FT_PROTOCOL = {"steps": 3000, "warmup": 200, "lr": 2e-5, "repeat": 50,
               "seed": ORACLE_SEED,
               "note": "route-protocol continued fine-tuning from the "
                       "trained baseline h (as executed in G3)"}


def run_oracles(args, ds, ctx, matrix, out_base):
    logger.info("=" * 60)
    logger.info(f"GX2: ORACLE FAMILIES ({ds}) -- finetune (baseline-h init) "
                "+ retrain (fresh LoRA init, route protocol)")
    logger.info("=" * 60)
    oracle_root = out_base / "oracles"
    oracle_root.mkdir(parents=True, exist_ok=True)
    args_o = argparse.Namespace(**vars(args))
    args_o.seed = ORACLE_SEED
    baseline = _baseline_ckpt(ds, out_base)
    results = {}
    session = None
    only = set(args.only_sets) if args.only_sets else None

    def _sess():
        nonlocal session
        if session is None:
            session = mx.ModelSession(args_o, f"e2c_gx_{ds}_oracle")
        return session

    for entry in matrix["sets"]:
        sid = entry["set_id"]
        if only and sid not in only:
            continue
        rec = {"set_id": sid}
        # ---- matched_finetune (G3 dirs: matched_<sid>) ----
        mdir = oracle_root / f"matched_{sid}"
        mckpt = mdir / "adapter_final" / "adapter_model.safetensors"
        msoft = mdir / "oracle_soft.json"
        if mckpt.exists() and msoft.exists():
            if not (mdir / "oracle_results.json").exists():
                with open(msoft) as f:
                    soft = json.load(f)
                _write_oracle_results(
                    mdir, "matched_finetune", "baseline_h", FT_PROTOCOL,
                    {"set_id": sid,
                     **_fit_metrics_from_soft(soft, ctx, entry)})
            rec["matched_finetune"] = {
                "dir": str(mdir), "mode": "cached",
                "sha256": rv.sha256_file(mckpt)}
        else:
            mdir.mkdir(parents=True, exist_ok=True)
            s = _sess()
            s.reset_to(baseline)
            mx.seed_everything(ORACLE_SEED)
            items = rv.build_supervised_items(
                s.adapter, s.processor, matched_pairs(ctx, entry),
                repeat=RETRAIN_REPEAT)
            rv.train_supervised(f"matched_ft_{sid}", s.adapter, s.model,
                                s.processor, items, mdir, args.device,
                                steps=RETRAIN_STEPS, warmup=RETRAIN_WARMUP,
                                lr=RETRAIN_LR)
            soft = _soft_all(s, ctx, args_o)
            with open(msoft, "w") as f:
                json.dump(soft, f, indent=2)
            _write_oracle_results(mdir, "matched_finetune", "baseline_h",
                                  FT_PROTOCOL, {"set_id": sid})
            rec["matched_finetune"] = {
                "dir": str(mdir), "mode": "trained",
                "sha256": rv.sha256_file(mckpt)}
            logger.info(f"GX2[{sid}]: matched_finetune trained")
        # ---- loo_finetune (G3 dirs: loo_<sid>; suppression reuse) ----
        reuse = loo_reuse_path(ds, entry)
        ldir = oracle_root / f"loo_{sid}"
        lckpt = ldir / "adapter_final" / "adapter_model.safetensors"
        lsoft = ldir / "oracle_soft.json"
        if reuse is not None:
            ldir.mkdir(parents=True, exist_ok=True)
            s = _sess()
            src_ckpt = reuse / "adapter_final" / "adapter_model.safetensors"
            if not lsoft.exists():
                s.reset_to(src_ckpt)
                mx.seed_everything(ORACLE_SEED)
                soft = _soft_all(s, ctx, args_o)
                with open(lsoft, "w") as f:
                    json.dump(soft, f, indent=2)
            (ldir / "adapter_final").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_ckpt, ldir / "adapter_final"
                         / "adapter_model.safetensors")
            if not (ldir / "oracle_results.json").exists():
                with open(lsoft) as f:
                    soft = json.load(f)
                _write_oracle_results(
                    ldir, "loo_finetune", "baseline_h", FT_PROTOCOL,
                    {"set_id": sid, "reused_from": str(reuse),
                     **_fit_metrics_from_soft(soft, ctx, entry)})
            rec["loo_finetune"] = {
                "dir": str(ldir), "mode": "reused_suppression_matrix",
                "source": str(reuse), "sha256": rv.sha256_file(src_ckpt)}
        elif lckpt.exists() and lsoft.exists():
            if not (ldir / "oracle_results.json").exists():
                with open(lsoft) as f:
                    soft = json.load(f)
                _write_oracle_results(
                    ldir, "loo_finetune", "baseline_h", FT_PROTOCOL,
                    {"set_id": sid, **_fit_metrics_from_soft(soft, ctx,
                                                             entry)})
            rec["loo_finetune"] = {"dir": str(ldir), "mode": "cached",
                                   "sha256": rv.sha256_file(lckpt)}
        else:
            # trained loo_finetune (salmu mixed sets AND numeric -- G3.1
            # adds the numeric LOO reference for symmetry)
            ldir.mkdir(parents=True, exist_ok=True)
            s = _sess()
            s.reset_to(baseline)
            mx.seed_everything(ORACLE_SEED)
            items = rv.build_supervised_items(
                s.adapter, s.processor, retained_pairs(ctx, entry),
                repeat=RETRAIN_REPEAT)
            rv.train_supervised(f"loo_ft_{sid}", s.adapter, s.model,
                                s.processor, items, ldir, args.device,
                                steps=RETRAIN_STEPS, warmup=RETRAIN_WARMUP,
                                lr=RETRAIN_LR)
            soft = _soft_all(s, ctx, args_o)
            with open(lsoft, "w") as f:
                json.dump(soft, f, indent=2)
            _write_oracle_results(ldir, "loo_finetune", "baseline_h",
                                  FT_PROTOCOL, {"set_id": sid})
            rec["loo_finetune"] = {"dir": str(ldir), "mode": "trained",
                                   "sha256": rv.sha256_file(lckpt)}
            logger.info(f"GX2[{sid}]: loo_finetune trained")
        # ---- retrain families (fresh init, route protocol) ----
        for family in ("matched_retrain", "loo_retrain"):
            rdir = oracle_root / f"{family}_{sid}"
            rckpt = (rdir / "adapter_final" / "adapter_model.safetensors")
            rres = rdir / "oracle_results.json"
            if rckpt.exists() and (rdir / "oracle_soft.json").exists() \
                    and rres.exists():
                with open(rres) as f:
                    rr = json.load(f)
                rec[family] = {"dir": str(rdir), "mode": "cached",
                               "sha256": rv.sha256_file(rckpt),
                               "fit_ok": rr.get("fit_ok"),
                               "strict_all_expected":
                                   rr.get("strict_all_expected"),
                               "min_candidate_mass":
                                   rr.get("min_candidate_mass")}
            else:
                s = _sess()
                rr = train_oracle_retrain(s, ds, ctx, entry, family, rdir,
                                          args_o)
                rec[family] = {"dir": str(rdir),
                               "sha256": rv.sha256_file(rckpt), **rr}
        results[sid] = rec
    if session is not None:
        session.release()
    with open(oracle_root / "oracles_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def _baseline_ckpt(ds, out_base):
    if ds == "salmu":
        return SALMU_ROUTE_H
    return out_base / "route_h" / "adapter_final" / "adapter_model.safetensors"


def _soft_all(session, ctx, args):
    out = {}
    for iid in ctx["identity_ids"]:
        probs = rv.full_sequence_label_probs(
            session.adapter, session.model, session.processor,
            ctx["code_of"][iid], ctx["vocab"], args.device)
        prob_by_label = {l: probs.get(l, {}).get("prob", 0.0)
                         for l in ctx["vocab"]}
        summary = rv.build_candidate_summary(prob_by_label, ctx["vocab"],
                                             gx.DELETED_LABEL)
        summary["log_probs"] = {l: probs.get(l, {}).get("log_prob", -1e9)
                                for l in ctx["vocab"]}
        out[iid] = summary
    return out


# ====================================================================== #
# GX3/GX4/GX5 (+GX6 inside cells): transformation-aware cell engine
# ====================================================================== #
def _classify(ctx, entry, iid, parsed):
    """Depth-aware classification of one identity's parsed output."""
    exp = expected_label(ctx, entry, iid)
    a = entry["assignments"].get(iid)
    if ctx["kind"] == "taxonomic":
        cls = gx.classify_taxonomic(parsed, exp, ctx["dag"])
        out = {"classification": cls}
        if a and a["operation"] == "taxonomic" and parsed is not None:
            out["requested_depth"] = a["target_depth"]
            # achieved depth: position of parsed in the identity chain
            chain = ctx["hierarchy_of"][iid]
            out["achieved_depth"] = (chain.index(parsed)
                                     if parsed in chain else None)
        return out
    prof = ctx["profiles"][iid]
    res = gx.classify_numeric(parsed, exp, ctx["schema"][prof["field"]],
                              prof["exact_value"], ctx["schema"])
    res["boundary_tags"] = prof["boundary_tags"]
    return res


def _hard_eval(session, ctx, entry, args):
    backend = session.backend()
    session.model.eval()
    preds = []
    with torch.no_grad():
        for iid in ctx["identity_ids"]:
            prompt = rd.CODE_TO_ALIAS_PROMPT.format(code=ctx["code_of"][iid])
            gen = backend.generate(None, prompt,
                                   max_new_tokens=args.max_gen_tokens)
            raw = gen.text.strip()
            labels = rv.recognized_labels_in(raw, ctx["vocab"])
            parsed = rv.parse_recognized_label(raw, ctx["vocab"])
            exp = expected_label(ctx, entry, iid)
            cls = _classify(ctx, entry, iid, parsed)
            preds.append({
                "identity_id": iid, "raw": raw, "parsed_label": parsed,
                "recognized_labels": labels,
                "multi_label_ambiguous": len(labels) > 1,
                "group": ("target" if iid in entry["assignments"]
                          else control_group(ctx, entry, iid)),
                "expected_post_edit": exp,
                "correct_post_edit": parsed == exp,
                "source_leaked": (iid in entry["assignments"]
                                  and entry["assignments"][iid]["source"]
                                  in labels),
                **cls,
            })
    return preds


def _e2e_replay_gx(session, ctx, entry, g_rows, args):
    """Cached frozen-g replay; outcomes use the TRANSFORMATION expectation.

    strict outcome   : parsed == expected label
    no-source-leak   : the original specific label appears nowhere
    Counts cover routed rows only (h-side health); unrouted rows are g
    routing failures, reported separately.
    """
    backend = session.backend()
    session.model.eval()
    rows = []
    with torch.no_grad():
        for gr in g_rows:
            iid = gr["identity_id"]
            base = {"identity_id": iid, "split": gr.get("split"),
                    "pred_code": gr.get("pred_code"),
                    "code_correct": bool(gr.get("code_correct")),
                    "g_routed": bool(gr.get("g_routed"))}
            if not base["g_routed"] or not base["pred_code"]:
                rows.append({**base, "h_raw_text": None, "pred_alias": None,
                             "recognized_labels": [], "outcome_ok": False,
                             "no_source_leak": False})
                continue
            prompt = rd.CODE_TO_ALIAS_PROMPT.format(code=base["pred_code"])
            gen = backend.generate(None, prompt,
                                   max_new_tokens=args.max_gen_tokens)
            raw = gen.text.strip()
            parsed = rv.parse_recognized_label(raw, ctx["vocab"])
            labels = rv.recognized_labels_in(raw, ctx["vocab"])
            exp = expected_label(ctx, entry, iid)
            src = (entry["assignments"][iid]["source"]
                   if iid in entry["assignments"] else None)
            rows.append({**base, "h_raw_text": raw, "pred_alias": parsed,
                         "recognized_labels": labels,
                         "multi_label_ambiguous": len(labels) > 1,
                         "outcome_ok": parsed == exp,
                         "no_source_leak": (src not in labels)
                         if src else parsed == exp})
    routed = [r for r in rows if r["g_routed"] and r["pred_code"]]
    cond = [r for r in rows if r["code_correct"]]
    return {
        "n_images": len(rows),
        "g_routed": len(routed),
        "g_routing_failures": len(rows) - len(routed),
        "g_code_correct": sum(r["code_correct"] for r in rows),
        "e2e_strict_conditional_acc": sum(r["outcome_ok"] for r in cond)
        / max(len(cond), 1),
        "e2e_strict_unconditional_acc": sum(r["outcome_ok"] for r in rows)
        / max(len(rows), 1),
        "no_source_leak_conditional_acc": sum(r["no_source_leak"]
                                              for r in cond)
        / max(len(cond), 1),
        "h_unparseable_outputs": sum(1 for r in routed
                                     if r["pred_alias"] is None),
        "h_multi_label_ambiguous_outputs": sum(
            1 for r in routed if r.get("multi_label_ambiguous")),
    }, rows


ORACLE_DIR_BY_FAMILY = {
    "matched_finetune": "matched_{sid}",       # G3 legacy dir names
    "loo_finetune": "loo_{sid}",
    "matched_retrain": "matched_retrain_{sid}",
    "loo_retrain": "loo_retrain_{sid}",
}


def load_oracle_soft(oracle_root, family, sid):
    p = (oracle_root / ORACLE_DIR_BY_FAMILY[family].format(sid=sid)
         / "oracle_soft.json")
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def oracle_family_distances(edit_summaries, entry, ctx, oracle_root, sid):
    """G3.1: gated distances to all FOUR oracle families + separate deltas.

    delta_ft_l2      = D(edit, loo_finetune)  - D(edit, matched_finetune)
    delta_retrain_l2 = D(edit, loo_retrain)   - D(edit, matched_retrain)
    Positive delta = the edit is CLOSER to the transformation-matched
    reference than to the leave-one-out (deletion) reference OF THE SAME
    FAMILY.  ``edit_summaries`` maps identity id -> candidate summary
    (as produced by _soft_all / rebuildable from soft_probs_full).
    Legacy keys: "matched"/"loo" alias the finetune families and
    "delta_oracle_l2" aliases delta_ft_l2 (G3 reports used them).
    """
    softs = {fam: load_oracle_soft(oracle_root, fam, sid)
             for fam in ORACLE_FAMILIES}
    out = {}
    for iid in ctx["identity_ids"]:
        rec = {}
        for fam in ORACLE_FAMILIES:
            o = softs[fam]
            if o is None or iid not in o:
                rec[fam] = {"distance": None, "reliable": False,
                            "reason": "oracle not available"}
                continue
            d, rel, why = rv.gated_distance(edit_summaries[iid], o[iid],
                                            "candidate", ctx["vocab"],
                                            MIN_CANDIDATE_MASS)
            rec[fam] = {"distance": d, "reliable": rel, "reason": why}
        rec["matched"] = rec["matched_finetune"]   # legacy alias
        rec["loo"] = rec["loo_finetune"]           # legacy alias

        def _l2(fam, _rec=rec):
            dd = _rec[fam]["distance"]
            return dd["l2"] if dd else None
        mf, lf = _l2("matched_finetune"), _l2("loo_finetune")
        mr, lr = _l2("matched_retrain"), _l2("loo_retrain")
        rec["delta_ft_l2"] = ((lf - mf) if mf is not None
                              and lf is not None else None)
        rec["delta_retrain_l2"] = ((lr - mr) if mr is not None
                                   and lr is not None else None)
        rec["delta_oracle_l2"] = rec["delta_ft_l2"]   # legacy alias
        rec["delta_reliable"] = rec["delta_ft_l2"] is not None
        rec["delta_retrain_reliable"] = rec["delta_retrain_l2"] is not None
        out[iid] = rec
    return out


def summaries_from_probs(probs_by_iid, ctx):
    """CPU rebuild of _soft_all-style summaries from stored probs."""
    return {iid: rv.build_candidate_summary(probs, ctx["vocab"],
                                            gx.DELETED_LABEL)
            for iid, probs in probs_by_iid.items()}


def reevaluate_oracles_cpu(ds, ctx, matrix, out_base):
    """GX2R (G3.1): recompute the oracle-family block of EXISTING cell
    results from STORED distributions.  Edited checkpoints are untouched;
    this is CPU-only.  Rewrites cell_results.json with 'oracle_families'
    (legacy 'dual_oracle' kept as the finetune-only G3 view)."""
    logger.info("=" * 60)
    logger.info(f"GX2R: CPU re-evaluation of stored cells against all "
                f"four oracle families ({ds})")
    logger.info("=" * 60)
    oracle_root = out_base / "oracles"
    n = 0
    for p in sorted((out_base / "cells").glob("*/seed_*/cell_results.json")):
        with open(p) as f:
            cell = json.load(f)
        probs = cell.get("soft_probs_full")
        if not probs:
            logger.warning("GX2R[%s]: no soft_probs_full; skipped",
                           cell["cell_id"])
            continue
        entry = next((e for e in matrix["sets"]
                      if e["set_id"] == cell["set_id"]), None)
        if entry is None:
            continue
        fam = oracle_family_distances(summaries_from_probs(probs, ctx),
                                      entry, ctx, oracle_root,
                                      cell["set_id"])
        cell["oracle_families"] = fam
        cell["oracle_block_version"] = "g3_1"
        cell["oracle_reevaluation_note"] = (
            "G3.1 CPU re-evaluation: distances recomputed from stored "
            "candidate-label distributions; edited checkpoint and its "
            "metrics unchanged; families distinguish baseline-h "
            "fine-tuning references from fresh-init retraining references")
        with open(p, "w") as f:
            json.dump(cell, f, indent=2)
        n += 1
        logger.info("GX2R[%s]: families recomputed", cell["cell_id"])
    logger.info(f"GX2R: {n} cell files re-evaluated")
    return n


# ====================================================================== #
# GX2B: matched_retrain_balanced ablation (target_boost=1)
#   Trains a fresh-init matched oracle WITHOUT the edit recipe's target
#   oversampling and compares D(E, matched_retrain_balanced),
#   D(E, matched_retrain_weighted) and D(E, loo_retrain) on the SAME edited
#   cells E.  The weighted reference is the existing matched_retrain dir.
# ====================================================================== #
def run_balanced_ablation(args, ds, ctx, matrix, out_base, set_ids):
    logger.info("=" * 60)
    logger.info(f"GX2B: BALANCED matched_retrain (boost=1) for {ds} "
                f"sets={set_ids}")
    logger.info("=" * 60)
    oracle_root = out_base / "oracles"
    oracle_root.mkdir(parents=True, exist_ok=True)
    args_o = argparse.Namespace(**vars(args))
    args_o.seed = ORACLE_SEED
    results = {}
    session = None
    for entry in matrix["sets"]:
        sid = entry["set_id"]
        if sid not in set_ids:
            continue
        bdir = oracle_root / f"{BALANCED_FAMILY}_{sid}"
        bckpt = bdir / "adapter_final" / "adapter_model.safetensors"
        bres = bdir / "oracle_results.json"
        if bckpt.exists() and (bdir / "oracle_soft.json").exists() \
                and bres.exists():
            with open(bres) as f:
                rr = json.load(f)
            proto = rr.get("protocol") or {}
            # A reused oracle was trained under SOME protocol, recorded in the
            # oracle_results.json beside it.  Fail closed on a mismatch:
            # silently reusing a reference trained at a different boost or seed
            # would report it as matched_retrain_balanced when it is not.
            want = {"steps": RETRAIN_STEPS, "warmup": RETRAIN_WARMUP,
                    "lr": RETRAIN_LR, "repeat": RETRAIN_REPEAT,
                    "target_boost": RETRAIN_TARGET_BOOST_BALANCED,
                    "seed": ORACLE_SEED}
            differs = {k: {"found": proto.get(k), "expected": v}
                       for k, v in want.items() if proto.get(k) != v}
            if differs:
                raise RuntimeError(
                    f"GX2B[{sid}]: the cached balanced oracle was NOT trained "
                    f"under the balanced recipe -- protocol differs: "
                    f"{differs}.  Reusing it would report a "
                    f"differently-trained reference as {BALANCED_LABEL}.")
            # The trained branch reports its own protocol, so the reused branch
            # reports the one on disk.  Dropping oracle_seed here made a CPU
            # re-run of GX2B say LESS about the reference than the GPU run that
            # trained it did, and lost the seed the ablation is scoped to.
            results[sid] = {"mode": "cached",
                            "target_boost": proto.get("target_boost"),
                            "oracle_seed": proto.get("seed"),
                            "protocol": proto,
                            "sha256": rv.sha256_file(bckpt),
                            "fit_ok": rr.get("fit_ok"),
                            "strict_all_expected":
                                rr.get("strict_all_expected"),
                            "min_candidate_mass":
                                rr.get("min_candidate_mass"),
                            "reused_not_trained_in_this_pass": (
                                "trained by an earlier committed pass; the "
                                "checkpoint sha256 and the protocol above are "
                                "read from the oracle_results.json beside it")}
            logger.info("GX2B[%s]: balanced oracle cached (seed=%s boost=%s)",
                        sid, proto.get("seed"), proto.get("target_boost"))
            continue
        if session is None:
            session = mx.ModelSession(args_o, f"e2c_gx_{ds}_bal")
        rr = train_oracle_retrain(session, ds, ctx, entry, BALANCED_FAMILY,
                                  bdir, args_o,
                                  target_boost=RETRAIN_TARGET_BOOST_BALANCED)
        results[sid] = {"sha256": rv.sha256_file(bckpt), **rr}
    if session is not None:
        session.release()
    with open(oracle_root / "balanced_ablation_oracles.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def compare_matched_boost(ds, ctx, matrix, out_base, set_ids):
    """CPU 3-way comparison of the edited cell E (edit seed 17, reused as-is)
    against matched_retrain_balanced, matched_retrain_weighted and
    loo_retrain, per transformation target.  Refusal controls are excluded
    from the promotion verdict but reported.  REFERENCE_NAMING declares why
    the emitted numeric keys still read Mx1/Mx5."""
    oracle_root = out_base / "oracles"

    def _load(dir_name):
        p = oracle_root / dir_name / "oracle_soft.json"
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    def _load_res(dir_name):
        p = oracle_root / dir_name / "oracle_results.json"
        if not p.exists():
            return {}
        with open(p) as f:
            return json.load(f)

    per_set = {}
    for entry in matrix["sets"]:
        sid = entry["set_id"]
        if sid not in set_ids:
            continue
        cellp = out_base / "cells" / sid / "seed_17" / "cell_results.json"
        if not cellp.exists():
            logger.warning("GX2B[%s]: no seed-17 edited cell; skipped", sid)
            continue
        with open(cellp) as f:
            cell = json.load(f)
        e_summ = summaries_from_probs(cell["soft_probs_full"], ctx)
        m1 = _load(f"{BALANCED_FAMILY}_{sid}")
        m5 = _load(f"{WEIGHTED_FAMILY}_{sid}")
        lo = _load(f"loo_retrain_{sid}")
        m1_fit = _load_res(f"{BALANCED_FAMILY}_{sid}")

        def _dist(o, t, _e=e_summ):
            if o is None or t not in o or t not in _e:
                return None
            d, rel, _ = rv.gated_distance(_e[t], o[t], "candidate",
                                          ctx["vocab"], MIN_CANDIDATE_MASS)
            return d["l2"] if (rel and d) else None

        trans = [t for t, a in entry["assignments"].items()
                 if a["operation"] != "refusal"]
        refusal = [t for t, a in entry["assignments"].items()
                   if a["operation"] == "refusal"]

        def _rows(id_list, _m1=m1, _m5=m5, _lo=lo, _ent=entry):
            rows = []
            for t in sorted(id_list):
                d1, d5, dl = _dist(_m1, t), _dist(_m5, t), _dist(_lo, t)
                rows.append({
                    "target": t,
                    "operation": _ent["assignments"][t]["operation"],
                    "d_E_Mx1": d1, "d_E_Mx5": d5, "d_E_L": dl,
                    "delta_x1": (dl - d1) if d1 is not None
                    and dl is not None else None,
                    "delta_x5": (dl - d5) if d5 is not None
                    and dl is not None else None,
                    "Mx1_closer_than_L": bool(
                        d1 is not None and dl is not None and d1 < dl),
                    "delta_x1_material": bool(
                        d1 is not None and dl is not None
                        and (dl - d1) >= DELTA_RETRAIN_MIN_MARGIN),
                    "Mx1_vs_Mx5": ("balanced_closer" if
                                   (d1 is not None and d5 is not None
                                    and d1 < d5) else
                                   "weighted_closer" if
                                   (d1 is not None and d5 is not None)
                                   else "n/a"),
                })
            return rows

        t_rows = _rows(trans)
        r_rows = _rows(refusal)
        set_promotes = bool(t_rows) and bool(m1_fit.get("fit_ok")) and all(
            r["Mx1_closer_than_L"] and r["delta_x1_material"]
            for r in t_rows)
        per_set[sid] = {
            "mode": entry["mode"],
            "balanced_fit": {
                "strict_all_expected": m1_fit.get("strict_all_expected"),
                "min_candidate_mass": m1_fit.get("min_candidate_mass"),
                "fit_ok": m1_fit.get("fit_ok"),
                "target_boost": m1_fit.get("protocol", {}).get(
                    "target_boost", RETRAIN_TARGET_BOOST_BALANCED),
            },
            "transformation_targets": t_rows,
            "refusal_controls": r_rows,
            "set_promotes": set_promotes,
            "failed_conditions": sorted(
                (["balanced_fit"] if not m1_fit.get("fit_ok") else [])
                + [f"{r['target']}:Mx1_not_closer_than_L"
                   for r in t_rows if not r["Mx1_closer_than_L"]]
                + [f"{r['target']}:delta_x1_below_margin"
                   for r in t_rows if not r["delta_x1_material"]]),
        }
        logger.info("GX2B[%s]: set_promotes=%s (trans targets=%d)",
                    sid, set_promotes, len(t_rows))
    promotes_all = bool(per_set) and all(
        v["set_promotes"] for v in per_set.values())
    return {
        "dataset": ds,
        "ablation": (f"{BALANCED_LABEL} (target_boost="
                     f"{RETRAIN_TARGET_BOOST_BALANCED}) vs {WEIGHTED_LABEL} "
                     f"(target_boost={RETRAIN_TARGET_BOOST}) vs loo_retrain; "
                     f"same edited cells E"),
        "distance_metric": "gated candidate-space L2 (unreliable -> null)",
        "margin_l2": DELTA_RETRAIN_MIN_MARGIN,
        "reference_naming": dict(REFERENCE_NAMING),
        "protocol_identical": {
            "steps": RETRAIN_STEPS, "warmup": RETRAIN_WARMUP,
            "lr": RETRAIN_LR, "repeat": RETRAIN_REPEAT,
            "seed": ORACLE_SEED,
            "only_difference": (f"target_boost {RETRAIN_TARGET_BOOST_BALANCED} "
                                f"({BALANCED_LABEL}) vs "
                                f"{RETRAIN_TARGET_BOOST} ({WEIGHTED_LABEL})")},
        "promotion_conditions": [
            (f"{BALANCED_LABEL} FITS transformed + retained mappings "
             "(strict 1.0, mass >= 0.99)"),
            (f"D(E, {BALANCED_LABEL}) < D(E, loo_retrain) for every "
             "transformation target"),
            (f"Delta_balanced = D(E, loo_retrain) - D(E, {BALANCED_LABEL}) "
             f">= margin ({DELTA_RETRAIN_MIN_MARGIN})"),
            ("conclusion agrees across ALL representative transformation "
             "types (refusal controls reported separately, not gating)")],
        "per_set": per_set,
        "promotes_all": promotes_all,
    }


def archive_balanced(ds, out_base, set_ids, commit):
    """Revision-pinned HF archive of the balanced matched_retrain oracles."""
    oracle_root = out_base / "oracles"
    rel = Path("releases") / f"e2c_gran_balanced_{ds}_{commit[:7]}"
    rel.mkdir(parents=True, exist_ok=True)
    entries = []
    for sid in set_ids:
        src = (oracle_root / f"{BALANCED_FAMILY}_{sid}" / "adapter_final"
               / "adapter_model.safetensors")
        if not src.exists():
            continue
        dest = rel / f"balanced_{sid}.safetensors"
        shutil.copy2(src, dest)
        entries.append({"kind": "oracle_matched_retrain_balanced",
                        "key": sid, "file": dest.name,
                        "sha256": rv.sha256_file(dest),
                        "bytes": dest.stat().st_size})
    hf_ok, hf_revision = False, None
    try:
        from huggingface_hub import HfApi, whoami
        whoami()
        api = HfApi()
        api.create_repo(HF_ARCHIVE_REPO, repo_type="model", exist_ok=True)
        url = api.upload_folder(
            folder_path=str(rel), repo_id=HF_ARCHIVE_REPO,
            path_in_repo=f"granularity_balanced_{ds}_{commit[:7]}",
            repo_type="model",
            commit_message=f"E2C-v3 balanced matched_retrain oracles ({ds}) "
                           f"@ {commit[:7]}")
        hf_ok = True
        hf_revision = url.rstrip("/").rsplit("/", 1)[-1] if url else None
        for e in entries:
            e["hf_revision"] = hf_revision
            e["hf_uri"] = (f"https://huggingface.co/{HF_ARCHIVE_REPO}/"
                           f"resolve/{hf_revision}/granularity_balanced_"
                           f"{ds}_{commit[:7]}/{e['file']}")
        logger.info("GX2B: uploaded %d balanced oracles, revision %s",
                    len(entries), hf_revision)
    except Exception as exc:
        logger.warning(f"GX2B: HF upload unavailable ({str(exc)[:120]})")
    with open(rel / "CHECKSUMS.txt", "w") as f:
        for e in entries:
            f.write(f"{e['sha256']}  {e['file']}\n")
    manifest = {"kind": "granularity_balanced_ablation_archive",
                "dataset": ds, "git_commit": commit, "release_dir": str(rel),
                "hf_repo": HF_ARCHIVE_REPO if hf_ok else None,
                "hf_upload_ok": hf_ok, "hf_revision": hf_revision,
                "n_files": len(entries),
                "uri_immutability_note": "resolve/<hf_commit_sha> pinned",
                "entries": entries}
    with open(rel / "archive_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def run_gx2b(args, ds, ctx, matrix, out_base, provenance, commit, t_start):
    """GX2B: balanced matched-retrain ablation (target_boost=1) on the
    representative transformation types, reusing the SAME edited cells E."""
    logger.info("=" * 60)
    logger.info(f"GX2B: BALANCED matched_retrain ABLATION ({ds})")
    logger.info("=" * 60)
    set_ids = (list(args.only_sets) if args.only_sets
               else list(BALANCED_REP_SETS.get(ds, [])))
    if not set_ids:
        raise RuntimeError(f"GX2B: no representative sets configured "
                           f"for {ds}")
    oracle_root = out_base / "oracles"
    by_id = {e["set_id"]: e for e in matrix["sets"]}
    for sid in set_ids:
        if sid not in by_id:
            raise RuntimeError(f"GX2B: {sid} not in the frozen matrix")
        need = {
            "E_seed17_cell": out_base / "cells" / sid / "seed_17"
            / "cell_results.json",
            f"{WEIGHTED_LABEL}": oracle_root
            / f"{WEIGHTED_FAMILY}_{sid}" / "oracle_soft.json",
            "loo_retrain": oracle_root
            / f"loo_retrain_{sid}" / "oracle_soft.json",
        }
        missing = sorted(k for k, p in need.items() if not p.exists())
        if missing:
            raise RuntimeError(
                f"GX2B[{sid}]: missing prerequisites {missing} (the "
                f"ablation reuses the existing edited cell, "
                f"{WEIGHTED_LABEL} and loo_retrain; run the main matrix "
                f"or G3.1 pilot first)")
    bal = run_balanced_ablation(args, ds, ctx, matrix, out_base, set_ids)
    cmp = compare_matched_boost(ds, ctx, matrix, out_base, set_ids)
    archive = {"release_dir": None, "hf_upload_ok": False,
               "hf_revision": None, "n_files": 0}
    if not args.smoke:
        archive = archive_balanced(ds, out_base, set_ids, commit)
    report = {
        "kind": "balanced_matched_retrain_ablation",
        "dataset": ds,
        "provenance": provenance,
        "representative_sets": set_ids,
        "balanced_oracles": bal,
        "comparison": cmp,
        "archive": {"release_dir": archive.get("release_dir"),
                    "hf_upload_ok": archive.get("hf_upload_ok"),
                    "hf_revision": archive.get("hf_revision"),
                    "n_files": archive.get("n_files", 0)},
        "promotes_all": cmp["promotes_all"],
        **_cumulative_elapsed(
            GRAN_ROOT / "reports" / f"balanced_ablation_{ds}.json",
            t_start, _device_name()),
    }
    rep_dir = GRAN_ROOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    with open(rep_dir / f"balanced_ablation_{ds}.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("GX2B: %s balanced ablation promotes_all=%s", ds,
                cmp["promotes_all"])
    logger.info("=" * 60)
    logger.info(f"GRANULARITY RUN ({ds}) PHASE GX2B COMPLETE "
                f"({round(time.time() - t_start, 1)}s)")
    logger.info("=" * 60)
    return 0


# ====================================================================== #
# GX2S: ORACLE-SEED SENSITIVITY
#   The full matrices vary the EDIT seed but hold the fresh-retrain
#   references at ORACLE_SEED=17.  GX2S trains matched_retrain +
#   loo_retrain at seeds 42/123 too, for the same representative sets, and
#   reports the paired (edit_seed x oracle_seed) Delta table.  The robust
#   claim rests on the SIGN/MARGIN of Delta, not exact distance equality.
# ====================================================================== #
def _load_soft(dir_path):
    p = Path(dir_path) / "oracle_soft.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _gated_l2(e_summ, o_summ, t, ctx):
    if o_summ is None or t not in o_summ or t not in e_summ:
        return None
    d, rel, _ = rv.gated_distance(e_summ[t], o_summ[t], "candidate",
                                  ctx["vocab"], MIN_CANDIDATE_MASS)
    return d["l2"] if (rel and d) else None


def run_oracle_seed_sensitivity(args, ds, ctx, matrix, out_base, set_ids,
                                oracle_seeds=ORACLE_SEEDS_SENSITIVITY):
    logger.info("=" * 60)
    logger.info(f"GX2S: ORACLE-SEED SENSITIVITY ({ds}) seeds={oracle_seeds} "
                f"sets={set_ids}")
    logger.info("=" * 60)
    oracle_root = out_base / "oracles"
    oracle_root.mkdir(parents=True, exist_ok=True)
    args_o = argparse.Namespace(**vars(args))
    args_o.seed = ORACLE_SEED
    session = None
    results = {}
    for entry in matrix["sets"]:
        sid = entry["set_id"]
        if sid not in set_ids:
            continue
        for oseed in oracle_seeds:
            for family in ("matched_retrain", "loo_retrain"):
                odir = oracle_root / retrain_oracle_dir(family, sid, oseed)
                ockpt = (odir / "adapter_final"
                         / "adapter_model.safetensors")
                ores = odir / "oracle_results.json"
                key = f"{family}_{sid}__oseed{oseed}"
                if ockpt.exists() and (odir / "oracle_soft.json").exists() \
                        and ores.exists():
                    with open(ores) as f:
                        rr = json.load(f)
                    results[key] = {
                        "mode": "cached", "oracle_seed": oseed,
                        "sha256": rv.sha256_file(ockpt),
                        "fit_ok": rr.get("fit_ok"),
                        "strict_all_expected": rr.get("strict_all_expected"),
                        "min_candidate_mass": rr.get("min_candidate_mass")}
                    logger.info("GX2S[%s %s oseed=%d]: cached", sid, family,
                                oseed)
                    continue
                if session is None:
                    session = mx.ModelSession(args_o, f"e2c_gx_{ds}_oseed")
                rr = train_oracle_retrain(session, ds, ctx, entry, family,
                                          odir, args_o, oracle_seed=oseed)
                results[key] = {"sha256": rv.sha256_file(ockpt), **rr}
    if session is not None:
        session.release()
    with open(oracle_root / "oracle_seed_sensitivity_oracles.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def compare_oracle_seed_sensitivity(ds, ctx, matrix, out_base, set_ids,
                                    oracle_seeds=ORACLE_SEEDS_SENSITIVITY):
    """Paired table: Set | Edit seed | Oracle seed | D_matched | D_LOO |
    Delta, over every available (edit_seed, oracle_seed) pair.  For a
    multi-target set the row reports the WORST-CASE transformation target
    (min Delta) plus the per-target detail and the mean."""
    oracle_root = out_base / "oracles"
    rows = []
    coverage = {}
    for entry in matrix["sets"]:
        sid = entry["set_id"]
        if sid not in set_ids:
            continue
        trans = sorted(t for t, a in entry["assignments"].items()
                       if a["operation"] != "refusal")
        cell_root = out_base / "cells" / sid
        edit_seeds = sorted(
            int(p.parent.name.split("_")[1])
            for p in cell_root.glob("seed_*/cell_results.json"))
        coverage[sid] = {"mode": entry["mode"], "edit_seeds": edit_seeds,
                         "transformation_targets": trans,
                         "pending_edit_seeds": [
                             s for s in gx.SEEDS_DEFAULT
                             if s not in edit_seeds]}
        for eseed in edit_seeds:
            cellp = cell_root / f"seed_{eseed}" / "cell_results.json"
            with open(cellp) as f:
                cell = json.load(f)
            e_summ = summaries_from_probs(cell["soft_probs_full"], ctx)
            for oseed in oracle_seeds:
                msoft = _load_soft(oracle_root / retrain_oracle_dir(
                    "matched_retrain", sid, oseed))
                lsoft = _load_soft(oracle_root / retrain_oracle_dir(
                    "loo_retrain", sid, oseed))
                base = {"set": sid, "mode": entry["mode"],
                        "edit_seed": eseed, "oracle_seed": oseed}
                if msoft is None or lsoft is None:
                    rows.append({**base, "status": "oracle_missing",
                                 "d_matched": None, "d_loo": None,
                                 "delta": None})
                    continue
                per_target = []
                for t in trans:
                    dm = _gated_l2(e_summ, msoft, t, ctx)
                    dl = _gated_l2(e_summ, lsoft, t, ctx)
                    per_target.append({
                        "target": t, "d_matched": dm, "d_loo": dl,
                        "delta": (dl - dm) if dm is not None
                        and dl is not None else None})
                valid = [r for r in per_target if r["delta"] is not None]
                if not valid:
                    rows.append({**base, "status": "unreliable_mass",
                                 "d_matched": None, "d_loo": None,
                                 "delta": None, "per_target": per_target})
                    continue
                worst = min(valid, key=lambda r: r["delta"])
                rows.append({
                    **base, "status": "ok",
                    "d_matched": worst["d_matched"],
                    "d_loo": worst["d_loo"],
                    "delta": worst["delta"],
                    "worst_target": worst["target"],
                    "mean_delta": round(statistics.fmean(
                        r["delta"] for r in valid), 8),
                    "all_targets_closer": all(r["delta"] > 0
                                              for r in valid),
                    "n_transformation_targets": len(trans),
                    "per_target": per_target})
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        gate = {"status": "no_valid_pairs", "n_pairs": 0}
    else:
        worst_delta = min(r["delta"] for r in ok_rows)
        gate_a = all(r["all_targets_closer"] for r in ok_rows)
        gate_b = worst_delta > 0
        gate = {
            "status": "evaluated",
            "n_pairs": len(ok_rows),
            "all_edit_seeds_closer_to_all_matched_oracle_seeds": gate_a,
            "worst_case_delta": round(worst_delta, 6),
            "worst_case_delta_positive": bool(gate_b),
            "worst_case_delta_ge_margin": bool(
                worst_delta >= DELTA_RETRAIN_MIN_MARGIN),
            "margin_l2": DELTA_RETRAIN_MIN_MARGIN,
            "passed": bool(gate_a or gate_b),
            "passed_strong": bool(gate_a and
                                  worst_delta >= DELTA_RETRAIN_MIN_MARGIN),
            "gate_definition": [
                ("A: every edit seed is closer to every matched-oracle seed "
                 "than to the corresponding LOO oracle (all targets)"),
                ("B: worst-case Delta = D_LOO - D_matched remains positive "
                 "(preferably >= 0.5)"),
                "passed = A OR B; passed_strong = A AND worst Delta >= 0.5"],
        }
    pending = {sid: c["pending_edit_seeds"] for sid, c in coverage.items()
               if c["pending_edit_seeds"]}
    return {
        "dataset": ds,
        "oracle_seeds": list(oracle_seeds),
        "distance_metric": "gated candidate-space L2 (unreliable -> null)",
        "coverage": coverage,
        "pending_edit_seeds": pending,
        "paired_table": [{"set": r["set"], "edit_seed": r["edit_seed"],
                          "oracle_seed": r["oracle_seed"],
                          "d_matched": r.get("d_matched"),
                          "d_loo": r.get("d_loo"), "delta": r.get("delta"),
                          "status": r["status"]} for r in rows],
        "rows": rows,
        "gate": gate,
    }


def archive_oracle_seed_sensitivity(ds, out_base, set_ids, commit,
                                    oracle_seeds=ORACLE_SEEDS_SENSITIVITY):
    oracle_root = out_base / "oracles"
    rel = Path("releases") / f"e2c_gran_oseed_{ds}_{commit[:7]}"
    rel.mkdir(parents=True, exist_ok=True)
    entries = []
    for sid in set_ids:
        for oseed in oracle_seeds:
            for family in ("matched_retrain", "loo_retrain"):
                src = (oracle_root / retrain_oracle_dir(family, sid, oseed)
                       / "adapter_final" / "adapter_model.safetensors")
                if not src.exists():
                    continue
                name = f"{family}_{sid}__oseed{oseed}.safetensors"
                dest = rel / name
                shutil.copy2(src, dest)
                entries.append({"kind": f"oracle_{family}_oseed", "key": sid,
                                "oracle_seed": oseed, "file": name,
                                "sha256": rv.sha256_file(dest),
                                "bytes": dest.stat().st_size})
    hf_ok, hf_revision = False, None
    try:
        from huggingface_hub import HfApi, whoami
        whoami()
        api = HfApi()
        api.create_repo(HF_ARCHIVE_REPO, repo_type="model", exist_ok=True)
        url = api.upload_folder(
            folder_path=str(rel), repo_id=HF_ARCHIVE_REPO,
            path_in_repo=f"granularity_oseed_{ds}_{commit[:7]}",
            repo_type="model",
            commit_message=f"E2C-v3 oracle-seed sensitivity oracles ({ds}) "
                           f"@ {commit[:7]}")
        hf_ok = True
        hf_revision = url.rstrip("/").rsplit("/", 1)[-1] if url else None
        for e in entries:
            e["hf_revision"] = hf_revision
            e["hf_uri"] = (f"https://huggingface.co/{HF_ARCHIVE_REPO}/"
                           f"resolve/{hf_revision}/granularity_oseed_{ds}_"
                           f"{commit[:7]}/{e['file']}")
        logger.info("GX2S: uploaded %d oracle-seed checkpoints, revision %s",
                    len(entries), hf_revision)
    except Exception as exc:
        logger.warning(f"GX2S: HF upload unavailable ({str(exc)[:120]})")
    with open(rel / "CHECKSUMS.txt", "w") as f:
        for e in entries:
            f.write(f"{e['sha256']}  {e['file']}\n")
    manifest = {"kind": "granularity_oracle_seed_sensitivity_archive",
                "dataset": ds, "git_commit": commit, "release_dir": str(rel),
                "hf_repo": HF_ARCHIVE_REPO if hf_ok else None,
                "hf_upload_ok": hf_ok, "hf_revision": hf_revision,
                "n_files": len(entries),
                "uri_immutability_note": "resolve/<hf_commit_sha> pinned",
                "entries": entries}
    with open(rel / "archive_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def run_gx2s(args, ds, ctx, matrix, out_base, provenance, commit, t_start):
    """GX2S: oracle-seed sensitivity ablation on the representative sets."""
    logger.info("=" * 60)
    logger.info(f"GX2S: ORACLE-SEED SENSITIVITY ABLATION ({ds})")
    logger.info("=" * 60)
    set_ids = (list(args.only_sets) if args.only_sets
               else list(BALANCED_REP_SETS.get(ds, [])))
    if not set_ids:
        raise RuntimeError(f"GX2S: no representative sets configured "
                           f"for {ds}")
    by_id = {e["set_id"]: e for e in matrix["sets"]}
    for sid in set_ids:
        if sid not in by_id:
            raise RuntimeError(f"GX2S: {sid} not in the frozen matrix")
    trained = run_oracle_seed_sensitivity(args, ds, ctx, matrix, out_base,
                                          set_ids)
    cmp = compare_oracle_seed_sensitivity(ds, ctx, matrix, out_base, set_ids)
    # human-readable paired table in the log
    logger.info("GX2S paired table (%s):", ds)
    logger.info("  %-30s %5s %6s %10s %10s %10s", "set", "edit", "oracle",
                "D_match", "D_LOO", "Delta")
    for r in cmp["paired_table"]:
        dm = f"{r['d_matched']:.5f}" if r["d_matched"] is not None else "n/a"
        dl = f"{r['d_loo']:.5f}" if r["d_loo"] is not None else "n/a"
        dd = f"{r['delta']:.5f}" if r["delta"] is not None else "n/a"
        logger.info("  %-30s %5s %6s %10s %10s %10s  %s", r["set"],
                    r["edit_seed"], r["oracle_seed"], dm, dl, dd,
                    r["status"])
    archive = {"release_dir": None, "hf_upload_ok": False,
               "hf_revision": None, "n_files": 0}
    if not args.smoke:
        archive = archive_oracle_seed_sensitivity(ds, out_base, set_ids,
                                                  commit)
    report = {
        "kind": "oracle_seed_sensitivity_ablation",
        "dataset": ds,
        "provenance": provenance,
        "representative_sets": set_ids,
        "oracle_seeds": cmp["oracle_seeds"],
        "trained_oracles": trained,
        "paired_table": cmp["paired_table"],
        "rows": cmp["rows"],
        "coverage": cmp["coverage"],
        "pending_edit_seeds": cmp["pending_edit_seeds"],
        "gate": cmp["gate"],
        "archive": {"release_dir": archive.get("release_dir"),
                    "hf_upload_ok": archive.get("hf_upload_ok"),
                    "hf_revision": archive.get("hf_revision"),
                    "n_files": archive.get("n_files", 0)},
        **_cumulative_elapsed(
            GRAN_ROOT / "reports" / f"oracle_seed_sensitivity_{ds}.json",
            t_start, _device_name()),
    }
    rep_dir = GRAN_ROOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    with open(rep_dir / f"oracle_seed_sensitivity_{ds}.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("GX2S: %s gate=%s (worst Delta=%s, all_closer=%s)", ds,
                cmp["gate"].get("passed"),
                cmp["gate"].get("worst_case_delta"),
                cmp["gate"].get(
                    "all_edit_seeds_closer_to_all_matched_oracle_seeds"))
    logger.info("=" * 60)
    logger.info(f"GRANULARITY RUN ({ds}) PHASE GX2S COMPLETE "
                f"({round(time.time() - t_start, 1)}s)")
    logger.info("=" * 60)
    return 0


def _pass_criteria(hard, soft, entry, ctx):
    """Frozen per-cell criteria (plan §12). Every criterion recorded."""
    crit = gx.PASS_CRITERIA
    targets = list(entry["assignments"])
    retained = [i for i in ctx["identity_ids"] if i not in targets]
    sib = [i for i in retained if control_group(ctx, entry, i) == "sibling"]
    hard_ok = {p["identity_id"]: p["correct_post_edit"] for p in hard}
    strict_acc = sum(hard_ok.values()) / len(hard_ok)
    retain_acc = (sum(hard_ok[i] for i in retained) / len(retained)
                  if retained else 1.0)
    sib_acc = (sum(hard_ok[i] for i in sib) / len(sib)) if sib else None
    min_p_desired = min(soft[t]["probs"].get(
        entry["assignments"][t]["target"], 0.0) for t in targets)
    max_p_source = max(soft[t]["probs"].get(
        entry["assignments"][t]["source"], 0.0) for t in targets)
    min_cand = min(v["candidate_mass"] for v in soft.values())
    wrong_branch = [p["identity_id"] for p in hard
                    if p.get("classification") == "wrong_branch"]
    wb_rate = len(wrong_branch) / len(hard)
    unparse = [p["identity_id"] for p in hard if p["parsed_label"] is None
               and not p["multi_label_ambiguous"]]
    multi = [p["identity_id"] for p in hard if p["multi_label_ambiguous"]]
    checks = {
        "strict_expected_accuracy==1.0": strict_acc == 1.0,
        "min_target_p_desired>=0.90": min_p_desired >= crit[
            "min_target_p_desired"],
        "max_target_p_source<=0.01": max_p_source <= crit[
            "max_target_p_source"],
        "min_candidate_mass>=0.99": min_cand >= crit["min_candidate_mass"],
        "retained_strict_accuracy==1.0": retain_acc == 1.0,
        "sibling_strict_accuracy==1.0": sib_acc is None or sib_acc == 1.0,
        "wrong_branch_rate==0": wb_rate == 0.0,
        "unparseable_outputs==0": len(unparse) == 0,
        "multi_label_outputs==0": len(multi) == 0,
    }
    return {
        "cell_pass": all(checks.values()),
        "failed_criteria": sorted(k for k, v in checks.items() if not v),
        "checks": checks,
        "strict_expected_accuracy": strict_acc,
        "retain_acc": retain_acc,
        "sibling_acc": sib_acc,
        "sibling_ids": sib,
        "min_target_p_desired": min_p_desired,
        "max_target_p_source": max_p_source,
        "min_candidate_mass": min_cand,
        "wrong_branch_ids": wrong_branch,
        "unparseable_ids": unparse,
        "multi_label_ids": multi,
    }


def run_cells(args, ds, ctx, matrix, out_base):
    modes = {"GX3": SINGLE_MODES, "GX4": SAME_DEPTH_MODES,
             "GX5": MIXED_MODES}[args.phase] if args.phase in \
        ("GX3", "GX4", "GX5") else \
        SINGLE_MODES | SAME_DEPTH_MODES | MIXED_MODES
    logger.info("=" * 60)
    logger.info(f"GX3-5: CELLS ({ds}) modes={sorted(modes)}")
    logger.info("=" * 60)
    cells_root = out_base / "cells"
    cells_root.mkdir(parents=True, exist_ok=True)
    oracle_root = out_base / "oracles"
    baseline = _baseline_ckpt(ds, out_base)
    g_rows = None
    if ds == "salmu":
        with open(SALMU_G_CACHE) as f:
            g_rows = json.load(f)["rows"]
    session = mx.ModelSession(args, f"e2c_gx_{ds}_cell")
    only = set(args.only_sets) if args.only_sets else None
    only_seeds = set(args.only_seeds) if args.only_seeds else None
    try:
        for entry in matrix["sets"]:
            if entry["mode"] not in modes:
                continue
            sid = entry["set_id"]
            if only and sid not in only:
                continue
            for seed in matrix["edit_seeds"]:
                if only_seeds and seed not in only_seeds:
                    continue
                cell_id = f"{sid}__seed{seed}"
                cell_dir = cells_root / sid / f"seed_{seed}"
                cell_dir.mkdir(parents=True, exist_ok=True)
                result_path = cell_dir / "cell_results.json"
                ckpt = (cell_dir / "edited_h" / "adapter_final"
                        / "adapter_model.safetensors")
                if result_path.exists() and ckpt.exists():
                    logger.info(f"[{cell_id}] cached, skipping")
                    continue
                logger.info("-" * 56)
                logger.info(f"[{cell_id}] mode={entry['mode']} seed={seed} "
                            f"targets={sorted(entry['assignments'])}")
                session.reset_to(baseline)
                mx.seed_everything(seed)
                args_c = argparse.Namespace(**vars(args))
                args_c.seed = seed
                pairs = [{"prompt": rd.CODE_TO_ALIAS_PROMPT.format(
                              code=ctx["code_of"][t]),
                          "answer": a["target"]}
                         for t, a in entry["assignments"].items()]
                retain_pairs = [{"prompt": rd.CODE_TO_ALIAS_PROMPT.format(
                                     code=ctx["code_of"][i]),
                                 "answer": ctx["baseline_alias_of"][i]}
                                for i in entry["retain_ids"]]
                items = (
                    rv.build_supervised_items(session.adapter,
                                              session.processor, pairs,
                                              repeat=args.ul_repeat
                                              * TARGET_BOOST)
                    + rv.build_supervised_items(session.adapter,
                                                session.processor,
                                                retain_pairs,
                                                repeat=args.ul_repeat
                                                * RETAIN_REPEAT))
                rv.train_supervised(f"gx_{cell_id}", session.adapter,
                                    session.model, session.processor, items,
                                    cell_dir / "edited_h", args.device,
                                    steps=args.ul_steps,
                                    warmup=args.ul_warmup, lr=args.ul_lr)
                mx.seed_everything(seed)
                hard = _hard_eval(session, ctx, entry, args)
                soft = _soft_all(session, ctx, args_c)
                criteria = _pass_criteria(hard, soft, entry, ctx)
                dist = oracle_family_distances(soft, entry, ctx,
                                               oracle_root, sid)
                e2e, e2e_rows = ({"level": "association-only",
                                  "reason": "no image router for numeric "
                                            "profiles (CelebA g redesign "
                                            "is separate)"}, [])
                if g_rows is not None:
                    e2e, e2e_rows = _e2e_replay_gx(session, ctx, entry,
                                                   g_rows, args)
                cell = {
                    "cell_id": cell_id, "dataset": ds, "set_id": sid,
                    "mode": entry["mode"], "seed": seed,
                    "assignments": entry["assignments"],
                    "controls": entry.get("controls", {}),
                    "control_notes": entry.get("control_notes", {}),
                    "checkpoint_sha256": rv.sha256_file(ckpt),
                    "hard_preds": hard,
                    "soft": {i: {"p_expected": soft[i]["probs"].get(
                                     expected_label(ctx, entry, i), 0.0),
                                 "p_baseline_alias": soft[i]["probs"].get(
                                     ctx["baseline_alias_of"][i], 0.0),
                                 "candidate_mass": soft[i]["candidate_mass"],
                                 "other_mass": soft[i]["other_mass"]}
                             for i in ctx["identity_ids"]},
                    "soft_probs_full": {i: soft[i]["probs"]
                                        for i in ctx["identity_ids"]},
                    "dual_oracle": dist,
                    "oracle_families": dist,
                    "oracle_block_version": "g3_1",
                    "e2e": e2e, "e2e_rows": e2e_rows,
                    "criteria": criteria,
                }
                with open(result_path, "w") as f:
                    json.dump(cell, f, indent=2)
                acc = criteria["strict_expected_accuracy"]
                logger.info(
                    f"[{cell_id}] pass={criteria['cell_pass']} "
                    f"strict={acc:.3f} retain={criteria['retain_acc']:.3f} "
                    f"sib={criteria['sibling_acc']} "
                    f"p_desired>={criteria['min_target_p_desired']:.4f} "
                    f"p_source<={criteria['max_target_p_source']:.2e}")
    finally:
        session.release()
    return cells_root


# ====================================================================== #
# GX7: aggregation (mean/std/min + per-seed + FAILURE COUNTS + boundary
# and sibling/cousin breakdowns) -- CPU-reconstructable from cell files
# ====================================================================== #
def load_all_cells(out_base):
    cells = []
    for p in sorted((out_base / "cells").glob("*/seed_*/cell_results.json")):
        with open(p) as f:
            cells.append(json.load(f))
    return cells


def _oracle_block(c):
    return c.get("oracle_families") or c.get("dual_oracle") or {}


def _fam_l2(c, t, fam):
    r = _oracle_block(c).get(t, {}).get(fam, {})
    if r.get("reliable") and r.get("distance"):
        return r["distance"]["l2"]
    return None


def _fam_delta(c, t, key):
    return _oracle_block(c).get(t, {}).get(key)


def aggregate_gx(ds, out_base, matrix, args=None):
    logger.info("=" * 60)
    logger.info(f"GX7: AGGREGATE ({ds})")
    logger.info("=" * 60)
    cells = load_all_cells(out_base)
    by_mode = {}
    for c in cells:
        by_mode.setdefault(c["mode"], []).append(c)

    def msd(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return {"mean": round(statistics.fmean(vals), 6),
                "std": round(statistics.stdev(vals), 6) if len(vals) > 1
                else 0.0,
                "min": round(min(vals), 8), "max": round(max(vals), 8),
                "n": len(vals)}

    def is_transformation(op):
        return op != "refusal"

    def oracle_metrics(cs, op_filter=None):
        """Per-family L2 + deltas over target identities, optionally
        restricted to transformations or refusal controls (G3.1: refusal
        targets must NOT contribute to the granularity headline)."""
        out = {}
        for fam in ORACLE_FAMILIES:
            out[f"l2_to_{fam}"] = msd(
                [_fam_l2(c, t, fam) for c in cs for t, a in
                 c["assignments"].items()
                 if op_filter is None or is_transformation(a["operation"])
                 == op_filter])
        for key in ("delta_ft_l2", "delta_retrain_l2"):
            out[key] = msd(
                [_fam_delta(c, t, key) for c in cs for t, a in
                 c["assignments"].items()
                 if op_filter is None or is_transformation(a["operation"])
                 == op_filter])
        return out

    per_mode = {}
    for mode, cs in sorted(by_mode.items()):
        cls_counts = {}
        for c in cs:
            for p in c["hard_preds"]:
                if p["group"] == "target":
                    cls_counts[p.get("classification", "?")] = \
                        cls_counts.get(p.get("classification", "?"), 0) + 1
        crit_names = cs[0]["criteria"]["checks"]
        fails = {k: sum(not c["criteria"]["checks"][k] for c in cs)
                 for k in crit_names}
        sib_ids_total = sum(len(c["criteria"]["sibling_ids"]) for c in cs)
        retain_l2 = {
            fam: msd([max((_fam_l2(c, r, fam) or 0.0)
                          for r in c["soft_probs_full"]
                          if r not in c["assignments"])
                      for c in cs
                      if any(_fam_l2(c, r, fam) is not None
                             for r in c["soft_probs_full"]
                             if r not in c["assignments"])])
            for fam in ORACLE_FAMILIES}
        per_mode[mode] = {
            "n_cells": len(cs),
            "cell_pass": sum(c["criteria"]["cell_pass"] for c in cs),
            "criterion_fail_counts": fails,
            "strict_expected_accuracy": msd(
                [c["criteria"]["strict_expected_accuracy"] for c in cs]),
            "retain_acc": msd([c["criteria"]["retain_acc"] for c in cs]),
            "sibling_acc": msd([c["criteria"]["sibling_acc"] for c in cs]),
            "sibling_coverage": {
                "cells_with_sibling_controls": sum(
                    1 for c in cs if c["criteria"]["sibling_ids"]),
                "n_cells": len(cs),
                "sibling_controls_total": sib_ids_total,
                "note": "sibling metric is null (never vacuous 1.0) for "
                        "targets whose sibling is co-targeted or unique in "
                        "the sampled hierarchy; coverage must always be "
                        "reported next to any sibling accuracy",
            },
            "min_target_p_desired": msd(
                [c["criteria"]["min_target_p_desired"] for c in cs]),
            "max_target_p_source": msd(
                [c["criteria"]["max_target_p_source"] for c in cs]),
            "min_candidate_mass": msd(
                [c["criteria"]["min_candidate_mass"] for c in cs]),
            "target_classification_counts": cls_counts,
            # G3.1: separate blocks -- refusal controls never contribute
            # to the granularity (transformation) headline
            "transformation_targets": oracle_metrics(cs, True),
            "refusal_controls": oracle_metrics(cs, False),
            "all_targets": oracle_metrics(cs, None),
            "retained_l2_max_by_family": retain_l2,
            "per_seed": {str(c["seed"]): {
                "cell_pass": c["criteria"]["cell_pass"],
                "strict": c["criteria"]["strict_expected_accuracy"],
                "failed_criteria": c["criteria"]["failed_criteria"],
                "checkpoint_sha256": c["checkpoint_sha256"],
            } for c in cs},
        }
        if cs[0]["e2e"].get("n_images") is not None:
            per_mode[mode]["e2e_strict_conditional"] = msd(
                [c["e2e"]["e2e_strict_conditional_acc"] for c in cs])
            per_mode[mode]["e2e_strict_unconditional"] = msd(
                [c["e2e"]["e2e_strict_unconditional_acc"] for c in cs])
            per_mode[mode]["no_source_leak_conditional"] = msd(
                [c["e2e"]["no_source_leak_conditional_acc"] for c in cs])
            per_mode[mode]["h_unparseable_total"] = sum(
                c["e2e"]["h_unparseable_outputs"] for c in cs)
            per_mode[mode]["h_multi_label_total"] = sum(
                c["e2e"]["h_multi_label_ambiguous_outputs"] for c in cs)
            per_mode[mode]["g_routing_failures_total"] = sum(
                c["e2e"]["g_routing_failures"] for c in cs)

    # numeric boundary analysis (per boundary tag: correct rate)
    boundary = {}
    if ds == "celeba_numeric":
        for c in cells:
            for p in c["hard_preds"]:
                if p["group"] != "target":
                    continue
                for tag in p.get("boundary_tags", []):
                    b = boundary.setdefault(tag, {"n": 0, "correct": 0})
                    b["n"] += 1
                    b["correct"] += int(p["correct_post_edit"])
        for b in boundary.values():
            b["accuracy"] = b["correct"] / max(b["n"], 1)

    # ---------------- G3.1 promotion gate (transformation targets) ----
    gate_per = []
    for c in cells:
        for t, a in c["assignments"].items():
            if a["operation"] == "refusal":
                continue
            r = _oracle_block(c).get(t, {})
            mr_block = r.get("matched_retrain") or {}
            loo_block = r.get("loo_retrain") or {}
            dmr = mr_block.get("distance")
            dlr = loo_block.get("distance")
            delta = r.get("delta_retrain_l2")
            fitp = (out_base / "oracles"
                    / f"matched_retrain_{c['set_id']}"
                    / "oracle_results.json")
            fit = {}
            if fitp.exists():
                with open(fitp) as f:
                    fit = json.load(f)
            gate_per.append({
                "cell_id": c["cell_id"], "target": t, "seed": c["seed"],
                "operation": a["operation"],
                "matched_retrain_fit_ok": fit.get("fit_ok"),
                "matched_retrain_strict": fit.get("strict_all_expected"),
                "matched_retrain_mass": fit.get("min_candidate_mass"),
                "matched_retrain_reliable": bool(mr_block.get("reliable")),
                "l2_to_matched_retrain": dmr["l2"] if dmr else None,
                "l2_to_loo_retrain": dlr["l2"] if dlr else None,
                "loo_retrain_reliable": bool(loo_block.get("reliable")),
                # The gate's own recorded reason for refusing the distance.
                # A null delta has a cause, and the cause is the difference
                # between "not established" and "established and failed".
                "why_no_loo_retrain_distance": loo_block.get("reason"),
                "delta_retrain_l2": delta,
                "delta_retrain_established": delta is not None,
                "delta_material": (delta is not None
                                   and delta >= DELTA_RETRAIN_MIN_MARGIN),
                "closer_to_matched_retrain": bool(
                    dmr and dlr and dmr["l2"] < dlr["l2"]),
            })
    have = [p for p in gate_per if p["delta_retrain_l2"] is not None]

    def _target_ok(p):
        return bool(p["matched_retrain_fit_ok"]
                    and (p["matched_retrain_mass"] or 0) >= 0.99
                    and p["delta_material"] and p["closer_to_matched_retrain"])

    #: A target whose delta is null was never compared: the frozen
    #: MIN_CANDIDATE_MASS gate refused to renormalize a distribution that left
    #: the recognized label set.  That is a COVERAGE limit, not a failed
    #: comparison, and reporting the two together is what makes a 117/120
    #: result read as three failures.
    not_established = [p for p in gate_per if p["delta_retrain_l2"] is None]
    #: The sets a declared sensitivity analysis would have to cover.  Derived
    #: from the rows, never hardcoded, so it cannot drift from the gate.
    uncovered_sets = sorted({p["cell_id"].rsplit("__seed", 1)[0]
                             for p in not_established})
    if not gate_per:
        g3_1_gate = {"status": "no_transformation_targets_evaluated"}
    elif not have:
        g3_1_gate = {"status": "retrain_oracles_not_yet_trained",
                     "n_transformation_targets": len(gate_per),
                     "per_target": gate_per}
    else:
        failed = [p for p in have if not _target_ok(p)]
        # Same truth value as before, decomposed so the report can say WHICH
        # of the two made it false.
        ok = not not_established and not failed
        g3_1_gate = {
            "status": "evaluated", "passed": bool(ok),
            "margin_l2": DELTA_RETRAIN_MIN_MARGIN,
            "n_transformation_targets": len(gate_per),
            "n_with_retrain_oracles": len(have),
            "conditions": [
                ("matched_retrain fits ALL transformed and retained "
                 "mappings (strict 1.0)"),
                "matched_retrain candidate mass >= 0.99",
                "delta_retrain_l2 >= margin (material separation)",
                "D(edit, matched_retrain) < D(edit, loo_retrain)",
                ("holds for EVERY transformation target, refusal controls "
                 "excluded")],
            "coverage": {
                "n_separation_established": len(have),
                "n_separation_not_established": len(not_established),
                "n_established_and_failing": len(failed),
                "gate_fails_on": ("conditions" if failed else
                                  "coverage" if not_established else
                                  "nothing"),
                "established_and_failing": [
                    {"cell_id": p["cell_id"], "target": p["target"],
                     "seed": p["seed"]}
                    for p in failed],
                "not_established": [
                    {"cell_id": p["cell_id"], "target": p["target"],
                     "seed": p["seed"],
                     "operation": p["operation"],
                     "l2_to_matched_retrain": p["l2_to_matched_retrain"],
                     "matched_retrain_reliable":
                         p["matched_retrain_reliable"],
                     "l2_to_loo_retrain": None,
                     "why_no_loo_retrain_distance":
                         p["why_no_loo_retrain_distance"]}
                    for p in not_established],
                "meaning": (
                    "'not established' is NOT 'failed'.  A null "
                    "delta_retrain_l2 means the frozen "
                    f"MIN_CANDIDATE_MASS={MIN_CANDIDATE_MASS} distance gate "
                    "refused to renormalize the loo_retrain distribution at "
                    "that target, so there is no LOO distance to compare "
                    "against and no comparison was made.  The gate above "
                    "requires EVERY transformation target, so an incomplete "
                    "coverage makes passed=false on its own even when not one "
                    "established comparison failed."),
                "not_a_pending_job": (
                    "the loo_retrain reference for these targets IS trained "
                    "and committed at the frozen protocol "
                    f"({RETRAIN_STEPS}/{RETRAIN_WARMUP}/{RETRAIN_LR}, "
                    f"repeat {RETRAIN_REPEAT}, seed {ORACLE_SEED}); "
                    "re-running it at that protocol reproduces the same "
                    "candidate-score support, so the gap is a property of the "
                    "frozen reference and not of an unfinished job"),
                "no_seed_substitution": (
                    "coverage may be widened only by a DECLARED sensitivity "
                    "analysis that reports every outcome; a favorable oracle "
                    "seed must never be substituted into this primary gate"),
                "sensitivity_analysis_available": (
                    # When coverage is complete there is nothing to widen, and
                    # interpolating an empty set list into "a sensitivity
                    # analysis over the uncovered set(s) []" is a sentence about
                    # nothing -- the same defect the prompt panel's attribution
                    # branches were rewritten to remove.
                    "coverage is COMPLETE: every transformation target has an "
                    "established matched-vs-LOO comparison, so no sensitivity "
                    "analysis is pending and none could widen this gate.  GX2S "
                    "(oracle seeds 42/123, separate report, never reads or "
                    "writes this gate) remains available as an ablation over "
                    "the representative sets if a seed-dependence question is "
                    "asked later."
                    if not uncovered_sets else
                    "GX2S trains matched_retrain and loo_retrain at ORACLE "
                    "seeds 42 and 123 and writes a SEPARATE report "
                    "(oracle_seed_sensitivity_<dataset>.json); it neither "
                    "reads nor writes the gate above, so it cannot change this "
                    "verdict.  A declared sensitivity analysis over the "
                    f"uncovered set(s) {uncovered_sets} is available at "
                    "'--phase GX2S --only-sets "
                    f"{' '.join(uncovered_sets + sorted(BALANCED_REP_SETS.get(ds, [])))}'"
                    ".  --only-sets MUST also name the already-covered "
                    "representatives: the GX2S report is rebuilt from the sets "
                    "given on the command line, so naming only the new set "
                    "would silently drop the existing pairs.  Every outcome "
                    "is reported, favorable or not."),
            },
            "per_target": gate_per}

    # ---------------- scoped claims (G3.1 wording discipline) ----------
    def _trans_msd(key):
        return msd([_fam_delta(c, t, key) if key.startswith("delta")
                    else _fam_l2(c, t, key)
                    for c in cells for t, a in c["assignments"].items()
                    if a["operation"] != "refusal"])
    sib_tot = sum(len(c["criteria"]["sibling_ids"]) for c in cells)
    sib_ok = 0
    for c in cells:
        hp = {p["identity_id"]: p for p in c["hard_preds"]}
        sib_ok += sum(hp[i]["correct_post_edit"]
                      for i in c["criteria"]["sibling_ids"] if i in hp)
    l2_mf, l2_mr = _trans_msd("matched_finetune"), _trans_msd(
        "matched_retrain")
    d_ft, d_rt = _trans_msd("delta_ft_l2"), _trans_msd("delta_retrain_l2")

    # ---------------- the TWO conclusions, kept apart ------------------ #
    # A complete behavioral matrix and an incomplete oracle comparison are
    # different statements with different denominators.  Reporting them as one
    # made 84/84 cells read as an unfinished experiment, and made 117/120
    # comparisons read as three failed separations.  Neither is what was
    # measured, so each gets its own claim with its own coverage.
    n_pass = sum(c["criteria"]["cell_pass"] for c in cells)
    n_expected = matrix["n_cells"]
    seeds_seen = sorted({c["seed"] for c in cells})
    if n_pass == len(cells) == n_expected:
        behavioral_claim = (
            f"All {n_pass} cells of the full {n_expected}-cell {ds} matrix "
            f"satisfy the frozen behavioral criteria (cell_pass), at edit "
            f"seeds {seeds_seen}.  This is the BEHAVIORAL conclusion -- "
            f"transformation accuracy, retained accuracy, the sibling and "
            f"retained controls, and source leakage -- and it is complete.  "
            f"It is reported separately from the oracle-separation coverage "
            f"below, which asks a different question with a different "
            f"denominator.")
    elif n_pass == len(cells):
        behavioral_claim = (
            f"All {n_pass} evaluated cells satisfy the frozen behavioral "
            f"criteria, but the matrix is INCOMPLETE: {len(cells)} of "
            f"{n_expected} cells, at edit seeds {seeds_seen} of "
            f"{sorted(matrix['edit_seeds'])}.  No behavioral conclusion about "
            f"the full matrix may be drawn from a partial one.")
    else:
        behavioral_claim = (
            f"{n_pass} of {len(cells)} evaluated cells satisfy the frozen "
            f"behavioral criteria ({n_expected} expected when complete), so "
            f"{len(cells) - n_pass} cell(s) FAIL and the behavioral "
            f"conclusion does not hold.")

    cov = g3_1_gate.get("coverage")
    n_tot = g3_1_gate.get("n_transformation_targets")
    if cov:
        n_est = cov["n_separation_established"]
        n_ne = cov["n_separation_not_established"]
        n_bad = cov["n_established_and_failing"]
        # Group the uncovered rows by (set, target) so three edit seeds of one
        # target read as ONE target with no LOO support, not three failures.
        by_target = {}
        for row in cov["not_established"]:
            by_target.setdefault(
                (row["cell_id"].rsplit("__seed", 1)[0], row["target"]),
                []).append(row["seed"])
        uncovered_detail = "; ".join(
            f"target {t} in {sid} at edit seed(s) {sorted(seeds)}"
            for (sid, t), seeds in sorted(by_target.items())) or "none"
        causes = sorted({r["why_no_loo_retrain_distance"] or "not recorded"
                         for r in cov["not_established"]})
        # Quoted verbatim and counted, because two causes that differ only in a
        # recorded digit read as a stutter unless the report says they are two.
        cause_text = (f"{len(causes)} distinct cause(s) recorded by the frozen "
                      f"distance gate: " + "; ".join(causes)) \
            if causes else "no cause recorded"
    else:
        n_est = n_ne = n_bad = None
        uncovered_detail = cause_text = None

    if g3_1_gate.get("passed"):
        oracle_separation_claim = (
            f"Matched-vs-LOO retraining separation is established for "
            f"{n_est} of {n_tot} target-seed comparisons -- every "
            f"transformation target evaluated -- and each is material "
            f"(Delta_retrain >= {DELTA_RETRAIN_MIN_MARGIN}) with the edit "
            f"closer to {WEIGHTED_LABEL} than to loo_retrain.")
    elif not cov:
        oracle_separation_claim = (
            "Matched-vs-LOO retraining separation is not evaluated "
            f"(status={g3_1_gate.get('status')}).")
    else:
        oracle_separation_claim = (
            f"Matched-vs-LOO retraining separation is established for "
            f"{n_est} of {n_tot} target-seed comparisons"
            + (f", and every one of those {n_est} is material "
               f"(Delta_retrain >= {DELTA_RETRAIN_MIN_MARGIN}) with the edit "
               f"closer to {WEIGHTED_LABEL} than to loo_retrain"
               if not n_bad else
               f", but {n_bad} of the established comparisons FAIL the "
               f"conditions")
            + f".  It is NOT established for the remaining {n_ne}: "
            f"{uncovered_detail}.  {cause_text}.  The G3.1 gate reports "
            f"passed=false because it requires EVERY transformation target"
            + (" -- it did not fail on any measured comparison"
               if not n_bad else "")
            + ".  Retraining the reference until it happens to pass is not a "
              "repair: that reference is already trained and committed at the "
              "frozen protocol, and re-running the same protocol reproduces "
              "the same candidate-score support.")

    claims = {
        "behavioral_claim": behavioral_claim,
        "oracle_separation_claim": oracle_separation_claim,
        "supported_finetune_reference": (
            "Over the evaluated code prompts and candidate-label space, the "
            "edit is extremely close to a transformation-matched "
            "CONTINUED-FINE-TUNING reference (baseline-h init) and far from "
            "the corresponding leave-one-out fine-tuning reference."
            + (f" L2-to-matched_finetune mean={l2_mf['mean']} (n={l2_mf['n']}),"
               f" Delta_FT mean={d_ft['mean']} (min={d_ft['min']})."
               if l2_mf and d_ft else "")),
        "retraining_claim": (
            ("Additionally, the edit is materially closer to a FRESH-init "
             "transformation-matched RETRAINING reference than to the "
             "fresh-init leave-one-out retraining reference "
             + (f"(Delta_retrain mean={d_rt['mean']}, min={d_rt['min']}, "
                f"margin>={DELTA_RETRAIN_MIN_MARGIN}; L2-to-matched_retrain "
                f"mean={l2_mr['mean']})." if d_rt and l2_mr else "")
             + " 'Close to policy-matched retraining' is supported, scoped "
               "to the evaluated code prompts and candidate-label space.")
            if g3_1_gate.get("passed") else
            ("'Close to policy-matched retraining' is supported WHERE IT IS "
             f"ESTABLISHED -- {n_est} of {n_tot} target-seed comparisons, "
             "every one material and every one closer to "
             f"{WEIGHTED_LABEL} than to loo_retrain -- but it is NOT "
             f"supported for the full matrix, because separation is not "
             f"established for {n_ne} target-seed comparison(s) "
             f"({uncovered_detail}).  The frozen G3.1 gate requires every "
             "transformation target, so it reports passed=false on COVERAGE "
             "and not on any measured comparison.  Scoped to the evaluated "
             "code prompts and candidate-label space.")
            if cov and not n_bad else
            "'Close to policy-matched retraining' is NOT yet supported: "
            "retrain-family oracles are missing or the G3.1 gate has not "
            f"passed (status={g3_1_gate.get('status')})."),
        "equivalence_scope": (
            "A tiny L2 between two nearly one-hot distributions over the "
            "evaluated candidate set does NOT establish global functional "
            "or parameter equivalence; every proximity claim is scoped to "
            "the evaluated code prompts and candidate-label space."),
        "sibling_coverage": (
            f"The available sibling control(s) passed {sib_ok}/{sib_tot}; "
            "all other retained controls passed. Targets without a retained "
            "sibling (unique branch or co-targeted) have a null sibling "
            "metric, never a vacuous 1.0."),
        "refusal_separation": (
            "Refusal controls are reported in a separate block and never "
            "contribute to the granularity (transformation) headline."),
    }
    if ds == "celeba_numeric":
        claims["numeric_boundary"] = (
            "Boundary wording: report lower/interior boundary targets that "
            "passed, including any member of a frozen adjacent-boundary "
            "pair; 'adjacent-boundary pair passed' is reserved for the full "
            "matrix once BOTH members (e.g. 19 and 20) are jointly "
            "evaluated. The pilot edits values individually and does not "
            "jointly edit and test both sides of a boundary.")

    executed_seeds = sorted({c["seed"] for c in cells})
    configured_seeds = list(matrix["edit_seeds"])
    if len(cells) == 0:
        run_stage = "no_cells"
    elif (len(cells) == matrix["n_cells"]
          and executed_seeds == sorted(configured_seeds)):
        run_stage = "full_matrix"
    elif len(executed_seeds) == 1:
        run_stage = "G3_single_seed_pilot(+G3.1_reevaluation)"
    else:
        run_stage = "partial_matrix"

    summary = {
        "dataset": ds,
        "configured_full_seeds": configured_seeds,
        "executed_seeds": executed_seeds,
        "run_stage": run_stage,
        "edit_seeds": configured_seeds,      # legacy alias
        "n_sets": matrix["n_sets"],
        "cells_evaluated": len(cells),
        "full_matrix_cells_expected": matrix["n_cells"],
        "cells_expected": matrix["n_cells"],  # legacy alias
        "cell_pass_total": sum(c["criteria"]["cell_pass"] for c in cells),
        "oracle_family_definitions": {
            "matched_finetune": "trained baseline h init; transformed full "
                                "mapping (continued fine-tuning reference)",
            "loo_finetune": "trained baseline h init; retained mapping "
                            "only (deletion-as-fine-tuning reference)",
            "matched_retrain": (f"FRESH base + fresh LoRA; transformed full "
                                f"mapping; route h's own protocol "
                                f"({RETRAIN_STEPS}/{RETRAIN_WARMUP}/"
                                f"{RETRAIN_LR}, uniform repeat "
                                f"{RETRAIN_REPEAT}, seed {ORACLE_SEED}) PLUS "
                                f"the EDIT recipe's target oversampling "
                                f"(target_boost={RETRAIN_TARGET_BOOST}); "
                                f"reported as {WEIGHTED_LABEL}.  The x5 is "
                                f"mx.TARGET_BOOST from the suppression recipe, "
                                f"NOT route h's recipe: rd.train_h builds "
                                f"every mapping at a uniform repeat with no "
                                f"target boost"),
            "matched_retrain_balanced": (
                f"as matched_retrain but target_boost="
                f"{RETRAIN_TARGET_BOOST_BALANCED} (GX2B): the transformed "
                f"mapping appears once per epoch like each retained mapping, "
                f"which is how route h itself is trained"),
            "loo_retrain": "FRESH base + fresh LoRA; retained mapping "
                           "only; same protocol, no transformed target to "
                           "weight",
            "delta_ft_l2": "D(edit, loo_finetune) - D(edit, "
                           "matched_finetune)",
            "delta_retrain_l2": "D(edit, loo_retrain) - D(edit, "
                                "matched_retrain)",
        },
        "correction": {
            "corrects_commit": "a1df9be",
            "issue": "G3 described edits as 'policy-matched-retraining "
                     "equivalent', but its matched/LOO oracles were "
                     "continued fine-tuning from the trained baseline h, "
                     "not fresh retraining; aggregation also let the "
                     "refusal control contribute to the granularity "
                     "headline and reported sibling accuracy without "
                     "coverage.",
            "repair": "G3.1: four named oracle families, separate "
                      "Delta_FT / Delta_retrain, taxonomic-vs-refusal "
                      "blocks, sibling coverage counts, "
                      "configured-vs-executed seed fields, numeric LOO "
                      "reference, CPU re-evaluation of existing edited "
                      "checkpoints (weights untouched).",
        },
        "reference_naming": dict(REFERENCE_NAMING),
        "g3_1_gate": g3_1_gate,
        "claims": claims,
        "scope": {
            "multi_seed_meaning": "stable across three EDIT-TRAINING seeds; "
                                  "router, baseline h, cached routing "
                                  "predictions, set selection and oracle "
                                  "seed are fixed",
            "transformations": "granularity (taxonomic/numeric) + refusal "
                               "controls; see per-mode breakdown",
        },
        "per_mode": per_mode,
        "boundary_analysis": boundary,
    }
    with open(out_base / "granularity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"GX7: {summary['cell_pass_total']}/{len(cells)} cells pass "
                f"(expected {matrix['n_cells']} when complete); G3.1 gate: "
                f"{g3_1_gate.get('status')}"
                + (f" passed={g3_1_gate['passed']}"
                   if "passed" in g3_1_gate else ""))
    return summary


def archive_gx(args, ds, out_base, matrix, oracle_results):
    logger.info("=" * 60)
    logger.info(f"GX7: ARCHIVE ({ds}) -- SHA-256 + revision-pinned HF URIs")
    logger.info("=" * 60)
    commit = rv.git_commit_sha()
    rel = Path("releases") / f"e2c_granularity_{ds}_{commit[:7]}"
    rel.mkdir(parents=True, exist_ok=True)
    entries = []

    def _add(src, rel_name, kind, key):
        src = Path(src)
        if not src.exists():
            logger.warning(f"archive: missing {src}")
            return
        dest = rel / rel_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        entries.append({"kind": kind, "key": key, "file": rel_name,
                        "sha256": rv.sha256_file(dest),
                        "bytes": dest.stat().st_size,
                        "source_path": str(src.resolve()),
                        "local_uri": dest.resolve().as_uri()})

    if ds in ("celeba_numeric", "mllmu"):
        _add(out_base / "route_h" / "adapter_final"
             / "adapter_model.safetensors",
             "route/route_h.safetensors", "route", "route_h")
    else:
        _add(SALMU_ROUTE_H, "route/h_C_to_Y.safetensors", "route",
             "salmu_pilot_h")
    legacy_name = {"matched_finetune": "matched", "loo_finetune": "loo"}
    for sid, rec in oracle_results.items():
        for fam in ORACLE_FAMILIES:
            fr = rec.get(fam) or rec.get(legacy_name.get(fam, ""), {})
            if fr.get("sha256") or fr.get("mode") in (
                    "trained", "cached", "trained_fresh",
                    "reused_suppression_matrix"):
                _add(out_base / "oracles"
                     / ORACLE_DIR_BY_FAMILY[fam].format(sid=sid)
                     / "adapter_final" / "adapter_model.safetensors",
                     f"oracles/{fam}_{sid}.safetensors", f"oracle_{fam}",
                     sid)
    for c in load_all_cells(out_base):
        _add(out_base / "cells" / c["set_id"] / f"seed_{c['seed']}"
             / "edited_h" / "adapter_final" / "adapter_model.safetensors",
             f"cells/{c['set_id']}/seed_{c['seed']}.safetensors", "cell",
             c["cell_id"])

    hf_ok, hf_repo, hf_revision = False, None, None
    try:
        from huggingface_hub import HfApi, whoami
        whoami()
        api = HfApi()
        api.create_repo(HF_ARCHIVE_REPO, repo_type="model", exist_ok=True)
        url = api.upload_folder(
            folder_path=str(rel), repo_id=HF_ARCHIVE_REPO,
            path_in_repo=f"granularity_{ds}_{commit[:7]}", repo_type="model",
            commit_message=f"E2C-v3 granularity checkpoints ({ds}) "
                           f"@ {commit[:7]}")
        hf_ok, hf_repo = True, HF_ARCHIVE_REPO
        hf_revision = url.rstrip("/").rsplit("/", 1)[-1] if url else None
        for e in entries:
            e["hf_revision"] = hf_revision
            e["hf_uri"] = (f"https://huggingface.co/{HF_ARCHIVE_REPO}/"
                           f"resolve/{hf_revision}/granularity_{ds}_"
                           f"{commit[:7]}/{e['file']}")
        logger.info(f"GX7: uploaded {len(entries)} checkpoints, revision "
                    f"{hf_revision}")
    except Exception as exc:
        logger.warning(f"GX7: HF upload unavailable ({str(exc)[:120]})")
    with open(rel / "CHECKSUMS.txt", "w") as f:
        for e in entries:
            f.write(f"{e['sha256']}  {e['file']}\n")
    manifest = {"kind": "granularity_archive", "dataset": ds,
                "git_commit": commit, "release_dir": str(rel),
                "hf_repo": hf_repo, "hf_upload_ok": hf_ok,
                "hf_revision": hf_revision, "n_files": len(entries),
                "uri_immutability_note": "resolve/<hf_commit_sha> pinned; "
                                         "resolve/main is mutable",
                "entries": entries}
    with open(rel / "archive_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def run_manifest_gx(args, ds, out_base, matrix, oracle_results, summary,
                    archive, t_start, provenance):
    if ds == "mllmu":
        # The MLLMU pilot's inputs live beside the G6.0 artifacts they are
        # derived from, and the hierarchy and target selection are inputs in
        # their own right: the pilot is licensed by them, so a run manifest
        # that did not bind their bytes could not show which audit it ran
        # under.
        inputs = {
            "matrix_manifest": rv.sha256_file(g6m.G6_MATRIX_PATH),
            "g6_pilot_manifest": rv.sha256_file(g6m.G6_MANIFEST_PATH),
            "g6_hierarchy": rv.sha256_file(g6m.HIERARCHY_PATH),
            "g6_target_selection": rv.sha256_file(g6m.SELECTION_PATH),
            "baseline_h": rv.sha256_file(_baseline_ckpt(ds, out_base)),
        }
    else:
        inputs = {
            "matrix_manifest": rv.sha256_file(
                MANIFEST_DIR / ("matrix_salmu.json" if ds == "salmu"
                                else "matrix_celeba_numeric.json")),
            "baseline_h": rv.sha256_file(_baseline_ckpt(ds, out_base)),
        }
    if ds == "salmu":
        inputs["dataset_manifest"] = rv.sha256_file(SALMU_MANIFEST)
        inputs["g_cache_e2e_rows"] = rv.sha256_file(SALMU_G_CACHE)
    elif ds == "celeba_numeric":
        inputs["numeric_manifest"] = rv.sha256_file(
            MANIFEST_DIR / "numeric_manifest.json")
    ckpts = {}
    legacy_name = {"matched_finetune": "matched", "loo_finetune": "loo"}
    for sid, rec in oracle_results.items():
        for fam in ORACLE_FAMILIES:
            fr = rec.get(fam) or rec.get(legacy_name.get(fam, ""), {})
            if fr.get("sha256"):
                ckpts[f"oracle_{fam}_{sid}"] = fr["sha256"]
    all_cells = load_all_cells(out_base)
    for c in all_cells:
        ckpts[f"cell_{c['cell_id']}"] = c["checkpoint_sha256"]

    g6_block = {}
    if ds == "mllmu":
        # The four pilot gates ARE the decision this run exists to make, so
        # they are evaluated here and bound into the manifest rather than left
        # to a separate pass that could be forgotten.  Written into the run
        # directory as well, so the gate verdict is hashed with the outputs it
        # was computed from.
        gate_cells = g6m.cells_for_gates(all_cells)
        g6_block = {
            "gates": g6m.evaluate_gates(gate_cells, matrix),
            "sets": g6m.per_set_provenance(matrix, gate_cells),
            "provenance": g6m.g6_provenance(out_base, args),
            "n_cells_adapted": len(gate_cells),
        }
        with open(out_base / "g6_pilot_gates.json", "w") as f:
            json.dump(g6_block, f, indent=2)
        logger.info(
            f"GX7/G6: pilot gates passed={g6_block['gates']['passed']} "
            f"failed={g6_block['gates']['failed_gates']} "
            f"proceed_to_full_matrix="
            f"{g6_block['gates']['proceed_to_full_matrix']}")

    manifest = {
        "experiment": f"e2c_v3_granularity_{ds}",
        "provenance": provenance,
        "inputs_sha256": inputs,
        "checkpoints_sha256": ckpts,
        "archive": {"release_dir": archive.get("release_dir"),
                    "hf_repo": archive.get("hf_repo"),
                    "hf_upload_ok": archive.get("hf_upload_ok"),
                    "hf_revision": archive.get("hf_revision"),
                    "n_files": archive.get("n_files", 0)},
        "results": {"cells_evaluated": summary["cells_evaluated"],
                    "cell_pass_total": summary["cell_pass_total"],
                    "run_stage": summary["run_stage"],
                    "executed_seeds": summary["executed_seeds"],
                    "g3_1_gate_status": summary["g3_1_gate"].get("status"),
                    "g3_1_gate_passed": summary["g3_1_gate"].get("passed"),
                    "per_mode_pass": {m: [v["cell_pass"], v["n_cells"]]
                                      for m, v in
                                      summary["per_mode"].items()}},
        "elapsed_sec_note": (
            "elapsed_sec is CUMULATIVE over every pass that produced this "
            "manifest; elapsed_this_pass_sec is this pass alone"),
        **({"g6_pilot": g6_block} if ds == "mllmu" else {}),
        **_cumulative_elapsed(out_base / "run_manifest.json", t_start,
                              _device_name()),
        "gpu": _device_name(),
    }
    with open(out_base / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"GX7: run manifest written (checkpoints={len(ckpts)})")
    return manifest


# ====================================================================== #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   choices=["salmu", "celeba_numeric", "mllmu"])
    p.add_argument("--phase", default="all",
                   choices=["all", "GX0", "GX1R", "GX2", "GX2R", "GX2B",
                            "GX2S", "GX3", "GX4", "GX5", "GX7"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=gx.SEEDS_DEFAULT)
    p.add_argument("--only-sets", nargs="*", default=None)
    p.add_argument("--only-seeds", type=int, nargs="*", default=None,
                   help="restrict EXECUTION to these edit seeds (the "
                        "frozen matrix always declares 17/42/123)")
    p.add_argument("--route-steps", type=int, default=mx.ROUTE_STEPS)
    p.add_argument("--route-warmup", type=int, default=mx.ROUTE_WARMUP)
    p.add_argument("--route-lr", type=float, default=mx.ROUTE_LR)
    p.add_argument("--route-repeat", type=int, default=mx.ROUTE_REPEAT)
    p.add_argument("--ul-steps", type=int, default=mx.UL_STEPS)
    p.add_argument("--ul-warmup", type=int, default=mx.UL_WARMUP)
    p.add_argument("--ul-lr", type=float, default=mx.UL_LR)
    p.add_argument("--ul-repeat", type=int, default=mx.UL_REPEAT)
    p.add_argument("--max-gen-tokens", type=int, default=12)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


#: The code this runner executes.  Result OUTPUTS are deliberately absent, and
#: so is every script this runner does not import.  Its own script first, then
#: the shared ones in the order the rule-generalization, prompt-panel and
#: method-baseline runners declare them -- this runner IS granularity_matrix, so
#: that entry is its own and is not repeated.
GX_CODE = ["scripts/e2c_v3_granularity_matrix.py",
           "scripts/e2c_v3_research_validity.py",
           "scripts/e2c_v3_granularity.py",
           "scripts/e2c_v3_matrix.py",
           "scripts/e2c_v3_realdata.py"]


def _dirty_tracked_code():
    """Tracked-file changes among the EXECUTED code (not result outputs).

    GX2B is a parallel ablation that coexists with a running main matrix;
    the main run's GX2R/GX7 legitimately dirty tracked RESULT files
    (cell_results.json, summaries).  Provenance for the ablation binds to
    the committed CODE, so we only refuse if a script itself is dirty.

    A declared script that is not on disk is reported as such: dropping it from
    the pathspec would silently widen ``git status`` to the WHOLE worktree and
    blame this ablation for edits made by the parallel runs.  Same fix as the
    rule-generalization, prompt-panel and method-baseline runners, which is the
    last of the four this reaches: the matrix runner is the script those runs
    were executing, so it could not be edited while they were.
    """
    import subprocess
    missing = [p for p in GX_CODE if not Path(p).exists()]
    if missing:
        return [f"<declared executed code missing: {p}>" for p in missing]
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no",
             *GX_CODE], text=True)
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return ["<git status failed>"]


def main():
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
    out_base = OUT_ROOT / f"{ds}{suffix}"
    out_base.mkdir(parents=True, exist_ok=True)

    commit = rv.git_commit_sha()
    dirty = rv.git_worktree_dirty()
    provenance = {
        "commit": commit,
        "runner_script_sha256": rv.sha256_file(Path(__file__).resolve()),
        "granularity_lib_sha256": rv.sha256_file(
            SCRIPT_DIR / "e2c_v3_granularity.py"),
        "shared_scoring_script_sha256": rv.script_sha256(),
        "dirty": dirty, "clean_required": not args.smoke,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    logger.info(f"Provenance: commit={commit} "
                f"runner={provenance['runner_script_sha256'][:12]} "
                f"gxlib={provenance['granularity_lib_sha256'][:12]} "
                f"rv={provenance['shared_scoring_script_sha256'][:12]} "
                f"dirty={dirty}")
    if dirty and not args.smoke:
        if args.phase in ("GX2B", "GX2S"):
            dirty_code = _dirty_tracked_code()
            if dirty_code:
                raise RuntimeError(
                    f"{args.phase}: executed CODE not committed: "
                    f"{dirty_code}")
            provenance["parallel_run_dirty_outputs_ok"] = True
            logger.warning("%s: tracked worktree has uncommitted OUTPUT "
                           "from a parallel main-matrix run (executed code "
                           "is committed); proceeding", args.phase)
        else:
            raise RuntimeError("tracked worktree dirty; commit before a "
                               "full granularity run")

    matrix, ctx, validation = build_or_verify(ds, args)
    with open(out_base / "gx0_validation.json", "w") as f:
        json.dump(validation, f, indent=2)
    if args.phase == "GX0":
        return 0
    if args.phase == "GX2B":
        # matched_retrain_balanced ablation: independent of the main
        # cell/oracle flow; reuses the existing edited cell,
        # matched_retrain_weighted and loo_retrain, trains only the balanced
        # reference
        return run_gx2b(args, ds, ctx, matrix, out_base, provenance,
                        commit, t_start)
    if args.phase == "GX2S":
        # oracle-seed sensitivity: reuses existing edited cells E, trains
        # matched/LOO fresh-retrain references at oracle seeds 42/123
        return run_gx2s(args, ds, ctx, matrix, out_base, provenance,
                        commit, t_start)
    if args.smoke:  # pilot subset: first single set only, one seed
        matrix = json.loads(json.dumps(matrix))
        keep = set(args.only_sets) if args.only_sets else \
            {matrix["sets"][0]["set_id"]}
        matrix["sets"] = [e for e in matrix["sets"] if e["set_id"] in keep]
        for e in matrix["sets"]:
            e["seeds"] = [17]
        matrix["edit_seeds"] = [17]

    if ds in ("celeba_numeric", "mllmu") and args.phase in (
            "all", "GX1R", "GX2", "GX3", "GX4", "GX5"):
        ensure_numeric_route(args, ctx, out_base, ds=ds)
    oracle_results = {}
    if args.phase in ("all", "GX2"):
        oracle_results = run_oracles(args, ds, ctx, matrix, out_base)
    else:
        p = out_base / "oracles" / "oracles_summary.json"
        if p.exists():
            with open(p) as f:
                oracle_results = json.load(f)
    if args.phase in ("all", "GX3", "GX4", "GX5"):
        run_cells(args, ds, ctx, matrix, out_base)
    if args.phase in ("all", "GX2R"):
        # G3.1: CPU re-evaluation of stored cells vs all four families
        reevaluate_oracles_cpu(ds, ctx, matrix, out_base)
    summary = None
    if args.phase in ("all", "GX2R", "GX7"):
        summary = aggregate_gx(ds, out_base, matrix, args)
        if args.phase in ("all", "GX7"):
            archive = {"release_dir": None, "hf_repo": None,
                       "hf_upload_ok": False, "hf_revision": None,
                       "n_files": 0}
            if not args.smoke:
                archive = archive_gx(args, ds, out_base, matrix,
                                     oracle_results)
            run_manifest_gx(args, ds, out_base, matrix, oracle_results,
                            summary, archive, t_start, provenance)
    logger.info("=" * 60)
    logger.info(f"GRANULARITY RUN ({ds}) PHASE {args.phase} COMPLETE "
                f"({round(time.time() - t_start, 1)}s)")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
