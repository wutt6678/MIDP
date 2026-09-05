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
- mllmu            : DEFERRED (G6) until the profession hierarchy is
                     externally justified and audited; the validator
                     refuses unaudited hierarchies.

Oracles (plan section 9, CORRECTED in G3.1): FOUR named families per set,
distinguishing initialization AND data --
  matched_finetune : trained-baseline-h init; transformed full mapping
                     (continued fine-tuning reference);
  loo_finetune     : trained-baseline-h init; retained mapping only
                     (deletion-as-fine-tuning reference; salmu reuses
                     suppression-matrix LOO oracles on exact set match);
  matched_retrain  : FRESH base + fresh LoRA; transformed full mapping;
                     ORIGINAL route-h protocol (3000/200/2e-5, targets x5,
                     retained x50, seed 17);
  loo_retrain      : FRESH base + fresh LoRA; retained mapping only;
                     same route-h protocol.
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

Phases: GX0 validate schemas/matrices | GX1 freeze matrix files
        GX1R numeric baseline route h | GX2 oracle families
        GX2R CPU re-evaluation of stored cells vs all families
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
# matched/loo *_retrain  = FRESH base + fresh LoRA, trained with the
# ORIGINAL route-h protocol (3000/200/2e-5, repeat 50, seed 17).
# Only the retrain families support retraining claims.
RETRAIN_STEPS = 3000
RETRAIN_WARMUP = 200
RETRAIN_LR = 2e-5
RETRAIN_REPEAT = 50             # original route retained repetition
RETRAIN_TARGET_BOOST = 5        # original route target oversampling
DELTA_RETRAIN_MIN_MARGIN = 0.5  # G3.1 materiality margin (L2)
ORACLE_FAMILIES = ("matched_finetune", "loo_finetune",
                   "matched_retrain", "loo_retrain")

SINGLE_MODES = {"single_level1", "single_level2", "single_exact_to_narrow",
                "single_exact_to_broad", "single_exact_to_rounded"}
SAME_DEPTH_MODES = {"simultaneous_same_depth_l1", "simultaneous_same_depth_l2",
                    "simultaneous_same_resolution"}
MIXED_MODES = {"simultaneous_mixed_depth", "simultaneous_mixed_resolution"}


def dataset_ctx(ds, matrix):
    """Build the per-dataset evaluation context (frozen inputs only)."""
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
    """target | sibling | cousin | unrelated | retain (for the report)."""
    if iid in entry["assignments"]:
        return "target"
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
    if ds == "salmu":
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
    return matrix, ctx, validation


def load_frozen(ds):
    name = ("matrix_salmu.json" if ds == "salmu"
            else "matrix_celeba_numeric.json")
    with open(MANIFEST_DIR / name) as f:
        return json.load(f)


# ====================================================================== #
# GX1R: numeric baseline route h (association route for the 24 codes)
# ====================================================================== #
def ensure_numeric_route(args, ctx, out_base):
    ckpt = out_base / "route_h" / "adapter_final" / "adapter_model.safetensors"
    if ckpt.exists():
        logger.info("GX1R: numeric baseline route h already present")
        return ckpt
    logger.info("GX1R: TRAIN numeric baseline route h (3000/200/2e-5, "
                "repeat 50, seed 17)")
    session = mx.ModelSession(args, "e2c_gx_num_h")
    try:
        mx.seed_everything(17)
        pairs = [{"prompt": rd.CODE_TO_ALIAS_PROMPT.format(
                      code=ctx["code_of"][i]),
                  "answer": ctx["baseline_alias_of"][i]}
                 for i in ctx["identity_ids"]]
        items = rv.build_supervised_items(session.adapter, session.processor,
                                          pairs, repeat=args.route_repeat)
        rv.train_supervised("gx_num_route_h", session.adapter, session.model,
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
#     lora_B zeros) + the ORIGINAL route-h protocol (3000/200/2e-5,
#     targets x5, retained x50, seed 17).
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


def session_reset_fresh(session):
    n_a, n_b = fresh_reinit_lora(
        list(session.adapter_model.named_parameters()), ORACLE_SEED)
    logger.info("reset_fresh(): LoRA re-initialized fresh "
                "(A kaiming / B zeros, seed %d; %d/%d tensors)",
                ORACLE_SEED, n_a, n_b)


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


def train_oracle_retrain(session, ds, ctx, entry, family, out_dir, args):
    """Fresh-init retraining oracle under the ORIGINAL route-h protocol."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session_reset_fresh(session)
    mx.seed_everything(ORACLE_SEED)
    items = []
    if family == "matched_retrain":
        assign = [{"prompt": rd.CODE_TO_ALIAS_PROMPT.format(
                       code=ctx["code_of"][i]),
                   "answer": expected_label(ctx, entry, i)}
                  for i in sorted(entry["assignments"])] * RETRAIN_TARGET_BOOST
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
    # G3.1 gate inputs.  matched_retrain must fit ALL transformed AND
    # retained mappings; loo_retrain never saw the targets, so its fit
    # check covers the RETAINED mapping only (targets are expected to
    # drift -- that is the deletion reference).
    if family == "matched_retrain":
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
    if family == "matched_retrain":
        fit_ok = bool(strict == 1.0 and (mass is None or mass >= 0.99))
        scope = "all_transformed_and_retained"
    else:
        fit_ok = bool(strict == 1.0)
        scope = "retained_only(targets_never_seen)"
    _write_oracle_results(
        out_dir, family, "fresh_lora",
        {"steps": RETRAIN_STEPS, "warmup": RETRAIN_WARMUP,
         "lr": RETRAIN_LR, "repeat": RETRAIN_REPEAT,
         "target_boost": RETRAIN_TARGET_BOOST if family == "matched_retrain"
         else 0, "seed": ORACLE_SEED},
        {"set_id": entry["set_id"], "strict_fit_scope": scope,
         "strict_all_expected": strict,
         "min_candidate_mass": mass, "fit_ok": fit_ok})
    logger.info("GX2[%s]: %s trained (fresh init; strict=%.4f mass=%s "
                "fit_ok=%s)", entry["set_id"], family, strict,
                f"{mass:.4f}" if mass is not None else "n/a", fit_ok)
    return {"mode": "trained_fresh", "strict_all_expected": strict,
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
            dmr = (r.get("matched_retrain") or {}).get("distance")
            dlr = (r.get("loo_retrain") or {}).get("distance")
            delta = r.get("delta_retrain_l2")
            fitp = (out_base / "oracles"
                    / f"matched_retrain_{c['set_id']}"
                    / "oracle_results.json")
            fit = {}
            if fitp.exists():
                with open(fitp) as f:
                    fit = json.load(f)
            gate_per.append({
                "cell_id": c["cell_id"], "target": t,
                "operation": a["operation"],
                "matched_retrain_fit_ok": fit.get("fit_ok"),
                "matched_retrain_strict": fit.get("strict_all_expected"),
                "matched_retrain_mass": fit.get("min_candidate_mass"),
                "l2_to_matched_retrain": dmr["l2"] if dmr else None,
                "l2_to_loo_retrain": dlr["l2"] if dlr else None,
                "delta_retrain_l2": delta,
                "delta_material": (delta is not None
                                   and delta >= DELTA_RETRAIN_MIN_MARGIN),
                "closer_to_matched_retrain": bool(
                    dmr and dlr and dmr["l2"] < dlr["l2"]),
            })
    have = [p for p in gate_per if p["delta_retrain_l2"] is not None]
    if not gate_per:
        g3_1_gate = {"status": "no_transformation_targets_evaluated"}
    elif not have:
        g3_1_gate = {"status": "retrain_oracles_not_yet_trained",
                     "n_transformation_targets": len(gate_per),
                     "per_target": gate_per}
    else:
        ok = (len(have) == len(gate_per) and all(
            p["matched_retrain_fit_ok"]
            and (p["matched_retrain_mass"] or 0) >= 0.99
            and p["delta_material"] and p["closer_to_matched_retrain"]
            for p in have))
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
    claims = {
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
            "matched_retrain": "FRESH base + fresh LoRA; transformed full "
                               "mapping; original route-h protocol "
                               "(3000/200/2e-5, targets x5, retained x50, "
                               "seed 17)",
            "loo_retrain": "FRESH base + fresh LoRA; retained mapping "
                           "only; same route-h protocol",
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

    if ds == "celeba_numeric":
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
    inputs = {
        "matrix_manifest": rv.sha256_file(
            MANIFEST_DIR / ("matrix_salmu.json" if ds == "salmu"
                            else "matrix_celeba_numeric.json")),
        "baseline_h": rv.sha256_file(_baseline_ckpt(ds, out_base)),
    }
    if ds == "salmu":
        inputs["dataset_manifest"] = rv.sha256_file(SALMU_MANIFEST)
        inputs["g_cache_e2e_rows"] = rv.sha256_file(SALMU_G_CACHE)
    else:
        inputs["numeric_manifest"] = rv.sha256_file(
            MANIFEST_DIR / "numeric_manifest.json")
    ckpts = {}
    legacy_name = {"matched_finetune": "matched", "loo_finetune": "loo"}
    for sid, rec in oracle_results.items():
        for fam in ORACLE_FAMILIES:
            fr = rec.get(fam) or rec.get(legacy_name.get(fam, ""), {})
            if fr.get("sha256"):
                ckpts[f"oracle_{fam}_{sid}"] = fr["sha256"]
    for c in load_all_cells(out_base):
        ckpts[f"cell_{c['cell_id']}"] = c["checkpoint_sha256"]
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
        "elapsed_sec": round(time.time() - t_start, 1),
        "gpu": torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else "cpu",
    }
    with open(out_base / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"GX7: run manifest written (checkpoints={len(ckpts)})")
    return manifest


# ====================================================================== #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   choices=["salmu", "celeba_numeric"])
    p.add_argument("--phase", default="all",
                   choices=["all", "GX0", "GX1R", "GX2", "GX2R", "GX3",
                            "GX4", "GX5", "GX7"])
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
        raise RuntimeError("tracked worktree dirty; commit before a full "
                           "granularity run")

    matrix, ctx, validation = build_or_verify(ds, args)
    with open(out_base / "gx0_validation.json", "w") as f:
        json.dump(validation, f, indent=2)
    if args.phase == "GX0":
        return 0
    if args.smoke:  # pilot subset: first single set only, one seed
        matrix = json.loads(json.dumps(matrix))
        keep = set(args.only_sets) if args.only_sets else \
            {matrix["sets"][0]["set_id"]}
        matrix["sets"] = [e for e in matrix["sets"] if e["set_id"] in keep]
        for e in matrix["sets"]:
            e["seeds"] = [17]
        matrix["edit_seeds"] = [17]

    if ds == "celeba_numeric" and args.phase in ("all", "GX1R", "GX2",
                                                 "GX3", "GX4", "GX5"):
        ensure_numeric_route(args, ctx, out_base)
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
