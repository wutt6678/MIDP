#!/usr/bin/env python3
"""E2C-v3 multi-seed / multi-forget-set matrix runner (PPUBench, MLLMU, SALMU).

Scale-up of the single-seed pilots to a declarative experiment matrix that
records BOTH the training seed and the forget-set identity for every cell.

Matrix design (per dataset, deterministic under --seed for set selection)
=========================================================================
- seeds: 17, 42, 123 (edit-training seeds; routes are the frozen, already
  validated per-dataset checkpoints from the committed pilot runs).
- single-target sets:
    * PPUBench: ALL 4 identity rotations.
    * MLLMU: 6 identities, one per profession (balanced over professions).
    * SALMU: 6 identities balanced across taxonomy depth (distinct
      level-2/level-1 groups via round-robin).
- simultaneous sets (ONE edited h per set -- all targets suppressed in the
  SAME training run and evaluated on the SAME model; never separate edits
  reported as simultaneous):
    * PPUBench: 3 pairwise 2-target stress sets (seed-17 draw over pairs).
    * MLLMU: 3 sets x 3 targets, each set spanning 3 distinct professions.
    * SALMU: 3 sets x 3 targets, each set spanning >= 2 level-2 groups.
- forget semantics: every target is driven to the refusal label 'Unknown'
  (refusal-targeted association suppression); all other identities retain
  their route aliases.

Per-cell metrics (all recorded, per seed, with mean/std/min and FAILURE
COUNTS -- not only averages -- at set and dataset level)
========================================================
- post-edit expectation accuracy (strict, multi-token aware, multi-label
  rejected);
- original-label suppression: P(original alias) per target (full-sequence);
- candidate mass and explicit OTHER mass per identity;
- gated distance to the leave-one-out oracle for the SAME forget set
  (L2 / JS / cosine + reliability gate at candidate mass >= 0.01);
- retained-set accuracy and worst-identity retained probability;
- held-out g routing accuracy (frozen route, from the committed pilot
  artifact; MLLMU has no held-out images and says so);
- conditional (g-routed-correctly subset) and unconditional (all images)
  E2E accuracy via cached g codes replayed through the edited h;
- unparseable-output and multi-label (ambiguous) counts;
- per-transformation results (per-target suppression, per-retained id);
- QA probes (MLLMU only, auxiliary) WITH the explicit warning that native
  QA preservation is NOT established (base model is at floor pre-edit).

Oracles: one leave-one-out h per forget set (retrained excluding exactly
that set, training-matched 3000/200/2e-5 x50, oracle seed fixed at 17 --
the reference does not vary with the edit seed). An existing pilot oracle
is reused when its exclusion set matches exactly (PPUBench {001}).

Checkpoint archival: every oracle and every cell checkpoint is archived
under releases/ with SHA-256 and, when Hugging Face authentication is
available, uploaded to an immutable HF repo referenced by URI + SHA-256,
so results are exactly reloadable, not merely auditable.

Phases: MX0 matrix manifest | MX1 oracles | MX2 cells | MX3 aggregate
        MX4 archive (+HF upload) | MX5 run manifest.
"""
import argparse
import gc
import importlib.util
import itertools
import json
import logging
import shutil
import statistics
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2c_v3_matrix")

SCRIPT_DIR = Path(__file__).resolve().parent

# Frozen routes / pilots (committed artifacts) per dataset
DATASET_ROOTS = {
    "ppubench": Path("e2c_v3_real"),
    "mllmu": Path("e2c_mllmu"),
    "salmu": Path("e2c_salmu"),
}
ROUTE_DIRS = {
    "ppubench": Path("e2c_v3_real/outputs/realdata"),
    "mllmu": Path("e2c_mllmu/outputs/mllmu"),
    "salmu": Path("e2c_salmu/outputs/salmu"),
}
MANIFEST_PATHS = {
    "ppubench": Path("e2c_v3_real/manifests/realdata_identity_mapping.json"),
    "mllmu": Path("e2c_mllmu/manifests/mllmu_manifest.json"),
    "salmu": Path("e2c_salmu/manifests/salmu_manifest.json"),
}
# committed pilot e2e rows carrying cached frozen-g pred codes per image
G_CACHE_PATHS = {
    "ppubench": ROUTE_DIRS["ppubench"] / "e2e_post" / "e2e_results.json",
    "mllmu": ROUTE_DIRS["mllmu"] / "e2e_post" / "e2e_results.json",
    "salmu": ROUTE_DIRS["salmu"] / "e2e_post" / "e2e_results.json",
}
# held-out g routing accuracy from the committed pilots (frozen g; recorded
# per dataset, not per cell)
HELD_OUT_ROUTING = {
    "ppubench": {"held_out_g_accuracy": 1.0, "source": "RD1 gate, commit 9bf3652"},
    "mllmu": {"held_out_g_accuracy": None,
              "source": "MLLMU-Bench has ONE image per identity; image-level "
                        "held-out routing does not exist (stated limitation)"},
    "salmu": {"held_out_g_accuracy": 0.8056,
              "source": "SB1 rescore at 1e11df4 (decode-fixed)"},
}
# existing pilot oracles reusable when the exclusion set matches exactly
EXISTING_ORACLES = {
    "ppubench": {"001": ROUTE_DIRS["ppubench"] / "h_oracle"},
    "mllmu": {},   # pilot oracle excluded {suppress, update} -- no match
    "salmu": {},   # pilot oracle excluded {suppress, gen_l1, gen_l2} -- no match
}

OUT_ROOT = Path("e2c_matrix")
MANIFEST_DIR = OUT_ROOT / "manifests"

SEEDS_DEFAULT = [17, 42, 123]
MATRIX_SEED = 17           # controls set selection only
ORACLE_SEED = 17           # oracles are references; fixed seed
ROUTE_STEPS = 3000
ROUTE_WARMUP = 200
ROUTE_LR = 2e-5
ROUTE_REPEAT = 50
UL_STEPS = 500
UL_WARMUP = 50
UL_LR = 2e-5
UL_REPEAT = 50
MIN_CANDIDATE_MASS = 0.01
TARGET_BOOST = 5           # suppression-pair repeat multiplier
RETAIN_REPEAT = 3

HF_ARCHIVE_REPO = "wutt6678/e2c-v3-matrix-checkpoints"

QA_WARNING = (
    "Native QA preservation is NOT ESTABLISHED by the probe delta: the base "
    "model scores at floor on Mask_Task probes pre-edit (1/36), so pre/post "
    "similarity cannot demonstrate preservation of native QA ability; the "
    "probes only detect large collateral shifts above that floor.")


def _load_sibling(module_name, filename):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rv = _load_sibling("e2c_rv_shared", "e2c_v3_research_validity.py")
rd = _load_sibling("e2c_rd_shared", "e2c_v3_realdata.py")
DELETED_LABEL = rd.DELETED_LABEL


def set_id(targets):
    return "fs_" + "-".join(sorted(targets))


def set_dir_name(targets):
    return set_id(targets)


# ====================================================================== #
# MX0: declarative matrix construction (deterministic, recorded rules)
# ====================================================================== #
def _balanced_mllmu_singles(manifest):
    """One identity per profession (first sorted id), preserving the
    manifest's profession frequency order."""
    by_prof = {}
    for iid in manifest["identity_ids"]:
        by_prof.setdefault(manifest["alias_of"][iid], []).append(iid)
    singles = []
    for prof in manifest["professions"]:
        singles.append(min(by_prof[prof]))
    return singles


def _balanced_mllmu_simultaneous(manifest, n_sets=3, set_size=3):
    """3 sets x 3 targets; each set spans distinct professions; ids drawn
    deterministically under MATRIX_SEED without reuse across sets."""
    g = torch.Generator().manual_seed(MATRIX_SEED)
    ids = list(manifest["identity_ids"])
    perm = torch.randperm(len(ids), generator=g).tolist()
    shuffled = [ids[i] for i in perm]
    sets, used = [], set()
    for _ in range(n_sets):
        cur, profs = [], set()
        for iid in shuffled:
            if iid in used:
                continue
            prof = manifest["alias_of"][iid]
            if prof in profs:
                continue
            cur.append(iid)
            profs.add(prof)
            used.add(iid)
            if len(cur) == set_size:
                break
        if len(cur) != set_size:
            raise RuntimeError("cannot build balanced MLLMU simultaneous set")
        sets.append(sorted(cur))
    return sets


def _balanced_salmu_singles(manifest):
    """6 identities balanced across taxonomy depth: round-robin over
    (level2, level1) groups in sorted order."""
    levels = manifest["job_levels"]
    groups = {}
    for iid in manifest["identity_ids"]:
        key = (levels[iid][2], levels[iid][1])
        groups.setdefault(key, []).append(iid)
    keys = sorted(groups)
    singles, k = [], 0
    while len(singles) < 6:
        advanced = False
        for key in keys:
            members = sorted(groups[key])
            if k < len(members):
                singles.append(members[k])
                advanced = True
                if len(singles) >= 6:
                    break
        if not advanced:
            break
        k += 1
    return singles


def _balanced_salmu_simultaneous(manifest, n_sets=3, set_size=3):
    """3 sets x 3 targets; each set spans >= 2 distinct level-2 groups;
    deterministic under MATRIX_SEED, no id reuse across sets."""
    levels = manifest["job_levels"]
    g = torch.Generator().manual_seed(MATRIX_SEED)
    ids = list(manifest["identity_ids"])
    perm = torch.randperm(len(ids), generator=g).tolist()
    shuffled = [ids[i] for i in perm]
    sets, used = [], set()
    for _ in range(n_sets):
        cur, l2s = [], set()
        for iid in shuffled:
            if iid in used:
                continue
            l2 = levels[iid][2]
            # first two picks free; third must add a new level-2 group if one
            # is still available among unused ids
            if len(cur) == 2:
                remaining_new = [x for x in shuffled if x not in used
                                 and x not in cur and levels[x][2] not in l2s]
                if remaining_new and l2 in l2s:
                    continue
            cur.append(iid)
            l2s.add(l2)
            used.add(iid)
            if len(cur) == set_size:
                break
        if len(cur) != set_size:
            raise RuntimeError("cannot build balanced SALMU simultaneous set")
        sets.append(sorted(cur))
    return sets


def _ppubench_sets(manifest):
    ids = sorted(manifest["identity_ids"])
    singles = [[i] for i in ids]
    g = torch.Generator().manual_seed(MATRIX_SEED)
    pairs = sorted(itertools.combinations(ids, 2))
    perm = torch.randperm(len(pairs), generator=g).tolist()
    simultaneous = [sorted(pairs[i]) for i in perm[:3]]
    return singles, simultaneous


def _matrix_path(args):
    suffix = "_smoke" if args.smoke else ""
    return MANIFEST_DIR / f"matrix_{args.dataset}{suffix}.json"


def build_matrix(args):
    logger.info("=" * 60)
    logger.info(f"MX0: BUILD MATRIX MANIFEST ({args.dataset})")
    logger.info("=" * 60)
    with open(MANIFEST_PATHS[args.dataset]) as f:
        manifest = json.load(f)
    ds = args.dataset
    if ds == "ppubench":
        singles, simultaneous = _ppubench_sets(manifest)
        rules = {"singles": "all 4 identity rotations",
                 "simultaneous": "seed-17 draw of 3 of the 6 pairwise "
                                 "2-target stress sets"}
    elif ds == "mllmu":
        singles = [[i] for i in _balanced_mllmu_singles(manifest)]
        simultaneous = _balanced_mllmu_simultaneous(manifest)
        rules = {"singles": "one identity per profession (6 professions, "
                            "first sorted id each)",
                 "simultaneous": "3 sets x 3 targets, each spanning 3 "
                                 "distinct professions, seed-17 draw, no "
                                 "id reuse across sets"}
    else:  # salmu
        singles = [[i] for i in _balanced_salmu_singles(manifest)]
        simultaneous = _balanced_salmu_simultaneous(manifest)
        rules = {"singles": "6 identities round-robin over (level2, level1) "
                            "taxonomy groups (balanced across depths)",
                 "simultaneous": "3 sets x 3 targets, each spanning >= 2 "
                                 "level-2 groups, seed-17 draw, no id reuse"}

    sets = ([{"targets": s, "mode": "single"} for s in singles]
            + [{"targets": s, "mode": "simultaneous"} for s in simultaneous])
    existing = EXISTING_ORACLES[ds]
    for entry in sets:
        entry["set_id"] = set_id(entry["targets"])
        match = "+".join(sorted(entry["targets"]))
        reuse = None
        for excl, path in existing.items():
            if "+".join(sorted(excl.split("|"))) == match:
                reuse = str(path)
        entry["oracle_reuse"] = reuse
    matrix = {
        "dataset": ds,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "matrix_seed": MATRIX_SEED,
        "edit_seeds": args.seeds,
        "oracle_seed": ORACLE_SEED,
        "selection_rules": rules,
        "forget_semantics": f"every target driven to refusal label "
                            f"'{DELETED_LABEL}' in ONE edited h per set; "
                            f"all other identities retain their aliases",
        "simultaneity_guarantee": "one training run and one checkpoint per "
                                  "(set, seed); all targets evaluated on the "
                                  "same model",
        "route": {"dir": str(ROUTE_DIRS[ds]),
                  "note": "frozen pilot route checkpoints; g is never "
                          "edited or retrained by the matrix"},
        "held_out_routing": HELD_OUT_ROUTING[ds],
        "budgets": {"oracle": [ROUTE_STEPS, ROUTE_WARMUP, ROUTE_LR,
                               ROUTE_REPEAT],
                    "edit": [args.ul_steps, args.ul_warmup, args.ul_lr,
                             args.ul_repeat, TARGET_BOOST, RETAIN_REPEAT]},
        "sets": sets,
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = _matrix_path(args)
    with open(path, "w") as f:
        json.dump(matrix, f, indent=2)
    logger.info(f"MX0: {len(sets)} forget sets x {len(args.seeds)} seeds = "
                f"{len(sets) * len(args.seeds)} cells")
    for e in sets:
        logger.info(f"  [{e['mode']:12s}] {e['set_id']} "
                    f"oracle_reuse={e['oracle_reuse']}")
    return matrix


def load_matrix(args):
    with open(_matrix_path(args)) as f:
        return json.load(f)


# ====================================================================== #
# Persistent model session (base model loaded ONCE per phase; LoRA tensors
# fully overwritten by the fail-closed loader on every reset, so cells are
# independent without paying the ~3 min base-model reload each time)
# ====================================================================== #
class ModelSession:
    def __init__(self, args, adapter_name):
        self.args = args
        self.adapter, self.model, self.processor = rv.create_adapter_model(
            args, args.device, adapter_name)
        self.model = rv.attach_lora(self.adapter, self.model)

    def reset_to(self, safetensors_path):
        rv.load_trained_weights(self.adapter, self.model,
                                Path(safetensors_path))

    def backend(self):
        from route_data.config import ModelConfig
        cfg = ModelConfig(
            backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
            revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            dtype="bfloat16", seed=self.args.seed)
        return self.adapter.to_eval_backend(
            model=self.model, processor=self.processor, model_config=cfg)

    def release(self):
        del self.model, self.processor, self.adapter
        gc.collect()
        torch.cuda.empty_cache()


def label_vocab_for(manifest):
    return sorted(set(manifest["alias_of"].values()) | {DELETED_LABEL})


def suppression_pairs(manifest, targets):
    return [{"prompt": rd.CODE_TO_ALIAS_PROMPT.format(
                code=manifest["code_of"][t]), "answer": DELETED_LABEL}
            for t in targets]


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ====================================================================== #
# MX1: leave-one-out oracles per forget set (seed fixed at ORACLE_SEED)
# ====================================================================== #
def run_oracles(args, out_base, matrix, manifest):
    logger.info("=" * 60)
    logger.info("MX1: LEAVE-ONE-OUT ORACLES PER FORGET SET")
    logger.info("=" * 60)
    oracle_root = out_base / "oracles"
    oracle_root.mkdir(parents=True, exist_ok=True)
    results = {}
    session = None
    args_o = argparse.Namespace(**vars(args))
    args_o.seed = ORACLE_SEED
    baseline = (Path(matrix["route"]["dir"]) / "h_C_to_Y" / "adapter_final"
                / "adapter_model.safetensors")
    for entry in matrix["sets"]:
        sid = entry["set_id"]
        odir = oracle_root / sid
        soft_path = odir / "oracle_soft.json"
        ckpt = odir / "adapter_final" / "adapter_model.safetensors"
        if entry["oracle_reuse"]:
            reuse_ckpt = (Path(entry["oracle_reuse"]) / "adapter_final"
                          / "adapter_model.safetensors")
            if not reuse_ckpt.exists():
                raise RuntimeError(f"oracle reuse path missing: {reuse_ckpt}")
            odir.mkdir(parents=True, exist_ok=True)
            if session is None:
                session = ModelSession(args_o, f"mx_{args.dataset}_oracle")
            session.reset_to(reuse_ckpt)
            seed_everything(ORACLE_SEED)
            soft, _ = rd.soft_eval_codes(args_o, session.adapter,
                                         session.model, session.processor,
                                         manifest)
            with open(soft_path, "w") as f:
                json.dump(soft, f, indent=2)
            results[sid] = {
                "mode": "reused_pilot_oracle",
                "source": str(reuse_ckpt),
                "source_sha256": rv.sha256_file(reuse_ckpt),
                "excluded": entry["targets"],
            }
            logger.info(f"MX1[{sid}]: reused pilot oracle "
                        f"({rv.sha256_file(reuse_ckpt)[:12]})")
            continue
        if ckpt.exists() and soft_path.exists():
            logger.info(f"MX1[{sid}]: oracle already present, skipping")
            results[sid] = {"mode": "cached", "excluded": entry["targets"],
                            "checkpoint_sha256": rv.sha256_file(ckpt)}
            continue
        odir.mkdir(parents=True, exist_ok=True)
        if session is None:
            session = ModelSession(args_o, f"mx_{args.dataset}_oracle")
        session.reset_to(baseline)
        seed_everything(ORACLE_SEED)
        pairs = rd.h_pairs(manifest, exclude=entry["targets"])
        items = rv.build_supervised_items(session.adapter, session.processor,
                                          pairs, repeat=args.route_repeat)
        rv.train_supervised(f"oracle_{sid}", session.adapter, session.model,
                            session.processor, items, odir, args.device,
                            steps=args.route_steps, warmup=args.route_warmup,
                            lr=args.route_lr)
        soft, _ = rd.soft_eval_codes(args_o, session.adapter, session.model,
                                     session.processor, manifest)
        with open(soft_path, "w") as f:
            json.dump(soft, f, indent=2)
        results[sid] = {
            "mode": "trained",
            "excluded": entry["targets"],
            "checkpoint_sha256": rv.sha256_file(ckpt),
            "training_matched_pilot_recipe": True,
        }
        logger.info(f"MX1[{sid}]: oracle trained, "
                    f"sha={rv.sha256_file(ckpt)[:12]}")
    if session is not None:
        session.release()
    with open(oracle_root / "oracles_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ====================================================================== #
# MX2: cells -- ONE edited h per (forget set, seed), full metric suite
# ====================================================================== #
def _hard_eval(session, manifest, vocab, targets, max_new_tokens):
    """Strict hard eval against POST-EDIT expectations (targets -> Unknown,
    retained -> alias).  Multi-token aware, multi-label rejected."""
    backend = session.backend()
    session.model.eval()
    preds = []
    with torch.no_grad():
        for iid in manifest["identity_ids"]:
            prompt = rd.CODE_TO_ALIAS_PROMPT.format(
                code=manifest["code_of"][iid])
            gen = backend.generate(None, prompt,
                                   max_new_tokens=max_new_tokens)
            raw = gen.text.strip()
            parsed = rv.parse_recognized_label(raw, vocab)
            labels = rv.recognized_labels_in(raw, vocab)
            expected = DELETED_LABEL if iid in targets else \
                manifest["alias_of"][iid]
            preds.append({
                "identity_id": iid, "raw": raw, "parsed_label": parsed,
                "recognized_labels": labels,
                "multi_label_ambiguous": len(labels) > 1,
                "role": "target" if iid in targets else "retain",
                "expected_post_edit": expected,
                "correct_post_edit": parsed == expected,
                "old_alias_leaked": (iid in targets
                                     and manifest["alias_of"][iid] in labels),
            })
    return preds


def _soft_eval(session, args, manifest, vocab):
    out = {}
    for iid in manifest["identity_ids"]:
        probs = rv.full_sequence_label_probs(
            session.adapter, session.model, session.processor,
            manifest["code_of"][iid], vocab, args.device)
        prob_by_label = {l: probs.get(l, {}).get("prob", 0.0) for l in vocab}
        logp = {l: probs.get(l, {}).get("log_prob", -1e9) for l in vocab}
        summary = rv.build_candidate_summary(prob_by_label, vocab,
                                             DELETED_LABEL)
        summary["log_probs"] = logp
        out[iid] = summary
    return out


def _e2e_replay(session, manifest, vocab, targets, g_rows, max_new_tokens):
    """Replay the CACHED frozen-g code predictions through the edited h.

    conditional   = accuracy over images g routed CORRECTLY
    unconditional = accuracy over ALL images (routing failures count wrong)
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
                             "recognized_labels": [], "outcome_ok": False})
                continue
            prompt = rd.CODE_TO_ALIAS_PROMPT.format(code=base["pred_code"])
            gen = backend.generate(None, prompt,
                                   max_new_tokens=max_new_tokens)
            raw = gen.text.strip()
            parsed = rv.parse_recognized_label(raw, vocab)
            labels = rv.recognized_labels_in(raw, vocab)
            if iid in targets:
                ok = manifest["alias_of"][iid] not in labels
            else:
                ok = parsed == manifest["alias_of"][iid]
            rows.append({**base, "h_raw_text": raw, "pred_alias": parsed,
                         "recognized_labels": labels,
                         "multi_label_ambiguous": len(labels) > 1,
                         "outcome_ok": ok})
    n_all = len(rows)
    routed = [r for r in rows if r["g_routed"] and r["pred_code"]]
    cond = [r for r in rows if r["code_correct"]]
    summary = {
        "n_images": n_all,
        "g_routed": len(routed),
        "g_routing_failures": n_all - len(routed),
        "g_code_correct": sum(r["code_correct"] for r in rows),
        "e2e_unconditional_ok": sum(r["outcome_ok"] for r in rows),
        "e2e_unconditional_acc": sum(r["outcome_ok"] for r in rows)
        / max(n_all, 1),
        "e2e_conditional_ok": sum(r["outcome_ok"] for r in cond),
        "e2e_conditional_n": len(cond),
        "e2e_conditional_acc": sum(r["outcome_ok"] for r in cond)
        / max(len(cond), 1),
        "h_unparseable_outputs": sum(
            1 for r in rows if r["pred_alias"] is None),
        "h_multi_label_ambiguous_outputs": sum(
            1 for r in rows if r.get("multi_label_ambiguous")),
        "by_split": {},
    }
    for split in sorted({r["split"] for r in rows}):
        sub = [r for r in rows if r["split"] == split]
        summary["by_split"][str(split)] = {
            "n": len(sub),
            "e2e_unconditional_acc": sum(r["outcome_ok"] for r in sub)
            / max(len(sub), 1),
        }
    return summary, rows


def _qa_probes_on_session(session, manifest):
    """MLLMU auxiliary native QA probes on the CURRENT session weights."""
    backend = session.backend()
    session.model.eval()
    results = {}
    with torch.no_grad():
        for iid, probes in manifest.get("qa_probes", {}).items():
            rows = []
            for p in probes:
                gen = backend.generate(None, p["question"], max_new_tokens=40)
                text = gen.text.strip()
                rows.append({"question": p["question"],
                             "ground_truth": p["ground_truth"],
                             "generated": text,
                             "correct": p["ground_truth"].lower()
                             in text.lower()})
            results[iid] = rows
    n_all = sum(len(v) for v in results.values())
    n_ok = sum(r["correct"] for v in results.values() for r in v)
    return {"accuracy": round(n_ok / max(n_all, 1), 4), "n": n_all,
            "warning": QA_WARNING, "per_identity": results}


def _failure_flags(hard_preds, soft, targets, manifest, vocab):
    retain_ids = [i for i in manifest["identity_ids"] if i not in targets]
    retain_acc = (sum(p["correct_post_edit"] for p in hard_preds
                      if p["identity_id"] in set(retain_ids))
                  / max(len(retain_ids), 1))
    leaks = [p["identity_id"] for p in hard_preds if p["old_alias_leaked"]]
    soft_leaks = [t for t in targets
                  if soft[t]["probs"].get(manifest["alias_of"][t], 0.0) > 0.5]
    incomplete = [p["identity_id"] for p in hard_preds
                  if p["role"] == "target" and not p["correct_post_edit"]]
    oov = [i for i in manifest["identity_ids"]
           if soft[i]["candidate_mass"] < MIN_CANDIDATE_MASS]
    ambiguous = [p["identity_id"] for p in hard_preds
                 if p["multi_label_ambiguous"]]
    worst = None
    if retain_ids:
        worst = min(retain_ids,
                    key=lambda i: soft[i]["probs"].get(
                        manifest["alias_of"][i], 0.0))
    flags = {
        "retain_acc": retain_acc,
        "worst_retained_identity": worst,
        "worst_retained_p_alias": (
            soft[worst]["probs"].get(manifest["alias_of"][worst], 0.0)
            if worst else None),
        "hard_leaks": leaks,
        "soft_leaks": soft_leaks,
        "suppress_incomplete": incomplete,
        "oov_garbage_ids": oov,
        "ambiguous_ids": ambiguous,
        "leak": bool(leaks or soft_leaks),
        "retain_broken": retain_acc < 1.0,
        "catastrophic": bool(leaks or soft_leaks) or retain_acc < 0.5
        or bool(oov),
        "cell_pass": not leaks and not soft_leaks and not incomplete
        and retain_acc >= 1.0 and not oov,
    }
    return flags


def run_cells(args, out_base, matrix, manifest, g_rows):
    logger.info("=" * 60)
    logger.info(f"MX2: CELLS ({args.dataset}) -- one edited h per "
                f"(set, seed)")
    logger.info("=" * 60)
    cells_root = out_base / "cells"
    cells_root.mkdir(parents=True, exist_ok=True)
    vocab = label_vocab_for(manifest)
    baseline = (Path(matrix["route"]["dir"]) / "h_C_to_Y" / "adapter_final"
                / "adapter_model.safetensors")
    oracle_root = out_base / "oracles"
    session = ModelSession(args, f"mx_{args.dataset}_cell")
    all_cells = {}
    try:
        for entry in matrix["sets"]:
            sid = entry["set_id"]
            targets = set(entry["targets"])
            oracle_soft_path = oracle_root / sid / "oracle_soft.json"
            with open(oracle_soft_path) as f:
                oracle_soft = json.load(f)
            for seed in matrix["edit_seeds"]:
                cell_id = f"{sid}__seed{seed}"
                cell_dir = cells_root / sid / f"seed_{seed}"
                cell_dir.mkdir(parents=True, exist_ok=True)
                result_path = cell_dir / "cell_results.json"
                ckpt = cell_dir / "edited_h" / "adapter_final" \
                    / "adapter_model.safetensors"
                if result_path.exists() and ckpt.exists() and not args.smoke:
                    logger.info(f"MX2[{cell_id}]: cached, skipping")
                    with open(result_path) as f:
                        all_cells[cell_id] = json.load(f)
                    continue
                logger.info("-" * 56)
                logger.info(f"MX2[{cell_id}]: reset->baseline, seed={seed}, "
                            f"targets={sorted(targets)}")
                session.reset_to(baseline)
                seed_everything(seed)
                args_c = argparse.Namespace(**vars(args))
                args_c.seed = seed
                items = (
                    rv.build_supervised_items(
                        session.adapter, session.processor,
                        suppression_pairs(manifest, targets),
                        repeat=args.ul_repeat * TARGET_BOOST)
                    + rv.build_supervised_items(
                        session.adapter, session.processor,
                        rd.h_pairs(manifest, exclude=targets),
                        repeat=args.ul_repeat * RETAIN_REPEAT))
                rv.train_supervised(f"cell_{cell_id}", session.adapter,
                                    session.model, session.processor, items,
                                    cell_dir / "edited_h", args.device,
                                    steps=args.ul_steps,
                                    warmup=args.ul_warmup, lr=args.ul_lr)
                seed_everything(seed)
                hard = _hard_eval(session, manifest, vocab, targets,
                                  args.max_gen_tokens)
                soft = _soft_eval(session, args_c, manifest, vocab)
                flags = _failure_flags(hard, soft, targets, manifest, vocab)
                e2e, e2e_rows = _e2e_replay(session, manifest, vocab,
                                            targets, g_rows,
                                            args.max_gen_tokens)
                dist = {}
                for iid in manifest["identity_ids"]:
                    d, reliable, reason = rv.gated_distance(
                        soft[iid], oracle_soft[iid], "candidate", vocab,
                        MIN_CANDIDATE_MASS)
                    dist[iid] = {"distance": d, "reliable": reliable,
                                 "reason": reason}
                cell = {
                    "cell_id": cell_id, "dataset": args.dataset,
                    "set_id": sid, "mode": entry["mode"],
                    "targets": sorted(targets), "seed": seed,
                    "checkpoint_sha256": rv.sha256_file(ckpt),
                    "post_edit_expectation_accuracy": sum(
                        p["correct_post_edit"] for p in hard) / len(hard),
                    "hard_preds": hard,
                    "soft": {i: {"p_original_alias": soft[i]["probs"].get(
                                     manifest["alias_of"][i], 0.0),
                                 "p_unknown": soft[i]["probs"].get(
                                     DELETED_LABEL, 0.0),
                                 "candidate_mass": soft[i]["candidate_mass"],
                                 "other_mass": soft[i]["other_mass"]}
                             for i in manifest["identity_ids"]},
                    "soft_log_probs": {i: soft[i]["log_probs"]
                                       for i in manifest["identity_ids"]},
                    "gated_distance_to_oracle": dist,
                    "e2e": e2e,
                    "e2e_rows": e2e_rows,
                    "failure_flags": flags,
                    "per_transformation": {
                        "suppress": {t: {
                            "hard": next(p["correct_post_edit"] for p in hard
                                         if p["identity_id"] == t),
                            "p_original": soft[t]["probs"].get(
                                manifest["alias_of"][t], 0.0),
                            "p_unknown": soft[t]["probs"].get(
                                DELETED_LABEL, 0.0)} for t in sorted(targets)},
                        "retain": {i: {
                            "hard": next(p["correct_post_edit"] for p in hard
                                         if p["identity_id"] == i),
                            "p_alias": soft[i]["probs"].get(
                                manifest["alias_of"][i], 0.0)}
                            for i in manifest["identity_ids"]
                            if i not in targets},
                    },
                }
                if args.dataset == "mllmu":
                    cell["qa_probes_post"] = _qa_probes_on_session(
                        session, manifest)
                with open(result_path, "w") as f:
                    json.dump(cell, f, indent=2)
                all_cells[cell_id] = cell
                acc = cell["post_edit_expectation_accuracy"]
                logger.info(
                    f"MX2[{cell_id}]: acc={acc:.3f} "
                    f"retain={flags['retain_acc']:.3f} "
                    f"leak={flags['leak']} pass={flags['cell_pass']} "
                    f"e2e_cond={e2e['e2e_conditional_acc']:.3f} "
                    f"e2e_uncond={e2e['e2e_unconditional_acc']:.3f}")
    finally:
        session.release()
    with open(cells_root / "cells_index.json", "w") as f:
        json.dump({k: {"set_id": v["set_id"], "seed": v["seed"],
                       "checkpoint_sha256": v["checkpoint_sha256"],
                       "pass": v["failure_flags"]["cell_pass"]}
                   for k, v in all_cells.items()}, f, indent=2)
    return all_cells


# ====================================================================== #
# MX3: aggregation across seeds -- mean/std/min, per-seed values, and
# FAILURE RATES (a set that passes 2 seeds and fails 1 stays visible)
# ====================================================================== #
def _msd(values):
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {"mean": statistics.fmean(values),
            "std": (statistics.stdev(values) if len(values) > 1 else 0.0),
            "min": min(values), "max": max(values)}


def aggregate(args, out_base, matrix, all_cells):
    logger.info("=" * 60)
    logger.info("MX3: AGGREGATE ACROSS SEEDS (mean/std + failure rates)")
    logger.info("=" * 60)
    per_set = {}
    for entry in matrix["sets"]:
        sid = entry["set_id"]
        cells = [all_cells[f"{sid}__seed{s}"] for s in matrix["edit_seeds"]
                 if f"{sid}__seed{s}" in all_cells]
        if not cells:
            continue
        n = len(cells)
        agg = {
            "set_id": sid, "mode": entry["mode"],
            "targets": entry["targets"], "n_seeds": n,
            "seeds": [c["seed"] for c in cells],
            "post_edit_expectation_accuracy": _msd(
                [c["post_edit_expectation_accuracy"] for c in cells]),
            "retain_acc": _msd(
                [c["failure_flags"]["retain_acc"] for c in cells]),
            "worst_retained_p_alias": _msd(
                [c["failure_flags"]["worst_retained_p_alias"] for c in cells
                 if c["failure_flags"]["worst_retained_p_alias"] is not None]),
            "max_target_p_original": _msd(
                [max(c["soft"][t]["p_original_alias"] for t in entry["targets"])
                 for c in cells]),
            "min_target_p_unknown": _msd(
                [min(c["soft"][t]["p_unknown"] for t in entry["targets"])
                 for c in cells]),
            "min_candidate_mass": _msd(
                [min(v["candidate_mass"] for v in c["soft"].values())
                 for c in cells]),
            "max_other_mass": _msd(
                [max(v["other_mass"] for v in c["soft"].values())
                 for c in cells]),
            "e2e_conditional_acc": _msd(
                [c["e2e"]["e2e_conditional_acc"] for c in cells]),
            "e2e_unconditional_acc": _msd(
                [c["e2e"]["e2e_unconditional_acc"] for c in cells]),
            "unparseable_outputs_total": _msd(
                [c["e2e"]["h_unparseable_outputs"] for c in cells]),
            "multi_label_outputs_total": _msd(
                [c["e2e"]["h_multi_label_ambiguous_outputs"] for c in cells]),
            "oracle_distance_l2_changed": _msd(
                [statistics.fmean(
                    [c["gated_distance_to_oracle"][t]["distance"]["l2"]
                     for t in entry["targets"]
                     if c["gated_distance_to_oracle"][t]["reliable"]])
                 for c in cells if all(
                     c["gated_distance_to_oracle"][t]["reliable"]
                     for t in entry["targets"])]),
            "failure_counts": {
                "cells": n,
                "leak": sum(c["failure_flags"]["leak"] for c in cells),
                "soft_leak": sum(bool(c["failure_flags"]["soft_leaks"])
                                 for c in cells),
                "suppress_incomplete": sum(
                    bool(c["failure_flags"]["suppress_incomplete"])
                    for c in cells),
                "retain_broken": sum(c["failure_flags"]["retain_broken"]
                                     for c in cells),
                "oov_garbage": sum(bool(c["failure_flags"]["oov_garbage_ids"])
                                   for c in cells),
                "catastrophic": sum(c["failure_flags"]["catastrophic"]
                                    for c in cells),
                "cell_pass": sum(c["failure_flags"]["cell_pass"]
                                 for c in cells),
            },
            "per_seed": {str(c["seed"]): {
                "post_edit_expectation_accuracy":
                    c["post_edit_expectation_accuracy"],
                "retain_acc": c["failure_flags"]["retain_acc"],
                "max_target_p_original": max(
                    c["soft"][t]["p_original_alias"] for t in entry["targets"]),
                "e2e_conditional_acc": c["e2e"]["e2e_conditional_acc"],
                "e2e_unconditional_acc": c["e2e"]["e2e_unconditional_acc"],
                "cell_pass": c["failure_flags"]["cell_pass"],
                "catastrophic": c["failure_flags"]["catastrophic"],
                "checkpoint_sha256": c["checkpoint_sha256"],
            } for c in cells},
        }
        if args.dataset == "mllmu":
            qa = [c["qa_probes_post"]["accuracy"] for c in cells
                  if "qa_probes_post" in c]
            agg["qa_probes_post_accuracy"] = _msd(qa)
            agg["qa_warning"] = QA_WARNING
        per_set[sid] = agg
        logger.info(
            f"MX3[{sid}] pass={agg['failure_counts']['cell_pass']}/{n} "
            f"retain_acc(mean)={agg['retain_acc']['mean']} "
            f"e2e_cond(mean)="
            f"{round(agg['e2e_conditional_acc']['mean'] or 0, 4)}")
    dataset_level = {
        "cells_total": sum(a["failure_counts"]["cells"]
                           for a in per_set.values()),
        "cell_pass_total": sum(a["failure_counts"]["cell_pass"]
                               for a in per_set.values()),
        "catastrophic_total": sum(a["failure_counts"]["catastrophic"]
                                  for a in per_set.values()),
        "leak_total": sum(a["failure_counts"]["leak"] for a in per_set.values()),
        "retain_broken_total": sum(a["failure_counts"]["retain_broken"]
                                   for a in per_set.values()),
    }
    dataset_level["cell_pass_rate"] = (
        dataset_level["cell_pass_total"] / max(dataset_level["cells_total"], 1))
    summary = {
        "dataset": args.dataset,
        "edit_seeds": matrix["edit_seeds"],
        "held_out_routing": matrix["held_out_routing"],
        "simultaneity_guarantee": matrix["simultaneity_guarantee"],
        "dataset_level": dataset_level,
        "per_set": per_set,
    }
    with open(out_base / "matrix_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"MX3: dataset cell_pass_rate="
                f"{dataset_level['cell_pass_rate']:.3f} "
                f"({dataset_level['cell_pass_total']}/"
                f"{dataset_level['cells_total']})")
    return summary


# ====================================================================== #
# MX4: checkpoint archival -- durable local releases/ + HF upload for
# immutable URIs (results must be exactly RELOADABLE, not just auditable)
# ====================================================================== #
def archive_checkpoints(args, out_base, matrix, oracle_results, all_cells):
    logger.info("=" * 60)
    logger.info("MX4: CHECKPOINT ARCHIVAL (SHA-256 + immutable URI)")
    logger.info("=" * 60)
    commit = rv.git_commit_sha()
    rel = Path("releases") / f"e2c_matrix_{args.dataset}_{commit[:7]}"
    rel.mkdir(parents=True, exist_ok=True)
    entries = []

    def _add(src, rel_name, kind, key):
        src = Path(src)
        if not src.exists():
            logger.warning(f"MX4: missing checkpoint {src}")
            return
        dest = rel / rel_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        sha = rv.sha256_file(dest)
        entries.append({
            "kind": kind, "key": key, "file": rel_name,
            "sha256": sha, "bytes": dest.stat().st_size,
            "source_path": str(src.resolve()),
            "local_uri": dest.resolve().as_uri(),
        })

    # route checkpoints (frozen bases every cell starts from)
    route = Path(matrix["route"]["dir"])
    for tag in ("g_X_to_C", "h_C_to_Y"):
        _add(route / tag / "adapter_final" / "adapter_model.safetensors",
             f"route/{tag}.safetensors", "route", tag)
    # oracles
    for sid, meta in oracle_results.items():
        src = meta.get("source") or (out_base / "oracles" / sid
                                     / "adapter_final"
                                     / "adapter_model.safetensors")
        _add(src, f"oracles/{sid}.safetensors", "oracle", sid)
    # cells
    for cell_id, cell in all_cells.items():
        _add(out_base / "cells" / cell["set_id"] / f"seed_{cell['seed']}"
             / "edited_h" / "adapter_final" / "adapter_model.safetensors",
             f"cells/{cell['set_id']}/seed_{cell['seed']}.safetensors",
             "cell", cell_id)

    hf_ok, hf_repo = False, None
    try:
        from huggingface_hub import HfApi, whoami
        whoami()
        api = HfApi()
        api.create_repo(HF_ARCHIVE_REPO, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(rel), repo_id=HF_ARCHIVE_REPO,
            path_in_repo=f"matrix_{args.dataset}_{commit[:7]}",
            repo_type="model",
            commit_message=f"E2C-v3 matrix checkpoints ({args.dataset}) "
                           f"@ {commit[:7]}")
        hf_ok = True
        hf_repo = HF_ARCHIVE_REPO
        for e in entries:
            e["hf_uri"] = (f"https://huggingface.co/{HF_ARCHIVE_REPO}/"
                           f"resolve/main/matrix_{args.dataset}_"
                           f"{commit[:7]}/{e['file']}")
        logger.info(f"MX4: uploaded to HF {HF_ARCHIVE_REPO} "
                    f"(immutable URIs recorded)")
    except Exception as exc:
        logger.warning(f"MX4: HF upload unavailable ({str(exc)[:120]}); "
                       f"local archive + SHA-256 retained, upload before "
                       f"publication")
    with open(rel / "CHECKSUMS.txt", "w") as f:
        for e in entries:
            f.write(f"{e['sha256']}  {e['file']}\n")
    archive_manifest = {
        "dataset": args.dataset, "git_commit": commit,
        "release_dir": str(rel),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hf_repo": hf_repo, "hf_upload_ok": hf_ok,
        "n_files": len(entries),
        "entries": entries,
        "reload_note": "each cell_results.json checkpoint_sha256 matches the "
                       "archived file sha256; adapters reload via "
                       "rv.load_trained_weights on a Qwen3.5-9B + matching "
                       "LoRA config",
    }
    with open(rel / "archive_manifest.json", "w") as f:
        json.dump(archive_manifest, f, indent=2)
    logger.info(f"MX4: archived {len(entries)} checkpoints under {rel}")
    return archive_manifest


# ====================================================================== #
# Pilot checkpoint archival (durability requirement: the committed pilot
# result dirs hash the adapter weights but do not contain them; archive
# every pilot checkpoint with SHA-256 + immutable HF URI)
# ====================================================================== #
PILOT_CHECKPOINTS = {
    "ppubench": ("e2c_v3_real/outputs/realdata",
                 ["g_X_to_C", "h_C_to_Y", "h_oracle", "edited_h"]),
    "mllmu": ("e2c_mllmu/outputs/mllmu",
              ["g_X_to_C", "h_C_to_Y", "h_oracle", "edited_h"]),
    "salmu": ("e2c_salmu/outputs/salmu",
              ["g_X_to_C", "h_C_to_Y", "h_oracle", "edited_h"]),
    "celeba": ("e2c_celeba/outputs/celeba",
               ["g_X_to_C", "h_C_to_Y", "h_oracle", "edited_h"]),
}


def archive_pilots(args):
    logger.info("=" * 60)
    logger.info("MX4P: PILOT CHECKPOINT ARCHIVAL")
    logger.info("=" * 60)
    commit = rv.git_commit_sha()
    rel = Path("releases") / f"e2c_pilots_{commit[:7]}"
    rel.mkdir(parents=True, exist_ok=True)
    entries = []
    for ds, (root, tags) in PILOT_CHECKPOINTS.items():
        for tag in tags:
            src = Path(root) / tag / "adapter_final" \
                / "adapter_model.safetensors"
            if not src.exists():
                logger.warning(f"MX4P: missing {src}")
                continue
            dest = rel / "pilots" / ds / f"{tag}.safetensors"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            entries.append({
                "kind": "pilot", "dataset": ds, "tag": tag,
                "file": str(dest.relative_to(rel)),
                "sha256": rv.sha256_file(dest),
                "bytes": dest.stat().st_size,
                "source_path": str(src.resolve()),
                "local_uri": dest.resolve().as_uri(),
            })
    hf_ok, hf_repo = False, None
    try:
        from huggingface_hub import HfApi, whoami
        whoami()
        api = HfApi()
        api.create_repo(HF_ARCHIVE_REPO, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(rel), repo_id=HF_ARCHIVE_REPO,
            path_in_repo=f"pilots_{commit[:7]}", repo_type="model",
            commit_message=f"E2C-v3 pilot checkpoints @ {commit[:7]}")
        hf_ok, hf_repo = True, HF_ARCHIVE_REPO
        for e in entries:
            e["hf_uri"] = (f"https://huggingface.co/{HF_ARCHIVE_REPO}/"
                           f"resolve/main/pilots_{commit[:7]}/{e['file']}")
        logger.info(f"MX4P: uploaded {len(entries)} pilot checkpoints to "
                    f"HF {HF_ARCHIVE_REPO}")
    except Exception as exc:
        logger.warning(f"MX4P: HF upload unavailable ({str(exc)[:120]}); "
                       f"local archive + SHA-256 retained")
    with open(rel / "CHECKSUMS.txt", "w") as f:
        for e in entries:
            f.write(f"{e['sha256']}  {e['file']}\n")
    manifest = {"kind": "pilot_archive", "git_commit": commit,
                "release_dir": str(rel), "hf_repo": hf_repo,
                "hf_upload_ok": hf_ok, "n_files": len(entries),
                "entries": entries}
    with open(rel / "archive_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"MX4P: archived {len(entries)} pilot checkpoints "
                f"under {rel} (hf_upload={hf_ok})")
    return manifest


# ====================================================================== #
# MX5: run manifest (provenance, as in every other E2C-v3 runner)
# ====================================================================== #
def run_manifest(args, out_base, matrix, oracle_results, all_cells, summary,
                 archive, t_start, provenance):
    inputs = {
        "dataset_manifest": rv.sha256_file(MANIFEST_PATHS[args.dataset]),
        "matrix_manifest": rv.sha256_file(_matrix_path(args)),
        "g_cache_e2e_rows": rv.sha256_file(G_CACHE_PATHS[args.dataset]),
        "route_g": rv.sha256_file(Path(matrix["route"]["dir"]) / "g_X_to_C"
                                  / "adapter_final" / "adapter_model.safetensors"),
        "route_h": rv.sha256_file(Path(matrix["route"]["dir"]) / "h_C_to_Y"
                                  / "adapter_final" / "adapter_model.safetensors"),
    }
    ckpts = {}
    for sid, meta in oracle_results.items():
        ckpts[f"oracle_{sid}"] = meta.get(
            "source_sha256") or meta.get("checkpoint_sha256")
    for cell_id, cell in all_cells.items():
        ckpts[f"cell_{cell_id}"] = cell["checkpoint_sha256"]
    manifest = {
        "experiment": f"e2c_v3_matrix_{args.dataset}",
        "provenance": provenance,
        "inputs_sha256": inputs,
        "checkpoints_sha256": ckpts,
        "archive": {"release_dir": archive.get("release_dir"),
                    "hf_repo": archive["hf_repo"],
                    "hf_upload_ok": archive["hf_upload_ok"],
                    "n_files": archive["n_files"]},
        "matrix": {"sets": {e["set_id"]: {"targets": e["targets"],
                                          "mode": e["mode"]}
                            for e in matrix["sets"]},
                   "edit_seeds": matrix["edit_seeds"],
                   "oracle_seed": ORACLE_SEED},
        "results": {"dataset_level": summary["dataset_level"],
                    "per_set_pass": {s: a["failure_counts"]
                                     for s, a in summary["per_set"].items()}},
        "elapsed_sec": round(time.time() - t_start, 1),
        "gpu": torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else "cpu",
    }
    with open(out_base / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"MX5: run manifest written "
                f"(commit={provenance['commit'][:7]}, "
                f"checkpoints={len(ckpts)})")
    return manifest


# ====================================================================== #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=sorted(DATASET_ROOTS),
                   help="required unless --archive-pilots")
    p.add_argument("--archive-pilots", action="store_true",
                   help="archive committed pilot checkpoints (all datasets) "
                        "to releases/ + HF, then exit")
    p.add_argument("--phase", default="all",
                   choices=["all", "MX0", "MX1", "MX2", "MX3", "MX4", "MX5"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    p.add_argument("--route-steps", type=int, default=ROUTE_STEPS)
    p.add_argument("--route-warmup", type=int, default=ROUTE_WARMUP)
    p.add_argument("--route-lr", type=float, default=ROUTE_LR)
    p.add_argument("--route-repeat", type=int, default=ROUTE_REPEAT)
    p.add_argument("--ul-steps", type=int, default=UL_STEPS)
    p.add_argument("--ul-warmup", type=int, default=UL_WARMUP)
    p.add_argument("--ul-lr", type=float, default=UL_LR)
    p.add_argument("--ul-repeat", type=int, default=UL_REPEAT)
    p.add_argument("--max-gen-tokens", type=int, default=12)
    p.add_argument("--smoke", action="store_true",
                   help="tiny budget, first 2 sets x 1 seed, no HF upload")
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    if args.archive_pilots:
        archive_pilots(args)
        return 0
    if not args.dataset:
        raise SystemExit("--dataset is required unless --archive-pilots")
    if args.smoke:
        args.route_steps, args.route_warmup = 20, 2
        args.ul_steps, args.ul_warmup = 20, 2
        args.route_repeat = 2
        args.seeds = [17]
    # rv.create_adapter_model reads args.seed (base-model config seed; the
    # loaded checkpoints override weights, per-cell seeds are applied in MX2)
    args.seed = args.seeds[0]
    suffix = "_smoke" if args.smoke else ""
    out_base = OUT_ROOT / "outputs" / f"{args.dataset}{suffix}"
    out_base.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATHS[args.dataset]) as f:
        manifest = json.load(f)
    with open(G_CACHE_PATHS[args.dataset]) as f:
        g_rows = json.load(f)["rows"]

    commit = rv.git_commit_sha()
    dirty = rv.git_worktree_dirty()
    provenance = {
        "commit": commit, "script_sha256": rv.script_sha256(),
        "dirty": dirty, "clean_required": not args.smoke,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    logger.info(f"Provenance: commit={commit} "
                f"script_sha256={provenance['script_sha256'][:12]} "
                f"dirty={dirty} clean_required={not args.smoke}")
    if dirty and not args.smoke:
        raise RuntimeError(
            "tracked worktree is dirty; commit code/data before a full "
            "matrix run so results bind to the executing commit")

    if args.phase in ("all", "MX0"):
        matrix = build_matrix(args)
    else:
        matrix = load_matrix(args)
    if args.smoke:
        matrix = dict(matrix)
        matrix["sets"] = matrix["sets"][:1] + [
            s for s in matrix["sets"] if s["mode"] == "simultaneous"][:1]
        matrix["edit_seeds"] = [17]
        matrix = json.loads(json.dumps(matrix))  # deep copy

    oracle_results = {}
    all_cells = {}
    if args.phase in ("all", "MX1"):
        oracle_results = run_oracles(args, out_base, matrix, manifest)
    else:
        p = out_base / "oracles" / "oracles_summary.json"
        if p.exists():
            with open(p) as f:
                oracle_results = json.load(f)
    if args.phase in ("all", "MX2"):
        all_cells = run_cells(args, out_base, matrix, manifest, g_rows)
    else:
        idx = out_base / "cells" / "cells_index.json"
        if idx.exists():
            with open(idx) as f:
                cell_ids = list(json.load(f))
            for cell_id in cell_ids:
                sid, seed = cell_id.split("__seed")
                rp = out_base / "cells" / sid / f"seed_{seed}" \
                    / "cell_results.json"
                if rp.exists():
                    with open(rp) as f:
                        all_cells[cell_id] = json.load(f)
    summary = None
    if args.phase in ("all", "MX3") and all_cells:
        summary = aggregate(args, out_base, matrix, all_cells)
    archive = {"entries": [], "hf_repo": None, "hf_upload_ok": False,
               "n_files": 0, "release_dir": None}
    if args.phase in ("all", "MX4") and not args.smoke and all_cells:
        archive = archive_checkpoints(args, out_base, matrix, oracle_results,
                                      all_cells)
    if args.phase in ("all", "MX5") and not args.smoke and summary:
        run_manifest(args, out_base, matrix, oracle_results, all_cells,
                     summary, archive, t_start, provenance)
    logger.info("=" * 60)
    logger.info(f"MATRIX RUN ({args.dataset}) PHASE {args.phase} COMPLETE "
                f"({round(time.time() - t_start, 1)}s)")
    logger.info("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
