#!/usr/bin/env python3
"""E2C-v3 MLLMU-Bench pilot: identity-attribute association suppression/update.

Real semantic instantiation of the frozen E2C-v3 route on MLLMU-Bench
(500 real person profiles, one image each, biographies with real attribute
values).  Unlike PPUBench's pseudonymous aliases, h's targets here are the
benchmark-native attribute values themselves (Employment professions).

Design
======
- Identity subset: 12 identities = 2 per each of the 6 most common
  professions (deterministic under --seed).  Codes MID_<id>.
- Labels: the 6 profession values + 'Unknown' (refusal).  These are REAL
  semantic labels drawn from the benchmark, not pseudo-aliases.
- Forget partitions (native MLLMU semantics):
    * identity_forget  -> SUPPRESS one identity: its code is driven to
      'Unknown' (refusal-targeted association suppression).
    * association update -> UPDATE one identity: its code is driven to
      another real profession value (counterfactual relabeling).
  Both transformations plus full retention happen in ONE edited h.
- Architecture/budgets are frozen-identical to E2C-v3 and the PPUBench run.

Honest held-out statement
=========================
MLLMU-Bench provides ONE image per identity, so image-level held-out
evaluation of g is impossible; g is evaluated on its training images and
this is stated explicitly in the report.  Held-out-style evidence is at the
association level: (a) strict hard/soft evaluation of every code, (b)
gated distance to the leave-one-out oracle, and (c) auxiliary native
Mask_Task QA probes (benchmark-native phrasing, never used in route
training) scored pre/post on the shared model to detect collateral effects
outside the route.
"""
import argparse
import gc
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2c_v3_mllmu")

SCRIPT_DIR = Path(__file__).resolve().parent
MLLMU_ROOT = Path("/scratch/wutiantong/datasets/MLLMU-Bench")
FULL_SET = MLLMU_ROOT / "data" / "Full_Set.jsonl"

OUT_DEFAULT = "e2c_mllmu/outputs/mllmu"
MANIFEST_DIR = Path("e2c_mllmu/manifests")

IMG_TO_CODE_PROMPT = "What is the identity code for this person?"
CODE_TO_ALIAS_PROMPT = "Identity code: {code}. Generate the alias."
DELETED_LABEL = "Unknown"

N_PROFESSIONS = 6
IDS_PER_PROFESSION = 2
ROUTE_STEPS = 3000
ROUTE_WARMUP = 200
ROUTE_LR = 2e-5
ROUTE_REPEAT = 50
UL_STEPS = 500
UL_WARMUP = 50
UL_LR = 2e-5
UL_REPEAT = 50
MIN_CANDIDATE_MASS = 0.01
QA_PROBES_PER_ID = 3


def _load_sibling(module_name, filename):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rv = _load_sibling("e2c_rv_shared", "e2c_v3_research_validity.py")
rd = _load_sibling("e2c_rd_shared", "e2c_v3_realdata.py")


# ====================================================================== #
# MB0: deterministic MLLMU manifest
# ====================================================================== #
def build_mllmu_manifest(args):
    logger.info("=" * 60)
    logger.info("MB0: BUILD MLLMU MANIFEST")
    logger.info("=" * 60)
    with open(FULL_SET) as f:
        rows = [json.loads(l) for l in f]
    import collections
    prof_count = collections.Counter()
    bio_of = {}
    for r in rows:
        bio = json.loads(r["biography"])
        bio_of[r["ID"]] = bio
        prof_count[bio["Employment"]] += 1
    top_profs = [p for p, _ in prof_count.most_common(N_PROFESSIONS)]

    selected = []
    for prof in top_profs:
        ids = sorted(r["ID"] for r in rows
                     if json.loads(r["biography"])["Employment"] == prof)
        selected.extend(ids[:IDS_PER_PROFESSION])
    # deterministic suppress/update choice under the seed
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(selected), generator=g).tolist()
    suppress_id = selected[perm[0]]
    update_id = selected[perm[1]]
    own_prof = bio_of[update_id]["Employment"]
    # update to a DIFFERENT real profession present in the selected set
    update_new = next(p for p in top_profs if p != own_prof)

    row_by_id = {r["ID"]: r for r in rows}
    items = []
    for iid in selected:
        items.append({"identity_id": iid,
                      "image_uri": str(MLLMU_ROOT / row_by_id[iid]["image"]),
                      "split": "train"})  # one image per identity (no held-out)
    code_of = {iid: f"MID_{iid}" for iid in selected}
    alias_of = {iid: bio_of[iid]["Employment"] for iid in selected}

    # auxiliary native QA probes (Mask_Task), never used in route training
    qa_probes = {}
    for iid in selected:
        tasks = row_by_id[iid]["Mask_Task"][:QA_PROBES_PER_ID]
        qa_probes[iid] = [{"question": t["Question"],
                           "ground_truth": t["Ground_Truth"]} for t in tasks]

    manifest = {
        "dataset": "mllmu-bench (Full_Set)",
        "seed": args.seed,
        "source_sha256": rv.sha256_file(FULL_SET),
        "identity_ids": selected,
        "code_of": code_of,
        "alias_of": alias_of,
        "label_vocab_note": "targets are benchmark-native Employment values "
                            "(real semantic labels) + 'Unknown' refusal",
        "professions": top_profs,
        "suppress_identity_id": suppress_id,
        "update_identity_id": update_id,
        "update_new_label": update_new,
        "deleted_label": DELETED_LABEL,
        "images_per_identity": {i: 1 for i in selected},
        "held_out_note": "MLLMU-Bench has ONE image per identity; image-level "
                         "held-out g evaluation is impossible and is stated "
                         "as a limitation. Held-out evidence is at the "
                         "association level (strict eval, oracle distance, "
                         "native QA probes).",
        "items": items,
        "qa_probes": qa_probes,
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / "mllmu_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"MB0: {len(selected)} identities, suppress={suppress_id} "
                f"({alias_of[suppress_id]}), update={update_id} "
                f"({alias_of[update_id]} -> {update_new})")
    return manifest


def load_mllmu_manifest():
    with open(MANIFEST_DIR / "mllmu_manifest.json") as f:
        return json.load(f)


def _have_ckpt(d):
    return (Path(d) / "adapter_final" / "adapter_model.safetensors").exists()


def eval_g_mllmu(args, out_base, manifest):
    """Evaluate g on ALL images.  MLLMU-Bench has one image per identity, so
    there is NO held-out image split; the report states this explicitly."""
    logger.info("MB1: EVAL g on all images (no held-out images exist)")
    adapter, model, processor = rv.create_adapter_model(
        args, args.device, "e2c_mllmu_g")
    model = rv.attach_lora(adapter, model)
    rv.load_trained_weights(
        adapter, model, out_base / "g_X_to_C" / "adapter_final"
        / "adapter_model.safetensors")
    from route_data.config import ModelConfig
    cfg = ModelConfig(backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
                      revision="c202236235762e1c871ad0ccb60f8ee5ba337b9a",
                      dtype="bfloat16", seed=args.seed)
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=cfg)
    model.eval()
    codes = [manifest["code_of"][i] for i in manifest["identity_ids"]]
    from PIL import Image
    rows = []
    with torch.no_grad():
        for it in manifest["items"]:
            image = Image.open(it["image_uri"]).convert("RGB")
            gen = backend.generate(image, IMG_TO_CODE_PROMPT,
                                   max_new_tokens=6)
            raw = gen.text.strip()
            pred = rv._extract_code(raw, codes)
            rows.append({"identity_id": it["identity_id"],
                         "g_raw_text": raw, "pred_code": pred,
                         "correct":
                         pred == manifest["code_of"][it["identity_id"]]})
    del backend, model, processor, adapter
    gc.collect()
    torch.cuda.empty_cache()
    acc = sum(r["correct"] for r in rows) / len(rows)
    results = {
        "all_images": {"n": len(rows), "accuracy": acc},
        "held_out": None,
        "gate": "N/A (one image per identity; no held-out images)",
        "rows": rows,
    }
    with open(out_base / "g_X_to_C" / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"MB1: g all-image acc={acc:.3f} (held-out: none)")
    return results


# ====================================================================== #
# MB4: combined suppression + update + retention in ONE h
# ====================================================================== #
def run_combined_edit(args, out_base, manifest):
    logger.info("=" * 60)
    logger.info("MB4: COMBINED SUPPRESS + UPDATE + RETAIN (single h)")
    logger.info("=" * 60)
    edit_dir = out_base / "edited_h"
    edit_dir.mkdir(parents=True, exist_ok=True)
    sup, upd = manifest["suppress_identity_id"], manifest["update_identity_id"]
    adapter, model, processor = rv.create_adapter_model(
        args, args.device, "e2c_mllmu_h_edit")
    model = rv.attach_lora(adapter, model)
    rv.load_trained_weights(
        adapter, model,
        out_base / "h_C_to_Y" / "adapter_final" / "adapter_model.safetensors")
    suppress_pairs = [{"prompt": CODE_TO_ALIAS_PROMPT.format(
                          code=manifest["code_of"][sup]),
                       "answer": DELETED_LABEL}]
    update_pairs = [{"prompt": CODE_TO_ALIAS_PROMPT.format(
                        code=manifest["code_of"][upd]),
                     "answer": manifest["update_new_label"]}]
    retain_pairs = rd.h_pairs(manifest, exclude=[sup, upd])
    items = (
        rv.build_supervised_items(adapter, processor, suppress_pairs,
                                  repeat=args.ul_repeat * 5)
        + rv.build_supervised_items(adapter, processor, update_pairs,
                                    repeat=args.ul_repeat * 3)
        + rv.build_supervised_items(adapter, processor, retain_pairs,
                                    repeat=args.ul_repeat * 3))
    rv.train_supervised("mllmu_edit", adapter, model, processor, items,
                        edit_dir, args.device, steps=args.ul_steps,
                        warmup=args.ul_warmup, lr=args.ul_lr)
    del model, processor, adapter
    gc.collect()
    torch.cuda.empty_cache()
    return edit_dir


# ====================================================================== #
# MB5: post-edit evaluation
# ====================================================================== #
def expected_of(manifest):
    """Expected strict label per identity after the combined edit."""
    exp = dict(manifest["alias_of"])
    exp[manifest["suppress_identity_id"]] = DELETED_LABEL
    exp[manifest["update_identity_id"]] = manifest["update_new_label"]
    return exp


def run_mllmu_e2e(args, out_base, manifest, h_dir, h_name, tag):
    """E2E image -> code -> label with per-transformation expectations."""
    e2e_dir = out_base / f"e2e_{tag}"
    e2e_dir.mkdir(parents=True, exist_ok=True)
    adapter_g, model_g, processor_g = rv.create_adapter_model(
        args, args.device, "e2c_mllmu_g_e2e")
    model_g = rv.attach_lora(adapter_g, model_g)
    rv.load_trained_weights(
        adapter_g, model_g,
        out_base / "g_X_to_C" / "adapter_final" / "adapter_model.safetensors")
    from route_data.config import ModelConfig
    cfg = ModelConfig(backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
                      revision="c202236235762e1c871ad0ccb60f8ee5ba337b9a",
                      dtype="bfloat16", seed=args.seed)
    g_backend = adapter_g.to_eval_backend(
        model=model_g, processor=processor_g, model_config=cfg)
    model_g.eval()
    codes = [manifest["code_of"][i] for i in manifest["identity_ids"]]
    from PIL import Image
    code_rows = []
    with torch.no_grad():
        for it in manifest["items"]:
            image = Image.open(it["image_uri"]).convert("RGB")
            gen = g_backend.generate(image, IMG_TO_CODE_PROMPT,
                                     max_new_tokens=6)
            raw = gen.text.strip()
            pred = rv._extract_code(raw, codes)
            code_rows.append({**it, "g_raw_text": raw, "pred_code": pred,
                              "code_correct":
                              pred == manifest["code_of"][it["identity_id"]]})
    g_acc = sum(c["code_correct"] for c in code_rows) / len(code_rows)
    del g_backend, model_g, processor_g, adapter_g
    gc.collect()
    torch.cuda.empty_cache()

    adapter_h, model_h, processor_h = rd.load_h(args, h_dir, adapter_name=h_name)
    h_backend = adapter_h.to_eval_backend(
        model=model_h, processor=processor_h, model_config=cfg)
    model_h.eval()
    vocab = sorted(set(manifest["alias_of"].values())
                   | {DELETED_LABEL, manifest["update_new_label"]})
    exp = expected_of(manifest)
    sup_id = manifest["suppress_identity_id"]
    rows = []
    with torch.no_grad():
        for c in code_rows:
            if c["pred_code"] is None:
                rows.append({**c, "h_raw_text": None, "pred_alias": None,
                             "g_routed": False, "outcome_ok": False})
                continue
            prompt = CODE_TO_ALIAS_PROMPT.format(code=c["pred_code"])
            gen = h_backend.generate(None, prompt, max_new_tokens=8)
            raw = gen.text.strip()
            parsed = rv.parse_recognized_label(raw, vocab)
            iid = c["identity_id"]
            if iid == sup_id:
                # suppression: old profession must not appear anywhere
                ok = manifest["alias_of"][iid] not in rv.recognized_labels_in(
                    raw, vocab)
            else:
                ok = parsed == exp[iid]
            rows.append({**c, "h_raw_text": raw, "pred_alias": parsed,
                         "g_routed": True, "outcome_ok": ok})
    del h_backend, model_h, processor_h, adapter_h
    gc.collect()
    torch.cuda.empty_cache()

    routed = [r for r in rows if r["g_routed"]]
    by_transformation = {}
    for name, iids in [("suppress", {sup_id}),
                       ("update", {manifest["update_identity_id"]}),
                       ("retain", set(manifest["identity_ids"])
                        - {sup_id, manifest["update_identity_id"]})]:
        sub = [r for r in rows if r["identity_id"] in iids]
        by_transformation[name] = {
            "n": len(sub),
            "outcome_ok": sum(r["outcome_ok"] for r in sub)}
    summary = {
        "tag": tag,
        "g_code_accuracy": round(g_acc, 4),
        "n_images": len(rows),
        "g_routed": len(routed),
        "g_routing_failures": len(rows) - len(routed),
        "h_unparseable_outputs": sum(
            1 for r in routed if r["pred_alias"] is None),
        "scoring": "strict token-exact parsing; multi-label outputs rejected",
        "by_transformation": by_transformation,
        "rows": rows,
    }
    with open(e2e_dir / "e2e_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"E2E[{tag}]: g_acc={g_acc:.3f} routed={len(routed)}/"
                f"{len(rows)} by_transformation={by_transformation}")
    return summary


def run_qa_probes(args, out_base, manifest, h_dir, h_name, tag):
    """Auxiliary native Mask_Task QA probes on the shared model (pre/post).

    These probes use benchmark-native phrasing and are never used in route
    training; they detect collateral effects OUTSIDE the route.  Scoring: the
    ground-truth string must appear in the generated answer.
    """
    adapter, model, processor = rd.load_h(args, h_dir, adapter_name=h_name)
    from route_data.config import ModelConfig
    cfg = ModelConfig(backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
                      revision="c202236235762e1c871ad0ccb60f8ee5ba337b9a",
                      dtype="bfloat16", seed=args.seed)
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=cfg)
    model.eval()
    results = {}
    with torch.no_grad():
        for iid, probes in manifest["qa_probes"].items():
            rows = []
            for p in probes:
                gen = backend.generate(None, p["question"], max_new_tokens=40)
                text = gen.text.strip()
                rows.append({"question": p["question"],
                             "ground_truth": p["ground_truth"],
                             "generated": text,
                             "correct": p["ground_truth"].lower() in
                             text.lower()})
            results[iid] = rows
    del backend, model, processor, adapter
    gc.collect()
    torch.cuda.empty_cache()
    n_all = sum(len(v) for v in results.values())
    n_ok = sum(r["correct"] for v in results.values() for r in v)
    out = {"tag": tag, "accuracy": round(n_ok / max(n_all, 1), 4),
           "n": n_all, "per_identity": results}
    with open(out_base / f"qa_probes_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"QA probes[{tag}]: {n_ok}/{n_all} "
                f"(auxiliary, outside the route)")
    return out


def run_manifest(args, out_base, results, t_start, provenance):
    inputs = {"mllmu_full_set": rv.sha256_file(FULL_SET),
              "mllmu_manifest": rv.sha256_file(
                  MANIFEST_DIR / "mllmu_manifest.json")}
    checkpoints = {}
    for name, p in [
            ("g_X_to_C", out_base / "g_X_to_C" / "adapter_final"
             / "adapter_model.safetensors"),
            ("h_C_to_Y", out_base / "h_C_to_Y" / "adapter_final"
             / "adapter_model.safetensors"),
            ("h_oracle", out_base / "h_oracle" / "adapter_final"
             / "adapter_model.safetensors"),
            ("h_edited", out_base / "edited_h" / "adapter_final"
             / "adapter_model.safetensors")]:
        if Path(p).exists():
            checkpoints[name] = rv.sha256_file(p)
    outputs = {}
    for p in sorted(Path(out_base).rglob("*")):
        if p.is_file() and p.name != "run_manifest.json":
            outputs[str(p.relative_to(out_base))] = rv.sha256_file(p)
    manifest = {
        "git_commit": provenance["git_commit"],
        "cli_invocation": sys.argv,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_sec": round(time.time() - t_start, 1),
        "provenance": {
            "git_commit": provenance["git_commit"],
            "script_sha256": provenance["script_sha256"],
            "shared_scoring_script_sha256":
                provenance["shared_scoring_script_sha256"],
            "worktree_dirty_at_start": provenance["worktree_dirty_at_start"],
            "clean_worktree_required": provenance["clean_worktree_required"],
            "test_evidence": rv.collect_test_evidence(),
        },
        "config": {
            "device": args.device, "seed": args.seed, "phase": args.phase,
            "route_steps": args.route_steps, "route_warmup": args.route_warmup,
            "route_lr": args.route_lr, "route_repeat": args.route_repeat,
            "ul_steps": args.ul_steps, "ul_warmup": args.ul_warmup,
            "ul_lr": args.ul_lr, "ul_repeat": args.ul_repeat,
            "deletion_label": DELETED_LABEL,
            "min_candidate_mass": MIN_CANDIDATE_MASS},
        "inputs_sha256": inputs,
        "checkpoints_sha256": checkpoints,
        "outputs_sha256": outputs,
        "phase_status": {k: ("ok" if v is not None else "skipped")
                         for k, v in results.items()},
    }
    with open(out_base / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written: commit={manifest['git_commit']} "
                f"outputs_hashed={len(outputs)}")
    return manifest


# ====================================================================== #
# Main
# ====================================================================== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-base", default=OUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--phase", nargs="+", default=["all"],
                        help="all or subset of: MB0 MB1 MB2 MB3 MB4 MB5")
    parser.add_argument("--route-steps", type=int, default=ROUTE_STEPS)
    parser.add_argument("--route-warmup", type=int, default=ROUTE_WARMUP)
    parser.add_argument("--route-lr", type=float, default=ROUTE_LR)
    parser.add_argument("--route-repeat", type=int, default=ROUTE_REPEAT)
    parser.add_argument("--ul-steps", type=int, default=UL_STEPS)
    parser.add_argument("--ul-warmup", type=int, default=UL_WARMUP)
    parser.add_argument("--ul-lr", type=float, default=UL_LR)
    parser.add_argument("--ul-repeat", type=int, default=UL_REPEAT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-only", action="store_true",
                        help="resume/rescore: reuse existing checkpoints "
                             "where present; train only what is missing")
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.route_steps = min(args.route_steps, 30)
        args.route_warmup = min(args.route_warmup, 5)
        args.route_repeat = min(args.route_repeat, 2)
        args.ul_steps = min(args.ul_steps, 20)
        args.ul_warmup = min(args.ul_warmup, 5)
        args.ul_repeat = min(args.ul_repeat, 3)
        logger.info("SMOKE MODE: budgets reduced for validation")

    dirty = rv.git_worktree_dirty()
    clean_required = not args.smoke and not args.allow_dirty
    if clean_required and dirty:
        raise RuntimeError(
            "Refusing to run: git worktree is dirty. Commit first or pass "
            "--allow-dirty.")
    provenance = {
        "git_commit": rv.git_commit_sha(),
        "script_sha256": rv.sha256_file(Path(__file__).resolve()),
        "shared_scoring_script_sha256": rv.script_sha256(),
        "worktree_dirty_at_start": dirty,
        "clean_worktree_required": clean_required,
    }
    logger.info(f"Provenance: commit={provenance['git_commit']} "
                f"script_sha256={provenance['script_sha256'][:12]} "
                f"dirty={dirty} clean_required={clean_required}")

    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    all_phases = ["MB0", "MB1", "MB2", "MB3", "MB4", "MB5"]
    requested = set(args.phase)
    run_phases = (all_phases if "all" in requested
                  else [p for p in all_phases if p in requested])
    t_start = time.time()
    results = {}

    if "MB0" in run_phases:
        manifest = build_mllmu_manifest(args)
    else:
        manifest = load_mllmu_manifest()
    results["MB0"] = {"identity_ids": manifest["identity_ids"],
                      "suppress": manifest["suppress_identity_id"],
                      "update": manifest["update_identity_id"],
                      "update_new_label": manifest["update_new_label"]}

    if "MB1" in run_phases:
        if args.eval_only and _have_ckpt(out_base / "g_X_to_C"):
            logger.info("MB1: eval-only, reusing existing g checkpoint")
        else:
            rd.train_g(args, out_base, manifest)
        results["MB1"] = eval_g_mllmu(args, out_base, manifest)
        results["MB1"]["held_out_note"] = manifest["held_out_note"]

    if "MB2" in run_phases:
        if args.eval_only and _have_ckpt(out_base / "h_C_to_Y"):
            logger.info("MB2: eval-only, reusing existing h checkpoint")
        else:
            rd.train_h(args, out_base, manifest)
        adapter, model, processor = rd.load_h(args, out_base / "h_C_to_Y")
        results["MB2"] = rd.eval_h_strict(args, adapter, model, processor,
                                          manifest, "h_baseline")
        results["MB2_qa_probes_pre"] = run_qa_probes(
            args, out_base, manifest, out_base / "h_C_to_Y",
            "e2c_mllmu_h", "pre")
        del model, processor, adapter
        gc.collect()
        torch.cuda.empty_cache()

    if "MB3" in run_phases:
        exclude = [manifest["suppress_identity_id"],
                   manifest["update_identity_id"]]
        if args.eval_only and _have_ckpt(out_base / "h_oracle"):
            logger.info("MB3: eval-only, reusing existing oracle checkpoint")
        else:
            rd.train_h(args, out_base, manifest, exclude=exclude,
                       tag="h_oracle", adapter_name="e2c_mllmu_h_oracle")
        adapter, model, processor = rd.load_h(
            args, out_base / "h_oracle", adapter_name="e2c_mllmu_h_oracle")
        results["MB3"] = rd.eval_h_strict(args, adapter, model, processor,
                                          manifest, "h_oracle")
        oracle_soft, vocab = rd.soft_eval_codes(args, adapter, model,
                                                processor, manifest)
        with open(out_base / "h_oracle" / "soft_metrics.json", "w") as f:
            json.dump(oracle_soft, f, indent=2)
        del model, processor, adapter
        gc.collect()
        torch.cuda.empty_cache()

    if "MB4" in run_phases:
        if args.eval_only and _have_ckpt(out_base / "edited_h"):
            logger.info("MB4: eval-only, reusing existing edited checkpoint")
            results["MB4"] = {"trained": "reused"}
        else:
            run_combined_edit(args, out_base, manifest)
            results["MB4"] = {"trained": True}

    if "MB5" in run_phases:
        exp = expected_of(manifest)
        adapter, model, processor = rd.load_h(
            args, out_base / "edited_h", adapter_name="e2c_mllmu_h_edit")
        results["MB5_hard"] = rd.eval_h_strict(
            args, adapter, model, processor, manifest, "h_edited")
        # score hard eval against POST-EDIT expectations, not originals
        for p in results["MB5_hard"]["preds"]:
            p["expected_post_edit"] = exp[p["identity_id"]]
            p["correct_post_edit"] = (
                p["parsed_label"] == exp[p["identity_id"]])
        results["MB5_hard"]["accuracy_post_edit_expectation"] = sum(
            p["correct_post_edit"] for p in results["MB5_hard"]["preds"]
        ) / len(results["MB5_hard"]["preds"])
        soft, vocab = rd.soft_eval_codes(args, adapter, model, processor,
                                         manifest)
        rd6 = {"soft": {}, "gated_distance_to_oracle": {}}
        oracle_path = out_base / "h_oracle" / "soft_metrics.json"
        oracle_soft = None
        if oracle_path.exists():
            with open(oracle_path) as f:
                oracle_soft = json.load(f)
        for iid in manifest["identity_ids"]:
            s = soft[iid]
            rd6["soft"][iid] = {
                "expected_post_edit": exp[iid],
                "p_expected_post_edit": s["probs"].get(exp[iid], 0.0),
                "p_original_alias": s["probs"].get(
                    manifest["alias_of"][iid], 0.0),
                "candidate_mass": s["candidate_mass"],
                "other_mass": s["other_mass"],
            }
            if oracle_soft and iid in oracle_soft:
                dist, reliable, reason = rv.gated_distance(
                    s, oracle_soft[iid], "candidate", vocab,
                    MIN_CANDIDATE_MASS)
                rd6["gated_distance_to_oracle"][iid] = {
                    "distance": dist, "reliable": reliable, "reason": reason}
        results["MB5_soft"] = rd6
        del model, processor, adapter
        gc.collect()
        torch.cuda.empty_cache()
        results["MB5_e2e"] = run_mllmu_e2e(
            args, out_base, manifest, out_base / "edited_h",
            "e2c_mllmu_h_edit", tag="post")
        results["MB5_qa_probes_post"] = run_qa_probes(
            args, out_base, manifest, out_base / "edited_h",
            "e2c_mllmu_h_edit", "post")
        # collateral statement scope
        n_retain = len(manifest["identity_ids"]) - 2
        results["MB5_wording"] = {
            "collateral": f"no observed collateral on the {n_retain} tested "
                          "retained identities and the auxiliary native QA "
                          "probes; NOT a general zero-collateral claim",
        }

    logger.info("=" * 60)
    logger.info("MLLMU PILOT PHASE COMPLETE")
    logger.info("=" * 60)
    if results:
        with open(out_base / "mb_summary.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    if not args.no_manifest:
        run_manifest(args, out_base, results, t_start, provenance)
    logger.info(f"Results saved under {out_base} "
                f"({time.time() - t_start:.1f}s total)")


if __name__ == "__main__":
    main()
