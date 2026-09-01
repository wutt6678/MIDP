#!/usr/bin/env python3
"""E2C-v3 real-dataset implementation (ppubench).

Ports the frozen E2C-v3 discrete-bottleneck pipeline from the synthetic
proof-of-concept to REAL data: the ppubench benchmark (5 real identities,
15 images each, CelebA-style weak attribute labels, pre-defined
identity_forget unlearning split).

Architecture (frozen, identical to E2C-v3):
    X -> C -> Y,  g: image -> identity code (LoRA, NEVER edited after route
    establishment), h: code -> pseudo-alias (LoRA, the only edited module).

Phases
======
- RD0: build the deterministic real-data manifest (identity -> code -> alias,
  train/test image split) from the ppubench annotation parquet.
- RD1: train g (X -> C) on train images; gate on held-out test accuracy.
- RD2: train h (C -> Y) on code -> alias pairs; gate on strict hard accuracy.
- RD3: pre-edit end-to-end baseline (image -> code -> alias) with strict
  scoring (multi-label outputs are rejected, never resolved to first match).
- RD4: leave-one-out h oracle (retrain h WITHOUT the forget identity),
  training-matched to RD2 -- the retrain-equivalence reference.
- RD5: refusal-targeted association suppression of the forget identity
  (identity_forget split) with supervised retention of the rest.  This is
  abstention editing; it is NOT claimed to be retrain-equivalent.
- RD6: post-edit evaluation: soft metrics (full-sequence probabilities,
  candidate/OTHER mass, gated distance-to-oracle), strict hard evaluation,
  post-edit e2e, and the structural route-freeze control (g accuracy
  pre/post must be unchanged because g is never edited).
- RD7: run manifest binding commit, script SHA-256, dirty flag, test
  evidence, inputs, checkpoints, and output hashes.  Full runs require a
  clean git worktree.

Scoring reuses the research-validity helpers verbatim (strict token-exact
parsing with multi-label rejection, full-sequence probabilities, candidate
mass gating), so synthetic and real results are directly comparable.
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
logger = logging.getLogger("e2c_v3_realdata")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("data/processed/Qwen_Qwen3.5-9B/ppubench")
ANNOTATIONS_PARQUET = DATA_ROOT / "ppubench_celeba40_image_annotations.parquet"
SPLIT_MANIFEST = DATA_ROOT / "ppubench_split_manifest.json"

OUT_DEFAULT = "e2c_v3_real/outputs/realdata"
MANIFEST_DIR = Path("e2c_v3_real/manifests")

# Same prompts as the frozen E2C-v3 pipeline.
IMG_TO_CODE_PROMPT = "What is the identity code for this person?"
CODE_TO_ALIAS_PROMPT = "Identity code: {code}. Generate the alias."

# Refusal / abstention target (outside the alias space), same as E2C-v3.
DELETED_LABEL = "Unknown"

# Neutral pseudo-alias pool (the benchmark is anonymized; identities carry no
# real names, so h targets are synthetic aliases exactly as in E2C-v3).
ALIAS_POOL = ["Kael", "Mira", "Tova", "Rin", "Osha",
              "Dax", "Lume", "Seri", "Vann", "Oden"]

# Training budgets: route establishment is training-matched to the frozen
# Phase-C defaults; editing uses the established U0-U8 budget.
ROUTE_STEPS = 3000
ROUTE_WARMUP = 200
ROUTE_LR = 2e-5
ROUTE_REPEAT = 50
UL_STEPS = 500
UL_WARMUP = 50
UL_LR = 2e-5
UL_REPEAT = 50
IMAGES_PER_ID_TRAIN = 12
IMAGES_PER_ID_TEST = 3
MIN_IMAGES_PER_ID = 5  # identities below this are excluded (no test split)
G_GATE_MIN_TEST_ACC = 0.90
MIN_CANDIDATE_MASS = 0.01


def _load_sibling(module_name, filename):
    """Import a sibling script as a module (shared helper reuse)."""
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rv = _load_sibling("e2c_rv_shared", "e2c_v3_research_validity.py")


# ====================================================================== #
# RD0: deterministic real-data manifest
# ====================================================================== #
def build_real_manifest(args, out_base):
    logger.info("=" * 60)
    logger.info("RD0: BUILD REAL-DATA MANIFEST (ppubench)")
    logger.info("=" * 60)
    import pandas as pd
    df = pd.read_parquet(ANNOTATIONS_PARQUET)
    img_by_id = {}
    for iid, grp in df.groupby("identity_id"):
        img_by_id[str(iid)] = sorted(set(grp["image_uri"].tolist()))

    excluded = sorted(i for i, imgs in img_by_id.items()
                      if len(imgs) < MIN_IMAGES_PER_ID)
    identity_ids = sorted(i for i in img_by_id if i not in set(excluded))
    if len(identity_ids) > len(ALIAS_POOL):
        raise ValueError("not enough aliases for identities")
    # deterministic alias assignment by seed
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(ALIAS_POOL), generator=g).tolist()
    alias_of = {iid: ALIAS_POOL[perm[i]] for i, iid in enumerate(identity_ids)}
    code_of = {iid: f"RID_{iid}" for iid in identity_ids}

    with open(SPLIT_MANIFEST) as f:
        split_manifest = json.load(f)
    forget_split = next(s for s in split_manifest["splits"]
                        if s["name"] == "identity_forget")
    forget_ids = [str(i) for i in forget_split["forget_identity_ids"]]
    unknown = [i for i in forget_ids if i not in identity_ids]
    if unknown:
        raise ValueError(f"forget identities missing/excluded: {unknown}")

    items = []
    for iid in identity_ids:
        imgs = img_by_id[iid]
        n_test = min(IMAGES_PER_ID_TEST, len(imgs) - 1)
        for j, uri in enumerate(imgs):
            split = ("test" if j >= len(imgs) - n_test else "train")
            items.append({"identity_id": iid, "image_uri": uri,
                          "split": split})
    manifest = {
        "dataset": "ppubench",
        "seed": args.seed,
        "annotations_sha256": rv.sha256_file(ANNOTATIONS_PARQUET),
        "split_manifest_sha256": rv.sha256_file(SPLIT_MANIFEST),
        "identity_ids": identity_ids,
        "excluded_identities": {i: len(img_by_id[i]) for i in excluded},
        "exclusion_rule": f"fewer than {MIN_IMAGES_PER_ID} distinct images",
        "code_of": code_of,
        "alias_of": alias_of,
        "deleted_label": DELETED_LABEL,
        "forget_identity_ids": forget_ids,
        "images_per_identity": {i: len(img_by_id[i]) for i in identity_ids},
        "split": {"rule": "last min(3, n-1) sorted images per identity are "
                          "test; the rest are train",
                  "test_per_id": IMAGES_PER_ID_TEST},
        "items": items,
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / "realdata_identity_mapping.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"RD0: {len(identity_ids)} identities, {len(items)} images, "
                f"forget={forget_ids}, manifest={path}")
    return manifest


def load_real_manifest():
    with open(MANIFEST_DIR / "realdata_identity_mapping.json") as f:
        return json.load(f)


# ====================================================================== #
# RD1: route g (X -> C)
# ====================================================================== #
def train_g(args, out_base, manifest):
    logger.info("=" * 60)
    logger.info("RD1: TRAIN g (image -> code) on real images")
    logger.info("=" * 60)
    g_dir = out_base / "g_X_to_C"
    g_dir.mkdir(parents=True, exist_ok=True)
    train_items = [it for it in manifest["items"] if it["split"] == "train"]
    from PIL import Image
    sup_items = []
    adapter, model, processor = rv.create_adapter_model(
        args, args.device, "e2c_real_g")
    model = rv.attach_lora(adapter, model)
    for it in train_items:
        image = Image.open(it["image_uri"]).convert("RGB")
        ex = adapter.build_supervised_example(
            processor, image=image, prompt=IMG_TO_CODE_PROMPT,
            answer_text=manifest["code_of"][it["identity_id"]])
        for _ in range(args.route_repeat):
            sup_items.append(ex)
    rv.train_supervised("g_real", adapter, model, processor, sup_items,
                        g_dir, args.device, steps=args.route_steps,
                        warmup=args.route_warmup, lr=args.route_lr)
    del model, processor, adapter
    gc.collect()
    torch.cuda.empty_cache()
    return g_dir


def eval_g(args, out_base, manifest):
    logger.info("RD1: EVAL g on held-out test images")
    adapter, model, processor = rv.create_adapter_model(
        args, args.device, "e2c_real_g")
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
                                   max_new_tokens=12)
            raw = gen.text.strip()
            pred = rv._extract_code(raw, codes)
            rows.append({"identity_id": it["identity_id"],
                         "split": it["split"], "g_raw_text": raw,
                         "pred_code": pred,
                         "correct": pred == manifest["code_of"][it["identity_id"]]})
    del backend, model, processor, adapter
    gc.collect()
    torch.cuda.empty_cache()
    by_split = {}
    for split in ("train", "test"):
        sub = [r for r in rows if r["split"] == split]
        by_split[split] = {"n": len(sub),
                           "accuracy": sum(r["correct"] for r in sub) / len(sub)}
    results = {"per_split": by_split, "rows": rows,
               "gate": ("PASS" if by_split["test"]["accuracy"]
                        >= G_GATE_MIN_TEST_ACC else "FAIL")}
    with open(out_base / "g_X_to_C" / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"RD1: g test_acc={by_split['test']['accuracy']:.3f} "
                f"gate={results['gate']}")
    return results


# ====================================================================== #
# RD2: route h (C -> Y)
# ====================================================================== #
def h_pairs(manifest, exclude=()):
    return [{"prompt": CODE_TO_ALIAS_PROMPT.format(
                code=manifest["code_of"][iid]),
             "answer": manifest["alias_of"][iid]}
            for iid in manifest["identity_ids"] if iid not in set(exclude)]


def train_h(args, out_base, manifest, exclude=(), tag="h_C_to_Y",
            adapter_name="e2c_real_h", steps=None):
    steps = steps or args.route_steps
    h_dir = out_base / tag
    h_dir.mkdir(parents=True, exist_ok=True)
    adapter, model, processor = rv.create_adapter_model(
        args, args.device, adapter_name)
    model = rv.attach_lora(adapter, model)
    pairs = h_pairs(manifest, exclude)
    items = rv.build_supervised_items(adapter, processor, pairs,
                                      repeat=args.route_repeat)
    rv.train_supervised(tag, adapter, model, processor, items, h_dir,
                        args.device, steps=steps, warmup=args.route_warmup,
                        lr=args.route_lr)
    del model, processor, adapter
    gc.collect()
    torch.cuda.empty_cache()
    return h_dir


def load_h(args, adapter_dir, adapter_name="e2c_real_h"):
    adapter, model, processor = rv.create_adapter_model(
        args, args.device, adapter_name)
    model = rv.attach_lora(adapter, model)
    rv.load_trained_weights(adapter, model,
                            Path(adapter_dir) / "adapter_final"
                            / "adapter_model.safetensors")
    return adapter, model, processor


def eval_h_strict(args, adapter, model, processor, manifest, tag):
    """Hard evaluation with strict parsing (multi-label rejection)."""
    from route_data.config import ModelConfig
    cfg = ModelConfig(backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
                      revision="c202236235762e1c871ad0ccb60f8ee5ba337b9a",
                      dtype="bfloat16", seed=args.seed)
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=cfg)
    model.eval()
    vocab = sorted(set(manifest["alias_of"].values()) | {DELETED_LABEL})
    preds = []
    with torch.no_grad():
        for iid in manifest["identity_ids"]:
            prompt = CODE_TO_ALIAS_PROMPT.format(code=manifest["code_of"][iid])
            gen = backend.generate(None, prompt, max_new_tokens=8)
            raw = gen.text.strip()
            parsed = rv.parse_recognized_label(raw, vocab)
            preds.append({"identity_id": iid, "raw": raw,
                          "parsed_label": parsed,
                          "correct": parsed == manifest["alias_of"][iid]})
    acc = sum(p["correct"] for p in preds) / len(preds)
    logger.info(f"[{tag}] strict hard accuracy={acc:.3f}")
    return {"accuracy": acc, "preds": preds}


# ====================================================================== #
# RD3/RD6: end-to-end image -> code -> alias
# ====================================================================== #
def run_e2e(args, out_base, manifest, h_adapter_dir, h_adapter_name, tag,
            delete_set):
    e2e_dir = out_base / f"e2e_{tag}"
    e2e_dir.mkdir(parents=True, exist_ok=True)
    # stage 1: frozen g
    adapter_g, model_g, processor_g = rv.create_adapter_model(
        args, args.device, "e2c_real_g_e2e")
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

    # stage 2: h (edited or baseline) on the routed codes
    adapter_h, model_h, processor_h = load_h(
        args, h_adapter_dir, adapter_name=h_adapter_name)
    h_backend = adapter_h.to_eval_backend(
        model=model_h, processor=processor_h, model_config=cfg)
    model_h.eval()
    vocab = sorted(set(manifest["alias_of"].values()) | {DELETED_LABEL})
    alias_of = manifest["alias_of"]
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
            if iid in delete_set:
                ok = alias_of.get(iid) not in rv.recognized_labels_in(
                    raw, vocab)
            else:
                ok = parsed == alias_of.get(iid)
            rows.append({**c, "h_raw_text": raw, "pred_alias": parsed,
                         "g_routed": True, "outcome_ok": ok})
    del h_backend, model_h, processor_h, adapter_h
    gc.collect()
    torch.cuda.empty_cache()

    routed = [r for r in rows if r["g_routed"]]
    forget_rows = [r for r in rows if r["identity_id"] in delete_set]
    retain_rows = [r for r in rows if r["identity_id"] not in delete_set]
    summary = {
        "tag": tag,
        "g_code_accuracy": round(g_acc, 4),
        "n_images": len(rows),
        "g_routed": len(routed),
        "g_routing_failures": len(rows) - len(routed),
        "h_unparseable_outputs": sum(
            1 for r in routed if r["pred_alias"] is None),
        "scoring": "strict token-exact parsing; multi-label outputs rejected",
        "deleted_images": len(forget_rows),
        "deleted_e2e_suppressed": sum(r["outcome_ok"] for r in forget_rows),
        "retained_images": len(retain_rows),
        "retained_e2e_correct": sum(r["outcome_ok"] for r in retain_rows),
        "rows": rows,
    }
    with open(e2e_dir / "e2e_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"E2E[{tag}]: g_acc={g_acc:.3f} routed={len(routed)}/"
                f"{len(rows)} deleted_suppressed="
                f"{summary['deleted_e2e_suppressed']}/{len(forget_rows)} "
                f"retained={summary['retained_e2e_correct']}/{len(retain_rows)}")
    return summary


# ====================================================================== #
# RD5: refusal-targeted suppression (abstention editing)
# ====================================================================== #
def run_suppression(args, out_base, manifest):
    logger.info("=" * 60)
    logger.info("RD5: REFUSAL-TARGETED SUPPRESSION (abstention editing)")
    logger.info("=" * 60)
    edit_dir = out_base / "edited_h"
    edit_dir.mkdir(parents=True, exist_ok=True)
    forget_ids = manifest["forget_identity_ids"]
    adapter, model, processor = rv.create_adapter_model(
        args, args.device, "e2c_real_h_edit")
    model = rv.attach_lora(adapter, model)
    rv.load_trained_weights(
        adapter, model,
        out_base / "h_C_to_Y" / "adapter_final" / "adapter_model.safetensors")
    forget_pairs = [{"prompt": CODE_TO_ALIAS_PROMPT.format(
                        code=manifest["code_of"][iid]),
                     "answer": DELETED_LABEL} for iid in forget_ids]
    items = (
        rv.build_supervised_items(adapter, processor, forget_pairs,
                                  repeat=args.ul_repeat * 5)
        + rv.build_supervised_items(adapter, processor,
                                    h_pairs(manifest, forget_ids),
                                    repeat=args.ul_repeat))
    rv.train_supervised("suppression", adapter, model, processor, items,
                        edit_dir, args.device, steps=args.ul_steps,
                        warmup=args.ul_warmup, lr=args.ul_lr)
    del model, processor, adapter
    gc.collect()
    torch.cuda.empty_cache()
    return edit_dir


def soft_eval_codes(args, adapter, model, processor, manifest):
    """Full-sequence candidate probabilities for every code (strict accounting:
    candidate mass, OTHER mass, gated summaries)."""
    vocab = sorted(set(manifest["alias_of"].values()) | {DELETED_LABEL})
    out = {}
    for iid in manifest["identity_ids"]:
        probs = rv.full_sequence_label_probs(
            adapter, model, processor, manifest["code_of"][iid], vocab,
            args.device)
        prob_by_label = {l: probs.get(l, {}).get("prob", 0.0) for l in vocab}
        logp_by_label = {l: probs.get(l, {}).get("log_prob", -1e9)
                         for l in vocab}
        summary = rv.build_candidate_summary(prob_by_label, vocab,
                                             DELETED_LABEL)
        summary["log_probs"] = logp_by_label
        out[iid] = summary
    return out, vocab


# ====================================================================== #
# RD7: manifest
# ====================================================================== #
def run_manifest(args, out_base, results, t_start, provenance):
    logger.info("=" * 60)
    logger.info("RD7: RUN MANIFEST")
    logger.info("=" * 60)
    inputs = {"annotations": rv.sha256_file(ANNOTATIONS_PARQUET),
              "split_manifest": rv.sha256_file(SPLIT_MANIFEST),
              "identity_mapping": rv.sha256_file(
                  MANIFEST_DIR / "realdata_identity_mapping.json")}
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
        # The EXECUTING commit (captured before any results were written).
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
                f"script_sha256={manifest['provenance']['script_sha256'][:12]} "
                f"dirty={manifest['provenance']['worktree_dirty_at_start']} "
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
                        help="all or subset of: RD0 RD1 RD2 RD3 RD4 RD5 RD6")
    parser.add_argument("--route-steps", type=int, default=ROUTE_STEPS)
    parser.add_argument("--route-warmup", type=int, default=ROUTE_WARMUP)
    parser.add_argument("--route-lr", type=float, default=ROUTE_LR)
    parser.add_argument("--route-repeat", type=int, default=ROUTE_REPEAT)
    parser.add_argument("--ul-steps", type=int, default=UL_STEPS)
    parser.add_argument("--ul-warmup", type=int, default=UL_WARMUP)
    parser.add_argument("--ul-lr", type=float, default=UL_LR)
    parser.add_argument("--ul-repeat", type=int, default=UL_REPEAT)
    parser.add_argument("--smoke", action="store_true")
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

    # Provenance guard (same policy as the research-validity pipeline).
    dirty = rv.git_worktree_dirty()
    clean_required = not args.smoke and not args.allow_dirty
    if clean_required and dirty:
        raise RuntimeError(
            "Refusing to run: git worktree is dirty. Commit first so the "
            "manifest commit equals the executed implementation, or pass "
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

    all_phases = ["RD0", "RD1", "RD2", "RD3", "RD4", "RD5", "RD6"]
    requested = set(args.phase)
    run_phases = (all_phases if "all" in requested
                  else [p for p in all_phases if p in requested])
    t_start = time.time()
    results = {}

    manifest = None
    if "RD0" in run_phases:
        manifest = build_real_manifest(args, out_base)
    else:
        manifest = load_real_manifest()
    results["RD0"] = {"identity_ids": manifest["identity_ids"],
                      "alias_of": manifest["alias_of"],
                      "forget_identity_ids": manifest["forget_identity_ids"]}
    forget_ids = manifest["forget_identity_ids"]

    if "RD1" in run_phases:
        train_g(args, out_base, manifest)
        results["RD1"] = eval_g(args, out_base, manifest)

    if "RD2" in run_phases:
        train_h(args, out_base, manifest)
        adapter, model, processor = load_h(args, out_base / "h_C_to_Y")
        results["RD2"] = eval_h_strict(args, adapter, model, processor,
                                       manifest, "h_baseline")
        del model, processor, adapter
        gc.collect()
        torch.cuda.empty_cache()

    if "RD3" in run_phases:
        results["RD3"] = run_e2e(args, out_base, manifest,
                                 out_base / "h_C_to_Y", "e2c_real_h",
                                 tag="pre", delete_set=set())

    if "RD4" in run_phases:
        # leave-one-out oracle: retrain h WITHOUT the forget identity,
        # training-matched to RD2 (the retrain-equivalence reference).
        train_h(args, out_base, manifest, exclude=forget_ids,
                tag="h_oracle", adapter_name="e2c_real_h_oracle")
        adapter, model, processor = load_h(
            args, out_base / "h_oracle", adapter_name="e2c_real_h_oracle")
        results["RD4"] = eval_h_strict(args, adapter, model, processor,
                                       manifest, "h_oracle")
        oracle_soft, vocab = soft_eval_codes(args, adapter, model, processor,
                                             manifest)
        results["RD4"]["oracle_candidate"] = {
            iid: oracle_soft[iid] for iid in forget_ids}
        with open(out_base / "h_oracle" / "soft_metrics.json", "w") as f:
            json.dump(oracle_soft, f, indent=2)
        del model, processor, adapter
        gc.collect()
        torch.cuda.empty_cache()

    if "RD5" in run_phases:
        run_suppression(args, out_base, manifest)
        results["RD5"] = {"trained": True}

    if "RD6" in run_phases:
        adapter, model, processor = load_h(args, out_base / "edited_h",
                                           adapter_name="e2c_real_h_edit")
        results["RD6_hard"] = eval_h_strict(args, adapter, model, processor,
                                            manifest, "h_edited")
        soft, vocab = soft_eval_codes(args, adapter, model, processor,
                                      manifest)
        alias_of = manifest["alias_of"]
        rd6 = {"soft": {}, "gated_distance_to_oracle": {}}
        oracle_path = out_base / "h_oracle" / "soft_metrics.json"
        oracle_soft = None
        if oracle_path.exists():
            with open(oracle_path) as f:
                oracle_soft = json.load(f)
        for iid in manifest["identity_ids"]:
            s = soft[iid]
            rd6["soft"][iid] = {
                "expected_alias": alias_of[iid],
                "p_expected_alias": s["probs"].get(alias_of[iid], 0.0),
                "p_deleted_label": s["probs"].get(DELETED_LABEL, 0.0),
                "candidate_mass": s["candidate_mass"],
                "other_mass": s["other_mass"],
            }
            if oracle_soft and iid in oracle_soft:
                dist, reliable, reason = rv.gated_distance(
                    s, oracle_soft[iid], "candidate", vocab,
                    MIN_CANDIDATE_MASS)
                rd6["gated_distance_to_oracle"][iid] = {
                    "distance": dist, "reliable": reliable, "reason": reason}
        results["RD6_soft"] = rd6
        del model, processor, adapter
        gc.collect()
        torch.cuda.empty_cache()
        results["RD6_e2e"] = run_e2e(args, out_base, manifest,
                                     out_base / "edited_h",
                                     "e2c_real_h_edit", tag="post",
                                     delete_set=set(forget_ids))
        # structural route-freeze control: g must be unchanged (it is frozen).
        pre = (out_base / "e2e_pre" / "e2e_results.json")
        if pre.exists():
            with open(pre) as f:
                pre_sum = json.load(f)
            results["RD6_route_freeze_control"] = {
                "check_type": "structural_route_freeze_control",
                "g_accuracy_pre": pre_sum["g_code_accuracy"],
                "g_accuracy_post": results["RD6_e2e"]["g_code_accuracy"],
                "note": "g (X->C) is never edited; its accuracy must be "
                        "identical pre/post (recomputed empirically).",
            }

    logger.info("=" * 60)
    logger.info("REAL-DATASET PHASE COMPLETE")
    logger.info("=" * 60)
    if results:
        with open(out_base / "rd_summary.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    if not args.no_manifest:
        run_manifest(args, out_base, results, t_start, provenance)
    logger.info(f"Results saved under {out_base} "
                f"({time.time() - t_start:.1f}s total)")


if __name__ == "__main__":
    main()
