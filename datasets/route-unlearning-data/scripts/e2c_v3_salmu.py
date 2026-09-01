#!/usr/bin/env python3
"""E2C-v3 SALMUBench pilot: semantic/taxonomic abstraction on real data.

Real-taxonomy instantiation of the frozen E2C-v3 route on SALMUBench
(774 sensitive identities, many images each, deterministic 3-level job
taxonomies per identity from the salmu_hierarchical associations).

Transformations (all in ONE edited h, RV6-style composition):
    * SUPPRESS one identity   -> 'Unknown' (refusal-targeted suppression)
    * GENERALIZE two identities along their REAL job taxonomy:
        - one specific -> level-1 ancestor (e.g. job -> occupation category)
        - one specific -> level-2 ancestor (e.g. job -> sector)
      Taxonomic abstraction: outputs remain TRUE but less specific.
    * RETAIN the rest at their fine-grained job label.

Masking/redaction is NOT part of this pilot (auxiliary generalization only,
owned by separate redaction experiments); this pilot validates semantic/
taxonomic granularity reduction under the frozen route architecture.

Held-out reporting (fixing the PPUBench limitation): SALMUBench has many
images per identity; the last-3 sorted images per identity are strictly
held out, and every e2e metric is reported BOTH as all-image composition
and held-out-only.
"""
import argparse
import gc
import importlib.util
import io
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
logger = logging.getLogger("e2c_v3_salmu")

SCRIPT_DIR = Path(__file__).resolve().parent
SALMU_TRAIN_SNAPSHOT = Path(
    "/scratch/wutiantong/.cache/huggingface/hub/"
    "datasets--cvc-mmu--salmu-512-redistributed/snapshots/"
    "11bc6fec2530a70ba222bf86a70fe4d7681f86dd")
GMUL_ASSOCIATIONS = Path(
    "/scratch/wutiantong/GMUL/granularity-unlearning/data/"
    "salmu_hierarchical/associations.json")

OUT_DEFAULT = "e2c_salmu/outputs/salmu"
MANIFEST_DIR = Path("e2c_salmu/manifests")
IMAGE_CACHE = Path("e2c_salmu/images")

IMG_TO_CODE_PROMPT = "What is the identity code for this person?"
CODE_TO_ALIAS_PROMPT = "Identity code: {code}. Generate the alias."
DELETED_LABEL = "Unknown"

N_IDENTITIES = 12
IMAGES_TRAIN = 8
IMAGES_TEST = 3
ROUTE_STEPS = 3000
ROUTE_WARMUP = 200
ROUTE_LR = 2e-5
ROUTE_REPEAT = 50
UL_STEPS = 500
UL_WARMUP = 50
UL_LR = 2e-5
UL_REPEAT = 50
MIN_CANDIDATE_MASS = 0.01


def _load_sibling(module_name, filename):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rv = _load_sibling("e2c_rv_shared", "e2c_v3_research_validity.py")
rd = _load_sibling("e2c_rd_shared", "e2c_v3_realdata.py")


# ====================================================================== #
# SB0: deterministic SALMU manifest + image extraction
# ====================================================================== #
def _extract_image(cell, dest):
    """Save a parquet image cell (PIL or {'bytes':...}) to dest; return sha."""
    from PIL import Image
    if isinstance(cell, dict):
        img = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
    else:
        img = cell.convert("RGB")
    img.save(dest)
    return rv.sha256_file(dest)


def build_salmu_manifest(args):
    logger.info("=" * 60)
    logger.info("SB0: BUILD SALMU MANIFEST (taxonomies + image extraction)")
    logger.info("=" * 60)
    import collections
    with open(GMUL_ASSOCIATIONS) as f:
        associations = json.load(f)
    jobs = {iid: a["job"]["levels"] for iid, a in associations.items()
            if "job" in a and len(a["job"]["levels"]) == 3}

    # gather per-identity image rows from the sensitive shards
    import glob

    import pandas as pd
    shards = sorted(glob.glob(str(SALMU_TRAIN_SNAPSHOT / "data"
                                        / "sensitive-*.parquet")))
    per_id_rows = collections.defaultdict(list)
    for sh in shards:
        df = pd.read_parquet(sh)
        for _, row in df.iterrows():
            iid = str(row["identity_id"])
            if iid in jobs:
                per_id_rows[iid].append(row)
    eligible = sorted(i for i, rows in per_id_rows.items()
                      if len(rows) >= IMAGES_TRAIN + IMAGES_TEST)
    logger.info(f"eligible identities (>={IMAGES_TRAIN + IMAGES_TEST} "
                f"images): {len(eligible)}")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(eligible), generator=g).tolist()
    selected = [eligible[i] for i in perm[:N_IDENTITIES]]
    # deterministic transformation assignment under the seed
    tperm = torch.randperm(N_IDENTITIES, generator=g).tolist()
    suppress_id = selected[tperm[0]]
    gen_l1_id = selected[tperm[1]]   # specific -> level-1 ancestor
    gen_l2_id = selected[tperm[2]]   # specific -> level-2 ancestor

    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
    code_of, alias_of, items, img_sha = {}, {}, [], {}
    for iid in selected:
        rows = sorted(per_id_rows[iid], key=lambda r: str(r["file_name"]))
        code_of[iid] = f"SAL_{iid}"
        alias_of[iid] = jobs[iid][0]  # fine-grained job = baseline target
        keep = rows[:IMAGES_TRAIN + IMAGES_TEST]
        n_test = min(IMAGES_TEST, len(keep) - 1)
        for j, row in enumerate(keep):
            split = ("test" if j >= len(keep) - n_test else "train")
            fname = f"SAL_{iid}_{j:02d}.png"
            dest = IMAGE_CACHE / fname
            sha = _extract_image(row["image"], dest)
            img_sha[fname] = sha
            items.append({"identity_id": iid, "image_uri": str(dest),
                          "split": split, "source_file_name":
                          str(row["file_name"]), "image_sha256": sha})
    generalizations = {
        gen_l1_id: {"from": jobs[gen_l1_id][0], "to": jobs[gen_l1_id][1],
                    "depth": "specific_to_level1"},
        gen_l2_id: {"from": jobs[gen_l2_id][0], "to": jobs[gen_l2_id][2],
                    "depth": "specific_to_level2"},
    }
    manifest = {
        "dataset": "salmubench-512-redistributed (sensitive split)",
        "snapshot_revision": "11bc6fec2530a70ba222bf86a70fe4d7681f86dd",
        "taxonomy_source": str(GMUL_ASSOCIATIONS),
        "taxonomy_source_sha256": rv.sha256_file(GMUL_ASSOCIATIONS),
        "seed": args.seed,
        "identity_ids": selected,
        "code_of": code_of,
        "alias_of": alias_of,
        "job_levels": {i: jobs[i] for i in selected},
        "suppress_identity_id": suppress_id,
        "generalizations": generalizations,
        "deleted_label": DELETED_LABEL,
        "split": {"rule": f"first {IMAGES_TRAIN} sorted images train, "
                           f"last {IMAGES_TEST} held-out test",
                  "train_per_id": IMAGES_TRAIN, "test_per_id": IMAGES_TEST},
        "masking_redaction": "out of scope for this pilot (auxiliary "
                             "generalization only, per plan)",
        "images_sha256": img_sha,
        "items": items,
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / "salmu_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"SB0: {len(selected)} identities, {len(items)} images; "
                f"suppress={suppress_id} ({alias_of[suppress_id]}); "
                f"gen_l1={gen_l1_id} ({generalizations[gen_l1_id]['from']} -> "
                f"{generalizations[gen_l1_id]['to']}); gen_l2={gen_l2_id} "
                f"({generalizations[gen_l2_id]['from']} -> "
                f"{generalizations[gen_l2_id]['to']})")
    return manifest


def load_salmu_manifest():
    with open(MANIFEST_DIR / "salmu_manifest.json") as f:
        return json.load(f)


def _have_ckpt(d):
    return (Path(d) / "adapter_final" / "adapter_model.safetensors").exists()


# ====================================================================== #
# SB4: combined suppress + generalize(l1,l2) + retain in ONE h
# ====================================================================== #
def run_combined_edit(args, out_base, manifest):
    logger.info("=" * 60)
    logger.info("SB4: COMBINED SUPPRESS + GENERALIZE + RETAIN (single h)")
    logger.info("=" * 60)
    edit_dir = out_base / "edited_h"
    edit_dir.mkdir(parents=True, exist_ok=True)
    sup = manifest["suppress_identity_id"]
    gens = manifest["generalizations"]
    adapter, model, processor = rv.create_adapter_model(
        args, args.device, "e2c_salmu_h_edit")
    model = rv.attach_lora(adapter, model)
    rv.load_trained_weights(
        adapter, model,
        out_base / "h_C_to_Y" / "adapter_final" / "adapter_model.safetensors")
    suppress_pairs = [{"prompt": CODE_TO_ALIAS_PROMPT.format(
                          code=manifest["code_of"][sup]),
                       "answer": DELETED_LABEL}]
    gen_pairs = [{"prompt": CODE_TO_ALIAS_PROMPT.format(
                     code=manifest["code_of"][iid]),
                  "answer": g["to"]} for iid, g in gens.items()]
    retain_pairs = rd.h_pairs(manifest, exclude=[sup, *gens])
    items = (
        rv.build_supervised_items(adapter, processor, suppress_pairs,
                                  repeat=args.ul_repeat * 5)
        + rv.build_supervised_items(adapter, processor, gen_pairs,
                                    repeat=args.ul_repeat * 3)
        + rv.build_supervised_items(adapter, processor, retain_pairs,
                                    repeat=args.ul_repeat * 3))
    rv.train_supervised("salmu_edit", adapter, model, processor, items,
                        edit_dir, args.device, steps=args.ul_steps,
                        warmup=args.ul_warmup, lr=args.ul_lr)
    del model, processor, adapter
    gc.collect()
    torch.cuda.empty_cache()
    return edit_dir


# ====================================================================== #
# SB5: post-edit evaluation (all-image + held-out breakdowns)
# ====================================================================== #
def expected_of(manifest):
    exp = dict(manifest["alias_of"])
    exp[manifest["suppress_identity_id"]] = DELETED_LABEL
    for iid, g in manifest["generalizations"].items():
        exp[iid] = g["to"]
    return exp


def soft_eval_full_vocab(args, adapter, model, processor, manifest, vocab):
    """Full-sequence candidate probabilities over an arbitrary label vocab."""
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


def run_salmu_e2e(args, out_base, manifest, h_dir, h_name, tag):
    e2e_dir = out_base / f"e2e_{tag}"
    e2e_dir.mkdir(parents=True, exist_ok=True)
    adapter_g, model_g, processor_g = rv.create_adapter_model(
        args, args.device, "e2c_salmu_g_e2e")
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
    del g_backend, model_g, processor_g, adapter_g
    gc.collect()
    torch.cuda.empty_cache()

    adapter_h, model_h, processor_h = rd.load_h(args, h_dir,
                                                adapter_name=h_name)
    h_backend = adapter_h.to_eval_backend(
        model=model_h, processor=processor_h, model_config=cfg)
    model_h.eval()
    vocab = sorted(set(manifest["alias_of"].values())
                   | {DELETED_LABEL}
                   | {g["to"] for g in manifest["generalizations"].values()})
    exp = expected_of(manifest)
    sup = manifest["suppress_identity_id"]
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
            if iid == sup:
                ok = manifest["alias_of"][iid] not in rv.recognized_labels_in(
                    raw, vocab)
            else:
                ok = parsed == exp[iid]
            rows.append({**c, "h_raw_text": raw, "pred_alias": parsed,
                         "g_routed": True, "outcome_ok": ok})
    del h_backend, model_h, processor_h, adapter_h
    gc.collect()
    torch.cuda.empty_cache()

    def block(sub):
        routed = [r for r in sub if r["g_routed"]]
        return {
            "n": len(sub),
            "g_code_correct": sum(r["code_correct"] for r in sub),
            "g_routed": len(routed),
            "h_unparseable_outputs": sum(
                1 for r in routed if r["pred_alias"] is None),
            "outcome_ok": sum(r["outcome_ok"] for r in sub),
        }
    summary = {
        "tag": tag,
        "scoring": "strict token-exact parsing; multi-label outputs rejected",
        "all_images_composition": block(rows),
        "train_only": block([r for r in rows if r["split"] == "train"]),
        "held_out_only": block([r for r in rows if r["split"] == "test"]),
        "by_transformation": {},
        "rows": rows,
    }
    gen_ids = {g["depth"]: iid
               for iid, g in manifest["generalizations"].items()}
    transformations = {
        "suppress": {sup},
        "generalize_level1": {gen_ids["specific_to_level1"]},
        "generalize_level2": {gen_ids["specific_to_level2"]},
        "retain": set(manifest["identity_ids"]) - {sup}
        - set(manifest["generalizations"]),
    }
    for name, iids in transformations.items():
        sub = [r for r in rows if r["identity_id"] in iids]
        summary["by_transformation"][name] = block(sub)
    with open(e2e_dir / "e2e_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"E2E[{tag}]: all={summary['all_images_composition']} "
                f"held_out={summary['held_out_only']}")
    return summary


def run_manifest(args, out_base, results, t_start, provenance):
    inputs = {
        "salmu_manifest": rv.sha256_file(MANIFEST_DIR / "salmu_manifest.json"),
        "taxonomy_associations": rv.sha256_file(GMUL_ASSOCIATIONS),
    }
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
                        help="all or subset of: SB0 SB1 SB2 SB3 SB4 SB5")
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

    all_phases = ["SB0", "SB1", "SB2", "SB3", "SB4", "SB5"]
    requested = set(args.phase)
    run_phases = (all_phases if "all" in requested
                  else [p for p in all_phases if p in requested])
    t_start = time.time()
    results = {}

    if "SB0" in run_phases:
        manifest = build_salmu_manifest(args)
    else:
        manifest = load_salmu_manifest()
    results["SB0"] = {
        "identity_ids": manifest["identity_ids"],
        "suppress": manifest["suppress_identity_id"],
        "generalizations": manifest["generalizations"],
    }

    if "SB1" in run_phases:
        if args.eval_only and _have_ckpt(out_base / "g_X_to_C"):
            logger.info("SB1: eval-only, reusing existing g checkpoint")
        else:
            rd.train_g(args, out_base, manifest)
        results["SB1"] = rd.eval_g(args, out_base, manifest)

    if "SB2" in run_phases:
        if args.eval_only and _have_ckpt(out_base / "h_C_to_Y"):
            logger.info("SB2: eval-only, reusing existing h checkpoint")
        else:
            rd.train_h(args, out_base, manifest)
        adapter, model, processor = rd.load_h(args, out_base / "h_C_to_Y")
        results["SB2"] = rd.eval_h_strict(args, adapter, model, processor,
                                          manifest, "h_baseline")
        del model, processor, adapter
        gc.collect()
        torch.cuda.empty_cache()

    if "SB3" in run_phases:
        exclude = [manifest["suppress_identity_id"],
                   *manifest["generalizations"]]
        if args.eval_only and _have_ckpt(out_base / "h_oracle"):
            logger.info("SB3: eval-only, reusing existing oracle checkpoint")
        else:
            rd.train_h(args, out_base, manifest, exclude=exclude,
                       tag="h_oracle", adapter_name="e2c_salmu_h_oracle")
        adapter, model, processor = rd.load_h(
            args, out_base / "h_oracle", adapter_name="e2c_salmu_h_oracle")
        results["SB3"] = rd.eval_h_strict(args, adapter, model, processor,
                                          manifest, "h_oracle")
        # oracle soft metrics over the SAME full vocab SB5 uses, so the
        # gated distance-to-oracle comparison is vocab-consistent
        oracle_vocab = sorted(set(manifest["alias_of"].values())
                              | {DELETED_LABEL}
                              | {g["to"] for g in
                                 manifest["generalizations"].values()})
        oracle_soft, vocab = soft_eval_full_vocab(args, adapter, model,
                                                 processor, manifest,
                                                 oracle_vocab)
        with open(out_base / "h_oracle" / "soft_metrics.json", "w") as f:
            json.dump(oracle_soft, f, indent=2)
        del model, processor, adapter
        gc.collect()
        torch.cuda.empty_cache()

    if "SB4" in run_phases:
        if args.eval_only and _have_ckpt(out_base / "edited_h"):
            logger.info("SB4: eval-only, reusing existing edited checkpoint")
            results["SB4"] = {"trained": "reused"}
        else:
            run_combined_edit(args, out_base, manifest)
            results["SB4"] = {"trained": True}

    if "SB5" in run_phases:
        exp = expected_of(manifest)
        adapter, model, processor = rd.load_h(
            args, out_base / "edited_h", adapter_name="e2c_salmu_h_edit")
        results["SB5_hard"] = rd.eval_h_strict(
            args, adapter, model, processor, manifest, "h_edited")
        full_vocab = sorted(set(manifest["alias_of"].values())
                            | {DELETED_LABEL}
                            | {g["to"] for g in
                               manifest["generalizations"].values()})
        for p in results["SB5_hard"]["preds"]:
            # re-parse with the FULL label vocabulary (generalization labels
            # are outside the alias-only vocab used by eval_h_strict)
            p["parsed_full_vocab"] = rv.parse_recognized_label(
                p["raw"], full_vocab)
            p["expected_post_edit"] = exp[p["identity_id"]]
            p["correct_post_edit"] = (
                p["parsed_full_vocab"] == exp[p["identity_id"]])
        results["SB5_hard"]["accuracy_post_edit_expectation"] = sum(
            p["correct_post_edit"] for p in results["SB5_hard"]["preds"]
        ) / len(results["SB5_hard"]["preds"])
        soft, vocab = soft_eval_full_vocab(args, adapter, model, processor,
                                            manifest, full_vocab)
        sb5 = {"soft": {}, "gated_distance_to_oracle": {}}
        oracle_path = out_base / "h_oracle" / "soft_metrics.json"
        oracle_soft = None
        if oracle_path.exists():
            with open(oracle_path) as f:
                oracle_soft = json.load(f)
        for iid in manifest["identity_ids"]:
            s = soft[iid]
            sb5["soft"][iid] = {
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
                sb5["gated_distance_to_oracle"][iid] = {
                    "distance": dist, "reliable": reliable, "reason": reason}
        results["SB5_soft"] = sb5
        del model, processor, adapter
        gc.collect()
        torch.cuda.empty_cache()
        results["SB5_e2e"] = run_salmu_e2e(
            args, out_base, manifest, out_base / "edited_h",
            "e2c_salmu_h_edit", tag="post")
        n_retain = len(manifest["identity_ids"]) - 3
        results["SB5_wording"] = {
            "collateral": f"no observed collateral on the {n_retain} tested "
                          "retained identities; NOT a general zero-collateral "
                          "claim",
            "masking_redaction": "out of scope (auxiliary generalization "
                                 "only)",
        }

    logger.info("=" * 60)
    logger.info("SALMU PILOT PHASE COMPLETE")
    logger.info("=" * 60)
    if results:
        with open(out_base / "sb_summary.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    if not args.no_manifest:
        run_manifest(args, out_base, results, t_start, provenance)
    logger.info(f"Results saved under {out_base} "
                f"({time.time() - t_start:.1f}s total)")


if __name__ == "__main__":
    main()
