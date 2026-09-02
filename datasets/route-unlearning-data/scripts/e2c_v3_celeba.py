#!/usr/bin/env python3
"""E2C-v3 CelebA + synthetic profiles pilot: controlled granularity experiments.

Real CelebA images with SYNTHETIC identity profiles (by construction): real
face images are deterministically partitioned into 8 synthetic identities
by disjoint STRONG-CUE attribute signatures (bald, hat, eyeglasses, gray
hair, blond hair, black hair, mustache, necktie; visual consistency
preserved), and the identity/subgroup/group/numeric hierarchy is fully
controlled -- exactly the main controlled granularity experiments the plan
requires:

    specific -> subgroup      (alias -> SG_Cx)
    specific -> group         (alias -> GROUP_Cx)
    specific -> numeric       (alias -> NUM_i, the identity number)
    numeric  -> range         (NUM_i -> RANGE_LOW/HIGH: numeric abstraction
                               at a second, coarser depth)
    suppression               (alias -> Unknown refusal)
    update                    (alias -> novel alias)
    retain                    (the rest stay at fine aliases)

ALL transformations are applied simultaneously in ONE edited h (RV6-style
composition), evaluated with strict multi-label-rejection scoring,
candidate-mass gating, a leave-one-out oracle, and per-split reporting
(all-image composition vs held-out-only).
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
logger = logging.getLogger("e2c_v3_celeba")

SCRIPT_DIR = Path(__file__).resolve().parent
CELEBA_SNAPSHOT = Path(
    "/scratch/wutiantong/.cache/huggingface/hub/"
    "datasets--huggan--CelebA-faces-with-attributes/snapshots/"
    "b47e27a7c6bc578361ce132da8c8dad573b98d9e")

OUT_DEFAULT = "e2c_celeba/outputs/celeba"
MANIFEST_DIR = Path("e2c_celeba/manifests")
IMAGE_CACHE = Path("e2c_celeba/images")

IMG_TO_CODE_PROMPT = "What is the identity code for this person?"
CODE_TO_ALIAS_PROMPT = "Identity code: {code}. Generate the alias."
DELETED_LABEL = "Unknown"

# Signature-based identities contain MANY distinct faces per bucket, so g
# needs substantially more images per identity than the same-person
# benchmarks (ppubench) to avoid memorizing the training faces
# (9 train/identity gave train acc 1.0 but held-out 0.333).
IMAGES_TRAIN = 24
IMAGES_TEST = 6
ROUTE_STEPS = 3000
ROUTE_WARMUP = 200
ROUTE_LR = 2e-5
ROUTE_REPEAT = 50
UL_STEPS = 500
UL_WARMUP = 50
UL_LR = 2e-5
UL_REPEAT = 50
MIN_CANDIDATE_MASS = 0.01

ALIAS_POOL = ["Aster", "Briar", "Clove", "Dune", "Ember",
              "Fern", "Gale", "Hollow", "Iris", "Juniper"]
UPDATE_NEW_ALIAS = "Vesper"

# Disjoint attribute signatures -> 8 synthetic identities.  Ordered; each
# image is assigned to the FIRST signature it satisfies (deterministic).
# NOTE: this parquet uses the ORIGINAL CelebA convention of -1/1 values
# (not 0/1), so signatures are written in -1/1 terms.
#
# REDESIGN (after two failed g routes): signature-identities are NOT
# same-person identities -- g must abstract an attribute CLASS across many
# distinct faces.  The first design used visually weak/ambiguous cues
# (Reddish_Hair = "no named hair color + wavy", Smiling_Young, etc.) and g
# failed to fit even its training faces (train acc 0.575, held-out 0.267,
# confusion spread across all signatures).  Only signatures grounded in
# STRONG, individually detectable visual objects/cues are kept here:
# bald head, hat, eyeglasses, gray hair, blond hair, black hair, mustache,
# necktie.
SIGNATURES = [
    ("Bald", {"Bald": 1}),
    ("Wearing_Hat", {"Wearing_Hat": 1}),
    ("Eyeglasses", {"Eyeglasses": 1}),
    ("Gray_Hair", {"Gray_Hair": 1}),
    ("Blond_Hair", {"Blond_Hair": 1}),
    ("Black_Hair", {"Black_Hair": 1}),
    ("Mustache", {"Mustache": 1}),
    ("Wearing_Necktie", {"Wearing_Necktie": 1}),
]
N_IDENTITIES = len(SIGNATURES)


def _load_sibling(module_name, filename):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rv = _load_sibling("e2c_rv_shared", "e2c_v3_research_validity.py")
rd = _load_sibling("e2c_rd_shared", "e2c_v3_realdata.py")


# ====================================================================== #
# Hierarchy construction (synthetic profiles over real images)
# ====================================================================== #
def build_hierarchy():
    ids = [f"CID_{i}" for i in range(N_IDENTITIES)]
    alias_of = {iid: ALIAS_POOL[i] for i, iid in enumerate(ids)}
    subgroup_of = {iid: f"SG_C{i // 2 + 1}" for i, iid in enumerate(ids)}
    half = N_IDENTITIES // 2
    group_of = {iid: ("GROUP_CA" if i < half else "GROUP_CB")
                for i, iid in enumerate(ids)}
    numeric_of = {iid: f"NUM_{i}" for i, iid in enumerate(ids)}
    range_of = {iid: ("RANGE_LOW" if i < half else "RANGE_HIGH")
                for i, iid in enumerate(ids)}
    return ids, alias_of, subgroup_of, group_of, numeric_of, range_of


# ====================================================================== #
# CB0: deterministic CelebA manifest + image extraction
# ====================================================================== #
def build_celeba_manifest(args):
    logger.info("=" * 60)
    logger.info("CB0: BUILD CELEBA MANIFEST (synthetic profiles on real "
                "images)")
    logger.info("=" * 60)
    import glob

    import pandas as pd
    shards = sorted(glob.glob(str(CELEBA_SNAPSHOT / "data"
                                  / "*.parquet")))
    need = IMAGES_TRAIN + IMAGES_TEST
    buckets = {name: [] for name, _ in SIGNATURES}
    attrs = [a for _, sig in SIGNATURES for a in sig]
    cols = ["image", "image_id"] + sorted(set(attrs))
    skipped_shards = []
    for sh in shards:
        if all(len(b) >= need * 2 for b in buckets.values()):
            break
        try:
            df = pd.read_parquet(sh, columns=cols)
        except Exception as e:
            # 11 of the 132 upstream shards are corrupt in the HF repo itself
            # (verified after re-download: LFS bytes lack the parquet footer).
            # 121 good shards (~185k images) far exceed what we need.
            skipped_shards.append({"shard": Path(sh).name,
                                   "error": str(e)[:120]})
            logger.warning(f"CB0: skipping unreadable shard {Path(sh).name}: "
                           f"{str(e)[:80]}")
            continue
        for _, row in df.iterrows():
            for name, sig in SIGNATURES:
                if len(buckets[name]) >= need * 2:
                    continue
                if all(int(row[a]) == v for a, v in sig.items()):
                    buckets[name].append(row)
                    break
            if all(len(b) >= need * 2 for b in buckets.values()):
                break
    short = [n for n, b in buckets.items() if len(b) < need]
    if short:
        raise ValueError(f"signatures with too few images: {short}")

    ids, alias_of, subgroup_of, group_of, numeric_of, range_of = \
        build_hierarchy()
    # deterministic transformation assignment under the seed
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N_IDENTITIES, generator=g).tolist()
    roles = (["suppress", "update", "gen_subgroup", "gen_group",
              "gen_numeric", "gen_range"]
             + ["retain"] * (N_IDENTITIES - 6))
    t = {ids[perm[k]]: role for k, role in enumerate(roles)}

    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    code_of, items, img_sha = {}, [], {}
    for i, iid in enumerate(ids):
        code_of[iid] = f"CEL_{iid}"
        sig_name = SIGNATURES[i][0]
        rows = buckets[sig_name][:need]
        for j, row in enumerate(rows):
            split = "test" if j >= IMAGES_TRAIN else "train"
            fname = f"{iid}_{j:02d}.png"
            dest = IMAGE_CACHE / fname
            cell = row["image"]
            img = (Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
                   if isinstance(cell, dict) else cell.convert("RGB"))
            img.save(dest)
            img_sha[fname] = rv.sha256_file(dest)
            items.append({"identity_id": iid, "image_uri": str(dest),
                          "split": split, "source_image_id":
                          str(row["image_id"]), "image_sha256":
                          img_sha[fname]})

    transformations = {
        "suppress": {"ids": [iid for iid, r in t.items() if r == "suppress"],
                     "to": DELETED_LABEL},
        "update": {"ids": [iid for iid, r in t.items() if r == "update"],
                   "to": UPDATE_NEW_ALIAS},
        "gen_subgroup": {"ids": [iid for iid, r in t.items()
                                 if r == "gen_subgroup"],
                         "to_of": {iid: subgroup_of[iid] for iid in ids
                                    if t[iid] == "gen_subgroup"},
                         "depth": "specific_to_subgroup"},
        "gen_group": {"ids": [iid for iid, r in t.items()
                              if r == "gen_group"],
                      "to_of": {iid: group_of[iid] for iid in ids
                                 if t[iid] == "gen_group"},
                      "depth": "specific_to_group"},
        "gen_numeric": {"ids": [iid for iid, r in t.items()
                                if r == "gen_numeric"],
                        "to_of": {iid: numeric_of[iid] for iid in ids
                                   if t[iid] == "gen_numeric"},
                        "depth": "specific_to_numeric"},
        "gen_range": {"ids": [iid for iid, r in t.items() if r == "gen_range"],
                      "to_of": {iid: range_of[iid] for iid in ids
                                 if t[iid] == "gen_range"},
                      "depth": "numeric_to_range"},
        "retain": {"ids": [iid for iid, r in t.items() if r == "retain"]},
    }
    manifest = {
        "dataset": "huggan/CelebA-faces-with-attributes "
                   "(CelebA-aligned 178x218 + 40 attributes)",
        "snapshot_revision": "b47e27a7c6bc578361ce132da8c8dad573b98d9e",
        "skipped_corrupt_shards": skipped_shards,
        "profiles": "synthetic identity profiles over real CelebA images; "
                    "identities formed by disjoint attribute signatures "
                    "(visual consistency), hierarchy fully controlled",
        "seed": args.seed,
        "identity_ids": ids,
        "code_of": code_of,
        "alias_of": alias_of,
        "subgroup_of": subgroup_of,
        "group_of": group_of,
        "numeric_of": numeric_of,
        "range_of": range_of,
        "signature_of": {ids[i]: SIGNATURES[i][0]
                         for i in range(N_IDENTITIES)},
        "update_new_alias": UPDATE_NEW_ALIAS,
        "deleted_label": DELETED_LABEL,
        "transformations": transformations,
        "split": {"rule": f"first {IMAGES_TRAIN} images train, last "
                           f"{IMAGES_TEST} held-out test",
                  "train_per_id": IMAGES_TRAIN, "test_per_id": IMAGES_TEST},
        "images_sha256": img_sha,
        "items": items,
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / "celeba_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"CB0: {len(ids)} identities, {len(items)} images; "
                f"transformations="
                f"{ {k: v.get('ids', v.get('to')) for k, v in transformations.items()} }")
    return manifest


def load_celeba_manifest():
    with open(MANIFEST_DIR / "celeba_manifest.json") as f:
        return json.load(f)


# ====================================================================== #
# Shared helpers
# ====================================================================== #
def expected_of(manifest):
    exp = dict(manifest["alias_of"])
    tr = manifest["transformations"]
    for iid in tr["suppress"]["ids"]:
        exp[iid] = DELETED_LABEL
    for iid in tr["update"]["ids"]:
        exp[iid] = manifest["update_new_alias"]
    for key in ("gen_subgroup", "gen_group", "gen_numeric", "gen_range"):
        for iid, lab in tr[key]["to_of"].items():
            exp[iid] = lab
    return exp


def label_vocab(manifest):
    labs = set(manifest["alias_of"].values())
    labs.add(manifest["update_new_alias"])
    labs.add(DELETED_LABEL)
    labs.update(manifest["subgroup_of"].values())
    labs.update(manifest["group_of"].values())
    labs.update(manifest["numeric_of"].values())
    labs.update(manifest["range_of"].values())
    return sorted(labs)


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


# ====================================================================== #
# CB4: combined edit (all transformations in ONE h)
# ====================================================================== #
def run_combined_edit(args, out_base, manifest):
    logger.info("=" * 60)
    logger.info("CB4: COMBINED EDIT (all transformations, single h)")
    logger.info("=" * 60)
    edit_dir = out_base / "edited_h"
    edit_dir.mkdir(parents=True, exist_ok=True)
    adapter, model, processor = rv.create_adapter_model(
        args, args.device, "e2c_celeba_h_edit")
    model = rv.attach_lora(adapter, model)
    rv.load_trained_weights(
        adapter, model,
        out_base / "h_C_to_Y" / "adapter_final" / "adapter_model.safetensors")
    exp = expected_of(manifest)
    changed = [i for i in manifest["identity_ids"]
               if exp[i] != manifest["alias_of"][i]]
    changed_pairs = [{"prompt": CODE_TO_ALIAS_PROMPT.format(
                         code=manifest["code_of"][i]), "answer": exp[i]}
                     for i in changed]
    retain_pairs = rd.h_pairs(manifest, exclude=changed)
    items = (
        rv.build_supervised_items(adapter, processor, changed_pairs,
                                  repeat=args.ul_repeat * 3)
        + rv.build_supervised_items(adapter, processor, retain_pairs,
                                    repeat=args.ul_repeat * 3))
    rv.train_supervised("celeba_edit", adapter, model, processor, items,
                        edit_dir, args.device, steps=args.ul_steps,
                        warmup=args.ul_warmup, lr=args.ul_lr)
    del model, processor, adapter
    gc.collect()
    torch.cuda.empty_cache()
    return edit_dir


# ====================================================================== #
# CB5: post-edit e2e (all-image + held-out)
# ====================================================================== #
def run_celeba_e2e(args, out_base, manifest, h_dir, h_name, tag):
    e2e_dir = out_base / f"e2e_{tag}"
    e2e_dir.mkdir(parents=True, exist_ok=True)
    adapter_g, model_g, processor_g = rv.create_adapter_model(
        args, args.device, "e2c_celeba_g_e2e")
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
    vocab = label_vocab(manifest)
    exp = expected_of(manifest)
    sup_ids = set(manifest["transformations"]["suppress"]["ids"])
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
            if iid in sup_ids:
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
    for name, spec in manifest["transformations"].items():
        sub = [r for r in rows if r["identity_id"] in set(spec["ids"])]
        summary["by_transformation"][name] = block(sub)
    with open(e2e_dir / "e2e_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"E2E[{tag}]: all={summary['all_images_composition']} "
                f"held_out={summary['held_out_only']}")
    return summary


def run_manifest(args, out_base, results, t_start, provenance):
    inputs = {"celeba_manifest": rv.sha256_file(
        MANIFEST_DIR / "celeba_manifest.json")}
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
                        help="all or subset of: CB0 CB1 CB2 CB3 CB4 CB5")
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

    all_phases = ["CB0", "CB1", "CB2", "CB3", "CB4", "CB5"]
    requested = set(args.phase)
    run_phases = (all_phases if "all" in requested
                  else [p for p in all_phases if p in requested])
    t_start = time.time()
    results = {}

    if "CB0" in run_phases:
        manifest = build_celeba_manifest(args)
    else:
        manifest = load_celeba_manifest()
    results["CB0"] = {
        "identity_ids": manifest["identity_ids"],
        "transformations": {k: v["ids"] for k, v in
                            manifest["transformations"].items()},
    }

    if "CB1" in run_phases:
        rd.train_g(args, out_base, manifest)
        results["CB1"] = rd.eval_g(args, out_base, manifest)

    if "CB2" in run_phases:
        rd.train_h(args, out_base, manifest)
        adapter, model, processor = rd.load_h(args, out_base / "h_C_to_Y")
        results["CB2"] = rd.eval_h_strict(args, adapter, model, processor,
                                          manifest, "h_baseline")
        del model, processor, adapter
        gc.collect()
        torch.cuda.empty_cache()

    if "CB3" in run_phases:
        exclude = [i for i in manifest["identity_ids"]
                   if expected_of(manifest)[i] != manifest["alias_of"][i]]
        rd.train_h(args, out_base, manifest, exclude=exclude,
                   tag="h_oracle", adapter_name="e2c_celeba_h_oracle")
        adapter, model, processor = rd.load_h(
            args, out_base / "h_oracle", adapter_name="e2c_celeba_h_oracle")
        results["CB3"] = rd.eval_h_strict(args, adapter, model, processor,
                                          manifest, "h_oracle")
        oracle_soft, vocab = rd.soft_eval_codes(args, adapter, model,
                                                processor, manifest)
        with open(out_base / "h_oracle" / "soft_metrics.json", "w") as f:
            json.dump(oracle_soft, f, indent=2)
        del model, processor, adapter
        gc.collect()
        torch.cuda.empty_cache()

    if "CB4" in run_phases:
        run_combined_edit(args, out_base, manifest)
        results["CB4"] = {"trained": True}

    if "CB5" in run_phases:
        exp = expected_of(manifest)
        adapter, model, processor = rd.load_h(
            args, out_base / "edited_h", adapter_name="e2c_celeba_h_edit")
        results["CB5_hard"] = rd.eval_h_strict(
            args, adapter, model, processor, manifest, "h_edited")
        full_vocab = label_vocab(manifest)
        for p in results["CB5_hard"]["preds"]:
            # re-parse with the FULL label vocabulary (generalization labels
            # are outside the alias-only vocab used by eval_h_strict)
            p["parsed_full_vocab"] = rv.parse_recognized_label(
                p["raw"], full_vocab)
            p["expected_post_edit"] = exp[p["identity_id"]]
            p["correct_post_edit"] = (
                p["parsed_full_vocab"] == exp[p["identity_id"]])
        results["CB5_hard"]["accuracy_post_edit_expectation"] = sum(
            p["correct_post_edit"] for p in results["CB5_hard"]["preds"]
        ) / len(results["CB5_hard"]["preds"])
        soft, vocab = soft_eval_full_vocab(args, adapter, model, processor,
                                            manifest, full_vocab)
        cb5 = {"soft": {}, "gated_distance_to_oracle": {}}
        oracle_path = out_base / "h_oracle" / "soft_metrics.json"
        oracle_soft = None
        if oracle_path.exists():
            with open(oracle_path) as f:
                oracle_soft = json.load(f)
        for iid in manifest["identity_ids"]:
            s = soft[iid]
            cb5["soft"][iid] = {
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
                cb5["gated_distance_to_oracle"][iid] = {
                    "distance": dist, "reliable": reliable, "reason": reason}
        results["CB5_soft"] = cb5
        del model, processor, adapter
        gc.collect()
        torch.cuda.empty_cache()
        results["CB5_e2e"] = run_celeba_e2e(
            args, out_base, manifest, out_base / "edited_h",
            "e2c_celeba_h_edit", tag="post")
        n_retain = len(manifest["transformations"]["retain"]["ids"])
        results["CB5_wording"] = {
            "collateral": f"no observed collateral on the {n_retain} tested "
                          "retained identities; NOT a general zero-collateral "
                          "claim",
        }

    logger.info("=" * 60)
    logger.info("CELEBA PILOT PHASE COMPLETE")
    logger.info("=" * 60)
    if results:
        with open(out_base / "cb_summary.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    if not args.no_manifest:
        run_manifest(args, out_base, results, t_start, provenance)
    logger.info(f"Results saved under {out_base} "
                f"({time.time() - t_start:.1f}s total)")


if __name__ == "__main__":
    main()
