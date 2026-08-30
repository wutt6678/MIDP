#!/usr/bin/env python3
"""E2C-v3 Phase A — Factorial Intervention Audit.

Runs the complete factorial probe set on an existing trained adapter
(M-latent, D, or M-latent-shuffled) and classifies the failure mode.

Output: e2c_v3/reports/factorial_eval_<condition>.json
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2c_v3_factorial")

# LoRA scope (must match training)
SCOPE_REGEX = r"^model\.language_model\.layers\.\d+\.self_attn\.[qkvo]_proj$"

# Prompt templates (must match training)
CODE_IMG_PROMPT = "Identity code: {code}. Identify the person shown. Return only the alias."
IMG_PROMPT = "Identify the synthetic person shown here. Return only the alias."
CODE_PROMPT = "Identity code: {code}. What is the alias?"


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def make_shuffled_map(identity_ids, alias_of, seed=17):
    """Create a deranged (no fixed points) alias mapping."""
    rng = torch.Generator().manual_seed(seed)
    aliases = [alias_of[iid] for iid in identity_ids]
    n = len(aliases)
    for _ in range(100):
        perm = aliases[:]
        for i in range(n - 1, 0, -1):
            j = torch.randint(0, i + 1, (1,), generator=rng).item()
            perm[i], perm[j] = perm[j], perm[i]
        if all(perm[i] != aliases[i] for i in range(n)):
            break
    return {identity_ids[i]: perm[i] for i in range(n)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--condition", default="M_latent",
                        choices=["M_latent", "D", "M_latent_shuffled"],
                        help="Which trained adapter to audit")
    parser.add_argument("--adapter-dir", default=None,
                        help="Path to adapter_final directory")
    parser.add_argument("--out-dir", default="e2c_v3/reports")
    parser.add_argument("--image-base-dir", default="e2c/data/processed")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-wrong", type=int, default=None,
                        help="Max wrong codes per image (None = all 9)")
    parser.add_argument("--use-identity-as-code", action="store_true",
                        help="Use identity_id (syn_XX) as code instead of C0X. "
                             "Required for adapters trained with syn_XX codes.")
    parser.add_argument("--adapter-name", default=None,
                        help="Override adapter name for PEFT loading "
                             "(e.g. e2c_v3_mcomposition)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_base = Path(args.image_base_dir)

    # Default adapter paths
    if args.adapter_dir is None:
        adapter_dirs = {
            "M_latent": "e2c_v3/outputs/M_latent/adapter_final",
            "D": "e2c_v3/outputs/D/adapter_final",
            "M_latent_shuffled": "e2c_v3/outputs/M_latent_shuffled/adapter_final",
        }
        adapter_dir = Path(adapter_dirs[args.condition])
    else:
        adapter_dir = Path(args.adapter_dir)

    logger.info(f"Factorial audit: condition={args.condition}")
    logger.info(f"Adapter: {adapter_dir}")

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    with open("e2c_v3/manifests/identity_code_mapping.json") as f:
        mapping = json.load(f)
    identity_ids = [m["identity_id"] for m in mapping["mappings"]]
    identity_to_code = mapping["identity_to_code"]
    code_to_alias = mapping["code_to_alias"]
    identity_to_alias = mapping["identity_to_alias"]

    # For existing adapters trained with syn_XX as code
    if args.use_identity_as_code:
        identity_to_code = {iid: iid for iid in identity_ids}
        code_to_alias = {iid: alias for iid, alias in identity_to_alias.items()}
        logger.info("Using identity_id as code (syn_XX format)")

    with open("e2c_v3/manifests/wrong_code_pairs.json") as f:
        wrong_pairs = json.load(f)

    # Rebuild wrong_pairs if using identity-as-code format
    if args.use_identity_as_code:
        rebuilt_pairs = {"pairs": {}}
        for iid in identity_ids:
            all_wrong = []
            for other_iid in identity_ids:
                if other_iid != iid:
                    all_wrong.append({
                        "code_id": other_iid,
                        "alias": identity_to_alias[other_iid],
                        "is_ring_next": (other_iid == identity_ids[
                            (identity_ids.index(iid) + 1) % len(identity_ids)]),
                    })
            rebuilt_pairs["pairs"][iid] = {
                "true_code": iid,
                "true_alias": identity_to_alias[iid],
                "primary_wrong_code": identity_ids[
                    (identity_ids.index(iid) + 1) % len(identity_ids)],
                "all_wrong_codes": all_wrong,
            }
        wrong_pairs = rebuilt_pairs

    with open("e2c_v2/manifests/e2c_image_split.json") as f:
        split_manifest = json.load(f)
    eval_sets = defaultdict(list)
    for e in split_manifest:
        if e["identity_id"] in identity_ids:
            eval_sets[e["split"]].append({
                "identity_id": e["identity_id"],
                "image_id": e["image_id"],
                "image_path": e["image_path"],
            })

    shuffled_map = make_shuffled_map(identity_ids, identity_to_alias, args.seed)
    # Convert identity-keyed map to code-keyed map for probe builder
    shuffled_code_to_alias = {
        identity_to_code[iid]: alias
        for iid, alias in shuffled_map.items()
    }
    logger.info(f"Shuffled map: {shuffled_map}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------------------------------------------------ #
    # Build probes
    # ------------------------------------------------------------------ #
    from route_data.e2c_v3.probe_builder import (
        build_bare_image_probes,
        build_code_only_probes,
        build_correct_code_probes,
        build_shuffled_code_probes,
        build_wrong_code_probes,
        validate_factorial_probes,
    )

    all_probes = []
    for split in ("train", "validation", "test"):
        items = eval_sets[split]
        all_probes.extend(build_bare_image_probes(items, identity_to_alias))
        all_probes.extend(build_correct_code_probes(
            items, identity_to_code, code_to_alias))
        all_probes.extend(build_wrong_code_probes(
            items, identity_to_code, code_to_alias, wrong_pairs,
            n_wrong=args.n_wrong))

    # Code-only probes (no split, just identities)
    all_probes.extend(build_code_only_probes(
        identity_ids, identity_to_code, code_to_alias))
    all_probes.extend(build_shuffled_code_probes(
        identity_ids, identity_to_code, shuffled_code_to_alias))

    errors = validate_factorial_probes(all_probes)
    if errors:
        for e in errors:
            logger.error(f"Probe validation error: {e}")
        sys.exit(1)

    logger.info(f"Total probes: {len(all_probes)}")
    probe_types = defaultdict(int)
    for p in all_probes:
        probe_types[p["probe_type"]] += 1
    for pt, count in sorted(probe_types.items()):
        logger.info(f"  {pt}: {count}")

    # ------------------------------------------------------------------ #
    # Load model + adapter
    # ------------------------------------------------------------------ #
    from route_data.models.trainable.base import ModelFamilyProfile
    from route_data.models.trainable.qwen35 import Qwen35Adapter

    adapter_names = {
        "M_latent": "e2c_v3_mlatent",
        "D": "e2c_v3_direct",
        "M_latent_shuffled": "e2c_v3_mshuffled",
    }

    # Allow override for custom adapters
    adapter_name = args.adapter_name or adapter_names[args.condition]

    profile = ModelFamilyProfile(
        key="qwen35_9b", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_id="Qwen/Qwen3.5-9B",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        adapter_name=adapter_name,
        trust_remote_code=True, dtype="bfloat16",
        attn_implementation="sdpa",
        candidate_positive="Yes", candidate_negative="No",
        lora_rank=8, lora_alpha=16, lora_dropout=0.05,
        lora_scope="custom_ablation",
        lora_target_leaf_names=("q_proj", "k_proj", "v_proj", "o_proj"),
        lora_scope_regex=SCOPE_REGEX,
        r2mu_candidate_layers=(8, 16, 24, 29), r2mu_n_select_layers=4,
        language_layer_path="model.language_model.layers",
        language_hidden_size=4096, intermediate_size=12288,
        num_language_layers=32, lora_expected_target_modules=128,
    )

    adapter = Qwen35Adapter(profile)
    model, processor = adapter.load_model_processor(
        model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", device=args.device, training=False,
    )
    model = adapter.load_unlearning_adapter(
        model, adapter_dir, adapter_name=profile.adapter_name)
    model.eval()

    from route_data.config import ModelConfig
    eval_config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=args.seed,
    )
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=eval_config)

    # ------------------------------------------------------------------ #
    # Run probes
    # ------------------------------------------------------------------ #
    logger.info("Running factorial probes...")
    results = []
    n_done = 0

    with torch.no_grad():
        for probe in all_probes:
            image = None
            if probe["has_image"]:
                image = load_image(str(image_base / probe["image_path"]))

            gen = backend.generate(image, probe["prompt"], max_new_tokens=5)
            pred = (gen.text.strip().split()[0].strip(".,!?")
                    if gen.text.strip() else "")

            record = {
                "probe_id": probe["probe_id"],
                "probe_type": probe["probe_type"],
                "identity_id": probe["identity_id"],
                "image_id": probe.get("image_id"),
                "code_id": probe.get("code_id"),
                "prediction": pred,
            }

            # Add expected values for metric computation
            if "expected_alias" in probe:
                record["expected_alias"] = probe["expected_alias"]
            if "expected_alias_if_follow_code" in probe:
                record["expected_alias_if_follow_code"] = probe[
                    "expected_alias_if_follow_code"]
            if "expected_alias_if_follow_image" in probe:
                record["expected_alias_if_follow_image"] = probe[
                    "expected_alias_if_follow_image"]

            results.append(record)
            n_done += 1
            if n_done % 50 == 0:
                logger.info(f"  {n_done}/{len(all_probes)} probes done")

    logger.info(f"All {n_done} probes complete")

    # ------------------------------------------------------------------ #
    # Compute metrics
    # ------------------------------------------------------------------ #
    from route_data.e2c_v3.causal_metrics import (
        code_target_alignment,
        compute_per_identity_metrics,
        image_target_alignment,
    )

    # Aggregate metrics by probe type
    bare_correct = sum(
        1 for r in results
        if r["probe_type"] == "bare_image"
        and r["prediction"].lower() == r.get("expected_alias", "").lower()
    )
    bare_total = sum(1 for r in results if r["probe_type"] == "bare_image")

    correct_code_correct = sum(
        1 for r in results
        if r["probe_type"] == "image_correct_code"
        and r["prediction"].lower() == r.get("expected_alias", "").lower()
    )
    correct_code_total = sum(
        1 for r in results if r["probe_type"] == "image_correct_code")

    wrong_follow_code = sum(
        1 for r in results
        if r["probe_type"] == "image_wrong_code"
        and r["prediction"].lower() == r.get(
            "expected_alias_if_follow_code", "").lower()
    )
    wrong_follow_image = sum(
        1 for r in results
        if r["probe_type"] == "image_wrong_code"
        and r["prediction"].lower() == r.get(
            "expected_alias_if_follow_image", "").lower()
    )
    wrong_total = sum(
        1 for r in results if r["probe_type"] == "image_wrong_code")

    # Code-only metrics
    code_only_correct = sum(
        1 for r in results
        if r["probe_type"] == "code_only_correct"
        and r["prediction"].lower() == r.get("expected_alias", "").lower()
    )
    code_only_total = sum(
        1 for r in results if r["probe_type"] == "code_only_correct")

    # Shuffled code-only metrics
    shuffled_correct = sum(
        1 for r in results
        if r["probe_type"] == "code_only_shuffled"
        and r["prediction"].lower() == r.get("expected_alias", "").lower()
    )
    shuffled_total = sum(
        1 for r in results if r["probe_type"] == "code_only_shuffled")

    # Intervention change rate: compare correct vs wrong code per (image)
    correct_preds = {}
    wrong_preds = {}
    for r in results:
        if r["probe_type"] == "image_correct_code":
            key = f"{r['identity_id']}__{r['image_id']}"
            correct_preds[key] = r["prediction"]
        elif r["probe_type"] == "image_wrong_code":
            key = f"{r['identity_id']}__{r['image_id']}__{r['code_id']}"
            wrong_preds[key] = r["prediction"]

    # For change rate: compare each wrong-code pred to its correct-code baseline
    changes = 0
    change_total = 0
    for key, wrong_pred in wrong_preds.items():
        parts = key.split("__")
        base_key = f"{parts[0]}__{parts[1]}"
        if base_key in correct_preds:
            change_total += 1
            if wrong_pred != correct_preds[base_key]:
                changes += 1
    change_rate = changes / change_total if change_total > 0 else 0.0

    # Code target alignment (for wrong codes)
    wrong_code_expected = {}
    wrong_image_expected = {}
    for r in results:
        if r["probe_type"] == "image_wrong_code":
            key = f"{r['identity_id']}__{r['image_id']}__{r['code_id']}"
            wrong_code_expected[key] = r.get("expected_alias_if_follow_code", "")
            wrong_image_expected[key] = r.get("expected_alias_if_follow_image", "")

    wrong_preds_flat = {}
    for r in results:
        if r["probe_type"] == "image_wrong_code":
            key = f"{r['identity_id']}__{r['image_id']}__{r['code_id']}"
            wrong_preds_flat[key] = r["prediction"]

    align_c = code_target_alignment(wrong_preds_flat, wrong_code_expected)
    align_x = image_target_alignment(wrong_preds_flat, wrong_image_expected)

    # Per-identity breakdown
    per_identity = compute_per_identity_metrics(results, identity_ids)

    # Prediction distribution
    pred_dist = defaultdict(int)
    for r in results:
        pred_dist[r["prediction"].lower()] += 1

    # ------------------------------------------------------------------ #
    # Classify failure mode (Phase A3)
    # ------------------------------------------------------------------ #
    bare_acc = bare_correct / bare_total if bare_total > 0 else 0.0
    correct_code_acc = (correct_code_correct / correct_code_total
                        if correct_code_total > 0 else 0.0)
    img_plus_correct_c = correct_code_acc

    # Case A: correct code rescues M (img+correct_C >> img bare)
    # Case B: correct code does NOT rescue M (both ≈ 0)
    # Case C: wrong code changes outputs but doesn't align

    if img_plus_correct_c > 0.20 and bare_acc < 0.10:
        failure_case = "A"
        failure_description = (
            "Correct code rescues M-latent. "
            "Composition may already be present but X→C inference is weak."
        )
    elif img_plus_correct_c < 0.10 and bare_acc < 0.10:
        if change_rate > 0.50 and align_c < 0.30:
            failure_case = "C"
            failure_description = (
                "Wrong code changes outputs but does not align. "
                "Code-conditioned perturbation, not causal mediation."
            )
        else:
            failure_case = "B"
            failure_description = (
                "Correct code does not rescue M. "
                "Model lacks composed image+code inference. "
                "Proceed to Phase B (explicit composition repair)."
            )
    else:
        failure_case = "other"
        failure_description = (
            f"Unexpected pattern: bare={bare_acc:.3f}, "
            f"img+correct_C={img_plus_correct_c:.3f}, "
            f"change_rate={change_rate:.3f}, align_C={align_c:.3f}"
        )

    # ------------------------------------------------------------------ #
    # Build report
    # ------------------------------------------------------------------ #
    report = {
        "experiment": "E2C-v3 Phase A Factorial Audit",
        "condition": args.condition,
        "adapter_dir": str(adapter_dir),
        "seed": args.seed,
        "n_wrong_codes_per_image": args.n_wrong if args.n_wrong else 9,
        "total_probes": len(all_probes),
        "aggregate_metrics": {
            "bare_accuracy": bare_acc,
            "bare_correct": bare_correct,
            "bare_total": bare_total,
            "correct_code_accuracy": correct_code_acc,
            "correct_code_correct": correct_code_correct,
            "correct_code_total": correct_code_total,
            "wrong_code_target_agreement": (
                wrong_follow_code / wrong_total if wrong_total > 0 else 0.0),
            "wrong_code_image_agreement": (
                wrong_follow_image / wrong_total if wrong_total > 0 else 0.0),
            "wrong_code_follow_code": wrong_follow_code,
            "wrong_code_follow_image": wrong_follow_image,
            "wrong_code_total": wrong_total,
            "change_rate": change_rate,
            "change_count": changes,
            "change_total": change_total,
            "code_target_alignment_C": align_c,
            "image_target_alignment_X": align_x,
            "code_only_accuracy": (
                code_only_correct / code_only_total
                if code_only_total > 0 else 0.0),
            "code_only_correct": code_only_correct,
            "code_only_total": code_only_total,
            "shuffled_code_accuracy": (
                shuffled_correct / shuffled_total
                if shuffled_total > 0 else 0.0),
            "shuffled_correct": shuffled_correct,
            "shuffled_total": shuffled_total,
        },
        "per_identity": per_identity,
        "prediction_distribution": dict(sorted(
            pred_dist.items(), key=lambda x: -x[1])),
        "failure_classification": {
            "case": failure_case,
            "description": failure_description,
        },
        "verdict_flags": {
            "CODE_CAUSAL_SENSITIVITY_ESTABLISHED": change_rate >= 0.50,
            "CODE_TARGET_ALIGNED_CONTROL": (
                "Full" if align_c >= 0.80
                else "Partial" if align_c >= 0.30
                else "None"
            ),
            "IMAGE_TO_CODE_TO_OUTPUT_ROUTE": (
                bare_acc >= 0.80 and align_c >= 0.80),
            "DIRECT_IMAGE_ROUTE": bare_acc >= 0.80,
            "SHUFFLED_CAUSAL_CONTROL": (
                (shuffled_correct / shuffled_total
                 if shuffled_total > 0 else 0.0) >= 0.80),
            "COMPOSITION_RESCUE": img_plus_correct_c >= 0.20,
            "ROUTE_ESTABLISHED": False,
        },
    }

    # Save report
    report_path = out_dir / f"factorial_eval_{args.condition}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    # Also save per-probe results
    probe_path = out_dir / f"factorial_probes_{args.condition}.jsonl"
    with open(probe_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # ------------------------------------------------------------------ #
    # Log summary
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info(f"FACTORIAL AUDIT COMPLETE: {args.condition}")
    logger.info("=" * 60)
    logger.info(f"  bare_accuracy          = {bare_acc:.3f} "
                f"({bare_correct}/{bare_total})")
    logger.info(f"  correct_code_accuracy   = {correct_code_acc:.3f} "
                f"({correct_code_correct}/{correct_code_total})")
    logger.info(f"  wrong_code_follow_code  = "
                f"{wrong_follow_code / wrong_total if wrong_total else 0:.3f} "
                f"({wrong_follow_code}/{wrong_total})")
    logger.info(f"  wrong_code_follow_image = "
                f"{wrong_follow_image / wrong_total if wrong_total else 0:.3f} "
                f"({wrong_follow_image}/{wrong_total})")
    logger.info(f"  change_rate             = {change_rate:.3f} "
                f"({changes}/{change_total})")
    logger.info(f"  code_target_alignment   = {align_c:.3f}")
    logger.info(f"  image_target_alignment  = {align_x:.3f}")
    logger.info(f"  code_only_accuracy      = "
                f"{code_only_correct / code_only_total if code_only_total else 0:.3f} "
                f"({code_only_correct}/{code_only_total})")
    logger.info(f"  shuffled_code_accuracy  = "
                f"{shuffled_correct / shuffled_total if shuffled_total else 0:.3f} "
                f"({shuffled_correct}/{shuffled_total})")
    logger.info(f"  FAILURE CASE: {failure_case}")
    logger.info(f"  {failure_description}")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  Probes: {probe_path}")


if __name__ == "__main__":
    main()
