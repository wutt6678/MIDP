#!/usr/bin/env python3
"""E2C-v2 diagnostic: trainable-scope ablation (STRIDE-4 language layers variant).

Language attention restricted to layers {3,7,11,15,19,23,27,31} (34 modules total
with merger) to fit contended GPU memory. S1/S0 definitions unchanged.

Fixed: dataset (N=10 experimental identities), M1-only task, 1000 steps,
LR 2e-5, grad_accum 8, LoRA rank 8, seed 17, prompts.
Varied: trainable parameter scope only.

Scopes (exact Qwen3.5 module names):
  S1: multimodal/fusion only
      model.visual.merger.linear_fc{1,2}                       (2 modules)
  S2: fusion + language attention
      S1 + model.language_model.layers.*.self_attn.{q,k,v,o}_proj  (130)
  S3: selected late visual + fusion + language attention
      S2 + model.visual.blocks.{22..26}.{attn.qkv,attn.proj,
      mlp.linear_fc1,mlp.linear_fc2}                            (150)

S0 (language attention only) is the existing negative baseline:
  e2c_v2/outputs/diag_i2n_capacity/n10 (same protocol, 1000 steps).

Primary outcome: M1 I2N (free-gen + candidate ranking) on train/val/test.

Usage:
    python scripts/e2c_v2_diag_scope_ablation.py --scope S1 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as tnn
from torch.utils.data import DataLoader

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from route_data.e2c.training_dataset import E2CTrainingDataset, load_records_from_jsonl

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("e2c_v2_diag_scope")

TRAIN_PROMPT = "Identify the synthetic person shown here. Return only the alias."
EVAL_PROMPT = "Who is shown? Answer with the synthetic identity name only."
STEPS = 1000
WARMUP = 100

SCOPE_REGEXES: dict[str, list[str]] = {
    "S1": [
        r"^model\.visual\.merger\.linear_fc[12]$",
    ],
    "S2": [
        r"^model\.visual\.merger\.linear_fc[12]$",
        r"^model\.language_model\.layers\.(3|7|11|15|19|23|27|31)\.self_attn\.[qkvo]_proj$",
    ],
    "S3": [
        r"^model\.visual\.merger\.linear_fc[12]$",
        r"^model\.language_model\.layers\.(3|7|11|15|19|23|27|31)\.self_attn\.[qkvo]_proj$",
        r"^model\.visual\.blocks\.(2[2-6])\.(attn\.qkv|attn\.proj|mlp\.linear_fc1|mlp\.linear_fc2)$",
    ],
}


def load_image(path: str) -> Any:
    from PIL import Image
    return Image.open(path).convert("RGB")


def resolve_scope_targets(model: tnn.Module, scope: str) -> list[str]:
    patterns = [re.compile(p) for p in SCOPE_REGEXES[scope]]
    targets = []
    for name, mod in model.named_modules():
        if isinstance(mod, tnn.Linear) and any(p.match(name) for p in patterns):
            targets.append(name)
    targets.sort()
    if not targets:
        raise RuntimeError(f"Scope {scope} matched no modules")
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", required=True, choices=["S1", "S2", "S3"])
    parser.add_argument("--out-base", default="e2c_v2/outputs/diag_scope_stride4")
    parser.add_argument("--image-base-dir", default="e2c/data/processed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    scope = args.scope
    output_dir = Path(args.out_base) / scope
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Data: all 10 experimental identities, I2N records only (same as S0)
    # ------------------------------------------------------------------ #
    with open("e2c_v2/manifests/e2c_image_split.json") as f:
        split_manifest = json.load(f)
    identity_ids = sorted({e["identity_id"] for e in split_manifest
                           if e["identity_id"].startswith("syn_")
                           and e["identity_id"][4:].isdigit()})[:10]

    all_records = load_records_from_jsonl("e2c_v2/data/experimental/M_train.jsonl")
    records = [r for r in all_records
               if r["task"] == "image_to_identity" and r["identity_id"] in identity_ids]
    alias_of = {r["identity_id"]: r["alias"] for r in records}
    all_aliases = sorted({r["alias"] for r in all_records
                          if r["task"] == "image_to_identity"})

    eval_sets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in split_manifest:
        if e["identity_id"] in identity_ids:
            eval_sets[e["split"]].append({
                "identity_id": e["identity_id"],
                "image_id": e["image_id"],
                "image_path": e["image_path"],
                "correct_alias": alias_of[e["identity_id"]],
            })

    logger.info(f"Scope {scope}: {len(records)} I2N records, steps={STEPS}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------------------------------------------------ #
    # Model + scope-specific LoRA
    # ------------------------------------------------------------------ #
    logger.info("Loading Qwen3.5-9B...")
    from route_data.models.trainable.base import ModelFamilyProfile
    from route_data.models.trainable.qwen35 import Qwen35Adapter

    profile = ModelFamilyProfile(
        key="qwen35_9b", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_id="Qwen/Qwen3.5-9B",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        adapter_name=f"e2c_v2_scope_{scope.lower()}",
        trust_remote_code=True, dtype="bfloat16", attn_implementation="sdpa",
        candidate_positive="Yes", candidate_negative="No",
        lora_rank=8, lora_alpha=16, lora_dropout=0.05,
        lora_scope="custom_ablation",
        lora_target_leaf_names=("q_proj", "k_proj", "v_proj", "o_proj"),
        lora_scope_regex=r"^model\.language_model\.layers\.\d+\.self_attn\.",
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
        dtype="bfloat16", device=args.device, training=True,
    )

    target_modules = resolve_scope_targets(model, scope)
    logger.info(f"Scope {scope} targets: {len(target_modules)} modules")
    for t in target_modules[:6]:
        logger.info(f"  {t}")
    if len(target_modules) > 6:
        logger.info(f"  ... (+{len(target_modules) - 6} more)")

    model = adapter.attach_unlearning_adapter(
        model, lora_rank=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=target_modules, adapter_name=profile.adapter_name,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable: {trainable:,} parameters")

    config = {
        "scope": scope, "scope_regexes": SCOPE_REGEXES[scope],
        "target_modules": target_modules,
        "n_target_modules": len(target_modules),
        "trainable_parameters": trainable,
        "lora_rank": 8, "lora_alpha": 16,
        "steps": STEPS, "warmup_steps": WARMUP, "lr": args.lr,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "seed": args.seed, "device": args.device,
        "n_identities": 10, "identity_ids": identity_ids,
        "n_train_records": len(records),
        "train_prompt": TRAIN_PROMPT, "eval_prompt": EVAL_PROMPT,
        "s0_baseline": "e2c_v2/outputs/diag_i2n_capacity/n10",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    # ------------------------------------------------------------------ #
    # Training (M1-only, 1000 steps — identical budget to S0)
    # ------------------------------------------------------------------ #
    dataset = E2CTrainingDataset(
        records=records, processor=processor, adapter=adapter,
        image_loader=load_image, image_base_dir=args.image_base_dir,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=adapter.collate, num_workers=0,
    )

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr, weight_decay=0.0)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP)
    cosine = CosineAnnealingLR(optimizer, T_max=STEPS - WARMUP)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[WARMUP])

    logger.info(f"Training {scope}: {STEPS} steps on {len(records)} records...")
    model.train()
    trace = []
    global_step = 0
    running_loss = 0.0
    running_count = 0

    for epoch in range(1000):
        for batch in dataloader:
            bd = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    bd[k] = v.to(args.device)
                else:
                    bd[k] = v

            outputs = model(
                input_ids=bd["input_ids"], attention_mask=bd["attention_mask"],
                labels=bd["labels"],
                **{k: v for k, v in bd.items()
                   if k not in ("input_ids", "attention_mask", "labels")
                   and isinstance(v, torch.Tensor) and not k.startswith("_")},
            )
            loss = outputs.loss / args.grad_accum
            loss.backward()
            running_loss += loss.item() * args.grad_accum
            running_count += 1

            if running_count % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 20 == 0 or global_step <= 5:
                    avg_loss = running_loss / running_count
                    trace.append({"step": global_step, "loss": avg_loss,
                                  "lr": optimizer.param_groups[0]["lr"], "epoch": epoch})
                    logger.info(f"[{scope}] Step {global_step}/{STEPS} loss={avg_loss:.4f}")

                if global_step >= STEPS:
                    break
            if global_step >= STEPS:
                break
        if global_step >= STEPS:
            break

    logger.info("Saving final adapter...")
    adapter.save_unlearning_adapter(model, output_dir / "adapter_final")
    with open(output_dir / "training_trace.jsonl", "w") as f:
        for e in trace:
            f.write(json.dumps(e) + "\n")

    # ------------------------------------------------------------------ #
    # Evaluation: free generation + candidate ranking, all splits
    # ------------------------------------------------------------------ #
    model.eval()
    from route_data.config import ModelConfig
    eval_config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=args.seed,
    )
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=eval_config,
    )

    image_base = Path(args.image_base_dir)
    results: dict[str, Any] = {"scope": scope, "steps": STEPS,
                               "trainable_parameters": trainable,
                               "n_target_modules": len(target_modules),
                               "final_loss": running_loss / max(running_count, 1)}
    confusion: Counter = Counter()

    for split in ("train", "validation", "test"):
        free_correct = 0
        rank_correct = 0
        items_out = []
        for item in eval_sets[split]:
            image = load_image(str(image_base / item["image_path"]))
            expected = item["correct_alias"]

            gen = backend.generate(image, EVAL_PROMPT, max_new_tokens=5)
            pred = gen.text.strip()
            pred_norm = pred.split()[0].strip(".,!?") if pred.split() else ""
            free_ok = pred_norm.lower() == expected.lower()
            free_correct += int(free_ok)

            resp = backend.score_candidates(image, EVAL_PROMPT, all_aliases)
            scores = {cs.candidate: cs.log_probability for cs in resp.candidate_scores}
            ranked = max(scores, key=scores.get)
            rank_ok = ranked == expected
            rank_correct += int(rank_ok)

            if split == "test":
                key = pred_norm if pred_norm.lower() in {a.lower() for a in all_aliases} else "OTHER"
                confusion[(expected, key)] += 1

            items_out.append({
                "image_id": item["image_id"], "expected_alias": expected,
                "free_gen_prediction": pred, "free_gen_correct": free_ok,
                "ranked_top": ranked, "rank_correct": rank_ok,
            })

        total = len(eval_sets[split])
        results[f"{split}_free_gen_accuracy"] = free_correct / total
        results[f"{split}_candidate_rank_accuracy"] = rank_correct / total
        with open(output_dir / f"eval_{split}.jsonl", "w") as f:
            f.writelines(json.dumps(it) + "\n" for it in items_out)
        logger.info(f"[{scope}] {split}: free-gen {free_correct}/{total} "
                    f"({free_correct / total:.3f}), ranking {rank_correct}/{total} "
                    f"({rank_correct / total:.3f})")

    matrix = {}
    for (true_a, pred_a), cnt in confusion.items():
        matrix.setdefault(true_a, {})[pred_a] = cnt
    results["test_confusion_matrix"] = matrix
    results["confusion_labels"] = all_aliases + ["OTHER"]

    with open(output_dir / "scope_summary.json", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    logger.info(f"Scope {scope} complete: "
                f"train={results['train_free_gen_accuracy']:.3f} "
                f"val={results['validation_free_gen_accuracy']:.3f} "
                f"test={results['test_free_gen_accuracy']:.3f}")


if __name__ == "__main__":
    main()
