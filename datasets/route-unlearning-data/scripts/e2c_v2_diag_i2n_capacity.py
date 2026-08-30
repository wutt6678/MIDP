#!/usr/bin/env python3
"""E2C-v2 diagnostic: I2N binding capacity curve N -> I2N(N).

Trains M1-only (image -> alias) on the first N experimental identities with a
step budget proportional to N (100 optimizer steps per identity, matching the
canonical M1 calibration of 200 steps for 2 identities). Everything else is
fixed: same images, same 10/3/3 split, rank-8 language_attention_only LoRA,
LR 2e-5, grad_accum 8, same prompts, seed 17.

Measured per N:
  - I2N free-generation accuracy on train / validation / test splits
  - candidate-ranking accuracy (all 10 aliases scored via log-probability)
  - alias confusion matrix (free generation, test split)

Usage:
    python scripts/e2c_v2_diag_i2n_capacity.py --n 4 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from route_data.e2c.training_dataset import E2CTrainingDataset, load_records_from_jsonl

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("e2c_v2_diag_i2n")

TRAIN_PROMPT = "Identify the synthetic person shown here. Return only the alias."
EVAL_PROMPT = "Who is shown? Answer with the synthetic identity name only."
STEPS_PER_IDENTITY = 100  # canonical M1 calibration: 200 steps / 2 identities


def load_image(path: str) -> Any:
    from PIL import Image
    return Image.open(path).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, choices=[2, 4, 6, 8, 10])
    parser.add_argument("--out-base", default="e2c_v2/outputs/diag_i2n_capacity")
    parser.add_argument("--image-base-dir", default="e2c/data/processed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    n = args.n
    steps = STEPS_PER_IDENTITY * n
    warmup_steps = max(1, steps // 10)
    output_dir = Path(args.out_base) / f"n{n}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Data: first N experimental identities, I2N records only
    # ------------------------------------------------------------------ #
    with open("e2c_v2/manifests/e2c_image_split.json") as f:
        split_manifest = json.load(f)
    # Experimental identities are syn_00..syn_09 (calibration ids are syn_cal_*)
    identity_ids = sorted({e["identity_id"] for e in split_manifest
                           if e["identity_id"].startswith("syn_")
                           and e["identity_id"][4:].isdigit()})[:n]

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

    logger.info(f"N={n}: identities={identity_ids}")
    logger.info(f"Training records: {len(records)}; steps={steps}, warmup={warmup_steps}")
    for split, items in eval_sets.items():
        logger.info(f"Eval {split}: {len(items)} images")

    config = {
        "n_identities": n, "identity_ids": identity_ids,
        "steps": steps, "steps_per_identity": STEPS_PER_IDENTITY,
        "warmup_steps": warmup_steps, "lr": args.lr,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "seed": args.seed, "device": args.device,
        "train_prompt": TRAIN_PROMPT, "eval_prompt": EVAL_PROMPT,
        "n_train_records": len(records), "candidate_aliases": all_aliases,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------------------------------------------------ #
    # Model + LoRA (identical to canonical)
    # ------------------------------------------------------------------ #
    logger.info("Loading Qwen3.5-9B...")
    from route_data.models.trainable.base import ModelFamilyProfile
    from route_data.models.trainable.qwen35 import Qwen35Adapter

    profile = ModelFamilyProfile(
        key="qwen35_9b", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_id="Qwen/Qwen3.5-9B",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        adapter_name="e2c_v2_diag_i2n",
        trust_remote_code=True, dtype="bfloat16", attn_implementation="sdpa",
        candidate_positive="Yes", candidate_negative="No",
        lora_rank=8, lora_alpha=16, lora_dropout=0.05,
        lora_scope="language_attention_only",
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

    target_modules = adapter.resolve_lora_targets(model)
    logger.info(f"LoRA targets: {len(target_modules)} modules")
    model = adapter.attach_unlearning_adapter(
        model, lora_rank=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=target_modules, adapter_name=profile.adapter_name,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable: {trainable:,} parameters")

    # ------------------------------------------------------------------ #
    # Training (M1-only)
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
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[warmup_steps])

    logger.info(f"Training M1-only: {steps} steps on {len(records)} records...")
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
                    logger.info(f"Step {global_step}/{steps} loss={avg_loss:.4f}")

                if global_step >= steps:
                    break
            if global_step >= steps:
                break
        if global_step >= steps:
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
    results: dict[str, Any] = {"n": n, "identity_ids": identity_ids,
                               "steps": steps,
                               "final_loss": running_loss / max(running_count, 1)}
    confusion: Counter = Counter()

    for split in ("train", "validation", "test"):
        free_correct = 0
        rank_correct = 0
        items_out = []
        for item in eval_sets[split]:
            image = load_image(str(image_base / item["image_path"]))
            expected = item["correct_alias"]

            # Free generation (same protocol as canonical I2N probes)
            gen = backend.generate(image, EVAL_PROMPT, max_new_tokens=5)
            pred = gen.text.strip()
            pred_norm = pred.split()[0].strip(".,!?") if pred.split() else ""
            free_ok = pred_norm.lower() == expected.lower()
            free_correct += int(free_ok)

            # Candidate ranking over all 10 aliases
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
                "candidate_scores": scores,
            })

        total = len(eval_sets[split])
        results[f"{split}_free_gen_accuracy"] = free_correct / total
        results[f"{split}_candidate_rank_accuracy"] = rank_correct / total
        with open(output_dir / f"eval_{split}.jsonl", "w") as f:
            f.writelines(json.dumps(it) + "\n" for it in items_out)
        logger.info(f"{split}: free-gen {free_correct}/{total} "
                    f"({free_correct / total:.3f}), ranking {rank_correct}/{total} "
                    f"({rank_correct / total:.3f})")

    # Confusion matrix (test split): rows = true alias, cols = predicted
    matrix = {}
    for (true_a, pred_a), cnt in confusion.items():
        matrix.setdefault(true_a, {})[pred_a] = cnt
    results["test_confusion_matrix"] = matrix
    results["confusion_labels"] = all_aliases + ["OTHER"]

    with open(output_dir / "diagnostic_summary.json", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    logger.info(f"N={n} diagnostic complete: "
                f"train={results['train_free_gen_accuracy']:.3f} "
                f"val={results['validation_free_gen_accuracy']:.3f} "
                f"test={results['test_free_gen_accuracy']:.3f}")


if __name__ == "__main__":
    main()
