#!/usr/bin/env python3
"""E2C-v2 staged calibration — M2-only (Alias→Fact) or D-only (Image→Fact).

Usage:
    # M2: text-only alias→fact on calibration identities
    python scripts/e2c_v2_calibrate_stage.py --stage M2 --device cuda:0

    # D: image→fact on calibration identities
    python scripts/e2c_v2_calibrate_stage.py --stage D --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from route_data.e2c.training_dataset import E2CTrainingDataset, load_records_from_jsonl

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("e2c_v2_calibrate")

STAGE_CONFIG = {
    "M2": {
        "source_file": "M_train.jsonl",
        "task_filter": "name_to_attribute",
        "description": "Alias → Fact (text-only)",
    },
    "D": {
        "source_file": "D_train.jsonl",
        "task_filter": "image_to_attribute",
        "description": "Image → Fact (direct)",
    },
}


def load_image(path: str) -> Any:
    from PIL import Image
    return Image.open(path).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["M2", "D"])
    parser.add_argument("--v2-dir", default="e2c_v2")
    parser.add_argument("--image-base-dir", default="e2c/data/processed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--checkpoint-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    stage_cfg = STAGE_CONFIG[args.stage]
    v2_dir = Path(args.v2_dir)
    output_dir = v2_dir / "outputs" / "calibration" / args.stage.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "stage": args.stage,
        "description": stage_cfg["description"],
        "lr": args.lr, "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "seed": args.seed, "device": args.device,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    logger.info(f"{args.stage} calibration: {stage_cfg['description']}")
    logger.info(f"lr={args.lr}, steps={args.steps}")

    # Load and filter records
    records_path = v2_dir / "data" / "calibration" / stage_cfg["source_file"]
    all_records = load_records_from_jsonl(records_path)
    records = [r for r in all_records if r["task"] == stage_cfg["task_filter"]]
    logger.info(f"Loaded {len(records)} {stage_cfg['task_filter']} records")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load model
    logger.info("Loading Qwen3.5-9B...")
    from route_data.models.trainable.qwen35 import Qwen35Adapter
    from route_data.models.trainable.base import ModelFamilyProfile

    profile = ModelFamilyProfile(
        key="qwen35_9b", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_id="Qwen/Qwen3.5-9B",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        adapter_name=f"e2c_v2_{args.stage.lower()}",
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

    # Build dataset
    dataset = E2CTrainingDataset(
        records=records, processor=processor, adapter=adapter,
        image_loader=load_image, image_base_dir=args.image_base_dir,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=adapter.collate, num_workers=0,
    )

    # Optimizer
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr, weight_decay=0.0)
    if args.warmup_steps > 0 and args.steps > args.warmup_steps:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=args.steps - args.warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                 milestones=[args.warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.steps)

    # Training loop
    logger.info(f"Training {args.steps} steps on {len(records)} records...")
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
                elif isinstance(v, list):
                    bd[k] = v
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

                if global_step % 10 == 0 or global_step <= 5:
                    avg_loss = running_loss / running_count
                    trace.append({"step": global_step, "loss": avg_loss,
                                  "lr": optimizer.param_groups[0]["lr"], "epoch": epoch})
                    logger.info(f"Step {global_step}/{args.steps} loss={avg_loss:.4f}")

                if global_step % args.checkpoint_steps == 0:
                    adapter.save_unlearning_adapter(
                        model, output_dir / "checkpoints" / f"step_{global_step}")

                if global_step >= args.steps:
                    break
            if global_step >= args.steps:
                break
        if global_step >= args.steps:
            break

    # Save final
    logger.info("Saving final adapter...")
    adapter.save_unlearning_adapter(model, output_dir / "adapter_final")

    with open(output_dir / "training_trace.jsonl", "w") as f:
        for e in trace:
            f.write(json.dumps(e) + "\n")

    summary = {
        "stage": args.stage,
        "total_steps": global_step,
        "final_loss": running_loss / max(running_count, 1),
        "trainable_parameters": trainable,
        "n_training_records": len(records),
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    logger.info(f"{args.stage} calibration complete. Loss: {summary['final_loss']:.4f}")


if __name__ == "__main__":
    main()
