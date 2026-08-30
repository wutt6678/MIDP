#!/usr/bin/env python3
"""E2C train route — train M, D, or M-shuffled with LoRA on Qwen3.5-9B.

Usage:
    python scripts/e2c_train_route.py \
        --condition M \
        --config e2c/configs/e2c_canonical.yaml \
        --manifest-dir e2c/manifests \
        --dataset-dir e2c/data/splits \
        --output-dir e2c/outputs/<code_sha>/M \
        --image-base-dir e2c/data/processed

Reuses existing Qwen35Adapter, HuggingFaceChatAdapter, LoRA, and scoring
infrastructure from the MIDP project.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

# Ensure project source is importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from route_data.e2c.provenance import (
    build_provenance,
    capture_lora_config,
    capture_model_fingerprint,
    capture_processor_fingerprint,
    capture_training_config,
    get_git_state,
    write_provenance,
)
from route_data.e2c.synthetic_manifest import sha256_file
from route_data.e2c.training_dataset import E2CTrainingDataset, load_records_from_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2c_train")


def load_image(path: str) -> Any:
    """Load an image from disk using PIL."""
    from PIL import Image
    return Image.open(path).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="E2C route training")
    parser.add_argument("--condition", required=True, choices=["M", "D", "M_shuffled"])
    parser.add_argument("--config", default="e2c/configs/e2c_canonical.yaml")
    parser.add_argument("--manifest-dir", default="e2c/manifests")
    parser.add_argument("--dataset-dir", default="e2c/data/splits")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-base-dir", default="e2c/data/processed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true",
                        help="Run smoke test with minimal steps")
    parser.add_argument("--smoke-steps", type=int, default=5)
    parser.add_argument("--checkpoint-steps", type=int, default=50,
                        help="Save checkpoint every N steps")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    e2c_cfg = config["e2c"]
    model_cfg = e2c_cfg["model"]
    lora_cfg = e2c_cfg["lora"]
    train_cfg = e2c_cfg["training"]
    seed = e2c_cfg["seed"]

    condition = args.condition
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"E2C training: condition={condition}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Seed: {seed}")

    # Determine training budget
    optimizer_steps = train_cfg["optimizer_steps"]
    if args.smoke:
        optimizer_steps = args.smoke_steps
        logger.info(f"SMOKE mode: {optimizer_steps} steps only")

    # Save resolved config
    resolved_config = {
        "condition": condition,
        "seed": seed,
        "model": model_cfg,
        "lora": lora_cfg,
        "training": {**train_cfg, "optimizer_steps": optimizer_steps},
        "smoke": args.smoke,
    }
    with open(output_dir / "config_resolved.json", "w") as f:
        json.dump(resolved_config, f, indent=2, sort_keys=True)

    # ------------------------------------------------------------------ #
    # Set seeds
    # ------------------------------------------------------------------ #
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------ #
    # Load model and processor
    # ------------------------------------------------------------------ #
    logger.info("Loading model and processor...")
    from route_data.models.trainable.base import ModelFamilyProfile
    from route_data.models.trainable.qwen35 import Qwen35Adapter

    profile = ModelFamilyProfile(
        key="qwen35_9b",
        model_id=model_cfg["id"],
        revision=model_cfg["revision"],
        processor_id=model_cfg["processor_id"],
        processor_revision=model_cfg["processor_revision"],
        adapter_name="e2c_route",
        trust_remote_code=model_cfg["trust_remote_code"],
        dtype=model_cfg["dtype"],
        attn_implementation=model_cfg["attn_implementation"],
        candidate_positive="Yes",
        candidate_negative="No",
        lora_rank=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        lora_scope="language_attention_only",
        lora_target_leaf_names=tuple(lora_cfg["target_modules"]),
        lora_scope_regex=lora_cfg["scope_regex"],
        r2mu_candidate_layers=(8, 16, 24, 29),
        r2mu_n_select_layers=4,
        language_layer_path="model.language_model.layers",
        language_hidden_size=4096,
        intermediate_size=12288,
        num_language_layers=32,
        lora_expected_target_modules=128,
    )

    adapter = Qwen35Adapter(profile)
    model, processor = adapter.load_model_processor(
        model_id=model_cfg["id"],
        revision=model_cfg["revision"],
        processor_revision=model_cfg["processor_revision"],
        dtype=model_cfg["dtype"],
        device=args.device,
        training=True,
    )

    # Resolve LoRA targets and attach adapter
    logger.info("Applying LoRA adapter...")
    target_modules = adapter.resolve_lora_targets(model)
    logger.info(f"LoRA targets: {len(target_modules)} modules")

    model = adapter.attach_unlearning_adapter(
        model,
        lora_rank=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=target_modules,
        adapter_name=profile.adapter_name,
    )

    # Snapshot protected parameters
    protected_snapshot = adapter.snapshot_protected_parameters(model)

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ------------------------------------------------------------------ #
    # Build dataset
    # ------------------------------------------------------------------ #
    dataset_dir = Path(args.dataset_dir)
    condition_file = {
        "M": "M_train.jsonl",
        "D": "D_train.jsonl",
        "M_shuffled": "M_shuffled_train.jsonl",
    }[condition]

    records = load_records_from_jsonl(dataset_dir / condition_file)
    logger.info(f"Loaded {len(records)} training records for {condition}")

    # Filter to calibration only for smoke test
    if args.smoke:
        records = [r for r in records if "cal" in r.get("identity_id", "")]
        if not records:
            logger.warning("No calibration records found, using first 4 records")
            all_records = load_records_from_jsonl(dataset_dir / condition_file)
            records = all_records[:4]
        logger.info(f"Smoke: using {len(records)} records")

    image_base_dir = Path(args.image_base_dir)
    dataset = E2CTrainingDataset(
        records=records,
        processor=processor,
        adapter=adapter,
        image_loader=load_image,
        image_base_dir=image_base_dir,
    )

    batch_size = train_cfg["batch_size"]
    grad_accum = train_cfg["gradient_accumulation_steps"]
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=adapter.collate,
        num_workers=0,
    )

    # ------------------------------------------------------------------ #
    # Optimizer
    # ------------------------------------------------------------------ #
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    lr = train_cfg["learning_rate"]
    warmup_steps = train_cfg["warmup_steps"]
    max_grad_norm = train_cfg["max_grad_norm"]

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=lr,
        weight_decay=train_cfg["weight_decay"],
    )

    if warmup_steps > 0 and optimizer_steps > warmup_steps:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps,
        )
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=optimizer_steps - warmup_steps,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=optimizer_steps)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    logger.info(f"Starting training: {optimizer_steps} optimizer steps")
    model.train()

    training_trace: list[dict[str, Any]] = []
    global_step = 0
    running_loss = 0.0
    running_count = 0

    for epoch in range(1000):  # safety cap
        for batch in dataloader:
            # Move to device
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(args.device)
                elif isinstance(v, list):
                    batch_device[k] = v
                else:
                    batch_device[k] = v

            # Forward pass
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
                labels=batch_device["labels"],
                **{k: v for k, v in batch_device.items()
                   if k not in ("input_ids", "attention_mask", "labels")
                   and isinstance(v, torch.Tensor)
                   and not k.startswith("_")},
            )
            loss = outputs.loss

            # Backward
            loss = loss / grad_accum
            loss.backward()

            running_loss += loss.item() * grad_accum
            running_count += 1

            if (running_count) % grad_accum == 0:
                # Gradient clipping (returns pre-clip norm for logging)
                grad_norm: float | None = None
                if max_grad_norm > 0:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        trainable_params, max_grad_norm,
                    ))
                else:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        trainable_params, float("inf"),
                    ))

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log
                if global_step % 10 == 0 or global_step <= 5:
                    avg_loss = running_loss / running_count
                    trace_entry = {
                        "step": global_step,
                        "loss": avg_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                        "grad_norm": grad_norm if grad_norm is not None
                        and torch.isfinite(torch.tensor(grad_norm)) else None,
                        "epoch": epoch,
                    }
                    training_trace.append(trace_entry)
                    logger.info(
                        f"Step {global_step}/{optimizer_steps} "
                        f"loss={avg_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
                    )

                # Checkpoint
                if global_step % args.checkpoint_steps == 0:
                    ckpt_dir = output_dir / "checkpoints" / f"step_{global_step}"
                    adapter.save_unlearning_adapter(model, ckpt_dir)
                    logger.info(f"Checkpoint saved at step {global_step}")

                if global_step >= optimizer_steps:
                    break

            if global_step >= optimizer_steps:
                break

        if global_step >= optimizer_steps:
            break

    # ------------------------------------------------------------------ #
    # Save final adapter
    # ------------------------------------------------------------------ #
    logger.info("Saving final adapter...")
    final_dir = output_dir / "adapter_final"
    adapter_meta = adapter.save_unlearning_adapter(model, final_dir)

    # Verify protected parameters
    protected_report = adapter.verify_protected_parameters(
        protected_snapshot, model,
    )
    logger.info(
        f"Protected params: {'PASS' if protected_report['pass'] else 'FAIL'} "
        f"({protected_report['n_changed']}/{protected_report['n_total']} changed)"
    )

    # ------------------------------------------------------------------ #
    # Write training artifacts
    # ------------------------------------------------------------------ #
    # Training trace
    with open(output_dir / "training_trace.jsonl", "w") as f:
        f.writelines(json.dumps(entry) + "\n" for entry in training_trace)

    # Training summary
    summary = {
        "condition": condition,
        "seed": seed,
        "total_steps": global_step,
        "final_loss": running_loss / max(running_count, 1),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "protected_parameters": protected_report,
        "adapter_metadata": adapter_meta,
        "smoke": args.smoke,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # Parameter inventory
    param_inventory = {
        "trainable": [],
        "protected_count": protected_report["n_total"],
    }
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_inventory["trainable"].append({
                "name": name,
                "shape": list(param.shape),
                "dtype": str(param.dtype),
            })
    with open(output_dir / "parameter_inventory.json", "w") as f:
        json.dump(param_inventory, f, indent=2, sort_keys=True)

    # Parameter change report
    change_report = {
        "lora_tensors_changed": 0,
        "lora_tensors_total": 0,
    }
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            change_report["lora_tensors_total"] += 1
            if param.abs().sum().item() > 0:
                change_report["lora_tensors_changed"] += 1
    with open(output_dir / "parameter_change_report.json", "w") as f:
        json.dump(change_report, f, indent=2, sort_keys=True)

    # ------------------------------------------------------------------ #
    # Provenance
    # ------------------------------------------------------------------ #
    logger.info("Writing provenance...")
    manifest_dir = Path(args.manifest_dir)
    manifest_shas = {}
    for fpath in manifest_dir.glob("*.json"):
        manifest_shas[fpath.stem] = sha256_file(fpath)

    dataset_shas = {}
    for fname in ["M_train.jsonl", "D_train.jsonl", "M_shuffled_train.jsonl"]:
        fpath = dataset_dir / fname
        if fpath.exists():
            dataset_shas[fname] = sha256_file(fpath)

    git_state = get_git_state(_project_root)
    prov = build_provenance(
        git_state=git_state,
        model_fingerprint=capture_model_fingerprint(
            model_id=model_cfg["id"],
            revision=model_cfg["revision"],
            processor_id=model_cfg["processor_id"],
            processor_revision=model_cfg["processor_revision"],
            dtype=model_cfg["dtype"],
            attn_implementation=model_cfg["attn_implementation"],
            trust_remote_code=model_cfg["trust_remote_code"],
        ),
        processor_fingerprint=capture_processor_fingerprint(processor),
        lora_config=capture_lora_config(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            scope_regex=lora_cfg["scope_regex"],
        ),
        training_config=capture_training_config(
            learning_rate=train_cfg["learning_rate"],
            optimizer_steps=optimizer_steps,
            warmup_steps=train_cfg["warmup_steps"],
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            max_grad_norm=max_grad_norm,
            weight_decay=train_cfg["weight_decay"],
            seed=seed,
            condition=condition,
        ),
        manifest_shas=manifest_shas,
        dataset_shas=dataset_shas,
        adapter_shas={"final": adapter_meta.get("checkpoint_sha256", "")},
    )
    write_provenance(prov, output_dir)

    logger.info(f"Training complete. Output: {output_dir}")
    logger.info(f"  Steps: {global_step}")
    logger.info(f"  Final loss: {summary['final_loss']:.4f}")


if __name__ == "__main__":
    main()
