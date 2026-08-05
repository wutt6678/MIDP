"""Fine-tune one condition/seed LoRA adapter (PLAN.md section 11.3).

Small custom training loop (Accelerate + PEFT), one condition, one seed.
Trains from the same immutable base checkpoint for every condition, freezes
the base model and vision encoder, injects LoRA, masks prompt tokens in the
labels, and saves adapter + config snapshot + losses + exposure counts.

Usage (from repo root):
    python experiments/route_train.py --config configs/route_direct.yaml
    python experiments/route_train.py --config configs/route_mediated.yaml --seed 1
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from route.celeba import load_manifest  # noqa: E402
from vlm_spatial.model import _MODEL_CLASS_MAP  # noqa: E402
from vlm_spatial.route_dataset import RouteCollator, RouteDataset  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_train_model(model_cfg):
    """Load the base model for LoRA training (bf16, eager attention)."""
    import importlib

    from transformers import AutoConfig, AutoProcessor

    name = model_cfg["name"]
    print(f"Loading base model {name} ...")
    config = AutoConfig.from_pretrained(name, trust_remote_code=True)
    arch = config.architectures[0] if config.architectures else None
    class_path = _MODEL_CLASS_MAP.get(arch)
    if class_path is None:
        raise ValueError(f"Unsupported model architecture: {arch}")

    module_name, class_name = class_path.rsplit(".", 1)
    model_cls = getattr(importlib.import_module(module_name), class_name)

    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    model = model_cls.from_pretrained(
        name,
        dtype=dtype,
        attn_implementation="eager",
    )

    # Freeze the base model entirely; LoRA re-enables its target modules.
    model.requires_grad_(False)
    if model_cfg.get("freeze_vision_encoder", True):
        n_frozen_visual = 0
        for pname, param in model.named_parameters():
            if "visual" in pname:
                param.requires_grad_(False)
                n_frozen_visual += 1
        print(f"  Froze {n_frozen_visual} vision-encoder parameters")

    if model_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    processor = AutoProcessor.from_pretrained(name)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model, processor


def build_lora_config(lora_cfg):
    from peft import LoraConfig
    return LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def count_exposures(dataset):
    """Exposure counts to verify parity across conditions (PLAN section 6)."""
    per_identity = Counter()
    per_type = Counter()
    per_property = Counter()
    n_text_only = 0
    for ex in dataset.examples:
        row = ex["row"]
        per_identity[str(row["celeba_identity_id"])] += 1
        per_type[ex["example_type"]] += 1
        per_property[row["property"]] += 1
        if ex["variant"] is None:
            n_text_only += 1
    return {
        "n_examples": len(dataset.examples),
        "n_text_only": n_text_only,
        "per_identity": dict(sorted(per_identity.items())),
        "per_example_type": dict(sorted(per_type.items())),
        "per_property": dict(sorted(per_property.items())),
    }


def evaluate_loss(accelerator, model, eval_loader):
    model.eval()
    total, n_batches = 0.0, 0
    for batch in eval_loader:
        with torch.no_grad():
            outputs = model(**batch)
        total += float(outputs.loss.detach())
        n_batches += 1
    model.train()
    return total / max(1, n_batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after this many optimizer steps (debug)")
    parser.add_argument("--output", default=None,
                        help="Override output dir "
                             "(default results/route_mvp/<condition>/"
                             "seed<seed>)")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    condition = cfg["condition"]
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    epochs = args.epochs if args.epochs is not None else cfg["training"]["epochs"]
    set_seed(seed)

    output_dir = Path(args.output) if args.output else (
        Path("results") / "route_mvp" / condition / f"seed{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    manifest_dir = Path(data_cfg["manifest_dir"])
    train_rows = load_manifest(manifest_dir / "train.jsonl")
    val_rows = load_manifest(manifest_dir / "validation.jsonl")

    train_ds = RouteDataset(train_rows, data_cfg["celeba_root"], condition,
                            image_size=data_cfg.get("image_size"), seed=seed)
    val_ds = RouteDataset(val_rows, data_cfg["celeba_root"], condition,
                          image_size=data_cfg.get("image_size"), seed=seed)
    exposures = count_exposures(train_ds)
    print(f"Condition={condition} seed={seed}: "
          f"{len(train_ds)} train / {len(val_ds)} val examples")

    from accelerate import Accelerator
    from peft import get_peft_model
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
    )
    device = accelerator.device

    model, processor = load_train_model(cfg["model"])
    # Same LoRA initialization seed in all conditions (PLAN section 8.3).
    set_seed(seed)
    model = get_peft_model(model, build_lora_config(cfg["lora"]))
    model.print_trainable_parameters()
    model.to(device)

    collator = RouteCollator(processor)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=0, collate_fn=collator, generator=g)
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=0, collate_fn=collator)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    steps_per_epoch = max(
        1, len(train_loader) // train_cfg["gradient_accumulation_steps"])
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(train_cfg.get("warmup_ratio", 0.05) * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, max(1, total_steps))

    (model, optimizer, train_loader, val_loader, scheduler) = (
        accelerator.prepare(model, optimizer, train_loader, val_loader,
                            scheduler))

    # Freeze snapshot of the run configuration.
    with open(output_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump({"config_path": str(args.config), **cfg,
                        "seed_effective": seed, "epochs_effective": epochs},
                       f, sort_keys=False)
    with open(output_dir / "exposure_counts.json", "w") as f:
        json.dump(exposures, f, indent=2)

    eval_every = train_cfg.get("evaluation_every_steps", 20)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    global_step = 0
    train_log, eval_log = [], []
    run_loss = 0.0
    t0 = time.time()
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            run_loss += float(loss.detach())
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.log_every == 0:
                    avg = run_loss / args.log_every
                    train_log.append({"step": global_step,
                                      "loss": round(avg, 4),
                                      "elapsed_s": round(time.time() - t0, 1)})
                    print(f"  step {global_step}/{total_steps}  "
                          f"loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
                    run_loss = 0.0
                if global_step % eval_every == 0:
                    val_loss = evaluate_loss(accelerator, model, val_loader)
                    eval_log.append({"step": global_step,
                                     "loss": round(val_loss, 4)})
                    print(f"  [eval] step {global_step}  val_loss={val_loss:.4f}")
                if args.max_steps and global_step >= args.max_steps:
                    break
        if args.max_steps and global_step >= args.max_steps:
            break

    final_val_loss = evaluate_loss(accelerator, model, val_loader)
    eval_log.append({"step": global_step, "loss": round(final_val_loss, 4),
                     "final": True})
    with open(output_dir / "losses.json", "w") as f:
        json.dump({"train": train_log, "eval": eval_log,
                   "total_steps": global_step}, f, indent=2)

    adapter_dir = output_dir / "adapter"
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(adapter_dir)
    print(f"\nSaved adapter to {adapter_dir}")
    print(f"Final val loss: {final_val_loss:.4f} "
          f"({global_step} optimizer steps, "
          f"{time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
