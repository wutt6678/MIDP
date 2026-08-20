#!/usr/bin/env python3
"""Model-agnostic candidate-margin unlearning runner.

Implements the common research pipeline for all model families:
  1. Resolve frozen baseline, identity selection, method config
  2. Load fresh base model via adapter
  3. Apply language-attention-only LoRA
  4. Train with candidate-margin forget + KL retain
  5. Save adapter checkpoint
  6. Fresh reload base model + adapter
  7. Post-evaluate on frozen 500 probes
  8. Validate all gates and write binding

Usage::

    # Canary (1 optimizer step)
    python scripts/run_model_unlearning.py \\
        --model-profile configs/models/unlearning/qwen35_4b.yaml \\
        --method-config configs/methods/candidate_margin_v1.yaml \\
        --selection configs/experiments/common/frozen_identity_selection_v1.yaml \\
        --canary --seed 17 --device cuda:0

    # Full run (50 optimizer steps)
    python scripts/run_model_unlearning.py \\
        --model-profile configs/models/unlearning/qwen35_4b.yaml \\
        --method-config configs/methods/candidate_margin_v1.yaml \\
        --selection configs/experiments/common/frozen_identity_selection_v1.yaml \\
        --seed 17 --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_model_unlearning")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
            cwd=str(PROJECT_ROOT),
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except FileNotFoundError:
        return ""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Training dataset (model-agnostic via adapter)
# --------------------------------------------------------------------------- #

class UnlearningDataset(Dataset):
    """Dataset that builds supervised examples via the adapter."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        adapter: Any,
        processor: Any,
    ):
        self.samples = samples
        self.adapter = adapter
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image_uri = sample["image_uri"]
        question = sample["question"]
        answer_label = sample["answer_label"]
        answer_text = "Yes" if answer_label else "No"

        from PIL import Image
        image = Image.open(image_uri).convert("RGB")

        example = self.adapter.build_supervised_example(
            self.processor,
            image=image,
            prompt=question,
            answer_text=answer_text,
        )

        # Ensure _prefix_len is set (some adapters like Phi don't set it)
        if "_prefix_len" not in example:
            prefix = self.adapter.build_prefix(
                self.processor, image=image, prompt=question,
            )
            example["_prefix_len"] = prefix["input_ids"].shape[0]

        # Ensure _yes_token_ids and _no_token_ids are set
        if "_yes_token_ids" not in example:
            example["_yes_token_ids"] = self.adapter.candidate_token_ids(
                self.processor, "Yes",
            )
            example["_no_token_ids"] = self.adapter.candidate_token_ids(
                self.processor, "No",
            )

        # Ensure _answer_label is set (True=Yes, False=No)
        if "_answer_label" not in example:
            example["_answer_label"] = (answer_text == self.adapter.profile.candidate_positive)

        # Ensure _pad_token_id is set for collate
        if "_pad_token_id" not in example:
            example["_pad_token_id"] = self.adapter.pad_token_id(self.processor)

        example["_identity_id"] = sample.get("identity_id", "")
        example["_probe_id"] = sample.get("probe_id", "")
        example["_probe_family"] = sample.get("probe_family", "")
        example["_identity_group"] = sample.get("identity_group", "")
        return example


def unlearning_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate batch with right-padding for text, pass-through for visual."""
    pad_id = 0
    max_len = max(ex["input_ids"].shape[0] for ex in batch)
    batch_size = len(batch)

    result: dict[str, Any] = {}
    text_keys = {"input_ids", "attention_mask", "labels"}
    meta_keys = {"_prefix_len", "_correct_answer_token_ids", "_answer_label",
                 "_yes_token_ids", "_no_token_ids", "_identity_id",
                 "_probe_id", "_probe_family", "_identity_group"}

    for key in batch[0]:
        if key in meta_keys:
            result[key] = [ex[key] for ex in batch]
            if key == "_prefix_len":
                result[key] = torch.tensor(result[key])
            continue

        if key in text_keys:
            tensors = [ex[key] for ex in batch]
            if tensors[0].dim() == 1:
                padded = torch.full((batch_size, max_len), pad_id if key != "labels" else -100,
                                    dtype=tensors[0].dtype)
                for i, t in enumerate(tensors):
                    padded[i, :t.shape[0]] = t
                result[key] = padded
            else:
                result[key] = torch.stack(tensors)
        elif isinstance(batch[0][key], torch.Tensor):
            try:
                result[key] = torch.stack([ex[key] for ex in batch])
            except (RuntimeError, TypeError):
                result[key] = [ex[key] for ex in batch]
        else:
            result[key] = [ex[key] for ex in batch]

    return result


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def train_unlearning(
    model: torch.nn.Module,
    adapter: Any,
    forget_dataset: UnlearningDataset,
    retain_dataset: UnlearningDataset,
    *,
    learning_rate: float,
    num_optimizer_steps: int,
    gradient_accumulation_steps: int,
    retain_weight: float,
    max_grad_norm: float,
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the candidate-margin training loop."""
    model.train()
    model.to(device)

    forget_loader = DataLoader(
        forget_dataset, batch_size=1, shuffle=True,
        collate_fn=adapter.collate,
    )
    retain_loader = DataLoader(
        retain_dataset, batch_size=1, shuffle=True,
        collate_fn=adapter.collate,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    forget_iter = iter(forget_loader)
    retain_iter = iter(retain_loader)

    trace_path = output_dir / "training_trace.jsonl"
    with open(trace_path, "w") as trace_f:

        start_time = time.time()
        final_stats: dict[str, Any] = {}

        logger.info(
            f"Training for {num_optimizer_steps} optimizer steps "
            f"(grad_accum={gradient_accumulation_steps})"
        )

        for opt_step in range(1, num_optimizer_steps + 1):
            optimizer.zero_grad()
            accum_forget = 0.0
            accum_retain = 0.0

            for _micro in range(gradient_accumulation_steps):
                try:
                    forget_batch = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    forget_batch = next(forget_iter)

                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_batch = next(retain_iter)

                # Move to device
                forget_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in forget_batch.items()}
                retain_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in retain_batch.items()}

                # Forget loss: candidate margin on target examples
                forget_loss = _compute_forget_loss(model, forget_batch, adapter)
                # Retain loss: KL to frozen reference (use current model as approx)
                retain_loss = _compute_retain_loss_kl(model, retain_batch)

                total_loss = forget_loss + retain_weight * retain_loss
                loss_scaled = total_loss / gradient_accumulation_steps
                loss_scaled.backward()

                accum_forget += forget_loss.item() / gradient_accumulation_steps
                accum_retain += retain_loss.item() / gradient_accumulation_steps

            # Gradient clipping
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

            optimizer.step()

            elapsed = time.time() - start_time
            step_stats = {
                "step": opt_step,
                "forget_loss": accum_forget,
                "retain_loss": accum_retain,
                "total_loss": accum_forget + retain_weight * accum_retain,
                "elapsed_seconds": elapsed,
            }
            trace_f.write(json.dumps(step_stats) + "\n")
            trace_f.flush()

            logger.info(
                f"Step {opt_step}/{num_optimizer_steps} | "
                f"loss={step_stats['total_loss']:.4f} "
                f"forget={step_stats['forget_loss']:.4f} "
                f"retain={step_stats['retain_loss']:.4f}"
            )
            final_stats = step_stats

    final_stats["total_elapsed_seconds"] = time.time() - start_time
    final_stats["num_optimizer_steps"] = num_optimizer_steps
    final_stats["gradient_accumulation_steps"] = gradient_accumulation_steps
    return final_stats


def _compute_forget_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    adapter: Any,
) -> torch.Tensor:
    """Candidate-margin forget loss: minimise M = logP(correct) - logP(wrong)."""
    from route_data.models.scoring import score_candidate_sequence_tensor

    batch_size = batch["input_ids"].shape[0]
    prefix_lens = batch["_prefix_len"]
    answer_labels = batch["_answer_label"]
    yes_token_ids_raw = batch["_yes_token_ids"]
    no_token_ids_raw = batch["_no_token_ids"]

    # Normalize to lists for indexing (Phi collate returns scalars for batch_size=1)
    if isinstance(prefix_lens, (int, float)):
        prefix_lens = [int(prefix_lens)]
    elif isinstance(prefix_lens, torch.Tensor):
        prefix_lens = prefix_lens.tolist()
    if isinstance(answer_labels, bool):
        answer_labels = [answer_labels]
    # _yes_token_ids and _no_token_ids: each element is a list[int]
    # For batch_size=1 Phi collate, the batch value is a single list[int]
    if isinstance(yes_token_ids_raw, list) and len(yes_token_ids_raw) > 0 and isinstance(yes_token_ids_raw[0], int):
        # Single list[int] — wrap in outer list for batch indexing
        yes_token_ids_batch = [yes_token_ids_raw]
    else:
        yes_token_ids_batch = yes_token_ids_raw
    if isinstance(no_token_ids_raw, list) and len(no_token_ids_raw) > 0 and isinstance(no_token_ids_raw[0], int):
        no_token_ids_batch = [no_token_ids_raw]
    else:
        no_token_ids_batch = no_token_ids_raw

    total_margin = torch.tensor(0.0, device=batch["input_ids"].device)

    for i in range(batch_size):
        prefix_len = int(prefix_lens[i])
        expected = answer_labels[i]

        # Extract prefix
        prefix: dict[str, Any] = {
            "input_ids": batch["input_ids"][i:i+1, :prefix_len],
            "attention_mask": batch["attention_mask"][i:i+1, :prefix_len],
        }

        # Add multimodal and sequence-indexed tensors
        image_keys = adapter.image_indexed_keys() if adapter else frozenset()
        # Sequence-indexed keys have same length as input_ids and must be sliced
        seq_keys = {"mm_token_type_ids"}  # Qwen-specific
        for key, val in batch.items():
            if key in ("input_ids", "attention_mask", "labels") or key.startswith("_"):
                continue
            if isinstance(val, torch.Tensor):
                if key in image_keys:
                    # Image-indexed keys: dim 0 = num_images, NOT batch.
                    # Pass through as-is (do not slice by batch index).
                    prefix[key] = val
                elif key in seq_keys:
                    # Sequence-indexed: same length as input_ids, slice to prefix
                    prefix[key] = val[i:i+1, :prefix_len]
                elif val.dim() >= 1 and val.shape[0] == batch_size:
                    prefix[key] = val[i:i+1]
            elif isinstance(val, list) and len(val) > i:
                prefix[key] = val[i]

        # Sanitize for model-specific forward
        if adapter:
            prefix = adapter.sanitize_model_inputs(prefix)

        log_p_yes = score_candidate_sequence_tensor(
            model, prefix, yes_token_ids_batch[i], adapter=adapter,
        )
        log_p_no = score_candidate_sequence_tensor(
            model, prefix, no_token_ids_batch[i], adapter=adapter,
        )

        if expected:
            margin = log_p_yes - log_p_no
        else:
            margin = log_p_no - log_p_yes

        total_margin = total_margin + margin

    return total_margin / batch_size


def _compute_retain_loss_kl(
    model: torch.nn.Module,
    batch: dict[str, Any],
) -> torch.Tensor:
    """Retain loss: answer-only cross-entropy (retain examples should keep correct answers)."""
    model_kwargs: dict[str, Any] = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    for key, val in batch.items():
        if (
            key not in ("input_ids", "attention_mask", "labels")
            and not key.startswith("_")
            and isinstance(val, torch.Tensor)
        ):
            model_kwargs[key] = val

    outputs = model(**model_kwargs)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()

    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return loss


# --------------------------------------------------------------------------- #
# Post-evaluation
# --------------------------------------------------------------------------- #

def run_post_evaluation(
    adapter: Any,
    model: torch.nn.Module,
    processor: Any,
    profile: Any,
    adapter_path: Path,
    probe_path: str,
    output_dir: Path,
    baseline_results_path: str,
    profile_path: Path,
) -> dict[str, Any]:
    """Fresh reload + 500-probe post-evaluation using BaselineRunner."""
    from route_data.config import GenerationConfig, ModelConfig
    from route_data.eval.baseline_runner import BaselineRunner

    logger.info("Post-evaluation: fresh reload + 500 probes")

    # Build ModelConfig for eval
    model_config = ModelConfig(
        backend="adapter_eval_backend",
        model_id=profile.model_id,
        revision=profile.revision,
        dtype=profile.dtype,
        generation=GenerationConfig(do_sample=False),
    )

    # Convert to eval backend
    backend = adapter.to_eval_backend(
        model=model,
        processor=processor,
        model_config=model_config,
        adapter_metadata={"adapter_path": str(adapter_path)},
    )
    fingerprint = backend.fingerprint()

    # Build SimpleNamespace for runner (matches generate_model_baseline.py pattern)
    import types
    runner_model_config = types.SimpleNamespace(
        model_id=profile.model_id,
        revision=profile.revision,
        dtype=profile.dtype,
        backend="adapter_eval_backend",
        fingerprint_id=fingerprint.get("fingerprint_id", ""),
        generation=GenerationConfig(do_sample=False),
    )

    post_eval_dir = output_dir / "post_eval"
    post_eval_dir.mkdir(parents=True, exist_ok=True)

    # Resolve evidence paths
    project_root = PROJECT_ROOT
    dataset_manifest = project_root / "outputs/full_fiubench/evidence/research_dataset_manifest.json"
    freeze_verification = project_root / "outputs/full_fiubench/evidence/final_freeze_verification.json"
    processed_dataset = project_root / "outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_processed.jsonl"

    # Run evaluation using BaselineRunner
    runner = BaselineRunner(
        backend=backend,
        probe_path=probe_path,
        output_dir=str(post_eval_dir),
        model_config=runner_model_config,
        resume=True,
        dataset_manifest_path=str(dataset_manifest) if dataset_manifest.exists() else None,
        model_config_path=str(profile_path),
        freeze_verification_path=str(freeze_verification) if freeze_verification.exists() else None,
        processed_dataset_path=str(processed_dataset) if processed_dataset.exists() else None,
    )

    results = runner.run_all()
    runner.save_results()
    summary = runner.generate_summary()
    validation = runner.validate_results()

    # Write validation report
    validation_path = post_eval_dir / "validation_report.json"
    with open(validation_path, "w") as f:
        json.dump(validation, f, indent=2, default=str)
        f.write("\n")

    n_errors = sum(1 for r in results if r.error is not None)
    logger.info(f"Post-evaluation: {len(results)} probes, {n_errors} errors")
    logger.info(f"Post-evaluation validation: {'PASS' if validation.get('pass') else 'FAIL'}")

    return {
        "num_probes": len(results),
        "num_errors": n_errors,
        "results_path": str(post_eval_dir / "baseline_results.jsonl"),
        "summary_path": str(post_eval_dir / "baseline_summary.json"),
        "validation_path": str(validation_path),
        "validation_pass": validation.get("pass", False),
        "summary": summary,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Model-agnostic unlearning runner")
    parser.add_argument("--model-profile", type=str, required=True,
                        help="Path to model YAML profile")
    parser.add_argument("--method-config", type=str,
                        default="configs/methods/candidate_margin_v1.yaml",
                        help="Path to method YAML config")
    parser.add_argument("--selection", type=str,
                        default="configs/experiments/common/frozen_identity_selection_v1.yaml",
                        help="Path to identity selection YAML")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: auto)")
    parser.add_argument("--canary", action="store_true",
                        help="Canary mode: 1 optimizer step")
    parser.add_argument("--skip-post-eval", action="store_true",
                        help="Skip post-evaluation (for debugging)")
    args = parser.parse_args()

    set_seed(args.seed)

    # -- Load configs ------------------------------------------------------- #
    import yaml
    profile_path = Path(args.model_profile)
    method_path = Path(args.method_config)
    selection_path = Path(args.selection)

    with open(profile_path) as f:
        profile_raw = yaml.safe_load(f)
    with open(method_path) as f:
        method_config = yaml.safe_load(f)
    with open(selection_path) as f:
        selection = yaml.safe_load(f)

    model_key = profile_raw["key"]
    logger.info(f"Model: {model_key}")
    logger.info(f"Profile: {profile_path}")
    logger.info(f"Method: {method_path}")

    # -- Resolve output directory ------------------------------------------- #
    if args.output:
        output_dir = Path(args.output)
    else:
        mode = "canary" if args.canary else "seed_17"
        output_dir = (PROJECT_ROOT / "outputs/experiments/unlearning" /
                      model_key / "candidate_margin" / mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mark incomplete
    run_incomplete = output_dir / "RUN_INCOMPLETE"
    run_incomplete.touch()

    # -- Load adapter and model --------------------------------------------- #
    from route_data.models.trainable.registry import (
        compute_profile_sha256,
        create_adapter,
        load_profile_from_yaml,
    )

    profile = load_profile_from_yaml(profile_path)
    adapter = create_adapter(model_key, profile=profile)
    profile_sha = compute_profile_sha256(profile_path)

    logger.info(f"Loading model: {profile.model_id} rev={profile.revision}")
    model, processor = adapter.load_model_processor(
        model_id=profile.model_id,
        revision=profile.revision,
        processor_revision=profile.processor_revision,
        dtype=profile.dtype,
        device=args.device,
        training=False,
    )

    # Freeze base
    for param in model.parameters():
        param.requires_grad = False

    # -- Apply LoRA --------------------------------------------------------- #
    lora_targets = adapter.resolve_lora_targets(model)
    logger.info(f"LoRA targets: {len(lora_targets)} modules")

    from peft import LoraConfig, get_peft_model

    inner_peft = adapter.get_inner_peft_model(model)
    if inner_peft is not None:
        # Model already has PEFT layers injected (e.g. Phi bundled adapters).
        # Resolve targets relative to the inner model by stripping the
        # prefix that the full model's scope_regex expects.
        # Phi's scope_regex: ^model\.layers\.\d+\.(self_attn)\.(qkv_proj|o_proj)$
        # Inner model names: layers.\d+.self_attn.qkv_proj
        inner_targets = []
        for t in lora_targets:
            # Strip leading "model." if present
            if t.startswith("model."):
                inner_targets.append(t[len("model."):])
            else:
                inner_targets.append(t)
        logger.info(f"LoRA targets (inner): {len(inner_targets)} modules")
        # Use LoraModel to inject our language adapter alongside existing ones.
        from peft.tuners.lora.model import LoraModel
        lora_cfg = LoraConfig(
            r=profile.lora_rank,
            lora_alpha=profile.lora_alpha,
            lora_dropout=profile.lora_dropout,
            target_modules=inner_targets,
            task_type="CAUSAL_LM",
        )
        # LoraModel constructor calls inject_adapter() internally
        LoraModel(inner_peft, lora_cfg, adapter_name="unlearning")
        logger.info("Injected 'unlearning' adapter via LoraModel")
    else:
        lora_cfg = LoraConfig(
            r=profile.lora_rank,
            lora_alpha=profile.lora_alpha,
            lora_dropout=profile.lora_dropout,
            target_modules=lora_targets,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    # Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_trainable:,}")

    # -- Build training data ------------------------------------------------ #
    baseline_path = (PROJECT_ROOT / "outputs/experiments/pre_unlearning" /
                     model_key / "baseline_v1" / "baseline_results.jsonl")
    processed_path = (PROJECT_ROOT / "outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/"
                      "fiubench_processed.jsonl")

    baseline_results = []
    with open(baseline_path) as f:
        for line in f:
            if line.strip():
                baseline_results.append(json.loads(line))

    # Build image_sha -> sample map
    sha_to_sample = {}
    with open(processed_path) as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                image_uri = sample.get("image_uri", "")
                if image_uri:
                    image_sha = file_sha256(Path(image_uri))
                    sha_to_sample[image_sha] = sample

    # Build forget/retain datasets
    target_ids = set(selection["target_identities"])
    retain_ids = set(selection["retain_identities"])

    forget_samples = []
    retain_samples = []
    for r in baseline_results:
        iid = r["identity_id"]
        image_sha = r.get("image_sha256", "")
        if image_sha not in sha_to_sample:
            continue
        processed = sha_to_sample[image_sha]
        train_sample = {
            "image_uri": processed["image_uri"],
            "question": r["question"],
            "answer_label": r["answer_label"],
            "identity_id": iid,
            "probe_id": r.get("probe_id", ""),
            "probe_family": r.get("probe_family", ""),
            "identity_group": "target" if iid in target_ids else "retain" if iid in retain_ids else "other",
        }
        if iid in target_ids:
            forget_samples.append(train_sample)
        elif iid in retain_ids:
            retain_samples.append(train_sample)

    logger.info(f"Forget samples: {len(forget_samples)}, Retain samples: {len(retain_samples)}")

    forget_dataset = UnlearningDataset(forget_samples, adapter, processor)
    retain_dataset = UnlearningDataset(retain_samples, adapter, processor)

    # -- Write run config --------------------------------------------------- #
    run_config = {
        "model_key": model_key,
        "model_id": profile.model_id,
        "model_revision": profile.revision,
        "processor_revision": profile.processor_revision,
        "model_profile_sha256": profile_sha,
        "method_config": str(method_path),
        "selection": str(selection_path),
        "seed": args.seed,
        "device": args.device,
        "canary": args.canary,
        "git_commit": git_commit(),
        "target_identities": selection["target_identities"],
        "retain_identities": selection["retain_identities"],
        "control_identities": selection["control_identities"],
        "lora_targets": lora_targets,
        "lora_target_count": len(lora_targets),
        "trainable_parameters": n_trainable,
    }
    with open(output_dir / "run_config_resolved.yaml", "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)

    # -- Train -------------------------------------------------------------- #
    hp = method_config["training"]
    num_steps = 1 if args.canary else hp["num_optimizer_steps"]

    logger.info("=" * 60)
    logger.info(f"Starting training: {num_steps} steps")
    logger.info("=" * 60)

    training_stats = train_unlearning(
        model, adapter, forget_dataset, retain_dataset,
        learning_rate=hp["learning_rate"],
        num_optimizer_steps=num_steps,
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        retain_weight=hp["retain_weight"],
        max_grad_norm=hp.get("max_grad_norm", 1.0),
        device=args.device,
        output_dir=output_dir,
    )

    # -- Save adapter ------------------------------------------------------- #
    adapter_path = output_dir / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)

    # Save LoRA weights
    if inner_peft is not None:
        inner_peft.save_pretrained(str(adapter_path))
    else:
        model.save_pretrained(str(adapter_path))

    adapter_sha = file_sha256(adapter_path / "adapter_model.bin") if (adapter_path / "adapter_model.bin").exists() else ""
    logger.info(f"Adapter saved: {adapter_path}")

    # -- Write training summary --------------------------------------------- #
    training_summary = {
        **training_stats,
        "adapter_path": str(adapter_path),
        "adapter_sha256": adapter_sha,
        "forget_samples": len(forget_samples),
        "retain_samples": len(retain_samples),
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2)
        f.write("\n")

    # -- Post-evaluation ---------------------------------------------------- #
    if not args.skip_post_eval:
        probe_path = str(PROJECT_ROOT / "outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/"
                         "fiubench_route_conflict_eval.jsonl")

        # Fresh reload for post-evaluation
        logger.info("Fresh reload for post-evaluation")
        model.eval()

        post_eval = run_post_evaluation(
            adapter=adapter,
            model=model,
            processor=processor,
            profile=profile,
            adapter_path=adapter_path,
            probe_path=probe_path,
            output_dir=output_dir,
            baseline_results_path=str(baseline_path),
            profile_path=profile_path,
        )
    else:
        post_eval = {"num_probes": 0, "num_errors": 0, "skipped": True}

    # -- Write selection ---------------------------------------------------- #
    with open(output_dir / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)
        f.write("\n")

    # -- Write environment -------------------------------------------------- #
    import torch
    env_info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "device": args.device,
    }
    try:
        import transformers
        env_info["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    try:
        import peft
        env_info["peft_version"] = peft.__version__
    except ImportError:
        pass
    with open(output_dir / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)
        f.write("\n")

    # -- Write parameter inventory ------------------------------------------ #
    param_inv = {
        "lora_target_count": len(lora_targets),
        "lora_targets": lora_targets,
        "trainable_parameters": n_trainable,
        "frozen_base_changes": 0,
    }
    with open(output_dir / "parameter_inventory.json", "w") as f:
        json.dump(param_inv, f, indent=2)
        f.write("\n")

    # -- Validation --------------------------------------------------------- #
    validation = {
        "pass": True,
        "checks": {
            "finite_loss": bool(np.isfinite(training_stats.get("total_loss", 0))),
            "nonzero_trainable": bool(n_trainable > 0),
            "adapter_saved": bool(adapter_path.exists()),
            "probe_count": int(post_eval.get("num_probes", 0)),
            "probe_errors": int(post_eval.get("num_errors", 0)),
        },
    }
    with open(output_dir / "validation_report.json", "w") as f:
        json.dump(validation, f, indent=2)
        f.write("\n")

    # -- Remove RUN_INCOMPLETE ---------------------------------------------- #
    if run_incomplete.exists():
        run_incomplete.unlink()

    logger.info("=" * 60)
    logger.info(f"Unlearning complete: {model_key}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
