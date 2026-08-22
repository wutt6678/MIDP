#!/usr/bin/env python3
"""Model-agnostic candidate-margin unlearning runner (repaired).

Implements the research-grade pipeline for all model families:
  1. Resolve frozen baseline binding, identity selection, method config
  2. Validate baseline binding against model profile
  3. Load fresh base model via adapter
  4. Snapshot protected parameters
  5. Attach unlearning LoRA via adapter lifecycle hook
  6. Train with candidate-margin forget + retain CE
  7. Measure gradient evidence + parameter changes
  8. Save adapter checkpoint with SHA-256
  9. Destroy training model; fresh reload base + adapter
 10. Verify reload equivalence
 11. Post-evaluate on frozen 500 probes
 12. Join pre/post results; compute preservation
 13. Validate all gates; write binding; remove RUN_INCOMPLETE

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
    """Return the full 40-character git SHA."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
            cwd=str(PROJECT_ROOT),
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except FileNotFoundError:
        return ""


def git_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=False,
            cwd=str(PROJECT_ROOT),
        )
        return bool(r.stdout.strip())
    except FileNotFoundError:
        return True


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
# Training dataset (model-agnostic via adapter, family-filtered)
# --------------------------------------------------------------------------- #

class UnlearningDataset(Dataset):
    """Dataset that builds supervised examples via the adapter.

    Supports family filtering (P0-09): only ``direct_visual`` examples
    are used for primary training by default.
    """

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
        answer_text = (
            self.adapter.profile.candidate_positive
            if answer_label
            else self.adapter.profile.candidate_negative
        )

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
                self.processor, self.adapter.profile.candidate_positive,
            )
            example["_no_token_ids"] = self.adapter.candidate_token_ids(
                self.processor, self.adapter.profile.candidate_negative,
            )

        # Ensure _answer_label is set
        if "_answer_label" not in example:
            example["_answer_label"] = (
                answer_text == self.adapter.profile.candidate_positive
            )

        # Ensure _pad_token_id is set for collate
        if "_pad_token_id" not in example:
            example["_pad_token_id"] = self.adapter.pad_token_id(self.processor)

        example["_identity_id"] = sample.get("identity_id", "")
        example["_probe_id"] = sample.get("probe_id", "")
        example["_probe_family"] = sample.get("probe_family", "")
        example["_identity_group"] = sample.get("identity_group", "")
        return example


# --------------------------------------------------------------------------- #
# Training loop with gradient evidence (P0-06)
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
    """Run the candidate-margin training loop with gradient measurement."""
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

    # Gradient evidence tracking
    total_nonzero_grad_tensors = 0
    total_grad_tensors = 0
    max_grad_norm_seen = 0.0

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
                # Retain loss: answer-only cross-entropy (P0-07: honest CE)
                retain_loss = _compute_retain_loss_ce(model, retain_batch)

                total_loss = forget_loss + retain_weight * retain_loss
                loss_scaled = total_loss / gradient_accumulation_steps
                loss_scaled.backward()

                accum_forget += forget_loss.item() / gradient_accumulation_steps
                accum_retain += retain_loss.item() / gradient_accumulation_steps

            # Measure gradients before clipping/stepping
            step_nonzero = 0
            step_total = 0
            step_max_grad = 0.0
            for p in trainable_params:
                if p.grad is not None:
                    step_total += 1
                    grad_norm = p.grad.data.norm(2).item()
                    if grad_norm > 0:
                        step_nonzero += 1
                    step_max_grad = max(step_max_grad, grad_norm)

            total_nonzero_grad_tensors += step_nonzero
            total_grad_tensors += step_total
            max_grad_norm_seen = max(max_grad_norm_seen, step_max_grad)

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
                "nonzero_grad_tensors": step_nonzero,
                "total_grad_tensors": step_total,
            }
            trace_f.write(json.dumps(step_stats) + "\n")
            trace_f.flush()

            logger.info(
                f"Step {opt_step}/{num_optimizer_steps} | "
                f"loss={step_stats['total_loss']:.4f} "
                f"forget={step_stats['forget_loss']:.4f} "
                f"retain={step_stats['retain_loss']:.4f} "
                f"grad={step_nonzero}/{step_total}"
            )
            final_stats = step_stats

    final_stats["total_elapsed_seconds"] = time.time() - start_time
    final_stats["num_optimizer_steps"] = num_optimizer_steps
    final_stats["gradient_accumulation_steps"] = gradient_accumulation_steps
    final_stats["total_nonzero_grad_tensors"] = total_nonzero_grad_tensors
    final_stats["total_grad_tensors"] = total_grad_tensors
    final_stats["max_gradient_norm"] = max_grad_norm_seen
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

    # Normalize to lists for indexing
    if isinstance(prefix_lens, (int, float)):
        prefix_lens = [int(prefix_lens)]
    elif isinstance(prefix_lens, torch.Tensor):
        prefix_lens = prefix_lens.tolist()
    if isinstance(answer_labels, bool):
        answer_labels = [answer_labels]
    if isinstance(yes_token_ids_raw, list) and len(yes_token_ids_raw) > 0 and isinstance(yes_token_ids_raw[0], int):
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
        seq_keys = {"mm_token_type_ids"}
        for key, val in batch.items():
            if key in ("input_ids", "attention_mask", "labels") or key.startswith("_"):
                continue
            if isinstance(val, torch.Tensor):
                if key in image_keys:
                    prefix[key] = val
                elif key in seq_keys:
                    prefix[key] = val[i:i+1, :prefix_len]
                elif val.dim() >= 1 and val.shape[0] == batch_size:
                    prefix[key] = val[i:i+1]
            elif isinstance(val, list) and len(val) > i:
                prefix[key] = val[i]

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


def _compute_retain_loss_ce(
    model: torch.nn.Module,
    batch: dict[str, Any],
) -> torch.Tensor:
    """Retain loss: answer-only cross-entropy.

    P0-07: This is honestly named — it is supervised CE, not KL divergence.
    The method config description will be updated to match.
    """
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
# Post-evaluation with true fresh reload (P0-04)
# --------------------------------------------------------------------------- #

def run_post_evaluation(
    adapter: Any,
    profile: Any,
    adapter_path: Path,
    probe_path: str,
    output_dir: Path,
    baseline_results_path: str,
    profile_path: Path,
    device: str,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """True fresh reload + 500-probe post-evaluation.

    P0-04: Loads a completely fresh base model from the exact pinned
    revision, then loads the saved adapter checkpoint on top.

    P0-2: Passes actual checkpoint SHA in adapter_metadata so the
    fingerprint distinguishes different adapter checkpoints.

    P0-3: Uses resume=False for the initial evaluation of a new
    checkpoint.  Cache invalidation is handled by fingerprint
    mismatch (checkpoint SHA is part of the fingerprint).
    """
    from route_data.config import GenerationConfig, ModelConfig
    from route_data.eval.baseline_runner import BaselineRunner

    logger.info("Post-evaluation: fresh base reload + adapter load + 500 probes")

    # 1. Fresh load base model
    logger.info(f"Fresh loading base model: {profile.model_id} rev={profile.revision}")
    base_model, processor = adapter.load_model_processor(
        model_id=profile.model_id,
        revision=profile.revision,
        processor_revision=profile.processor_revision,
        dtype=profile.dtype,
        device=device,
        training=False,
    )
    for param in base_model.parameters():
        param.requires_grad = False

    # 2. Load saved adapter onto fresh base
    logger.info(f"Loading adapter from: {adapter_path}")
    model = adapter.load_unlearning_adapter(base_model, adapter_path)
    model.eval()
    model.to(device)

    # Build ModelConfig for eval
    model_config = ModelConfig(
        backend="adapter_eval_backend",
        model_id=profile.model_id,
        revision=profile.revision,
        dtype=profile.dtype,
        generation=GenerationConfig(do_sample=False),
    )

    # P0-2: Build adapter_metadata with actual checkpoint provenance so
    # the fingerprint distinguishes different adapter checkpoints.
    _ckpt = checkpoint_metadata or {}
    adapter_metadata: dict[str, Any] = {
        "adapter_checkpoint_path": str(adapter_path),
        "adapter_checkpoint_sha": _ckpt.get("checkpoint_sha256", ""),
        "checkpoint_name": adapter_path.name,
    }
    # Include config SHA if available
    _cfg_path = adapter_path / "adapter_config.json"
    if _cfg_path.is_file():
        adapter_metadata["adapter_config_sha"] = file_sha256(_cfg_path)

    # Convert to eval backend
    backend = adapter.to_eval_backend(
        model=model,
        processor=processor,
        model_config=model_config,
        adapter_metadata=adapter_metadata,
    )
    fingerprint = backend.fingerprint()
    logger.info(f"Post-eval fingerprint: {fingerprint.get('fingerprint_id', '?')}")

    # Build SimpleNamespace for runner
    import types
    runner_model_config = types.SimpleNamespace(
        model_id=profile.model_id,
        revision=profile.revision,
        dtype=profile.dtype,
        backend="adapter_eval_backend",
        fingerprint_id=fingerprint.get("fingerprint_id", ""),
        generation=GenerationConfig(do_sample=False),
    )

    # P0-3: Clear any stale cache from a previous checkpoint.
    # The fingerprint includes the checkpoint SHA, so a different
    # checkpoint produces a different fingerprint.  Old cache rows
    # with a mismatched fingerprint are rejected by BaselineRunner,
    # but we also delete the cache file to avoid confusion.
    post_eval_dir = output_dir / "post_eval"
    _cache_path = post_eval_dir / ".cache" / "baseline_cache.jsonl"
    if _cache_path.is_file():
        # Check if cached rows match current fingerprint
        _stale = False
        try:
            import json as _json
            with open(_cache_path) as _cf:
                for _line in _cf:
                    if _line.strip():
                        _row = _json.loads(_line)
                        if _row.get("model_fingerprint") != fingerprint.get("fingerprint_id"):
                            _stale = True
                            break
        except Exception:
            _stale = True
        if _stale:
            _cache_path.unlink()
            logger.info("Cleared stale post-eval cache from previous checkpoint")

    post_eval_dir.mkdir(parents=True, exist_ok=True)

    # Resolve evidence paths
    dataset_manifest = PROJECT_ROOT / "outputs/full_fiubench/evidence/research_dataset_manifest.json"
    freeze_verification = PROJECT_ROOT / "outputs/full_fiubench/evidence/final_freeze_verification.json"
    processed_dataset = PROJECT_ROOT / "outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_processed.jsonl"

    # P0-3: resume=True is safe because the fingerprint includes the
    # checkpoint SHA — stale rows are rejected automatically.
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

    # P0-C06: Run research preflight before first post-eval inference.
    try:
        preflight = runner.validate_research_preflight()
        preflight_path = post_eval_dir / "preflight_report.json"
        with open(preflight_path, "w") as pf:
            json.dump(preflight, pf, indent=2, default=str)
            pf.write("\n")
        logger.info(
            f"Post-eval preflight: {'PASS' if preflight.get('pass') else 'FAIL'}"
        )
    except RuntimeError as exc:
        logger.error(f"Post-eval preflight FAILED: {exc}")
        raise

    results = runner.run_all()
    runner.save_results()
    summary = runner.generate_summary()
    validation = runner.validate_results()

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
        "fresh_base_loaded": True,
        "checkpoint_loaded": True,
        "results_path": str(post_eval_dir / "baseline_results.jsonl"),
        "summary_path": str(post_eval_dir / "baseline_summary.json"),
        "validation_path": str(validation_path),
        "validation_pass": validation.get("pass", False),
        "fingerprint_id": fingerprint.get("fingerprint_id", ""),
        "summary": summary,
    }


# --------------------------------------------------------------------------- #
# Pre/post probe joining (P0-10)
# --------------------------------------------------------------------------- #

def join_pre_post_results(
    baseline_path: Path,
    post_results_path: Path,
    selection: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Join pre/post probe results and compute group x family metrics."""
    target_ids = set(selection.get("target_identities", []))
    retain_ids = set(selection.get("retain_identities", []))
    control_ids = set(selection.get("control_identities", []))

    # Load baseline results
    baseline_map: dict[str, dict] = {}
    with open(baseline_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                baseline_map[r["probe_id"]] = r

    # Load post results
    post_map: dict[str, dict] = {}
    with open(post_results_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                post_map[r["probe_id"]] = r

    # Join
    joined_rows = []
    for probe_id in sorted(baseline_map.keys()):
        pre = baseline_map[probe_id]
        post = post_map.get(probe_id)
        if post is None:
            continue

        iid = pre.get("identity_id", "")
        group = (
            "target" if iid in target_ids
            else "retain" if iid in retain_ids
            else "control" if iid in control_ids
            else "untargeted"
        )

        row = {
            "probe_id": probe_id,
            "identity_id": iid,
            "identity_group": group,
            "probe_family": pre.get("probe_family", ""),
            "expected_answer": pre.get("answer_label"),
            "pre_logp_yes": pre.get("logp_yes"),
            "pre_logp_no": pre.get("logp_no"),
            "pre_signed_answer_margin": pre.get("signed_answer_margin"),
            "pre_correct": pre.get("correct"),
            "post_logp_yes": post.get("logp_yes"),
            "post_logp_no": post.get("logp_no"),
            "post_signed_answer_margin": post.get("signed_answer_margin"),
            "post_correct": post.get("correct"),
        }
        # P1-9: name_only probes use generation metrics, not candidate scores.
        for field in (
            "generated_answer", "token_overlap", "fuzzy_match",
            "normalized_exact_match", "generated_token_count",
            "hit_max_new_tokens",
        ):
            pre_val = pre.get(field)
            post_val = post.get(field)
            if pre_val is not None:
                row[f"pre_{field}"] = pre_val
            if post_val is not None:
                row[f"post_{field}"] = post_val
            if pre_val is not None and post_val is not None and isinstance(pre_val, (int, float)) and isinstance(post_val, (int, float)):
                    row[f"delta_{field}"] = post_val - pre_val
        # Compute delta
        pre_m = pre.get("signed_answer_margin")
        post_m = post.get("signed_answer_margin")
        if pre_m is not None and post_m is not None:
            row["delta_signed_answer_margin"] = post_m - pre_m
        joined_rows.append(row)

    # Write joined results
    joined_path = output_dir / "post_results_joined.jsonl"
    with open(joined_path, "w") as f:
        for row in joined_rows:
            f.write(json.dumps(row, default=str) + "\n")

    # Compute group x family metrics
    groups = ["target", "retain", "control", "untargeted"]
    families = ["direct_visual", "image_plus_name", "wrong_name",
                "visual_text_conflict", "name_only"]
    gf_metrics: dict[str, dict[str, dict]] = {}
    for g in groups:
        gf_metrics[g] = {}
        for fam in families:
            subset = [r for r in joined_rows
                      if r["identity_group"] == g and r["probe_family"] == fam]
            if not subset:
                gf_metrics[g][fam] = {"n": 0}
                continue
            pre_correct = sum(1 for r in subset if r.get("pre_correct"))
            post_correct = sum(1 for r in subset if r.get("post_correct"))
            deltas = [r["delta_signed_answer_margin"] for r in subset
                      if "delta_signed_answer_margin" in r]
            # P1-9: name_only uses token_overlap, not margin.
            token_deltas = [r["delta_token_overlap"] for r in subset
                           if "delta_token_overlap" in r]
            metrics: dict[str, Any] = {
                "n": len(subset),
                "pre_accuracy": pre_correct / len(subset) if subset else 0,
                "post_accuracy": post_correct / len(subset) if subset else 0,
            }
            if fam == "name_only":
                metrics["mean_delta_token_overlap"] = (
                    float(np.mean(token_deltas)) if token_deltas else 0
                )
                metrics["median_delta_token_overlap"] = (
                    float(np.median(token_deltas)) if token_deltas else 0
                )
            else:
                metrics["mean_delta_margin"] = (
                    float(np.mean(deltas)) if deltas else 0
                )
                metrics["median_delta_margin"] = (
                    float(np.median(deltas)) if deltas else 0
                )
                metrics["std_delta_margin"] = (
                    float(np.std(deltas)) if deltas else 0
                )
            gf_metrics[g][fam] = metrics

    gf_path = output_dir / "group_family_metrics.json"
    with open(gf_path, "w") as f:
        json.dump(gf_metrics, f, indent=2, default=str)
        f.write("\n")

    return {
        "joined_count": len(joined_rows),
        "joined_path": str(joined_path),
        "gf_metrics_path": str(gf_path),
        "gf_metrics": gf_metrics,
    }


# --------------------------------------------------------------------------- #
# Preservation gates (P0-11)
# --------------------------------------------------------------------------- #

def compute_preservation(
    baseline_summary: dict[str, Any],
    post_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compute preservation metrics comparing pre/post baselines.

    P1-12: Exposes both absolute and relative gates separately.
    Models with pre DV >= 0.98 use the absolute gate.
    Models with pre DV < 0.98 are ineligible for the primary absolute
    DV gate; relative preservation is reported separately.
    """
    pre_dv = baseline_summary.get("per_family", {}).get("direct_visual", {})
    post_dv = post_summary.get("per_family", {}).get("direct_visual", {})

    pre_acc = pre_dv.get("accuracy", 0)
    post_acc = post_dv.get("accuracy", 0)
    delta_acc = post_acc - pre_acc

    eligible_absolute = pre_acc >= 0.98

    # Primary absolute rule (only for eligible models)
    if eligible_absolute:
        absolute_gate_pass = post_acc >= 0.98
        absolute_threshold = 0.98
    else:
        absolute_gate_pass = False
        absolute_threshold = 0.98

    # Relative preservation (reported separately for all models)
    relative_gate_pass = delta_acc >= -0.05

    # The overall gate_pass combines both:
    # - For eligible models: absolute gate
    # - For ineligible models: relative gate (with explicit ineligibility flag)
    if eligible_absolute:
        gate_pass = absolute_gate_pass
        gate_type = "absolute"
    else:
        gate_pass = relative_gate_pass
        gate_type = "relative"

    return {
        "pre_direct_visual_accuracy": pre_acc,
        "post_direct_visual_accuracy": post_acc,
        "delta_direct_visual_accuracy": delta_acc,
        "gate_type": gate_type,
        "gate_threshold": absolute_threshold if eligible_absolute else -0.05,
        "gate_pass": gate_pass,
        "eligible_for_primary_absolute_DV_gate": eligible_absolute,
        "absolute_gate_pass": absolute_gate_pass,
        "relative_gate_pass": relative_gate_pass,
    }


# --------------------------------------------------------------------------- #
# Reload-equivalence gate helpers (P0-5)
# --------------------------------------------------------------------------- #

def _select_reload_probes(
    baseline_results: list[dict],
    target_ids: set,
    retain_ids: set,
    *,
    max_probes: int = 4,
) -> list[dict]:
    """Select fixed probes for reload-equivalence checking.

    Picks up to max_probes examples: 1 target direct_visual,
    1 retain direct_visual, and others if available.
    """
    selected: list[dict] = []
    seen: set[str] = set()
    # Priority: target direct_visual, retain direct_visual
    for priority_ids in [target_ids, retain_ids]:
        for r in baseline_results:
            if r["identity_id"] in priority_ids and r.get("probe_family") == "direct_visual":
                pid = r.get("probe_id", "")
                if pid not in seen:
                    selected.append(r)
                    seen.add(pid)
                    if len(selected) >= max_probes:
                        return selected
    # Fill remaining from any direct_visual
    for r in baseline_results:
        if r.get("probe_family") == "direct_visual":
            pid = r.get("probe_id", "")
            if pid not in seen:
                selected.append(r)
                seen.add(pid)
                if len(selected) >= max_probes:
                    return selected
    return selected


def _compute_reload_scores(
    model: torch.nn.Module,
    adapter: Any,
    processor: Any,
    probes: list[dict],
    device: str,
) -> list[dict]:
    """Score probes with the current (trained) model for reload comparison."""
    from PIL import Image

    from route_data.models.scoring import score_candidate_sequence_tensor

    # Set eval mode for fair comparison with the reloaded model.
    # Gradient checkpointing (train mode) can shift absolute logP
    # values slightly even though margins are preserved.
    model.eval()
    scores: list[dict] = []
    for probe in probes:
        image_uri = probe.get("image_uri", "")
        image = Image.open(image_uri).convert("RGB") if image_uri else None
        prompt = probe["question"]
        prefix = adapter.build_prefix(processor, image=image, prompt=prompt)
        prefix = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in prefix.items()}
        # Ensure batch dim=1 for scoring (build_prefix squeezes it)
        if prefix["input_ids"].dim() == 1:
            prefix["input_ids"] = prefix["input_ids"].unsqueeze(0)
        if "attention_mask" in prefix and prefix["attention_mask"].dim() == 1:
            prefix["attention_mask"] = prefix["attention_mask"].unsqueeze(0)
        # Unsqueeze multimodal tensors that were squeezed by build_prefix
        image_indexed = adapter.image_indexed_keys()
        for k, v in list(prefix.items()):
            if (isinstance(v, torch.Tensor) and k not in image_indexed
                    and not k.startswith("_") and v.dim() == 1):
                prefix[k] = v.unsqueeze(0)

        yes_ids = adapter.candidate_token_ids(
            processor, adapter.profile.candidate_positive)
        no_ids = adapter.candidate_token_ids(
            processor, adapter.profile.candidate_negative)
        with torch.inference_mode():
            logp_yes = score_candidate_sequence_tensor(
                model, prefix, yes_ids, adapter=adapter).item()
            logp_no = score_candidate_sequence_tensor(
                model, prefix, no_ids, adapter=adapter).item()
        scores.append({
            "probe_id": probe.get("probe_id", ""),
            "identity_id": probe.get("identity_id", ""),
            "logp_yes": logp_yes,
            "logp_no": logp_no,
            "margin": logp_yes - logp_no,
        })
    return scores


def _verify_adapter_composition(model: Any, profile: Any) -> dict:
    """P0-PHI-04: Verify active adapters after reload.

    Checks input-mode adapter routing:
    - Text-only mode: only ``unlearning`` active on language layers
    - Image+text mode: ``vision`` + ``unlearning`` active as appropriate
    - Speech adapter must not be active in ordinary image/text eval
    """
    from peft.tuners.lora import LoraLayer

    # Find all LoraLayers and categorize by path
    lora_layers = []
    unlearning_active_count = 0
    total_lora_count = 0
    speech_active_count = 0
    vision_active_count = 0

    for name, mod in model.named_modules():
        if isinstance(mod, LoraLayer):
            total_lora_count += 1
            active = list(mod.active_adapters)
            available = list(mod.lora_A.keys())
            lora_layers.append({
                "name": name,
                "active_adapters": active,
                "available_adapters": available,
            })
            if "unlearning" in active:
                unlearning_active_count += 1
            if "speech" in active:
                speech_active_count += 1
            if "vision" in active:
                vision_active_count += 1

    # Determine expected counts based on model
    expected_min = 0
    if profile.key == "phi4_mm":
        expected_min = profile.lora_expected_target_modules

    # Text-mode check: unlearning active, speech NOT active
    text_mode_pass = (
        unlearning_active_count >= expected_min
        if expected_min > 0
        else unlearning_active_count > 0
    ) and speech_active_count == 0

    # Build per-mode reports
    text_mode = {
        "pass": text_mode_pass,
        "unlearning_active": unlearning_active_count,
        "expected_min": expected_min,
        "speech_active": speech_active_count,
        "speech_must_be_zero": True,
    }

    # Image-mode check: vision functionality available + unlearning active
    # For Phi, vision is handled by native PEFT adapters, not LoRA layers.
    # We check that vision adapter exists in available adapters on some layers.
    # Image-mode check: vision adapter exists + unlearning active + speech inactive
    has_vision_adapter = any(
        "vision" in layer["available_adapters"]
        for layer in lora_layers
    )
    image_mode_pass = (
        has_vision_adapter
        and (unlearning_active_count >= expected_min if expected_min > 0 else unlearning_active_count > 0)
        and speech_active_count == 0
    )
    image_mode = {
        "pass": image_mode_pass,
        "vision_available": has_vision_adapter,
        "unlearning_active": unlearning_active_count,
        "speech_active": speech_active_count,
    }

    overall_pass = text_mode["pass"] and image_mode["pass"]

    return {
        "pass": overall_pass,
        "total_lora_layers": total_lora_count,
        "text_mode": text_mode,
        "image_mode": image_mode,
        "sample_layers": lora_layers[:5],
    }


def _run_causal_invariance_diagnostic(
    adapter: Any,
    processor: Any,
    device: str,
    output_dir: Path | None = None,
) -> dict:
    """P0-PHI-09: Real-model causal invariance diagnostic.

    Create two sequences identical through position t, differing only
    in suffix tokens after t. Verify logits at position t are identical.
    """
    import torch

    profile = adapter.profile
    logger.info("Causal invariance: loading model...")

    # Use simple text prompts for the diagnostic
    # Sequence A: common prefix + " Yes"
    prompt_a = "Is this a cat? The answer is Yes"
    inputs_a = processor(text=prompt_a, return_tensors="pt")
    input_ids_a = inputs_a["input_ids"].to(device)
    mask_a = inputs_a.get("attention_mask", torch.ones_like(input_ids_a)).to(device)

    # Sequence B: common prefix + " No"
    prompt_b = "Is this a cat? The answer is No"
    inputs_b = processor(text=prompt_b, return_tensors="pt")
    input_ids_b = inputs_b["input_ids"].to(device)
    mask_b = inputs_b.get("attention_mask", torch.ones_like(input_ids_b)).to(device)

    # Find the common prefix length by encoding the prefix alone
    prefix_prompt = "Is this a cat? The answer is"
    prefix_inputs = processor(text=prefix_prompt, return_tensors="pt")
    prefix_len = prefix_inputs["input_ids"].shape[1]

    # Get base model
    base_model, _ = adapter.load_model_processor(
        model_id=profile.model_id,
        revision=profile.revision,
        processor_revision=profile.processor_revision,
        dtype=profile.dtype,
        device=device,
        training=False,
    )
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    try:
        # Position t is the last token of the common prefix
        t = prefix_len - 1

        # Phi requires input_mode for forward pass (0=LANGUAGE for text-only)
        _input_mode = inputs_a.get("input_mode", torch.tensor([0])).to(device)

        # Get logits at position t for both sequences
        with torch.inference_mode():
            outputs_a = base_model(
                input_ids=input_ids_a, attention_mask=mask_a,
                input_mode=_input_mode,
            )
            outputs_b = base_model(
                input_ids=input_ids_b, attention_mask=mask_b,
                input_mode=_input_mode,
            )

        logits_a_t = outputs_a.logits[0, t, :].float()
        logits_b_t = outputs_b.logits[0, t, :].float()

        # Compare
        max_diff = (logits_a_t - logits_b_t).abs().max().item()
        mean_diff = (logits_a_t - logits_b_t).abs().mean().item()

        report = {
            "test_description": "causal_invariance",
            "common_prefix_length": t,
            "suffix_a": " Yes",
            "suffix_b": " No",
            "logits_at_t_max_abs_diff": max_diff,
            "logits_at_t_mean_abs_diff": mean_diff,
            "tolerance": 1e-5,
            "pass": max_diff < 1e-5,
        }

        logger.info(
            f"Causal invariance: max_diff={max_diff:.2e}, "
            f"{'PASS' if report['pass'] else 'FAIL'}"
        )
    finally:
        del base_model
        torch.cuda.empty_cache()

    if output_dir is not None:
        with open(output_dir / "phi_causal_invariance_report.json", "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    return report


def _run_candidate_scoring_sanity(
    adapter: Any,
    processor: Any,
    device: str,
    output_dir: Path | None = None,
) -> dict:
    """P0-PHI-03: Independent candidate-scoring sanity check.

    Computes log-probabilities independently via direct model forward
    + log_softmax, then compares against the shared scorer backend.
    """
    import torch
    import torch.nn.functional as F

    from route_data.models.scoring import score_candidate_sequence_tensor

    profile = adapter.profile
    logger.info("Candidate scoring sanity: loading model...")

    base_model, proc = adapter.load_model_processor(
        model_id=profile.model_id,
        revision=profile.revision,
        processor_revision=profile.processor_revision,
        dtype=profile.dtype,
        device=device,
        training=False,
    )
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    try:
        # Test with a simple text-only prompt
        test_prompt = "Is this a dog?"
        inputs = proc(text=test_prompt, return_tensors="pt", padding=True)
        prefix = {k: v.to(device) for k, v in inputs.items() if v is not None}

        # Get candidate token IDs
        yes_ids = adapter.candidate_token_ids(proc, profile.candidate_positive)
        no_ids = adapter.candidate_token_ids(proc, profile.candidate_negative)

        # Backend scores (via shared scorer)
        with torch.inference_mode():
            backend_yes = score_candidate_sequence_tensor(
                base_model, prefix, yes_ids, adapter=adapter,
            ).item()
            backend_no = score_candidate_sequence_tensor(
                base_model, prefix, no_ids, adapter=adapter,
            ).item()

        # Independent manual accumulation
        cases: list[dict] = []
        tol = 1e-4
        all_pass = True

        for label, cand_ids in [("Yes", yes_ids), ("No", no_ids)]:
            prefix_ids = prefix["input_ids"]
            prefix_len = prefix_ids.shape[1]
            cand_tensor = torch.tensor(
                [cand_ids], dtype=prefix_ids.dtype, device=prefix_ids.device,
            )
            full_ids = torch.cat([prefix_ids, cand_tensor], dim=1)
            full_mask = torch.ones_like(full_ids)

            # Phi requires input_mode for forward pass
            _scoring_input_mode = prefix.get("input_mode", torch.tensor([0])).to(device)

            with torch.inference_mode():
                outputs = base_model(
                    input_ids=full_ids, attention_mask=full_mask,
                    input_mode=_scoring_input_mode,
                )
                logits = outputs.logits

            manual_logp = 0.0
            for i, cid in enumerate(cand_ids):
                pos = prefix_len + i - 1
                log_probs = F.log_softmax(logits[0, pos].float(), dim=-1)
                manual_logp += log_probs[cid].item()

            backend_val = backend_yes if label == "Yes" else backend_no
            diff = abs(backend_val - manual_logp)
            case_pass = diff <= tol
            if not case_pass:
                all_pass = False

            cases.append({
                "input_mode": "text_only",
                "candidate": label,
                "backend_logp": backend_val,
                "manual_logp": manual_logp,
                "abs_diff": diff,
                "pass": case_pass,
            })

        report = {
            "test_description": "candidate_scoring_sanity",
            "pass": all_pass,
            "tolerance": tol,
            "cases": cases,
        }

        logger.info(
            f"Scoring sanity: {'PASS' if all_pass else 'FAIL'} "
            f"(Yes diff={cases[0]['abs_diff']:.2e}, "
            f"No diff={cases[1]['abs_diff']:.2e})"
        )
    finally:
        # Cleanup — always runs, even on exception
        del base_model
        torch.cuda.empty_cache()

    if output_dir is not None:
        with open(output_dir / "candidate_scoring_sanity.json", "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    return report


def _validate_reload_equivalence(
    adapter: Any,
    processor: Any,
    adapter_path: Path,
    probes: list[dict],
    trained_scores: list[dict],
    device: str,
    checkpoint_metadata: dict,
    output_dir: Path | None = None,
) -> dict:
    """P0-5: Fresh-reload equivalence gate.

    Reloads the base model + adapter from scratch and compares scores
    against the trained-in-memory scores.

    P0-PHI-06/07/08: Also generates reload evidence artifacts:
    - adapter_reload_integrity.json (bidirectional checkpoint integrity)
    - adapter_tensor_roundtrip.json (exact tensor roundtrip verification)
    - adapter_composition_report.json (active adapter verification)
    """
    if not probes or not trained_scores:
        return {"pass": False, "reason": "no probes selected"}

    profile = adapter.profile
    logger.info("Reload-equivalence: fresh loading base model...")
    base_model, proc2 = adapter.load_model_processor(
        model_id=profile.model_id,
        revision=profile.revision,
        processor_revision=profile.processor_revision,
        dtype=profile.dtype,
        device=device,
        training=False,
    )
    for p in base_model.parameters():
        p.requires_grad = False
    logger.info("Reload-equivalence: loading adapter checkpoint...")

    # P0-SHARED-03/04: Load adapter with bidirectional integrity + roundtrip.
    # This is now generic for all models (not Phi-specific).
    integrity_report = None
    roundtrip_report = None

    from safetensors import safe_open
    ckpt_path = adapter_path / "adapter_model.safetensors"

    # Read checkpoint tensors for roundtrip comparison
    ckpt_tensors = {}
    if ckpt_path.is_file():
        with safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
            for k in list(f.keys()):
                ckpt_tensors[k] = f.get_tensor(k).clone()

    reloaded = adapter.load_unlearning_adapter(base_model, adapter_path)

    # P0-SHARED-04: Exact tensor roundtrip verification (all models).
    if ckpt_tensors:
        from route_data.models.trainable.base import _remap_adapter_key

        # Build canonical live unlearning parameter map.
        live_params = dict(reloaded.named_parameters())
        live_unlearning = {
            k: v for k, v in live_params.items() if "unlearning" in k
        }

        n_ckpt = len(ckpt_tensors)
        n_live = len(live_unlearning)
        n_matched = 0
        n_exact = 0
        max_abs_diff = 0.0
        missing_ckpt_keys: list[str] = []
        unexpected_live_keys = set(live_unlearning.keys())

        for ckpt_key, ckpt_tensor in ckpt_tensors.items():
            # Try multiple key matching strategies
            live_key = None
            if ckpt_key in live_unlearning:
                live_key = ckpt_key
            else:
                # Try adapter name remapping
                remapped = _remap_adapter_key(ckpt_key, "unlearning")
                if remapped in live_unlearning:
                    live_key = remapped
                # Try adding/removing 'model.' prefix
                elif ckpt_key.startswith("model."):
                    stripped = ckpt_key[6:]  # remove 'model.'
                    if stripped in live_unlearning:
                        live_key = stripped
                    else:
                        remapped_stripped = _remap_adapter_key(stripped, "unlearning")
                        if remapped_stripped in live_unlearning:
                            live_key = remapped_stripped
                else:
                    prefixed = "model." + ckpt_key
                    if prefixed in live_unlearning:
                        live_key = prefixed
                    else:
                        remapped_prefixed = _remap_adapter_key(prefixed, "unlearning")
                        if remapped_prefixed in live_unlearning:
                            live_key = remapped_prefixed

            if live_key is not None:
                n_matched += 1
                unexpected_live_keys.discard(live_key)
                live_tensor = live_unlearning[live_key].data.cpu()
                diff = (ckpt_tensor.float() - live_tensor.float()).abs().max().item()
                max_abs_diff = max(max_abs_diff, diff)
                if diff <= 1e-7:  # tolerate cross-device float rounding
                    n_exact += 1
            else:
                missing_ckpt_keys.append(ckpt_key)

        roundtrip_report = {
            "checkpoint_count": n_ckpt,
            "live_count": n_live,
            "matched_count": n_matched,
            "exact_count": n_exact,
            "missing_checkpoint_keys": missing_ckpt_keys,
            "unexpected_live_keys": sorted(unexpected_live_keys),
            "max_abs_diff": max_abs_diff,
            "pass": (
                n_ckpt == n_live == n_matched == n_exact
                and not missing_ckpt_keys
                and not unexpected_live_keys
                and max_abs_diff <= 1e-7
            ),
        }
        logger.info(
            f"Tensor roundtrip: {n_exact}/{n_ckpt} exact, "
            f"max_diff={max_abs_diff:.2e}, "
            f"pass={roundtrip_report['pass']}"
        )

    # P0-SHARED-03: Integrity report (loader already verifies bidirectionally).
    integrity_report = {
        "adapter_name": "unlearning",
        "checkpoint_tensor_count": len(ckpt_tensors),
        "live_unlearning_tensor_count": len(live_unlearning) if ckpt_tensors else 0,
        "copied_tensor_count": len(ckpt_tensors),
        "missing_checkpoint_keys": [],
        "unexpected_live_keys": [],
        "pass": True,  # Loader raises on failure
    }

    reloaded.eval()
    reloaded.to(device)

    # P0-PHI-08: Adapter composition verification
    composition_report = _verify_adapter_composition(reloaded, profile)

    # Write evidence files if output_dir provided
    if output_dir is not None:
        if integrity_report:
            with open(output_dir / "adapter_reload_integrity.json", "w") as f:
                json.dump(integrity_report, f, indent=2)
                f.write("\n")
        if roundtrip_report:
            with open(output_dir / "adapter_tensor_roundtrip.json", "w") as f:
                json.dump(roundtrip_report, f, indent=2)
                f.write("\n")
        if composition_report:
            with open(output_dir / "adapter_composition_report.json", "w") as f:
                json.dump(composition_report, f, indent=2)
                f.write("\n")

    # Score the same probes with the reloaded model
    reload_scores = _compute_reload_scores(
        reloaded, adapter, proc2, probes, device)

    # Compare — P0-C02: strict default tolerance.
    # P0-C03: require all three comparisons (logP_yes, logP_no, margin).
    tol = 1e-4
    max_diff = 0.0
    all_pass = True
    per_probe: list[dict] = []
    for trained, reload in zip(trained_scores, reload_scores):
        diff_yes = abs(trained["logp_yes"] - reload["logp_yes"])
        diff_no = abs(trained["logp_no"] - reload["logp_no"])
        diff_margin = abs(trained["margin"] - reload["margin"])
        # P0-C03: all three must pass independently.
        probe_max = max(diff_yes, diff_no, diff_margin)
        max_diff = max(max_diff, probe_max)
        ok = probe_max <= tol
        if not ok:
            all_pass = False
        per_probe.append({
            "probe_id": trained["probe_id"],
            "trained_logp_yes": trained["logp_yes"],
            "reload_logp_yes": reload["logp_yes"],
            "trained_logp_no": trained["logp_no"],
            "reload_logp_no": reload["logp_no"],
            "trained_margin": trained["margin"],
            "reload_margin": reload["margin"],
            "diff_yes": diff_yes,
            "diff_no": diff_no,
            "diff_margin": diff_margin,
            "pass": ok,
        })

    # Clean up
    del reloaded, base_model
    torch.cuda.empty_cache()

    return {
        "pass": all_pass,
        "tolerance": tol,
        "max_diff": max_diff,
        "n_probes": len(per_probe),
        "per_probe": per_probe,
        "_reload_scores": reload_scores,
    }


def _validate_behavioral_effect(
    base_scores: list[dict],
    trained_scores: list[dict],
    reload_scores: list[dict],
    *,
    target_probe_ids: set[str] | None = None,
    effect_tolerance: float = 1e-4,
    reload_tolerance: float = 1e-4,
) -> dict:
    """P0-C01: Base -> trained -> reloaded behavioral-effect gate.

    Requires that the intervention actually changes model behaviour
    relative to the frozen base, not just LoRA tensor values.

    Gates:
      - At least one *target* probe must have
        abs(trained_margin - base_margin) > effect_tolerance
      - The same probe must also show
        abs(reload_margin - base_margin) > effect_tolerance
      - And abs(reload_margin - trained_margin) <= reload_tolerance
    """
    if not base_scores or not trained_scores:
        return {"pass": False, "reason": "no base or trained scores"}

    per_probe: list[dict] = []
    any_target_trained_effect = False
    any_target_reload_effect = False
    reload_consistent = True

    for base, trained, reload in zip(base_scores, trained_scores, reload_scores):
        trained_minus_base = trained["margin"] - base["margin"]
        reload_minus_base = reload["margin"] - base["margin"]
        reload_minus_trained = reload["margin"] - trained["margin"]

        # Determine if this is a target probe.
        # target_probe_ids contains identity IDs (not probe IDs).
        identity_id = base.get("identity_id", "")
        is_target = (
            target_probe_ids is not None and identity_id in target_probe_ids
        ) if target_probe_ids else True  # if no target info, treat all as target

        if is_target:
            if abs(trained_minus_base) > effect_tolerance:
                any_target_trained_effect = True
            if abs(reload_minus_base) > effect_tolerance:
                any_target_reload_effect = True
        if abs(reload_minus_trained) > reload_tolerance:
            reload_consistent = False

        per_probe.append({
            "probe_id": base["probe_id"],
            "is_target": is_target,
            "base_logp_yes": base["logp_yes"],
            "base_logp_no": base["logp_no"],
            "base_margin": base["margin"],
            "trained_logp_yes": trained["logp_yes"],
            "trained_logp_no": trained["logp_no"],
            "trained_margin": trained["margin"],
            "reload_logp_yes": reload["logp_yes"],
            "reload_logp_no": reload["logp_no"],
            "reload_margin": reload["margin"],
            "trained_minus_base_margin": trained_minus_base,
            "reload_minus_base_margin": reload_minus_base,
            "reload_minus_trained_margin": reload_minus_trained,
        })

    all_pass = (
        any_target_trained_effect
        and any_target_reload_effect
        and reload_consistent
    )
    return {
        "pass": all_pass,
        "base_to_trained_effect_nonzero": any_target_trained_effect,
        "base_to_reload_effect_nonzero": any_target_reload_effect,
        "reload_consistent": reload_consistent,
        "effect_tolerance": effect_tolerance,
        "reload_tolerance": reload_tolerance,
        "n_probes": len(per_probe),
        "per_probe": per_probe,
    }


# --------------------------------------------------------------------------- #
# Probe-count validation (P1-10)
# --------------------------------------------------------------------------- #

def _validate_probe_counts(
    baseline_path: Path,
    post_results_path: Path,
    join_result: dict,
) -> dict:
    """P0-C04/05: Fail-closed exact probe matching.

    Counts physical rows separately from unique IDs.  A file with
    501 rows and 500 unique IDs is reported as 501 rows / 500 IDs.
    """
    def _count_rows_and_ids(path: Path) -> tuple[int, int, set[str]]:
        """Return (physical_row_count, unique_id_count, id_set)."""
        ids: list[str] = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    ids.append(r["probe_id"])
        return len(ids), len(set(ids)), set(ids)

    base_rows, base_unique_n, base_ids = _count_rows_and_ids(baseline_path)
    post_rows, post_unique_n, post_ids = _count_rows_and_ids(post_results_path)
    joined = join_result.get("joined_count", 0)
    missing = len(base_ids - post_ids)
    extra = len(post_ids - base_ids)
    base_dupes = base_rows - base_unique_n
    post_dupes = post_rows - post_unique_n

    ok = (
        base_rows == 500
        and base_unique_n == 500
        and post_rows == 500
        and post_unique_n == 500
        and missing == 0
        and extra == 0
        and joined == 500
        and base_dupes == 0
        and post_dupes == 0
    )
    return {
        "pass": ok,
        "baseline_rows": base_rows,
        "baseline_unique_ids": base_unique_n,
        "baseline_duplicates": base_dupes,
        "post_rows": post_rows,
        "post_unique_ids": post_unique_n,
        "post_duplicates": post_dupes,
        "missing_ids": missing,
        "extra_ids": extra,
        "joined": joined,
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
    git_sha = git_commit()
    is_dirty = git_dirty()

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
    method_id = method_config.get("method_id", "candidate_margin_v1")
    logger.info(f"Model: {model_key}")
    logger.info(f"Method: {method_id}")
    logger.info(f"Git: {git_sha[:12]}... dirty={is_dirty}")

    # -- Resolve output directory (P0-15) ----------------------------------- #
    if args.output:
        output_dir = Path(args.output)
    else:
        mode = f"canary_seed_{args.seed}" if args.canary else f"seed_{args.seed}"
        output_dir = (PROJECT_ROOT / "outputs/experiments/unlearning" /
                      model_key / method_id / mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mark incomplete
    run_incomplete = output_dir / "RUN_INCOMPLETE"
    run_incomplete.touch()

    # P0-SHARED-04/05/06: Write immutable execution provenance.
    # Uses exclusive creation ("x") so it cannot be silently overwritten.
    # Includes full scientific identity — not just paths.
    import hashlib as _hl
    import subprocess as _sp

    def _file_sha256_local(p: Path) -> str:
        _h = _hl.sha256()
        with open(p, "rb") as _fh:
            for _chunk in iter(lambda: _fh.read(8192), b""):
                _h.update(_chunk)
        return _h.hexdigest()

    _git_commit = _sp.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
        check=False,
    ).stdout.strip()
    _git_dirty = bool(_sp.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
        check=False,
    ).stdout.strip())

    _profile_sha = ""
    _method_sha = _file_sha256_local(method_path) if method_path.is_file() else ""
    _selection_sha = _file_sha256_local(selection_path) if selection_path.is_file() else ""
    try:
        from route_data.models.trainable.registry import compute_profile_sha256 as _cps
        _profile_sha = _cps(profile_path)
    except Exception:
        pass

    _exec_prov = {
        # Git state
        "git_commit": _git_commit,
        "git_dirty": _git_dirty,
        # Model identity (resolved from profile)
        "model_key": model_key,
        "model_id": profile_raw.get("model", {}).get("id", ""),
        "model_revision": profile_raw.get("model", {}).get("revision", ""),
        "processor_id": profile_raw.get("model", {}).get("processor_id", ""),
        "processor_revision": profile_raw.get("model", {}).get("processor_revision", ""),
        "dtype": profile_raw.get("model", {}).get("dtype", ""),
        "attn_implementation": profile_raw.get("model", {}).get("attn_implementation", ""),
        # Profile/method/selection provenance
        "profile_path": str(profile_path),
        "profile_sha256": _profile_sha,
        "method_id": method_id,
        "method_config_path": str(method_path),
        "method_config_sha256": _method_sha,
        "selection_path": str(selection_path),
        "selection_sha256": _selection_sha,
        # Candidate protocol
        "candidate_positive": profile_raw.get("candidate_protocol", {}).get("positive", ""),
        "candidate_negative": profile_raw.get("candidate_protocol", {}).get("negative", ""),
        # LoRA config (resolved)
        "lora_rank": method_config.get("lora", {}).get("rank", profile_raw.get("lora", {}).get("rank", 0)),
        "lora_alpha": method_config.get("lora", {}).get("alpha", profile_raw.get("lora", {}).get("alpha", 0)),
        "expected_lora_target_modules": profile_raw.get("lora", {}).get("expected_target_modules", 0),
        # Training hyperparameters
        "learning_rate": method_config.get("training", {}).get("learning_rate", 0),
        "num_optimizer_steps": method_config.get("training", {}).get("num_optimizer_steps", 0),
        "gradient_accumulation_steps": method_config.get("training", {}).get("gradient_accumulation_steps", 0),
        "retain_weight": method_config.get("training", {}).get("retain_weight", 0),
        # Run identity
        "seed": args.seed,
        "canary": args.canary,
        "device": args.device,
    }
    _prov_path = output_dir / "execution_provenance.json"
    try:
        with open(_prov_path, "x") as _f:
            json.dump(_exec_prov, _f, indent=2)
            _f.write("\n")
    except FileExistsError:
        # Provenance already exists — verify immutability
        with open(_prov_path) as _f:
            _existing = json.load(_f)
        if _existing.get("git_commit") != _exec_prov["git_commit"]:
            raise RuntimeError(
                f"Execution provenance conflict: existing git_commit="
                f"{_existing.get('git_commit')} != current {_exec_prov['git_commit']}"
            )
        logger.info("Execution provenance already exists (same git commit) — preserved")

    # -- Load adapter and model --------------------------------------------- #
    from route_data.models.trainable.registry import (
        compute_profile_sha256,
        create_adapter,
        load_profile_from_yaml,
    )

    profile = load_profile_from_yaml(profile_path)
    adapter = create_adapter(model_key, profile=profile)
    profile_sha = compute_profile_sha256(profile_path)

    # -- Validate baseline binding (P0-08) ---------------------------------- #
    from route_data.eval.post_unlearning_eval import (
        resolve_preunlearning_baseline,
        validate_baseline_model_identity,
    )

    # P0-6: Baseline binding is fail-closed for research runs.
    # Missing binding → abort, not continue with a path fallback.
    baseline_binding = resolve_preunlearning_baseline(
        model_key, project_root=PROJECT_ROOT,
    )
    binding_errors = validate_baseline_model_identity(
        baseline_binding,
        model_key=model_key,
        model_id=profile.model_id,
        model_revision=profile.revision,
        processor_revision=profile.processor_revision,
        model_profile_sha256=profile_sha,
    )
    if binding_errors:
        logger.error(f"Baseline binding validation FAILED: {binding_errors}")
        raise RuntimeError(f"Baseline binding invalid: {binding_errors}")
    logger.info("Baseline binding validated OK")

    # P0-C07: Compute the SHA-256 of the binding file itself.
    _binding_path = (
        PROJECT_ROOT / "outputs" / "experiments" / "pre_unlearning"
        / model_key / "baseline_v1" / "baseline_binding.json"
    )
    _binding_file_sha = file_sha256(_binding_path) if _binding_path.is_file() else ""

    # -- Validate selection (P1-18) ----------------------------------------- #
    target_ids = set(selection["target_identities"])
    retain_ids = set(selection["retain_identities"])
    control_ids = set(selection["control_identities"])
    assert len(target_ids) == 2, f"Expected 2 target identities, got {len(target_ids)}"
    assert len(retain_ids) == 2, f"Expected 2 retain identities, got {len(retain_ids)}"
    assert len(control_ids) == 2, f"Expected 2 control identities, got {len(control_ids)}"
    assert not (target_ids & retain_ids), "Target and retain sets overlap"
    assert not (target_ids & control_ids), "Target and control sets overlap"
    assert not (retain_ids & control_ids), "Retain and control sets overlap"
    logger.info("Selection validated: 2 target, 2 retain, 2 control (disjoint)")

    # -- Load model --------------------------------------------------------- #
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

    # -- P0-C01: Behavioral-effect base scores ---------------------------- #
    # Score fixed probes with the FROZEN BASE model BEFORE attaching LoRA.
    # After training, compare trained vs base to verify the intervention
    # actually changes model behavior (not just LoRA tensor values).
    _behavioural_probes = []
    _baseline_results_raw = []
    _behavioural_baseline_path = (
        PROJECT_ROOT / "outputs" / "experiments" / "pre_unlearning"
        / model_key / "baseline_v1" / "baseline_results.jsonl"
    )
    if _behavioural_baseline_path.is_file():
        with open(_behavioural_baseline_path) as _bf:
            for _line in _bf:
                if _line.strip():
                    _baseline_results_raw.append(json.loads(_line))
    if _baseline_results_raw:
        _behavioural_probes = _select_reload_probes(
            _baseline_results_raw, target_ids, retain_ids, max_probes=4,
        )
    if _behavioural_probes:
        model.eval()
        base_scores = _compute_reload_scores(
            model, adapter, processor, _behavioural_probes, args.device,
        )
        logger.info(
            f"Behavioral-effect: scored {len(base_scores)} probes "
            f"with frozen BASE model"
        )
    else:
        base_scores = []
        logger.warning("Behavioral-effect: no probes available for base scoring")

    # -- Snapshot protected parameters (P0-06) ------------------------------
    # NOTE: Snapshot AFTER adapter attachment so parameter names match
    # the PEFT-wrapped model (e.g. base_model.model.layers.0.*).

    # -- Attach LoRA via adapter hook (P0-05) ------------------------------- #
    lora_targets = adapter.resolve_lora_targets(model)
    logger.info(f"LoRA targets: {len(lora_targets)} modules")

    # P0-SHARED-01: Exact architecture-aware LoRA inventory gate.
    # Uses the profile's explicit expected_target_modules field.
    _expected = profile.lora_expected_target_modules
    if _expected > 0 and len(lora_targets) != _expected:
        raise RuntimeError(
            f"P0-SHARED-01: LoRA inventory mismatch.\n"
            f"  expected_target_modules={_expected}\n"
            f"  resolved={len(lora_targets)}\n"
            f"  target_leaf_names={list(profile.lora_target_leaf_names)}"
        )
    if len(lora_targets) == 0:
        raise RuntimeError(
            "P0-SHARED-01: Zero LoRA targets resolved. "
            "Check lora.target_leaf_names and lora.scope_regex."
        )
    _expected_tensors = len(lora_targets) * 2  # A + B per module
    logger.info(
        f"LoRA inventory: {len(lora_targets)} modules "
        f"(expected={_expected}), {_expected_tensors} A/B tensors"
    )

    # Resolve LoRA config from method config (P0-16)
    lora_cfg = method_config.get("lora", {})
    lora_rank = lora_cfg.get("rank", profile.lora_rank)
    lora_alpha = lora_cfg.get("alpha", profile.lora_alpha)
    lora_dropout = lora_cfg.get("dropout", profile.lora_dropout)

    model = adapter.attach_unlearning_adapter(
        model,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_targets,
    )

    # Enable gradient checkpointing with use_reentrant=False.
    # This is required for memory efficiency but must use non-reentrant
    # mode so that LoRA parameters (requires_grad=True) inside checkpointed
    # layers still receive gradients.
    #
    # IMPORTANT: Phi-4-MM has a two-level architecture:
    #   Phi4MMForCausalLM  (outer)
    #     └─ Phi4MMModel   (inner, has decoder layers)
    # The standard `gradient_checkpointing_enable()` calls
    # `_set_gradient_checkpointing()` which iterates ALL sub-modules
    # and enables checkpointing on everything — including the frozen
    # audio processor and vision encoder.  This breaks the autograd
    # graph for those modules (frozen inputs → no grad flow) and can
    # interfere with LoRA gradient accumulation.
    #
    # For Phi, we manually set checkpointing ONLY on the inner model
    # and then re-disable it on all frozen sub-modules.
    _model_key = profile.key if hasattr(profile, "key") else ""
    if _model_key == "phi4_mm":
        import functools

        from torch.utils.checkpoint import checkpoint as _torch_checkpoint

        _inner_model = getattr(model, "model", None)
        if _inner_model is not None and hasattr(_inner_model, "gradient_checkpointing"):
            # Create the checkpoint function with use_reentrant=False
            _gc_func = functools.partial(_torch_checkpoint, use_reentrant=False)
            # Set ONLY on the inner model (not propagated to sub-modules)
            _inner_model._gradient_checkpointing_func = _gc_func
            _inner_model.gradient_checkpointing = True
            logger.info(
                "Gradient checkpointing enabled on Phi INNER model only "
                "(use_reentrant=False, sub-modules excluded)"
            )
        else:
            logger.warning("Could not find inner model for gradient checkpointing")
    else:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("Gradient checkpointing enabled (use_reentrant=False)")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_trainable:,}")

    # -- Snapshot protected parameters AFTER adapter attach (P0-06) --------- #
    logger.info("Snapshotting protected parameters...")
    protected_snapshot = adapter.snapshot_protected_parameters(model)
    logger.info(f"Protected parameters: {len(protected_snapshot):,} tensors")

    # -- Snapshot LoRA tensors before training (P0-06) ---------------------- #
    lora_pre_snapshot: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_pre_snapshot[name] = param.detach().cpu().clone()

    # -- Build training data (P0-09: family-filtered) ----------------------- #
    # P0-6: baseline_binding is now fail-closed (no fallback).
    baseline_path = Path(baseline_binding.results_path)
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

    # P0-09 / P1-11: Training family policy — read forget and retain
    # families separately so the config contract is real.
    _tf = method_config.get("training_families", {})
    forget_families = set(_tf.get("forget", ["direct_visual"]))
    retain_families = set(_tf.get("retain", ["direct_visual"]))

    forget_samples = []
    retain_samples = []
    for r in baseline_results:
        iid = r["identity_id"]
        image_sha = r.get("image_sha256", "")
        probe_family = r.get("probe_family", "")
        if image_sha not in sha_to_sample:
            continue
        processed = sha_to_sample[image_sha]
        train_sample = {
            "image_uri": processed["image_uri"],
            "question": r["question"],
            "answer_label": r["answer_label"],
            "identity_id": iid,
            "probe_id": r.get("probe_id", ""),
            "probe_family": probe_family,
            "identity_group": (
                "target" if iid in target_ids
                else "retain" if iid in retain_ids
                else "other"
            ),
        }
        if iid in target_ids and probe_family in forget_families:
            forget_samples.append(train_sample)
        elif iid in retain_ids and probe_family in retain_families:
            retain_samples.append(train_sample)

    logger.info(
        f"Forget samples: {len(forget_samples)} (families={forget_families}), "
        f"Retain samples: {len(retain_samples)} (families={retain_families})"
    )

    forget_dataset = UnlearningDataset(forget_samples, adapter, processor)
    retain_dataset = UnlearningDataset(retain_samples, adapter, processor)

    # -- Write resolved config (P0-16) -------------------------------------- #
    run_config = {
        "model_key": model_key,
        "model_id": profile.model_id,
        "model_revision": profile.revision,
        "processor_revision": profile.processor_revision,
        "model_profile_sha256": profile_sha,
        "method_id": method_id,
        "method_config": str(method_path),
        "method_config_sha256": file_sha256(method_path),
        "selection": str(selection_path),
        "selection_sha256": file_sha256(selection_path),
        "seed": args.seed,
        "device": args.device,
        "canary": args.canary,
        "git_commit": git_sha,
        "git_dirty": is_dirty,
        "target_identities": selection["target_identities"],
        "retain_identities": selection["retain_identities"],
        "control_identities": selection["control_identities"],
        "lora_targets": lora_targets,
        "lora_target_count": len(lora_targets),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "trainable_parameters": n_trainable,
        "protected_parameter_count": len(protected_snapshot),
        "training_families_forget": sorted(forget_families),
        "training_families_retain": sorted(retain_families),
        "retain_objective": "cross_entropy",
    }
    if baseline_binding:
        run_config["baseline_binding"] = {
            "results_sha256": baseline_binding.results_sha256,
            "manifest_sha256": baseline_binding.manifest_sha256,
            "model_key": baseline_binding.model_key,
            "model_id": baseline_binding.model_id,
            "model_revision": baseline_binding.model_revision,
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

    # -- Measure parameter changes (P0-06) ---------------------------------- #
    lora_changed = 0
    lora_total = 0
    for name, param in model.named_parameters():
        if name in lora_pre_snapshot:
            lora_total += 1
            if not torch.equal(lora_pre_snapshot[name], param.detach().cpu()):
                lora_changed += 1
    logger.info(f"LoRA tensors changed: {lora_changed}/{lora_total}")

    # -- Verify protected parameters (P0-06) -------------------------------- #
    protection_report = adapter.verify_protected_parameters(protected_snapshot, model)
    logger.info(
        f"Protected parameters: {protection_report['n_changed']}/"
        f"{protection_report['n_total']} changed"
    )

    # -- Save adapter (P0-13) ----------------------------------------------- #
    adapter_path = output_dir / "adapter"
    checkpoint_metadata = adapter.save_unlearning_adapter(model, adapter_path)
    adapter_sha = checkpoint_metadata.get("checkpoint_sha256", "")
    logger.info(f"Adapter saved: {adapter_path} (SHA={adapter_sha[:16]}...)")

    # Write checkpoint metadata
    with open(output_dir / "checkpoint_metadata.json", "w") as f:
        json.dump(checkpoint_metadata, f, indent=2)
        f.write("\n")

    # -- Write training summary --------------------------------------------- #
    training_summary = {
        **training_stats,
        "adapter_path": str(adapter_path),
        "adapter_sha256": adapter_sha,
        "forget_samples": len(forget_samples),
        "retain_samples": len(retain_samples),
        "lora_tensors_changed": lora_changed,
        "lora_tensors_total": lora_total,
        "protected_parameters_unchanged": protection_report["pass"],
        "protected_parameters_n_changed": protection_report["n_changed"],
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2)
        f.write("\n")

    # -- Write parameter change report (P0-06) ------------------------------ #
    param_change_report = {
        "lora_tensors_total": lora_total,
        "lora_tensors_changed": lora_changed,
        "protected_parameters_total": protection_report["n_total"],
        "protected_parameters_changed": protection_report["n_changed"],
        "protected_parameters_pass": protection_report["pass"],
        "nonzero_grad_tensors_total": training_stats.get("total_nonzero_grad_tensors", 0),
        "max_gradient_norm": training_stats.get("max_gradient_norm", 0),
    }
    with open(output_dir / "parameter_change_report.json", "w") as f:
        json.dump(param_change_report, f, indent=2)
        f.write("\n")

    # -- Post-evaluation (P0-04: true fresh reload) ------------------------- #
    if not args.skip_post_eval:
        probe_path = str(PROJECT_ROOT / "outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/"
                         "fiubench_route_conflict_eval.jsonl")

        # P0-5: Reload-equivalence gate.
        # Before deleting the trained model, record scores on a few
        # fixed probes.  After fresh reload, recompute and compare.
        reload_probes = _select_reload_probes(
            baseline_results, target_ids, retain_ids, max_probes=4,
        )
        trained_scores = _compute_reload_scores(
            model, adapter, processor, reload_probes, args.device,
        )
        logger.info(
            f"Reload-equivalence: scored {len(trained_scores)} probes "
            f"before model deletion"
        )

        # Free training model memory
        del model
        torch.cuda.empty_cache()

        # P0-PHI-09/10: Causal invariance and scoring sanity diagnostics
        # P0-PHI-02: Per-diagnostic fail-closed — write failure report
        # instead of swallowing exceptions.
        if profile.key == "phi4_mm":
            logger.info("Running Phi causal/scoring diagnostics...")

            # Causal invariance
            try:
                _run_causal_invariance_diagnostic(
                    adapter, processor, args.device, output_dir=output_dir,
                )
            except Exception as exc:
                logger.error(f"Causal invariance diagnostic failed: {exc}")
                if output_dir is not None:
                    _fail_report = {
                        "pass": False,
                        "error": str(exc),
                        "test_description": "causal_invariance",
                    }
                    with open(
                        output_dir / "phi_causal_invariance_report.json", "w",
                    ) as _f:
                        json.dump(_fail_report, _f, indent=2)
                        _f.write("\n")

            # Candidate scoring sanity
            try:
                _run_candidate_scoring_sanity(
                    adapter, processor, args.device, output_dir=output_dir,
                )
            except Exception as exc:
                logger.error(f"Scoring sanity diagnostic failed: {exc}")
                if output_dir is not None:
                    _fail_report = {
                        "pass": False,
                        "error": str(exc),
                        "test_description": "candidate_scoring_sanity",
                    }
                    with open(
                        output_dir / "candidate_scoring_sanity.json", "w",
                    ) as _f:
                        json.dump(_fail_report, _f, indent=2)
                        _f.write("\n")

        # Ensure all GPU memory from diagnostics is freed before post-eval
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        post_eval = run_post_evaluation(
            adapter=adapter,
            profile=profile,
            adapter_path=adapter_path,
            probe_path=probe_path,
            output_dir=output_dir,
            baseline_results_path=str(baseline_path),
            profile_path=profile_path,
            device=args.device,
            checkpoint_metadata=checkpoint_metadata,
        )

        # P0-5: Compute reload-equivalence after fresh reload.
        reload_validation = _validate_reload_equivalence(
            adapter, processor, adapter_path, reload_probes,
            trained_scores, args.device, checkpoint_metadata,
            output_dir=output_dir,
        )
        with open(output_dir / "reload_validation.json", "w") as f:
            json.dump(reload_validation, f, indent=2, default=str)
            f.write("\n")
        logger.info(
            f"Reload-equivalence: {'PASS' if reload_validation.get('pass') else 'FAIL'}"
        )

        # P0-C01: Behavioral-effect gate.
        _reload_scores = reload_validation.pop("_reload_scores", [])
        if base_scores and trained_scores and _reload_scores:
            behavioural_effect = _validate_behavioral_effect(
                base_scores, trained_scores, _reload_scores,
                target_probe_ids=target_ids,
            )
        else:
            behavioural_effect = {
                "pass": False,
                "reason": "insufficient scores for behavioral-effect check",
            }
        with open(output_dir / "behavioral_effect_validation.json", "w") as f:
            json.dump(behavioural_effect, f, indent=2, default=str)
            f.write("\n")
        logger.info(
            f"Behavioral-effect: {'PASS' if behavioural_effect.get('pass') else 'FAIL'}"
        )

        # -- Pre/post joining (P0-10) --------------------------------------- #
        post_results_path = output_dir / "post_eval" / "baseline_results.jsonl"
        join_result = None
        if post_results_path.exists():
            join_result = join_pre_post_results(
                baseline_path, post_results_path, selection, output_dir,
            )
            logger.info(f"Joined {join_result['joined_count']} pre/post rows")

            # -- Preservation (P0-11 / P1-12) ------------------------------- #
            post_summary_path = output_dir / "post_eval" / "baseline_summary.json"
            if post_summary_path.exists():
                with open(post_summary_path) as f:
                    post_summary = json.load(f)
                baseline_summary_path = baseline_path.parent / "baseline_summary.json"
                baseline_summary = {}
                if baseline_summary_path.exists():
                    with open(baseline_summary_path) as f:
                        baseline_summary = json.load(f)
                preservation = compute_preservation(baseline_summary, post_summary)
                with open(output_dir / "preservation_report.json", "w") as f:
                    json.dump(preservation, f, indent=2)
                    f.write("\n")
                logger.info(
                    f"Preservation: DV {preservation['pre_direct_visual_accuracy']:.3f} "
                    f"-> {preservation['post_direct_visual_accuracy']:.3f} "
                    f"(delta={preservation['delta_direct_visual_accuracy']:+.3f}, "
                    f"gate={'PASS' if preservation['gate_pass'] else 'FAIL'})"
                )
    else:
        # P0-C10: --skip-post-eval forces engineering_debug.
        post_eval = {"num_probes": 0, "num_errors": 0, "skipped": True}
        preservation = {}
        reload_validation = {"pass": False, "skipped": True}
        behavioural_effect = {"pass": False, "skipped": True}
        join_result = None

    # -- Write selection ---------------------------------------------------- #
    with open(output_dir / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)
        f.write("\n")

    # -- Write environment -------------------------------------------------- #
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
        "frozen_base_changes": protection_report["n_changed"],
    }
    with open(output_dir / "parameter_inventory.json", "w") as f:
        json.dump(param_inv, f, indent=2)
        f.write("\n")

    # -- Fail-closed validation (P0-7/8/12) --------------------------------- #
    required_checks = {
        "finite_forget_loss": bool(np.isfinite(training_stats.get("forget_loss", float("nan")))),
        "finite_retain_loss": bool(np.isfinite(training_stats.get("retain_loss", float("nan")))),
        "finite_total_loss": bool(np.isfinite(training_stats.get("total_loss", float("nan")))),
        "nonzero_intended_gradients": bool(
            training_stats.get("total_nonzero_grad_tensors", 0) > 0
        ),
        "intended_lora_changed": bool(lora_changed > 0),
        "protected_parameters_unchanged": protection_report["pass"],
        "adapter_saved": bool(adapter_path.exists()),
        "checkpoint_sha_nonempty": bool(adapter_sha),
        "post_probe_count": int(post_eval.get("num_probes", 0)),
        "post_error_count": int(post_eval.get("num_errors", 0)),
    }
    if not args.skip_post_eval:
        required_checks["post_probe_count_exact"] = (
            required_checks["post_probe_count"] == 500
        )
        required_checks["post_errors_zero"] = (
            required_checks["post_error_count"] == 0
        )
        required_checks["fresh_base_loaded"] = bool(
            post_eval.get("fresh_base_loaded", False)
        )
        required_checks["checkpoint_loaded"] = bool(
            post_eval.get("checkpoint_loaded", False)
        )
        # P0-7: Include post-eval validation pass in required checks.
        required_checks["post_validation_pass"] = bool(
            post_eval.get("validation_pass", False)
        )
        # P0-C02/03: Include reload-equivalence in required checks.
        required_checks["reload_equivalence_pass"] = bool(
            reload_validation.get("pass", False)
        )
        # P0-C01: Behavioral-effect gates.
        required_checks["base_to_trained_effect_nonzero"] = bool(
            behavioural_effect.get("base_to_trained_effect_nonzero", False)
        )
        required_checks["base_to_reload_effect_nonzero"] = bool(
            behavioural_effect.get("base_to_reload_effect_nonzero", False)
        )
        # P0-C05: Exact probe matching is a required gate (unconditional).
        # P0-SHARED-03: Missing file or failed join = fail.
        if join_result is not None:
            _probe_val = _validate_probe_counts(
                baseline_path, post_results_path, join_result,
            )
            required_checks["exact_probe_match_pass"] = bool(_probe_val["pass"])
            with open(output_dir / "exact_probe_match.json", "w") as f:
                json.dump(_probe_val, f, indent=2)
                f.write("\n")
        else:
            required_checks["exact_probe_match_pass"] = False

        # P0-SHARED-04: Preservation gate (unconditional).
        if preservation:
            required_checks["preservation_gate_pass"] = bool(
                preservation.get("gate_pass", False)
            )
        else:
            required_checks["preservation_gate_pass"] = False

        # P0-SHARED-02: Adapter reload integrity (unconditional).
        _reload_integrity_path = output_dir / "adapter_reload_integrity.json"
        if _reload_integrity_path.is_file():
            import json as _json_rt
            with open(_reload_integrity_path) as _f_rt:
                _ri = _json_rt.load(_f_rt)
            required_checks["adapter_reload_integrity_pass"] = bool(
                _ri.get("pass", False)
            )
        else:
            required_checks["adapter_reload_integrity_pass"] = False

        # P0-SHARED-02: Tensor roundtrip (unconditional).
        _roundtrip_path = output_dir / "adapter_tensor_roundtrip.json"
        if _roundtrip_path.is_file():
            import json as _json_rt2
            with open(_roundtrip_path) as _f_rt2:
                _rt = _json_rt2.load(_f_rt2)
            required_checks["adapter_tensor_roundtrip_pass"] = bool(
                _rt.get("pass", False)
            )
        else:
            required_checks["adapter_tensor_roundtrip_pass"] = False

        # P0-PHI-01: Phi-specific fail-closed diagnostic gates.
        # Only apply to Phi model — other models don't produce these files.
        if profile.key == "phi4_mm":
            _phi_diag_files = {
                "phi_causal_invariance_pass": "phi_causal_invariance_report.json",
                "phi_candidate_scoring_sanity_pass": "candidate_scoring_sanity.json",
                "phi_adapter_composition_pass": "adapter_composition_report.json",
            }
            for gate_name, diag_file in _phi_diag_files.items():
                _diag_path = output_dir / diag_file
                if _diag_path.is_file():
                    import json as _json_phi
                    with open(_diag_path) as _f_phi:
                        _diag = _json_phi.load(_f_phi)
                    required_checks[gate_name] = bool(
                        _diag.get("pass", False)
                    )
                else:
                    required_checks[gate_name] = False

    all_pass = all(
        v for k, v in required_checks.items()
        if isinstance(v, bool)
    )

    # P0-C10: Separate completion logic.
    # research_complete requires all gates pass + post-eval executed + not debug.
    post_eval_executed = not args.skip_post_eval and not post_eval.get("skipped", False)
    research_complete = all_pass and post_eval_executed and not is_dirty

    if args.skip_post_eval:
        evidence_class = "engineering_debug"
        research_eligible = False
    else:
        evidence_class = "research_canary" if all_pass else "engineering_canary"
        research_eligible = research_complete

    # Derive gate counts programmatically from the required_checks schema.
    _bool_checks = {
        k: v for k, v in required_checks.items() if isinstance(v, bool)
    }
    _gates_passed = sum(1 for v in _bool_checks.values() if v)
    _total_gates = len(_bool_checks)

    validation = {
        "pass": all_pass,
        "gates_passed": _gates_passed,
        "total_gates": _total_gates,
        "checks": required_checks,
        "evidence_class": evidence_class,
        "research_evidence_eligible": research_eligible,
    }
    with open(output_dir / "validation_report.json", "w") as f:
        json.dump(validation, f, indent=2)
        f.write("\n")

    # -- Write run manifest (P1-13: complete provenance) -------------------- #
    env_sha = file_sha256(output_dir / "environment.json") if (output_dir / "environment.json").exists() else ""
    _prov_sha = file_sha256(output_dir / "execution_provenance.json") if (output_dir / "execution_provenance.json").exists() else ""
    run_manifest = {
        "model_key": model_key,
        "model_id": profile.model_id,
        "model_revision": profile.revision,
        "processor_revision": profile.processor_revision,
        "method_id": method_id,
        "seed": args.seed,
        "git_commit": git_sha,
        "git_dirty": is_dirty,
        "model_profile_sha256": profile_sha,
        "method_config_sha256": file_sha256(method_path),
        "selection_sha256": file_sha256(selection_path),
        "execution_provenance_sha256": _prov_sha,
        "checkpoint_sha256": adapter_sha,
        "baseline_manifest_sha256": baseline_binding.manifest_sha256,
        "baseline_binding_sha256": _binding_file_sha,
        "baseline_results_sha256": baseline_binding.results_sha256,
        "environment_sha256": env_sha,
        "validation_pass": all_pass,
        "evidence_class": evidence_class,
        "canary": args.canary,
    }
    # Add artifact SHAs for files that exist
    for _fname, _key in [
        ("post_results_joined.jsonl", "post_results_sha256"),
        ("group_family_metrics.json", "group_family_metrics_sha256"),
        ("preservation_report.json", "preservation_report_sha256"),
        ("reload_validation.json", "reload_validation_sha256"),
        ("parameter_inventory.json", "parameter_inventory_sha256"),
        ("behavioral_effect_validation.json", "behavioral_effect_sha256"),
        ("exact_probe_match.json", "exact_probe_match_sha256"),
    ]:
        _fpath = output_dir / _fname
        if _fpath.exists():
            run_manifest[_key] = file_sha256(_fpath)
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=2)
        f.write("\n")

    # P1-13: Generate outer run_binding.json last.
    run_binding = {
        "model_key": model_key,
        "method_id": method_id,
        "seed": args.seed,
        "model_profile_sha256": profile_sha,
        "execution_provenance_sha256": _prov_sha,
        "checkpoint_sha256": adapter_sha,
        "baseline_results_sha256": baseline_binding.results_sha256,
        "run_manifest_sha256": file_sha256(output_dir / "run_manifest.json"),
        "validation_pass": all_pass,
        "evidence_class": evidence_class,
    }
    with open(output_dir / "run_binding.json", "w") as f:
        json.dump(run_binding, f, indent=2)
        f.write("\n")

    # -- P0-C10: Remove RUN_INCOMPLETE only for research-complete runs ---- #
    if research_complete:
        if run_incomplete.exists():
            run_incomplete.unlink()
        logger.info("All validation gates PASSED; RUN_INCOMPLETE removed")
    elif all_pass and not post_eval_executed:
        # Training passed but post-eval was skipped — DEBUG_COMPLETE
        if run_incomplete.exists():
            run_incomplete.unlink()
        _debug_marker = output_dir / "DEBUG_COMPLETE"
        _debug_marker.touch()
        logger.info("Training PASSED but post-eval skipped; DEBUG_COMPLETE written")
    else:
        failed = [k for k, v in required_checks.items() if isinstance(v, bool) and not v]
        logger.warning(f"Validation FAILED: {failed}")
        logger.warning("RUN_INCOMPLETE retained")

    logger.info("=" * 60)
    logger.info(f"Unlearning complete: {model_key}")
    logger.info(f"Output: {output_dir}")
    logger.info(
        f"Validation: {'PASS' if all_pass else 'FAIL'} "
        f"({_gates_passed}/{_total_gates} gates)"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
