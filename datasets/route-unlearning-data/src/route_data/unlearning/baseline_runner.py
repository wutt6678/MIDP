"""Unified baseline training runner.

This module provides a training loop that accepts any UnlearningObjective
and handles checkpointing, trace logging, and manifest writing. It reuses
the validated MIDP infrastructure (datasets, collator, LoRA, model loading)
from the existing unlearning_harness.

Public API
----------
.. autoclass:: BaselineTrainingConfig
.. autoclass:: BaselineTrainer
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..eval.unlearning_harness import (
    qwen_collate_fn,
)
from .objectives import (
    GradientAscent,
    GradientDifference,
    KLMinimization,
    NegativePreferenceOptimization,
    RetainOnlyCE,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class BaselineTrainingConfig:
    """Configuration for baseline training runs."""

    # Method
    method_name: str = "mllmu_ga"

    # Base model
    model_id: str = "Qwen/Qwen3.5-9B"
    model_revision: str = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    seed: int = 17

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Training
    learning_rate: float = 2e-5
    num_optimizer_steps: int = 125
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 0

    # Method-specific
    retain_weight: float = 1.0  # For GD
    kl_temperature: float = 1.0  # For KL
    kl_weight: float = 1.0  # For KL
    include_retain_ce: bool = False  # For KL
    npo_beta: float = 0.9  # For NPO (paper value, NOT repo default 0.4)
    retain_only: bool = False  # For npo_oracle: train on retain data only

    # Data
    forget_identity_ids: list[str] = field(default_factory=list)
    retain_identity_ids: list[str] = field(default_factory=list)
    processed_dataset_path: str = ""

    # Output
    output_dir: str = ""
    checkpoint_steps: list[int] = field(
        default_factory=lambda: [1, 5, 10, 25, 50, 60, 75, 90, 125]
    )

    # Reference/oracle model paths (for KL and NPO)
    reference_model_path: str = ""
    oracle_adapter_path: str = ""

    # Provenance
    selection_manifest_sha256: str = ""
    code_commit: str = ""

    # Profile-driven provenance (P0-8)
    model_key: str = ""
    processor_id: str = ""
    processor_revision: str = ""
    model_profile_sha256: str = ""
    adapter_family: str = ""
    lora_target_inventory_sha256: str = ""

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


# --------------------------------------------------------------------------- #
# Dummy loader for retain-only training
# --------------------------------------------------------------------------- #

class _DummyLoader:
    """Yields empty dicts for retain-only training (no forget dataset)."""

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Any]:
        return {}


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

class BaselineTrainer:
    """Unified training loop for MLLMU-Bench baselines.

    Accepts any UnlearningObjective and handles:
    - Deterministic seeding
    - Gradient accumulation
    - Checkpointing at specified steps
    - Training trace logging (JSONL)
    - Manifest writing
    - Numerical safety checks
    """

    def __init__(
        self,
        config: BaselineTrainingConfig,
        objective: Any,
        model: Any,
        processor: Any,
        forget_dataset: Any,
        retain_dataset: Any | None = None,
        reference_model: Any | None = None,
        oracle_model: Any | None = None,
        adapter: Any | None = None,
    ):
        self.config = config
        self.objective = objective
        self.model = model
        self.processor = processor
        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset
        self.reference_model = reference_model
        self.oracle_model = oracle_model
        self.adapter = adapter

        # Collator selection (P0-1): adapter-aware or legacy Qwen fallback
        if self.adapter is not None:
            collate_fn = self.adapter.collate
        else:
            collate_fn = qwen_collate_fn

        # Freeze reference/oracle models
        if self.reference_model is not None:
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False

        if self.oracle_model is not None:
            self.oracle_model.eval()
            for param in self.oracle_model.parameters():
                param.requires_grad = False

        # Deterministic seeding
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        # Data loaders
        if forget_dataset is not None:
            self.forget_loader = DataLoader(
                forget_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,
                generator=self.generator,
                collate_fn=collate_fn,
            )
        else:
            # Retain-only mode: dummy forget loader yielding empty dicts
            self.forget_loader = _DummyLoader()

        self.retain_loader = None
        if retain_dataset is not None:
            self.retain_loader = DataLoader(
                retain_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,
                generator=self.generator,
                collate_fn=collate_fn,
            )

        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # Warmup LR scheduler
        if config.warmup_steps > 0:
            def lr_lambda(step: int) -> float:
                if step < config.warmup_steps:
                    return (step + 1) / config.warmup_steps
                return 1.0
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda,
            )
        else:
            self.scheduler = None

        # Training state
        self.global_step = 0
        self.training_log: list[dict[str, float]] = []

        # Device
        self.device = torch.device(config.device)

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Trace file
        self.trace_path = self.output_dir / "training_trace.jsonl"

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move batch tensors to the training device."""
        result: dict[str, Any] = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result

    def train(self) -> dict[str, Any]:
        """Run the training loop.

        Returns
        -------
        training_summary:
            Summary dict with final losses, checkpoint paths, etc.
        """
        self.model.train()
        self.model.to(self.device)

        forget_iter = iter(self.forget_loader)
        retain_iter = iter(self.retain_loader) if self.retain_loader else None

        num_opt_steps = self.config.num_optimizer_steps
        grad_accum = self.config.gradient_accumulation_steps
        checkpoint_steps = set(self.config.checkpoint_steps)

        total_microsteps = num_opt_steps * grad_accum
        logger.info(
            f"Starting {self.config.method_name} training for {num_opt_steps} "
            f"optimizer steps ({total_microsteps} microbatches, "
            f"grad_accum={grad_accum})"
        )
        start_time = time.time()

        # Open trace file
        with open(self.trace_path, "w") as trace_fh:
            for opt_step in range(1, num_opt_steps + 1):
                self.optimizer.zero_grad()
                accum_total = 0.0
                accum_forget = 0.0
                accum_retain = 0.0
                accum_kl = 0.0
                accum_npo = 0.0

                for micro_idx in range(grad_accum):
                    # Get forget batch
                    try:
                        forget_batch = next(forget_iter)
                    except StopIteration:
                        forget_iter = iter(self.forget_loader)
                        forget_batch = next(forget_iter)
                    forget_batch = self._move_batch(forget_batch)

                    # Get retain batch if needed
                    retain_batch = None
                    if retain_iter is not None:
                        try:
                            retain_batch = next(retain_iter)
                        except StopIteration:
                            retain_iter = iter(self.retain_loader)
                            retain_batch = next(retain_iter)
                        retain_batch = self._move_batch(retain_batch)

                    # Compute loss via objective
                    loss_dict = self.objective.compute_loss(
                        model=self.model,
                        forget_batch=forget_batch,
                        retain_batch=retain_batch,
                        reference_model=self.reference_model,
                        oracle_model=self.oracle_model,
                    )

                    total_loss = loss_dict["total_loss"]

                    # Numerical safety checks BEFORE backward
                    if not torch.isfinite(total_loss):
                        self._write_failure_metadata(
                            opt_step, micro_idx, "non_finite_total_loss"
                        )
                        raise RuntimeError(
                            f"Non-finite total loss at step {opt_step}, "
                            f"micro {micro_idx}: {total_loss.item()}"
                        )

                    # Scale for gradient accumulation
                    scaled_loss = total_loss / grad_accum
                    scaled_loss.backward()

                    accum_total += total_loss.item()
                    if loss_dict.get("forget_loss") is not None:
                        accum_forget += loss_dict["forget_loss"].item()
                    if loss_dict.get("retain_loss") is not None:
                        accum_retain += loss_dict["retain_loss"].item()
                    if loss_dict.get("kl_loss") is not None:
                        accum_kl += loss_dict["kl_loss"].item()
                    if loss_dict.get("npo_loss") is not None:
                        accum_npo += loss_dict["npo_loss"].item()

                # Gradient norm check BEFORE optimizer step
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                if not torch.isfinite(grad_norm):
                    self._write_failure_metadata(
                        opt_step, None, "non_finite_gradient_norm"
                    )
                    raise RuntimeError(
                        f"Non-finite gradient norm at step {opt_step}: {grad_norm}"
                    )

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step = opt_step

                # Average losses
                avg_total = accum_total / grad_accum
                avg_forget = accum_forget / grad_accum
                avg_retain = accum_retain / grad_accum
                avg_kl = accum_kl / grad_accum
                avg_npo = accum_npo / grad_accum

                elapsed = time.time() - start_time

                # Log entry
                log_entry = {
                    "method": self.config.method_name,
                    "optimizer_step": opt_step,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "total_loss": avg_total,
                    "forget_ce": avg_forget,
                    "retain_ce": avg_retain,
                    "kl_loss": avg_kl,
                    "npo_loss": avg_npo,
                    "gradient_norm": float(grad_norm.item()),
                    "finite": True,
                    "elapsed_seconds": elapsed,
                }
                self.training_log.append(log_entry)

                # Write trace row
                trace_fh.write(json.dumps(log_entry) + "\n")
                trace_fh.flush()

                logger.info(
                    f"Step {opt_step}/{num_opt_steps} | "
                    f"loss={avg_total:.4f} "
                    f"forget={avg_forget:.4f} "
                    f"retain={avg_retain:.4f} "
                    f"kl={avg_kl:.4f} "
                    f"npo={avg_npo:.4f} "
                    f"grad_norm={grad_norm:.4f}"
                )

                # Post-step safety check
                if not math.isfinite(avg_total):
                    self._write_failure_metadata(
                        opt_step, None, "non_finite_avg_loss_post_step"
                    )
                    raise RuntimeError(
                        f"Non-finite loss at optimizer step {opt_step}"
                    )

                # Checkpoint
                if opt_step in checkpoint_steps:
                    ckpt_name = f"optimizer_step_{opt_step:03d}"
                    self._save_checkpoint(ckpt_name)

            # Save final checkpoint if not already saved
            if num_opt_steps not in checkpoint_steps:
                self._save_checkpoint(f"optimizer_step_{num_opt_steps:03d}_final")

        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed:.1f}s")

        # Write manifest
        self._write_manifest(elapsed)

        return {
            "method": self.config.method_name,
            "num_optimizer_steps": num_opt_steps,
            "gradient_accumulation_steps": grad_accum,
            "total_microsteps": total_microsteps,
            "final_loss": self.training_log[-1]["total_loss"] if self.training_log else None,
            "elapsed_seconds": elapsed,
            "checkpoints_saved": len(checkpoint_steps) + 1,
        }

    def _save_checkpoint(self, name: str) -> Path:
        """Save a checkpoint."""
        ckpt_dir = self.output_dir / "checkpoints" / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save adapter weights
        self.model.save_pretrained(ckpt_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
            "config": {
                "method_name": self.config.method_name,
                "learning_rate": self.config.learning_rate,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
            },
        }
        torch.save(state, ckpt_dir / "training_state.pt")

        logger.info(f"Saved checkpoint: {ckpt_dir}")
        return ckpt_dir

    def _write_manifest(self, elapsed: float) -> None:
        """Write the training manifest with full provenance."""
        manifest = {
            "method": self.config.method_name,
            "base_model": {
                "model_id": self.config.model_id,
                "revision": self.config.model_revision,
                "dtype": self.config.dtype,
            },
            "model_profile": {
                "model_key": self.config.model_key,
                "processor_id": self.config.processor_id,
                "processor_revision": self.config.processor_revision,
                "model_profile_sha256": self.config.model_profile_sha256,
                "adapter_family": self.config.adapter_family,
                "lora_target_inventory_sha256": self.config.lora_target_inventory_sha256,
            },
            "lora": {
                "rank": self.config.lora_rank,
                "alpha": self.config.lora_alpha,
                "dropout": self.config.lora_dropout,
                "target_modules": self.config.lora_target_modules,
            },
            "training": {
                "seed": self.config.seed,
                "learning_rate": self.config.learning_rate,
                "num_optimizer_steps": self.config.num_optimizer_steps,
                "batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "max_grad_norm": self.config.max_grad_norm,
            },
            "method_hyperparameters": {},
            "selection_manifest_sha256": self.config.selection_manifest_sha256,
            "code_commit": self.config.code_commit,
            "elapsed_seconds": elapsed,
            "checkpoint_steps": self.config.checkpoint_steps,
            "output_dir": str(self.output_dir),
        }

        # Add method-specific hyperparameters
        if self.config.method_name == "mllmu_ga_difference":
            manifest["method_hyperparameters"]["retain_weight"] = self.config.retain_weight
        elif self.config.method_name == "mllmu_kl_min":
            manifest["method_hyperparameters"]["kl_weight"] = self.config.kl_weight
            manifest["method_hyperparameters"]["temperature"] = self.config.kl_temperature
            manifest["method_hyperparameters"]["include_retain_ce"] = self.config.include_retain_ce
        elif self.config.method_name == "npo_oracle":
            manifest["method_hyperparameters"]["retain_only"] = True
            manifest["method_hyperparameters"]["description"] = "Oracle training on retain data only for NPO"
        elif self.config.method_name == "mllmu_npo":
            manifest["method_hyperparameters"]["beta"] = self.config.npo_beta
            manifest["method_hyperparameters"]["upstream_paper_beta"] = 0.9
            manifest["method_hyperparameters"]["upstream_repo_default_beta"] = 0.4
            manifest["method_hyperparameters"]["chosen_beta"] = self.config.npo_beta

        manifest_path = self.output_dir / "baseline_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")

        logger.info(f"Wrote manifest: {manifest_path}")

    def _write_failure_metadata(
        self, opt_step: int, micro_idx: int | None, reason: str,
    ) -> None:
        """Write failure metadata on numerical failure."""
        failure = {
            "reason": reason,
            "optimizer_step": opt_step,
            "micro_idx": micro_idx,
            "timestamp": time.time(),
            "training_log_entries": len(self.training_log),
        }
        path = self.output_dir / "failure_metadata.json"
        with open(path, "w") as f:
            json.dump(failure, f, indent=2)
        logger.error(f"Training aborted: {reason} at step {opt_step}")


# --------------------------------------------------------------------------- #
# Factory functions
# --------------------------------------------------------------------------- #

def build_objective(config: BaselineTrainingConfig) -> Any:
    """Build the appropriate objective from config.

    Parameters
    ----------
    config:
        Training configuration.

    Returns
    -------
    objective:
        An UnlearningObjective instance.
    """
    if config.method_name == "mllmu_ga":
        return GradientAscent()
    elif config.method_name == "mllmu_ga_difference":
        return GradientDifference(retain_weight=config.retain_weight)
    elif config.method_name == "mllmu_kl_min":
        return KLMinimization(
            kl_weight=config.kl_weight,
            temperature=config.kl_temperature,
            include_retain_ce=config.include_retain_ce,
        )
    elif config.method_name == "npo_oracle":
        return RetainOnlyCE()
    elif config.method_name == "mllmu_npo":
        return NegativePreferenceOptimization(beta=config.npo_beta)
    else:
        raise ValueError(f"Unknown method: {config.method_name}")


def load_config_from_yaml(path: str | Path) -> BaselineTrainingConfig:
    """Load a BaselineTrainingConfig from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML config file.

    Returns
    -------
    config:
        BaselineTrainingConfig instance.
    """
    import yaml

    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Extract fields from nested structure
    method = raw.get("method", {})
    base_model = raw.get("base_model", {})
    training = raw.get("training", {})
    lora = raw.get("lora", {})
    runtime = raw.get("runtime", {})

    config = BaselineTrainingConfig(
        method_name=method.get("name", "mllmu_ga"),
        model_id=base_model.get("id", "Qwen/Qwen3.5-9B"),
        model_revision=base_model.get("revision", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
        dtype=base_model.get("dtype", "bfloat16"),
        device=runtime.get("device", "cuda:0"),
        seed=training.get("seed", 17),
        lora_rank=lora.get("rank", 8),
        lora_alpha=lora.get("alpha", 16),
        lora_dropout=lora.get("dropout", 0.0),
        learning_rate=training.get("learning_rate", 2e-5),
        num_optimizer_steps=training.get("max_optimizer_steps", 125),
        batch_size=training.get("batch_size", 1),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 4),
        max_grad_norm=training.get("max_grad_norm", 1.0),
        retain_weight=method.get("retain_weight", 1.0),
        kl_temperature=method.get("temperature", 1.0),
        kl_weight=method.get("kl_weight", 1.0),
        include_retain_ce=method.get("include_retain_ce", False),
        npo_beta=method.get("beta", 0.9),
        retain_only=training.get("retain_only", False),
        output_dir=runtime.get("output_dir", ""),
        checkpoint_steps=training.get("checkpoint_steps", [1, 5, 10, 25, 50, 60, 75, 90, 125]),
    )

    return config
