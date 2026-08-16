"""Minimal LoRA-based unlearning harness for the Stage 3 pilot.

This module implements a targeted intervention that selectively weakens
identity-conditioned behavior while preserving general visual capability.

Public API
----------
.. autoclass:: UnlearningConfig
.. autoclass:: UnlearningTrainer
.. autofunction:: load_base_model
.. autofunction:: apply_lora
.. autofunction:: build_forget_dataset
.. autofunction:: build_retain_dataset
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class UnlearningConfig:
    """Configuration for the unlearning pilot."""

    # Base model
    model_id: str = "Qwen/Qwen3.5-9B"
    model_revision: str = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    seed: int = 17

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Training
    learning_rate: float = 1e-4
    num_steps: int = 50
    retain_weight: float = 0.1
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 5

    # Data
    forget_identity_ids: list[str] = field(default_factory=list)
    retain_identity_ids: list[str] = field(default_factory=list)
    processed_dataset_path: str = ""
    route_probe_path: str = ""

    # Output
    output_dir: str = "outputs/experiments/unlearning_pilot/Qwen_Qwen3.5-9B/pilot_v1"
    checkpoint_steps: list[int] = field(default_factory=lambda: [0, 10, 25, 50])

    # Provenance
    selection_manifest_sha256: str = ""
    code_commit: str = ""

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


# --------------------------------------------------------------------------- #
# Dataset builders
# --------------------------------------------------------------------------- #

class ForgetDataset(Dataset):
    """Dataset of samples from target identities to unlearn."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        processor: Any,
        max_length: int = 512,
    ):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image_uri = sample["image_uri"]
        question = sample["question"]

        # Build chat messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        try:
            prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
            )

        # Load and process image
        from PIL import Image
        image = Image.open(image_uri).convert("RGB")

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs.get("pixel_values", torch.zeros(1, 3, 224, 224)),
        }


class RetainDataset(Dataset):
    """Dataset of samples from retain identities or general visual examples."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        processor: Any,
        max_length: int = 512,
    ):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image_uri = sample["image_uri"]
        question = sample["question"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        try:
            prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
            )

        from PIL import Image
        image = Image.open(image_uri).convert("RGB")

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs.get("pixel_values", torch.zeros(1, 3, 224, 224)),
        }


def build_forget_dataset(
    processed_dataset_path: str | Path,
    target_identity_ids: list[str],
    processor: Any,
    *,
    max_samples: int | None = None,
    seed: int = 17,
) -> ForgetDataset:
    """Build a dataset of samples from target identities.

    Parameters
    ----------
    processed_dataset_path:
        Path to fiubench_processed.jsonl.
    target_identity_ids:
        List of identity IDs to unlearn.
    processor:
        Qwen processor for tokenization and image processing.
    max_samples:
        Optional cap on dataset size.
    seed:
        Random seed for sampling.
    """
    rng = random.Random(seed)
    target_set = set(target_identity_ids)

    samples: list[dict[str, Any]] = []
    with open(processed_dataset_path) as fh:
        for line in fh:
            row = json.loads(line)
            if row["identity_id"] in target_set:
                samples.append(row)

    rng.shuffle(samples)
    if max_samples is not None:
        samples = samples[:max_samples]

    logger.info(f"Forget dataset: {len(samples)} samples from {len(target_set)} identities")
    return ForgetDataset(samples, processor)


def build_retain_dataset(
    processed_dataset_path: str | Path,
    retain_identity_ids: list[str],
    processor: Any,
    *,
    max_samples: int | None = None,
    seed: int = 17,
) -> RetainDataset:
    """Build a dataset of samples from retain identities.

    Parameters
    ----------
    processed_dataset_path:
        Path to fiubench_processed.jsonl.
    retain_identity_ids:
        List of identity IDs to preserve.
    processor:
        Qwen processor for tokenization and image processing.
    max_samples:
        Optional cap on dataset size.
    seed:
        Random seed for sampling.
    """
    rng = random.Random(seed)
    retain_set = set(retain_identity_ids)

    samples: list[dict[str, Any]] = []
    with open(processed_dataset_path) as fh:
        for line in fh:
            row = json.loads(line)
            if row["identity_id"] in retain_set:
                samples.append(row)

    rng.shuffle(samples)
    if max_samples is not None:
        samples = samples[:max_samples]

    logger.info(f"Retain dataset: {len(samples)} samples from {len(retain_set)} identities")
    return RetainDataset(samples, processor)


# --------------------------------------------------------------------------- #
# Model loading and LoRA setup
# --------------------------------------------------------------------------- #

def load_base_model(
    model_id: str,
    revision: str,
    dtype: str = "bfloat16",
    device: str = "cuda:0",
) -> tuple[Any, Any]:
    """Load the base Qwen model and processor.

    Returns
    -------
    model : AutoModelForImageTextToText
        The base model in eval mode.
    processor : AutoProcessor
        The processor for tokenization and image processing.
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_dtype = getattr(torch, dtype)
    snapshot_download(model_id, revision=revision)

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        revision=revision,
        device_map=device,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )

    model.eval()
    logger.info(f"Loaded base model {model_id} revision {revision}")
    return model, processor


def apply_lora(
    model: Any,
    *,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> Any:
    """Apply LoRA adapters to the model.

    Parameters
    ----------
    model : PreTrainedModel
        The base model.
    r : int
        LoRA rank.
    lora_alpha : int
        LoRA alpha scaling.
    lora_dropout : float
        Dropout for LoRA layers.
    target_modules : list[str]
        Module names to apply LoRA to.

    Returns
    -------
    model : PeftModel
        The model with LoRA adapters attached.
    """
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=None,  # Custom task
    )

    model = get_peft_model(model, lora_config)
    logger.info(f"Applied LoRA: r={r}, alpha={lora_alpha}, targets={target_modules}")
    return model


# --------------------------------------------------------------------------- #
# Loss functions
# --------------------------------------------------------------------------- #

def compute_forget_loss(
    model: Any,
    batch: dict[str, torch.Tensor],
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the forget loss (negative log-likelihood on target samples).

    For unlearning, we want to *increase* the loss on forget samples,
    so we return the negative of the standard LM loss.
    """
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch.get("pixel_values"),
        labels=batch["input_ids"] if labels is None else labels,
    )
    # Return negative loss so that minimizing this *increases* the LM loss
    return -outputs.loss


def compute_retain_loss(
    model: Any,
    batch: dict[str, torch.Tensor],
    reference_model: Any | None = None,
) -> torch.Tensor:
    """Compute the retain loss (standard LM loss on retain samples).

    If reference_model is provided, use KL divergence instead.
    """
    if reference_model is not None:
        # KL divergence to frozen reference
        with torch.no_grad():
            ref_outputs = reference_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values"),
            )
            ref_logits = ref_outputs.logits

        curr_outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"),
        )
        curr_logits = curr_outputs.logits

        # KL divergence: sum over vocab, mean over sequence
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(curr_logits, dim=-1),
            torch.softmax(ref_logits, dim=-1),
            reduction="batchmean",
        )
        return kl
    else:
        # Standard LM loss
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"),
            labels=batch["input_ids"],
        )
        return outputs.loss


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

class UnlearningTrainer:
    """Minimal training loop for the unlearning pilot."""

    def __init__(
        self,
        config: UnlearningConfig,
        model: Any,
        processor: Any,
        forget_dataset: Dataset,
        retain_dataset: Dataset,
        reference_model: Any | None = None,
    ):
        self.config = config
        self.model = model
        self.processor = processor
        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset
        self.reference_model = reference_model

        # Freeze reference if provided
        if self.reference_model is not None:
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False

        # Data loaders
        self.forget_loader = DataLoader(
            forget_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.retain_loader = DataLoader(
            retain_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # Training state
        self.global_step = 0
        self.training_log: list[dict[str, float]] = []

        # Move model to device
        self.device = torch.device(config.device)

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move batch tensors to the training device."""
        return {k: v.to(self.device) for k, v in batch.items()}

    def train(self) -> dict[str, Any]:
        """Run the training loop.

        Returns
        -------
        training_summary : dict
            Summary of training including final losses and checkpoint paths.
        """
        self.model.train()
        self.model.to(self.device)

        forget_iter = iter(self.forget_loader)
        retain_iter = iter(self.retain_loader)

        total_steps = self.config.num_steps
        checkpoint_steps = set(self.config.checkpoint_steps)

        # Save initial checkpoint if step 0 is requested
        if 0 in checkpoint_steps:
            self._save_checkpoint("step_000")

        logger.info(f"Starting training for {total_steps} steps")
        start_time = time.time()

        for step in range(1, total_steps + 1):
            # Get batches (cycle through datasets if needed)
            try:
                forget_batch = next(forget_iter)
            except StopIteration:
                forget_iter = iter(self.forget_loader)
                forget_batch = next(forget_iter)

            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(self.retain_loader)
                retain_batch = next(retain_iter)

            forget_batch = self._move_batch(forget_batch)
            retain_batch = self._move_batch(retain_batch)

            # Compute losses
            forget_loss = compute_forget_loss(self.model, forget_batch)
            retain_loss = compute_retain_loss(
                self.model, retain_batch, self.reference_model
            )

            total_loss = forget_loss + self.config.retain_weight * retain_loss

            # Scale loss for gradient accumulation
            scaled_loss = total_loss / self.config.gradient_accumulation_steps
            scaled_loss.backward()

            # Optimizer step
            if step % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Log
                elapsed = time.time() - start_time
                log_entry = {
                    "step": step,
                    "total_loss": total_loss.item(),
                    "forget_loss": forget_loss.item(),
                    "retain_loss": retain_loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": elapsed,
                }
                self.training_log.append(log_entry)
                logger.info(
                    f"Step {step}/{total_steps} | "
                    f"loss={total_loss.item():.4f} "
                    f"forget={forget_loss.item():.4f} "
                    f"retain={retain_loss.item():.4f}"
                )

                # Safety checks
                if not math.isfinite(total_loss.item()):
                    raise RuntimeError(f"Non-finite loss at step {step}")

                # Checkpoint
                if step in checkpoint_steps:
                    ckpt_name = f"step_{step:03d}"
                    self._save_checkpoint(ckpt_name)

        # Save final checkpoint
        if "final" not in [f"step_{s:03d}" for s in checkpoint_steps]:
            self._save_checkpoint("final")

        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed:.1f}s")

        return {
            "total_steps": total_steps,
            "final_loss": self.training_log[-1]["total_loss"] if self.training_log else None,
            "elapsed_seconds": elapsed,
            "checkpoints_saved": len(checkpoint_steps) + 1,
        }

    def _save_checkpoint(self, name: str) -> Path:
        """Save a checkpoint with the given name."""
        ckpt_dir = Path(self.config.output_dir) / "checkpoints" / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save adapter weights only
        self.model.save_pretrained(ckpt_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
            "config": {
                "learning_rate": self.config.learning_rate,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
            },
        }
        torch.save(state, ckpt_dir / "training_state.pt")

        logger.info(f"Saved checkpoint: {ckpt_dir}")
        return ckpt_dir


# --------------------------------------------------------------------------- #
# Provenance and reporting
# --------------------------------------------------------------------------- #

def generate_trainable_parameter_report(model: Any) -> dict[str, Any]:
    """Generate a report of trainable parameters.

    Returns
    -------
    report : dict
        Includes total/trainable counts, percentage, and module names.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_modules = []
    for name, param in model.named_parameters():
        if param.requires_grad and name not in trainable_modules:
            # Extract module prefix (e.g., "base_model.model.layers.0.self_attn.q_proj")
            module_name = ".".join(name.split(".")[:-1])
            if module_name not in trainable_modules:
                trainable_modules.append(module_name)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": 100.0 * trainable_params / total_params if total_params > 0 else 0.0,
        "trainable_module_count": len(trainable_modules),
        "trainable_modules": sorted(trainable_modules)[:50],  # Cap at 50 for readability
    }


def generate_run_manifest(
    config: UnlearningConfig,
    training_summary: dict[str, Any],
    param_report: dict[str, Any],
    code_commit: str,
    git_dirty: bool,
) -> dict[str, Any]:
    """Generate the unlearning run manifest.

    Returns
    -------
    manifest : dict
        Full provenance record for the training run.
    """
    return {
        "experiment_id": "fiubench_unlearning_pilot_v1",
        "base_model": {
            "model_id": config.model_id,
            "revision": config.model_revision,
            "dtype": config.dtype,
        },
        "selection_manifest_sha256": config.selection_manifest_sha256,
        "method": {
            "name": "lora_targeted_update",
            "hyperparameters": {
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout,
                "lora_target_modules": config.lora_target_modules,
                "learning_rate": config.learning_rate,
                "num_steps": config.num_steps,
                "retain_weight": config.retain_weight,
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
            },
        },
        "seed": config.seed,
        "forget_identities": config.forget_identity_ids,
        "retain_identities": config.retain_identity_ids,
        "training_summary": training_summary,
        "trainable_parameters": param_report,
        "code_provenance": {
            "git_commit": code_commit,
            "git_dirty": git_dirty,
        },
        "output_dir": config.output_dir,
    }


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
