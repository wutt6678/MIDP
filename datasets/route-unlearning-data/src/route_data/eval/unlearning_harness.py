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

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Visual probe families that use signed_answer_margin
VISUAL_FAMILIES = frozenset({
    "direct_visual",
    "image_plus_name",
    "wrong_name",
    "visual_text_conflict",
})


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
    num_optimizer_steps: int = 50
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
    """Dataset of samples from target identities to unlearn.

    Each sample builds a full supervised chat example with user question
    and assistant Yes/No answer.  Prompt tokens are masked with -100 so
    that only the assistant answer tokens contribute to the forget loss.

    All processor tensor outputs are preserved (P0-2).
    """

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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image_uri = sample["image_uri"]
        question = sample["question"]
        answer_label = sample["answer_label"]  # bool: True=Yes, False=No
        answer_text = "Yes" if answer_label else "No"

        # -- Build user-only prompt to find assistant start position ------ #
        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        try:
            user_prompt = self.processor.apply_chat_template(
                user_messages,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            user_prompt = self.processor.apply_chat_template(
                user_messages,
                add_generation_prompt=True,
            )

        user_tokens = self.processor.tokenizer(
            user_prompt, return_tensors="pt", truncation=False,
        )
        assistant_start = user_tokens["input_ids"].shape[1]

        # -- Build full conversation with assistant answer ---------------- #
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text}],
            },
        ]

        try:
            full_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            full_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
            )

        # Load image
        from PIL import Image
        image = Image.open(image_uri).convert("RGB")

        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        # -- Preserve ALL processor tensor outputs (P0-2) --------------- #
        result: dict[str, Any] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                result[key] = value.squeeze(0)

        # Hard-fail if required multimodal tensors are missing (P0-2)
        if "pixel_values" not in result:
            raise RuntimeError(
                f"Required multimodal tensor 'pixel_values' missing from "
                f"processor output for sample idx={idx}. "
                f"Available keys: {sorted(inputs.keys())}"
            )

        # -- Build labels with prompt masking (P0-1) -------------------- #
        labels = result["input_ids"].clone()
        labels[:assistant_start] = -100
        result["labels"] = labels

        # -- Metadata for candidate-margin forget loss ------------------ #
        result["_answer_position"] = assistant_start
        result["_correct_answer_token_ids"] = self._get_answer_token_ids(
            answer_text
        )
        result["_answer_label"] = answer_label

        return result

    def _get_answer_token_ids(self, answer_text: str) -> list[int]:
        """Get token IDs for an answer string (e.g., 'Yes' or 'No')."""
        tokenizer = self.processor.tokenizer
        ids = tokenizer.encode(
            answer_text, add_special_tokens=False, return_tensors="pt",
        )
        if ids.numel() > 0:
            return ids[0].tolist()
        # Fallback via vocab
        tid = tokenizer.vocab.get(answer_text)
        if tid is not None:
            return [tid]
        raise RuntimeError(
            f"Cannot find token ID for answer text: {answer_text!r}"
        )


class RetainDataset(Dataset):
    """Dataset of samples from retain identities or general visual examples.

    Preserves all processor tensor outputs (P0-2) and builds labels with
    prompt masking so only assistant answer tokens contribute to the
    retain loss.
    """

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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image_uri = sample["image_uri"]
        question = sample["question"]
        answer_label = sample["answer_label"]
        answer_text = "Yes" if answer_label else "No"

        # -- Build user-only prompt for assistant start position -------- #
        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        try:
            user_prompt = self.processor.apply_chat_template(
                user_messages,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            user_prompt = self.processor.apply_chat_template(
                user_messages,
                add_generation_prompt=True,
            )

        user_tokens = self.processor.tokenizer(
            user_prompt, return_tensors="pt", truncation=False,
        )
        assistant_start = user_tokens["input_ids"].shape[1]

        # -- Full conversation with assistant answer -------------------- #
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text}],
            },
        ]

        try:
            full_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            full_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
            )

        from PIL import Image
        image = Image.open(image_uri).convert("RGB")

        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        # -- Preserve ALL processor tensor outputs (P0-2) --------------- #
        result: dict[str, Any] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                result[key] = value.squeeze(0)

        if "pixel_values" not in result:
            raise RuntimeError(
                f"Required multimodal tensor 'pixel_values' missing from "
                f"processor output for sample idx={idx}. "
                f"Available keys: {sorted(inputs.keys())}"
            )

        # -- Labels with prompt masking --------------------------------- #
        labels = result["input_ids"].clone()
        labels[:assistant_start] = -100
        result["labels"] = labels

        return result


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
# Qwen-aware collator (P0-2)
# --------------------------------------------------------------------------- #

def qwen_collate_fn(
    batch: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Collate function for Qwen multimodal training batches.

    Pads ``input_ids`` and ``labels`` with appropriate pad/ignore tokens,
    stacks all other tensor fields, and preserves metadata keys prefixed
    with ``_``.
    """
    pad_token_id = 0  # Qwen default pad token

    # -- Determine max sequence length --------------------------------- #
    max_len = max(item["input_ids"].shape[0] for item in batch)

    padded_input_ids: list[torch.Tensor] = []
    padded_attention_mask: list[torch.Tensor] = []
    padded_labels: list[torch.Tensor] = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        padded_input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        padded_attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_labels.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

    result: dict[str, torch.Tensor] = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels),
    }

    # -- Stack other tensor fields (pixel_values, image_grid_thw, …) --- #
    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())

    for key in all_keys:
        if key in ("input_ids", "attention_mask", "labels") or key.startswith("_"):
            continue
        tensors = [item[key] for key in [key] for item in batch if key in item]
        if len(tensors) == len(batch) and torch.is_tensor(tensors[0]):
            try:
                result[key] = torch.stack(tensors)
            except RuntimeError:
                # Variable-length non-padded tensors — keep as list
                result[key] = tensors

    # -- Preserve metadata for candidate-margin loss ------------------- #
    for key in ("_answer_position", "_correct_answer_token_ids", "_answer_label"):
        if key in batch[0]:
            result[key] = [item[key] for item in batch]

    return result


# --------------------------------------------------------------------------- #
# Loss functions
# --------------------------------------------------------------------------- #

def compute_forget_loss(
    model: Any,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Candidate-margin forget loss (P0-1).

    Computes ``M = logP(correct_answer) − logP(wrong_answer)`` at the
    assistant answer position and returns ``M`` so that minimising the
    loss *reduces* the candidate margin.
    """
    # Forward with labels so the model computes CE loss (unused here, but
    # ensures correct graph if needed).
    model_kwargs: dict[str, Any] = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    # Pass all multimodal tensors
    for key, value in batch.items():
        if (
            key not in ("input_ids", "attention_mask", "labels")
            and not key.startswith("_")
            and (torch.is_tensor(value) or (isinstance(value, list) and len(value) > 0))
        ):
            model_kwargs[key] = value

    outputs = model(**model_kwargs)
    logits = outputs.logits  # (B, T, V)

    answer_positions = batch["_answer_position"]
    correct_token_ids_list = batch["_correct_answer_token_ids"]

    yes_id = model_kwargs.get("_yes_token_id")
    if yes_id is None:
        # Resolve from the first sample's metadata
        yes_id = correct_token_ids_list[0][0] if correct_token_ids_list[0][0] != 16484 else 16484

    total_margin = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    for i in range(logits.shape[0]):
        pos = answer_positions[i]
        correct_ids = correct_token_ids_list[i]
        # Use first token of the answer for margin computation
        correct_id = correct_ids[0] if isinstance(correct_ids, list) else correct_ids
        wrong_id = 16484 if correct_id != 16484 else 7681  # Yes=16484, No=7681

        log_probs = torch.log_softmax(logits[i, pos, :], dim=-1)
        log_p_correct = log_probs[correct_id]
        log_p_wrong = log_probs[wrong_id]

        # M = logP(correct) - logP(wrong)
        margin = log_p_correct - log_p_wrong
        total_margin = total_margin + margin

    # L_forget = mean(M) — minimising reduces the margin
    return total_margin / logits.shape[0]


def compute_retain_loss(
    model: Any,
    batch: dict[str, torch.Tensor],
    reference_model: Any | None = None,
) -> torch.Tensor:
    """Compute the retain loss.

    If *reference_model* is provided, use KL divergence to the frozen
    reference on the assistant-answer tokens (where ``labels != -100``).
    Otherwise fall back to standard LM loss with the masked labels.
    """
    model_kwargs: dict[str, Any] = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    for key, value in batch.items():
        if (
            key not in ("input_ids", "attention_mask", "labels")
            and not key.startswith("_")
            and (torch.is_tensor(value) or (isinstance(value, list) and len(value) > 0))
        ):
            model_kwargs[key] = value

    if reference_model is not None:
        # KL divergence to frozen reference
        with torch.no_grad():
            ref_outputs = reference_model(**model_kwargs)
            ref_logits = ref_outputs.logits

        curr_outputs = model(**model_kwargs)
        curr_logits = curr_outputs.logits

        # Mask to assistant-answer positions only
        labels = batch["labels"]
        answer_mask = labels != -100  # (B, T)

        # Shift for next-token prediction
        shift_mask = answer_mask[:, 1:]
        shift_curr = curr_logits[:, :-1, :]
        shift_ref = ref_logits[:, :-1, :]

        if shift_mask.sum() == 0:
            return torch.tensor(0.0, device=curr_logits.device, requires_grad=True)

        # KL divergence over answer tokens only
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(shift_curr, dim=-1),
            torch.softmax(shift_ref, dim=-1),
            reduction="batchmean",
        )
        return kl
    else:
        # Standard LM loss with masked labels
        model_kwargs["labels"] = batch["labels"]
        outputs = model(**model_kwargs)
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

        # -- Deterministic seeding (P0-13) ----------------------------- #
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        # Data loaders (seeded generator + Qwen collator)
        self.forget_loader = DataLoader(
            forget_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            generator=self.generator,
            collate_fn=qwen_collate_fn,
        )
        self.retain_loader = DataLoader(
            retain_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            generator=self.generator,
            collate_fn=qwen_collate_fn,
        )

        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # -- Warmup LR scheduler (P1-3) -------------------------------- #
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

        # Move model to device
        self.device = torch.device(config.device)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move batch tensors to the training device.

        Non-tensor values (metadata lists) are passed through unchanged.
        """
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
        training_summary : dict
            Summary of training including final losses and checkpoint paths.
        """
        self.model.train()
        self.model.to(self.device)

        forget_iter = iter(self.forget_loader)
        retain_iter = iter(self.retain_loader)

        # -- P0-12: num_optimizer_steps × gradient_accumulation_steps -- #
        num_opt_steps = self.config.num_optimizer_steps
        grad_accum = self.config.gradient_accumulation_steps
        checkpoint_steps = set(self.config.checkpoint_steps)

        # Save initial checkpoint if step 0 is requested
        if 0 in checkpoint_steps:
            self._save_checkpoint("optimizer_step_000")

        total_microsteps = num_opt_steps * grad_accum
        logger.info(
            f"Starting training for {num_opt_steps} optimizer steps "
            f"({total_microsteps} microbatches, grad_accum={grad_accum})"
        )
        start_time = time.time()

        for opt_step in range(1, num_opt_steps + 1):
            # Accumulate gradients over grad_accum microbatches
            self.optimizer.zero_grad()
            accum_forget = 0.0
            accum_retain = 0.0
            accum_total = 0.0

            for micro_idx in range(grad_accum):
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
                scaled_loss = total_loss / grad_accum
                scaled_loss.backward()

                accum_forget += forget_loss.item()
                accum_retain += retain_loss.item()
                accum_total += total_loss.item()

            # -- Optimizer step ---------------------------------------- #
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step = opt_step

            # Average losses over accumulation
            avg_forget = accum_forget / grad_accum
            avg_retain = accum_retain / grad_accum
            avg_total = accum_total / grad_accum

            # Log
            elapsed = time.time() - start_time
            log_entry = {
                "optimizer_step": opt_step,
                "total_loss": avg_total,
                "forget_loss": avg_forget,
                "retain_loss": avg_retain,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "elapsed_seconds": elapsed,
            }
            self.training_log.append(log_entry)
            logger.info(
                f"Optimizer step {opt_step}/{num_opt_steps} | "
                f"loss={avg_total:.4f} "
                f"forget={avg_forget:.4f} "
                f"retain={avg_retain:.4f}"
            )

            # Safety checks
            if not math.isfinite(avg_total):
                raise RuntimeError(
                    f"Non-finite loss at optimizer step {opt_step}"
                )

            # Checkpoint
            if opt_step in checkpoint_steps:
                ckpt_name = f"optimizer_step_{opt_step:03d}"
                self._save_checkpoint(ckpt_name)

        # Save final checkpoint
        if num_opt_steps not in checkpoint_steps:
            self._save_checkpoint(f"optimizer_step_{num_opt_steps:03d}_final")

        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed:.1f}s")

        return {
            "num_optimizer_steps": num_opt_steps,
            "gradient_accumulation_steps": grad_accum,
            "total_microsteps": total_microsteps,
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
            "name": "lora_targeted_candidate_margin",
            "hyperparameters": {
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout,
                "lora_target_modules": config.lora_target_modules,
                "learning_rate": config.learning_rate,
                "num_optimizer_steps": config.num_optimizer_steps,
                "retain_weight": config.retain_weight,
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "warmup_steps": config.warmup_steps,
            },
        },
        "seed": config.seed,
        "determinism": {
            "random_seed": config.seed,
            "numpy_seed": config.seed,
            "torch_manual_seed": config.seed,
            "torch_cuda_manual_seed_all": config.seed,
            "dataloader_generator_seed": config.seed,
        },
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
