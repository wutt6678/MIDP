"""MMUnlearner (B7): Saliency-guided structural unlearning.

Paper: Huo et al., ACL Findings 2025 (arXiv:2502.11051)

This module implements the MMUnlearner baseline adapted to the MIDP
route-unlearning pipeline. MMUnlearner identifies parameters that are
important for forget data and selectively modifies them while preserving
parameters important for retain/text-preservation data.

Key steps:
1. Parameter inventory — classify all Qwen parameters into categories
2. Saliency data — accumulate forget/retain/text importance scores
3. Gradient mask generation — construct saliency ranking
4. Training — answer-only GA + zero gradients outside mask

Public API
----------
.. autoclass:: MMUnlearnerConfig
.. autoclass:: MMUnlearner
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Parameter categories
# --------------------------------------------------------------------------- #

PARAMETER_CATEGORIES = {
    "vision_encoder": ["visual.", "vision_tower."],
    "vlm_projector": ["multi_modal_projector.", "visual_proj.", "mm_projector."],
    "attention_qkv": [".q_proj.", ".k_proj.", ".v_proj."],
    "attention_output": [".o_proj."],
    "mlp": [".mlp."],
    "embeddings": ["embed_tokens.", "wte.", "embedding."],
    "lm_head": ["lm_head."],
    "norm": ["norm.", "ln_f.", "layer_norm."],
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class MMUnlearnerConfig:
    """Configuration for MMUnlearner."""

    # Saliency estimation
    saliency_n_samples: int = 32
    saliency_batch_size: int = 1

    # Mask parameters
    mask_granularity: str = "element"  # "element" or "row"
    target_sparsity: float = 0.5  # Fraction of parameters to mask out
    min_mask_fraction: float = 0.01  # Minimum fraction to select
    max_mask_fraction: float = 0.99  # Maximum fraction to select

    # Training
    learning_rate: float = 2e-5
    num_optimizer_steps: int = 125
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    seed: int = 17

    # Modality ablation
    modality: str = "both"  # "both", "vision_only", "language_only"

    # Output
    output_dir: str = ""

    @property
    def effective_sparsity(self) -> float:
        """Fraction of parameters excluded from training."""
        return max(self.min_mask_fraction, min(self.target_sparsity, self.max_mask_fraction))


# --------------------------------------------------------------------------- #
# Parameter inventory
# --------------------------------------------------------------------------- #

def build_parameter_inventory(model: nn.Module) -> dict[str, Any]:
    """Classify all model parameters into categories.

    Parameters
    ----------
    model:
        The model to inventory.

    Returns
    -------
    inventory:
        Dict mapping category names to parameter info.
    """
    inventory: dict[str, dict[str, Any]] = {cat: {"params": [], "numel": 0} for cat in PARAMETER_CATEGORIES}
    inventory["other"] = {"params": [], "numel": 0}

    total_params = 0
    for name, param in model.named_parameters():
        categorized = False
        for cat, prefixes in PARAMETER_CATEGORIES.items():
            if any(prefix in name for prefix in prefixes):
                inventory[cat]["params"].append(name)
                inventory[cat]["numel"] += param.numel()
                categorized = True
                break
        if not categorized:
            inventory["other"]["params"].append(name)
            inventory["other"]["numel"] += param.numel()
        total_params += param.numel()

    inventory["_summary"] = {
        "total_parameters": total_params,
        "categories": {
            cat: {"n_params": len(info["params"]), "numel": info["numel"]}
            for cat, info in inventory.items() if cat != "_summary"
        },
    }

    logger.info(f"Parameter inventory: {total_params} total parameters")
    for cat, info in inventory.items():
        if cat != "_summary" and info["numel"] > 0:
            logger.info(f"  {cat}: {info['numel']} params ({len(info['params'])} tensors)")

    return inventory


def save_parameter_inventory(
    inventory: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save parameter inventory to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-safe format
    safe = {}
    for cat, info in inventory.items():
        if cat == "_summary":
            # _summary has its own structure
            safe[cat] = info
        elif isinstance(info, dict):
            safe[cat] = {
                "params": info.get("params", []),
                "numel": info.get("numel", 0),
            }
        else:
            safe[cat] = info

    with open(output_path, "w") as f:
        json.dump(safe, f, indent=2)
        f.write("\n")
    logger.info(f"Saved parameter inventory: {output_path}")


# --------------------------------------------------------------------------- #
# Saliency estimation
# --------------------------------------------------------------------------- #

def estimate_saliency(
    model: nn.Module,
    forget_loader: Any,
    retain_loader: Any | None,
    text_loader: Any | None,
    config: MMUnlearnerConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Estimate per-parameter saliency scores.

    Accumulates gradient magnitudes for forget, retain, and text data.
    High forget + low retain + low text = candidate for modification.

    Parameters
    ----------
    model:
        The model.
    forget_loader:
        DataLoader for forget/target examples.
    retain_loader:
        DataLoader for retain examples (optional).
    text_loader:
        DataLoader for text-only preservation set (optional).
    config:
        MMUnlearner configuration.
    device:
        Compute device.

    Returns
    -------
    saliency:
        Dict mapping parameter names to importance score tensors.
    """
    # Accumulate gradient magnitudes
    forget_importance: dict[str, torch.Tensor] = {}
    retain_importance: dict[str, torch.Tensor] = {}
    text_importance: dict[str, torch.Tensor] = {}

    def _accumulate(loader, importance_dict, n_batches):
        count = 0
        for batch in loader:
            if count >= n_batches:
                break
            # Move batch to device
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            model.zero_grad()
            # Simple CE forward/backward
            model_kwargs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            for key, value in batch.items():
                if (key not in ("input_ids", "attention_mask", "labels")
                        and not key.startswith("_")
                        and (torch.is_tensor(value) or (isinstance(value, list) and len(value) > 0))):
                    model_kwargs[key] = value

            outputs = model(**model_kwargs)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in importance_dict:
                        importance_dict[name] = torch.zeros_like(param.data)
                    importance_dict[name] += param.grad.data.abs()

            count += 1
        # Normalize
        if count > 0:
            for name in importance_dict:
                importance_dict[name] /= count

    logger.info("Estimating forget saliency ...")
    _accumulate(forget_loader, forget_importance, config.saliency_n_samples)

    if retain_loader is not None:
        logger.info("Estimating retain saliency ...")
        _accumulate(retain_loader, retain_importance, config.saliency_n_samples)

    if text_loader is not None:
        logger.info("Estimating text preservation saliency ...")
        _accumulate(text_loader, text_importance, config.saliency_n_samples)

    # Compute combined saliency score (per-element):
    # High forget, low retain, low text → high score
    saliency: dict[str, torch.Tensor] = {}
    for name, forget_score in forget_importance.items():
        retain_score = retain_importance.get(name, 0.0)
        text_score = text_importance.get(name, 0.0)

        # Saliency = forget importance - retain importance - text importance
        score = forget_score - retain_score - text_score
        saliency[name] = score

    return saliency


# --------------------------------------------------------------------------- #
# Mask generation
# --------------------------------------------------------------------------- #

def generate_saliency_mask(
    model: nn.Module,
    saliency: dict[str, torch.Tensor],
    config: MMUnlearnerConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Generate an element-level binary mask from saliency scores.

    Individual elements (not whole tensors) are ranked by saliency.
    Uses **exact global top-k index selection** over trainable parameters
    only (Option A — LoRA-compatible mask population).

    Parameters
    ----------
    model:
        The model.
    saliency:
        Per-element saliency score tensors (one tensor per parameter).
    config:
        MMUnlearner configuration.

    Returns
    -------
    mask:
        Dict mapping parameter names to binary mask tensors.
    mask_meta:
        Metadata about mask construction including exact cardinality.
    """
    # Option A: trainable-only population (coherent with LoRA optimizer).
    trainable_params = [
        (n, p) for n, p in model.named_parameters() if p.requires_grad
    ]
    total_numel = sum(p.numel() for _, p in trainable_params)

    if total_numel == 0:
        logger.warning("No trainable parameters — generating empty mask")
        mask = {n: torch.zeros_like(p.data) for n, p in trainable_params}
        meta = {
            "requested_sparsity": config.target_sparsity,
            "requested_selected_fraction": 1.0 - config.effective_sparsity,
            "population_numel": 0,
            "requested_selected_numel": 0,
            "actual_selected_numel": 0,
            "measured_sparsity": 0.0,
            "mask_population": "trainable_only",
        }
        return mask, meta

    # Collect all element scores on CPU for global ranking — trainable only.
    all_scores_list: list[torch.Tensor] = []
    param_layout: list[tuple[str, torch.Size, int, torch.dtype]] = []
    for name, param in trainable_params:
        score = saliency.get(name)
        if score is not None:
            all_scores_list.append(score.detach().flatten().cpu())
        else:
            all_scores_list.append(torch.zeros(param.numel()))
        param_layout.append((name, param.data.shape, param.numel(), param.data.dtype))

    all_scores_flat = torch.cat(all_scores_list)
    assert all_scores_flat.numel() == total_numel, (
        f"Score vector numel {all_scores_flat.numel()} != "
        f"trainable population {total_numel}"
    )

    # Exact global top-k index selection.
    requested_selected_numel = int(total_numel * (1.0 - config.effective_sparsity))
    requested_selected_numel = max(1, min(requested_selected_numel, total_numel))

    topk_indices = torch.topk(
        all_scores_flat, k=requested_selected_numel, largest=True,
    ).indices

    global_mask = torch.zeros(total_numel, dtype=torch.bool)
    global_mask[topk_indices] = True

    # Split global_mask back into the original parameter tensor shapes.
    mask: dict[str, torch.Tensor] = {}
    offset = 0
    n_selected = 0
    for name, shape, numel, dtype in param_layout:
        chunk = global_mask[offset:offset + numel]
        m = chunk.reshape(shape).to(dtype)
        mask[name] = m
        n_selected += int(m.sum().item())
        offset += numel

    actual_selected_numel = n_selected
    measured_sparsity = 1.0 - (actual_selected_numel / max(total_numel, 1))

    # Hard validation gate — exact or ±1 tolerance.
    if abs(actual_selected_numel - requested_selected_numel) > 1:
        raise RuntimeError(
            f"Mask cardinality gate FAILED: requested "
            f"{requested_selected_numel} selected elements, got "
            f"{actual_selected_numel} (tolerance: 1)"
        )

    mask_meta: dict[str, Any] = {
        "requested_sparsity": config.target_sparsity,
        "requested_selected_fraction": 1.0 - config.effective_sparsity,
        "population_numel": total_numel,
        "requested_selected_numel": requested_selected_numel,
        "actual_selected_numel": actual_selected_numel,
        "measured_sparsity": measured_sparsity,
        "mask_population": "trainable_only",
    }

    logger.info(
        f"Mask generated (element-level, trainable-only): "
        f"{actual_selected_numel}/{total_numel} elements selected "
        f"(measured sparsity: {measured_sparsity:.3f})"
    )

    return mask, mask_meta


def apply_mask_to_gradients(
    model: nn.Module,
    mask: dict[str, torch.Tensor],
) -> None:
    """Zero out gradients for parameters outside the mask.

    Parameters
    ----------
    model:
        The model (after backward pass).
    mask:
        Binary mask per parameter name.
    """
    for name, param in model.named_parameters():
        if param.grad is not None and name in mask:
            param.grad.data *= mask[name].to(param.grad.data.device)


# --------------------------------------------------------------------------- #
# MMUnlearner class
# --------------------------------------------------------------------------- #

class MMUnlearner:
    """MMUnlearner structural unlearning baseline (B7).

    Implements saliency-guided gradient masking for selective parameter
    modification during unlearning training.
    """

    name = "mmunlearner"

    def __init__(self, config: MMUnlearnerConfig):
        self.config = config

    def run(
        self,
        model: nn.Module,
        forget_loader: Any,
        retain_loader: Any | None = None,
        text_loader: Any | None = None,
        device: str = "cuda:0",
    ) -> dict[str, Any]:
        """Run the full MMUnlearner pipeline.

        Parameters
        ----------
        model:
            The model to unlearn.
        forget_loader:
            DataLoader for forget examples.
        retain_loader:
            DataLoader for retain examples.
        text_loader:
            DataLoader for text-only preservation set.
        device:
            Compute device.

        Returns
        -------
        results:
            Pipeline results dict.
        """
        dev = torch.device(device)
        model.to(dev)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Parameter inventory
        logger.info("Step 1: Building parameter inventory ...")
        inventory = build_parameter_inventory(model)
        save_parameter_inventory(
            inventory, output_dir / "mmunlearner_parameter_inventory.json",
        )

        # Step 2: Saliency estimation
        logger.info("Step 2: Estimating saliency ...")
        saliency = estimate_saliency(
            model, forget_loader, retain_loader, text_loader,
            self.config, dev,
        )

        # Step 3: Mask generation
        logger.info("Step 3: Generating saliency mask ...")
        mask, mask_meta = generate_saliency_mask(model, saliency, self.config)

        # Save mask info — merge config + exact cardinality metadata
        mask_info = {
            "granularity": self.config.mask_granularity,
            "modality": self.config.modality,
            **mask_meta,
        }
        with open(output_dir / "mask_info.json", "w") as f:
            json.dump(mask_info, f, indent=2)
            f.write("\n")

        # Step 4: Training with masked gradients
        logger.info("Step 4: Training with masked gradients ...")
        training_results = self._train_with_mask(
            model, forget_loader, mask, dev,
        )

        # Step 5: Save LoRA adapter checkpoint (P0-5)
        adapter_path = output_dir / "checkpoints" / "adapter_final"
        adapter_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_path))
        logger.info(f"Saved LoRA adapter checkpoint: {adapter_path}")

        return {
            "inventory": inventory.get("_summary", {}),
            "mask_info": mask_info,
            "training": training_results,
            "adapter_path": str(adapter_path),
        }

    def _train_with_mask(
        self,
        model: nn.Module,
        forget_loader: Any,
        mask: dict[str, torch.Tensor],
        device: torch.device,
    ) -> dict[str, Any]:
        """Train with masked gradients (answer-only GA + mask)."""
        model.train()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config.learning_rate)

        forget_iter = iter(forget_loader)
        grad_accum = self.config.gradient_accumulation_steps
        losses = []

        for step in range(1, self.config.num_optimizer_steps + 1):
            optimizer.zero_grad()
            accum_loss = 0.0

            for micro_idx in range(grad_accum):
                try:
                    batch = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    batch = next(forget_iter)

                batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                model_kwargs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                }
                for key, value in batch.items():
                    if (key not in ("input_ids", "attention_mask", "labels")
                            and not key.startswith("_")
                            and (torch.is_tensor(value) or (isinstance(value, list) and len(value) > 0))):
                        model_kwargs[key] = value

                outputs = model(**model_kwargs)
                logits = outputs.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch["labels"][:, 1:].contiguous()

                # Answer-only GA (negative CE)
                ce = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss = -ce / grad_accum
                loss.backward()
                accum_loss += (-ce).item()

            # Apply mask to gradients
            apply_mask_to_gradients(model, mask)

            # Clip and step
            torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
            optimizer.step()

            avg_loss = accum_loss / grad_accum
            losses.append(avg_loss)

            if step % 25 == 0 or step == 1:
                logger.info(f"  Step {step}: loss={avg_loss:.4f}")

        return {
            "num_steps": self.config.num_optimizer_steps,
            "final_loss": losses[-1] if losses else None,
            "losses": losses,
        }
