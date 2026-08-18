"""MANU (B8): Modality-Aware Neuron Unlearning.

Paper: Liu et al., ACL 2025 (arXiv:2502.15910)

This module implements the MANU baseline adapted to the MIDP
route-unlearning pipeline. MANU identifies MLP neurons that are
important for forget data and prunes them while preserving neurons
important for retain data.

Key steps:
1. Neuron inventory — MLP intermediate channels as primary unit
2. Modality-aware importance — per-neuron contribution estimates
3. Pruning — clip/zero selected neuron outputs + associated weights
4. Evaluation — 5% and 10% prune rates

Public API
----------
.. autoclass:: MANUConfig
.. autoclass:: MANU
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class MANUConfig:
    """Configuration for MANU neuron pruning."""

    # Pruning rates
    primary_prune_fraction: float = 0.10  # 10% primary
    secondary_prune_fraction: float = 0.05  # 5% secondary

    # Importance estimation
    importance_n_samples: int = 32
    importance_batch_size: int = 1

    # Neuron selection
    neuron_unit: str = "mlp_intermediate"  # primary unit: MLP intermediate channels

    # Training (for post-pruning fine-tuning if needed)
    learning_rate: float = 2e-5
    num_optimizer_steps: int = 0  # MANU is pruning-only by default
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    seed: int = 17

    # Output
    output_dir: str = ""

    @property
    def prune_rates(self) -> list[float]:
        """Return list of prune rates to evaluate."""
        return [self.secondary_prune_fraction, self.primary_prune_fraction]


# --------------------------------------------------------------------------- #
# Neuron inventory
# --------------------------------------------------------------------------- #

def build_neuron_inventory(model: nn.Module) -> dict[str, Any]:
    """Identify all MLP neurons in the model.

    Primary unit: MLP intermediate channels (the activation dimension
    after the first linear layer in each MLP block).

    Parameters
    ----------
    model:
        The model to inventory.

    Returns
    -------
    inventory:
        Dict mapping layer names to neuron info.
    """
    inventory: dict[str, dict[str, Any]] = {}

    for name, module in model.named_modules():
        # Identify MLP layers — typically nn.Linear or gated variants
        # In Qwen-style models: mlp.gate_proj, mlp.up_proj, mlp.down_proj
        if _is_mlp_layer(name, module):
            # Determine intermediate size
            intermediate_size = _get_intermediate_size(module)
            if intermediate_size > 0:
                # Extract layer path (e.g., "model.layers.0.mlp")
                layer_path = _extract_mlp_layer_path(name)
                if layer_path not in inventory:
                    inventory[layer_path] = {
                        "layer_path": layer_path,
                        "modules": {},
                        "n_neurons": intermediate_size,
                        "total_parameters": 0,
                    }

                inventory[layer_path]["modules"][name] = {
                    "module_name": name,
                    "module_type": type(module).__name__,
                    "shape": list(module.weight.shape) if hasattr(module, "weight") else [],
                    "numel": module.weight.numel() if hasattr(module, "weight") else 0,
                }
                inventory[layer_path]["total_parameters"] += (
                    module.weight.numel() if hasattr(module, "weight") else 0
                )

    summary = {
        "n_layers": len(inventory),
        "n_total_neurons": sum(info["n_neurons"] for info in inventory.values()),
        "n_total_parameters": sum(info["total_parameters"] for info in inventory.values()),
    }

    logger.info(
        f"Neuron inventory: {summary['n_layers']} MLP layers, "
        f"{summary['n_total_neurons']} total neurons"
    )

    return {
        "layers": inventory,
        "_summary": summary,
    }


def _is_mlp_layer(name: str, module: nn.Module) -> bool:
    """Check if a module is an MLP sub-layer.

    Only matches ``up_proj`` to avoid dimension mismatches between
    gate/up (intermediate_size) and down (hidden_size) projections.
    """
    if not isinstance(module, nn.Linear):
        return False
    return ".mlp.up_proj" in name


def _get_intermediate_size(module: nn.Module) -> int:
    """Get the intermediate (neuron) dimension of an MLP layer."""
    if hasattr(module, "weight"):
        # For gate_proj/up_proj: out_features = intermediate_size
        # For down_proj: in_features = intermediate_size
        return max(module.weight.shape[0], module.weight.shape[1])
    return 0


def _extract_mlp_layer_path(module_name: str) -> str:
    """Extract the parent MLP layer path from a module name.

    E.g., 'model.layers.0.mlp.gate_proj' -> 'model.layers.0.mlp'
    """
    parts = module_name.split(".")
    # Find the 'mlp' segment and return everything up to and including it
    for i, part in enumerate(parts):
        if part == "mlp":
            return ".".join(parts[:i + 1])
    return module_name


def save_neuron_inventory(
    inventory: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save neuron inventory to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-safe format
    safe = {
        "_summary": inventory.get("_summary", {}),
        "layers": {},
    }
    for layer_path, info in inventory.get("layers", {}).items():
        safe["layers"][layer_path] = {
            "layer_path": info["layer_path"],
            "n_neurons": info["n_neurons"],
            "total_parameters": info["total_parameters"],
            "modules": {
                name: {
                    "module_name": m["module_name"],
                    "module_type": m["module_type"],
                    "shape": m["shape"],
                    "numel": m["numel"],
                }
                for name, m in info["modules"].items()
            },
        }

    with open(output_path, "w") as f:
        json.dump(safe, f, indent=2)
        f.write("\n")
    logger.info(f"Saved neuron inventory: {output_path}")


# --------------------------------------------------------------------------- #
# Modality-aware importance estimation
# --------------------------------------------------------------------------- #

def estimate_modality_importance(
    model: nn.Module,
    forget_loader: Any,
    retain_loader: Any | None,
    config: MANUConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Estimate per-neuron importance for each modality.

    For each MLP neuron, estimate its contribution to:
    - forget loss (visual + text)
    - retain loss (visual + text)

    Neurons with high forget importance and low retain importance
    are candidates for pruning.

    Parameters
    ----------
    model:
        The model.
    forget_loader:
        DataLoader for forget/target examples.
    retain_loader:
        DataLoader for retain examples.
    config:
        MANU configuration.
    device:
        Compute device.

    Returns
    -------
    importance:
        Dict mapping layer_path to per-neuron importance tensors.
    """
    # Collect per-neuron activation magnitudes
    forget_activations: dict[str, torch.Tensor] = {}
    retain_activations: dict[str, torch.Tensor] = {}

    # Register hooks to capture MLP outputs
    hooks = []
    activation_buffer: dict[str, torch.Tensor] = {}

    def _make_hook(layer_path: str):
        def hook_fn(module, input, output):
            # output shape: (batch, seq_len, intermediate_size)
            # Accumulate absolute activation magnitude per neuron
            act = output.detach().abs().mean(dim=(0, 1))  # (intermediate_size,)
            if layer_path in activation_buffer:
                activation_buffer[layer_path] += act
            else:
                activation_buffer[layer_path] = act.clone()
        return hook_fn

    # Register hooks on all MLP layers
    for name, module in model.named_modules():
        if _is_mlp_layer(name, module):
            layer_path = _extract_mlp_layer_path(name)
            hooks.append(module.register_forward_hook(_make_hook(layer_path)))

    # Accumulate forget activations
    _accumulate_activations(model, forget_loader, activation_buffer,
                            forget_activations, config, device)

    # Accumulate retain activations
    if retain_loader is not None:
        activation_buffer.clear()
        _accumulate_activations(model, retain_loader, activation_buffer,
                                retain_activations, config, device)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute importance: forget - retain (high = prune candidate)
    importance: dict[str, torch.Tensor] = {}
    for layer_path, forget_score in forget_activations.items():
        retain_score = retain_activations.get(layer_path, torch.zeros_like(forget_score))
        importance[layer_path] = forget_score - retain_score

    return importance


def _accumulate_activations(
    model: nn.Module,
    loader: Any,
    activation_buffer: dict[str, torch.Tensor],
    result_dict: dict[str, torch.Tensor],
    config: MANUConfig,
    device: torch.device,
) -> None:
    """Run forward passes and accumulate per-neuron activations."""
    model.eval()
    count = 0

    for batch in loader:
        if count >= config.importance_n_samples:
            break

        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            model_kwargs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            for key, value in batch.items():
                if (key not in ("input_ids", "attention_mask", "labels")
                        and not key.startswith("_")
                        and (torch.is_tensor(value) or (isinstance(value, list) and len(value) > 0))):
                    model_kwargs[key] = value

            model(**model_kwargs)

        count += 1

    # Normalize
    if count > 0:
        for name, value in activation_buffer.items():
            result_dict[name] = value / count


# --------------------------------------------------------------------------- #
# Neuron pruning
# --------------------------------------------------------------------------- #

def select_neurons_to_prune(
    importance: dict[str, torch.Tensor],
    prune_fraction: float,
) -> dict[str, list[int]]:
    """Select neurons to prune based on importance scores.

    Parameters
    ----------
    importance:
        Per-layer importance tensors.
    prune_fraction:
        Fraction of neurons to prune (0.0 to 1.0).

    Returns
    -------
    neurons_to_prune:
        Dict mapping layer_path to list of neuron indices to prune.
    """
    # Collect all (layer_path, neuron_idx, score) triples
    all_neurons = []
    for layer_path, scores in importance.items():
        for idx in range(scores.shape[0]):
            all_neurons.append((layer_path, idx, scores[idx].item()))

    if not all_neurons:
        logger.warning("Empty importance — no neurons to prune")
        return {}

    # Sort by importance (lower = more prune-worthy, since importance = forget - retain)
    # Actually, higher forget importance = more important to forget = should be pruned
    all_neurons.sort(key=lambda x: x[2], reverse=True)

    n_total = len(all_neurons)
    n_prune = max(1, int(n_total * prune_fraction))
    n_prune = min(n_prune, n_total)

    selected = all_neurons[:n_prune]

    # Group by layer
    neurons_to_prune: dict[str, list[int]] = {}
    for layer_path, idx, _ in selected:
        if layer_path not in neurons_to_prune:
            neurons_to_prune[layer_path] = []
        neurons_to_prune[layer_path].append(idx)

    total_pruned = sum(len(idxs) for idxs in neurons_to_prune.values())
    logger.info(
        f"Selected {total_pruned}/{n_total} neurons to prune "
        f"({total_pruned / max(n_total, 1) * 100:.1f}%) "
        f"across {len(neurons_to_prune)} layers"
    )

    return neurons_to_prune


def prune_neurons(
    model: nn.Module,
    neurons_to_prune: dict[str, list[int]],
) -> dict[str, Any]:
    """Zero out selected neuron weights.

    For each MLP layer, zero the output weights corresponding to
    selected neurons. This effectively removes their contribution
    to downstream computation.

    Parameters
    ----------
    model:
        The model.
    neurons_to_prune:
        Dict mapping layer_path to neuron indices.

    Returns
    -------
    prune_info:
        Information about what was pruned.
    """
    n_parameters_modified = 0
    n_weight_slices_modified = 0
    pruned_layers = []
    # Track unique (layer_path, neuron_idx) pairs to avoid double-counting
    # when the same neuron is zeroed in both up_proj and down_proj.
    pruned_neuron_set: set[tuple[str, int]] = set()

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        layer_path = _extract_mlp_layer_path(name)
        if layer_path not in neurons_to_prune:
            continue

        neuron_indices = neurons_to_prune[layer_path]

        if hasattr(module, "weight"):
            with torch.no_grad():
                out_features = module.weight.shape[0]
                in_features = module.weight.shape[1]

                if ".mlp.up_proj" in name:
                    # up_proj: out_features = intermediate_size → zero rows
                    for idx in neuron_indices:
                        if idx < out_features:
                            module.weight[idx, :] = 0.0
                            n_parameters_modified += in_features
                            n_weight_slices_modified += 1
                            pruned_neuron_set.add((layer_path, idx))
                elif ".mlp.down_proj" in name:
                    # down_proj: in_features = intermediate_size → zero columns
                    for idx in neuron_indices:
                        if idx < in_features:
                            module.weight[:, idx] = 0.0
                            n_parameters_modified += out_features
                            n_weight_slices_modified += 1
                            pruned_neuron_set.add((layer_path, idx))
                # gate_proj is intentionally skipped — pruning is defined
                # by up_proj row zeroing + down_proj column zeroing.

        if layer_path not in pruned_layers:
            pruned_layers.append(layer_path)

    n_unique = len(pruned_neuron_set)

    prune_info = {
        "unique_neurons_pruned": n_unique,
        "weight_slices_modified": n_weight_slices_modified,
        "n_parameters_modified": n_parameters_modified,
        "n_layers_affected": len(pruned_layers),
        "pruned_layers": pruned_layers,
    }

    logger.info(
        f"Pruned {n_unique} unique neurons ({n_weight_slices_modified} weight "
        f"slices, {n_parameters_modified} parameters) "
        f"across {len(pruned_layers)} layers"
    )

    return prune_info


# --------------------------------------------------------------------------- #
# MANU class
# --------------------------------------------------------------------------- #

class MANU:
    """MANU neuron pruning baseline (B8).

    Implements modality-aware neuron identification and pruning for
    selective unlearning.
    """

    name = "manu"

    def __init__(self, config: MANUConfig):
        self.config = config

    def run(
        self,
        model: nn.Module,
        forget_loader: Any,
        retain_loader: Any | None = None,
        device: str = "cuda:0",
        eval_callback: Callable[[str, nn.Module], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run the full MANU pipeline.

        Parameters
        ----------
        model:
            The model to unlearn.
        forget_loader:
            DataLoader for forget examples.
        retain_loader:
            DataLoader for retain examples.
        device:
            Compute device.
        eval_callback:
            Optional callback ``f(rate_str, model)`` invoked on the **live
            pruned model** before restoration.  Must return an eval-results
            dict (e.g. from ``evaluate_intervention``).

        Returns
        -------
        results:
            Pipeline results.
        """
        dev = torch.device(device)
        model.to(dev)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Neuron inventory
        logger.info("Step 1: Building neuron inventory ...")
        inventory = build_neuron_inventory(model)
        save_neuron_inventory(
            inventory, output_dir / "manu_neuron_inventory.json",
        )

        # Step 2: Importance estimation
        logger.info("Step 2: Estimating modality-aware importance ...")
        importance = estimate_modality_importance(
            model, forget_loader, retain_loader, self.config, dev,
        )

        # Save importance scores
        importance_info = {}
        for layer_path, scores in importance.items():
            importance_info[layer_path] = {
                "n_neurons": scores.shape[0],
                "mean": scores.mean().item(),
                "std": scores.std().item(),
                "min": scores.min().item(),
                "max": scores.max().item(),
            }
        with open(output_dir / "importance_info.json", "w") as f:
            json.dump(importance_info, f, indent=2)
            f.write("\n")

        # Step 3: Pruning at multiple rates
        # Save original weights to CPU to restore between prune rates (avoids OOM from deepcopy)
        original_state = {n: p.cpu().clone() for n, p in model.state_dict().items()}

        # P1-18: Compute pre-prune state hash for restore verification.
        pre_prune_hash = _state_dict_sha256(original_state)
        logger.info(f"Pre-prune state hash: {pre_prune_hash[:16]}...")

        results = {}
        for prune_rate in self.config.prune_rates:
            logger.info(f"Step 3: Pruning at {prune_rate * 100:.0f}% ...")

            # Select neurons
            neurons = select_neurons_to_prune(importance, prune_rate)

            # Prune in-place (avoids OOM from deepcopy)
            prune_info = prune_neurons(model, neurons)

            # Verify forward pass works
            try:
                _verify_forward(model, forget_loader, dev)
                prune_info["forward_pass_ok"] = True
            except Exception as e:
                logger.error(f"Forward pass failed after pruning: {e}")
                prune_info["forward_pass_ok"] = False

            # Deterministic pruning specification with selection hash (P0-8)
            selection_str = json.dumps(
                {k: sorted(v) for k, v in neurons.items()}, sort_keys=True,
            )
            selection_sha = hashlib.sha256(selection_str.encode()).hexdigest()

            # P0-12: Use zero-padded rate string (05, 10) for consistency.
            rate_str = f"{round(prune_rate * 100):02d}"

            prune_spec = {
                "prune_fraction": prune_rate,
                "selected_neurons": {k: sorted(v) for k, v in neurons.items()},
                "n_unique_neurons_pruned": prune_info["unique_neurons_pruned"],
                "n_parameters_modified": prune_info["n_parameters_modified"],
                "n_weight_slices_modified": prune_info["weight_slices_modified"],
                "n_layers_affected": prune_info["n_layers_affected"],
                "selection_sha256": selection_sha,
            }

            # Persist pruning specification (P0-8)
            with open(output_dir / f"prune_spec_{rate_str}.json", "w") as f:
                json.dump(prune_spec, f, indent=2)
                f.write("\n")
            logger.info(
                f"Saved pruning specification (rate={rate_str}%, "
                f"sha256={selection_sha[:16]}...)"
            )

            # Evaluate the live pruned model BEFORE restoration (P0-8).
            # PeftModel.save_pretrained() cannot preserve base-weight zeroing,
            # so we must evaluate the in-memory pruned model directly.
            if eval_callback is not None and prune_info.get("forward_pass_ok"):
                logger.info(
                    f"Evaluating live pruned model at {rate_str}% ..."
                )
                eval_result = eval_callback(rate_str, model)
                prune_info["eval_result"] = eval_result
                if not eval_result.get("strict_validation_pass"):
                    logger.error(
                        f"MANU prune_{rate_str}: strict validation FAILED"
                    )
            else:
                eval_result = None

            prune_info["prune_fraction"] = prune_rate
            results[f"prune_{rate_str}"] = prune_info

            # Restore original weights for next prune rate
            model.load_state_dict(original_state)
            # P1-18: Verify restoration.
            post_restore_hash = _state_dict_sha256(
                {n: p.cpu().clone() for n, p in model.state_dict().items()}
            )
            restore_verified = post_restore_hash == pre_prune_hash
            prune_info["restore_verified"] = restore_verified
            if not restore_verified:
                logger.error(
                    f"MANU prune_{rate_str}: restoration FAILED "
                    f"(pre={pre_prune_hash[:16]}..., "
                    f"post={post_restore_hash[:16]}...)"
                )
            else:
                logger.info(
                    f"Restored original model after {rate_str}% "
                    f"(verified: {post_restore_hash[:16]}...)"
                )

        # Clean up
        del original_state

        return {
            "inventory": inventory.get("_summary", {}),
            "importance": importance_info,
            "pruning": results,
            # P1-18: Restore verification metadata.
            "pre_prune_state_hash": pre_prune_hash,
            "post_restore_state_hash": post_restore_hash,
            "restore_verified": restore_verified,
        }


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Compute SHA-256 of a model state dict for restore verification (P1-18).

    Hashes the raw bytes of each tensor in sorted key order.
    """
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        t = state_dict[key]
        raw = t.detach().contiguous().cpu().numpy().tobytes()
        h.update(key.encode("utf-8"))
        h.update(raw)
    return h.hexdigest()


def _verify_forward(
    model: nn.Module,
    loader: Any,
    device: torch.device,
) -> None:
    """Verify that forward pass works after pruning."""
    model.eval()
    batch = next(iter(loader))
    batch = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }

    with torch.no_grad():
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
        if not hasattr(outputs, "logits"):
            raise RuntimeError("Model output has no logits attribute")
