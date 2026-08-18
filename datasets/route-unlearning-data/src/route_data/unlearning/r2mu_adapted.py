"""R²MU-adapted (B9): Representation-level unlearning.

Paper: Wang et al., EMNLP 2025 (arXiv:2506.12963)

This is a cross-domain adaptation, NOT a paper-faithful multimodal
reproduction. The original R²MU uses CoT reasoning traces for
representation misdirection; since Qwen3.5-9B route probes have no
reasoning traces, we use only the representation misdirection core.

Key steps:
1. Layer selection — pre-register candidate layers, select via
   target-vs-retain linear separability on training-only examples
2. Forget objective — MSE between hidden representation and fixed
   random target vector z (norm-matched)
3. Retain objective — MSE between current and frozen-model
   representations (gamma=1.0 primary)
4. Representation pooling — pre-answer decision position
   (last token before ground-truth answer, derived from _prefix_len)
5. Trainable scope — same rank-8 LoRA as E2B

Public API
----------
.. autoclass:: R2MUAdaptedConfig
.. autoclass:: R2MUAdapted
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Decision-position extraction (P0-R2MU)
# --------------------------------------------------------------------------- #

def get_decision_positions(
    batch: dict[str, Any],
    seq_len: int,
) -> torch.Tensor:
    """Compute pre-answer decision positions from ``_prefix_len``.

    The decision representation is the hidden state *immediately before*
    the ground-truth answer tokens, i.e. at index ``prefix_len - 1``.

    Parameters
    ----------
    batch:
        A collated batch dict.  Must contain ``_prefix_len`` (one entry
        per example) and ``attention_mask`` (for non-padding validation).
    seq_len:
        Total sequence length (for bounds validation).

    Returns
    -------
    positions:
        1-D ``torch.LongTensor`` of shape ``(batch_size,)`` on the same
        device as ``batch["attention_mask"]``.

    Raises
    ------
    RuntimeError
        If ``_prefix_len`` is missing, any value is out of bounds,
        batch cardinality mismatches, or prefix extends into padding.
    """
    prefix_len_raw = batch.get("_prefix_len")
    if prefix_len_raw is None:
        raise RuntimeError(
            "R2MU decision-position extraction requires _prefix_len "
            "in the batch.  Ensure both ForgetDataset and RetainDataset "
            "store _prefix_len and the collate function preserves it."
        )

    # P1-3: Require attention_mask for non-padding validation.
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        raise RuntimeError(
            "R2MU decision-position extraction requires attention_mask "
            "in the batch for non-padding boundary validation."
        )

    # Normalise to a list of ints.
    if torch.is_tensor(prefix_len_raw):
        prefix_lens = prefix_len_raw.tolist()
    elif isinstance(prefix_len_raw, (list, tuple)):
        prefix_lens = [int(v) for v in prefix_len_raw]
    else:
        prefix_lens = [int(prefix_len_raw)]

    # P1-3: Validate batch cardinality.
    batch_size = attention_mask.shape[0]
    if len(prefix_lens) != batch_size:
        raise RuntimeError(
            f"R2MU _prefix_len cardinality mismatch: "
            f"got {len(prefix_lens)} entries for batch_size={batch_size}"
        )

    # P1-3: Compute non-padding lengths from attention_mask.
    non_padding_lengths = attention_mask.sum(dim=1).long()
    if torch.is_tensor(non_padding_lengths):
        non_padding_list = non_padding_lengths.tolist()
    else:
        non_padding_list = list(non_padding_lengths)

    # Determine output device from attention_mask.
    device = attention_mask.device

    positions = []
    for i, pl in enumerate(prefix_lens):
        np_len = int(non_padding_list[i])
        if pl <= 0:
            raise RuntimeError(
                f"_prefix_len[{i}]={pl} must be > 0"
            )
        # P1-3: Validate against actual non-padding length.
        if pl > np_len:
            raise RuntimeError(
                f"_prefix_len[{i}]={pl} exceeds non_padding_length={np_len} "
                f"(prefix extends into padding)"
            )
        # P1-3: Supervised B9 requires answer tokens after prefix.
        if pl >= np_len:
            raise RuntimeError(
                f"R2MU _prefix_len[{i}]={pl} reaches the end of the "
                f"non-padding sequence (length={np_len}); "
                f"expected supervised answer tokens after the prefix"
            )
        if pl > seq_len:
            raise RuntimeError(
                f"_prefix_len[{i}]={pl} exceeds seq_len={seq_len}"
            )
        positions.append(pl - 1)

    return torch.tensor(positions, dtype=torch.long, device=device)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class R2MUAdaptedConfig:
    """Configuration for R²MU-adapted representation unlearning."""

    # Layer selection
    candidate_layers: list[int] = field(default_factory=lambda: [8, 16, 24, 32])
    n_select_layers: int = 2  # Number of layers to select
    separability_n_samples: int = 32

    # Forget target
    target_seed: int = 42  # Seed for random target vector generation
    target_norm: float = 1.0  # Norm of the random target vector

    # Retain objective
    gamma: float = 1.0  # Weight for retain MSE term

    # Training
    learning_rate: float = 2e-5
    num_optimizer_steps: int = 125
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    seed: int = 17

    # Checkpointing
    checkpoint_steps: list[int] = field(
        default_factory=lambda: [1, 5, 10, 25, 50, 60, 75, 90, 125]
    )

    # Output
    output_dir: str = ""

    @property
    def method_name(self) -> str:
        return "r2mu_adapted"


# --------------------------------------------------------------------------- #
# Random target generation
# --------------------------------------------------------------------------- #

def generate_random_target(
    hidden_size: int,
    seed: int = 42,
    target_norm: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate a fixed random target vector for representation misdirection.

    Parameters
    ----------
    hidden_size:
        Dimension of the hidden representation.
    seed:
        Random seed for reproducibility.
    target_norm:
        Desired L2 norm of the target vector.
    device:
        Target device.

    Returns
    -------
    target:
        Random target vector with specified norm.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    target = torch.randn(hidden_size, generator=gen)
    # Normalize to target norm
    target = target / target.norm() * target_norm
    if device is not None:
        target = target.to(device)
    return target


def target_sha256(target: torch.Tensor) -> str:
    """Compute SHA-256 hash of the target vector for provenance."""
    raw = target.detach().float().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------- #
# Layer selection
# --------------------------------------------------------------------------- #

def select_unlearning_layers(
    model: nn.Module,
    forget_loader: Any,
    retain_loader: Any,
    candidate_layers: list[int],
    n_select: int,
    hidden_size: int,
    device: torch.device,
) -> list[int]:
    """Select layers with highest target-vs-retain separability.

    For each candidate layer, compute the linear separability between
    forget representations and retain representations. Select the
    layers with the highest separability scores.

    Parameters
    ----------
    model:
        The model.
    forget_loader:
        DataLoader for forget examples.
    retain_loader:
        DataLoader for retain examples.
    candidate_layers:
        Pre-registered candidate layer indices.
    n_select:
        Number of layers to select.
    hidden_size:
        Hidden dimension size.
    device:
        Compute device.

    Returns
    -------
    selected_layers:
        List of selected layer indices.
    """
    model.eval()
    separability = {}

    for layer_idx in candidate_layers:
        forget_reprs = _collect_representations(
            model, forget_loader, layer_idx, hidden_size, device, n_samples=16,
        )
        retain_reprs = _collect_representations(
            model, retain_loader, layer_idx, hidden_size, device, n_samples=16,
        )

        if forget_reprs is None or retain_reprs is None:
            separability[layer_idx] = 0.0
            continue

        # Simple separability: distance between means
        forget_mean = forget_reprs.mean(dim=0)
        retain_mean = retain_reprs.mean(dim=0)
        dist = (forget_mean - retain_mean).norm().item()
        separability[layer_idx] = dist

    # Sort by separability (descending) and select top-k
    sorted_layers = sorted(separability.items(), key=lambda x: x[1], reverse=True)
    selected = [layer for layer, _ in sorted_layers[:n_select]]

    logger.info(f"Layer separability: {separability}")
    logger.info(f"Selected layers: {selected}")

    return sorted(selected)


def _collect_representations(
    model: nn.Module,
    loader: Any,
    layer_idx: int,
    hidden_size: int,
    device: torch.device,
    n_samples: int = 16,
) -> torch.Tensor | None:
    """Collect hidden representations at a specific layer."""
    representations = []
    hook_output = {}

    def hook_fn(module, input, output):
        # output is typically (batch, seq_len, hidden_size)
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        hook_output["repr"] = h.detach()

    # Find the target layer
    target_layer = None
    layer_count = 0
    for name, module in model.named_modules():
        if _is_transformer_layer(name):
            if layer_count == layer_idx:
                target_layer = module
                break
            layer_count += 1

    if target_layer is None:
        logger.warning(f"Layer {layer_idx} not found (only {layer_count} layers)")
        return None

    handle = target_layer.register_forward_hook(hook_fn)

    for count, batch in enumerate(loader):
        if count >= n_samples:
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

        if "repr" in hook_output:
            # P0-R2MU / P1-3: Extract hidden state at the pre-answer
            # decision position (_prefix_len - 1).
            h = hook_output["repr"]
            if h.dim() == 3:
                # Determine which dim is hidden_size.
                hidden_dim = None
                for d in [1, 2]:
                    if h.shape[d] == hidden_size:
                        hidden_dim = d
                        break
                # P1-3: Determine actual seq dimension.
                if hidden_dim == 2:
                    seq_dim_len = h.shape[1]  # (batch, seq, hidden)
                elif hidden_dim == 1:
                    seq_dim_len = h.shape[2]  # (batch, hidden, seq)
                else:
                    raise RuntimeError(
                        f"Unable to identify hidden dimension for "
                        f"tensor shape {tuple(h.shape)} "
                        f"(expected hidden_size={hidden_size})"
                    )
                decision_pos = get_decision_positions(batch, seq_dim_len)
                batch_idx = torch.arange(h.size(0), device=h.device)
                if hidden_dim == 2:
                    repr_extracted = h[batch_idx, decision_pos]
                elif hidden_dim == 1:
                    repr_extracted = h[batch_idx, :, decision_pos]
            else:
                repr_extracted = h
            representations.append(repr_extracted.cpu())

    handle.remove()

    if not representations:
        return None

    return torch.cat(representations, dim=0)  # (n_samples, hidden_size)


def _is_transformer_layer(name: str) -> bool:
    """Check if a module is a transformer layer.

    Matches patterns like 'model.layers.0' or 'base_model.model.model.layers.0'.
    """
    if ".layers." not in name:
        return False
    # Check if the part after the last '.layers.' is a number
    parts = name.split(".layers.")
    suffix = parts[-1]
    return suffix.isdigit()


# --------------------------------------------------------------------------- #
# Representation loss functions
# --------------------------------------------------------------------------- #

def forget_representation_loss(
    current_repr: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """MSE between current representation and random target.

    Parameters
    ----------
    current_repr:
        Current model hidden representation, shape (batch, hidden_size).
    target:
        Fixed random target vector, shape (hidden_size,).

    Returns
    -------
    loss:
        MSE loss scalar.
    """
    target_expanded = target.unsqueeze(0).expand_as(current_repr)
    return torch.nn.functional.mse_loss(current_repr, target_expanded)


def retain_representation_loss(
    current_repr: torch.Tensor,
    frozen_repr: torch.Tensor,
) -> torch.Tensor:
    """MSE between current and frozen-model representations.

    Parameters
    ----------
    current_repr:
        Current model hidden representation, shape (batch, hidden_size).
    frozen_repr:
        Frozen model hidden representation, shape (batch, hidden_size).

    Returns
    -------
    loss:
        MSE loss scalar.
    """
    return torch.nn.functional.mse_loss(current_repr, frozen_repr)


# --------------------------------------------------------------------------- #
# R²MU-adapted class
# --------------------------------------------------------------------------- #

class R2MUAdapted:
    """R²MU-adapted representation unlearning baseline (B9).

    Cross-domain adaptation: uses only the representation misdirection
    core (no CoT components). Same rank-8 LoRA as E2B to isolate
    output-level vs representation-level objective effects.
    """

    name = "r2mu_adapted"

    def __init__(self, config: R2MUAdaptedConfig):
        self.config = config

    def run(
        self,
        model: nn.Module,
        frozen_model: nn.Module,
        forget_loader: Any,
        retain_loader: Any,
        device: str = "cuda:0",
    ) -> dict[str, Any]:
        """Run the full R²MU-adapted pipeline.

        Parameters
        ----------
        model:
            The model to unlearn (with LoRA adapters).
        frozen_model:
            The frozen reference model (pre-unlearning).
        forget_loader:
            DataLoader for forget examples.
        retain_loader:
            DataLoader for retain examples.
        device:
            Compute device.

        Returns
        -------
        results:
            Pipeline results dict.
        """
        dev = torch.device(device)
        model.to(dev)
        frozen_model.to(dev)
        frozen_model.eval()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get hidden size
        hidden_size = self._get_hidden_size(model)

        # Step 1: Generate random target
        logger.info("Step 1: Generating random target vector ...")
        target = generate_random_target(
            hidden_size=hidden_size,
            seed=self.config.target_seed,
            target_norm=self.config.target_norm,
            device=dev,
        )
        # Match model dtype (e.g. bfloat16) to avoid dtype mismatch in backward
        model_dtype = next(model.parameters()).dtype
        target = target.to(dtype=model_dtype)
        sha = target_sha256(target)
        logger.info(f"Target vector SHA-256: {sha}")

        target_info = {
            "hidden_size": hidden_size,
            "seed": self.config.target_seed,
            "target_norm": self.config.target_norm,
            "sha256": sha,
        }
        with open(output_dir / "representation_targets.json", "w") as f:
            json.dump(target_info, f, indent=2)
            f.write("\n")

        # Step 2: Layer selection
        logger.info("Step 2: Selecting unlearning layers ...")
        selected_layers = select_unlearning_layers(
            model, forget_loader, retain_loader,
            self.config.candidate_layers, self.config.n_select_layers,
            hidden_size, dev,
        )
        logger.info(f"Selected layers: {selected_layers}")

        # Step 3: Training
        logger.info("Step 3: Training with representation objectives ...")
        training_results = self._train(
            model, frozen_model, forget_loader, retain_loader,
            target, selected_layers, dev, output_dir,
        )

        # Propagate adapter_path from training results
        adapter_path = training_results.get("adapter_path")

        return {
            "target_info": target_info,
            "selected_layers": selected_layers,
            "training": training_results,
            "adapter_path": adapter_path,
            # P0-5: Decision-position provenance for B9.
            "representation_position_definition": (
                "last_token_before_ground_truth_answer"
            ),
            "representation_position_source": "_prefix_len",
            "answer_tokens_excluded": True,
        }

    def _train(
        self,
        model: nn.Module,
        frozen_model: nn.Module,
        forget_loader: Any,
        retain_loader: Any,
        target: torch.Tensor,
        selected_layers: list[int],
        device: torch.device,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Train with representation-level objectives."""
        model.train()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config.learning_rate)

        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)
        grad_accum = self.config.gradient_accumulation_steps
        model_dtype = next(model.parameters()).dtype

        losses = []
        forget_repr_distances = []

        # P1-16: Boundary diagnostics for prefix-length validation.
        boundary_n_checked = 0
        boundary_n_valid = 0
        boundary_n_equal_full_length = 0
        boundary_min_prefix_len: int | None = None
        boundary_max_prefix_len: int = 0

        for step in range(1, self.config.num_optimizer_steps + 1):
            optimizer.zero_grad()
            accum_loss = 0.0
            accum_forget_dist = 0.0

            for micro_idx in range(grad_accum):
                # Get forget batch
                try:
                    forget_batch = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    forget_batch = next(forget_iter)

                # Get retain batch
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_batch = next(retain_iter)

                forget_batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in forget_batch.items()
                }
                retain_batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in retain_batch.items()
                }

                # P1-16: Accumulate boundary diagnostics.
                for _batch_for_diag in (forget_batch, retain_batch):
                    _pl = _batch_for_diag.get("_prefix_len")
                    _am = _batch_for_diag.get("attention_mask")
                    if _pl is not None and _am is not None:
                        if torch.is_tensor(_pl):
                            _pl_list = _pl.tolist()
                        elif isinstance(_pl, (list, tuple)):
                            _pl_list = list(_pl)
                        else:
                            _pl_list = [_pl]
                        if torch.is_tensor(_am):
                            _non_pad = _am.sum(dim=1).tolist()
                        else:
                            _non_pad = [0] * len(_pl_list)
                        for _i, (_pv, _np) in enumerate(
                            zip(_pl_list, _non_pad)
                        ):
                            boundary_n_checked += 1
                            _pv_int = int(_pv)
                            if 0 < _pv_int <= _np:
                                boundary_n_valid += 1
                            if _pv_int == _np:
                                boundary_n_equal_full_length += 1
                            if boundary_min_prefix_len is None or (
                                _pv_int < boundary_min_prefix_len
                            ):
                                boundary_min_prefix_len = _pv_int
                            boundary_max_prefix_len = max(
                                boundary_max_prefix_len, _pv_int
                            )

                # Collect representations at selected layers
                forget_reprs = self._hook_representations(
                    model, forget_batch, selected_layers, device,
                )
                retain_reprs_current = self._hook_representations(
                    model, retain_batch, selected_layers, device,
                )

                with torch.no_grad():
                    retain_reprs_frozen = self._hook_representations(
                        frozen_model, retain_batch, selected_layers, device,
                    )

                # Forget loss: MSE to random target
                f_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
                f_dist = 0.0
                for layer_repr in forget_reprs:
                    f_loss = f_loss + forget_representation_loss(layer_repr, target)
                    with torch.no_grad():
                        f_dist += (layer_repr.mean(dim=0).float() - target.float()).norm().item()
                f_loss = f_loss / len(forget_reprs) / grad_accum

                # Retain loss: MSE to frozen model
                r_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
                for curr, frozen in zip(retain_reprs_current, retain_reprs_frozen):
                    r_loss = r_loss + retain_representation_loss(curr, frozen)
                r_loss = r_loss / max(len(retain_reprs_current), 1) / grad_accum

                total = f_loss + self.config.gamma * r_loss
                total.backward()

                accum_loss += total.item()
                accum_forget_dist += f_dist / len(forget_reprs)

            # Clip and step
            torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
            optimizer.step()

            avg_loss = accum_loss / grad_accum
            avg_dist = accum_forget_dist / grad_accum
            losses.append(avg_loss)
            forget_repr_distances.append(avg_dist)

            if step % 25 == 0 or step == 1:
                logger.info(
                    f"  Step {step}: loss={avg_loss:.4f} "
                    f"forget_repr_dist={avg_dist:.4f}"
                )

            # Checkpoint diagnostics + adapter save
            if step in self.config.checkpoint_steps:
                diag = {
                    "step": step,
                    "loss": avg_loss,
                    "forget_repr_distance": avg_dist,
                }
                with open(output_dir / f"diagnostic_step{step}.json", "w") as f:
                    json.dump(diag, f, indent=2)
                    f.write("\n")
                # Save adapter checkpoint at this step
                ckpt_path = output_dir / "checkpoints" / f"adapter_step{step}"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(ckpt_path))
                logger.info(f"Saved adapter checkpoint: {ckpt_path}")

        # Save final adapter checkpoint
        adapter_path = output_dir / "checkpoints" / "adapter_final"
        adapter_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_path))
        logger.info(f"Saved final adapter checkpoint: {adapter_path}")

        return {
            "num_steps": self.config.num_optimizer_steps,
            "final_loss": losses[-1] if losses else None,
            "final_forget_repr_distance": forget_repr_distances[-1] if forget_repr_distances else None,
            "losses": losses,
            "forget_repr_distances": forget_repr_distances,
            "adapter_path": str(adapter_path),
            # P1-16: Boundary diagnostics for B9 auditability.
            "boundary_diagnostics": {
                "n_examples_checked": boundary_n_checked,
                "n_prefix_boundaries_valid": boundary_n_valid,
                "n_prefix_boundaries_equal_full_length": boundary_n_equal_full_length,
                "min_prefix_len": boundary_min_prefix_len,
                "max_prefix_len": boundary_max_prefix_len,
                "answer_tokens_excluded": (
                    boundary_n_valid == boundary_n_checked
                    and boundary_n_equal_full_length == 0
                ),
            },
        }

    def _hook_representations(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        selected_layers: list[int],
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Collect hidden representations at selected layers."""
        representations = []
        hook_outputs = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                hook_outputs[layer_idx] = h
            return hook_fn

        # Register hooks
        handles = []
        layer_count = 0
        for name, module in model.named_modules():
            if _is_transformer_layer(name):
                if layer_count in selected_layers:
                    handles.append(
                        module.register_forward_hook(make_hook(layer_count))
                    )
                layer_count += 1

        # Forward pass
        model_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        for key, value in batch.items():
            if (key not in ("input_ids", "attention_mask", "labels")
                    and not key.startswith("_")
                    and (torch.is_tensor(value) or (isinstance(value, list) and len(value) > 0))):
                model_kwargs[key] = value

        _ = model(**model_kwargs)

        # Remove hooks
        for h in handles:
            h.remove()

        # P0-R2MU / P1-3: Extract representations at the pre-answer
        # decision position (_prefix_len - 1).
        _hs = self._get_hidden_size(model)
        for layer_idx in selected_layers:
            if layer_idx in hook_outputs:
                h = hook_outputs[layer_idx]
                if h.dim() == 3:
                    # Determine which dim is hidden_size.
                    hidden_dim = None
                    for d in [1, 2]:
                        if h.shape[d] == _hs:
                            hidden_dim = d
                            break
                    # P1-3: Determine actual seq dimension.
                    if hidden_dim == 2:
                        seq_dim_len = h.shape[1]
                    elif hidden_dim == 1:
                        seq_dim_len = h.shape[2]
                    else:
                        raise RuntimeError(
                            f"Unable to identify hidden dimension for "
                            f"tensor shape {tuple(h.shape)} "
                            f"(expected hidden_size={_hs})"
                        )
                    decision_pos = get_decision_positions(
                        batch, seq_dim_len,
                    )
                    batch_idx = torch.arange(
                        h.size(0), device=h.device,
                    )
                    if hidden_dim == 2:
                        repr_extracted = h[batch_idx, decision_pos]
                    elif hidden_dim == 1:
                        repr_extracted = h[batch_idx, :, decision_pos]
                else:
                    repr_extracted = h
                representations.append(repr_extracted)

        return representations

    def _get_hidden_size(self, model: nn.Module) -> int:
        """Extract hidden size from model config.

        Handles both simple configs (``config.hidden_size``) and
        multimodal configs like ``Qwen3_5Config`` where the text
        hidden size lives at ``config.text_config.hidden_size``.
        """
        if hasattr(model, "config"):
            cfg = model.config
            # Direct hidden_size (e.g. LlamaConfig, Qwen2Config)
            if hasattr(cfg, "hidden_size"):
                return cfg.hidden_size
            # Multimodal config (e.g. Qwen3_5Config)
            if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
                return cfg.text_config.hidden_size
        # Fallback: infer from lm_head or first language-model linear
        for name, module in model.named_modules():
            if name.endswith("lm_head") and isinstance(module, nn.Linear):
                return module.weight.shape[1]
        raise RuntimeError("Cannot determine hidden size")
