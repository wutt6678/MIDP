"""Unlearning objective implementations for MLLMU-Bench baselines.

This module implements the UnlearningObjective Protocol and concrete
objective classes for GA, GD, KL, and NPO baselines.

All objectives use answer-only cross-entropy: CE is computed only on
assistant answer tokens, never on prompts, system messages, or image
placeholders.

Public API
----------
.. autoclass:: UnlearningObjective
.. autoclass:: GradientAscent
.. autoclass:: GradientDifference
.. autoclass:: KLMinimization
.. autoclass:: NegativePreferenceOptimization
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.nn.functional as F


class UnlearningObjective(Protocol):
    """Protocol for unlearning objectives.

    Each objective computes a structured loss dict containing the total
    loss and individual components. Unused components may be None.
    """

    name: str

    def compute_loss(
        self,
        model: torch.nn.Module,
        forget_batch: dict[str, Any],
        retain_batch: dict[str, Any] | None = None,
        reference_model: torch.nn.Module | None = None,
        oracle_model: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute the unlearning loss.

        Parameters
        ----------
        model:
            The trainable model (with LoRA adapters).
        forget_batch:
            Batch of forget/target examples.
        retain_batch:
            Batch of retain examples (required for GD, KL).
        reference_model:
            Frozen reference model for KL (pre-unlearning model).
        oracle_model:
            Frozen oracle model for NPO (retain-only fine-tuned).

        Returns
        -------
        loss_dict:
            Dict with keys: total_loss, forget_loss, retain_loss, kl_loss, npo_loss.
            Unused values are None.
        """
        ...


# --------------------------------------------------------------------------- #
# Answer-only cross-entropy helper
# --------------------------------------------------------------------------- #

def answer_only_cross_entropy(
    model: torch.nn.Module,
    batch: dict[str, Any],
) -> torch.Tensor:
    """Compute cross-entropy on assistant answer tokens only.

    Uses the labels tensor where prompt tokens are masked with -100.
    Only positions where labels != -100 contribute to the loss.

    Parameters
    ----------
    model:
        The language model.
    batch:
        Batch dict containing input_ids, attention_mask, labels, and
        multimodal tensors (pixel_values, image_grid_thw, etc.).

    Returns
    -------
    ce_loss:
        Scalar cross-entropy loss over answer tokens only.
    """
    # Build model kwargs with all multimodal tensors
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

    # Forward pass (no labels — we compute loss manually for answer-only masking)
    outputs = model(**model_kwargs)
    logits = outputs.logits  # [B, T, V]

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()

    # Flatten for cross-entropy
    # shift_logits: [B*(T-1), V]
    # shift_labels: [B*(T-1)]
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )

    return loss


# --------------------------------------------------------------------------- #
# Baseline B1: Gradient Ascent (GA)
# --------------------------------------------------------------------------- #

class GradientAscent:
    """Gradient Ascent objective (MLLMU-Bench B1).

    Maximizes the cross-entropy on the forget set:
        loss = -ce_forget

    No retain term. Answer-only CE.
    """

    name = "mllmu_ga"

    def compute_loss(
        self,
        model: torch.nn.Module,
        forget_batch: dict[str, Any],
        retain_batch: dict[str, Any] | None = None,
        reference_model: torch.nn.Module | None = None,
        oracle_model: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute GA loss.

        Returns
        -------
        loss_dict:
            total_loss = -ce_forget
            forget_loss = -ce_forget
            retain_loss = None
            kl_loss = None
            npo_loss = None
        """
        ce_forget = answer_only_cross_entropy(model, forget_batch)
        loss = -ce_forget

        return {
            "total_loss": loss,
            "forget_loss": loss,
            "retain_loss": None,
            "kl_loss": None,
            "npo_loss": None,
        }


# --------------------------------------------------------------------------- #
# Baseline B2: Gradient Difference (GD)
# --------------------------------------------------------------------------- #

class GradientDifference:
    """Gradient Difference objective (MLLMU-Bench B2).

    Combines gradient ascent on forget set with gradient descent on retain set:
        loss = -ce_forget + retain_weight * ce_retain

    For the primary MLLMU-faithful comparison, retain_weight = 1.0 (NOT E2B's 0.1).
    """

    name = "mllmu_ga_difference"

    def __init__(self, retain_weight: float = 1.0):
        """Initialize GD objective.

        Parameters
        ----------
        retain_weight:
            Coefficient for the retain CE term. Default 1.0 for MLLMU-faithful
            comparison (paper uses unweighted sum).
        """
        self.retain_weight = retain_weight

    def compute_loss(
        self,
        model: torch.nn.Module,
        forget_batch: dict[str, Any],
        retain_batch: dict[str, Any] | None = None,
        reference_model: torch.nn.Module | None = None,
        oracle_model: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute GD loss.

        Returns
        -------
        loss_dict:
            total_loss = -ce_forget + retain_weight * ce_retain
            forget_loss = -ce_forget
            retain_loss = ce_retain
            kl_loss = None
            npo_loss = None
        """
        if retain_batch is None:
            raise ValueError("GD requires retain_batch")

        ce_forget = answer_only_cross_entropy(model, forget_batch)
        ce_retain = answer_only_cross_entropy(model, retain_batch)

        forget_loss = -ce_forget
        retain_loss = ce_retain
        total_loss = forget_loss + self.retain_weight * retain_loss

        return {
            "total_loss": total_loss,
            "forget_loss": forget_loss,
            "retain_loss": retain_loss,
            "kl_loss": None,
            "npo_loss": None,
        }


# --------------------------------------------------------------------------- #
# Baseline B3: KL Minimization
# --------------------------------------------------------------------------- #

class KLMinimization:
    """KL Minimization objective (MLLMU-Bench B3).

    Maximizes forget CE while keeping retain predictions close to the
    frozen reference model:
        loss = -ce_forget + kl_weight * kl_retain

    The paper equation uses only the KL term (no second retain CE term).
    Temperature T = 1.0 for the primary baseline.
    """

    name = "mllmu_kl_min"

    def __init__(
        self,
        kl_weight: float = 1.0,
        temperature: float = 1.0,
        include_retain_ce: bool = False,
    ):
        """Initialize KL objective.

        Parameters
        ----------
        kl_weight:
            Coefficient for the KL divergence term.
        temperature:
            Temperature for KL computation. Default 1.0.
        include_retain_ce:
            If True, add retain CE term (for compatibility with official repo).
            Default False for paper-faithful implementation.
        """
        self.kl_weight = kl_weight
        self.temperature = temperature
        self.include_retain_ce = include_retain_ce

    def compute_loss(
        self,
        model: torch.nn.Module,
        forget_batch: dict[str, Any],
        retain_batch: dict[str, Any] | None = None,
        reference_model: torch.nn.Module | None = None,
        oracle_model: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute KL loss.

        Returns
        -------
        loss_dict:
            total_loss = -ce_forget + kl_weight * kl_retain [+ ce_retain if include_retain_ce]
            forget_loss = -ce_forget
            retain_loss = ce_retain (if include_retain_ce else None)
            kl_loss = kl_retain
            npo_loss = None
        """
        if reference_model is None:
            raise ValueError("KL requires reference_model")
        if retain_batch is None:
            raise ValueError("KL requires retain_batch")

        # Forget term: gradient ascent on forget set
        ce_forget = answer_only_cross_entropy(model, forget_batch)
        forget_loss = -ce_forget

        # Retain term: KL divergence to frozen reference on answer tokens
        kl_retain = self._compute_kl_retain(model, reference_model, retain_batch)

        # Total loss
        total_loss = forget_loss + self.kl_weight * kl_retain

        # Optional retain CE term (for compatibility with official repo)
        retain_loss = None
        if self.include_retain_ce:
            retain_loss = answer_only_cross_entropy(model, retain_batch)
            total_loss = total_loss + retain_loss

        return {
            "total_loss": total_loss,
            "forget_loss": forget_loss,
            "retain_loss": retain_loss,
            "kl_loss": kl_retain,
            "npo_loss": None,
        }

    def _compute_kl_retain(
        self,
        model: torch.nn.Module,
        reference_model: torch.nn.Module,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Compute KL divergence on answer tokens only.

        KL(current || reference) over assistant answer positions.
        """
        # Build model kwargs
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

        # Current model forward (with gradients)
        curr_outputs = model(**model_kwargs)
        curr_logits = curr_outputs.logits

        # Reference model forward (frozen, no gradients)
        with torch.no_grad():
            ref_outputs = reference_model(**model_kwargs)
            ref_logits = ref_outputs.logits

        # Mask to answer tokens only
        labels = batch["labels"]
        answer_mask = labels != -100  # [B, T]

        # Shift for next-token prediction
        shift_mask = answer_mask[:, 1:]  # [B, T-1]
        shift_curr = curr_logits[:, :-1, :]  # [B, T-1, V]
        shift_ref = ref_logits[:, :-1, :]  # [B, T-1, V]

        if shift_mask.sum() == 0:
            return torch.tensor(0.0, device=curr_logits.device, requires_grad=True)

        # Apply temperature
        shift_curr_scaled = shift_curr / self.temperature
        shift_ref_scaled = shift_ref / self.temperature

        # Compute KL divergence per token
        # KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))
        # where P = reference, Q = current
        current_log_probs = F.log_softmax(shift_curr_scaled, dim=-1)
        reference_probs = F.softmax(shift_ref_scaled, dim=-1)

        # KL divergence (reduction="none" to get per-token values)
        kl_per_token = F.kl_div(
            current_log_probs,
            reference_probs,
            reduction="none",
        )  # [B, T-1, V]

        # Sum over vocabulary dimension
        kl_per_token = kl_per_token.sum(dim=-1)  # [B, T-1]

        # Mask to answer tokens only
        kl_per_token = kl_per_token * shift_mask.float()

        # Average over answer tokens
        answer_token_count = shift_mask.sum().float()
        kl = kl_per_token.sum() / answer_token_count

        return kl


# --------------------------------------------------------------------------- #
# Baseline B5: Negative Preference Optimization (NPO)
# --------------------------------------------------------------------------- #

class NegativePreferenceOptimization:
    """Negative Preference Optimization objective (MLLMU-Bench B5).

    Treats forget examples as dispreferred data and compares the current
    model to a reference/oracle model (retain-only fine-tuned).

    Loss:
        loss = -(2/beta) * log_sigmoid(-beta * log_ratio)
        where log_ratio = logp_current - logp_oracle

    Paper beta = 0.9 (NOT repo default 0.4).
    """

    name = "mllmu_npo"

    def __init__(self, beta: float = 0.9):
        """Initialize NPO objective.

        Parameters
        ----------
        beta:
            Temperature parameter. Paper value = 0.9.
            Do NOT use repo default 0.4 for primary baseline.
        """
        self.beta = beta

    def compute_loss(
        self,
        model: torch.nn.Module,
        forget_batch: dict[str, Any],
        retain_batch: dict[str, Any] | None = None,
        reference_model: torch.nn.Module | None = None,
        oracle_model: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute NPO loss.

        Returns
        -------
        loss_dict:
            total_loss = npo_loss
            forget_loss = None
            retain_loss = None
            kl_loss = None
            npo_loss = npo_loss
        """
        if oracle_model is None:
            raise ValueError("NPO requires oracle_model")

        # Compute sequence log-probabilities for forget answers
        batch_size = forget_batch["input_ids"].shape[0]
        prefix_lens = forget_batch["_prefix_len"]
        answer_labels = forget_batch["_answer_label"]
        yes_token_ids = forget_batch["_yes_token_ids"][0]
        no_token_ids = forget_batch["_no_token_ids"][0]

        from ..models.scoring import score_candidate_sequence_tensor

        total_npo_loss = torch.tensor(0.0, device=forget_batch["input_ids"].device)

        for i in range(batch_size):
            prefix_len = prefix_lens[i]
            expected_answer = answer_labels[i]

            # Build prefix dict for this sample
            prefix: dict[str, torch.Tensor] = {
                "input_ids": forget_batch["input_ids"][i:i+1, :prefix_len],
                "attention_mask": forget_batch["attention_mask"][i:i+1, :prefix_len],
            }

            # Add multimodal tensors
            _SEQ_KEYS = {"mm_token_type_ids"}
            for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids", "image_sizes"):
                if key in forget_batch:
                    val = forget_batch[key]
                    if torch.is_tensor(val):
                        if key in _SEQ_KEYS:
                            prefix[key] = val[i:i+1, :prefix_len]
                        else:
                            prefix[key] = val[i:i+1]
                    elif isinstance(val, list) and len(val) > i:
                        prefix[key] = val[i]

            # Compute log-probabilities under current model
            log_p_yes_current = score_candidate_sequence_tensor(model, prefix, yes_token_ids)
            log_p_no_current = score_candidate_sequence_tensor(model, prefix, no_token_ids)

            # Compute log-probabilities under oracle model (frozen)
            with torch.no_grad():
                log_p_yes_oracle = score_candidate_sequence_tensor(oracle_model, prefix, yes_token_ids)
                log_p_no_oracle = score_candidate_sequence_tensor(oracle_model, prefix, no_token_ids)

            # Select the correct answer log-prob
            if expected_answer:
                logp_current = log_p_yes_current
                logp_oracle = log_p_yes_oracle
            else:
                logp_current = log_p_no_current
                logp_oracle = log_p_no_oracle

            # Compute log ratio
            log_ratio = logp_current - logp_oracle

            # NPO loss: -(2/beta) * log_sigmoid(-beta * log_ratio)
            npo_loss_i = -(2.0 / self.beta) * F.logsigmoid(-self.beta * log_ratio)
            total_npo_loss = total_npo_loss + npo_loss_i

        # Average over batch
        npo_loss = total_npo_loss / batch_size

        return {
            "total_loss": npo_loss,
            "forget_loss": None,
            "retain_loss": None,
            "kl_loss": None,
            "npo_loss": npo_loss,
        }
