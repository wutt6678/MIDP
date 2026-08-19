"""Candidate sequence scoring (coding plan section 8.4).

For better-calibrated binary decisions, score the full candidate strings
(e.g. "Yes" and "No") conditioned on image + prompt instead of assuming a
single-token answer:

    score(c) = sum_t log P(c_t | image, prompt, c_<t)
    P(yes)   = exp(score_yes) / (exp(score_yes) + exp(score_no))

Candidate scoring is the preferred source for AUROC and calibration metrics;
free generation remains the user-facing behavior metric. Implemented with no
backend-specific assumptions so it is unit-testable with synthetic logits.
"""

from __future__ import annotations

import math
from typing import Any

import torch

# Increment whenever the scoring implementation changes in a way that
# invalidates previously cached scores (e.g. different normalization,
# tokenization, or sequence-log-prob algorithm).  The value is embedded
# in the score-cache key so stale entries are automatically discarded.
SCORING_VERSION = "2"

# Increment when the candidate scoring protocol changes (e.g. different
# candidates, different decision rule, different normalization).
# Embedded in baseline cache provenance for strict invalidation.
CANDIDATE_PROTOCOL_VERSION = "1"


def gather_sequence_log_probs(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
    """Total log-probability of a token sequence.

    Args:
        logits: ``[L, vocab]`` tensor where row ``t`` parameterizes the
            distribution over ``target_ids[t]`` (i.e. logits are already
            aligned/sliced to the positions that predict the candidate).
        target_ids: ``[L]`` long tensor of candidate token ids.

    Returns:
        ``sum_t log P(target_ids[t] | logits[t])`` as a float.
    """
    if logits.dim() != 2 or target_ids.dim() != 1:
        raise ValueError(
            f"Expected logits [L, V] and targets [L]; got {tuple(logits.shape)} / "
            f"{tuple(target_ids.shape)}"
        )
    if logits.shape[0] != target_ids.shape[0]:
        raise ValueError(
            f"Length mismatch: logits {logits.shape[0]} vs targets {target_ids.shape[0]}"
        )
    if target_ids.shape[0] == 0:
        raise ValueError("Empty candidate sequence")
    if (target_ids < 0).any() or (target_ids >= logits.shape[-1]).any():
        raise ValueError("target id outside vocabulary range")
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(gathered.sum().item())


def normalize_binary_scores(scores: dict[str, float]) -> dict[str, float]:
    """Softmax-normalize two (or more) sequence scores into probabilities."""
    if len(scores) < 2:
        raise ValueError("Need at least two candidates to normalize")
    offset = max(scores.values())
    weights = {k: math.exp(v - offset) for k, v in scores.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


def binary_probability(score_positive: float, score_negative: float) -> float:
    """P(positive) from two sequence log-probability scores (numerically stable)."""
    return normalize_binary_scores({"positive": score_positive, "negative": score_negative})[
        "positive"
    ]


def score_candidate_sequence_tensor(
    model: torch.nn.Module,
    prefix: dict[str, torch.Tensor],
    candidate_token_ids: list[int],
    *,
    adapter: Any | None = None,
) -> torch.Tensor:
    """Differentiable candidate sequence scorer for training and evaluation.

    Computes ``sum_j log P(candidate_j | prefix, candidate_<j)`` using the
    exact multimodal assistant prefix and candidate token IDs.

    Parameters
    ----------
    model:
        The language model (typically Qwen3.5-9B or similar).
    prefix:
        Dict containing the multimodal prefix tensors:
        - ``input_ids``: [1, prefix_len]
        - ``attention_mask``: [1, prefix_len]
        - Visual tensors (model-specific: ``pixel_values``, etc.)
        - Sequence-indexed text tensors (e.g. ``mm_token_type_ids``)
    candidate_token_ids:
        List of token IDs for the candidate sequence (e.g., [16484] for "Yes").
    adapter:
        Optional :class:`TrainableVLMAdapter` instance.  When provided,
        the adapter builds the forward dict via ``append_candidate()``,
        making the scorer model-agnostic.  When ``None``, falls back to
        the legacy hardcoded Qwen-style visual field handling.

    Returns
    -------
    log_prob : torch.Tensor
        Differentiable scalar: ``sum_j log P(candidate_j | prefix, candidate_<j)``.

    Alignment
    ---------
    - ``full_input_ids = prefix_input_ids + candidate_ids``
    - Prediction row for candidate token 1: ``prefix_len - 1``
    - Prediction row for candidate token j: ``prefix_len - 1 + j``
    """
    if not candidate_token_ids:
        raise ValueError("candidate_token_ids must be non-empty")

    prefix_input_ids = prefix["input_ids"]
    prefix_len = prefix_input_ids.shape[1]
    if prefix_len == 0:
        raise ValueError("prefix length must be > 0")

    # Build the full forward dict (prefix + candidate)
    if adapter is not None:
        # Adapter-driven path: model-agnostic
        forward_kwargs = adapter.append_candidate(prefix, candidate_token_ids)
    else:
        # Legacy path: hardcoded visual field handling (backward compat)
        forward_kwargs = _build_forward_kwargs_legacy(
            prefix, candidate_token_ids,
        )

    full_input_ids = forward_kwargs["input_ids"]
    m = len(candidate_token_ids)

    # Forward pass (no inference_mode — must support gradients for training)
    outputs = model(**forward_kwargs)
    logits = outputs.logits  # [1, full_len, vocab]

    # Extract prediction rows for candidate tokens
    # Row prefix_len - 1 predicts token prefix_len (first candidate token)
    # Row prefix_len - 1 + j predicts token prefix_len + j
    pred_rows = logits[0, prefix_len - 1: prefix_len - 1 + m, :]
    target_ids = full_input_ids[0, prefix_len: prefix_len + m]

    # Compute log probabilities
    log_probs = torch.log_softmax(pred_rows.float(), dim=-1)
    gathered = log_probs.gather(-1, target_ids.to(log_probs.device).unsqueeze(-1)).squeeze(-1)
    log_prob = gathered.sum()

    return log_prob


def _build_forward_kwargs_legacy(
    prefix: dict[str, torch.Tensor],
    candidate_token_ids: list[int],
) -> dict[str, torch.Tensor]:
    """Build forward kwargs with hardcoded Qwen-style visual fields.

    This is the legacy path used by the Qwen inference backend when
    no adapter is available.  New code should pass an adapter instead.
    """
    prefix_input_ids = prefix["input_ids"]
    device = prefix_input_ids.device
    dtype = prefix_input_ids.dtype

    cand_ids = torch.tensor(
        [candidate_token_ids], dtype=dtype, device=device,
    )

    full_input_ids = torch.cat([prefix_input_ids, cand_ids], dim=1)
    full_attention_mask = torch.cat(
        [prefix["attention_mask"], torch.ones_like(cand_ids)], dim=1,
    )

    forward_kwargs: dict[str, torch.Tensor] = {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask,
    }

    # Extend mm_token_type_ids for candidate tokens (text-only, type 0)
    if "mm_token_type_ids" in prefix:
        prefix_mm = prefix["mm_token_type_ids"]
        cand_mm = torch.zeros_like(cand_ids, dtype=prefix_mm.dtype)
        forward_kwargs["mm_token_type_ids"] = torch.cat([prefix_mm, cand_mm], dim=1)

    # Forward visual tensors from prefix (legacy hardcoded list)
    for key in ("pixel_values", "image_sizes", "image_grid_thw"):
        if key in prefix:
            forward_kwargs[key] = prefix[key]

    return forward_kwargs


def compute_candidate_margin(
    model: torch.nn.Module,
    prefix: dict[str, torch.Tensor],
    yes_token_ids: list[int],
    no_token_ids: list[int],
    expected_answer: bool,
    *,
    adapter: Any | None = None,
) -> torch.Tensor:
    """Compute the candidate margin for forget loss.

    Parameters
    ----------
    model:
        The language model.
    prefix:
        Multimodal prefix tensors.
    yes_token_ids:
        Token IDs for "Yes" (dynamically resolved, not hard-coded).
    no_token_ids:
        Token IDs for "No" (dynamically resolved, not hard-coded).
    expected_answer:
        True if the expected answer is "Yes", False if "No".
    adapter:
        Optional trainable adapter for model-agnostic forward building.
        When provided, uses ``adapter.append_candidate()`` instead of
        the legacy hardcoded path.

    Returns
    -------
    margin : torch.Tensor
        Differentiable scalar: ``M = logP(correct) - logP(wrong)``.
        Minimizing this reduces the candidate margin.

    Forget objective
    ----------------
    - Expected Yes: ``M = logP(Yes) - logP(No)``
    - Expected No:  ``M = logP(No) - logP(Yes)``
    - ``L_forget = mean(M)``
    """
    log_p_yes = score_candidate_sequence_tensor(
        model, prefix, yes_token_ids, adapter=adapter,
    )
    log_p_no = score_candidate_sequence_tensor(
        model, prefix, no_token_ids, adapter=adapter,
    )

    if expected_answer:
        # Expected Yes: margin = logP(Yes) - logP(No)
        margin = log_p_yes - log_p_no
    else:
        # Expected No: margin = logP(No) - logP(Yes)
        margin = log_p_no - log_p_yes

    return margin
