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

import torch

# Increment whenever the scoring implementation changes in a way that
# invalidates previously cached scores (e.g. different normalization,
# tokenization, or sequence-log-prob algorithm).  The value is embedded
# in the score-cache key so stale entries are automatically discarded.
SCORING_VERSION = "2"


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
