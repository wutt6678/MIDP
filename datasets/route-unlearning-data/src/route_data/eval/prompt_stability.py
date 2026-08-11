"""Prompt-stability analysis (coding plan section 8.8).

For each attribute at least three semantically equivalent prompts are evaluated
on the validation pilot. This module quantifies how sensitive the model is to the
wording:

- standard deviation of balanced accuracy across prompt variants;
- pairwise prediction agreement;
- flip rate (fraction of images where variants disagree);
- output-format failure rate per variant.

The production prompt is selected on validation data only (highest balanced
accuracy, tie-broken by fewer format failures) and must be frozen before any test
evaluation. Everything here is pure numpy and unit-testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "VariantPredictions",
    "select_production_prompt",
    "stability_report",
]


@dataclass
class VariantPredictions:
    """Aligned predictions for one prompt variant.

    Attributes:
        prompt_id: stable identifier (e.g. ``celeba_binary_v1.Smiling#2``).
        pred: 0/1 predictions for parseable rows (aligned across variants).
        parse_failures: how many outputs of this variant failed to parse.
        total: total attempts (defaults to ``len(pred) + parse_failures``).
    """

    prompt_id: str
    pred: Any
    parse_failures: int = 0
    total: int | None = None

    def __post_init__(self):
        self.pred = np.asarray(list(self.pred)).astype(int)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    y = y_true.astype(bool)
    p = y_pred.astype(bool)
    pos = y
    neg = ~y
    if not pos.any() or not neg.any():
        return None
    tpr = float((p[pos] == y[pos]).mean())
    tnr = float((p[neg] == y[neg]).mean())
    return (tpr + tnr) / 2


def stability_report(
    variants: list[VariantPredictions],
    y_true: Any = None,
) -> dict[str, Any]:
    """Compute the prompt-stability report for one attribute.

    Args:
        variants: two or more aligned :class:`VariantPredictions`.
        y_true: optional ground-truth labels (aligned). Required for balanced
            accuracy statistics; agreement/flip-rate work without it.
    """
    if len(variants) < 2:
        raise ValueError("Prompt-stability analysis needs at least two variants")
    n = variants[0].pred.shape[0]
    for v in variants[1:]:
        if v.pred.shape[0] != n:
            raise ValueError(
                f"Variant {v.prompt_id} has {v.pred.shape[0]} rows; expected {n}"
            )

    stack = np.stack([v.pred for v in variants], axis=0)  # [V, N]

    report: dict[str, Any] = {
        "n_variants": len(variants),
        "n_images": int(n),
        "variant_ids": [v.prompt_id for v in variants],
    }

    # Balanced accuracy per variant (requires ground truth).
    if y_true is not None:
        y = np.asarray(list(y_true)).astype(int)
        if y.shape[0] != n:
            raise ValueError(f"y_true length {y.shape[0]} != {n}")
        balanced = [ _balanced_accuracy(y, v.pred) for v in variants ]
        report["balanced_accuracy_by_variant"] = dict(
            zip([v.prompt_id for v in variants], balanced)
        )
        present = [b for b in balanced if b is not None]
        report["std_balanced_accuracy"] = (
            float(np.std(present)) if len(present) >= 2 else None
        )
        report["mean_balanced_accuracy"] = float(np.mean(present)) if present else None
    else:
        report["balanced_accuracy_by_variant"] = None
        report["std_balanced_accuracy"] = None
        report["mean_balanced_accuracy"] = None

    # Pairwise agreement.
    ids = [v.prompt_id for v in variants]
    pairwise: dict[str, float] = {}
    agreements: list[float] = []
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            if n == 0:
                continue
            agree = float((stack[i] == stack[j]).mean())
            pairwise[f"{ids[i]}|{ids[j]}"] = agree
            agreements.append(agree)
    report["pairwise_agreement"] = pairwise
    report["mean_pairwise_agreement"] = float(np.mean(agreements)) if agreements else None

    # Flip rate: fraction of images where not all variants agree.
    if n == 0:
        report["flip_rate"] = None
    else:
        all_equal = (stack == stack[0:1]).all(axis=0)
        report["flip_rate"] = float(1.0 - all_equal.mean())

    # Format-failure rate per variant.
    failure_rates: dict[str, float | None] = {}
    for v in variants:
        total = v.total if v.total is not None else (v.pred.shape[0] + v.parse_failures)
        failure_rates[v.prompt_id] = (
            float(v.parse_failures / total) if total else None
        )
    report["format_failure_rate_by_variant"] = failure_rates

    return report


def select_production_prompt(
    variants: list[VariantPredictions],
    y_true: Any,
) -> str:
    """Pick the production prompt on validation data only (plan 8.8).

    Highest balanced accuracy wins; ties broken by lower format-failure rate, then
    by prompt_id for determinism.
    """
    y = np.asarray(list(y_true)).astype(int)
    scored: list[tuple[float, float, str]] = []
    for v in variants:
        if v.pred.shape[0] != y.shape[0]:
            raise ValueError(f"Variant {v.prompt_id} misaligned with ground truth")
        bal = _balanced_accuracy(y, v.pred)
        bal = bal if bal is not None else -1.0
        total = v.total if v.total is not None else (v.pred.shape[0] + v.parse_failures)
        fail = (v.parse_failures / total) if total else 1.0
        scored.append((bal, fail, v.prompt_id))
    # Sort by balanced acc desc, failure rate asc, id asc.
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return scored[0][2]
