"""Binary-attribute metric suite (coding plan section 8.6).

Every metric is computed per attribute and then macro-averaged. Because CelebA
labels are heavily imbalanced, ordinary accuracy is never the sole headline:
balanced accuracy, macro-F1, AUROC/average-precision, Brier score, and expected
calibration error are all reported together.

All functions operate on plain numpy arrays so they are unit-testable without a
model. ``p_positive`` (candidate-sequence probability) is optional; metrics that
need a score degrade gracefully to ``None`` when it is absent (plan section 8.6:
"AUROC and average precision *when candidate probabilities are available*").
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "compute_binary_metrics",
    "macro_average",
    "expected_calibration_error",
    "brier_score",
    "prevalence",
    "always_negative_baseline",
    "prevalence_threshold_baseline",
]


def _as_bool_array(values: Any) -> np.ndarray:
    return np.asarray(list(values)).astype(bool)


def prevalence(labels: np.ndarray) -> float:
    """Fraction of positive labels (ground-truth or prediction prevalence)."""
    labels = _as_bool_array(labels)
    if labels.size == 0:
        return float("nan")
    return float(labels.mean())


# --------------------------------------------------------------------------- #
# Non-neural baselines (plan section 8.7)
# --------------------------------------------------------------------------- #


def always_negative_baseline(n: int) -> np.ndarray:
    """Predict 0 for every row (the always-negative baseline)."""
    return np.zeros(int(n), dtype=int)


def prevalence_threshold_baseline(
    train_labels: Any, n: int, threshold: float | None = None
) -> np.ndarray:
    """Predict 1 everywhere if training prevalence exceeds a threshold.

    Reproduces the training-partition prevalence baseline (plan 8.7): a single
    global decision derived only from training-set label frequency. ``threshold``
    defaults to 0.5. Never uses validation/test labels.
    """
    prev = prevalence(_as_bool_array(train_labels))
    cutoff = 0.5 if threshold is None else float(threshold)
    value = 1 if prev >= cutoff else 0
    return np.full(int(n), value, dtype=int)


def brier_score(y_true: np.ndarray, p_positive: np.ndarray) -> float:
    """Mean squared error between the binary label and predicted probability."""
    y = _as_bool_array(y_true).astype(float)
    p = np.asarray(list(p_positive), dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {p.shape}")
    if y.size == 0:
        return float("nan")
    p = np.clip(p, 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, p_positive: np.ndarray, n_bins: int = 10
) -> float:
    """Equal-width expected calibration error (ECE).

    Bins predictions by confidence and weights the per-bin |accuracy - confidence|
    gap by the number of samples in each bin.
    """
    y = _as_bool_array(y_true).astype(float)
    p = np.asarray(list(p_positive), dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {p.shape}")
    if y.size == 0:
        return float("nan")
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = float(y.size)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        ece += (n_bin / total) * abs(acc - conf)
    return float(ece)


def _safe_div(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def compute_binary_metrics(
    y_true: Any,
    y_pred: Any,
    p_positive: Any = None,
    parse_failures: int = 0,
    total_queries: int | None = None,
    latency_ms: Any = None,
    ece_bins: int = 10,
) -> dict[str, float | None]:
    """Compute the full per-attribute metric dictionary (plan section 8.6).

    Args:
        y_true: ground-truth 0/1 labels for the evaluated rows.
        y_pred: predicted 0/1 labels for the same rows. Rows whose parse failed
            should be *excluded* from ``y_true``/``y_pred`` and counted via
            ``parse_failures`` instead.
        p_positive: optional candidate probability P(label=1) for AUROC/AP/Brier/ECE.
        parse_failures: number of outputs that failed to parse.
        total_queries: total number of queries attempted (for parse-failure rate).
            Defaults to ``len(y_true) + parse_failures``.
        latency_ms: optional per-query latencies for the mean-latency metric.

    Returns a dict whose keys match the report schema. Metrics requiring a score
    are ``None`` when ``p_positive`` is not provided.
    """
    y = _as_bool_array(y_true)
    p = _as_bool_array(y_pred)
    if y.shape != p.shape:
        raise ValueError(f"y_true/y_pred shape mismatch: {y.shape} vs {p.shape}")

    n = int(y.size)
    out: dict[str, float | None] = {}

    if n == 0:
        # No parseable rows: report zeros/None but keep the schema intact.
        out.update(
            {
                "n": 0,
                "accuracy": None,
                "balanced_accuracy": None,
                "precision_pos": None,
                "recall_pos": None,
                "f1_pos": None,
                "precision_neg": None,
                "recall_neg": None,
                "f1_neg": None,
                "macro_f1": None,
                "brier": None,
                "ece": None,
                "auroc": None,
                "average_precision": None,
                "pred_prevalence": None,
                "gt_prevalence": None,
                "parse_failure_rate": _parse_failure_rate(parse_failures, total_queries, n),
                "mean_latency_ms": float(np.mean(latency_ms)) if latency_ms is not None else None,
            }
        )
        return out

    tp = int(np.logical_and(y, p).sum())
    tn = int(np.logical_and(~y, ~p).sum())
    fp = int(np.logical_and(~y, p).sum())
    fn = int(np.logical_and(y, ~p).sum())

    accuracy = (tp + tn) / n
    sens = _safe_div(tp, tp + fn)          # recall on positives (TPR)
    spec = _safe_div(tn, tn + fp)          # recall on negatives (TNR)
    balanced_accuracy = (
        (sens + spec) / 2 if sens is not None and spec is not None else None
    )

    precision_pos = _safe_div(tp, tp + fp)
    recall_pos = sens
    f1_pos = _f1(precision_pos, recall_pos)
    precision_neg = _safe_div(tn, tn + fn)
    recall_neg = spec
    f1_neg = _f1(precision_neg, recall_neg)
    macro_f1 = (
        (f1_pos + f1_neg) / 2 if f1_pos is not None and f1_neg is not None else None
    )

    out.update(
        {
            "n": n,
            "accuracy": float(accuracy),
            "balanced_accuracy": balanced_accuracy,
            "precision_pos": precision_pos,
            "recall_pos": recall_pos,
            "f1_pos": f1_pos,
            "precision_neg": precision_neg,
            "recall_neg": recall_neg,
            "f1_neg": f1_neg,
            "macro_f1": macro_f1,
            "pred_prevalence": prevalence(p),
            "gt_prevalence": prevalence(y),
            "parse_failure_rate": _parse_failure_rate(parse_failures, total_queries, n),
            "mean_latency_ms": float(np.mean(latency_ms)) if latency_ms is not None else None,
        }
    )

    # Score-dependent metrics.
    if p_positive is not None:
        proba = np.asarray(list(p_positive), dtype=float)
        if proba.shape[0] != n:
            raise ValueError(
                f"p_positive length {proba.shape[0]} != evaluated rows {n}"
            )
        out["brier"] = brier_score(y, proba)
        out["ece"] = expected_calibration_error(y, proba, n_bins=ece_bins)
        out["auroc"] = _auroc(y, proba)
        out["average_precision"] = _average_precision(y, proba)
    else:
        out["brier"] = None
        out["ece"] = None
        out["auroc"] = None
        out["average_precision"] = None

    return out


def _parse_failure_rate(
    parse_failures: int, total_queries: int | None, n_parsed: int
) -> float | None:
    total = total_queries if total_queries is not None else (n_parsed + parse_failures)
    if total <= 0:
        return None
    return float(parse_failures / total)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or (precision + recall) == 0:
        return None if (precision is None or recall is None) else 0.0
    return 2 * precision * recall / (precision + recall)


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """AUROC via the Mann-Whitney U statistic (handles ties with rank averaging)."""
    if len(np.unique(y_true)) < 2:
        return None  # single-class: AUROC undefined
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true.astype(int), scores))
    except ImportError:
        return _mann_whitney_auroc(y_true, scores)


def _mann_whitney_auroc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average ranks for ties.
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = avg
        i = j + 1
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    sum_pos_ranks = ranks[y_true].sum()
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def _average_precision(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_true.astype(int), scores))
    except ImportError:
        order = np.argsort(-scores, kind="mergesort")
        y_sorted = y_true[order]
        tp_cum = np.cumsum(y_sorted)
        total_pos = y_sorted.sum()
        if total_pos == 0:
            return None
        precision_at_k = tp_cum / np.arange(1, len(y_sorted) + 1)
        return float((precision_at_k * y_sorted).sum() / total_pos)


# --------------------------------------------------------------------------- #
# Macro averaging
# --------------------------------------------------------------------------- #

# Keys to macro-average (mean across attributes). ``n`` is summed instead.
_AVERAGE_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "precision_pos",
    "recall_pos",
    "f1_pos",
    "precision_neg",
    "recall_neg",
    "f1_neg",
    "macro_f1",
    "brier",
    "ece",
    "auroc",
    "average_precision",
    "parse_failure_rate",
    "mean_latency_ms",
)


def macro_average(per_attribute: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    """Macro-average the per-attribute metric dicts (plan section 8.6).

    ``n`` is summed across attributes. Score-dependent keys are averaged only
    over attributes that produced a non-None value, so a generation-only run is
    not dragged down by missing AUROC entries.
    """
    out: dict[str, float | None] = {}
    total_n = 0
    for metrics in per_attribute.values():
        total_n += int(metrics.get("n") or 0)
    out["n"] = total_n
    out["attributes"] = len(per_attribute)
    for key in _AVERAGE_KEYS:
        values = [
            metrics[key]
            for metrics in per_attribute.values()
            if metrics.get(key) is not None
        ]
        out[key] = float(np.mean(values)) if values else None
    # Prevalence gaps are more informative than raw prevalence at macro level.
    return out
