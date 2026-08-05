"""Per-attribute confidence calibration (coding plan section 11.2).

Calibrators are fit on CelebA **validation** candidate-sequence scores only, one
per attribute, saved, and then applied frozen to benchmark images. Two methods
are supported:

- ``platt``: Platt scaling (a logistic mapping of the raw score);
- ``isotonic``: isotonic regression (monotone piecewise-constant mapping).

Because of domain shift from CelebA to synthetic/public-figure faces, calibrated
outputs are confidence indicators, not guaranteed probabilities. Both calibrators
are implemented in pure numpy so they serialize to JSON without pickling.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "Calibrator",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "fit_calibrator",
    "save_calibrators",
    "load_calibrators",
]

_METHODS = ("platt", "isotonic")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


# --------------------------------------------------------------------------- #
# Platt scaling
# --------------------------------------------------------------------------- #


@dataclass
class PlattCalibrator:
    """Maps a raw score to a probability with a fitted logistic function."""

    weight: float = 1.0
    bias: float = 0.0
    method: str = "platt"
    n_fit: int = 0

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray, iters: int = 200, lr: float = 0.1):
        scores = np.asarray(scores, dtype=float).ravel()
        labels = np.asarray(labels, dtype=float).ravel()
        if scores.shape != labels.shape or scores.size == 0:
            raise ValueError("scores/labels must be non-empty and same length")
        w, b = 0.0, 0.0
        # Center/scale the score for stable optimization.
        mean, std = float(scores.mean()), float(scores.std() or 1.0)
        x = (scores - mean) / std
        for _ in range(iters):
            p = _sigmoid(w * x + b)
            grad_w = float(np.mean((p - labels) * x))
            grad_b = float(np.mean(p - labels))
            w -= lr * grad_w
            b -= lr * grad_b
        # Fold the standardization back into weight/bias on the raw score.
        return cls(weight=w / std, bias=b - (w * mean) / std, n_fit=int(scores.size))

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        return _sigmoid(self.weight * scores + self.bias)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlattCalibrator":
        return cls(
            weight=float(data["weight"]),
            bias=float(data["bias"]),
            n_fit=int(data.get("n_fit", 0)),
        )


# --------------------------------------------------------------------------- #
# Isotonic regression (pool adjacent violators)
# --------------------------------------------------------------------------- #


@dataclass
class IsotonicCalibrator:
    """Monotone piecewise-linear mapping fit with the PAV algorithm."""

    x_thresholds: list[float] = field(default_factory=list)
    y_values: list[float] = field(default_factory=list)
    method: str = "isotonic"
    n_fit: int = 0

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray):
        scores = np.asarray(scores, dtype=float).ravel()
        labels = np.asarray(labels, dtype=float).ravel()
        if scores.shape != labels.shape or scores.size == 0:
            raise ValueError("scores/labels must be non-empty and same length")
        order = np.argsort(scores, kind="mergesort")
        x = scores[order]
        y = labels[order]
        # Pool adjacent violators via a single pass using stacks.
        stack_y: list[float] = []
        stack_w: list[int] = []
        stack_xsum: list[float] = []
        for xi, yi in zip(x, y):
            stack_y.append(float(yi))
            stack_w.append(1)
            stack_xsum.append(float(xi))
            while len(stack_y) > 1 and stack_y[-2] > stack_y[-1]:
                y2, w2, xs2 = stack_y.pop(), stack_w.pop(), stack_xsum.pop()
                y1, w1, xs1 = stack_y.pop(), stack_w.pop(), stack_xsum.pop()
                stack_y.append((y1 * w1 + y2 * w2) / (w1 + w2))
                stack_w.append(w1 + w2)
                stack_xsum.append(xs1 + xs2)
        # Reconstruct block mean x positions and monotone y values.
        x_thresholds: list[float] = []
        y_values: list[float] = []
        for yv, w, xs in zip(stack_y, stack_w, stack_xsum):
            x_thresholds.append(xs / w)
            y_values.append(float(np.clip(yv, 0.0, 1.0)))
        return cls(x_thresholds=x_thresholds, y_values=y_values, n_fit=int(scores.size))

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if not self.x_thresholds:
            return np.full_like(scores, 0.5)
        x = np.asarray(self.x_thresholds, dtype=float)
        y = np.asarray(self.y_values, dtype=float)
        return np.interp(scores, x, y)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IsotonicCalibrator":
        return cls(
            x_thresholds=[float(v) for v in data["x_thresholds"]],
            y_values=[float(v) for v in data["y_values"]],
            n_fit=int(data.get("n_fit", 0)),
        )


# --------------------------------------------------------------------------- #
# Dispatch / serialization
# --------------------------------------------------------------------------- #

Calibrator = PlattCalibrator | IsotonicCalibrator


def fit_calibrator(method: str, scores: Any, labels: Any) -> Calibrator:
    if method not in _METHODS:
        raise ValueError(f"Unknown calibration method {method!r}; use one of {_METHODS}")
    if method == "platt":
        return PlattCalibrator.fit(scores, labels)
    return IsotonicCalibrator.fit(scores, labels)


def save_calibrators(calibrators: dict[str, Calibrator], path: str | Path) -> None:
    """Persist one calibrator per attribute to a JSON file (atomic)."""
    from ..data.io import write_json

    payload = {attr: cal.to_dict() for attr, cal in sorted(calibrators.items())}
    write_json(payload, path)


def load_calibrators(path: str | Path) -> dict[str, Calibrator]:
    """Load per-attribute calibrators from a JSON file."""
    path = Path(path)
    with open(path) as f:
        payload = json.load(f)
    out: dict[str, Calibrator] = {}
    for attr, data in payload.items():
        method = data.get("method", "platt")
        if method == "platt":
            out[attr] = PlattCalibrator.from_dict(data)
        elif method == "isotonic":
            out[attr] = IsotonicCalibrator.from_dict(data)
        else:
            raise ValueError(f"Unknown calibrator method {method!r} for {attr}")
    return out
