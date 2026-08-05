"""Per-attribute metrics, macro aggregation, and result persistence."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .config import Config


def compute_metrics(samples: list[dict], preds: list[bool],
                    attrs: list[str]) -> tuple[dict, float, float]:
    """Per-attribute metrics + macro averages.

    Returns (per_attr, macro_accuracy, macro_balanced_accuracy).
    """
    per_attr = {}
    for attr in attrs:
        sel = [i for i, s in enumerate(samples) if s["attribute"] == attr]
        y_true = [samples[i]["label"] for i in sel]
        y_pred = [preds[i] for i in sel]
        pos_rate = sum(y_true) / len(y_true)
        per_attr[attr] = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
            "positive_rate": round(pos_rate, 4),
            "majority_baseline": round(max(pos_rate, 1 - pos_rate), 4),
            "n": len(y_true),
        }
    macro_acc = round(sum(v["accuracy"] for v in per_attr.values())
                      / len(per_attr), 4)
    macro_bal_acc = round(sum(v["balanced_accuracy"] for v in per_attr.values())
                          / len(per_attr), 4)
    return per_attr, macro_acc, macro_bal_acc


def _sanitize(name: str) -> str:
    """Make an id safe for use in a file name (e.g. org/model -> org_model)."""
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_results(cfg: Config, per_attr: dict, macro_acc: float,
                 macro_bal_acc: float, n_images: int) -> tuple[Path, Path]:
    """Write JSON + CSV result files; returns their paths.

    Files are named ``{dataset}_{model}_{date}_{time}.{json,csv}`` unless an
    explicit prefix is given via ``output.name``. Relative output dirs are
    resolved against the project root (celeba_test/).
    """
    out_dir = Path(cfg.output.dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = cfg.output.name or "{}_{}_{}".format(
        _sanitize(cfg.dataset.dataset_id), _sanitize(cfg.model.model_id), stamp)
    result = {
        "config": cfg.to_dict(),
        "model": cfg.model.model_id,
        "dataset": cfg.dataset.dataset_id,
        "split": cfg.dataset.split,
        "n_images": n_images,
        "n_attributes": len(per_attr),
        "seed": cfg.eval.seed,
        "macro_accuracy": macro_acc,
        "macro_balanced_accuracy": macro_bal_acc,
        "per_attribute": per_attr,
    }
    json_path = out_dir / f"{prefix}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    csv_path = out_dir / f"{prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attribute", "accuracy", "balanced_accuracy",
                    "positive_rate", "majority_baseline", "n"])
        for attr, v in per_attr.items():
            w.writerow([attr, v["accuracy"], v["balanced_accuracy"],
                        v["positive_rate"], v["majority_baseline"], v["n"]])
    return json_path, csv_path


def print_results(per_attr: dict, macro_acc: float, macro_bal_acc: float) -> None:
    print("\n===== attribute classification results =====")
    print(f"{'attribute':<22} {'acc':>6} {'bal_acc':>8} {'pos_rate':>9} {'maj_base':>9}")
    for attr, v in per_attr.items():
        print(f"{attr:<22} {v['accuracy']:>6.3f} {v['balanced_accuracy']:>8.3f} "
              f"{v['positive_rate']:>9.3f} {v['majority_baseline']:>9.3f}")
    print(f"\nmacro accuracy:          {macro_acc}")
    print(f"macro balanced accuracy: {macro_bal_acc}")
