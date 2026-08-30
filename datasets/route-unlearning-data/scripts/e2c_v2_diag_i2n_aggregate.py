#!/usr/bin/env python3
"""Aggregate the E2C-v2 I2N capacity sweep into the N -> I2N(N) curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-base", default="e2c_v2/outputs/diag_i2n_capacity")
    args = parser.parse_args()
    out_base = Path(args.out_base)

    curve = []
    for n in (2, 4, 6, 8, 10):
        path = out_base / f"n{n}" / "diagnostic_summary.json"
        if not path.exists():
            curve.append({"n": n, "status": "MISSING"})
            continue
        with open(path) as f:
            s = json.load(f)
        curve.append({
            "n": n,
            "steps": s["steps"],
            "final_loss": round(s["final_loss"], 4),
            "train_free_gen": round(s["train_free_gen_accuracy"], 3),
            "val_free_gen": round(s["validation_free_gen_accuracy"], 3),
            "test_free_gen": round(s["test_free_gen_accuracy"], 3),
            "train_rank": round(s["train_candidate_rank_accuracy"], 3),
            "val_rank": round(s["validation_candidate_rank_accuracy"], 3),
            "test_rank": round(s["test_candidate_rank_accuracy"], 3),
            "confusion_matrix": s["test_confusion_matrix"],
            "confusion_labels": s["confusion_labels"],
        })

    report = {"experiment": "I2N binding capacity curve (M1-only)",
              "fixed": {"lora_rank": 8, "scope": "language_attention_only",
                        "lr": 2e-5, "grad_accum": 8, "seed": 17,
                        "steps_per_identity": 100, "split": "10/3/3",
                        "train_prompt": "Identify the synthetic person shown here. Return only the alias.",
                        "eval_prompt": "Who is shown? Answer with the synthetic identity name only."},
              "curve": curve}
    with open(out_base / "capacity_curve.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print("N | steps | loss  | train | val   | test  | rank(test)")
    print("--+-------+-------+-------+-------+-------+-----------")
    for c in curve:
        if "status" in c:
            print(f"{c['n']:>1} | MISSING")
            continue
        print(f"{c['n']:>1} | {c['steps']:>5} | {c['final_loss']:.3f} | "
              f"{c['train_free_gen']:.3f} | {c['val_free_gen']:.3f} | "
              f"{c['test_free_gen']:.3f} | {c['test_rank']:.3f}")


if __name__ == "__main__":
    main()
