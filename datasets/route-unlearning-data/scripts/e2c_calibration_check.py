#!/usr/bin/env python3
"""E2C calibration check — quick evaluation of calibration adapters.

Evaluates all 3 conditions on I2N, NAME, DV-syn and visual control probes
to help decide whether the calibration hyperparameters are adequate.

Usage:
    python scripts/e2c_calibration_check.py \
        --cal-dir e2c/outputs/calibration/lr1e-5_steps150 \
        --probe-dir e2c/data/splits \
        --image-base-dir e2c/data/processed \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from route_data.e2c.route_metrics import (
    compute_accuracy_from_probes,
    compute_i2n_accuracy,
)


def load_eval_results(eval_dir: Path) -> dict[str, list[dict]]:
    results = {}
    for family in ["I2N", "NAME", "DV_syn", "IPN_syn", "WN", "VTC", "VISUAL_CONTROL"]:
        path = eval_dir / f"{family}.jsonl"
        if path.exists():
            with open(path) as f:
                results[family] = [json.loads(l) for l in f if l.strip()]
        else:
            results[family] = []
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal-dir", default="e2c/outputs/calibration/lr1e-5_steps150")
    parser.add_argument("--probe-dir", default="e2c/data/splits")
    parser.add_argument("--image-base-dir", default="e2c/data/processed")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cal_dir = Path(args.cal_dir)

    for condition in ["M", "D", "M_shuffled"]:
        adapter_dir = cal_dir / condition / "adapter_final"
        eval_dir = cal_dir / condition / "eval"

        if not adapter_dir.exists():
            print(f"[SKIP] {condition}: no adapter at {adapter_dir}")
            continue

        if not (eval_dir / "eval_summary.json").exists():
            print(f"[PENDING] {condition}: eval not yet run")
            continue

        with open(eval_dir / "eval_summary.json") as f:
            json.load(f)  # validate existence

        results = load_eval_results(eval_dir)

        i2n = compute_i2n_accuracy(results.get("I2N", []))
        name = compute_accuracy_from_probes(results.get("NAME", []))
        dv = compute_accuracy_from_probes(results.get("DV_syn", []))
        vc = compute_accuracy_from_probes(results.get("VISUAL_CONTROL", []))

        print(f"\n{'='*50}")
        print(f"Condition {condition}")
        print(f"  I2N:      {i2n:.3f}  (target >= 0.90)")
        print(f"  NAME:     {name:.3f}  (target >= 0.90 for M, <= 0.65 for D)")
        print(f"  DV-syn:   {dv:.3f}  (target >= 0.80)")
        print(f"  VISUAL-C: {vc:.3f}  (should remain high)")
        print(f"{'='*50}")

        # Quick assessment
        if condition == "M":
            if i2n >= 0.90 and name >= 0.90:
                print("  -> M mappings LEARNED")
            else:
                print("  -> M mappings NOT YET LEARNED (need more steps or higher LR)")
        elif condition == "D":
            if dv >= 0.80 and name <= 0.65:
                print("  -> D direct learned, NAME appropriately low")
            elif dv >= 0.80:
                print("  -> D direct learned but NAME may be too high")
            else:
                print("  -> D needs more training")


if __name__ == "__main__":
    main()
