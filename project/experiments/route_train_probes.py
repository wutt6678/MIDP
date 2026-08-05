"""Train layer-wise linear probes on extracted activations (PLAN section
11.8).

For each layer and token group: standardize using training activations
only, fit identity (10-way) and property (binary) logistic regressions,
evaluate on held-out images, report balanced accuracy and cross-entropy.
Repeated for several fit seeds; image-level splits prevent leakage (each
image's representations stay in one fold).

Usage (from repo root):
    python experiments/route_train_probes.py \
        --train results/route_mvp/direct/seed0/activations_train.npz \
        --test results/route_mvp/direct/seed0/activations_validation.npz \
        --output results/route_mvp/direct/seed0/probes.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_activations(npz_path):
    meta_path = Path(npz_path).with_suffix(".json")
    data = np.load(npz_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return data, meta


def identity_labels(index):
    return np.array([row["identity"] for row in index])


def property_labels(index):
    return np.array([1 if row["property"] == "WUG" else 0
                     for row in index])


def fit_probe(x_train, y_train, x_test, y_test, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, log_loss

    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0) + 1e-6
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    # Cross-entropy over a shared label set (handles subsets/smoke runs).
    all_labels = np.unique(np.concatenate([y_train, y_test]))
    proba_full = np.zeros((len(y_test), len(all_labels)))
    for j, c in enumerate(clf.classes_):
        col = int(np.searchsorted(all_labels, c))
        proba_full[:, col] = clf.predict_proba(x_test)[:, j]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "cross_entropy": float(log_loss(y_test, proba_full,
                                        labels=all_labels)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True,
                        help="Training activations .npz (fit set)")
    parser.add_argument("--test", required=True,
                        help="Held-out activations .npz (eval set)")
    parser.add_argument("--condition", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="Probe fit seeds")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_data, train_meta = load_activations(args.train)
    test_data, test_meta = load_activations(args.test)

    layers = train_meta["layers"]
    groups = train_meta["groups"]
    y_train_id = identity_labels(train_meta["index"])
    y_test_id = identity_labels(test_meta["index"])
    y_train_prop = property_labels(train_meta["index"])
    y_test_prop = property_labels(test_meta["index"])

    # Verify no image appears in both folds.
    train_files = {row["image_file"] for row in train_meta["index"]}
    test_files = {row["image_file"] for row in test_meta["index"]}
    overlap = train_files & test_files
    if overlap:
        raise ValueError(f"Image leakage between folds: {len(overlap)} "
                         f"shared image files")
    print(f"Fit: {len(y_train_id)} examples, eval: {len(y_test_id)} "
          f"examples, no image overlap")

    condition = args.condition or train_meta.get("condition", "unknown")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(out_path, "w") as f:
        for li in layers:
            for group in groups:
                key = f"L{li}_{group}"
                if key not in train_data or key not in test_data:
                    continue
                x_train = train_data[key]
                x_test = test_data[key]
                for task, y_tr, y_te in (("identity", y_train_id, y_test_id),
                                         ("property", y_train_prop,
                                          y_test_prop)):
                    runs = [fit_probe(x_train, y_tr, x_test, y_te, s)
                            for s in args.seeds]
                    row = {
                        "condition": condition,
                        "layer": li, "group": group, "task": task,
                        "fit_seeds": args.seeds,
                        "balanced_accuracy_mean": float(np.mean(
                            [r["balanced_accuracy"] for r in runs])),
                        "balanced_accuracy_std": float(np.std(
                            [r["balanced_accuracy"] for r in runs])),
                        "cross_entropy_mean": float(np.mean(
                            [r["cross_entropy"] for r in runs])),
                        "runs": runs,
                    }
                    f.write(json.dumps(row) + "\n")
                    n_written += 1
                    print(f"  L{li:>2} {group:<8} {task:<8} "
                          f"balAcc={row['balanced_accuracy_mean']:.3f}"
                          f"\u00b1{row['balanced_accuracy_std']:.3f}  "
                          f"ce={row['cross_entropy_mean']:.3f}")

    print(f"\nWrote {n_written} probe rows to {out_path}")


if __name__ == "__main__":
    main()
