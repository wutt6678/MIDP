#!/usr/bin/env python3
"""E2C build dataset — generate all frozen manifests and training datasets.

Usage:
    python scripts/e2c_build_dataset.py \
        --config e2c/configs/e2c_canonical.yaml \
        --output-dir e2c/manifests \
        --dataset-dir e2c/data/splits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project source is importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "datasets" / "route-unlearning-data" / "src"))
sys.path.insert(0, str(_project_root / "src"))

from route_data.e2c.dataset_builder import (
    build_condition_d,
    build_condition_m,
    build_condition_m_shuffled,
    build_condition_matching_report,
    validate_condition_invariants,
    write_training_jsonl,
)
from route_data.e2c.probe_builder import (
    build_all_probes,
    validate_probes,
    write_probe_jsonl,
)
from route_data.e2c.synthetic_manifest import (
    DEFAULT_SEED,
    assign_aliases,
    generate_e2c_manifests,
    generate_identity_ids,
    load_json_manifest,
    write_json_manifest,
)


def main():
    parser = argparse.ArgumentParser(description="Build E2C frozen manifests and datasets")
    parser.add_argument(
        "--manifest-dir",
        default="e2c/manifests",
        help="Output directory for frozen manifests",
    )
    parser.add_argument(
        "--dataset-dir",
        default="e2c/data/splits",
        help="Output directory for training JSONL files",
    )
    parser.add_argument(
        "--probe-dir",
        default="e2c/data/splits",
        help="Output directory for probe JSONL files",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--image-dir", default=None,
        help="Base directory for images (for path resolution)",
    )
    parser.add_argument(
        "--optimizer-steps", type=int, default=200,
        help="Optimizer step budget for matching report",
    )
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    dataset_dir = Path(args.dataset_dir)
    probe_dir = Path(args.probe_dir)
    seed = args.seed

    print(f"[E2C] Building manifests with seed={seed}")
    print(f"[E2C] Manifest dir: {manifest_dir}")
    print(f"[E2C] Dataset dir: {dataset_dir}")

    # ------------------------------------------------------------------ #
    # Step 1: Generate all frozen manifests
    # ------------------------------------------------------------------ #
    shas = generate_e2c_manifests(
        manifest_dir,
        seed=seed,
        image_dir=args.image_dir,
    )
    print(f"[E2C] Generated {len(shas)} manifests")
    for name, sha in sorted(shas.items()):
        print(f"  {name}: {sha[:16]}...")

    # ------------------------------------------------------------------ #
    # Step 2: Load manifests for dataset building
    # ------------------------------------------------------------------ #
    identity_manifest = load_json_manifest(  # noqa: F841
        manifest_dir / "synthetic_identity_manifest.json"
    )
    image_splits_manifest = load_json_manifest(
        manifest_dir / "e2c_image_split.json"
    )
    true_mapping = load_json_manifest(
        manifest_dir / "synthetic_attribute_mapping.json"
    )
    shuffled_mapping = load_json_manifest(
        manifest_dir / "synthetic_attribute_mapping_shuffled.json"
    )
    wn_pairs = load_json_manifest(
        manifest_dir / "e2c_wrong_name_pairs.json"
    )
    visual_controls = load_json_manifest(
        manifest_dir / "e2c_visual_controls.json"
    )

    # Build lookup structures
    exp_ids, cal_ids = generate_identity_ids()
    alias_map = assign_aliases(exp_ids, cal_ids)

    # Build image split dict: identity_id -> {"train": [...], ...}
    splits_lookup: dict[str, dict[str, list[int]]] = {}
    for rec in image_splits_manifest:
        id_ = rec["identity_id"]
        img_idx = int(rec["image_id"].split("_img_")[-1])
        if id_ not in splits_lookup:
            splits_lookup[id_] = {"train": [], "validation": [], "test": []}
        splits_lookup[id_][rec["split"]].append(img_idx)

    # ------------------------------------------------------------------ #
    # Step 3: Build condition-specific training datasets
    # ------------------------------------------------------------------ #
    print("[E2C] Building condition M records...")
    m_records = build_condition_m(
        alias_map=alias_map,
        true_mapping=true_mapping,
        image_splits=splits_lookup,
        experimental_ids=exp_ids,
        seed=seed,
    )

    print("[E2C] Building condition D records...")
    d_records = build_condition_d(
        alias_map=alias_map,
        true_mapping=true_mapping,
        image_splits=splits_lookup,
        experimental_ids=exp_ids,
        seed=seed,
    )

    print("[E2C] Building condition M-shuffled records...")
    ms_records = build_condition_m_shuffled(
        alias_map=alias_map,
        shuffled_mapping=shuffled_mapping,
        image_splits=splits_lookup,
        experimental_ids=exp_ids,
        seed=seed,
    )

    # ------------------------------------------------------------------ #
    # Step 4: Validate condition invariants
    # ------------------------------------------------------------------ #
    print("[E2C] Validating condition invariants...")
    invariant_report = validate_condition_invariants(
        m_records, d_records, ms_records,
        true_mapping=true_mapping,
        shuffled_mapping=shuffled_mapping,
    )
    print(f"  Invariants: {'PASS' if invariant_report['pass'] else 'FAIL'}")
    print(f"  M: {invariant_report['m_total']} records")
    print(f"  D: {invariant_report['d_total']} records")
    print(f"  M-shuffled: {invariant_report['ms_total']} records")

    # Condition matching report
    matching_report = build_condition_matching_report(
        m_records, d_records, ms_records,
        alias_map=alias_map,
        optimizer_steps=args.optimizer_steps,
    )
    sha = write_json_manifest(
        matching_report, manifest_dir / "e2c_condition_matching_report.json",
    )
    shas["e2c_condition_matching_report"] = sha

    # ------------------------------------------------------------------ #
    # Step 5: Write training JSONL files
    # ------------------------------------------------------------------ #
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("[E2C] Writing training datasets...")
    m_sha = write_training_jsonl(m_records, dataset_dir / "M_train.jsonl")
    d_sha = write_training_jsonl(d_records, dataset_dir / "D_train.jsonl")
    ms_sha = write_training_jsonl(ms_records, dataset_dir / "M_shuffled_train.jsonl")
    shas["M_train"] = m_sha
    shas["D_train"] = d_sha
    shas["M_shuffled_train"] = ms_sha

    # ------------------------------------------------------------------ #
    # Step 6: Build route probes
    # ------------------------------------------------------------------ #
    print("[E2C] Building route probes...")
    probes = build_all_probes(
        image_splits=image_splits_manifest,
        alias_map=alias_map,
        true_mapping=true_mapping,
        wn_pairs=wn_pairs,
        visual_controls=visual_controls,
        experimental_ids=exp_ids,
    )

    # Count test images
    test_image_count = len([
        r for r in image_splits_manifest
        if r["split"] == "test" and r["identity_id"] in set(exp_ids)
    ])

    # Validate probes
    probe_report = validate_probes(
        probes,
        experimental_ids=exp_ids,
        test_image_count=test_image_count,
    )
    print(f"  Probes: {'PASS' if probe_report['pass'] else 'FAIL'}")
    for family, count in probe_report["family_counts"].items():
        print(f"    {family}: {count}")

    # Write probe JSONL files
    probe_dir.mkdir(parents=True, exist_ok=True)
    for family, family_probes in probes.items():
        fname = f"{family}.jsonl"
        sha = write_probe_jsonl(family_probes, probe_dir / fname)
        shas[f"probe_{family}"] = sha

    # ------------------------------------------------------------------ #
    # Step 7: Write leakage report
    # ------------------------------------------------------------------ #
    from route_data.e2c.route_validation import validate_leakage

    print("[E2C] Running leakage validation...")
    test_images = [
        r for r in image_splits_manifest
        if r["split"] == "test" and r["identity_id"] in set(exp_ids)
    ]
    leakage_report = validate_leakage(
        train_records={"M": m_records, "D": d_records, "M_shuffled": ms_records},
        test_images=test_images,
        image_splits=image_splits_manifest,
        wn_pairs=wn_pairs,
        alias_map=alias_map,
        true_mapping=true_mapping,
        shuffled_mapping=shuffled_mapping,
        experimental_ids=exp_ids,
        calibration_ids=cal_ids,
    )
    sha = write_json_manifest(
        leakage_report, manifest_dir / "e2c_leakage_report.json",
    )
    shas["e2c_leakage_report"] = sha
    print(f"  Leakage: {'PASS' if leakage_report['pass'] else 'FAIL'}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n[E2C] Build complete. Manifest SHAs:")
    for name, sha in sorted(shas.items()):
        print(f"  {name}: {sha[:16]}...")

    # Write summary
    summary = {
        "seed": seed,
        "experimental_identities": len(exp_ids),
        "calibration_identities": len(cal_ids),
        "conditions": ["M", "D", "M_shuffled"],
        "manifest_shas": shas,
        "condition_counts": {
            "M": len(m_records),
            "D": len(d_records),
            "M_shuffled": len(ms_records),
        },
        "probe_family_counts": probe_report["family_counts"],
        "leakage_pass": leakage_report["pass"],
        "invariants_pass": invariant_report["pass"],
    }
    write_json_manifest(summary, manifest_dir / "e2c_build_summary.json")

    print(f"\n[E2C] All invariants: {'PASS' if invariant_report['pass'] else 'FAIL'}")
    print(f"[E2C] All probes: {'PASS' if probe_report['pass'] else 'FAIL'}")
    print(f"[E2C] Leakage check: {'PASS' if leakage_report['pass'] else 'FAIL'}")
    print("[E2C] Ready for GPU training.")


if __name__ == "__main__":
    main()
