#!/usr/bin/env python3
"""E2C-v2 build dataset — generates calibration + experimental datasets with
full lineage, SHA validation, population isolation, and preflight gates.

Produces:
    e2c_v2/
      data/
        calibration/{M,D,M_shuffled}_train.jsonl + probes/
        experimental/{M,D,M_shuffled}_train.jsonl + probes/
      manifests/  (all frozen JSON manifests)
      reports/e2c_v2_preflight_report.json

Usage:
    # Draft build (generates manifests, marks audit pending)
    python scripts/e2c_build_dataset_v2.py --mode draft

    # Finalize (computes SHAs from real images, validates lineage)
    python scripts/e2c_build_dataset_v2.py --mode finalize --image-base-dir e2c/data/processed

    # Preflight (runs all CPU gates, writes preflight report)
    python scripts/e2c_build_dataset_v2.py --mode preflight --image-base-dir e2c/data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from route_data.e2c.dataset_builder import (
    build_condition_d,
    build_condition_m,
    build_condition_m_shuffled,
    validate_audit_completeness,
    validate_condition_invariants,
    validate_population_isolation,
    write_training_jsonl,
)
from route_data.e2c.probe_builder import (
    build_all_probes,
    validate_probes,
    write_probe_jsonl,
)
from route_data.e2c.route_validation import validate_leakage
from route_data.e2c.synthetic_manifest import (
    DEFAULT_SEED,
    assign_aliases,
    finalize_alias_tokenization,
    generate_calibration_mapping,
    generate_identity_ids,
    generate_image_splits,
    generate_shuffled_mapping,
    generate_true_mapping,
    generate_wrong_name_pairs,
    load_json_manifest,
    validate_alias_tokenization,
    write_json_manifest,
)


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Draft build
# --------------------------------------------------------------------------- #

def cmd_draft(args):
    """Generate all manifests with pending audit and empty SHAs."""
    out = Path(args.output_dir)
    manifest_dir = out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    shas: dict[str, str] = {}

    exp_ids, cal_ids = generate_identity_ids()
    all_ids = exp_ids + cal_ids
    alias_map = assign_aliases(exp_ids, cal_ids)

    # Identity manifest (alias_token_ids filled later)
    identity_records = []
    for id_ in all_ids:
        role = "experimental" if id_ in exp_ids else "calibration"
        identity_records.append({
            "identity_id": id_,
            "alias": alias_map[id_],
            "alias_token_ids": [],
            "alias_token_count": 0,
            "tokenizer_id": "",
            "tokenizer_revision": "",
            "role": role,
        })
    sha = write_json_manifest(identity_records, manifest_dir / "synthetic_identity_manifest.json")
    shas["synthetic_identity_manifest"] = sha

    # Image splits
    splits = generate_image_splits(all_ids, seed=seed)
    image_split_records = []
    for id_ in all_ids:
        for split_name in ("train", "validation", "test"):
            for idx in splits[id_][split_name]:
                img_id = f"{id_}_img_{idx:03d}"
                # Fix 1: path relative to image_base_dir (not repo-root)
                img_path = f"{id_}/{img_id}.png"
                image_split_records.append({
                    "identity_id": id_,
                    "image_id": img_id,
                    "image_path": img_path,
                    "image_sha256": "",
                    # Fix 2: source_render_id empty until ingestion populates
                    "source_render_id": "",
                    "generation_type": "",
                    "augmentation_parent_id": None,
                    "split": split_name,
                })
    sha = write_json_manifest(image_split_records, manifest_dir / "e2c_image_split.json")
    shas["e2c_image_split"] = sha

    # Experimental mappings
    true_mapping = generate_true_mapping(exp_ids, seed=seed)
    sha = write_json_manifest(true_mapping, manifest_dir / "synthetic_attribute_mapping.json")
    shas["synthetic_attribute_mapping"] = sha

    shuffled_mapping = generate_shuffled_mapping(true_mapping, seed=seed)
    sha = write_json_manifest(shuffled_mapping, manifest_dir / "synthetic_attribute_mapping_shuffled.json")
    shas["synthetic_attribute_mapping_shuffled"] = sha

    # Calibration mappings
    cal_true, cal_shuf = generate_calibration_mapping(cal_ids, seed=seed)
    sha = write_json_manifest(cal_true, manifest_dir / "synthetic_attribute_mapping_calibration.json")
    shas["synthetic_attribute_mapping_calibration"] = sha
    sha = write_json_manifest(cal_shuf, manifest_dir / "synthetic_attribute_mapping_calibration_shuffled.json")
    shas["synthetic_attribute_mapping_calibration_shuffled"] = sha

    # Wrong-name pairs (experimental only)
    wn_pairs = generate_wrong_name_pairs(true_mapping, alias_map, seed=seed)
    sha = write_json_manifest(wn_pairs, manifest_dir / "e2c_wrong_name_pairs.json")
    shas["e2c_wrong_name_pairs"] = sha

    # Identity audit (fail-closed: all pending)
    audit_records = []
    for id_ in all_ids:
        for idx in range(16):
            img_id = f"{id_}_img_{idx:03d}"
            audit_records.append({
                "identity_id": id_,
                "image_id": img_id,
                "audit_status": "pending",
                "reviewer": None,
                "review_timestamp": None,
                "identity_consistent": None,
                "duplicate": None,
                "corrupted": None,
                "watermark": None,
                "alias_leakage": None,
                "target_fact_leakage": None,
                "notes": "",
            })
    sha = write_json_manifest(audit_records, manifest_dir / "e2c_identity_audit.json")
    shas["e2c_identity_audit"] = sha

    # Visual controls (Fix 7: mark as pending until real metadata ingested)
    visual_control_records = []
    for id_ in all_ids:
        for idx in range(16):
            img_id = f"{id_}_img_{idx:03d}"
            visual_control_records.append({
                "image_id": img_id,
                "identity_id": id_,
                "controls": {"smiling": None, "eyeglasses": None, "hat": None},
                "source": "pending",
            })
    sha = write_json_manifest(visual_control_records, manifest_dir / "e2c_visual_controls.json")
    shas["e2c_visual_controls"] = sha

    print(f"[E2C-v2] Draft build complete. {len(shas)} manifests written.")
    for name, s in sorted(shas.items()):
        print(f"  {name}: {s[:16]}...")
    print("[E2C-v2] Next: attach real images, then run --mode finalize")


# --------------------------------------------------------------------------- #
# Finalize: compute SHAs, build datasets/probes
# --------------------------------------------------------------------------- #

def cmd_finalize(args):
    """Compute image SHAs, build calibration + experimental datasets and probes."""
    out = Path(args.output_dir)
    manifest_dir = out / "manifests"
    image_base = Path(args.image_base_dir)
    seed = args.seed

    # Load manifests
    image_splits = load_json_manifest(manifest_dir / "e2c_image_split.json")
    alias_manifest = load_json_manifest(manifest_dir / "synthetic_identity_manifest.json")
    exp_true = load_json_manifest(manifest_dir / "synthetic_attribute_mapping.json")
    exp_shuf = load_json_manifest(manifest_dir / "synthetic_attribute_mapping_shuffled.json")
    cal_true = load_json_manifest(manifest_dir / "synthetic_attribute_mapping_calibration.json")
    cal_shuf = load_json_manifest(manifest_dir / "synthetic_attribute_mapping_calibration_shuffled.json")

    exp_ids, cal_ids = generate_identity_ids()
    alias_map = {r["identity_id"]: r["alias"] for r in alias_manifest}
    splits_lookup = _build_splits_lookup(image_splits)

    # Compute SHA-256 for every image
    print("[E2C-v2] Computing image SHA-256 hashes...")
    missing_count = 0
    for rec in image_splits:
        img_path = image_base / rec["image_path"]
        if img_path.exists():
            rec["image_sha256"] = _sha256_file(img_path)
        else:
            missing_count += 1
            rec["image_sha256"] = ""

    if missing_count:
        print(f"[WARNING] {missing_count} image files not found")

    # Write updated split manifest
    sha = write_json_manifest(image_splits, manifest_dir / "e2c_image_split.json")
    print(f"  e2c_image_split SHA: {sha[:16]}...")

    # Build experimental datasets
    print("[E2C-v2] Building experimental datasets...")
    exp_dir = out / "data" / "experimental"
    exp_dir.mkdir(parents=True, exist_ok=True)

    exp_m = build_condition_m(
        alias_map=alias_map, true_mapping=exp_true,
        image_splits=splits_lookup, identity_ids=exp_ids, seed=seed,
    )
    exp_d = build_condition_d(
        alias_map=alias_map, true_mapping=exp_true,
        image_splits=splits_lookup, identity_ids=exp_ids, seed=seed,
    )
    exp_ms = build_condition_m_shuffled(
        alias_map=alias_map, shuffled_mapping=exp_shuf,
        image_splits=splits_lookup, identity_ids=exp_ids, seed=seed,
    )

    # Validate experimental condition invariants
    validate_condition_invariants(
        exp_m, exp_d, exp_ms,
        true_mapping=exp_true, shuffled_mapping=exp_shuf,
    )
    print("  Experimental invariants: PASS")

    # Write JSONL
    write_training_jsonl(exp_m, exp_dir / "M_train.jsonl")
    write_training_jsonl(exp_d, exp_dir / "D_train.jsonl")
    write_training_jsonl(exp_ms, exp_dir / "M_shuffled_train.jsonl")

    # Build experimental probes
    print("[E2C-v2] Building experimental probes...")
    test_images = [r for r in image_splits
                   if r["split"] == "test" and r["identity_id"] in set(exp_ids)]
    wn_pairs = generate_wrong_name_pairs(exp_true, alias_map, seed=seed)
    visual_controls = load_json_manifest(manifest_dir / "e2c_visual_controls.json")

    exp_probes = build_all_probes(
        image_splits=image_splits, alias_map=alias_map,
        true_mapping=exp_true, wn_pairs=wn_pairs,
        visual_controls=visual_controls, experimental_ids=exp_ids,
    )
    test_img_count = len(test_images)
    validate_probes(exp_probes, experimental_ids=exp_ids,
                    test_image_count=test_img_count)
    print("  Experimental probes: PASS")

    exp_probe_dir = exp_dir / "probes"
    exp_probe_dir.mkdir(exist_ok=True)
    for family, probes in exp_probes.items():
        write_probe_jsonl(probes, exp_probe_dir / f"{family}.jsonl")

    # Build calibration datasets
    print("[E2C-v2] Building calibration datasets...")
    cal_dir = out / "data" / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    # Merge cal_true into exp_true for the combined mapping needed by builders
    combined_true = {**exp_true, **cal_true}
    combined_shuf = {**exp_shuf, **cal_shuf}

    cal_m = build_condition_m(
        alias_map=alias_map, true_mapping=combined_true,
        image_splits=splits_lookup, identity_ids=cal_ids, seed=seed,
    )
    cal_d = build_condition_d(
        alias_map=alias_map, true_mapping=combined_true,
        image_splits=splits_lookup, identity_ids=cal_ids, seed=seed,
    )
    cal_ms = build_condition_m_shuffled(
        alias_map=alias_map, shuffled_mapping=combined_shuf,
        image_splits=splits_lookup, identity_ids=cal_ids, seed=seed,
    )

    write_training_jsonl(cal_m, cal_dir / "M_train.jsonl")
    write_training_jsonl(cal_d, cal_dir / "D_train.jsonl")
    write_training_jsonl(cal_ms, cal_dir / "M_shuffled_train.jsonl")

    # Build calibration probes
    print("[E2C-v2] Building calibration probes...")
    cal_wn = generate_wrong_name_pairs(cal_true, alias_map, seed=seed)
    cal_probes = build_all_probes(
        image_splits=image_splits, alias_map=alias_map,
        true_mapping=combined_true, wn_pairs=cal_wn,
        visual_controls=visual_controls, experimental_ids=cal_ids,
    )
    cal_probe_dir = cal_dir / "probes"
    cal_probe_dir.mkdir(exist_ok=True)
    for family, probes in cal_probes.items():
        write_probe_jsonl(probes, cal_probe_dir / f"{family}.jsonl")

    # Population isolation
    print("[E2C-v2] Validating population isolation...")
    iso_report = validate_population_isolation(
        calibration_records=cal_m + cal_d + cal_ms,
        experimental_records=exp_m + exp_d + exp_ms,
        calibration_ids=cal_ids,
        experimental_ids=exp_ids,
    )
    print(f"  Population isolation: {'PASS' if iso_report['pass'] else 'FAIL'}")

    # Fix 5: Calibration condition invariants
    print("[E2C-v2] Validating calibration condition invariants...")
    try:
        validate_condition_invariants(
            cal_m, cal_d, cal_ms,
            true_mapping=cal_true, shuffled_mapping=cal_shuf,
        )
        print("  Calibration invariants: PASS")
    except ValueError as e:
        print(f"  Calibration invariants: FAIL\n    {e}")

    # Fix 5: Calibration probe validation
    print("[E2C-v2] Validating calibration probes...")
    cal_test_count = len([r for r in image_splits
                          if r["split"] == "test"
                          and r["identity_id"] in set(cal_ids)])
    try:
        validate_probes(cal_probes, experimental_ids=cal_ids,
                        test_image_count=cal_test_count)
        print("  Calibration probes: PASS")
    except ValueError as e:
        print(f"  Calibration probes: FAIL\n    {e}")

    print("\n[E2C-v2] Finalize complete.")
    print(f"  Experimental: {len(exp_m)} M, {len(exp_d)} D, {len(exp_ms)} M_shuf records")
    print(f"  Calibration:  {len(cal_m)} M, {len(cal_d)} D, {len(cal_ms)} M_shuf records")
    print("  Next: complete manual audit, then run --mode preflight")


# --------------------------------------------------------------------------- #
# Preflight: run all CPU gates
# --------------------------------------------------------------------------- #

def cmd_preflight(args):
    """Run all CPU-only validation gates and write preflight report."""
    out = Path(args.output_dir)
    manifest_dir = out / "manifests"

    print("[E2C-v2] Running preflight validation...")
    gates: dict[str, dict[str, Any]] = {}

    # Load manifests
    image_splits = load_json_manifest(manifest_dir / "e2c_image_split.json")
    audit = load_json_manifest(manifest_dir / "e2c_identity_audit.json")
    alias_manifest = load_json_manifest(manifest_dir / "synthetic_identity_manifest.json")

    exp_ids, cal_ids = generate_identity_ids()

    # Gate 1: Image metadata completeness
    print("  Gate 1: Image metadata completeness...")
    meta_errors = []
    for rec in image_splits:
        if not rec.get("image_sha256"):
            meta_errors.append(f"{rec['image_id']}: missing SHA")
        if not rec.get("source_render_id"):
            meta_errors.append(f"{rec['image_id']}: missing source_render_id")
        if "generation_type" not in rec:
            meta_errors.append(f"{rec['image_id']}: missing generation_type")
    gates["image_metadata"] = {
        "pass": len(meta_errors) == 0,
        "n_errors": len(meta_errors),
        "errors": meta_errors[:10],
    }

    # Gate 2: SHA-level leakage
    print("  Gate 2: SHA-level leakage...")
    exp_true = load_json_manifest(manifest_dir / "synthetic_attribute_mapping.json")
    exp_shuf = load_json_manifest(manifest_dir / "synthetic_attribute_mapping_shuffled.json")
    alias_map = {r["identity_id"]: r["alias"] for r in alias_manifest}
    splits_lookup = _build_splits_lookup(image_splits)

    # Build minimal train records for leakage check
    exp_m = build_condition_m(
        alias_map=alias_map, true_mapping=exp_true,
        image_splits=splits_lookup, identity_ids=exp_ids, seed=args.seed,
    )
    exp_d = build_condition_d(
        alias_map=alias_map, true_mapping=exp_true,
        image_splits=splits_lookup, identity_ids=exp_ids, seed=args.seed,
    )
    exp_ms = build_condition_m_shuffled(
        alias_map=alias_map, shuffled_mapping=exp_shuf,
        image_splits=splits_lookup, identity_ids=exp_ids, seed=args.seed,
    )
    test_images = [r for r in image_splits
                   if r["split"] == "test" and r["identity_id"] in set(exp_ids)]
    wn_pairs = generate_wrong_name_pairs(exp_true, alias_map, seed=args.seed)

    try:
        leakage_report = validate_leakage(
            train_records={"M": exp_m, "D": exp_d, "M_shuffled": exp_ms},
            test_images=test_images,
            image_splits=image_splits,
            wn_pairs=wn_pairs,
            alias_map=alias_map,
            true_mapping=exp_true,
            shuffled_mapping=exp_shuf,
            experimental_ids=exp_ids,
            calibration_ids=cal_ids,
        )
        gates["leakage"] = leakage_report
    except ValueError as e:
        gates["leakage"] = {"pass": False, "error": str(e)[:500]}

    # Gate 3: Audit completeness (hardened: semantic + coverage check)
    print("  Gate 3: Identity audit completeness...")
    expected_img_ids = [r["image_id"] for r in image_splits]
    audit_report = validate_audit_completeness(
        audit, expected_image_ids=expected_img_ids,
    )
    gates["audit"] = audit_report

    # Gate 4: Alias tokenization
    print("  Gate 4: Alias tokenization...")
    if alias_manifest and alias_manifest[0].get("alias_token_ids"):
        tok_report = validate_alias_tokenization(alias_manifest)
        gates["alias_tokenization"] = tok_report
    else:
        gates["alias_tokenization"] = {
            "pass": False,
            "errors": ["Alias token IDs not yet populated"],
        }

    # Gate 5: Population isolation (Fix 6: check ALL 3 conditions)
    print("  Gate 5: Population isolation...")
    cal_true = load_json_manifest(manifest_dir / "synthetic_attribute_mapping_calibration.json")
    cal_shuf = load_json_manifest(manifest_dir / "synthetic_attribute_mapping_calibration_shuffled.json")
    combined_true = {**exp_true, **cal_true}
    combined_shuf = {**exp_shuf, **cal_shuf}
    cal_m = build_condition_m(
        alias_map=alias_map, true_mapping=combined_true,
        image_splits=splits_lookup, identity_ids=cal_ids, seed=args.seed,
    )
    cal_d = build_condition_d(
        alias_map=alias_map, true_mapping=combined_true,
        image_splits=splits_lookup, identity_ids=cal_ids, seed=args.seed,
    )
    cal_ms = build_condition_m_shuffled(
        alias_map=alias_map, shuffled_mapping=combined_shuf,
        image_splits=splits_lookup, identity_ids=cal_ids, seed=args.seed,
    )
    all_cal = cal_m + cal_d + cal_ms
    all_exp = exp_m + exp_d + exp_ms
    iso_report = validate_population_isolation(
        calibration_records=all_cal,
        experimental_records=all_exp,
        calibration_ids=cal_ids,
        experimental_ids=exp_ids,
    )
    gates["population_isolation"] = iso_report

    # Gate 6: VTC semantic validation
    print("  Gate 6: VTC semantic validation...")
    from route_data.e2c.probe_builder import build_vtc_probes
    vtc_probes = build_vtc_probes(test_images, wn_pairs, alias_map, exp_true)
    from route_data.e2c.route_validation import validate_vtc_semantics
    vtc_report = validate_vtc_semantics(vtc_probes)
    gates["vtc_semantics"] = vtc_report

    # Gate 7: Condition invariants
    print("  Gate 7: Condition invariants...")
    try:
        cond_report = validate_condition_invariants(
            exp_m, exp_d, exp_ms,
            true_mapping=exp_true, shuffled_mapping=exp_shuf,
        )
        gates["condition_invariants"] = cond_report
    except ValueError as e:
        gates["condition_invariants"] = {"pass": False, "error": str(e)[:500]}

    # Gate 8: Visual controls metadata (Fix 7)
    print("  Gate 8: Visual controls metadata...")
    vc_records = load_json_manifest(manifest_dir / "e2c_visual_controls.json")
    vc_errors = []
    mandatory_families = ["smiling", "eyeglasses", "hat"]
    for rec in vc_records:
        controls = rec.get("controls", {})
        for fam in mandatory_families:
            val = controls.get(fam)
            if val is None:
                vc_errors.append(
                    f"{rec['image_id']}: {fam} is None (pending)")
            elif not isinstance(val, bool):
                vc_errors.append(
                    f"{rec['image_id']}: {fam}={val!r} not bool")
    gates["visual_controls"] = {
        "pass": len(vc_errors) == 0,
        "n_errors": len(vc_errors),
        "errors": vc_errors[:10],
    }

    # Gate 9: generation_type enum (Fix 8)
    print("  Gate 9: generation_type validation...")
    valid_gen_types = {"independent_render", "augmentation"}
    gen_errors = []
    for rec in image_splits:
        gt = rec.get("generation_type", "")
        if not gt:
            gen_errors.append(
                f"{rec['image_id']}: generation_type is empty")
        elif gt not in valid_gen_types:
            gen_errors.append(
                f"{rec['image_id']}: generation_type={gt!r} not in "
                f"{valid_gen_types}")
        if gt == "augmentation" and not rec.get("augmentation_parent_id"):
            gen_errors.append(
                f"{rec['image_id']}: augmentation without parent_id")
    gates["generation_type"] = {
        "pass": len(gen_errors) == 0,
        "n_errors": len(gen_errors),
        "errors": gen_errors[:10],
    }

    # Overall
    all_pass = all(g.get("pass", False) for g in gates.values())
    preflight = {
        "schema_version": "e2c_v2_preflight_v1",
        "overall_pass": all_pass,
        "gates": gates,
        "canonical_training_allowed": all_pass,
    }

    report_dir = out / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "e2c_v2_preflight_report.json", "w") as f:
        json.dump(preflight, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"\n{'='*50}")
    print("E2C-v2 Preflight Report")
    print(f"{'='*50}")
    for gate_name, gate in sorted(gates.items()):
        status = "PASS" if gate.get("pass") else "FAIL"
        print(f"  {gate_name}: {status}")
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"  Canonical training: {'ALLOWED' if all_pass else 'BLOCKED'}")

    if not all_pass:
        print("\nFailed gates:")
        for gate_name, gate in sorted(gates.items()):
            if not gate.get("pass"):
                errs = gate.get("errors", gate.get("error", ""))
                if isinstance(errs, list):
                    for e in errs[:5]:
                        print(f"  [{gate_name}] {e}")
                else:
                    print(f"  [{gate_name}] {errs}")

    return 0 if all_pass else 1


# --------------------------------------------------------------------------- #
# Tokenize aliases (Fix 4)
# --------------------------------------------------------------------------- #

def cmd_tokenize_aliases(args):
    """Tokenize all aliases with the frozen Qwen tokenizer."""
    out = Path(args.output_dir)
    manifest_dir = out / "manifests"

    print(f"[E2C-v2] Loading tokenizer: {args.tokenizer} @ {args.tokenizer_revision}")
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
    )
    tokenizer = getattr(proc, "tokenizer", proc)

    alias_manifest = load_json_manifest(
        manifest_dir / "synthetic_identity_manifest.json"
    )
    alias_map = {r["identity_id"]: r["alias"] for r in alias_manifest}

    print(f"[E2C-v2] Tokenizing {len(alias_map)} aliases...")
    token_records = finalize_alias_tokenization(
        alias_map, tokenizer,
        tokenizer_id=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
    )

    # Validate
    tok_report = validate_alias_tokenization(token_records)
    print(f"  Tokenization: {'PASS' if tok_report['pass'] else 'FAIL'}")
    print(f"  min={tok_report['min_token_count']}, "
          f"max={tok_report['max_token_count']}, "
          f"mean={tok_report['mean_token_count']:.1f}")

    # Write updated manifest
    sha = write_json_manifest(token_records,
                              manifest_dir / "synthetic_identity_manifest.json")
    print(f"  Updated manifest SHA: {sha[:16]}...")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _build_splits_lookup(image_splits):
    """Build identity_id -> {train: [...], validation: [...], test: [...]}."""
    lookup: dict[str, dict[str, list[int]]] = {}
    for rec in image_splits:
        id_ = rec["identity_id"]
        idx = int(rec["image_id"].split("_img_")[-1])
        if id_ not in lookup:
            lookup[id_] = {"train": [], "validation": [], "test": []}
        lookup[id_][rec["split"]].append(idx)
    return lookup


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="E2C-v2 dataset builder")
    parser.add_argument("--mode", required=True,
                        choices=["draft", "finalize", "preflight",
                                 "tokenize-aliases"],
                        help="Build mode")
    parser.add_argument("--output-dir", default="e2c_v2",
                        help="Output directory for E2C-v2 artifacts")
    parser.add_argument("--image-base-dir", default="e2c/data/processed",
                        help="Base directory for generated images")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # Fix 4: tokenize-aliases params
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B",
                        help="Tokenizer model ID")
    parser.add_argument("--tokenizer-revision",
                        default="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                        help="Tokenizer model revision")
    args = parser.parse_args()

    if args.mode == "draft":
        cmd_draft(args)
    elif args.mode == "finalize":
        cmd_finalize(args)
    elif args.mode == "preflight":
        sys.exit(cmd_preflight(args))
    elif args.mode == "tokenize-aliases":
        cmd_tokenize_aliases(args)


if __name__ == "__main__":
    main()
