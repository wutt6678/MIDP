#!/usr/bin/env python3
"""Package compact evidence bundles (P0-EVIDENCE-01/02).

Reads run_manifest.json and run_binding.json from a canary or full-run
output directory and copies every referenced artifact into the compact
evidence directory.

Usage::

    python scripts/package_canary_evidence.py --model qwen35_4b
    python scripts/package_canary_evidence.py --model qwen35_4b --mode full
    python scripts/package_canary_evidence.py --all
    python scripts/package_canary_evidence.py --all --mode full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    "qwen35_4b",
    "glm46v_flash",
    "internvl35_8b_hf",
    "phi4_mm",
    "gemma3_12b",
]

REQUIRED_FILES = [
    "validation_report.json",
    "run_manifest.json",
    "run_binding.json",
    "environment.json",
    "parameter_inventory.json",
    "parameter_change_report.json",
    "training_summary.json",
    "behavioral_effect_validation.json",
    "reload_validation.json",
    "adapter_reload_integrity.json",
    "adapter_tensor_roundtrip.json",
    "exact_probe_match.json",
    "preservation_report.json",
]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_model(model_key: str, mode: str = "canary") -> tuple[bool, list[str]]:
    """Package evidence for one model. Returns (pass, errors)."""
    if mode == "canary":
        run_dir = PROJECT_ROOT / "outputs" / "experiments" / "unlearning" / model_key / "candidate_margin_v1" / "canary_seed_17"
        evidence_dir = PROJECT_ROOT / "outputs" / "experiments" / "canary_evidence" / model_key
    else:  # full
        run_dir = PROJECT_ROOT / "outputs" / "experiments" / "unlearning" / model_key / "candidate_margin_v1" / "seed_17"
        evidence_dir = PROJECT_ROOT / "outputs" / "experiments" / "full_run_evidence" / model_key
    if not run_dir.is_dir():
        return False, [f"{mode.capitalize()} output not found: {run_dir}"]

    errors: list[str] = []

    # Create evidence directory
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Copy required files
    for fname in REQUIRED_FILES:
        src = run_dir / fname
        if src.is_file():
            dst = evidence_dir / fname
            shutil.copy2(src, dst)
        else:
            errors.append(f"Missing: {fname}")

    # Copy model-specific diagnostics
    for extra in [
        "phi_causal_invariance_report.json",
        "candidate_scoring_sanity.json",
        "adapter_composition_report.json",
        "checkpoint_metadata.json",
        "selection.json",
        "group_family_metrics.json",
        "execution_provenance.json",
        "final_classification.json",
    ]:
        src = run_dir / extra
        if src.is_file():
            shutil.copy2(src, evidence_dir / extra)

    # P0-EVIDENCE-01: Verify artifact SHAs using explicit mapping
    SHA_FILE_MAP = {
        "execution_provenance_sha256": "execution_provenance.json",
        "preservation_report_sha256": "preservation_report.json",
        "reload_validation_sha256": "reload_validation.json",
        "parameter_inventory_sha256": "parameter_inventory.json",
        "behavioral_effect_sha256": "behavioral_effect_validation.json",
        "exact_probe_match_sha256": "exact_probe_match.json",
        "group_family_metrics_sha256": "group_family_metrics.json",
    }

    manifest_path = evidence_dir / "run_manifest.json"
    manifest_sha_valid = True
    if manifest_path.is_file():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for sha_key, filename in SHA_FILE_MAP.items():
            stored_sha = manifest.get(sha_key, "")
            if stored_sha:
                fpath = evidence_dir / filename
                if fpath.is_file():
                    actual = _file_sha256(fpath)
                    if actual != stored_sha:
                        errors.append(f"SHA mismatch: {filename}")
                        manifest_sha_valid = False

    # P0-EVIDENCE-02: Validate run binding
    binding_path = evidence_dir / "run_binding.json"
    run_binding_valid = True
    if binding_path.is_file() and manifest_path.is_file():
        with open(binding_path) as f:
            binding = json.load(f)
        # Verify manifest SHA in binding
        stored_manifest_sha = binding.get("run_manifest_sha256", "")
        if stored_manifest_sha:
            actual_manifest_sha = _file_sha256(manifest_path)
            if actual_manifest_sha != stored_manifest_sha:
                errors.append("run_manifest SHA mismatch in binding")
                run_binding_valid = False
        # Verify execution provenance SHA in binding
        stored_prov_sha = binding.get("execution_provenance_sha256", "")
        if stored_prov_sha:
            prov_path = evidence_dir / "execution_provenance.json"
            if prov_path.is_file():
                actual_prov_sha = _file_sha256(prov_path)
                if actual_prov_sha != stored_prov_sha:
                    errors.append("execution_provenance SHA mismatch in binding")
                    run_binding_valid = False

    # P0-EVIDENCE-03: Write bundle-level validation
    required_complete = len(errors) == 0
    validation = {
        "pass": required_complete,
        "required_files_complete": required_complete,
        "manifest_sha_valid": manifest_sha_valid,
        "run_binding_valid": run_binding_valid,
        "artifact_sha_valid": manifest_sha_valid,
        "mode": mode,
        "model_key": model_key,
    }
    with open(evidence_dir / "evidence_bundle_validation.json", "w") as f:
        json.dump(validation, f, indent=2)
        f.write("\n")

    return len(errors) == 0, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Package evidence")
    parser.add_argument("--model", help="Model key to package")
    parser.add_argument("--all", action="store_true", help="Package all models")
    parser.add_argument("--mode", choices=["canary", "full"], default="canary",
                        help="Evidence mode (default: canary)")
    args = parser.parse_args()

    models = MODELS if args.all else [args.model] if args.model else []
    if not models:
        parser.error("Specify --model or --all")

    all_pass = True
    for model_key in models:
        passed, errors = _package_model(model_key, mode=args.mode)
        status = "PASS" if passed else "FAIL"
        print(f"  {model_key:<20s} [{args.mode}] {status}")
        if errors:
            for e in errors:
                print(f"    - {e}")
            all_pass = False

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
