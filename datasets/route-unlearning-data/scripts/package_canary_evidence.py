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

    # Verify manifest SHA references
    manifest_path = evidence_dir / "run_manifest.json"
    if manifest_path.is_file():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for key, val in manifest.items():
            if key.endswith("_sha256") and isinstance(val, str) and val:
                # Find corresponding file
                file_key = key.replace("_sha256", "")
                file_name = manifest.get(file_key, "")
                if isinstance(file_name, str) and file_name:
                    fpath = evidence_dir / file_name
                    if fpath.is_file():
                        actual = _file_sha256(fpath)
                        if actual != val:
                            errors.append(f"SHA mismatch: {file_name}")

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
