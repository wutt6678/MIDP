#!/usr/bin/env python3
"""Repository-wide research evidence verifier (P0-SHARED-07).

Validates all model evidence bundles: baselines, canaries, and full runs.

Usage::

    python scripts/verify_research_evidence.py --all
    python scripts/verify_research_evidence.py --model qwen35_4b
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    "qwen35_4b",
    "glm46v_flash",
    "internvl35_8b_hf",
    "phi4_mm",
    "gemma3_12b",
]

REQUIRED_CANARY_FILES = [
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


def _check_json_valid(path: Path) -> tuple[bool, str]:
    """Check if a JSON/JSONL file exists and is valid."""
    if not path.is_file():
        return False, "missing"
    try:
        if path.suffix == ".jsonl":
            # JSONL: each line must be valid JSON
            with open(path) as f:
                for i, line in enumerate(f, 1):
                    if line.strip():
                        json.loads(line)
            return True, "valid"
        else:
            with open(path) as f:
                json.load(f)
            return True, "valid"
    except (json.JSONDecodeError, Exception) as e:
        return False, f"invalid: {e}"


def _verify_baseline(model_key: str) -> tuple[str, list[str]]:
    """Verify a model's baseline bundle. Returns (status, details)."""
    baseline_dir = PROJECT_ROOT / "outputs/experiments/pre_unlearning" / model_key / "baseline_v1"
    if not baseline_dir.is_dir():
        return "MISSING", ["baseline directory not found"]

    issues: list[str] = []

    # Check required files
    for fname in ["baseline_results.jsonl", "baseline_summary.json",
                   "validation_report.json", "baseline_binding.json",
                   "baseline_manifest.json", "preflight_report.json"]:
        fpath = baseline_dir / fname
        valid, msg = _check_json_valid(fpath)
        if not valid:
            issues.append(f"{fname}: {msg}")

    # Check binding validates
    binding_path = baseline_dir / "baseline_binding.json"
    if binding_path.is_file():
        with open(binding_path) as f:
            binding = json.load(f)
        profile_sha = binding.get("model_profile_sha256", "")
        if not profile_sha:
            issues.append("binding: missing profile_sha256")

    if issues:
        return "INVALID", issues
    return "VALID", []


def _verify_canary(model_key: str) -> tuple[str, list[str]]:
    """Verify a model's canary evidence. Returns (status, details)."""
    canary_dir = PROJECT_ROOT / "outputs/experiments/unlearning" / model_key / "candidate_margin_v1" / "canary_seed_17"
    evidence_dir = PROJECT_ROOT / "outputs/experiments/canary_evidence" / model_key

    issues: list[str] = []

    # Check canary output directory
    if not canary_dir.is_dir():
        # Check evidence directory as fallback
        if not evidence_dir.is_dir():
            return "INCOMPLETE", ["no canary output or evidence found"]
        canary_dir = evidence_dir

    # Check required files
    for fname in REQUIRED_CANARY_FILES:
        fpath = canary_dir / fname
        if not fpath.is_file():
            # Also check evidence dir
            epath = evidence_dir / fname
            if not epath.is_file():
                issues.append(f"{fname}: missing")

    # Check validation report
    vr_path = canary_dir / "validation_report.json"
    if not vr_path.is_file():
        vr_path = evidence_dir / "validation_report.json"
    if vr_path.is_file():
        with open(vr_path) as f:
            vr = json.load(f)
        passed = vr.get("pass", False)
        gates = vr.get("gates_passed", 0)
        total = vr.get("total_gates", 0)
        if passed:
            return "VALID_POSITIVE", [f"{gates}/{total} gates passed"]
        else:
            # Check if it's a valid negative
            checks = vr.get("checks", {})
            preservation = checks.get("preservation_gate_pass", True)
            if not preservation:
                return "VALID_NEGATIVE", [f"{gates}/{total} gates, preservation FAIL"]
            return "VALID_NEGATIVE", [f"{gates}/{total} gates, FAIL"]

    if issues:
        return "INCOMPLETE", issues
    return "INCOMPLETE", ["no validation report found"]


def _verify_full_run(model_key: str) -> tuple[str, list[str]]:
    """Check for full-run evidence. Returns (status, details)."""
    full_dir = PROJECT_ROOT / "outputs/experiments/unlearning" / model_key / "candidate_margin_v1" / "seed_17"
    if not full_dir.is_dir():
        return "NOT_RUN", []

    vr_path = full_dir / "validation_report.json"
    if vr_path.is_file():
        with open(vr_path) as f:
            vr = json.load(f)
        passed = vr.get("pass", False)
        if passed:
            return "VALID_POSITIVE", []
        else:
            return "VALID_NEGATIVE", ["full run failed validation"]

    return "INCOMPLETE", ["full run directory exists but no validation report"]


def _verify_model(model_key: str) -> dict[str, Any]:
    """Verify all evidence for one model."""
    result: dict[str, Any] = {"model": model_key}

    # Baseline
    bl_status, bl_issues = _verify_baseline(model_key)
    result["baseline"] = {"status": bl_status, "issues": bl_issues}

    # Canary
    can_status, can_issues = _verify_canary(model_key)
    result["canary"] = {"status": can_status, "issues": can_issues}

    # Full run
    fr_status, fr_issues = _verify_full_run(model_key)
    result["full_run"] = {"status": fr_status, "issues": fr_issues}

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify research evidence")
    parser.add_argument("--model", help="Model key to verify")
    parser.add_argument("--all", action="store_true", help="Verify all models")
    args = parser.parse_args()

    models = MODELS if args.all else [args.model] if args.model else []
    if not models:
        parser.error("Specify --model or --all")

    all_results = []
    for model_key in models:
        result = _verify_model(model_key)
        all_results.append(result)

        # Print summary
        print(f"\n{model_key}")
        for stage in ["baseline", "canary", "full_run"]:
            info = result[stage]
            status = info["status"]
            print(f"  {stage:<12s} {status}")
            for issue in info.get("issues", [])[:3]:
                print(f"    - {issue}")

    # Write summary
    summary_path = PROJECT_ROOT / "outputs/experiments/research_status_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"models": {r["model"]: {k: v for k, v in r.items() if k != "model"} for r in all_results}}, f, indent=2)
        f.write("\n")
    print(f"\nSummary written: {summary_path}")


if __name__ == "__main__":
    main()
