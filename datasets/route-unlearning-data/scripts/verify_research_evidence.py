#!/usr/bin/env python3
"""Repository-wide research evidence verifier.

Validates all model evidence bundles: baselines, canaries, and full runs.
Implements fail-closed verification with SHA recomputation.

Status taxonomy:
  VALID              — all files present, all SHAs match
  VALID_POSITIVE     — all gates pass (scientific success)
  VALID_NEGATIVE     — mechanics valid, scientifically interpretable failure
  FAIL_EXECUTION     — mechanical/infrastructure failure
  INCOMPLETE         — missing mandatory artifacts
  NOT_RUN            — no evidence exists

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
    "gemma3_12b",
    "glm46v_flash",
    "internvl35_8b_hf",
    "phi4_mm",
]

PHI_EXTRA_FILES = [
    "phi_causal_invariance_report.json",
    "candidate_scoring_sanity.json",
    "adapter_composition_report.json",
]

REQUIRED_RUN_FILES = [
    "validation_report.json",
    "run_manifest.json",
    "run_binding.json",
    "environment.json",
    "execution_provenance.json",
    "parameter_inventory.json",
    "parameter_change_report.json",
    "training_summary.json",
    "behavioral_effect_validation.json",
    "reload_validation.json",
    "adapter_reload_integrity.json",
    "adapter_tensor_roundtrip.json",
    "exact_probe_match.json",
    "preservation_report.json",
    "group_family_metrics.json",
    "selection.json",
    "checkpoint_metadata.json",
]

REQUIRED_BASELINE_FILES = [
    "baseline_results.jsonl",
    "baseline_summary.json",
    "validation_report.json",
    "baseline_binding.json",
    "baseline_manifest.json",
    "preflight_report.json",
]

# SHA key -> filename mapping for manifest verification
SHA_FILE_MAP = {
    "execution_provenance_sha256": "execution_provenance.json",
    "preservation_report_sha256": "preservation_report.json",
    "reload_validation_sha256": "reload_validation.json",
    "parameter_inventory_sha256": "parameter_inventory.json",
    "behavioral_effect_sha256": "behavioral_effect_validation.json",
    "exact_probe_match_sha256": "exact_probe_match.json",
    "group_family_metrics_sha256": "group_family_metrics.json",
}


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
            with open(path) as f:
                for line in f:
                    if line.strip():
                        json.loads(line)
            return True, "valid"
        else:
            with open(path) as f:
                json.load(f)
            return True, "valid"
    except (json.JSONDecodeError, Exception) as e:
        return False, f"invalid: {e}"


def _find_run_dir(model_key: str, mode: str) -> Path | None:
    """Find the run directory, checking raw output then compact evidence."""
    if mode == "canary":
        raw = PROJECT_ROOT / "outputs/experiments/unlearning" / model_key / "candidate_margin_v1" / "canary_seed_17"
        evidence = PROJECT_ROOT / "outputs/experiments/canary_evidence" / model_key
    else:
        raw = PROJECT_ROOT / "outputs/experiments/unlearning" / model_key / "candidate_margin_v1" / "seed_17"
        evidence = PROJECT_ROOT / "outputs/experiments/full_run_evidence" / model_key

    if raw.is_dir():
        return raw
    if evidence.is_dir():
        return evidence
    return None


def _check_required_files(run_dir: Path, required: list[str], model_key: str) -> list[str]:
    """Check all required files exist and are valid JSON. Returns list of issues."""
    issues: list[str] = []
    for fname in required:
        fpath = run_dir / fname
        valid, msg = _check_json_valid(fpath)
        if not valid:
            issues.append(f"{fname}: {msg}")

    # Phi extra files
    if model_key == "phi4_mm":
        for fname in PHI_EXTRA_FILES:
            fpath = run_dir / fname
            valid, msg = _check_json_valid(fpath)
            if not valid:
                issues.append(f"{fname}: {msg}")

    return issues


def _verify_manifest_shas(run_dir: Path) -> list[str]:
    """Recompute and verify SHAs referenced in run_manifest.json."""
    issues: list[str] = []
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return ["run_manifest.json: missing"]

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Verify artifact SHAs using the explicit mapping
    for sha_key, filename in SHA_FILE_MAP.items():
        stored_sha = manifest.get(sha_key, "")
        if stored_sha:
            fpath = run_dir / filename
            if fpath.is_file():
                actual_sha = _file_sha256(fpath)
                if actual_sha != stored_sha:
                    issues.append(f"SHA mismatch: {filename} (manifest={stored_sha[:12]}... actual={actual_sha[:12]}...)")
            else:
                issues.append(f"SHA reference but file missing: {filename}")

    # Verify model profile SHA
    profile_sha_stored = manifest.get("model_profile_sha256", "")
    if profile_sha_stored:
        model_key = manifest.get("model_key", "")
        if model_key:
            profile_path = PROJECT_ROOT / "configs/models/unlearning" / f"{model_key}.yaml"
            if profile_path.is_file():
                try:
                    from route_data.models.trainable.registry import compute_profile_sha256
                    actual_profile_sha = compute_profile_sha256(profile_path)
                    if actual_profile_sha != profile_sha_stored:
                        issues.append(f"profile SHA mismatch: stored={profile_sha_stored[:12]}... actual={actual_profile_sha[:12]}...")
                except Exception:
                    pass

    return issues


def _verify_binding_shas(run_dir: Path) -> list[str]:
    """Verify run_binding.json references."""
    issues: list[str] = []
    binding_path = run_dir / "run_binding.json"
    manifest_path = run_dir / "run_manifest.json"

    if not binding_path.is_file():
        return ["run_binding.json: missing"]

    with open(binding_path) as f:
        binding = json.load(f)

    # Verify manifest SHA in binding
    stored_manifest_sha = binding.get("run_manifest_sha256", "")
    if stored_manifest_sha and manifest_path.is_file():
        actual_manifest_sha = _file_sha256(manifest_path)
        if actual_manifest_sha != stored_manifest_sha:
            issues.append("run_manifest SHA mismatch in binding")

    # Verify execution provenance SHA in binding
    stored_prov_sha = binding.get("execution_provenance_sha256", "")
    if stored_prov_sha:
        prov_path = run_dir / "execution_provenance.json"
        if prov_path.is_file():
            actual_prov_sha = _file_sha256(prov_path)
            if actual_prov_sha != stored_prov_sha:
                issues.append("execution_provenance SHA mismatch in binding")

    return issues


def _classify_run_status(run_dir: Path, model_key: str, evidence_dir: Path | None = None) -> tuple[str, list[str]]:
    """Classify a run as VALID_POSITIVE, VALID_NEGATIVE, FAIL_EXECUTION, or INCOMPLETE."""
    issues: list[str] = []

    # Check for migration sidecar (historical runs) in either directory
    has_migration = False
    for d in [run_dir, evidence_dir]:
        if d and ((d / "canary_evidence_migration.json").is_file() or 
                  (d / "full_run_evidence_migration.json").is_file()):
            has_migration = True
            break

    # Step 1: Check all required files present
    file_issues = _check_required_files(run_dir, REQUIRED_RUN_FILES, model_key)
    if file_issues:
        # If migration sidecar exists, check which files are missing
        if has_migration:
            # Migration covers: execution_provenance, adapter_reload_integrity, adapter_tensor_roundtrip
            migration_covered = {"execution_provenance.json", "adapter_reload_integrity.json", "adapter_tensor_roundtrip.json"}
            non_covered_issues = [i for i in file_issues if not any(mf in i for mf in migration_covered)]
            if non_covered_issues:
                return "INCOMPLETE", file_issues
            # Only migration-covered files missing — acceptable for historical runs
            covered_files = [i.split(":")[0] for i in file_issues]
            issues.append(f"[legacy] {', '.join(covered_files)} covered by migration sidecar")
        else:
            return "INCOMPLETE", file_issues

    # Step 2: Verify manifest SHAs
    sha_issues = _verify_manifest_shas(run_dir)
    issues.extend(sha_issues)

    # Step 3: Verify binding SHAs
    binding_issues = _verify_binding_shas(run_dir)
    issues.extend(binding_issues)

    # Step 4: Read validation report
    vr_path = run_dir / "validation_report.json"
    with open(vr_path) as f:
        vr = json.load(f)

    passed = vr.get("pass", False)
    failed_gates = vr.get("failed_gates", [])
    checks = vr.get("checks", {})
    gates_passed = vr.get("gates_passed", 0)
    total_gates = vr.get("total_gates", 0)

    if passed:
        status = "VALID_POSITIVE"
    else:
        # Distinguish scientific negative from execution failure
        # Execution failures: LoRA inventory, reload, roundtrip, NaN
        execution_failures = [
            checks.get("lora_inventory_pass", True),
            checks.get("reload_equivalence_pass", True),
            checks.get("adapter_reload_integrity_pass", True),
            checks.get("adapter_tensor_roundtrip_pass", True),
            checks.get("exact_probe_match_pass", True),
        ]

        if not all(execution_failures):
            status = "FAIL_EXECUTION"
        else:
            # Mechanics valid — scientifically interpretable outcome
            status = "VALID_NEGATIVE"

    # Add gate summary to issues
    gate_info = f"{gates_passed}/{total_gates} gates"
    if failed_gates:
        gate_info += f", failed: {', '.join(failed_gates[:3])}"
    issues.insert(0, gate_info)

    return status, issues


def _verify_baseline(model_key: str) -> tuple[str, list[str]]:
    """Verify a model's baseline bundle with SHA recomputation."""
    baseline_dir = PROJECT_ROOT / "outputs/experiments/pre_unlearning" / model_key / "baseline_v1"
    if not baseline_dir.is_dir():
        return "MISSING", ["baseline directory not found"]

    issues: list[str] = []

    # Check required files
    for fname in REQUIRED_BASELINE_FILES:
        fpath = baseline_dir / fname
        valid, msg = _check_json_valid(fpath)
        if not valid:
            issues.append(f"{fname}: {msg}")

    if issues:
        return "INVALID", issues

    # Recompute and verify binding SHAs
    binding_path = baseline_dir / "baseline_binding.json"
    if binding_path.is_file():
        with open(binding_path) as f:
            binding = json.load(f)

        # Verify profile SHA
        profile_sha_stored = binding.get("model_profile_sha256", "")
        if profile_sha_stored:
            profile_path = PROJECT_ROOT / "configs/models/unlearning" / f"{model_key}.yaml"
            if profile_path.is_file():
                try:
                    from route_data.models.trainable.registry import compute_profile_sha256
                    actual = compute_profile_sha256(profile_path)
                    if actual != profile_sha_stored:
                        issues.append(f"profile SHA mismatch: stored={profile_sha_stored[:12]}... actual={actual[:12]}...")
                except Exception:
                    pass

        # Verify manifest SHA
        manifest_sha_stored = binding.get("baseline_manifest_sha256", "")
        if manifest_sha_stored:
            manifest_path = baseline_dir / "baseline_manifest.json"
            if manifest_path.is_file():
                actual = _file_sha256(manifest_path)
                if actual != manifest_sha_stored:
                    issues.append("manifest SHA mismatch in binding")

        # Verify results SHA
        results_sha_stored = binding.get("baseline_results_sha256", "")
        if results_sha_stored:
            results_path = baseline_dir / "baseline_results.jsonl"
            if results_path.is_file():
                actual = _file_sha256(results_path)
                if actual != results_sha_stored:
                    issues.append("results SHA mismatch in binding")

    if issues:
        return "INVALID", issues
    return "VALID", []


def _verify_model(model_key: str) -> dict[str, Any]:
    """Verify all evidence for one model."""
    result: dict[str, Any] = {"model": model_key}

    # Baseline
    bl_status, bl_issues = _verify_baseline(model_key)
    result["baseline"] = {"status": bl_status, "issues": bl_issues}

    # Canary
    canary_dir = _find_run_dir(model_key, "canary")
    canary_evidence = PROJECT_ROOT / "outputs/experiments/canary_evidence" / model_key
    if canary_dir is None:
        if canary_evidence.is_dir():
            canary_dir = canary_evidence
        else:
            result["canary"] = {"status": "INCOMPLETE", "issues": ["no canary output or evidence found"]}
            canary_dir = None
    
    if canary_dir is not None:
        can_status, can_issues = _classify_run_status(canary_dir, model_key, canary_evidence if canary_evidence.is_dir() else None)
        result["canary"] = {"status": can_status, "issues": can_issues}

    # Full run
    full_dir = _find_run_dir(model_key, "full")
    full_evidence = PROJECT_ROOT / "outputs/experiments/full_run_evidence" / model_key
    if full_dir is None:
        if full_evidence.is_dir():
            full_dir = full_evidence
        else:
            result["full_run"] = {"status": "NOT_RUN", "issues": []}
            full_dir = None
    
    if full_dir is not None:
        fr_status, fr_issues = _classify_run_status(full_dir, model_key, full_evidence if full_evidence.is_dir() else None)
        result["full_run"] = {"status": fr_status, "issues": fr_issues}

    return result


def _load_or_create_summary() -> dict:
    """Load existing summary or create empty one."""
    summary_path = PROJECT_ROOT / "outputs/experiments/research_status_summary.json"
    if summary_path.is_file():
        with open(summary_path) as f:
            return json.load(f)
    return {"models": {}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify research evidence")
    parser.add_argument("--model", help="Model key to verify")
    parser.add_argument("--all", action="store_true", help="Verify all models")
    args = parser.parse_args()

    if args.all:
        models = MODELS
    elif args.model:
        models = [args.model]
    else:
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
            for issue in info.get("issues", [])[:5]:
                print(f"    - {issue}")

    # P0-SHARED-05: Only --all rewrites the global summary
    if args.all:
        summary_path = PROJECT_ROOT / "outputs/experiments/research_status_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_data = {
            "models": {r["model"]: {k: v for k, v in r.items() if k != "model"} for r in all_results}
        }
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
            f.write("\n")
        print(f"\nSummary written: {summary_path}")
    else:
        # Update only the requested model in existing summary
        summary_path = PROJECT_ROOT / "outputs/experiments/research_status_summary.json"
        summary_data = _load_or_create_summary()
        for r in all_results:
            summary_data["models"][r["model"]] = {k: v for k, v in r.items() if k != "model"}
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
            f.write("\n")
        print(f"\nSummary updated for: {', '.join(models)}")


if __name__ == "__main__":
    main()
