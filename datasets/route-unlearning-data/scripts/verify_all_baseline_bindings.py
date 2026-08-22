#!/usr/bin/env python3
"""Verify all model baseline bindings (P0-SHARED-02).

Validates every model's baseline binding against current profile hash
semantics and on-disk artifact integrity.

Usage::

    python scripts/verify_all_baseline_bindings.py

Expected output::

    qwen35_4b          PASS
    glm46v_flash       PASS
    internvl35_8b_hf   PASS
    phi4_mm            PASS
    gemma3_12b         PASS

Returns nonzero exit status if any model fails.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = PROJECT_ROOT / "configs" / "models" / "unlearning"
BASELINES_DIR = PROJECT_ROOT / "outputs" / "experiments" / "pre_unlearning"
PROBE_PATH = PROJECT_ROOT / "outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_route_conflict_eval.jsonl"
PROCESSED_PATH = PROJECT_ROOT / "outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_processed.jsonl"

MODELS = [
    "qwen35_4b",
    "glm46v_flash",
    "internvl35_8b_hf",
    "phi4_mm",
    "gemma3_12b",
]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_model(model_key: str) -> tuple[bool, list[str]]:
    """Verify a single model's baseline binding.

    Returns (pass, failures).
    """
    from route_data.models.trainable.registry import compute_profile_sha256

    failures: list[str] = []
    profile_path = PROFILES_DIR / f"{model_key}.yaml"
    output_dir = BASELINES_DIR / model_key / "baseline_v1"
    binding_path = output_dir / "baseline_binding.json"

    if not profile_path.is_file():
        return False, [f"profile not found: {profile_path}"]
    if not binding_path.is_file():
        return False, [f"binding not found: {binding_path}"]

    # Load profile
    from route_data.models.trainable.registry import load_profile_from_yaml
    profile = load_profile_from_yaml(str(profile_path))
    current_sha = compute_profile_sha256(str(profile_path))

    # Load binding
    with open(binding_path) as f:
        binding = json.load(f)

    # Check model identity
    if binding.get("model_key", model_key) != model_key:
        failures.append("model_key mismatch")
    if binding.get("model_id") != profile.model_id:
        failures.append("model_id mismatch")
    if binding.get("model_revision") != profile.revision:
        failures.append("model_revision mismatch")
    if binding.get("processor_revision") != profile.processor_revision:
        failures.append("processor_revision mismatch")

    # Check profile SHA
    binding_sha = binding.get("model_profile_sha256", "")
    if binding_sha != current_sha:
        failures.append(
            f"profile_sha256 mismatch: "
            f"binding={binding_sha[:16]}... current={current_sha[:16]}..."
        )

    # Check artifact file hashes
    hash_checks = [
        ("results_sha256", "results_file", "baseline_results.jsonl"),
        ("summary_sha256", "summary_file", "baseline_summary.json"),
        ("validation_report_sha256", "validation_report_file", "validation_report.json"),
        ("preflight_report_sha256", "preflight_report_file", "preflight_report.json"),
        ("manifest_sha256", "manifest_file", "baseline_manifest.json"),
    ]
    for sha_key, file_key, default_fname in hash_checks:
        expected = binding.get(sha_key, "")
        fname = binding.get(file_key, default_fname)
        fpath = output_dir / fname
        if not fpath.is_file():
            failures.append(f"{fname} not found")
            continue
        actual = _file_sha256(fpath)
        if actual != expected:
            failures.append(f"{sha_key} mismatch")

    # Check dataset SHAs
    if PROBE_PATH.is_file():
        expected = binding.get("route_probe_sha256", "")
        if expected and _file_sha256(PROBE_PATH) != expected:
            failures.append("route_probe_sha256 mismatch")

    if PROCESSED_PATH.is_file():
        expected = binding.get("processed_dataset_sha256", "")
        if expected and _file_sha256(PROCESSED_PATH) != expected:
            failures.append("processed_dataset_sha256 mismatch")

    # Check non-empty SHA fields
    for key, val in binding.items():
        if isinstance(val, str) and key.endswith("_sha256") and not val:
            failures.append(f"{key} is empty")

    return len(failures) == 0, failures


def main() -> None:
    all_pass = True

    for model_key in MODELS:
        passed, failures = _verify_model(model_key)
        status = "PASS" if passed else "FAIL"
        print(f"  {model_key:<20s} {status}")
        if failures:
            for f in failures:
                print(f"    - {f}")
            all_pass = False

    if not all_pass:
        print("\nSome bindings FAILED.")
        sys.exit(1)
    else:
        print("\nAll bindings PASS.")


if __name__ == "__main__":
    main()
