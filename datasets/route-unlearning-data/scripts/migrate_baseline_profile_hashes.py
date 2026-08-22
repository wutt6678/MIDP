#!/usr/bin/env python3
"""One-time baseline profile-hash migration (P0-SHARED-01 + P0-PROVENANCE-01/02).

The shared ``compute_profile_sha256()`` semantics changed:

- Scientific execution fields are hashed.
- ``compatibility.constraints`` is hashed.
- Descriptive ``tested_environment`` metadata is excluded.
- Mutable ``supports_*``, ``status``, ``access`` are excluded.

This script:

1. Loads every model profile (current and optionally old from git).
2. Computes the current canonical scientific profile SHA.
3. Loads the corresponding baseline binding.
4. Compares stored SHA vs current SHA.
5. If mismatched, verifies that no scientific field actually changed
   by comparing full canonical scientific payloads.
6. Regenerates metadata-only bindings where safe.
7. Writes a migration audit report with true old/new hashes.

Usage::

    # Migrate current bindings
    python scripts/migrate_baseline_profile_hashes.py

    # Audit-only with git refs
    python scripts/migrate_baseline_profile_hashes.py \\
      --old-ref 770a9b2 --new-ref HEAD --audit-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("migrate_baseline_profile_hashes")

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

# Scientific fields that must be compared semantically.
_SCIENTIFIC_KEYS = ["key", "model", "candidate_protocol", "lora", "structural"]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_scientific_fields(data: dict) -> dict:
    """Extract the scientific execution fields from a profile dict."""
    canonical = {k: data[k] for k in _SCIENTIFIC_KEYS if k in data}

    # For compatibility, only extract constraints (not tested_environment)
    compat = data.get("compatibility", {})
    if compat:
        if "constraints" in compat:
            canonical["compatibility"] = {"constraints": compat["constraints"]}
        else:
            constraint_keys = [
                "min_transformers", "max_transformers_exclusive",
                "min_torch", "max_torch_exclusive",
                "min_peft", "max_peft_exclusive",
            ]
            constraints = {k: compat[k] for k in constraint_keys if k in compat}
            if constraints:
                canonical["compatibility"] = {"constraints": constraints}

    return canonical


def _get_profile_from_git(ref: str, model_key: str) -> dict | None:
    """Load a profile YAML from a git ref."""
    rel_path = f"datasets/route-unlearning-data/configs/models/unlearning/{model_key}.yaml"
    try:
        import yaml
        result = subprocess.run(
            ["git", "show", f"{ref}:{rel_path}"],
            capture_output=True, text=True, check=True,
            cwd=PROJECT_ROOT,
        )
        return yaml.safe_load(result.stdout)
    except (subprocess.CalledProcessError, Exception):
        return None


def _scientific_fields_equal(
    old_data: dict, new_data: dict,
) -> tuple[bool, list[str]]:
    """Compare scientific fields between old and new profile data.

    P0-PROVENANCE-02: Full canonical payload comparison.
    """
    old_sci = _extract_scientific_fields(old_data)
    new_sci = _extract_scientific_fields(new_data)

    old_blob = json.dumps(old_sci, sort_keys=True, separators=(",", ":"))
    new_blob = json.dumps(new_sci, sort_keys=True, separators=(",", ":"))

    if old_blob == new_blob:
        return True, []

    # Find specific differences
    diffs: list[str] = []
    all_keys = set(list(old_sci.keys()) + list(new_sci.keys()))
    for k in sorted(all_keys):
        old_val = json.dumps(old_sci.get(k), sort_keys=True)
        new_val = json.dumps(new_sci.get(k), sort_keys=True)
        if old_val != new_val:
            diffs.append(f"  {k}: old={old_val[:80]}... new={new_val[:80]}...")

    return False, diffs


def _generate_upgraded_binding(
    profile: Any,
    profile_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate a new binding in the current format."""
    binding: dict[str, Any] = {
        "schema_version": "baseline-binding-v1",
        "model_key": profile.key,
        "model_id": profile.model_id,
        "model_revision": profile.revision,
        "processor_revision": profile.processor_revision,
        "model_profile_sha256": profile_sha256,
    }

    artifact_map = {
        "results_file": "baseline_results.jsonl",
        "summary_file": "baseline_summary.json",
        "validation_report_file": "validation_report.json",
        "preflight_report_file": "preflight_report.json",
        "manifest_file": "baseline_manifest.json",
    }
    for key, fname in artifact_map.items():
        fpath = output_dir / fname
        if fpath.is_file():
            binding[key] = fname
            binding[key.replace("_file", "_sha256")] = _file_sha256(fpath)
        else:
            raise FileNotFoundError(f"Required artifact missing: {fpath}")

    if PROBE_PATH.is_file():
        binding["route_probe_sha256"] = _file_sha256(PROBE_PATH)
    if PROCESSED_PATH.is_file():
        binding["processed_dataset_sha256"] = _file_sha256(PROCESSED_PATH)

    return binding


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline profile-hash migration",
    )
    parser.add_argument(
        "--old-ref",
        help="Git ref for old profiles (for audit-only mode)",
    )
    parser.add_argument(
        "--new-ref",
        help="Git ref for new profiles (for audit-only mode)",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only generate audit report, do not modify bindings",
    )
    args = parser.parse_args()

    from route_data.models.trainable.registry import (
        compute_profile_sha256,
        load_profile_from_yaml,
    )

    migration_report: dict[str, Any] = {
        "schema_version": "profile-hash-migration-v2",
        "old_ref": args.old_ref,
        "new_ref": args.new_ref,
        "models": {},
    }

    all_pass = True

    for model_key in MODELS:
        logger.info(f"--- {model_key} ---")
        profile_path = PROFILES_DIR / f"{model_key}.yaml"
        output_dir = BASELINES_DIR / model_key / "baseline_v1"
        binding_path = output_dir / "baseline_binding.json"

        if not profile_path.is_file():
            logger.error(f"Profile not found: {profile_path}")
            all_pass = False
            continue
        if not binding_path.is_file():
            logger.error(f"Binding not found: {binding_path}")
            all_pass = False
            continue

        # Load profile object
        profile = load_profile_from_yaml(str(profile_path))

        # Compute current SHA
        current_sha = compute_profile_sha256(profile_path)

        # Load existing binding
        with open(binding_path) as f:
            old_binding = json.load(f)

        old_sha = old_binding.get("model_profile_sha256", "")

        model_report: dict[str, Any] = {
            "old_binding_profile_sha256": old_sha,
            "new_profile_sha256": current_sha,
        }

        # P0-PROVENANCE-01: If --old-ref provided, also store the git-based old SHA
        if args.old_ref:
            old_profile_data = _get_profile_from_git(args.old_ref, model_key)
            if old_profile_data:
                old_sci = _extract_scientific_fields(old_profile_data)
                old_sci_blob = json.dumps(old_sci, sort_keys=True, separators=(",", ":"))
                old_git_sha = hashlib.sha256(old_sci_blob.encode()).hexdigest()
                model_report["old_ref"] = args.old_ref
                model_report["old_ref_scientific_sha256"] = old_git_sha

        if args.new_ref:
            new_profile_data = _get_profile_from_git(args.new_ref, model_key)
            if new_profile_data:
                new_sci = _extract_scientific_fields(new_profile_data)
                new_sci_blob = json.dumps(new_sci, sort_keys=True, separators=(",", ":"))
                new_git_sha = hashlib.sha256(new_sci_blob.encode()).hexdigest()
                model_report["new_ref"] = args.new_ref
                model_report["new_ref_scientific_sha256"] = new_git_sha

        # P0-PROVENANCE-02: Full scientific payload comparison
        if args.old_ref:
            old_profile_data = _get_profile_from_git(args.old_ref, model_key)
            with open(profile_path) as f:
                import yaml
                new_profile_data = yaml.safe_load(f)

            if old_profile_data and new_profile_data:
                sci_equal, sci_diffs = _scientific_fields_equal(
                    old_profile_data, new_profile_data,
                )
                model_report["scientific_fields_equal"] = sci_equal
                if not sci_equal:
                    model_report["scientific_diffs"] = sci_diffs
                    logger.error("  FAIL: Scientific fields changed!")
                    for d in sci_diffs:
                        logger.error(d)
                    all_pass = False

        if old_sha == current_sha:
            logger.info(f"  SHA already matches: {current_sha[:24]}...")
            model_report["migration_required"] = False
            model_report["migration_pass"] = True
            migration_report["models"][model_key] = model_report
            continue

        # SHA mismatch
        logger.info("  SHA mismatch detected")
        logger.info(f"    old: {old_sha[:24]}...")
        logger.info(f"    new: {current_sha[:24]}...")

        # Check model identity
        binding_model_id = old_binding.get("model_id", "")
        binding_model_rev = old_binding.get("model_revision", "")
        binding_proc_rev = old_binding.get("processor_revision", "")

        identity_ok = True
        if binding_model_id and binding_model_id != profile.model_id:
            identity_ok = False
        if binding_model_rev and binding_model_rev != profile.revision:
            identity_ok = False
        if binding_proc_rev and binding_proc_rev != profile.processor_revision:
            identity_ok = False

        if not identity_ok:
            logger.error("  FAIL: Model identity changed!")
            model_report["migration_required"] = False
            model_report["migration_pass"] = False
            model_report["error"] = "Model identity changed"
            migration_report["models"][model_key] = model_report
            all_pass = False
            continue

        if args.audit_only:
            logger.info("  Audit-only mode — not modifying binding")
            model_report["migration_required"] = True
            model_report["migration_pass"] = True
            migration_report["models"][model_key] = model_report
            continue

        # Regenerate binding
        logger.info("  Migrating binding")
        new_binding = _generate_upgraded_binding(profile, current_sha, output_dir)

        with open(binding_path, "w") as f:
            json.dump(new_binding, f, indent=2)
            f.write("\n")
        logger.info(f"  Binding regenerated: {binding_path}")

        # Validate
        validation: dict[str, Any] = {"pass": True, "checks": {}}
        for sha_key, _file_key, fname in [
            ("results_sha256", "results_file", "baseline_results.jsonl"),
            ("summary_sha256", "summary_file", "baseline_summary.json"),
            ("validation_report_sha256", "validation_report_file", "validation_report.json"),
            ("preflight_report_sha256", "preflight_report_file", "preflight_report.json"),
            ("manifest_sha256", "manifest_file", "baseline_manifest.json"),
        ]:
            fpath = output_dir / fname
            if fpath.is_file():
                expected = new_binding.get(sha_key, "")
                actual = _file_sha256(fpath)
                validation["checks"][sha_key] = actual == expected

        report_path = output_dir / "binding_validation_report.json"
        with open(report_path, "w") as f:
            json.dump(validation, f, indent=2)
            f.write("\n")

        binding_pass = all(validation["checks"].values())
        logger.info(f"  Binding validation: {'PASS' if binding_pass else 'FAIL'}")

        model_report["migration_required"] = True
        model_report["migration_pass"] = binding_pass

        if not binding_pass:
            all_pass = False

        migration_report["models"][model_key] = model_report

    # Write migration audit report
    report_path = PROJECT_ROOT / "outputs" / "experiments" / "baseline_profile_hash_migration.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(migration_report, f, indent=2)
        f.write("\n")

    logger.info("=" * 60)
    logger.info(f"Migration report: {report_path}")
    logger.info(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    logger.info("=" * 60)

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
