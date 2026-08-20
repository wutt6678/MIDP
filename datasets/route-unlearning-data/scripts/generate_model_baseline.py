#!/usr/bin/env python3
"""Generic model-agnostic pre-unlearning baseline generator.

Produces a research-complete baseline bundle:

    1. validate research preflight
    2. run all 500 probes (or 50 balanced smoke)
    3. save baseline_results.jsonl
    4. generate baseline_summary.json
    5. validate all results
    6. generate validation_report.json
    7. generate baseline_manifest.json
    8. generate baseline_binding.json LAST
    9. validate the final binding itself

Usage::

    # Full 500-probe baseline
    python scripts/generate_model_baseline.py \\
      --model-profile configs/models/unlearning/glm46v_flash.yaml \\
      --device cuda:0

    # Balanced 50-probe smoke (10/family)
    python scripts/generate_model_baseline.py \\
      --model-profile configs/models/unlearning/glm46v_flash.yaml \\
      --device cuda:0 --smoke-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_model_baseline")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_is_clean(project_root: Path) -> bool:
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=False,
            cwd=project_root,
        )
        return not bool(r.stdout.strip())
    except Exception:
        return False


def select_balanced_smoke_probes(
    probes: list,
    per_family: int = 10,
    seed: int = 17,
) -> list:
    """Select exactly *per_family* probes from each of the 5 families.

    Deterministic: sorts by probe_id within each family, then takes the
    first *per_family*.  Returns exactly ``per_family * 5`` probes.
    Works with both dict and BaselineProbe dataclass inputs.
    """
    import random
    rng = random.Random(seed)

    def _get(p, key):
        if isinstance(p, dict):
            return p[key]
        return getattr(p, key)

    by_family: dict[str, list] = defaultdict(list)
    for p in probes:
        by_family[_get(p, "probe_family")].append(p)

    expected_families = {
        "direct_visual", "image_plus_name", "wrong_name",
        "visual_text_conflict", "name_only",
    }
    missing = expected_families - set(by_family.keys())
    if missing:
        raise RuntimeError(f"Missing probe families: {missing}")

    selected: list = []
    for fam in sorted(expected_families):
        fam_probes = sorted(by_family[fam], key=lambda p: _get(p, "probe_id"))
        if len(fam_probes) < per_family:
            raise RuntimeError(
                f"Family {fam!r} has only {len(fam_probes)} probes, "
                f"need {per_family}"
            )
        selected.extend(fam_probes[:per_family])

    # Verify balance.
    fam_counts = defaultdict(int)
    for p in selected:
        fam_counts[_get(p, "probe_family")] += 1
    for fam in expected_families:
        assert fam_counts[fam] == per_family, (
            f"Family {fam!r} has {fam_counts[fam]} probes, expected {per_family}"
        )

    return selected


# --------------------------------------------------------------------------- #
# Binding generation and validation
# --------------------------------------------------------------------------- #

def _generate_binding_file(
    output_dir: Path,
    profile: Any,
    profile_sha256: str,
    probe_path: Path,
    processed_path: Path,
) -> Path:
    """Generate baseline_binding.json as the outer integrity envelope.

    The binding hashes all final immutable artifacts.  It must be
    generated LAST, after all other artifacts are written.
    """
    binding_path = output_dir / "baseline_binding.json"

    binding: dict[str, Any] = {
        "schema_version": "baseline-binding-v1",
        "model_key": profile.key,
        "model_id": profile.model_id,
        "model_revision": profile.revision,
        "processor_revision": profile.processor_revision,
        "model_profile_sha256": profile_sha256,
    }

    # Hash each artifact file.
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
            raise FileNotFoundError(
                f"Required artifact missing: {fpath}. "
                f"Binding must be generated LAST."
            )

    # Frozen dataset SHAs.
    if probe_path.is_file():
        binding["route_probe_sha256"] = _file_sha256(probe_path)
    else:
        raise FileNotFoundError(f"Probe file not found: {probe_path}")
    if processed_path.is_file():
        binding["processed_dataset_sha256"] = _file_sha256(processed_path)
    else:
        raise FileNotFoundError(f"Processed dataset not found: {processed_path}")

    with open(binding_path, "w") as f:
        json.dump(binding, f, indent=2)
        f.write("\n")

    logger.info(f"Generated binding: {binding_path}")
    return binding_path


def _validate_final_binding(
    output_dir: Path,
    profile: Any,
    profile_sha256: str,
    probe_path: Path,
    processed_path: Path,
) -> dict[str, Any]:
    """Validate the final binding against actual files on disk.

    Returns a validation report dict.  Raises RuntimeError on failure.
    """
    binding_path = output_dir / "baseline_binding.json"
    if not binding_path.is_file():
        raise FileNotFoundError(f"Binding not found: {binding_path}")

    with open(binding_path) as f:
        binding = json.load(f)

    checks: dict[str, Any] = {}

    # Model identity.
    checks["model_key_match"] = binding.get("model_key") == profile.key
    checks["model_id_match"] = binding.get("model_id") == profile.model_id
    checks["model_revision_match"] = binding.get("model_revision") == profile.revision
    checks["processor_revision_match"] = binding.get("processor_revision") == profile.processor_revision
    checks["model_profile_sha256_match"] = binding.get("model_profile_sha256") == profile_sha256

    # File hashes.
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
            checks[f"{sha_key}_file_exists"] = False
            continue
        checks[f"{sha_key}_file_exists"] = True
        actual = _file_sha256(fpath)
        checks[sha_key] = actual == expected

    # Frozen dataset SHAs.
    if probe_path.is_file():
        checks["route_probe_sha256"] = (
            binding.get("route_probe_sha256") == _file_sha256(probe_path)
        )
    if processed_path.is_file():
        checks["processed_dataset_sha256"] = (
            binding.get("processed_dataset_sha256") == _file_sha256(processed_path)
        )

    # No empty fields.
    for key, val in binding.items():
        if isinstance(val, str) and key.endswith("_sha256"):
            checks[f"{key}_nonempty"] = len(val) > 0

    all_pass = all(checks.values())
    report = {
        "pass": all_pass,
        "checks": checks,
        "binding_file": str(binding_path),
    }

    # Write validation report.
    report_path = output_dir / "binding_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    logger.info(f"Binding validation: {'PASS' if all_pass else 'FAIL'} → {report_path}")

    if not all_pass:
        failed = [k for k, v in checks.items() if not v]
        raise RuntimeError(
            f"Final binding validation FAILED:\n  " + "\n  ".join(failed)
        )

    return report


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic model-agnostic pre-unlearning baseline generator",
    )
    parser.add_argument("--model-profile", required=True)
    parser.add_argument(
        "--probe-file",
        default="outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_route_conflict_eval.jsonl",
    )
    parser.add_argument(
        "--processed-dataset",
        default="outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_processed.jsonl",
    )
    parser.add_argument("--output-root", default="outputs/experiments/pre_unlearning")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument(
        "--dataset-manifest",
        default="outputs/full_fiubench/evidence/research_dataset_manifest.json",
    )
    parser.add_argument(
        "--freeze-verification",
        default="outputs/full_fiubench/evidence/final_freeze_verification.json",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    profile_path = project_root / args.model_profile
    probe_path = project_root / args.probe_file
    processed_path = project_root / args.processed_dataset
    dataset_manifest_path = project_root / args.dataset_manifest
    freeze_verification_path = project_root / args.freeze_verification

    # ---- Load profile ---- #
    from route_data.models.trainable.registry import (
        compute_profile_sha256,
        create_adapter,
        load_profile_from_yaml,
    )

    profile = load_profile_from_yaml(str(profile_path))
    profile.validate_revision_immutable()
    profile_sha256 = compute_profile_sha256(str(profile_path))

    # ---- Determine output directory ---- #
    # Separate smoke and full baseline directories (§6).
    if args.smoke_only:
        protocol_dir = "smoke_v1"
    else:
        protocol_dir = "baseline_v1"

    output_dir = project_root / args.output_root / profile.key / protocol_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Git clean check for full baseline (§7 Test F) ---- #
    if not args.smoke_only and not _git_is_clean(project_root):
        raise RuntimeError(
            "Full research baseline requires a clean git working tree. "
            "Commit or stash changes before running."
        )

    # ---- RUN_INCOMPLETE marker (§5) ---- #
    run_id = uuid.uuid4().hex[:8]
    marker = output_dir / "RUN_INCOMPLETE"
    marker.write_text(f"run_id={run_id}\nmodel_key={profile.key}\n")
    logger.info(f"RUN_INCOMPLETE marker written: {marker}")

    logger.info("=" * 60)
    logger.info("Generic Pre-Unlearning Baseline Generation")
    logger.info("=" * 60)
    logger.info(f"Model key:          {profile.key}")
    logger.info(f"Model ID:           {profile.model_id}")
    logger.info(f"Model revision:     {profile.revision}")
    logger.info(f"Processor revision: {profile.processor_revision}")
    logger.info(f"Profile SHA-256:    {profile_sha256}")
    logger.info(f"Output dir:         {output_dir}")
    logger.info(f"Mode:               {'smoke (50)' if args.smoke_only else 'full (500)'}")
    logger.info("=" * 60)

    # ---- Clear cache if requested ---- #
    if args.clear_cache:
        import shutil
        cache_dir = output_dir / ".cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info(f"Cleared cache: {cache_dir}")

    # ---- Load model via adapter ---- #
    logger.info(f"Loading model via adapter ({profile.adapter_name})...")
    adapter = create_adapter(profile.key, profile=profile)
    model, processor = adapter.load_model_processor(
        model_id=profile.model_id,
        revision=profile.revision,
        processor_revision=profile.processor_revision,
        dtype=profile.dtype,
        device=args.device,
        training=False,
    )

    # ---- Build eval backend ---- #
    from route_data.config import GenerationConfig, ModelConfig

    model_config = ModelConfig(
        backend="adapter_eval_backend",
        model_id=profile.model_id,
        revision=profile.revision,
        dtype=profile.dtype,
        generation=GenerationConfig(do_sample=False),
    )
    backend = adapter.to_eval_backend(
        model=model, processor=processor,
        model_config=model_config, adapter_metadata=None,
    )
    fingerprint = backend.fingerprint()
    logger.info(f"Fingerprint: {fingerprint.get('fingerprint_id', '?')}")

    # ---- Build SimpleNamespace for BaselineRunner ---- #
    import types
    runner_model_config = types.SimpleNamespace(
        model_id=profile.model_id,
        revision=profile.revision,
        dtype=profile.dtype,
        backend="adapter_eval_backend",
        fingerprint_id=fingerprint.get("fingerprint_id", ""),
        generation=GenerationConfig(do_sample=False),
    )

    # ---- Create BaselineRunner ---- #
    from route_data.eval.baseline_runner import BaselineRunner

    runner = BaselineRunner(
        backend=backend,
        probe_path=str(probe_path),
        output_dir=str(output_dir),
        model_config=runner_model_config,
        resume=True,
        dataset_manifest_path=str(dataset_manifest_path),
        model_config_path=str(profile_path),
        freeze_verification_path=str(freeze_verification_path),
        processed_dataset_path=str(processed_path),
    )

    # ================================================================== #
    # Step 1: Validate research preflight
    # ================================================================== #
    logger.info("Step 1: Research preflight...")
    preflight = runner.validate_research_preflight()
    preflight_path = output_dir / "preflight_report.json"
    with open(preflight_path, "w") as f:
        json.dump(preflight, f, indent=2, default=str)
        f.write("\n")
    if not preflight.get("pass", False):
        raise RuntimeError(f"Research preflight FAILED: {preflight}")
    logger.info("Preflight: PASS")

    # ================================================================== #
    # Step 2: Run probes
    # ================================================================== #
    if args.smoke_only:
        # Balanced 50-probe smoke (§2).
        logger.info("Step 2: Balanced 50-probe smoke...")
        smoke_probe_objs = select_balanced_smoke_probes(runner.probes, per_family=10)
        assert len(smoke_probe_objs) == 50, f"Expected 50 smoke probes, got {len(smoke_probe_objs)}"

        results = runner.run_selected(smoke_probe_objs)
        smoke_ids = {p.probe_id for p in smoke_probe_objs}
    else:
        logger.info("Step 2: Running all 500 probes...")
        results = runner.run_all()
        smoke_ids = None

    logger.info(f"Step 2 complete: {len(results)} results")

    # ================================================================== #
    # Step 3: Save results
    # ================================================================== #
    logger.info("Step 3: Saving results...")
    results_path = runner.save_results()

    # ================================================================== #
    # Step 4: Generate summary
    # ================================================================== #
    logger.info("Step 4: Generating summary...")
    summary = runner.generate_summary()

    # ================================================================== #
    # Step 5: Validate results
    # ================================================================== #
    logger.info("Step 5: Validating results...")
    validation = runner.validate_results(smoke_probe_ids=smoke_ids)
    validation_path = output_dir / "validation_report.json"
    with open(validation_path, "w") as f:
        json.dump(validation, f, indent=2, default=str)
        f.write("\n")

    if not validation.get("pass", False):
        failed = [k for k, v in validation.get("checks", {}).items() if not v]
        raise RuntimeError(f"Strict validation FAILED: {failed}")
    logger.info("Validation: PASS")

    # ================================================================== #
    # Step 6-7: Generate manifest (full only)
    # ================================================================== #
    if not args.smoke_only:
        logger.info("Step 6-7: Generating manifest...")
        manifest = runner.generate_baseline_manifest()
    else:
        logger.info("Step 6-7: Skipping manifest (smoke mode)")

    # ================================================================== #
    # Step 8: Generate binding LAST
    # ================================================================== #
    if not args.smoke_only:
        logger.info("Step 8: Generating binding...")
        _generate_binding_file(
            output_dir=output_dir,
            profile=profile,
            profile_sha256=profile_sha256,
            probe_path=probe_path,
            processed_path=processed_path,
        )

        # ============================================================== #
        # Step 9: Validate the final binding
        # ============================================================== #
        logger.info("Step 9: Validating final binding...")
        _validate_final_binding(
            output_dir=output_dir,
            profile=profile,
            profile_sha256=profile_sha256,
            probe_path=probe_path,
            processed_path=processed_path,
        )
        logger.info("Final binding: PASS")
    else:
        logger.info("Step 8-9: Skipping binding (smoke mode)")

    # ---- Remove RUN_INCOMPLETE marker ---- #
    marker.unlink(missing_ok=True)
    logger.info("RUN_INCOMPLETE marker removed")

    logger.info("=" * 60)
    logger.info("Baseline generation COMPLETE!")
    logger.info(f"Model: {profile.key}")
    logger.info(f"Mode:  {'smoke (50)' if args.smoke_only else 'full (500)'}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
