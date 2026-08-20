#!/usr/bin/env python3
"""Generic model-agnostic pre-unlearning baseline generator.

Replaces model-specific scripts (``generate_4b_baseline.py``, etc.) with a
single CLI that derives all model identity from the profile YAML.

Usage::

    # Qwen3.5-4B baseline
    python scripts/generate_model_baseline.py \\
      --model-profile configs/models/unlearning/qwen35_4b.yaml \\
      --device cuda:0

    # GLM-4.6V-Flash baseline
    python scripts/generate_model_baseline.py \\
      --model-profile configs/models/unlearning/glm46v_flash.yaml \\
      --device cuda:0

    # Smoke test (50 probes)
    python scripts/generate_model_baseline.py \\
      --model-profile configs/models/unlearning/qwen35_4b.yaml \\
      --smoke-only

Output::

    outputs/experiments/pre_unlearning/<model_key>/baseline_v1/
        baseline_results.jsonl
        baseline_summary.json
        baseline_manifest.json
        baseline_binding.json
        validation_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import types
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_model_baseline")


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic model-agnostic pre-unlearning baseline generator",
    )
    parser.add_argument(
        "--model-profile", required=True,
        help="Path to model profile YAML (e.g. configs/models/unlearning/qwen35_4b.yaml).",
    )
    parser.add_argument(
        "--probe-file",
        default="outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_route_conflict_eval.jsonl",
        help="Path to the frozen 500-probe evaluation JSONL.",
    )
    parser.add_argument(
        "--processed-dataset",
        default="outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_processed.jsonl",
        help="Path to the frozen processed dataset JSONL.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/experiments/pre_unlearning",
        help="Root output directory (model_key/baseline_v1 appended automatically).",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device for model loading (default: cuda:0).",
    )
    parser.add_argument(
        "--smoke-only", action="store_true",
        help="Run only 50-probe smoke test.",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Clear the baseline cache before running.",
    )
    parser.add_argument(
        "--dataset-manifest",
        default="outputs/full_fiubench/evidence/research_dataset_manifest.json",
        help="Path to the research dataset manifest JSON.",
    )
    parser.add_argument(
        "--freeze-verification",
        default="outputs/full_fiubench/evidence/final_freeze_verification.json",
        help="Path to the freeze verification JSON.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    profile_path = project_root / args.model_profile
    probe_path = project_root / args.probe_file
    processed_path = project_root / args.processed_dataset
    dataset_manifest_path = project_root / args.dataset_manifest
    freeze_verification_path = project_root / args.freeze_verification

    # ---- Load profile and create adapter ---- #
    from route_data.models.trainable.registry import (
        compute_profile_sha256,
        create_adapter,
        load_profile_from_yaml,
    )

    profile = load_profile_from_yaml(str(profile_path))
    profile.validate_revision_immutable()
    profile_sha256 = compute_profile_sha256(str(profile_path))

    # Derive output directory from model_key.
    output_dir = project_root / args.output_root / profile.key / "baseline_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Generic Pre-Unlearning Baseline Generation")
    logger.info("=" * 60)
    logger.info(f"Model key:          {profile.key}")
    logger.info(f"Model ID:           {profile.model_id}")
    logger.info(f"Model revision:     {profile.revision}")
    logger.info(f"Processor revision: {profile.processor_revision}")
    logger.info(f"Adapter name:       {profile.adapter_name}")
    logger.info(f"Profile SHA-256:    {profile_sha256}")
    logger.info(f"Probe file:         {probe_path}")
    logger.info(f"Output dir:         {output_dir}")
    logger.info(f"Device:             {args.device}")
    logger.info(f"Smoke only:         {args.smoke_only}")
    logger.info("=" * 60)

    # ---- Clear cache if requested ---- #
    if args.clear_cache:
        import shutil
        cache_dir = output_dir / ".cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info(f"Cleared cache: {cache_dir}")
        results_file = output_dir / "baseline_results.jsonl"
        if results_file.exists():
            results_file.unlink()
            logger.info(f"Cleared results: {results_file}")

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
    logger.info(f"Model loaded: {type(model).__name__}")
    logger.info(f"Processor loaded: {type(processor).__name__}")

    # ---- Build eval backend via adapter (model-agnostic) ---- #
    from route_data.config import GenerationConfig, ModelConfig

    model_config = ModelConfig(
        backend="adapter_eval_backend",
        model_id=profile.model_id,
        revision=profile.revision,
        dtype=profile.dtype,
        generation=GenerationConfig(do_sample=False),
    )

    backend = adapter.to_eval_backend(
        model=model,
        processor=processor,
        model_config=model_config,
        adapter_metadata=None,
    )

    fingerprint = backend.fingerprint()
    logger.info(f"Model fingerprint: {fingerprint.get('fingerprint_id', 'unknown')}")

    # ---- Build SimpleNamespace for BaselineRunner ---- #
    runner_model_config = types.SimpleNamespace(
        model_id=profile.model_id,
        revision=profile.revision,
        dtype=profile.dtype,
        backend="adapter_eval_backend",
        fingerprint_id=fingerprint.get("fingerprint_id", ""),
        generation=GenerationConfig(do_sample=False),
    )

    # ---- Load probes ---- #
    probes = []
    with open(probe_path) as f:
        for line in f:
            if line.strip():
                probes.append(json.loads(line))
    logger.info(f"Loaded {len(probes)} probes")

    # ---- Run evaluation ---- #
    from route_data.eval.baseline_runner import BaselineRunner

    smoke_limit = 50 if args.smoke_only else None

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

    # Research preflight (validates dataset manifest, freeze verification).
    runner.validate_research_preflight()

    # Run evaluation.
    results = runner.run_all(limit=smoke_limit)
    logger.info(f"Evaluation complete: {len(results)} results")

    # Save results.
    runner.save_results()
    results_path = output_dir / "baseline_results.jsonl"
    results_sha256 = _file_sha256(results_path)
    logger.info(f"Results SHA-256: {results_sha256}")

    # Generate summary.
    runner.generate_summary()

    # ---- Generate binding file (outer integrity envelope) ---- #
    _generate_binding_file(
        output_dir=output_dir,
        profile=profile,
        profile_sha256=profile_sha256,
        probe_path=probe_path,
        processed_path=processed_path,
    )

    logger.info("=" * 60)
    logger.info("Baseline generation complete!")
    logger.info(f"Model: {profile.key}")
    logger.info(f"Total probes: {len(probes)}")
    logger.info(f"Results: {results_path}")
    logger.info(f"Binding: {output_dir / 'baseline_binding.json'}")
    logger.info("=" * 60)


def _generate_binding_file(
    output_dir: Path,
    profile: Any,
    profile_sha256: str,
    probe_path: Path,
    processed_path: Path,
) -> None:
    """Generate baseline_binding.json as the outer integrity envelope.

    Schema follows §1.4 of the multi-model integration plan:

    - manifest_sha256: SHA-256 of the final baseline_manifest.json
    - results_sha256: SHA-256 of baseline_results.jsonl
    - summary_sha256: SHA-256 of baseline_summary.json
    - route_probe_sha256: SHA-256 of the frozen probe file
    - processed_dataset_sha256: SHA-256 of the frozen processed dataset
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

    # Hash each artifact file if it exists.
    artifact_files = {
        "manifest_file": "baseline_manifest.json",
        "results_file": "baseline_results.jsonl",
        "summary_file": "baseline_summary.json",
    }
    for key, fname in artifact_files.items():
        fpath = output_dir / fname
        if fpath.is_file():
            binding[key] = fname
            binding[key.replace("_file", "_sha256")] = _file_sha256(fpath)

    # Frozen dataset SHAs.
    if probe_path.is_file():
        binding["route_probe_sha256"] = _file_sha256(probe_path)
    if processed_path.is_file():
        binding["processed_dataset_sha256"] = _file_sha256(processed_path)

    with open(binding_path, "w") as f:
        json.dump(binding, f, indent=2)
        f.write("\n")

    logger.info(f"Generated binding file: {binding_path}")


if __name__ == "__main__":
    main()
