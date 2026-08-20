#!/usr/bin/env python3
"""Generate pre-unlearning baseline for Qwen3.5-4B canary.

This script evaluates the frozen base model on the 500-probe FIUBench
dataset to establish the pre-unlearning baseline for the 4B canary.

Usage::

    # 50-probe smoke test (first 50 probes)
    python scripts/generate_4b_baseline.py --smoke-only

    # Full 500-probe baseline
    python scripts/generate_4b_baseline.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_4b_baseline")


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_model_fingerprint(
    model_id: str,
    model_revision: str,
    processor_class: str,
    tokenizer_class: str,
    dtype: str = "bfloat16",
) -> tuple[str, dict[str, Any]]:
    """Compute a deterministic model fingerprint."""
    import torch
    import transformers

    payload = {
        "backend": "qwen_hf",
        "model_id": model_id,
        "revision": model_revision,
        "dtype": dtype,
        "quantization": "none",
        "attn": "sdpa",
        "thinking": "disabled",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "processor_class": processor_class,
        "tokenizer_class": tokenizer_class,
    }

    # Compute fingerprint ID from payload
    payload_str = json.dumps(payload, sort_keys=True)
    fingerprint_id = hashlib.sha256(payload_str.encode()).hexdigest()[:16]
    payload["fingerprint_id"] = fingerprint_id

    return fingerprint_id, payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Qwen3.5-4B pre-unlearning baseline",
    )
    parser.add_argument(
        "--smoke-only", action="store_true",
        help="Run only 50-probe smoke test (first 50 probes).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiments/pre_unlearning/qwen35_4b/baseline_v1",
        help="Output directory for baseline artifacts.",
    )
    parser.add_argument(
        "--probe-path",
        default="outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_route_conflict_eval.jsonl",
        help="Path to the 500-probe evaluation dataset.",
    )
    parser.add_argument(
        "--profile",
        default="configs/models/unlearning/qwen35_4b.yaml",
        help="Path to the 4B model profile YAML.",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Clear the baseline cache before running.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear cache if requested
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

    probe_path = project_root / args.probe_path
    profile_path = project_root / args.profile

    # Load profile
    from route_data.models.trainable.registry import (
        compute_profile_sha256,
        create_adapter,
        load_profile_from_yaml,
    )

    profile = load_profile_from_yaml(profile_path)
    profile_sha256 = compute_profile_sha256(profile_path)

    logger.info("=" * 60)
    logger.info("Qwen3.5-4B Pre-Unlearning Baseline Generation")
    logger.info("=" * 60)
    logger.info(f"Model key: {profile.key}")
    logger.info(f"Model ID: {profile.model_id}")
    logger.info(f"Model revision: {profile.revision}")
    logger.info(f"Profile SHA-256: {profile_sha256}")
    logger.info(f"Probe path: {probe_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Smoke only: {args.smoke_only}")
    logger.info("=" * 60)

    # Load model via adapter
    logger.info("Loading model via trainable adapter...")
    adapter = create_adapter(profile.key, profile=profile)
    model, processor = adapter.load_model_processor(
        model_id=profile.model_id,
        revision=profile.revision,
        processor_revision=profile.processor_revision,
        dtype=profile.dtype,
        device="cuda:0",
        training=False,
    )

    logger.info(f"Model loaded: {type(model).__name__}")
    logger.info(f"Processor loaded: {type(processor).__name__}")

    # Compute model fingerprint
    processor_class = type(processor).__name__
    tokenizer_class = type(processor.tokenizer).__name__
    fingerprint_id, _ = _compute_model_fingerprint(
        model_id=profile.model_id,
        model_revision=profile.revision,
        processor_class=processor_class,
        tokenizer_class=tokenizer_class,
    )

    logger.info(f"Model fingerprint: {fingerprint_id}")

    # Load probes
    probes = []
    with open(probe_path) as f:
        for line in f:
            if line.strip():
                probes.append(json.loads(line))

    logger.info(f"Loaded {len(probes)} probes")

    # Run evaluation
    logger.info("Running evaluation on probes...")
    smoke_limit = 50 if args.smoke_only else None
    from route_data.config import GenerationConfig, ModelConfig
    from route_data.eval.baseline_runner import BaselineRunner
    from route_data.models.qwen import QwenHFBackend

    # Build ModelConfig + QwenHFBackend (correct BaselineRunner interface).
    qwen_config = ModelConfig(
        backend="qwen_hf",
        model_id=profile.model_id,
        revision=profile.revision,
        dtype=profile.dtype,
        generation=GenerationConfig(do_sample=False),
    )
    backend = QwenHFBackend.from_loaded_model(
        config=qwen_config,
        model=model,
        processor=processor,
        adapter_metadata=None,
        resolved_revision=profile.revision,
    )

    # model_config namespace for BaselineRunner attribute access (.revision etc.)
    import types
    _fp = backend.fingerprint()
    model_config = types.SimpleNamespace(
        model_id=profile.model_id,
        revision=profile.revision,
        dtype=profile.dtype,
        backend="qwen_hf",
        fingerprint_id=_fp.get("fingerprint_id", ""),
        generation=GenerationConfig(do_sample=False),
    )

    runner = BaselineRunner(
        backend=backend,
        probe_path=str(probe_path),
        output_dir=str(output_dir),
        model_config=model_config,
        resume=True,
        dataset_manifest_path=str(
            project_root / "outputs/full_fiubench/evidence/research_dataset_manifest.json"
        ),
        model_config_path=str(profile_path),
        freeze_verification_path=str(
            project_root / "outputs/full_fiubench/evidence/final_freeze_verification.json"
        ),
        processed_dataset_path=str(
            project_root / "outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_processed.jsonl"
        ),
    )

    # Run preflight
    runner.validate_research_preflight()

    # Run evaluation
    results = runner.run_all(limit=smoke_limit)
    logger.info(f"Evaluation complete: {len(results)} results")

    # Save results to JSONL
    runner.save_results()
    results_path = output_dir / "baseline_results.jsonl"

    # Compute results SHA-256
    results_sha256 = _file_sha256(results_path)
    logger.info(f"Results SHA-256: {results_sha256}")

    # Generate summary (writes baseline_summary.json as side effect)
    runner.generate_summary()

    logger.info("Baseline evaluation complete. Run generate_4b_baseline_manifest.py to generate the canonical manifest.")

    logger.info("=" * 60)
    logger.info("Baseline generation complete!")
    logger.info(f"Total probes: {len(probes)}")
    logger.info(f"Results: {results_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
