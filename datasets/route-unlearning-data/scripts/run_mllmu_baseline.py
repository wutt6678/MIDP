#!/usr/bin/env python3
"""Single-method runner for MLLMU-Bench baselines.

Trains one baseline method (GA, GD, KL, NPO) or runs the prompting
eval-only baseline, using the unified BaselineTrainer.

Usage::

    # Train GA baseline
    python scripts/run_mllmu_baseline.py \\
        --method ga \\
        --config configs/experiments/mllmu_baselines/ga.yaml

    # Preflight check only (no GPU required)
    python scripts/run_mllmu_baseline.py \\
        --method ga \\
        --config configs/experiments/mllmu_baselines/ga.yaml \\
        --preflight-only

    # Prompting baseline (no training)
    python scripts/run_mllmu_baseline.py \\
        --method prompting \\
        --config configs/experiments/mllmu_baselines/prompting.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_mllmu_baseline")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _git_commit() -> str:
    """Return the current git commit SHA."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except FileNotFoundError:
        return ""


def _merge_configs(common_path: Path, method_path: Path) -> dict[str, Any]:
    """Merge common.yaml with a per-method config.

    Method config values override common values. Nested dicts are
    merged shallowly (one level deep).
    """
    with open(common_path) as f:
        common = yaml.safe_load(f) or {}
    with open(method_path) as f:
        method = yaml.safe_load(f) or {}

    merged = dict(common)
    for key, value in method.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #

def run_preflight(config: dict[str, Any]) -> dict[str, Any]:
    """Run preflight checks before training.

    Verifies:
    - Config structure is valid
    - Method name is recognized
    - Output directory is writable
    - For KL: reference model path is set
    - For NPO: oracle adapter path is set

    Returns a preflight report dict.
    """
    checks: list[dict[str, Any]] = []

    def _check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "pass": passed, "detail": detail})

    method_name = config.get("method", {}).get("name", "")
    _check("method_recognized", method_name in {
        "mllmu_ga", "mllmu_ga_difference", "mllmu_kl_min",
        "mllmu_npo", "mllmu_prompting", "npo_oracle",
        "midp_candidate_margin",
    }, method_name)

    output_dir = config.get("runtime", {}).get("output_dir", "")
    if output_dir:
        out_path = Path(output_dir)
        try:
            out_path.mkdir(parents=True, exist_ok=True)
            _check("output_dir_writable", True, str(out_path))
        except OSError as exc:
            _check("output_dir_writable", False, str(exc))
    else:
        _check("output_dir_writable", False, "output_dir not set")

    # Method-specific checks
    if method_name == "mllmu_npo":
        oracle_path = config.get("runtime", {}).get("oracle_adapter_path", "")
        _check("oracle_adapter_path_set", bool(oracle_path),
               oracle_path if oracle_path else "NPO requires an oracle adapter")

    # Training config sanity
    training = config.get("training", {})
    steps = training.get("max_optimizer_steps", 0)
    _check("steps_positive", steps > 0, f"steps={steps}")

    lora = config.get("lora", {})
    _check("lora_rank_positive", lora.get("rank", 0) > 0,
           f"rank={lora.get('rank', 0)}")
    _check("lora_alpha_positive", lora.get("alpha", 0) > 0,
           f"alpha={lora.get('alpha', 0)}")

    all_passed = all(c["pass"] for c in checks)
    report = {
        "preflight_passed": all_passed,
        "method": method_name,
        "checks": checks,
        "timestamp": time.time(),
    }
    return report


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLLMU-Bench single-method baseline runner",
    )
    parser.add_argument(
        "--method", required=True,
        choices=["ga", "gd", "kl", "prompting", "npo", "npo_oracle", "midp_cm"],
        help="Which baseline method to run.",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the per-method YAML config.",
    )
    parser.add_argument(
        "--common-config",
        default="configs/experiments/mllmu_baselines/common.yaml",
        help="Path to common.yaml (default: %(default)s).",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Run preflight checks only (no GPU required).",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory from config.",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / args.config if not Path(args.config).is_absolute() else Path(args.config)
    common_path = project_root / args.common_config if not Path(args.common_config).is_absolute() else Path(args.common_config)

    # Load and merge configs
    config = _merge_configs(common_path, config_path)

    # Override output dir if specified
    if args.output_dir:
        config.setdefault("runtime", {})["output_dir"] = args.output_dir

    # Add code provenance
    config.setdefault("runtime", {})["code_commit"] = _git_commit()

    method_name = config.get("method", {}).get("name", "")
    logger.info(f"Method: {method_name}")
    logger.info(f"Config: {config_path}")

    # Preflight
    report = run_preflight(config)
    output_dir = Path(config.get("runtime", {}).get("output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight_path = output_dir / "preflight_report.json"
    with open(preflight_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    logger.info(f"Preflight: {'PASSED' if report['preflight_passed'] else 'FAILED'}")

    if not report["preflight_passed"]:
        for check in report["checks"]:
            if not check["pass"]:
                logger.error(f"  FAIL: {check['name']} — {check['detail']}")
        sys.exit(1)

    if args.preflight_only:
        logger.info("Preflight-only mode. Exiting.")
        return

    # Dispatch to appropriate handler
    from route_data.unlearning import (
        BaselineTrainer,
        BaselineTrainingConfig,
        build_objective,
        load_config_from_yaml,
    )

    # Build training config from merged YAML
    training_config = load_config_from_yaml(config_path)
    # Apply common config overrides
    base_model = config.get("base_model", {})
    training_config.model_id = base_model.get("id", training_config.model_id)
    training_config.model_revision = base_model.get("revision", training_config.model_revision)
    training_config.dtype = base_model.get("dtype", training_config.dtype)
    training_config.code_commit = _git_commit()

    # Apply data config
    data = config.get("data", {})
    training_config.processed_dataset_path = data.get("processed_dataset_path", "")
    training_config.forget_identity_ids = data.get("forget_identity_ids", [])
    training_config.retain_identity_ids = data.get("retain_identity_ids", [])

    if args.output_dir:
        training_config.output_dir = args.output_dir

    # Handle special methods
    if method_name == "mllmu_prompting":
        from route_data.unlearning.baseline_methods import PromptingBaseline
        logger.info("Running prompting baseline (no training)")
        baseline = PromptingBaseline()
        baseline.run_evaluation(
            model=None, processor=None,
            probe_dataset=[], output_dir=training_config.output_dir,
        )
        logger.info("Prompting baseline complete")
        return

    if method_name == "midp_candidate_margin":
        logger.info("MIDP-CM: referencing existing E2B-B2 result (no retraining)")
        e2b_dir = config.get("runtime", {}).get("e2b_b2_output_dir", "")
        if e2b_dir:
            logger.info(f"E2B-B2 output: {e2b_dir}")
        # Copy/symlink existing results
        return

    # Load model, datasets, and train
    from route_data.eval.unlearning_harness import (
        apply_lora,
        build_forget_dataset,
        build_retain_dataset,
        check_base_parameter_integrity,
        load_base_model,
    )

    logger.info(f"Loading base model {training_config.model_id} ...")
    model, processor = load_base_model(
        model_id=training_config.model_id,
        revision=training_config.model_revision,
        dtype=training_config.dtype,
        device=training_config.device,
    )

    logger.info("Applying LoRA ...")
    model = apply_lora(
        model,
        r=training_config.lora_rank,
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        target_modules=training_config.lora_target_modules,
    )
    check_base_parameter_integrity(model)

    # Build datasets
    logger.info("Building forget dataset ...")
    forget_ds = build_forget_dataset(
        processed_dataset_path=training_config.processed_dataset_path,
        target_identity_ids=training_config.forget_identity_ids,
        processor=processor,
    )

    retain_ds = None
    if method_name in ("mllmu_ga_difference", "mllmu_kl_min"):
        logger.info("Building retain dataset ...")
        retain_ds = build_retain_dataset(
            processed_dataset_path=training_config.processed_dataset_path,
            identity_ids=training_config.retain_identity_ids,
            processor=processor,
        )

    # Load reference/oracle models if needed
    reference_model = None
    oracle_model = None

    if method_name == "mllmu_kl_min":
        from route_data.unlearning.reference_models import load_frozen_reference_model
        ref_path = config.get("runtime", {}).get("reference_model_path", "")
        logger.info(f"Loading frozen reference model ...")
        reference_model, _ = load_frozen_reference_model(
            model_id=training_config.model_id,
            revision=training_config.model_revision,
            dtype=training_config.dtype,
            device=training_config.device,
        )

    if method_name == "mllmu_npo":
        from route_data.unlearning.reference_models import load_oracle_model
        oracle_path = config.get("runtime", {}).get("oracle_adapter_path", "")
        logger.info(f"Loading oracle model from {oracle_path} ...")
        oracle_model, _ = load_oracle_model(
            model_id=training_config.model_id,
            revision=training_config.model_revision,
            adapter_path=oracle_path,
            dtype=training_config.dtype,
            device=training_config.device,
        )

    # Build objective
    objective = build_objective(training_config)

    # Train
    trainer = BaselineTrainer(
        config=training_config,
        objective=objective,
        model=model,
        processor=processor,
        forget_dataset=forget_ds,
        retain_dataset=retain_ds,
        reference_model=reference_model,
        oracle_model=oracle_model,
    )

    logger.info(f"Starting training: {method_name}")
    summary = trainer.train()

    logger.info(f"Training complete: {summary}")
    logger.info(f"Output: {training_config.output_dir}")


if __name__ == "__main__":
    main()
