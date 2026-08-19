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
# Common evaluation wiring (P0-1)
# --------------------------------------------------------------------------- #

def _run_eval(
    method_label: str,
    model,
    processor,
    adapter_path,
    config: dict[str, Any],
    training_config,
    *,
    eval_subdir: str = "",
    backend_override: Any = None,
    objective_name: str = "",
    trainable_adapter: Any = None,
) -> dict[str, Any]:
    """Call evaluate_intervention with a complete PostEvalConfig (P0-1).

    Parameters
    ----------
    method_label:
        Human-readable method identifier (canonical suite ID).
    model, processor:
        The model in its final intervention state and its processor.
    adapter_path:
        Path to the LoRA adapter checkpoint, or *None*.
    config:
        Merged common + method YAML config dict.
    training_config:
        The training config object (carries model/dtype/device/seed).
    eval_subdir:
        Optional subdirectory name under the eval output directory.
        Used by MANU to isolate prune-rate outputs (P0-9/10).
    backend_override:
        Optional pre-built backend to pass through to
        ``evaluate_intervention()`` (P0-7 prompting).
    objective_name:
        P0-10: Training objective name (e.g. ``mllmu_ga``).  Defaults
        to *method_label* when empty.
    """
    from route_data.eval.post_unlearning_eval import (
        PostEvalConfig,
        evaluate_intervention,
    )

    data = config.get("data", {})
    project_root = Path(__file__).resolve().parent.parent

    probe_path = project_root / data.get(
        "route_probe_path",
        "outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/"
        "fiubench_route_conflict_eval.jsonl",
    )

    # Resolve output directory, optionally with a rate-specific subdir.
    eval_output_dir = Path(training_config.output_dir) / "eval"
    if eval_subdir:
        eval_output_dir = eval_output_dir / eval_subdir
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve frozen-contract paths from the merged config.
    baseline_results_path = data.get("baseline_results_path", "")
    baseline_manifest_path = data.get("baseline_manifest_path", "")
    research_dataset_manifest_path = data.get("research_dataset_manifest_path", "")
    freeze_verification_path = data.get("freeze_verification_path", "")
    processed_dataset_path = data.get("processed_dataset_path", "")
    selection_manifest_sha256 = data.get("selection_manifest_sha256", "")
    selection_manifest_path = data.get("selection_manifest_path", "")
    route_probe_sha256 = data.get("route_probe_sha256", "")

    # P0-1: Build a complete PostEvalConfig with the full frozen contract.
    post_config = PostEvalConfig(
        model_id=getattr(training_config, "model_id", "Qwen/Qwen3.5-9B"),
        model_revision=getattr(training_config, "model_revision", ""),
        dtype=getattr(training_config, "dtype", "bfloat16"),
        device=str(getattr(training_config, "device", "cuda:0")),
        seed=getattr(training_config, "seed", 17),
        probe_path=str(probe_path),
        baseline_results_path=str(baseline_results_path),
        baseline_manifest_path=str(baseline_manifest_path),
        output_dir=str(eval_output_dir),
        selection_manifest_sha256=selection_manifest_sha256,
        selection_manifest_path=str(selection_manifest_path),
        code_commit=_git_commit(),
        dataset_manifest_path=str(research_dataset_manifest_path),
        freeze_verification_path=str(freeze_verification_path),
        processed_dataset_path=str(processed_dataset_path),
        model_config_path="",
        route_probe_sha256=route_probe_sha256,
        # P0-8: Profile provenance
        model_key=getattr(trainable_adapter.profile, "key", "") if trainable_adapter else getattr(training_config, "model_key", ""),
        processor_id=getattr(trainable_adapter.profile, "processor_id", "") if trainable_adapter else getattr(training_config, "processor_id", ""),
        processor_revision=getattr(trainable_adapter.profile, "processor_revision", "") if trainable_adapter else getattr(training_config, "processor_revision", ""),
        model_profile_sha256=getattr(training_config, "model_profile_sha256", ""),
        adapter_family=getattr(trainable_adapter.profile, "adapter_name", "") if trainable_adapter else getattr(training_config, "adapter_family", ""),
    )

    # P0-1: Fail immediately if any mandatory frozen-contract field is empty.
    mandatory = {
        "probe_path": post_config.probe_path,
        "baseline_results_path": post_config.baseline_results_path,
        "baseline_manifest_path": post_config.baseline_manifest_path,
        "dataset_manifest_path": post_config.dataset_manifest_path,
        "freeze_verification_path": post_config.freeze_verification_path,
        "processed_dataset_path": post_config.processed_dataset_path,
        "selection_manifest_path": post_config.selection_manifest_path,
    }
    empty_fields = [k for k, v in mandatory.items() if not v]
    if empty_fields:
        raise RuntimeError(
            f"_run_eval({method_label}): mandatory PostEvalConfig fields "
            f"are empty: {empty_fields}. Check common.yaml provenance."
        )

    # P0-7: Validate baseline/model identity when both baseline manifest
    # and profile provenance are available.  This prevents cross-model
    # deltas (e.g. GLM post - Qwen pre).
    _bl_manifest = Path(post_config.baseline_manifest_path)
    if _bl_manifest.is_file() and post_config.model_key:
        from route_data.eval.post_unlearning_eval import (
            BaselineBinding,
            validate_baseline_model_identity,
        )
        with open(_bl_manifest) as _blf:
            _bl_data = json.load(_blf)
        _bl_model = _bl_data.get("model", {})
        _binding = BaselineBinding(
            manifest_path=str(_bl_manifest),
            model_id=_bl_model.get("id", ""),
            model_revision=_bl_model.get("revision", ""),
            processor_revision=_bl_model.get("processor_revision", ""),
        )
        _id_errors = validate_baseline_model_identity(
            _binding,
            model_key=post_config.model_key,
            model_id=post_config.model_id,
            model_revision=post_config.model_revision,
            processor_revision=post_config.processor_revision,
        )
        if _id_errors:
            raise RuntimeError(
                f"Baseline/model identity mismatch for "
                f"model_key={post_config.model_key!r}: {_id_errors}"
            )

    logger.info(f"Running common 500-probe evaluation: {method_label}")
    result = evaluate_intervention(
        model=model,
        processor=processor,
        adapter_path=str(adapter_path) if adapter_path else None,
        probe_dataset_path=str(probe_path),
        output_dir=str(eval_output_dir),
        config=post_config,
        baseline_results_path=str(baseline_results_path),
        method_name=method_label,
        objective_name=objective_name or method_label,
        backend_override=backend_override,
        trainable_adapter=trainable_adapter,
    )
    logger.info(
        f"Evaluation complete: {result.get('exact_pair_count', 0)} pairs, "
        f"{result.get('inference_errors', 0)} errors"
    )
    return result


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
        "mmunlearner", "manu", "r2mu_adapted",
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
        choices=["ga", "gd", "kl", "prompting", "npo", "npo_oracle", "midp_cm",
                 "mmunlearner", "manu", "r2mu_adapted"],
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
    parser.add_argument(
        "--model-profile",
        help="Path to model profile YAML (e.g. configs/models/unlearning/glm46v_flash.yaml). "
             "When provided, uses the trainable adapter for model-agnostic dispatch.",
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

    # Load trainable adapter if model profile is provided
    _model_adapter = None
    _profile = None
    _profile_sha256 = ""
    if args.model_profile:
        from route_data.models.trainable.registry import (
            compute_profile_sha256,
            create_adapter,
            load_profile_from_yaml,
            validate_research_profile,
        )
        profile_path = (
            project_root / args.model_profile
            if not Path(args.model_profile).is_absolute()
            else Path(args.model_profile)
        )
        _profile = load_profile_from_yaml(profile_path)
        _profile_sha256 = compute_profile_sha256(profile_path)

        # Validate profile for research mode
        errors = validate_research_profile(_profile)
        if errors:
            logger.error(f"Profile validation failed for {_profile.key}:")
            for err in errors:
                logger.error(f"  - {err}")
            sys.exit(1)

        # Enforce capability flags BEFORE model download
        method_name_for_cap = config.get("method", {}).get("name", "")
        _METHOD_CAPABILITY_MAP = {
            "mllmu_prompting": "supports_prompting",
            "midp_candidate_margin": "supports_candidate_margin",
            "mllmu_ga": "supports_ga",
            "mllmu_ga_difference": "supports_gd",
            "mllmu_kl_min": "supports_kl",
            "mllmu_npo": "supports_npo",
            "npo_oracle": "supports_npo",
            "mmunlearner": "supports_mmunlearner",
            "manu": "supports_manu",
            "r2mu_adapted": "supports_r2mu",
        }
        required_cap = _METHOD_CAPABILITY_MAP.get(method_name_for_cap)
        if required_cap and not getattr(_profile, required_cap, False):
            logger.error(
                f"{_profile.key} does not support {method_name_for_cap} "
                f"({required_cap}=false). Fix the profile or choose a "
                f"different method."
            )
            sys.exit(1)

        # Create adapter with the YAML profile (single source of truth)
        _model_adapter = create_adapter(_profile.key, profile=_profile)

        # Provenance log
        logger.info("=" * 60)
        logger.info("Model profile provenance")
        logger.info(f"  model_key          = {_profile.key}")
        logger.info(f"  model_id           = {_profile.model_id}")
        logger.info(f"  model_revision     = {_profile.revision}")
        logger.info(f"  processor_id       = {_profile.processor_id}")
        logger.info(f"  processor_revision = {_profile.processor_revision}")
        logger.info(f"  adapter_name       = {_profile.adapter_name}")
        logger.info(f"  dtype              = {_profile.dtype}")
        logger.info(f"  LoRA rank          = {_profile.lora_rank}")
        logger.info(f"  LoRA alpha         = {_profile.lora_alpha}")
        logger.info(f"  LoRA scope_regex   = {_profile.lora_scope_regex}")
        logger.info(f"  profile_sha256     = {_profile_sha256}")
        logger.info("=" * 60)

        # Store model key in config for downstream use
        config.setdefault("model", {})["key"] = _profile.key

        # P0-7: Override baseline paths with model-specific locations.
        # The pre-unlearning baseline must be per-model to prevent
        # cross-model deltas (e.g. GLM post - Qwen pre).
        _baseline_dir = (
            Path(__file__).resolve().parent.parent
            / "outputs" / "experiments" / "pre_unlearning"
            / _profile.key / "baseline_v1"
        )
        _data_cfg = config.setdefault("data", {})
        _data_cfg["baseline_results_path"] = str(
            _baseline_dir / "baseline_results.jsonl"
        )
        _data_cfg["baseline_manifest_path"] = str(
            _baseline_dir / "baseline_manifest.json"
        )
        logger.info(
            f"Model-specific baseline: {_baseline_dir}"
        )

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

    # P0: Profile is the single source of truth — override legacy config
    if _profile is not None:
        training_config.model_id = _profile.model_id
        training_config.model_revision = _profile.revision
        training_config.dtype = _profile.dtype
        training_config.lora_rank = _profile.lora_rank
        training_config.lora_alpha = _profile.lora_alpha
        training_config.lora_dropout = _profile.lora_dropout
        # P0-8: Bind profile provenance into training config
        training_config.model_key = _profile.key
        training_config.processor_id = _profile.processor_id
        training_config.processor_revision = _profile.processor_revision
        training_config.model_profile_sha256 = _profile_sha256
        training_config.adapter_family = _profile.adapter_name

    # Apply data config
    data = config.get("data", {})
    training_config.processed_dataset_path = data.get("processed_dataset_path", "")
    training_config.forget_identity_ids = data.get("forget_identity_ids", [])
    training_config.retain_identity_ids = data.get("retain_identity_ids", [])

    if args.output_dir:
        training_config.output_dir = args.output_dir

    # ------------------------------------------------------------------ #
    # Load base model (needed for ALL methods including prompting)        #
    # ------------------------------------------------------------------ #

    logger.info(f"Loading base model {training_config.model_id} ...")

    if _model_adapter is not None:
        # P0: Profile-active path — adapter is authoritative
        from route_data.eval.unlearning_harness import load_base_model_via_adapter
        model, processor = load_base_model_via_adapter(
            _model_adapter,
            device=training_config.device,
            training=True,
        )
    else:
        # Legacy Qwen-only path
        from route_data.eval.unlearning_harness import load_base_model
        model, processor = load_base_model(
            model_id=training_config.model_id,
            revision=training_config.model_revision,
            dtype=training_config.dtype,
            device=training_config.device,
        )

    # Handle special methods that do NOT use BaselineTrainer
    if method_name == "mllmu_prompting":
        # P0-7: The canonical prompting evaluation uses the system-prompt-
        # aware backend through the common 500-probe evaluator.  There is
        # NO duplicate plain-model evaluation.
        from route_data.config import ModelConfig
        from route_data.unlearning.baseline_methods import (
            MLLMU_PRIVACY_SYSTEM_PROMPT,
            _PromptingBackend,
        )

        logger.info("Running prompting via common evaluator with PromptingBackend")

        # Build the same ModelConfig used by other methods.
        prompting_model_config = ModelConfig(
            backend="qwen_hf",
            model_id=getattr(training_config, "model_id", "Qwen/Qwen3.5-9B"),
            revision=getattr(training_config, "model_revision", ""),
            dtype=getattr(training_config, "dtype", "bfloat16"),
            device_map=str(getattr(training_config, "device", "cuda:0")),
            seed=getattr(training_config, "seed", 17),
        )

        # Create the inner backend — use adapter if available, else Qwen
        if _model_adapter is not None:
            inner_backend = _model_adapter.to_eval_backend(
                model=model,
                processor=processor,
                model_config=prompting_model_config,
            )
        else:
            from route_data.models.qwen import QwenHFBackend
            inner_backend = QwenHFBackend.from_loaded_model(
                config=prompting_model_config,
                model=model,
                processor=processor,
                resolved_revision=training_config.model_revision,
            )
        prompting_backend = _PromptingBackend(
            inner=inner_backend,
            system_prompt=MLLMU_PRIVACY_SYSTEM_PROMPT,
        )

        # Single canonical evaluation through the common evaluator (P0-7/8).
        # The full frozen contract is passed via PostEvalConfig inside _run_eval.
        eval_result = _run_eval(
            "prompting", model, processor, None, config, training_config,
            backend_override=prompting_backend,
            trainable_adapter=_model_adapter,
        )
        if not eval_result.get("strict_validation_pass"):
            logger.error("Prompting: strict validation FAILED")
            sys.exit(1)
        return

    if method_name == "midp_candidate_margin":
        # P0-14: Bind validated E2B-B2 evidence via the real artifact binder.
        # No retraining; reads actual E2B-B2 artifacts and transforms them
        # into the common comparison schema.
        logger.info("MIDP-CM: binding existing E2B-B2 result (no retraining)")
        e2b_dir = config.get("runtime", {}).get("e2b_b2_output_dir", "")
        output_dir = Path(training_config.output_dir)
        eval_dir = output_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        if not e2b_dir:
            logger.error("MIDP-CM: e2b_b2_output_dir not set in config")
            sys.exit(1)

        e2b_path = Path(e2b_dir)
        if not e2b_path.exists():
            logger.error(
                f"MIDP-CM: E2B-B2 output directory not found: {e2b_path}"
            )
            sys.exit(1)

        # P0-14: Use the real evidence binder.
        from route_data.unlearning.e2b_evidence_binding import bind_e2b_b2_result

        selection_path = config.get("data", {}).get("selection_manifest_path", "")
        try:
            bind_e2b_b2_result(
                e2b_dir=e2b_path,
                output_dir=eval_dir,
                selection_manifest_path=selection_path,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error(f"MIDP-CM: evidence binding FAILED — {exc}")
            sys.exit(1)

        logger.info(
            f"MIDP-CM: evidence bound from {e2b_path} → {eval_dir / 'eval_results.json'}"
        )
        return

    logger.info("Applying LoRA ...")
    if _model_adapter is not None:
        # P0: Profile-active path — adapter resolves targets
        from route_data.eval.unlearning_harness import (
            apply_lora_via_adapter,
            check_base_parameter_integrity,
        )
        model = apply_lora_via_adapter(model, _model_adapter)
    else:
        # Legacy Qwen-only path
        from route_data.eval.unlearning_harness import (
            apply_lora,
            check_base_parameter_integrity,
        )
        model = apply_lora(
            model,
            r=training_config.lora_rank,
            lora_alpha=training_config.lora_alpha,
            lora_dropout=training_config.lora_dropout,
            target_modules=training_config.lora_target_modules,
        )
    check_base_parameter_integrity(model)

    # Build datasets
    # npo_oracle is retain-only: no forget dataset needed
    is_retain_only = config.get("training", {}).get("retain_only", False)

    forget_ds = None
    if not is_retain_only:
        logger.info("Building forget dataset ...")
        if _model_adapter is not None:
            from route_data.eval.unlearning_harness import (
                build_multimodal_forget_dataset,
            )
            forget_ds = build_multimodal_forget_dataset(
                processed_dataset_path=training_config.processed_dataset_path,
                target_identity_ids=training_config.forget_identity_ids,
                processor=processor,
                adapter=_model_adapter,
            )
        else:
            from route_data.eval.unlearning_harness import build_forget_dataset
            forget_ds = build_forget_dataset(
                processed_dataset_path=training_config.processed_dataset_path,
                target_identity_ids=training_config.forget_identity_ids,
                processor=processor,
            )

    retain_ds = None
    if method_name in ("mllmu_ga_difference", "mllmu_kl_min", "npo_oracle",
                       "mmunlearner", "manu", "r2mu_adapted"):
        logger.info("Building retain dataset ...")
        if _model_adapter is not None:
            from route_data.eval.unlearning_harness import (
                build_multimodal_retain_dataset,
            )
            retain_ds = build_multimodal_retain_dataset(
                processed_dataset_path=training_config.processed_dataset_path,
                retain_identity_ids=training_config.retain_identity_ids,
                processor=processor,
                adapter=_model_adapter,
            )
        else:
            from route_data.eval.unlearning_harness import build_retain_dataset
            retain_ds = build_retain_dataset(
                processed_dataset_path=training_config.processed_dataset_path,
                retain_identity_ids=training_config.retain_identity_ids,
                processor=processor,
            )

    # Load reference/oracle models if needed
    reference_model = None
    oracle_model = None

    if method_name == "mllmu_kl_min":
        logger.info("Loading frozen reference model ...")
        if _model_adapter is not None:
            from route_data.unlearning.reference_models import (
                load_frozen_reference_model_via_adapter,
            )
            reference_model, _ = load_frozen_reference_model_via_adapter(
                _model_adapter,
                device=training_config.device,
            )
        else:
            from route_data.unlearning.reference_models import load_frozen_reference_model
            config.get("runtime", {}).get("reference_model_path", "")
            reference_model, _ = load_frozen_reference_model(
                model_id=training_config.model_id,
                revision=training_config.model_revision,
                dtype=training_config.dtype,
                device=training_config.device,
            )

    if method_name == "mllmu_npo":
        oracle_path = config.get("runtime", {}).get("oracle_adapter_path", "")
        logger.info(f"Loading oracle model from {oracle_path} ...")
        if _model_adapter is not None:
            from route_data.unlearning.reference_models import (
                load_oracle_model_via_adapter,
            )
            oracle_model, _ = load_oracle_model_via_adapter(
                _model_adapter,
                adapter_path=oracle_path,
                device=training_config.device,
            )
        else:
            from route_data.unlearning.reference_models import load_oracle_model
            oracle_model, _ = load_oracle_model(
                model_id=training_config.model_id,
                revision=training_config.model_revision,
                adapter_path=oracle_path,
                dtype=training_config.dtype,
                device=training_config.device,
            )

    # ------------------------------------------------------------------ #
    # Structural / representation baselines (B7–B9)
    # These have their own training loops and do NOT use BaselineTrainer.
    # ------------------------------------------------------------------ #
    if method_name in ("mmunlearner", "manu", "r2mu_adapted"):
        from torch.utils.data import DataLoader

        # Use adapter collation if available, else fall back to Qwen
        if _model_adapter is not None:
            collate_fn = _model_adapter.collate
        else:
            from route_data.eval.unlearning_harness import qwen_collate_fn
            collate_fn = qwen_collate_fn

        batch_size = config.get("training", {}).get("batch_size", 1)
        forget_loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=True,
                                   collate_fn=collate_fn)
        retain_loader = (
            DataLoader(retain_ds, batch_size=batch_size, shuffle=True,
                       collate_fn=collate_fn)
            if retain_ds is not None else None
        )

        if method_name == "mmunlearner":
            from route_data.unlearning import MMUnlearner, MMUnlearnerConfig
            method_cfg = MMUnlearnerConfig(
                saliency_n_samples=config.get("saliency_n_samples", 32),
                target_sparsity=config.get("target_sparsity", 0.5),
                mask_granularity=config.get("mask_granularity", "element"),
                modality=config.get("modality", "both"),
                learning_rate=config.get("training", {}).get("learning_rate", 2e-5),
                num_optimizer_steps=config.get("training", {}).get("max_optimizer_steps", 125),
                gradient_accumulation_steps=config.get("training", {}).get("gradient_accumulation_steps", 4),
                max_grad_norm=config.get("training", {}).get("max_grad_norm", 1.0),
                output_dir=training_config.output_dir,
            )
            runner = MMUnlearner(method_cfg)
            summary = runner.run(model, forget_loader, retain_loader, device=str(training_config.device))

        elif method_name == "manu":
            from route_data.unlearning import MANU, MANUConfig
            method_cfg = MANUConfig(
                primary_prune_fraction=config.get("primary_prune_fraction", 0.10),
                secondary_prune_fraction=config.get("secondary_prune_fraction", 0.05),
                importance_n_samples=config.get("importance_n_samples", 32),
                neuron_unit=config.get("neuron_unit", "mlp_intermediate"),
                num_optimizer_steps=config.get("training", {}).get("max_optimizer_steps", 0),
                output_dir=training_config.output_dir,
            )

            # Eval callback: evaluates the live pruned model BEFORE restoration.
            # PeftModel.save_pretrained() cannot preserve base-weight zeroing,
            # so evaluation must happen on the in-memory pruned model (P0-8).
            # P0-9/10: Each prune rate gets an isolated eval subdirectory.
            def _manu_eval_callback(
                rate_str: str, pruned_model,
            ) -> dict[str, Any]:
                return _run_eval(
                    f"manu_prune_{rate_str}",
                    pruned_model, processor, None,
                    config, training_config,
                    eval_subdir=f"prune_{rate_str}",
                    trainable_adapter=_model_adapter,
                )

            runner = MANU(method_cfg)
            summary = runner.run(
                model, forget_loader, retain_loader,
                device=str(training_config.device),
                eval_callback=_manu_eval_callback,
            )

        elif method_name == "r2mu_adapted":
            from route_data.unlearning import R2MUAdapted, R2MUAdaptedConfig
            logger.info("Loading frozen reference model for R²MU-adapted ...")
            if _model_adapter is not None:
                from route_data.unlearning.reference_models import (
                    load_frozen_reference_model_via_adapter,
                )
                frozen_model, _ = load_frozen_reference_model_via_adapter(
                    _model_adapter,
                    device=training_config.device,
                )
            else:
                from route_data.unlearning.reference_models import load_frozen_reference_model
                config.get("runtime", {}).get("reference_model_path", "")
                frozen_model, _ = load_frozen_reference_model(
                    model_id=training_config.model_id,
                    revision=training_config.model_revision,
                    dtype=training_config.dtype,
                    device=training_config.device,
                )
            method_cfg = R2MUAdaptedConfig(
                candidate_layers=config.get("candidate_layers", [8, 16, 24, 32]),
                n_select_layers=config.get("n_select_layers", 2),
                target_seed=config.get("target_seed", 42),
                target_norm=config.get("target_norm", 1.0),
                gamma=config.get("gamma", 1.0),
                learning_rate=config.get("training", {}).get("learning_rate", 2e-5),
                num_optimizer_steps=config.get("training", {}).get("max_optimizer_steps", 125),
                gradient_accumulation_steps=config.get("training", {}).get("gradient_accumulation_steps", 4),
                max_grad_norm=config.get("training", {}).get("max_grad_norm", 1.0),
                checkpoint_steps=config.get("training", {}).get("checkpoint_steps", [1, 5, 10, 25, 50, 60, 75, 90, 125]),
                output_dir=training_config.output_dir,
            )
            runner = R2MUAdapted(method_cfg)
            summary = runner.run(model, frozen_model, forget_loader, retain_loader, device=str(training_config.device))

        logger.info(f"Training complete: {summary}")
        logger.info(f"Output: {training_config.output_dir}")

        # Common 500-probe evaluation for structural methods (P0-1).
        if method_name in ("mmunlearner", "r2mu_adapted"):
            adapter_p = (
                Path(training_config.output_dir)
                / "checkpoints" / "adapter_final"
            )
            eval_result = _run_eval(
                method_name, model, processor, adapter_p, config, training_config,
                trainable_adapter=_model_adapter,
            )
            if not eval_result.get("strict_validation_pass"):
                logger.error(f"{method_name}: strict validation FAILED")
                sys.exit(1)

        # P0-11: MANU hard-fail — verify each prune-rate evaluation produced
        # a valid eval_results.json.  Missing results are a hard error.
        if method_name == "manu":
            for rate_str in ("05", "10"):
                eval_results_path = (
                    Path(training_config.output_dir)
                    / "eval" / f"prune_{rate_str}" / "eval_results.json"
                )
                if not eval_results_path.is_file():
                    logger.error(
                        f"MANU: eval_results.json missing for prune rate "
                        f"{rate_str}: {eval_results_path}"
                    )
                    sys.exit(1)
                with open(eval_results_path) as f:
                    rate_result = json.load(f)
                if not rate_result.get("strict_validation_pass"):
                    logger.error(
                        f"MANU prune_{rate_str}: strict validation FAILED"
                    )
                    sys.exit(1)
                logger.info(
                    f"MANU prune_{rate_str}: "
                    f"{rate_result.get('exact_pair_count', 0)} pairs, "
                    f"{rate_result.get('inference_errors', 0)} errors"
                )

        return

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
        adapter=_model_adapter,
    )

    logger.info(f"Starting training: {method_name}")
    summary = trainer.train()

    logger.info(f"Training complete: {summary}")
    logger.info(f"Output: {training_config.output_dir}")

    # Save LoRA adapter and run common 500-probe evaluation (P0-1).
    adapter_save_path = (
        Path(training_config.output_dir) / "checkpoints" / "adapter_final"
    )
    adapter_save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_save_path))
    logger.info(f"Saved LoRA adapter: {adapter_save_path}")

    # P0-10: Map config objective name → canonical suite method ID.
    _OBJECTIVE_TO_CANONICAL = {
        "mllmu_ga": "ga",
        "mllmu_ga_difference": "gd",
        "mllmu_kl_min": "kl",
        "mllmu_npo": "npo",
        "npo_oracle": "npo_oracle",
        "midp_candidate_margin": "midp_cm",
    }
    canonical_id = _OBJECTIVE_TO_CANONICAL.get(method_name, method_name)

    eval_result = _run_eval(
        canonical_id, model, processor, adapter_save_path, config, training_config,
        objective_name=method_name,
        trainable_adapter=_model_adapter,
    )
    if not eval_result.get("strict_validation_pass"):
        logger.error(f"{method_name}: strict validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
