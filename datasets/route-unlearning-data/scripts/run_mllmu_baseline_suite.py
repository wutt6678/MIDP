#!/usr/bin/env python3
"""Sequential orchestrator for the full MLLMU-Bench baseline suite.

Runs all baseline methods in order:
  Phase A: Preflight checks (all methods)
  Phase B: Prompting baseline (no training)
  Phase C: GA training
  Phase D: GD training
  Phase E: KL training (requires reference model)
  Phase F: NPO oracle training → NPO unlearning
  Phase G: MIDP-CM reference (existing E2B-B2)

Usage::

    # Full suite
    python scripts/run_mllmu_baseline_suite.py

    # Preflight only
    python scripts/run_mllmu_baseline_suite.py --preflight-only

    # Skip specific methods
    python scripts/run_mllmu_baseline_suite.py --skip prompting --skip midp_cm

    # Run specific methods only
    python scripts/run_mllmu_baseline_suite.py --only ga --only gd
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_mllmu_baseline_suite")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs" / "experiments" / "mllmu_baselines"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "mllmu_baselines"

# Execution order per plan Section 1.7
METHOD_ORDER = [
    ("prompting", "B4", "Prompting (eval-only)"),
    ("ga", "B1", "Gradient Ascent"),
    ("gd", "B2", "Gradient Difference"),
    ("kl", "B3", "KL Minimization"),
    ("npo_oracle", "B5-oracle", "NPO Oracle Training"),
    ("npo", "B5", "Negative Preference Optimization"),
    ("midp_cm", "B6", "MIDP Candidate Margin (existing)"),
    ("mmunlearner", "B7", "MMUnlearner (saliency gradient masks)"),
    ("manu", "B8", "MANU (neuron pruning)"),
    ("r2mu_adapted", "B9", "R²MU-adapted (representation misdirection)"),
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except FileNotFoundError:
        return ""


def _load_common_config() -> dict[str, Any]:
    """Load the common configuration."""
    common_path = CONFIGS_DIR / "common.yaml"
    with open(common_path) as f:
        return yaml.safe_load(f) or {}


def _load_method_config(method: str) -> dict[str, Any]:
    """Load a per-method configuration, merged with common."""
    common = _load_common_config()
    method_path = CONFIGS_DIR / f"{method}.yaml"
    if not method_path.exists():
        raise FileNotFoundError(f"Config not found: {method_path}")
    with open(method_path) as f:
        method_cfg = yaml.safe_load(f) or {}

    merged = dict(common)
    for key, value in method_cfg.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


# Methods for which NPO-oracle is a training dependency, not a final
# comparison result.  Everything else in METHOD_ORDER is a comparison method.
_ORACLE_ONLY_METHODS = {"npo_oracle"}

# Comparison methods that MUST produce a valid eval_results.json.
COMPARISON_METHODS = [m[0] for m in METHOD_ORDER if m[0] not in _ORACLE_ONLY_METHODS]


def _validate_eval_result(eval_metrics: dict[str, Any], expected_method: str) -> str | None:
    """Validate a loaded eval_results.json.

    Returns None on success or an error-message string on failure.
    """
    # exact_pair_count must be 500
    pair_count = eval_metrics.get("exact_pair_count", eval_metrics.get("actual_pair_count", 0))
    if pair_count != 500:
        return f"exact_pair_count={pair_count}, expected 500"

    # inference_errors must be 0
    errors = eval_metrics.get("inference_errors", -1)
    if errors != 0:
        return f"inference_errors={errors}, expected 0"

    # strict_validation_pass must be true
    if not eval_metrics.get("strict_validation_pass", False):
        return "strict_validation_pass is not true"

    # exact_pairing_pass must be true
    if not eval_metrics.get("exact_pairing_pass", False):
        return "exact_pairing_pass is not true"

    # results_path must exist (if provided)
    results_path = eval_metrics.get("results_path", "")
    if results_path and not Path(results_path).is_file():
        return f"results_path does not exist: {results_path}"

    # manifest SHA must be non-empty (if the field is present)
    manifest_sha = eval_metrics.get("manifest_sha256", "")
    if "manifest_sha256" in eval_metrics and not manifest_sha:
        return "manifest_sha256 is empty"

    # delta fields must be present when applicable
    for field in ("delta_target", "delta_retain", "delta_control"):
        if field in eval_metrics and not eval_metrics[field]:
            return f"{field} is present but empty"

    return None


def _run_single_method(
    method: str,
    config: dict[str, Any],
    suite_state: dict[str, Any],
) -> dict[str, Any]:
    """Run a single baseline method as a subprocess.

    Parameters
    ----------
    method:
        Method key (ga, gd, kl, etc.).
    config:
        Merged config dict.
    suite_state:
        Accumulated state from previous methods (e.g., oracle adapter path).

    Returns
    -------
    result:
        Dict with status, output_dir, elapsed_seconds, and optionally
        eval_metrics loaded from the subprocess eval_results.json.
    """
    output_dir = config.get("runtime", {}).get("output_dir", "")
    if not output_dir:
        output_dir = str(OUTPUT_ROOT / method)
        config.setdefault("runtime", {})["output_dir"] = output_dir

    # Inject runtime dependencies from suite state
    if method in ("kl", "r2mu_adapted"):
        ref_path = suite_state.get("reference_model_path", "")
        if ref_path:
            config.setdefault("runtime", {})["reference_model_path"] = ref_path

    if method == "npo":
        oracle_path = suite_state.get("oracle_adapter_path", "")
        if oracle_path:
            config.setdefault("runtime", {})["oracle_adapter_path"] = oracle_path

    # Write merged config (with injected runtime values) to temp YAML
    # so the subprocess can read the complete config from disk
    tmp_config = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w", suffix=f"_{method}.yaml", delete=False,
    )
    yaml.dump(config, tmp_config, default_flow_style=False)
    tmp_config.close()

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_mllmu_baseline.py"),
        "--method", method,
        "--config", tmp_config.name,
    ]

    logger.info(f"{'='*60}")
    logger.info(f"Running: {method}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*60}")

    t0 = time.time()
    try:
        subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=False,
            check=True,
        )
        elapsed = time.time() - t0
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - t0
        logger.error(f"Method {method} FAILED (exit code {exc.returncode})")
        return {
            "method": method,
            "status": "training_failed",
            "error": str(exc),
            "output_dir": output_dir,
            "elapsed_seconds": elapsed,
        }
    finally:
        # Clean up temp config file
        import os
        try:
            os.unlink(tmp_config.name)
        except OSError:
            pass

    # --- Subprocess exited 0.  Now load eval_results.json (P0-2). ---
    # MANU produces per-rate eval dirs; other methods write to <output>/eval/.
    if method == "manu":
        # MANU evaluation is handled per-prune-rate inside the subprocess.
        # We look for eval_results.json under each prune rate dir.
        eval_results_by_rate = {}
        all_rates_valid = True
        for rate in [0.05, 0.10]:
            rate_str = f"{rate * 100:.0f}"
            eval_path = Path(output_dir) / "eval" / f"manu_prune_{rate_str}" / "eval_results.json"
            if not eval_path.is_file():
                # Also check directly under checkpoints
                eval_path = Path(output_dir) / "checkpoints" / f"prune_{rate_str}" / "eval_results.json"
            if eval_path.is_file():
                with open(eval_path) as f:
                    metrics = json.load(f)
                err = _validate_eval_result(metrics, f"manu_prune_{rate_str}")
                if err:
                    logger.error(f"MANU prune_{rate_str} eval invalid: {err}")
                    all_rates_valid = False
                eval_results_by_rate[f"prune_{rate_str}"] = metrics
            else:
                logger.error(f"MANU prune_{rate_str}: eval_results.json not found")
                all_rates_valid = False

        if not eval_results_by_rate:
            return {
                "method": method,
                "status": "missing_eval",
                "output_dir": output_dir,
                "elapsed_seconds": elapsed,
            }
        if not all_rates_valid:
            return {
                "method": method,
                "status": "invalid_eval",
                "output_dir": output_dir,
                "elapsed_seconds": elapsed,
                "eval_metrics": eval_results_by_rate,
            }
        return {
            "method": method,
            "status": "success",
            "output_dir": output_dir,
            "elapsed_seconds": elapsed,
            "eval_metrics": eval_results_by_rate,
        }

    # Standard single-eval methods
    eval_path = Path(output_dir) / "eval" / "eval_results.json"
    if not eval_path.is_file():
        logger.error(f"Method {method}: eval_results.json not found at {eval_path}")
        return {
            "method": method,
            "status": "missing_eval",
            "output_dir": output_dir,
            "elapsed_seconds": elapsed,
        }

    with open(eval_path) as f:
        eval_metrics = json.load(f)

    validation_error = _validate_eval_result(eval_metrics, method)
    if validation_error:
        logger.error(f"Method {method}: eval invalid — {validation_error}")
        return {
            "method": method,
            "status": "invalid_eval",
            "output_dir": output_dir,
            "elapsed_seconds": elapsed,
            "eval_metrics": eval_metrics,
            "eval_error": validation_error,
        }

    logger.info(f"Method {method}: eval_results.json loaded and validated")
    return {
        "method": method,
        "status": "success",
        "output_dir": output_dir,
        "elapsed_seconds": elapsed,
        "eval_metrics": eval_metrics,
    }


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #

def run_suite_preflight(
    methods: list[str],
    common_config: dict[str, Any],
) -> dict[str, Any]:
    """Run preflight checks for all methods.

    Returns a suite-level preflight report.
    """
    logger.info("Running suite-wide preflight ...")

    checks: list[dict[str, Any]] = []

    def _check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "pass": passed, "detail": detail})

    # Common config checks
    training = common_config.get("training", {})
    _check("seed_set", training.get("seed") == 17,
           f"seed={training.get('seed', 'NOT SET')}")
    _check("steps_set", training.get("max_optimizer_steps") == 125,
           f"steps={training.get('max_optimizer_steps', 'NOT SET')}")
    _check("lr_set", training.get("learning_rate") == 2e-5,
           f"lr={training.get('learning_rate', 'NOT SET')}")

    lora = common_config.get("lora", {})
    _check("lora_rank", lora.get("rank") == 8, f"rank={lora.get('rank', 'NOT SET')}")
    _check("lora_alpha", lora.get("alpha") == 16, f"alpha={lora.get('alpha', 'NOT SET')}")
    _check("lora_dropout", lora.get("dropout") == 0.0, f"dropout={lora.get('dropout', 'NOT SET')}")

    # Per-method config files exist
    for method in methods:
        cfg_path = CONFIGS_DIR / f"{method}.yaml"
        _check(f"config_{method}", cfg_path.exists(), str(cfg_path))

    # P0-11: Frozen artifact SHA verification.
    base_model = common_config.get("base_model", {})
    frozen_revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    _check(
        "base_model_revision_frozen",
        base_model.get("revision") == frozen_revision,
        f"revision={base_model.get('revision', 'NOT SET')}",
    )

    data_cfg = common_config.get("data", {})
    processed_sha = data_cfg.get("processed_dataset_sha256", "")
    _check(
        "processed_dataset_sha256_nonempty",
        bool(processed_sha) and len(processed_sha) == 64,
        f"sha256_len={len(processed_sha)}",
    )

    # Also verify route_probe_sha256 is present.
    probe_sha = data_cfg.get("route_probe_sha256", "")
    _check(
        "route_probe_sha256_nonempty",
        bool(probe_sha) and len(probe_sha) == 64,
        f"sha256_len={len(probe_sha)}",
    )

    # If the processed dataset file exists, verify its SHA-256.
    dataset_path = data_cfg.get("processed_dataset_path", "")
    if dataset_path:
        from pathlib import Path as _Path
        import hashlib as _hashlib

        ds_file = _Path(dataset_path)
        if ds_file.is_file():
            h = _hashlib.sha256()
            with open(ds_file, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
            actual_sha = h.hexdigest()
            _check(
                "processed_dataset_sha256_match",
                actual_sha == processed_sha,
                f"expected={processed_sha[:16]}... actual={actual_sha[:16]}...",
            )
        else:
            _check(
                "processed_dataset_exists",
                False,
                f"not found: {dataset_path}",
            )

    all_passed = all(c["pass"] for c in checks)
    report = {
        "preflight_passed": all_passed,
        "checks": checks,
        "methods": methods,
        "timestamp": time.time(),
        "code_commit": _git_commit(),
    }

    # Write report
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_ROOT / "suite_preflight_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    logger.info(f"Preflight: {'PASSED' if all_passed else 'FAILED'}")

    return report


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLLMU-Bench baseline suite orchestrator",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Run preflight checks only (no GPU required).",
    )
    parser.add_argument(
        "--skip", action="append", default=[],
        choices=[m[0] for m in METHOD_ORDER],
        help="Skip a specific method (can be repeated).",
    )
    parser.add_argument(
        "--only", action="append", default=[],
        choices=[m[0] for m in METHOD_ORDER],
        help="Run only specific methods (can be repeated).",
    )
    parser.add_argument(
        "--reference-model-path",
        help="Path to frozen reference model for KL (overrides config).",
    )
    args = parser.parse_args()

    # Determine which methods to run
    if args.only:
        methods_to_run = [m for m in METHOD_ORDER if m[0] in args.only]
    else:
        methods_to_run = [m for m in METHOD_ORDER if m[0] not in args.skip]

    method_keys = [m[0] for m in methods_to_run]
    logger.info(f"Methods to run: {', '.join(f'{m[1]}:{m[0]}' for m in methods_to_run)}")

    # Load common config
    common_config = _load_common_config()

    # Preflight
    preflight_report = run_suite_preflight(method_keys, common_config)
    if not preflight_report["preflight_passed"]:
        for check in preflight_report["checks"]:
            if not check["pass"]:
                logger.error(f"  FAIL: {check['name']} — {check['detail']}")
        sys.exit(1)

    if args.preflight_only:
        logger.info("Preflight-only mode. Exiting.")
        return

    # Suite state (shared across methods)
    suite_state: dict[str, Any] = {
        "code_commit": _git_commit(),
    }
    if args.reference_model_path:
        suite_state["reference_model_path"] = args.reference_model_path

    # Run methods in order
    results: list[dict[str, Any]] = []
    suite_start = time.time()

    for method_key, baseline_id, description in methods_to_run:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# {baseline_id}: {description}")
        logger.info(f"{'#'*60}")

        config = _load_method_config(method_key)
        result = _run_single_method(method_key, config, suite_state)
        results.append(result)

        # Update suite state based on completed method
        if result["status"] == "success" and method_key == "npo_oracle":
            # Store oracle adapter path for NPO
            oracle_dir = Path(result["output_dir"])
            # Find the last checkpoint
            ckpts = sorted((oracle_dir / "checkpoints").glob("optimizer_step_*"))
            if ckpts:
                suite_state["oracle_adapter_path"] = str(ckpts[-1])
                logger.info(f"Oracle adapter: {suite_state['oracle_adapter_path']}")

    suite_elapsed = time.time() - suite_start

    # Suite summary (P0-2: explicit status states)
    per_method_status = {}
    for r in results:
        has_eval = bool(r.get("eval_metrics"))
        per_method_status[r["method"]] = {
            "status": r["status"],
            "has_eval_result": has_eval,
            "elapsed_seconds": r.get("elapsed_seconds", 0),
        }
        if r.get("eval_error"):
            per_method_status[r["method"]]["eval_error"] = r["eval_error"]

    # eval_complete: all comparison methods have valid eval results.
    # NPO-oracle is a training dependency, not a comparison method.
    eval_complete = all(
        r.get("status") == "success" and bool(r.get("eval_metrics"))
        for r in results
        if r["method"] in COMPARISON_METHODS
    ) if results else False

    # all_succeeded: every requested method completed its required phase
    # AND every comparison method produced a valid eval result.
    all_succeeded = all(
        r["status"] == "success" for r in results
    ) and eval_complete

    summary = {
        "suite": "mllmu_baselines",
        "methods_run": [r["method"] for r in results],
        "results": results,
        "total_elapsed_seconds": suite_elapsed,
        "code_commit": suite_state["code_commit"],
        "per_method_status": per_method_status,
        "eval_complete": eval_complete,
        "all_succeeded": all_succeeded,
        "comparison_methods": COMPARISON_METHODS,
    }

    summary_path = OUTPUT_ROOT / "suite_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    logger.info(f"\n{'='*60}")
    logger.info(f"Suite complete in {suite_elapsed:.1f}s")
    for r in results:
        status_icon = "OK" if r["status"] == "success" else "FAIL"
        has_eval = bool(r.get("eval_metrics"))
        eval_icon = "eval" if has_eval else "NO_EVAL"
        logger.info(
            f"  [{status_icon}] {r['method']} "
            f"({r.get('elapsed_seconds', 0):.1f}s) [{eval_icon}] "
            f"status={r['status']}"
        )
    logger.info(f"Summary: {summary_path}")
    logger.info(f"{'='*60}")

    if not summary["all_succeeded"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
