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
import hashlib
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


def _sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_working_tree_clean() -> bool:
    """Return True if ``git status --porcelain`` produces no output."""
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=False,
        )
        return r.returncode == 0 and r.stdout.strip() == ""
    except FileNotFoundError:
        return False


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


def _validate_eval_result(
    eval_metrics: dict[str, Any],
    expected_method: str,
    common_config: dict[str, Any] | None = None,
) -> str | None:
    """Validate a loaded eval_results.json.

    P0-19/20: Strengthened with method-ID, revision, route-probe SHA,
    and on-disk artifact existence checks.

    Returns None on success or an error-message string on failure.
    """
    # P0-19: expected_pair_count must be 500
    expected_pc = eval_metrics.get("expected_pair_count", 0)
    if expected_pc != 500:
        return f"expected_pair_count={expected_pc}, must be 500"

    # P0-19: actual_pair_count must be 500
    actual_pc = eval_metrics.get("actual_pair_count", 0)
    if actual_pc != 500:
        return f"actual_pair_count={actual_pc}, must be 500"

    # exact_pair_count must be 500
    pair_count = eval_metrics.get("exact_pair_count", 0)
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

    # results_path must exist on disk
    results_path = eval_metrics.get("results_path", "")
    if results_path and not Path(results_path).is_file():
        return f"results_path does not exist: {results_path}"

    # P0-19: manifest path must exist (derive from eval_output_dir)
    eval_output_dir = eval_metrics.get("eval_output_dir", "")
    if eval_output_dir:
        manifest_path = Path(eval_output_dir) / "manifest.json"
        if not manifest_path.is_file():
            return f"manifest.json not found: {manifest_path}"

    # manifest SHA must be non-empty
    manifest_sha = eval_metrics.get("manifest_sha256", "")
    if "manifest_sha256" in eval_metrics and not manifest_sha:
        return "manifest_sha256 is empty"

    # delta fields must be present and non-empty
    for field in ("delta_target", "delta_retain", "delta_control",
                  "delta_untargeted"):
        if not eval_metrics.get(field):
            return f"{field} is missing or empty"

    # P0-26: name_only_delta must exist
    if "name_only_delta" not in eval_metrics:
        return "name_only_delta is missing"

    # P0-26: dv_accuracy must have required keys
    dv_acc = eval_metrics.get("dv_accuracy", {})
    for key in ("global", "target", "retain", "control", "untargeted"):
        if key not in dv_acc:
            return f"dv_accuracy missing key: {key}"

    # P0-26: group counts must match 2/2/2/94 (P0-16: fail-closed).
    group_counts = eval_metrics.get("group_identity_counts", {})
    expected_counts = {"target": 2, "retain": 2, "control": 2, "untargeted": 94}
    for grp, expected in expected_counts.items():
        actual = group_counts.get(grp)
        if actual is None:
            return f"group_identity_counts[{grp}] is missing (fail-closed)"
        if actual != expected:
            return f"group_identity_counts[{grp}]={actual}, expected {expected}"
    
    # P0-18: Require all four binary families in every delta group.
    required_families = {"DV", "IPN", "WN", "VTC"}
    for field in ("delta_target", "delta_retain", "delta_control", "delta_untargeted"):
        delta_dict = eval_metrics.get(field, {})
        if not isinstance(delta_dict, dict):
            return f"{field} is not a dict"
        missing = required_families - set(delta_dict.keys())
        if missing:
            return f"{field} missing binary families: {sorted(missing)}"
    
    # P0-17: Fail-closed provenance validation.
    # For full comparison evidence, require non-empty provenance fields.
    for prov_field in ("route_probe_sha256", "selection_manifest_sha256", "model_revision"):
        prov_value = eval_metrics.get(prov_field, "")
        if not prov_value:
            return f"{prov_field} is empty (fail-closed)"
    
    # P0-20: Enforce per-family 2/2/2/94 counts.
    group_probe_counts = eval_metrics.get("group_probe_counts", {})
    if group_probe_counts:
        expected_per_family = {
            "target": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
            "retain": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
            "control": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
            "untargeted": {"DV": 94, "IPN": 94, "WN": 94, "VTC": 94, "name_only": 94},
        }
        for grp, expected_families in expected_per_family.items():
            actual_families = group_probe_counts.get(grp, {})
            for fam, expected_count in expected_families.items():
                actual_count = actual_families.get(fam)
                if actual_count is None:
                    return f"group_probe_counts[{grp}][{fam}] is missing"
                if actual_count != expected_count:
                    return (
                        f"group_probe_counts[{grp}][{fam}]={actual_count}, "
                        f"expected {expected_count}"
                    )

    # P0-20: method identifier must match expected_method
    actual_method = eval_metrics.get("method", "")
    if actual_method != expected_method:
        return (
            f"method mismatch: got '{actual_method}', "
            f"expected '{expected_method}'"
        )

    # P0-19: base model revision must match frozen revision
    if common_config is not None:
        frozen_revision = common_config.get("base_model", {}).get("revision", "")
        actual_revision = eval_metrics.get("model_revision", "")
        if frozen_revision and actual_revision and actual_revision != frozen_revision:
            return (
                f"model_revision mismatch: got '{actual_revision}', "
                f"expected '{frozen_revision}'"
            )

        # P0-19: route probe SHA must match frozen SHA
        frozen_probe_sha = common_config.get("data", {}).get("route_probe_sha256", "")
        actual_probe_sha = eval_metrics.get("route_probe_sha256", "")
        if frozen_probe_sha and actual_probe_sha and actual_probe_sha != frozen_probe_sha:
            return (
                f"route_probe_sha256 mismatch: got '{actual_probe_sha}', "
                f"expected '{frozen_probe_sha}'"
            )

        # P0-26: selection manifest SHA must match
        frozen_sel_sha = common_config.get("data", {}).get("selection_manifest_sha256", "")
        actual_sel_sha = eval_metrics.get("selection_manifest_sha256", "")
        if frozen_sel_sha and actual_sel_sha and actual_sel_sha != frozen_sel_sha:
            return (
                f"selection_manifest_sha256 mismatch: got '{actual_sel_sha}', "
                f"expected '{frozen_sel_sha}'"
            )

    # P0-19: strict_validation report must exist
    if eval_output_dir:
        strict_path = Path(eval_output_dir) / "strict_validation.json"
        if not strict_path.is_file():
            return f"strict_validation.json not found: {strict_path}"

        # P0-19: pairing report must exist
        pairing_path = Path(eval_output_dir) / "pairing_validation.json"
        if not pairing_path.is_file():
            return f"pairing_validation.json not found: {pairing_path}"

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
            # P0-12: Zero-padded rate string (05, 10).
            rate_str = f"{round(rate * 100):02d}"
            eval_path = Path(output_dir) / "eval" / f"prune_{rate_str}" / "eval_results.json"
            if not eval_path.is_file():
                # Also check directly under checkpoints
                eval_path = Path(output_dir) / "checkpoints" / f"prune_{rate_str}" / "eval_results.json"
            if eval_path.is_file():
                with open(eval_path) as f:
                    metrics = json.load(f)
                # P0-10/12: Canonical method ID is manu_prune_05/manu_prune_10.
                err = _validate_eval_result(metrics, f"manu_prune_{rate_str}", config)
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

    validation_error = _validate_eval_result(eval_metrics, method, config)
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
    *,
    expected_code_sha: str = "",
) -> dict[str, Any]:
    """Run suite-wide preflight — full frozen-contract verification (P0-21).

    Gates checked:
      * Model: ID, revision, dtype, backend
      * Data: processed dataset, route probe, research manifest,
              freeze verification, baseline results & manifest,
              identity selection — all existence + SHA verified
      * Identity: target/retain/control IDs match frozen selection
      * Code: git HEAD, clean working tree, optional exact-SHA gate
      * Output: directories writable, method isolation

    Returns a suite-level preflight report.
    """
    logger.info("Running suite-wide preflight (full frozen contract) ...")

    checks: list[dict[str, Any]] = []

    def _check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "pass": passed, "detail": detail})

    # -- Model checks (P0-21) ------------------------------------------- #
    base_model = common_config.get("base_model", {})
    _check("model_id", base_model.get("id") == "Qwen/Qwen3.5-9B",
           f"id={base_model.get('id', 'NOT SET')}")
    frozen_revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    _check("model_revision_frozen",
           base_model.get("revision") == frozen_revision,
           f"revision={base_model.get('revision', 'NOT SET')}")
    _check("model_dtype", base_model.get("dtype") == "bfloat16",
           f"dtype={base_model.get('dtype', 'NOT SET')}")
    _check("model_backend", base_model.get("backend") == "qwen_hf",
           f"backend={base_model.get('backend', 'NOT SET')}")

    # -- Training config checks ----------------------------------------- #
    training = common_config.get("training", {})
    _check("seed_set", training.get("seed") == 17,
           f"seed={training.get('seed', 'NOT SET')}")
    _check("steps_set", training.get("max_optimizer_steps") == 125,
           f"steps={training.get('max_optimizer_steps', 'NOT SET')}")
    _check("lr_set", training.get("learning_rate") == 2e-5,
           f"lr={training.get('learning_rate', 'NOT SET')}")
    _check("warmup_set", training.get("warmup_steps") == 5,
           f"warmup={training.get('warmup_steps', 'NOT SET')}")

    lora = common_config.get("lora", {})
    _check("lora_rank", lora.get("rank") == 8,
           f"rank={lora.get('rank', 'NOT SET')}")
    _check("lora_alpha", lora.get("alpha") == 16,
           f"alpha={lora.get('alpha', 'NOT SET')}")
    _check("lora_dropout", lora.get("dropout") == 0.05,
           f"dropout={lora.get('dropout', 'NOT SET')}")

    # Per-method config files exist
    for method in methods:
        cfg_path = CONFIGS_DIR / f"{method}.yaml"
        _check(f"config_{method}", cfg_path.exists(), str(cfg_path))

    data_cfg = common_config.get("data", {})

    # -- Processed dataset (P0-3, P0-21) -------------------------------- #
    processed_sha = data_cfg.get("processed_dataset_sha256", "")
    _check("processed_dataset_sha256_configured",
           bool(processed_sha) and len(processed_sha) == 64,
           f"sha256_len={len(processed_sha)}")
    dataset_path = data_cfg.get("processed_dataset_path", "")
    if dataset_path and Path(dataset_path).is_file():
        actual_sha = _sha256_file(dataset_path)
        _check("processed_dataset_sha256_match",
               actual_sha == processed_sha,
               f"expected={processed_sha[:16]}... actual={actual_sha[:16]}...")
    else:
        _check("processed_dataset_exists", False,
               f"not found: {dataset_path}")

    # -- Route probe (P0-3, P0-21) -------------------------------------- #
    probe_sha = data_cfg.get("route_probe_sha256", "")
    _check("route_probe_sha256_configured",
           bool(probe_sha) and len(probe_sha) == 64,
           f"sha256_len={len(probe_sha)}")
    probe_path = data_cfg.get("route_probe_path", "")
    if probe_path:
        _check("route_probe_exists", Path(probe_path).is_file(),
               str(probe_path))
        if Path(probe_path).is_file():
            actual_probe_sha = _sha256_file(probe_path)
            _check("route_probe_sha256_match",
                   actual_probe_sha == probe_sha,
                   f"expected={probe_sha[:16]}... actual={actual_probe_sha[:16]}...")

    # -- Baseline results & manifest (P0-4) ----------------------------- #
    bl_results_path = data_cfg.get("baseline_results_path", "")
    bl_results_sha = data_cfg.get("baseline_results_sha256", "")
    _check("baseline_results_exists",
           bool(bl_results_path) and Path(bl_results_path).is_file(),
           str(bl_results_path))
    if bl_results_path and Path(bl_results_path).is_file():
        actual_bl_sha = _sha256_file(bl_results_path)
        _check("baseline_results_sha256_match",
               actual_bl_sha == bl_results_sha,
               f"expected={bl_results_sha[:16]}... actual={actual_bl_sha[:16]}...")
        # Structural: 500 rows, 500 unique probe IDs
        bl_rows: list[dict[str, Any]] = []
        with open(bl_results_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    bl_rows.append(json.loads(line))
        _check("baseline_500_rows", len(bl_rows) == 500,
               f"rows={len(bl_rows)}")
        bl_probe_ids = {r.get("probe_id") for r in bl_rows}
        _check("baseline_500_unique_probes", len(bl_probe_ids) == 500,
               f"unique_probe_ids={len(bl_probe_ids)}")

    bl_manifest_path = data_cfg.get("baseline_manifest_path", "")
    bl_manifest_sha = data_cfg.get("baseline_manifest_sha256", "")
    _check("baseline_manifest_exists",
           bool(bl_manifest_path) and Path(bl_manifest_path).is_file(),
           str(bl_manifest_path))
    if bl_manifest_path and Path(bl_manifest_path).is_file():
        actual_bm_sha = _sha256_file(bl_manifest_path)
        _check("baseline_manifest_sha256_match",
               actual_bm_sha == bl_manifest_sha,
               f"expected={bl_manifest_sha[:16]}... actual={actual_bm_sha[:16]}...")

    # -- Research dataset manifest (P0-5) ------------------------------- #
    rdm_path = data_cfg.get("research_dataset_manifest_path", "")
    rdm_sha = data_cfg.get("research_dataset_manifest_sha256", "")
    _check("research_dataset_manifest_exists",
           bool(rdm_path) and Path(rdm_path).is_file(),
           str(rdm_path))
    if rdm_path and Path(rdm_path).is_file():
        actual_rdm_sha = _sha256_file(rdm_path)
        _check("research_dataset_manifest_sha256_match",
               actual_rdm_sha == rdm_sha,
               f"expected={rdm_sha[:16]}... actual={actual_rdm_sha[:16]}...")
        # Validate manifest using fields that actually exist (P0-1).
        with open(rdm_path) as fh:
            rdm_data = json.load(fh)
        _check("research_manifest_version",
               bool(rdm_data.get("manifest_version")),
               f"version={rdm_data.get('manifest_version', 'N/A')}")
        _check("research_manifest_purpose",
               bool(rdm_data.get("manifest_purpose")),
               "manifest_purpose present")
        _check("research_model_id",
               rdm_data.get("model_provenance", {}).get("model_id")
               == "Qwen/Qwen3.5-9B",
               f"model_id={rdm_data.get('model_provenance', {}).get('model_id', 'N/A')}")
        _check("research_model_revision",
               rdm_data.get("model_provenance", {}).get("resolved_revision")
               == "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
               f"revision={rdm_data.get('model_provenance', {}).get('resolved_revision', 'N/A')}")
        _check("research_protocol_sha",
               rdm_data.get("protocol", {}).get("protocol_sha256")
               == "b08795380a310c86bfab34d916988431472a028d0abe6aa487e42df43351e924",
               f"sha={rdm_data.get('protocol', {}).get('protocol_sha256', 'N/A')[:16]}...")
        _check("research_processed_dataset_sha",
               rdm_data.get("dataset_artifacts", {}).get("processed_dataset", {}).get("sha256")
               == "7200df4ec361ee52ad8a183b1181271980f35fb3f79690931f17481080c0d8c1",
               "processed_dataset sha match")
        _check("research_route_probes_sha",
               rdm_data.get("dataset_artifacts", {}).get("route_probes", {}).get("sha256")
               == "aeca4ee889e429ad717afb4d83c265b3990aebd5c1464b8afb4b4a2ad4dfd864",
               "route_probes sha match")
        _check("research_route_probes_count",
               rdm_data.get("dataset_artifacts", {}).get("route_probes", {}).get("total_probes")
               == 500,
               f"total_probes={rdm_data.get('dataset_artifacts', {}).get('route_probes', {}).get('total_probes', 'N/A')}")
        _check("research_ready_for_experiments",
               rdm_data.get("definition_of_done", {}).get("ready_for_experiments")
               is True,
               f"ready={rdm_data.get('definition_of_done', {}).get('ready_for_experiments')}")

    # -- Freeze verification (P0-5) ------------------------------------- #
    fv_path = data_cfg.get("freeze_verification_path", "")
    fv_sha = data_cfg.get("freeze_verification_sha256", "")
    _check("freeze_verification_exists",
           bool(fv_path) and Path(fv_path).is_file(),
           str(fv_path))
    if fv_path and Path(fv_path).is_file():
        actual_fv_sha = _sha256_file(fv_path)
        _check("freeze_verification_sha256_match",
               actual_fv_sha == fv_sha,
               f"expected={fv_sha[:16]}... actual={actual_fv_sha[:16]}...")
        with open(fv_path) as fh:
            fv_data = json.load(fh)
        _check("freeze_dataset_version",
               fv_data.get("dataset_version") == "fiubench-route-v1",
               f"version={fv_data.get('dataset_version', 'N/A')}")
        _check("freeze_ready_for_experiments",
               fv_data.get("ready_for_experiments") is True,
               f"ready={fv_data.get('ready_for_experiments')}")
        _check("freeze_strict_final_verify",
               fv_data.get("strict_final_verify_pass") is True,
               f"pass={fv_data.get('strict_final_verify_pass')}")
        _check("freeze_bundle_verifier_pass",
               fv_data.get("bundle_verifier_pass") is True,
               f"pass={fv_data.get('bundle_verifier_pass')}")
        _check("freeze_manual_audit_pass",
               fv_data.get("manual_audit_pass") is True,
               f"pass={fv_data.get('manual_audit_pass')}")
        # Cross-check route probe SHA in freeze data
        freeze_probe_sha = fv_data.get("route_probe_sha256", "")
        if freeze_probe_sha and probe_sha:
            _check("freeze_route_probe_sha_consistent",
                   freeze_probe_sha == probe_sha,
                   "freeze vs config route probe SHA")
        # Cross-check processed dataset SHA in freeze data (informational).
        fv_data.get("dataset_manifest_sha256", "")

    # -- Identity selection (P0-6) -------------------------------------- #
    sel_path = data_cfg.get("selection_manifest_path", "")
    sel_sha = data_cfg.get("selection_manifest_sha256", "")
    _check("selection_manifest_exists",
           bool(sel_path) and Path(sel_path).is_file(),
           str(sel_path))
    if sel_path and Path(sel_path).is_file():
        actual_sel_sha = _sha256_file(sel_path)
        _check("selection_manifest_sha256_match",
               actual_sel_sha == sel_sha,
               f"expected={sel_sha[:16]}... actual={actual_sel_sha[:16]}...")
        with open(sel_path) as fh:
            sel_data = json.load(fh)
        cfg_target = set(data_cfg.get("forget_identity_ids", []))
        cfg_retain = set(data_cfg.get("retain_identity_ids", []))
        cfg_control = set(data_cfg.get("control_identity_ids", []))
        frozen_target = set(sel_data.get("target_identities", []))
        frozen_retain = set(sel_data.get("retain_identities", []))
        frozen_control = set(sel_data.get("control_identities", []))
        _check("target_ids_match",
               cfg_target == frozen_target,
               f"config={sorted(cfg_target)} frozen={sorted(frozen_target)}")
        _check("retain_ids_match",
               cfg_retain == frozen_retain,
               f"config={sorted(cfg_retain)} frozen={sorted(frozen_retain)}")
        _check("control_ids_match",
               cfg_control == frozen_control,
               f"config={sorted(cfg_control)} frozen={sorted(frozen_control)}")
        # P0-23: Check duplicates BEFORE converting to sets.
        raw_ids = (
            list(data_cfg.get("forget_identity_ids", []))
            + list(data_cfg.get("retain_identity_ids", []))
            + list(data_cfg.get("control_identity_ids", []))
        )
        _check("no_duplicate_ids",
               len(raw_ids) == len(set(raw_ids)),
               f"total={len(raw_ids)} unique={len(set(raw_ids))}")
        # Pairwise disjointness.
        _check("no_group_overlap",
               not (cfg_target & cfg_retain)
               and not (cfg_target & cfg_control)
               and not (cfg_retain & cfg_control),
               "target/retain/control pairwise disjoint")

    # -- Code state (P0-21, P0-22, P0-23) -------------------------------- #
    code_commit = _git_commit()
    _check("git_head_available", bool(code_commit),
           f"HEAD={code_commit[:16] if code_commit else 'N/A'}")
    if expected_code_sha:
        _check("expected_code_sha_match",
               code_commit == expected_code_sha,
               f"expected={expected_code_sha[:16]}... HEAD={code_commit[:16] if code_commit else 'N/A'}")
    _check("git_clean_working_tree", _git_working_tree_clean(),
           "git status --porcelain must be empty")

    # -- Output directories (P0-21) ------------------------------------- #
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _check("output_root_writable", OUTPUT_ROOT.is_dir(),
           str(OUTPUT_ROOT))
    for method in methods:
        method_dir = OUTPUT_ROOT / method
        try:
            method_dir.mkdir(parents=True, exist_ok=True)
            _check(f"output_{method}_writable", method_dir.is_dir(),
                   str(method_dir))
        except OSError as exc:
            _check(f"output_{method}_writable", False, str(exc))
    
    # P0-3: Verify runtime output root is git-ignored.
    runtime_root_str = str(OUTPUT_ROOT)
    try:
        r = subprocess.run(
            ["git", "check-ignore", "-q", runtime_root_str],
            capture_output=True, text=True, check=False,
            cwd=PROJECT_ROOT,
        )
        runtime_ignored = (r.returncode == 0)
        _check("runtime_output_root_ignored", runtime_ignored,
               f"path={runtime_root_str} ignored={runtime_ignored}")
    except FileNotFoundError:
        _check("runtime_output_root_ignored", False, "git not available")

    # -- Assemble report ------------------------------------------------- #
    all_passed = all(c["pass"] for c in checks)
    report = {
        "preflight_passed": all_passed,
        "checks": checks,
        "methods": methods,
        "timestamp": time.time(),
        "code_commit": code_commit,
        "runtime_output_root": runtime_root_str,
        "runtime_output_root_ignored": runtime_ignored if 'runtime_ignored' in locals() else False,
    }

    report_path = OUTPUT_ROOT / "suite_preflight_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    logger.info(f"Preflight: {'PASSED' if all_passed else 'FAILED'}")
    if not all_passed:
        for c in checks:
            if not c["pass"]:
                logger.error(f"  FAIL: {c['name']} — {c['detail']}")

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
    # P0-23: Exact code-SHA gate
    parser.add_argument(
        "--expected-code-sha",
        help="Require git HEAD to match this exact SHA before proceeding.",
    )
    # P0-2: Runtime output root (must be git-ignored).
    parser.add_argument(
        "--runtime-output-root",
        help="Override output root for runtime artefacts (must be git-ignored).",
    )
    args = parser.parse_args()

    # Determine which methods to run
    if args.only:
        methods_to_run = [m for m in METHOD_ORDER if m[0] in args.only]
    else:
        methods_to_run = [m for m in METHOD_ORDER if m[0] not in args.skip]

    method_keys = [m[0] for m in methods_to_run]
    logger.info(f"Methods to run: {', '.join(f'{m[1]}:{m[0]}' for m in methods_to_run)}")

    # P0-2: Override OUTPUT_ROOT when --runtime-output-root is given.
    global OUTPUT_ROOT
    runtime_root = getattr(args, "runtime_output_root", "") or ""
    if runtime_root:
        OUTPUT_ROOT = Path(runtime_root)
        logger.info(f"Runtime output root: {OUTPUT_ROOT}")

    # Load common config
    common_config = _load_common_config()

    # P0-22/23: Git state gates (before preflight)
    current_sha = _git_commit()
    expected_sha = getattr(args, "expected_code_sha", "") or ""
    if expected_sha and current_sha != expected_sha:
        logger.error(
            f"Expected code SHA {expected_sha} but HEAD is {current_sha}"
        )
        sys.exit(1)
    if not _git_working_tree_clean():
        logger.error(
            "Git working tree is dirty — commit or stash changes "
            "before running the suite."
        )
        sys.exit(1)

    # Preflight
    preflight_report = run_suite_preflight(
        method_keys, common_config,
        expected_code_sha=expected_sha,
    )
    if not preflight_report["preflight_passed"]:
        for check in preflight_report["checks"]:
            if not check["pass"]:
                logger.error(f"  FAIL: {check['name']} — {check['detail']}")
        sys.exit(1)

    if args.preflight_only:
        logger.info("Preflight-only mode. Exiting.")
        return

    # Suite state (shared across methods)
    # P0-3: Record git provenance invariant at experiment start.
    suite_state: dict[str, Any] = {
        "code_commit": _git_commit(),
        "code_sha": _git_commit(),
        "git_dirty_at_start": not _git_working_tree_clean(),
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
        
        # P0-1: CLI --runtime-output-root overrides YAML runtime.output_dir.
        if runtime_root:
            config.setdefault("runtime", {})
            config["runtime"]["output_dir"] = str(OUTPUT_ROOT / method_key)
        
        result = _run_single_method(method_key, config, suite_state)
        results.append(result)

        # Update suite state based on completed method
        if result["status"] == "success" and method_key == "npo_oracle":
            # Store oracle adapter path for NPO
            oracle_dir = Path(result["output_dir"])
            # P0-29: Prefer the canonical adapter_final directory.
            adapter_final = oracle_dir / "checkpoints" / "adapter_final"
            if adapter_final.is_dir():
                suite_state["oracle_adapter_path"] = str(adapter_final)
            else:
                # Fall back to the last optimizer-step checkpoint.
                ckpts = sorted((oracle_dir / "checkpoints").glob("optimizer_step_*"))
                if ckpts:
                    suite_state["oracle_adapter_path"] = str(ckpts[-1])
            if suite_state.get("oracle_adapter_path"):
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

    # P0-3: Post-run git cleanliness check.
    git_clean_at_end = _git_working_tree_clean()

    summary = {
        "suite": "mllmu_baselines",
        "methods_run": [r["method"] for r in results],
        "results": results,
        "total_elapsed_seconds": suite_elapsed,
        # P0-3: Git provenance invariant.
        "code_sha": suite_state["code_sha"],
        "code_commit": suite_state["code_commit"],
        "git_dirty_at_start": suite_state["git_dirty_at_start"],
        "git_clean_at_end": git_clean_at_end,
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
