"""Pilot runner — orchestrates the full Stage 3 unlearning experiment.

This module wires together all pipeline stages:

1. Load experiment config
2. Run identity selection
3. Run LoRA unlearning training
4. Run post-unlearning evaluation
5. Run paired analysis
6. Generate pilot validation report

Public API
----------
.. autoclass:: PilotRunner
.. autofunction:: load_experiment_config
.. autofunction:: generate_pilot_validation_report
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .paired_analysis import PairedAnalysis, PairedAnalysisConfig
from .pilot_selection import (
    build_identity_stats,
    generate_intervention_manifest,
    run_leakage_detection,
    select_pilot_identities,
    validate_pilot_frozen_inputs,
    write_selection_manifest,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load the experiment config YAML."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def validate_experiment_config(cfg: dict[str, Any]) -> None:
    """Validate the experiment config (P0-7).

    Rejects legacy field names and missing required paths.

    Raises
    ------
    ValueError
        If any required field is missing or a legacy field is detected.
    """
    errors: list[str] = []

    # -- Method name ---------------------------------------------------- #
    method_name = cfg.get("method", {}).get("name", "")
    if method_name == "lora_targeted_update":
        errors.append(
            "method.name 'lora_targeted_update' is legacy; "
            "use 'lora_targeted_candidate_margin'"
        )
    elif method_name != "lora_targeted_candidate_margin":
        errors.append(
            f"method.name must be 'lora_targeted_candidate_margin', "
            f"got '{method_name}'"
        )

    # -- num_steps vs num_optimizer_steps -------------------------------- #
    hp = cfg.get("method", {}).get("hyperparameters", {})
    if "num_steps" in hp and "num_optimizer_steps" not in hp:
        errors.append(
            "hyperparameters.num_steps is legacy; "
            "use 'num_optimizer_steps'"
        )

    # -- Required paths -------------------------------------------------- #
    dataset_cfg = cfg.get("dataset", {})
    if not dataset_cfg.get("research_manifest_path"):
        errors.append("dataset.research_manifest_path is required")
    if not dataset_cfg.get("freeze_verification_path"):
        errors.append("dataset.freeze_verification_path is required")

    base_model_cfg = cfg.get("base_model", {})
    if not base_model_cfg.get("model_config_path"):
        errors.append("base_model.model_config_path is required")
    if not base_model_cfg.get("revision"):
        errors.append("base_model.revision is required")

    if errors:
        raise ValueError(
            "Experiment config validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def _git_commit() -> str:
    """Return the current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except FileNotFoundError:
        return ""


def _sha256_file(path: str | Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_dirty() -> bool:
    """Return True if the git working tree is dirty."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=False,
        )
        return bool(result.stdout.strip()) if result.returncode == 0 else False
    except FileNotFoundError:
        return False


# --------------------------------------------------------------------------- #
# Shared validation extraction (Fix 1/2/3)
# --------------------------------------------------------------------------- #

def _accuracy_gate(value: float | None, threshold: float) -> bool | None:
    """Return True if *value* >= *threshold*, None if *value* is None."""
    if value is None:
        return None
    return value >= threshold


def _build_validation_gates(
    analysis_results: dict[str, Any],
    post_eval_summary: dict[str, Any] | None,
    *,
    dv_gate: float = 0.98,
    tolerance: float = 0.1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract analysis_summary and GO/NO-GO gates from *analysis_results*.

    Returns ``(analysis_summary, gates)``.

    Both ``PilotRunner.generate_validation_report`` and the standalone
    ``generate_pilot_validation_report`` delegate here so that the two
    code paths always produce identical metric routing and gate
    semantics for identical inputs.
    """
    ge = analysis_results.get("group_effects", {})
    pr = analysis_results.get("preservation_report", {})
    pv = analysis_results.get("pairing_validation", {})

    # -- P0-9: Use overall_visual (visual-only signed-margin). --
    target_vis = ge.get("target", {}).get("overall_visual", {})
    retain_vis = ge.get("retain", {}).get("overall_visual", {})
    control_vis = ge.get("control", {}).get("overall_visual", {})
    untargeted_vis = ge.get("untargeted", {}).get("overall_visual", {})

    target_mean = target_vis.get("mean")
    retain_mean = retain_vis.get("mean")
    control_mean = control_vis.get("mean")
    untargeted_mean = untargeted_vis.get("mean")

    # -- Fix 1: group-specific direct_visual fields. --
    global_dv = pr.get("global_direct_visual", {})
    target_dv = pr.get("target_direct_visual", {})
    retain_dv = pr.get("retain_direct_visual", {})
    control_dv = pr.get("control_direct_visual", {})
    untargeted_dv = pr.get("untargeted_direct_visual", {})

    global_post_acc = global_dv.get("post_accuracy")

    target_dv_pre = target_dv.get("pre_accuracy")
    target_dv_post = target_dv.get("post_accuracy")
    target_margin_pre = target_dv.get("pre_mean_margin")
    target_margin_post = target_dv.get("post_mean_margin")

    retain_dv_post = retain_dv.get("post_accuracy")
    control_dv_post = control_dv.get("post_accuracy")
    untargeted_dv_post = untargeted_dv.get("post_accuracy")

    # -- Post-eval inference errors. --
    post_eval_errors = 0
    if post_eval_summary is not None:
        post_eval_errors = post_eval_summary.get("inference_errors", 0)

    # -- Fix 3: Pairing validation (mode-neutral name). --
    pairing_pass = pv.get("pass", False)
    pairing_expected_n = pv.get("expected_n")
    pairing_baseline_rows = pv.get("baseline_rows")
    pairing_post_rows = pv.get("post_rows")

    # -- Assemble analysis_summary. --
    analysis_summary: dict[str, Any] = {
        "target_visual_delta_mean": target_mean,
        "retain_visual_delta_mean": retain_mean,
        "control_visual_delta_mean": control_mean,
        "untargeted_visual_delta_mean": untargeted_mean,
        "global_direct_visual_post_accuracy": global_post_acc,
        "target_direct_visual_pre_accuracy": target_dv_pre,
        "target_direct_visual_post_accuracy": target_dv_post,
        "target_direct_visual_pre_margin": target_margin_pre,
        "target_direct_visual_post_margin": target_margin_post,
        "retain_direct_visual_post_accuracy": retain_dv_post,
        "control_direct_visual_post_accuracy": control_dv_post,
        "untargeted_direct_visual_post_accuracy": untargeted_dv_post,
        "post_eval_inference_errors": post_eval_errors,
        "pairing_exact_match": pairing_pass,
        "pairing_expected_n": pairing_expected_n,
        "pairing_baseline_rows": pairing_baseline_rows,
        "pairing_post_rows": pairing_post_rows,
    }

    # -- GO/NO-GO gates. --
    target_mag = abs(target_mean) if target_mean is not None else 0.0
    retain_drift = abs(retain_mean) if retain_mean is not None else 0.0
    control_drift = abs(control_mean) if control_mean is not None else 0.0

    gates: dict[str, Any] = {
        # Primary selectivity gates.
        "target_exceeds_retain_plus_tolerance": (
            target_mag > retain_drift + tolerance
            if target_mean is not None and retain_mean is not None
            else None
        ),
        "target_exceeds_control_plus_tolerance": (
            target_mag > control_drift + tolerance
            if target_mean is not None and control_mean is not None
            else None
        ),
        # Fix 1: group-specific direct-visual preservation gates.
        "retain_direct_visual_accuracy_gate": _accuracy_gate(
            retain_dv_post, dv_gate,
        ),
        "control_direct_visual_accuracy_gate": _accuracy_gate(
            control_dv_post, dv_gate,
        ),
        "untargeted_direct_visual_accuracy_gate": _accuracy_gate(
            untargeted_dv_post, dv_gate,
        ),
        "global_direct_visual_accuracy_gate": _accuracy_gate(
            global_post_acc, dv_gate,
        ),
        # Quality gates.
        "zero_post_eval_inference_errors": post_eval_errors == 0,
        # Fix 3: mode-neutral pairing gate name.
        "exact_pairing": pairing_pass,
        # Configuration.
        "_gate_tolerance": tolerance,
        "_direct_visual_accuracy_gate": dv_gate,
    }

    return analysis_summary, gates


# --------------------------------------------------------------------------- #
# Pilot runner
# --------------------------------------------------------------------------- #

@dataclass
class PilotRunner:
    """Orchestrates the full Stage 3 unlearning pilot.

    This class coordinates all pipeline stages. The actual model loading,
    training, and evaluation require GPU and are delegated to the
    respective modules.
    """

    config_path: str | Path
    base_dir: str | Path = "."

    def __init__(
        self,
        config_path: str | Path,
        base_dir: str | Path = ".",
    ) -> None:
        self.config_path = Path(config_path)
        self.base_dir = Path(base_dir)
        self._config: dict[str, Any] | None = None
        self._output_dir: Path | None = None

    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            self._config = load_experiment_config(self.config_path)
        return self._config

    @property
    def output_dir(self) -> Path:
        if self._output_dir is None:
            out = self.config.get("runtime", {}).get("output_dir", "")
            self._output_dir = self.base_dir / out if out else self.base_dir / "pilot_v1"
        return self._output_dir

    def _resolve(self, rel_path: str) -> Path:
        """Resolve a path relative to base_dir."""
        return self.base_dir / rel_path

    # -- Stage 1: Selection ----------------------------------------------- #

    def run_selection(self) -> dict[str, Any]:
        """Run identity selection with preflight, leakage detection, and manifest.

        Returns the selection manifest as a dict.
        """
        cfg = self.config
        sel_dir = self.output_dir / "selection"
        evidence_dir = self.output_dir / "evidence"
        sel_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir.mkdir(parents=True, exist_ok=True)

        baseline_manifest = self._resolve(cfg["baseline"]["manifest_path"])
        baseline_results = self._resolve(cfg["baseline"]["results_path"])
        route_probe = self._resolve(cfg["dataset"]["route_probe_path"])
        processed_ds_str = cfg["dataset"].get("processed_dataset_path", "")
        processed_ds = self._resolve(processed_ds_str) if processed_ds_str else None

        # -- P0-5: Frozen SHA preflight --------------------------------- #
        actual_manifest_sha = _sha256_file(baseline_manifest)
        actual_results_sha = _sha256_file(baseline_results)
        actual_route_sha = _sha256_file(route_probe)
        actual_processed_sha = (
            _sha256_file(processed_ds) if processed_ds is not None else ""
        )

        logger.info("Running frozen SHA preflight...")
        validate_pilot_frozen_inputs(
            baseline_manifest_path=baseline_manifest,
            baseline_results_path=baseline_results,
            route_probe_path=route_probe,
            processed_dataset_path=processed_ds,
            expected_manifest_sha=cfg["baseline"].get("manifest_sha256", ""),
            expected_results_sha=cfg["baseline"].get("results_sha256", ""),
            expected_route_probe_sha=cfg["dataset"].get("route_probe_sha256", ""),
            expected_processed_ds_sha=cfg["dataset"].get("processed_dataset_sha256", ""),
            output_path=evidence_dir / "pilot_preflight_report.json",
        )

        # -- Build stats and select identities -------------------------- #
        logger.info("Building identity stats from baseline...")
        stats = build_identity_stats(baseline_results, route_probe, processed_ds)

        sel_cfg = cfg["selection"]
        logger.info("Selecting pilot identities...")
        selection = select_pilot_identities(
            stats,
            target_count=sel_cfg["target_identity_count"],
            retain_count=sel_cfg["retain_identity_count"],
            control_count=sel_cfg["control_identity_count"],
            seed=sel_cfg["seed"],
            preferred_role=sel_cfg.get("preferred_role", "train"),
        )

        # Write manifest with actual computed SHAs
        manifest_path = sel_dir / "pilot_identity_selection.json"
        write_selection_manifest(
            selection,
            manifest_path,
            baseline_manifest_sha256=actual_manifest_sha,
            baseline_results_sha256=actual_results_sha,
            route_probe_sha256=actual_route_sha,
            processed_dataset_sha256=actual_processed_sha,
            code_commit=_git_commit(),
        )

        logger.info(
            "Selection complete: target=%s, retain=%s, control=%s",
            selection.target_identities,
            selection.retain_identities,
            selection.control_identities,
        )

        # -- P0-3: Leakage detection ------------------------------------ #
        if processed_ds is not None:
            logger.info("Running leakage detection...")
            run_leakage_detection(
                processed_dataset_path=processed_ds,
                route_probe_path=route_probe,
                target_identity_ids=selection.target_identities,
                retain_identity_ids=selection.retain_identities,
                output_path=sel_dir / "leakage_report.json",
            )
            leakage_sha = _sha256_file(sel_dir / "leakage_report.json")
        else:
            leakage_sha = ""

        # -- P0-4: Intervention dataset manifest ------------------------ #
        if processed_ds is not None:
            sel_manifest_sha = _sha256_file(manifest_path)
            logger.info("Generating intervention dataset manifest...")
            generate_intervention_manifest(
                processed_dataset_path=processed_ds,
                target_identity_ids=selection.target_identities,
                retain_identity_ids=selection.retain_identities,
                selection_manifest_sha256=sel_manifest_sha,
                leakage_report_sha256=leakage_sha,
                experiment_config=cfg,
                output_path=sel_dir / "intervention_dataset_manifest.json",
                seed=sel_cfg.get("seed", 17),
                code_commit=_git_commit(),
            )

        return json.loads(manifest_path.read_text())

    # -- Stage 2: Training (requires GPU) --------------------------------- #

    def get_training_config(self, selection_manifest: dict[str, Any]) -> dict[str, Any]:
        """Build the training configuration from the experiment config.

        Returns a dict suitable for constructing ``UnlearningConfig``.
        """
        cfg = self.config
        method_cfg = cfg["method"]["hyperparameters"]

        return {
            "model_id": cfg["base_model"]["model_id"],
            "model_revision": cfg["base_model"]["revision"],
            "dtype": cfg["base_model"]["dtype"],
            "seed": cfg["runtime"]["seed"],
            "lora_rank": method_cfg["lora_rank"],
            "lora_alpha": method_cfg["lora_alpha"],
            "learning_rate": method_cfg["learning_rate"],
            "num_optimizer_steps": method_cfg["num_optimizer_steps"],
            "retain_weight": method_cfg["retain_weight"],
            "batch_size": method_cfg["train_batch_size"],
            "gradient_accumulation_steps": method_cfg["gradient_accumulation_steps"],
            "forget_identity_ids": selection_manifest["target_identities"],
            "retain_identity_ids": selection_manifest["retain_identities"],
            "processed_dataset_path": str(
                self._resolve(cfg["dataset"]["processed_dataset_path"])
            ) if cfg["dataset"].get("processed_dataset_path", "") else "",
            "route_probe_path": str(
                self._resolve(cfg["dataset"]["route_probe_path"])
            ),
            "output_dir": str(self.output_dir),
            "selection_manifest_sha256": _sha256_file(
                self.output_dir / "selection" / "pilot_identity_selection.json"
            ),
            "code_commit": _git_commit(),
        }

    # -- Stage 3: Post-evaluation (requires GPU) -------------------------- #

    def get_post_eval_config(
        self,
        checkpoint_path: str,
        checkpoint_name: str = "final",
    ) -> dict[str, Any]:
        """Build the post-evaluation configuration.

        Returns a dict suitable for constructing ``PostEvalConfig``.
        """
        cfg = self.config
        # P0-7: Resolve all frozen provenance paths.
        processed_ds_str = cfg["dataset"].get("processed_dataset_path", "")
        model_cfg_str = cfg["base_model"].get("model_config_path", "")
        freeze_verif_str = cfg["dataset"].get("freeze_verification_path", "")

        return {
            "model_id": cfg["base_model"]["model_id"],
            "model_revision": cfg["base_model"]["revision"],
            "dtype": cfg["base_model"]["dtype"],
            "seed": cfg["runtime"]["seed"],
            "checkpoint_path": checkpoint_path,
            "checkpoint_name": checkpoint_name,
            "probe_path": str(self._resolve(cfg["dataset"]["route_probe_path"])),
            "baseline_results_path": str(
                self._resolve(cfg["baseline"]["results_path"])
            ),
            "baseline_manifest_path": str(
                self._resolve(cfg["baseline"]["manifest_path"])
            ),
            # P0-11: output_dir is the base; PostUnlearningEvaluator
            # appends checkpoint_name for checkpoint-specific subdirs.
            "output_dir": str(self.output_dir / "post_eval"),
            "selection_manifest_sha256": _sha256_file(
                self.output_dir / "selection" / "pilot_identity_selection.json"
            ),
            "code_commit": _git_commit(),
            # P0-6: Use research_manifest_path from dataset section,
            # NOT baseline.manifest_path.
            "dataset_manifest_path": str(
                self._resolve(cfg["dataset"]["research_manifest_path"])
            ),
            "freeze_verification_path": str(
                self._resolve(freeze_verif_str)
            ) if freeze_verif_str else "",
            "processed_dataset_path": str(
                self._resolve(processed_ds_str)
            ) if processed_ds_str else "",
            "model_config_path": str(
                self._resolve(model_cfg_str)
            ) if model_cfg_str else "",
        }

    # -- Stage 4: Paired analysis ----------------------------------------- #

    def run_paired_analysis(
        self,
        post_results_path: str | Path,
        smoke_probe_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """Run paired analysis after post-evaluation.

        Parameters
        ----------
        post_results_path:
            Path to the post-eval ``post_results.jsonl``.
        smoke_probe_ids:
            If provided, filter baseline to this subset and validate
            N↔N pairing instead of 500↔500 (P0-6).
        """
        cfg = self.config
        analysis_dir = self.output_dir / "analysis"

        pa_config = PairedAnalysisConfig(
            baseline_results_path=str(
                self._resolve(cfg["baseline"]["results_path"])
            ),
            post_results_path=str(post_results_path),
            selection_manifest_path=str(
                self.output_dir / "selection" / "pilot_identity_selection.json"
            ),
            output_dir=str(analysis_dir),
            code_commit=_git_commit(),
            smoke_probe_ids=smoke_probe_ids,
        )

        pa = PairedAnalysis(pa_config)
        pa.load_data()
        results = pa.run_all()
        pa.write_artifacts(results)

        logger.info("Paired analysis complete. Artifacts in %s", analysis_dir)
        return results

    # -- Stage 5: Validation report --------------------------------------- #

    def generate_validation_report(
        self,
        training_summary: dict[str, Any] | None = None,
        post_eval_summary: dict[str, Any] | None = None,
        analysis_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate the pilot validation report (Section 53).

        P0-14: Gates use visual-only signed-margin metrics and enforce
        target effect magnitude > retain/control drift + tolerance.
        Also requires 0 post-eval inference errors and exact probe
        pairing (mode-neutral: N↔N where N depends on smoke/full).
        """
        evidence_dir = self.output_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)

        report: dict[str, Any] = {
            "experiment_id": self.config.get("experiment_id", ""),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_commit": _git_commit(),
            "stages": {
                "selection_completed": (
                    self.output_dir / "selection" / "pilot_identity_selection.json"
                ).exists(),
                "training_completed": training_summary is not None,
                "post_eval_completed": post_eval_summary is not None,
                "analysis_completed": analysis_results is not None,
            },
            "training_summary": training_summary,
            "post_eval_summary": post_eval_summary,
            "analysis_summary": {},
        }

        if analysis_results is not None:
            dv_gate = self.config.get("evaluation", {}).get(
                "direct_visual_accuracy_gate", 0.98,
            )
            tolerance = self.config.get("evaluation", {}).get(
                "gate_tolerance", 0.1,
            )
            analysis_summary, gates = _build_validation_gates(
                analysis_results,
                post_eval_summary,
                dv_gate=dv_gate,
                tolerance=tolerance,
            )
            report["analysis_summary"] = analysis_summary
            report["gates"] = gates

        report_path = evidence_dir / "pilot_validation_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        logger.info("Validation report written to %s", report_path)
        return report


# --------------------------------------------------------------------------- #
# Standalone validation report generator (for use without full pipeline)
# --------------------------------------------------------------------------- #

def generate_pilot_validation_report(
    output_dir: str | Path,
    *,
    experiment_id: str = "",
    training_summary: dict[str, Any] | None = None,
    post_eval_summary: dict[str, Any] | None = None,
    analysis_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a standalone validation report.

    This is a convenience function for generating the report without
    instantiating the full ``PilotRunner``.

    P0-14: Uses visual-only metrics and strengthened gates.
    """
    evidence_dir = Path(output_dir) / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": _git_commit(),
        "stages": {
            "selection_completed": (
                Path(output_dir) / "selection" / "pilot_identity_selection.json"
            ).exists(),
            "training_completed": training_summary is not None,
            "post_eval_completed": post_eval_summary is not None,
            "analysis_completed": analysis_results is not None,
        },
        "training_summary": training_summary,
        "post_eval_summary": post_eval_summary,
    }

    if analysis_results is not None:
        analysis_summary, gates = _build_validation_gates(
            analysis_results,
            post_eval_summary,
        )
        report["analysis_summary"] = analysis_summary
        report["gates"] = gates

    report_path = evidence_dir / "pilot_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
