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
    select_pilot_identities,
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
        """Run identity selection and write the manifest.

        Returns the selection manifest as a dict.
        """
        cfg = self.config
        sel_dir = self.output_dir / "selection"
        sel_dir.mkdir(parents=True, exist_ok=True)

        baseline_results = self._resolve(cfg["baseline"]["results_path"])
        route_probe = self._resolve(cfg["dataset"]["route_probe_path"])
        processed_ds_str = cfg["dataset"].get("processed_dataset_path", "")
        processed_ds = self._resolve(processed_ds_str) if processed_ds_str else None

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

        manifest_path = sel_dir / "pilot_identity_selection.json"
        write_selection_manifest(
            selection,
            manifest_path,
            baseline_manifest_sha256=cfg["baseline"].get("manifest_sha256", ""),
            baseline_results_sha256=cfg["baseline"].get("results_sha256", ""),
            route_probe_sha256=cfg["dataset"].get("route_probe_sha256", ""),
            processed_dataset_sha256=cfg["dataset"].get("processed_dataset_sha256", ""),
            code_commit=_git_commit(),
        )

        logger.info(
            "Selection complete: target=%s, retain=%s, control=%s",
            selection.target_identities,
            selection.retain_identities,
            selection.control_identities,
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
            "num_steps": method_cfg["num_steps"],
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
            "output_dir": str(self.output_dir / "post_eval"),
            "selection_manifest_sha256": _sha256_file(
                self.output_dir / "selection" / "pilot_identity_selection.json"
            ),
            "code_commit": _git_commit(),
        }

    # -- Stage 4: Paired analysis ----------------------------------------- #

    def run_paired_analysis(
        self,
        post_results_path: str | Path,
    ) -> dict[str, Any]:
        """Run paired analysis after post-evaluation.

        Parameters
        ----------
        post_results_path:
            Path to the post-eval ``post_results.jsonl``.
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

        This report summarizes whether all pipeline stages completed
        successfully and provides the GO/NO-GO decision inputs.
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
            ge = analysis_results.get("group_effects", {})
            pr = analysis_results.get("preservation_report", {})
            target_mean = ge.get("target", {}).get("overall", {}).get("mean")
            retain_mean = ge.get("retain", {}).get("overall", {}).get("mean")
            control_mean = ge.get("control", {}).get("overall", {}).get("mean")

            dv_post = pr.get("global_direct_visual", {}).get("post_accuracy")

            report["analysis_summary"] = {
                "target_mean_delta": target_mean,
                "retain_mean_delta": retain_mean,
                "control_mean_delta": control_mean,
                "post_direct_visual_accuracy": dv_post,
            }

            # GO/NO-GO gates
            dv_gate = self.config.get("evaluation", {}).get(
                "direct_visual_accuracy_gate", 0.98
            )
            report["gates"] = {
                "direct_visual_accuracy_gate": dv_gate,
                "direct_visual_pass": (
                    dv_post is not None and dv_post >= dv_gate
                    if dv_post is not None
                    else None
                ),
                "target_effect_visible": (
                    target_mean is not None and target_mean < 0
                ),
                "retain_preserved": (
                    retain_mean is not None and abs(retain_mean) < abs(target_mean or 0)
                    if target_mean is not None and retain_mean is not None
                    else None
                ),
            }

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
        ge = analysis_results.get("group_effects", {})
        pr = analysis_results.get("preservation_report", {})
        report["analysis_summary"] = {
            "target_mean_delta": ge.get("target", {}).get("overall", {}).get("mean"),
            "retain_mean_delta": ge.get("retain", {}).get("overall", {}).get("mean"),
            "control_mean_delta": ge.get("control", {}).get("overall", {}).get("mean"),
            "post_direct_visual_accuracy": pr.get(
                "global_direct_visual", {}
            ).get("post_accuracy"),
        }

    report_path = evidence_dir / "pilot_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
