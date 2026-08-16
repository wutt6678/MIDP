"""Post-unlearning evaluator for the Stage 3 pilot.

This module reuses the frozen 500-probe baseline infrastructure to
evaluate the model after LoRA-based unlearning intervention. It ensures
exact probe-ID matching with the pre-unlearning baseline and generates
a post-evaluation manifest with full provenance.

Public API
----------
.. autoclass:: PostEvalConfig
.. autoclass:: PostUnlearningEvaluator
.. autofunction:: load_lora_checkpoint
.. autofunction:: validate_exact_probe_matching
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baseline_runner import (
    BaselineResult,
    BaselineRunner,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class PostEvalConfig:
    """Configuration for post-unlearning evaluation."""

    # Base model (must match pre-unlearning)
    model_id: str = "Qwen/Qwen3.5-9B"
    model_revision: str = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    seed: int = 17

    # Checkpoint to evaluate
    checkpoint_path: str = ""
    checkpoint_name: str = ""  # e.g., "step_050"

    # Probe file (must be the exact frozen 500 probes)
    probe_path: str = ""

    # Baseline reference for comparison
    baseline_results_path: str = ""
    baseline_manifest_path: str = ""

    # Output
    output_dir: str = ""

    # Provenance
    selection_manifest_sha256: str = ""
    unlearning_run_manifest_sha256: str = ""
    code_commit: str = ""

    @property
    def experiment_id(self) -> str:
        return "fiubench_unlearning_pilot_v1"


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #

def load_lora_checkpoint(
    base_model_id: str,
    base_revision: str,
    checkpoint_path: str | Path,
    dtype: str = "bfloat16",
    device: str = "cuda:0",
) -> tuple[Any, Any]:
    """Load a LoRA adapter checkpoint on top of the frozen base model.

    Parameters
    ----------
    base_model_id:
        HuggingFace model ID for the base model.
    base_revision:
        Exact revision of the base model.
    checkpoint_path:
        Path to the LoRA adapter checkpoint directory.
    dtype:
        Torch dtype string.
    device:
        Device to load the model on.

    Returns
    -------
    model : PeftModel
        The model with LoRA adapters loaded.
    processor : AutoProcessor
        The processor for tokenization and image processing.
    """
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_dtype = getattr(torch, dtype)
    snapshot_download(base_model_id, revision=base_revision)

    # Load base model
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        revision=base_revision,
        device_map=device,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(
        base_model_id,
        revision=base_revision,
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        torch_dtype=torch_dtype,
    )
    model.eval()

    logger.info(
        f"Loaded LoRA checkpoint from {checkpoint_path} "
        f"on base {base_model_id} revision {base_revision}"
    )
    return model, processor


# --------------------------------------------------------------------------- #
# Probe validation
# --------------------------------------------------------------------------- #

def validate_exact_probe_matching(
    baseline_results: list[dict[str, Any]],
    post_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate that post-eval probe IDs exactly match baseline.

    Parameters
    ----------
    baseline_results:
        List of baseline result dicts (must have 'probe_id' field).
    post_results:
        List of post-eval result dicts (must have 'probe_id' field).

    Returns
    -------
    validation : dict
        Report with pass/fail status and details.
    """
    baseline_ids = {r["probe_id"] for r in baseline_results}
    post_ids = {r["probe_id"] for r in post_results}

    missing = baseline_ids - post_ids
    extra = post_ids - baseline_ids
    duplicates_baseline = len(baseline_results) - len(baseline_ids)
    duplicates_post = len(post_results) - len(post_ids)

    passed = (
        len(missing) == 0
        and len(extra) == 0
        and duplicates_baseline == 0
        and duplicates_post == 0
    )

    return {
        "passed": passed,
        "baseline_probe_count": len(baseline_ids),
        "post_probe_count": len(post_ids),
        "missing_probes": sorted(missing),
        "extra_probes": sorted(extra),
        "duplicate_baseline": duplicates_baseline,
        "duplicate_post": duplicates_post,
        "exact_match": passed,
    }


# --------------------------------------------------------------------------- #
# Post-unlearning evaluator
# --------------------------------------------------------------------------- #

class PostUnlearningEvaluator:
    """Evaluate the model after unlearning intervention.

    This evaluator reuses the exact frozen 500 probes and the baseline
    evaluation infrastructure to ensure comparable measurements.

    Parameters
    ----------
    config:
        Post-evaluation configuration.
    backend:
        VisionLanguageModel backend (with LoRA adapters loaded).
    """

    def __init__(
        self,
        config: PostEvalConfig,
        backend: Any,
        model_config: Any,
    ):
        self.config = config
        self.backend = backend
        self.model_config = model_config

        # Create the baseline runner for post-eval
        self._runner = BaselineRunner(
            backend=backend,
            probe_path=config.probe_path,
            output_dir=config.output_dir,
            model_config=model_config,
            resume=True,
            dataset_manifest_path=None,
            model_config_path=None,
        )

        self._results: list[BaselineResult] = []

    def run_evaluation(self, limit: int | None = None) -> list[BaselineResult]:
        """Run post-unlearning evaluation on the frozen 500 probes.

        Parameters
        ----------
        limit:
            Optional limit on number of probes (for smoke testing).

        Returns
        -------
        results : list[BaselineResult]
            The evaluation results.
        """
        logger.info(
            f"Starting post-unlearning evaluation: "
            f"checkpoint={self.config.checkpoint_name}"
        )
        start_time = time.perf_counter()

        self._results = self._runner.run_all(limit=limit)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Post-eval complete: {len(self._results)} results in {elapsed:.1f}s"
        )
        return self._results

    def save_results(self) -> Path:
        """Write post-eval results to ``results.jsonl``."""
        from dataclasses import asdict

        self.config.output_dir = Path(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.output_dir / "results.jsonl"

        rows = [asdict(r) for r in self._results]
        with open(path, "w") as f:
            f.writelines(json.dumps(row, default=str) + "\n" for row in rows)

        logger.info(f"Wrote {len(rows)} post-eval results to {path}")
        return path

    def generate_summary(self) -> dict[str, Any]:
        """Generate post-eval summary with per-family metrics."""
        return self._runner.generate_summary()

    def validate_against_baseline(self) -> dict[str, Any]:
        """Validate post-eval probe IDs match the frozen baseline.

        Returns
        -------
        validation : dict
            Report from :func:`validate_exact_probe_matching`.
        """
        # Load baseline results
        baseline_path = Path(self.config.baseline_results_path)
        baseline_results = []
        with open(baseline_path) as f:
            for line in f:
                baseline_results.append(json.loads(line))

        # Extract probe IDs from post results (works with dataclass or mock)
        post_results = [{"probe_id": r.probe_id} for r in self._results]

        return validate_exact_probe_matching(baseline_results, post_results)

    def generate_post_eval_manifest(self) -> dict[str, Any]:
        """Generate the post-evaluation manifest with full provenance.

        Returns
        -------
        manifest : dict
            The post-eval manifest with SHA-256 bindings.
        """
        output_dir = Path(self.config.output_dir)

        # Compute result file SHA
        results_path = output_dir / "results.jsonl"
        results_sha = ""
        if results_path.exists():
            results_sha = _sha256_file(results_path)

        # Compute summary
        summary = self.generate_summary()

        # Validate probe matching
        validation = self.validate_against_baseline()

        # Get git state
        git_commit = self.config.code_commit
        git_dirty = False
        try:
            import subprocess
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
                cwd=Path(__file__).parent.parent.parent.parent,
            )
            git_dirty = bool(result.stdout.strip())
            if not git_commit:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=Path(__file__).parent.parent.parent.parent,
                )
                git_commit = result.stdout.strip()
        except Exception:
            pass

        manifest = {
            "experiment_id": self.config.experiment_id,
            "evaluation_type": "post_unlearning",
            "checkpoint": {
                "name": self.config.checkpoint_name,
                "path": self.config.checkpoint_path,
            },
            "base_model": {
                "model_id": self.config.model_id,
                "revision": self.config.model_revision,
            },
            "probe_file": {
                "path": self.config.probe_path,
                "probe_count": len(self._results),
            },
            "baseline_reference": {
                "results_path": self.config.baseline_results_path,
                "manifest_path": self.config.baseline_manifest_path,
            },
            "provenance": {
                "selection_manifest_sha256": self.config.selection_manifest_sha256,
                "unlearning_run_manifest_sha256": self.config.unlearning_run_manifest_sha256,
                "results_sha256": results_sha,
            },
            "validation": validation,
            "summary": summary,
            "code_provenance": {
                "git_commit": git_commit,
                "git_dirty": git_dirty,
            },
            "seed": self.config.seed,
        }

        # Write manifest
        manifest_path = output_dir / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
            f.write("\n")

        logger.info(f"Wrote post-eval manifest to {manifest_path}")
        return manifest

    def validate_results(self) -> dict[str, Any]:
        """Run validation checks on post-eval results.

        Returns
        -------
        report : dict
            Validation report with pass/fail status.
        """
        results = self._results
        n_total = len(results)
        n_errors = sum(1 for r in results if r.error is not None)

        # Check all probe families present
        families = {r.probe_family for r in results}
        expected_families = {
            "direct_visual",
            "image_plus_name",
            "wrong_name",
            "visual_text_conflict",
            "name_only",
        }
        families_complete = families == expected_families

        # Check no duplicate probe IDs
        probe_ids = [r.probe_id for r in results]
        no_duplicates = len(probe_ids) == len(set(probe_ids))

        # Check finite scores
        finite_scores = True
        for r in results:
            if (
                r.signed_answer_margin is not None
                and r.signed_answer_margin != r.signed_answer_margin  # NaN check
            ):
                finite_scores = False
                break

        passed = (
            n_errors == 0
            and families_complete
            and no_duplicates
            and finite_scores
            and n_total == 500
        )

        return {
            "passed": passed,
            "total_results": n_total,
            "errors": n_errors,
            "families_complete": families_complete,
            "families_present": sorted(families),
            "no_duplicate_probes": no_duplicates,
            "all_scores_finite": finite_scores,
            "expected_probe_count": n_total == 500,
        }


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
