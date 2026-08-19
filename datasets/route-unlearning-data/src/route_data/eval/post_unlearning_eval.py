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
.. autofunction:: evaluate_intervention
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

    # P0-7: Full frozen evaluation contract
    dataset_manifest_path: str = ""
    freeze_verification_path: str = ""
    processed_dataset_path: str = ""
    model_config_path: str = ""
    # P0-4: Selection manifest path for identity-based grouping.
    selection_manifest_path: str = ""
    # P0-19/20: Route probe SHA for suite-level verification
    route_probe_sha256: str = ""

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
) -> tuple[Any, Any, dict[str, str]]:
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
    adapter_metadata : dict[str, str]
        Adapter provenance for the adapter-aware fingerprint (P0-6).
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
    checkpoint_path = Path(checkpoint_path)
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        torch_dtype=torch_dtype,
    )
    model.eval()

    # Build adapter metadata for the adapter-aware fingerprint (P0-6).
    adapter_metadata = _build_adapter_metadata(checkpoint_path)

    logger.info(
        f"Loaded LoRA checkpoint from {checkpoint_path} "
        f"on base {base_model_id} revision {base_revision}"
    )
    return model, processor, adapter_metadata


def _build_adapter_metadata(checkpoint_path: Path) -> dict[str, str]:
    """Compute adapter provenance for the adapter-aware fingerprint (P0-6).

    Reads the LoRA adapter config and computes SHA-256 of the checkpoint
    artifacts so that different adapter checkpoints produce different
    cache keys.
    """
    metadata: dict[str, str] = {
        "adapter_checkpoint_path": str(checkpoint_path),
    }

    # SHA-256 of adapter_model.safetensors (or adapter_model.bin).
    for fname in ("adapter_model.safetensors", "adapter_model.bin"):
        fpath = checkpoint_path / fname
        if fpath.is_file():
            metadata["adapter_checkpoint_sha"] = _sha256_file(fpath)
            break

    # SHA-256 of adapter_config.json + extract LoRA hyperparameters.
    config_path = checkpoint_path / "adapter_config.json"
    if config_path.is_file():
        metadata["adapter_config_sha"] = _sha256_file(config_path)
        try:
            with open(config_path) as f:
                lora_cfg = json.load(f)
            # LoRA hyperparameters.
            if "r" in lora_cfg:
                metadata["lora_rank"] = str(lora_cfg["r"])
            if "lora_alpha" in lora_cfg:
                metadata["lora_alpha"] = str(lora_cfg["lora_alpha"])
            if "target_modules" in lora_cfg:
                targets = lora_cfg["target_modules"]
                if isinstance(targets, list):
                    metadata["lora_target_modules"] = ",".join(sorted(targets))
                else:
                    metadata["lora_target_modules"] = str(targets)
        except Exception:
            pass

    # Derive checkpoint name and step from the directory name.
    metadata["checkpoint_name"] = checkpoint_path.name
    # Try to extract step number from directory name (e.g., "step_050").
    name = checkpoint_path.name
    import re
    step_match = re.search(r"(?:step_?|optimizer_step_?)(\d+)", name, re.IGNORECASE)
    if step_match:
        metadata["checkpoint_step"] = step_match.group(1)

    return metadata


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
    model_config:
        Model configuration for the BaselineRunner.
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

        # P0-11: Use checkpoint-specific output directory to prevent
        # cache reuse across different adapter checkpoints.
        output_dir = Path(config.output_dir)
        if config.checkpoint_name:
            output_dir = output_dir / config.checkpoint_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # P0-7: Pass full frozen evaluation contract to BaselineRunner.
        self._runner = BaselineRunner(
            backend=backend,
            probe_path=config.probe_path,
            output_dir=str(output_dir),
            model_config=model_config,
            resume=True,
            dataset_manifest_path=config.dataset_manifest_path or None,
            model_config_path=config.model_config_path or None,
            freeze_verification_path=config.freeze_verification_path or None,
            processed_dataset_path=config.processed_dataset_path or None,
        )

        self._results: list[BaselineResult] = []

    def run_evaluation(
        self,
        limit: int | None = None,
        smoke_probes: list | None = None,
    ) -> list[BaselineResult]:
        """Run post-unlearning evaluation on the frozen 500 probes.

        Parameters
        ----------
        limit:
            Optional limit on number of probes (for smoke testing).
        smoke_probes:
            If provided, run exactly these probes instead of all probes.
            Used for deterministic smoke testing (P0-2).

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

        # P0-8: Run research preflight before any inference.
        logger.info("Running research preflight...")
        self._runner.validate_research_preflight()

        if smoke_probes is not None:
            logger.info(
                "Smoke mode: running %d deterministic probes via run_selected()",
                len(smoke_probes),
            )
            self._results = self._runner.run_selected(smoke_probes)
        else:
            self._results = self._runner.run_all(limit=limit)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Post-eval complete: {len(self._results)} results in {elapsed:.1f}s"
        )
        return self._results

    def save_results(self) -> Path:
        """Write post-eval results to ``results.jsonl``."""
        from dataclasses import asdict

        # P0-11: Use the same checkpoint-specific directory as the runner.
        output_dir = Path(self.config.output_dir)
        if self.config.checkpoint_name:
            output_dir = output_dir / self.config.checkpoint_name
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "results.jsonl"

        rows = [asdict(r) for r in self._results]
        with open(path, "w") as f:
            f.writelines(json.dumps(row, default=str) + "\n" for row in rows)

        logger.info(f"Wrote {len(rows)} post-eval results to {path}")
        return path

    def generate_summary(self) -> dict[str, Any]:
        """Generate post-eval summary with per-family metrics."""
        return self._runner.generate_summary()

    def validate_against_baseline(
        self,
        smoke_probe_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """Validate post-eval probe IDs match the frozen baseline.

        Parameters
        ----------
        smoke_probe_ids:
            If provided, validate that post-eval IDs match exactly this
            subset (which must also be a subset of baseline IDs). Used
            for deterministic smoke testing (P0-4).

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

        if smoke_probe_ids is not None:
            # In smoke mode, compare against the smoke subset of baseline
            baseline_results = [
                r for r in baseline_results if r["probe_id"] in smoke_probe_ids
            ]

        return validate_exact_probe_matching(baseline_results, post_results)

    def generate_post_eval_manifest(
        self,
        smoke_probe_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """Generate the post-evaluation manifest with full provenance.

        Parameters
        ----------
        smoke_probe_ids:
            If provided, validate against this smoke subset instead of
            the full baseline.  Adds ``evaluation_scope`` metadata to
            the manifest indicating smoke vs full mode.

        Returns
        -------
        manifest : dict
            The post-eval manifest with SHA-256 bindings.
        """
        # P0-11: Use the same checkpoint-specific directory as the runner.
        output_dir = Path(self.config.output_dir)
        if self.config.checkpoint_name:
            output_dir = output_dir / self.config.checkpoint_name

        # Compute result file SHA
        results_path = output_dir / "results.jsonl"
        results_sha = ""
        if results_path.exists():
            results_sha = _sha256_file(results_path)

        # Compute summary
        summary = self.generate_summary()

        # Validate probe matching (smoke-aware).
        validation = self.validate_against_baseline(
            smoke_probe_ids=smoke_probe_ids,
        )

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
            "evaluation_scope": {
                "mode": "smoke" if smoke_probe_ids is not None else "full",
                "expected_probe_count": (
                    len(smoke_probe_ids)
                    if smoke_probe_ids is not None
                    else 500
                ),
            },
        }

        # P0-14: Include per-family per-group probe counts in manifest.
        _fam_map = {
            "direct_visual": "DV",
            "image_plus_name": "IPN",
            "wrong_name": "WN",
            "visual_text_conflict": "VTC",
            "name_only": "name_only",
        }
        _grp_counts: dict[str, dict[str, int]] = {
            g: {f: 0 for f in ("DV", "IPN", "WN", "VTC", "name_only")}
            for g in ("target", "retain", "control", "untargeted")
        }
        # Load selection for group classification.
        _tgt: set[str] = set()
        _ret: set[str] = set()
        _ctl: set[str] = set()
        _sel_path = self.config.selection_manifest_path
        if _sel_path and Path(_sel_path).is_file():
            with open(_sel_path) as _sf:
                _sd = json.load(_sf)
            _tgt = set(_sd.get("target_identities", []))
            _ret = set(_sd.get("retain_identities", []))
            _ctl = set(_sd.get("control_identities", []))
        for _r in self._results:
            _iid = _r.get("identity_id", "") if isinstance(_r, dict) else getattr(_r, "identity_id", "")
            _grp = "target" if _iid in _tgt else "retain" if _iid in _ret else "control" if _iid in _ctl else "untargeted"
            _pf = _r.get("probe_family", "") if isinstance(_r, dict) else getattr(_r, "probe_family", "")
            _fk = _fam_map.get(_pf, "")
            if _fk and _grp in _grp_counts:
                _grp_counts[_grp][_fk] += 1
        manifest["group_probe_counts"] = _grp_counts

        # Write manifest
        manifest_path = output_dir / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
            f.write("\n")

        logger.info(f"Wrote post-eval manifest to {manifest_path}")
        return manifest

    def validate_results(
        self,
        smoke_probe_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """Strict post-eval validation matching baseline standards (P0-8).

        Delegates to the BaselineRunner's ``validate_results()`` which
        performs the full set of research-grade checks:

        1. Exact probe-ID set equality
        2. Family counts match
        3. Binary score completeness (finite logp, p_yes, margins)
        4. Name-only generation/metric completeness
        5. Source metadata consistency
        6. Run provenance consistency
        7. Zero inference errors
        8. Protocol-role completeness
        9. Processed-dataset SHA match

        The only allowed difference from baseline is the model fingerprint
        (which differs due to the adapter).

        Parameters
        ----------
        smoke_probe_ids:
            If provided, validate against this subset of probe IDs instead
            of all probes. Used for deterministic smoke testing (P0-4).

        Returns
        -------
        report : dict
            Validation report with ``pass`` (bool) and ``checks`` (dict).

        Raises
        ------
        RuntimeError
            If any check fails.
        """
        # Delegate to the BaselineRunner's strict validation.
        return self._runner.validate_results(smoke_probe_ids=smoke_probe_ids)


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


# --------------------------------------------------------------------------- #
# Common evaluation orchestrator (P0-1)
# --------------------------------------------------------------------------- #

def evaluate_intervention(
    model: Any,
    processor: Any,
    adapter_path: str | Path | None,
    probe_dataset_path: str | Path,
    output_dir: str | Path,
    config: PostEvalConfig | None = None,
    *,
    baseline_results_path: str | Path = "",
    method_name: str = "unknown",
    objective_name: str = "",
    model_config_obj: Any = None,
    backend_override: Any = None,
    trainable_adapter: Any = None,
) -> dict[str, Any]:
    """Common evaluation orchestrator for all unlearning methods.

    All methods (training-based, pruning-based, prompting) call this
    function after their intervention to produce a standardised result
    dict that the comparison framework can consume directly.

    Parameters
    ----------
    model:
        The model in its final intervention state.  For adapter methods
        this is the base model with the trained LoRA adapter still
        attached.  For prompting this is the unmodified base model.
    processor:
        The Qwen ``AutoProcessor`` matching the base model.
    adapter_path:
        Path to the saved LoRA adapter checkpoint directory, or *None*
        for prompting (no adapter).
    probe_dataset_path:
        Path to the frozen 500-probe JSONL file.
    output_dir:
        Directory where evaluation artefacts are written.
    config:
        Optional :class:`PostEvalConfig` with advanced settings
        (baseline paths, provenance, frozen-contract paths).  When
        *None* a minimal default config is constructed from the other
        parameters.
    baseline_results_path:
        Path to the pre-unlearning baseline ``results.jsonl``.  Required
        for ΔM computation.  Falls back to
        ``config.baseline_results_path`` when empty.
    method_name:
        Human-readable method identifier included in the result dict.
    model_config_obj:
        Optional :class:`~route_data.config.ModelConfig` for the
        backend.  When *None* a default config is built from
        ``PostEvalConfig`` defaults.
    backend_override:
        Optional pre-built backend instance (e.g. ``_PromptingBackend``)
        to use *instead of* constructing a fresh ``QwenHFBackend`` from
        the model.  Used by the prompting baseline (P0-7) so that the
        canonical 500-probe evaluation sees the system-prompt-aware
        backend rather than the plain base model.

    Returns
    -------
    dict[str, Any]
        Standardised result dict with keys: ``method``,
        ``delta_target``, ``delta_retain``, ``delta_control``,
        ``delta_untargeted``,
        ``exact_pair_count``, ``inference_errors``,
        ``manifest_sha256``, ``per_family_post``, ``summary``,
        ``eval_output_dir``.
    """
    from ..config import ModelConfig

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Build effective config ------------------------------------------------ #
    if config is None:
        config = PostEvalConfig()
    if baseline_results_path:
        config.baseline_results_path = str(baseline_results_path)
    config.probe_path = str(probe_dataset_path)
    config.output_dir = str(output_dir)

    # -- Build ModelConfig for the backend ------------------------------------ #
    if model_config_obj is None:
        model_config_obj = ModelConfig(
            backend="qwen_hf",
            model_id=config.model_id,
            revision=config.model_revision,
            dtype=config.dtype,
            device_map=config.device,
            seed=config.seed,
        )

    # -- Wrap model in a VisionLanguageModel backend -------------------------- #
    adapter_metadata = None
    if backend_override is not None:
        # P0-7: Use the caller-supplied backend (e.g. _PromptingBackend)
        # instead of constructing a fresh backend from the model.
        backend = backend_override
    else:
        if adapter_path is not None:
            adapter_path = Path(adapter_path)
            adapter_metadata = _build_adapter_metadata(adapter_path)

        if trainable_adapter is not None:
            # Model-agnostic path: use the trainable adapter to build
            # the correct eval backend for this model family.
            backend = trainable_adapter.to_eval_backend(
                model=model,
                processor=processor,
                model_config=model_config_obj,
                adapter_metadata=adapter_metadata,
            )
        else:
            # Legacy path: hardcoded Qwen backend
            from ..models.qwen import QwenHFBackend
            backend = QwenHFBackend.from_loaded_model(
                config=model_config_obj,
                model=model,
                processor=processor,
                adapter_metadata=adapter_metadata,
                resolved_revision=config.model_revision,
            )

    # -- Run evaluation ------------------------------------------------------- #
    evaluator = PostUnlearningEvaluator(
        config=config,
        backend=backend,
        model_config=model_config_obj,
    )
    post_results: list[BaselineResult] = evaluator.run_evaluation()
    results_path = evaluator.save_results()

    # -- Strict frozen-contract validation (P0-2) -------------------------- #
    strict_report = evaluator.validate_results()
    if not strict_report.get("pass", False):
        failed = [
            k for k, v in strict_report.get("checks", {}).items()
            if not v
        ]
        raise RuntimeError(
            f"evaluate_intervention({method_name}): strict validation FAILED "
            f"— {failed}"
        )

    pair_report = evaluator.validate_against_baseline()
    if not pair_report.get("exact_match", False):
        raise RuntimeError(
            f"evaluate_intervention({method_name}): exact pairing FAILED — "
            f"{pair_report}"
        )

    # -- P0-14: Persist validation reports to disk ------------------------- #
    strict_report_path = output_dir / "strict_validation.json"
    with open(strict_report_path, "w") as f:
        json.dump(strict_report, f, indent=2, default=str)
        f.write("\n")
    logger.info(f"Persisted strict_validation.json: {strict_report_path}")

    pair_report_path = output_dir / "pairing_validation.json"
    with open(pair_report_path, "w") as f:
        json.dump(pair_report, f, indent=2, default=str)
        f.write("\n")
    logger.info(f"Persisted pairing_validation.json: {pair_report_path}")

    summary = evaluator.generate_summary()

    # -- Load selection manifest for identity-based grouping (P0-5) ------ #
    selection_manifest_path = getattr(config, "selection_manifest_path", "")
    target_ids: set[str] = set()
    retain_ids: set[str] = set()
    control_ids: set[str] = set()
    selection_manifest_sha = ""
    if selection_manifest_path and Path(selection_manifest_path).is_file():
        with open(selection_manifest_path) as _sf:
            _sel = json.load(_sf)
        target_ids = set(_sel.get("target_identities", []))
        retain_ids = set(_sel.get("retain_identities", []))
        control_ids = set(_sel.get("control_identities", []))
        selection_manifest_sha = _sha256_file(selection_manifest_path)

    def _classify_identity(identity_id: str) -> str:
        """Classify an identity into target/retain/control/untargeted."""
        if identity_id in target_ids:
            return "target"
        if identity_id in retain_ids:
            return "retain"
        if identity_id in control_ids:
            return "control"
        return "untargeted"

    # -- Load baseline results for ΔM computation ----------------------------- #
    baseline_path = Path(config.baseline_results_path)
    baseline_results: list[dict[str, Any]] = []
    if baseline_path.is_file():
        with open(baseline_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    baseline_results.append(json.loads(line))

    # -- Compute per-probe ΔM and aggregate ----------------------------------- #
    baseline_by_id: dict[str, dict[str, Any]] = {
        r["probe_id"]: r for r in baseline_results
    }

    # Per-probe metric extraction.
    def _metric(r: Any, is_dataclass: bool) -> float | None:
        """Extract the primary metric from a result (dataclass or dict)."""
        if is_dataclass:
            fam = r.probe_family
            if fam == "name_only":
                return r.normalized_exact_match
            return r.signed_answer_margin
        else:
            fam = r.get("probe_family", "")
            if fam == "name_only":
                return r.get("normalized_exact_match")
            return r.get("signed_answer_margin")

    # Aggregate: group by (probe_family, group) → list of ΔM.
    from collections import defaultdict
    delta_accum: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # P0-8: Separate accumulator for name_only metrics.
    name_only_accum: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # P0-16: Track per-group DV accuracy for preservation reporting.
    dv_correct_per_group: dict[str, list[bool]] = defaultdict(list)
    # Track all identity IDs seen for count enforcement.
    identity_groups: dict[str, str] = {}
    n_pairs = 0
    n_inference_errors = 0

    for pr in post_results:
        pid = pr.probe_id
        br = baseline_by_id.get(pid)
        if br is None:
            continue
        n_pairs += 1

        # Count inference errors.
        if pr.error is not None:
            n_inference_errors += 1

        # P0-5: Classify by identity, not protocol_role.
        identity_id = pr.identity_id if hasattr(pr, "identity_id") else ""
        group = _classify_identity(identity_id)

        # Track identity → group mapping for count enforcement.
        if identity_id:
            identity_groups[identity_id] = group

        # P0-16: Track DV accuracy per group.
        if pr.probe_family == "direct_visual" and pr.correct is not None:
            dv_correct_per_group[group].append(pr.correct)

        post_m = _metric(pr, is_dataclass=True)
        pre_m = _metric(br, is_dataclass=False)
        if post_m is None or pre_m is None:
            continue
        delta = post_m - pre_m

        family = pr.probe_family
        role_key = f"delta_{group}"

        # P0-8: name_only uses a separate accumulator.
        if family == "name_only":
            name_only_accum[family][role_key].append(delta)
        else:
            delta_accum[family][role_key].append(delta)

    # P0-6: Enforce frozen 2/2/2/94 identity counts.
    _EXPECTED_IDENTITY_COUNTS = {
        "target": 2, "retain": 2, "control": 2, "untargeted": 94,
    }
    group_identity_counts: dict[str, int] = {}
    for grp in ("target", "retain", "control", "untargeted"):
        group_identity_counts[grp] = sum(
            1 for g in identity_groups.values() if g == grp
        )
    identity_counts_valid = all(
        group_identity_counts.get(g, 0) == expected
        for g, expected in _EXPECTED_IDENTITY_COUNTS.items()
    )
    
    # P0-10/11: Compute per-family per-group probe counts.
    _FAMILIES = ("DV", "IPN", "WN", "VTC", "name_only")
    _GROUPS = ("target", "retain", "control", "untargeted")
    _FAMILY_MAP = {
        "direct_visual": "DV",
        "image_plus_name": "IPN",
        "wrong_name": "WN",
        "visual_text_conflict": "VTC",
        "name_only": "name_only",
    }
    group_probe_counts: dict[str, dict[str, int]] = {
        grp: {fam: 0 for fam in _FAMILIES} for grp in _GROUPS
    }
    for pr in post_results:
        identity_id = pr.identity_id if hasattr(pr, "identity_id") else ""
        grp = _classify_identity(identity_id)
        fam_key = _FAMILY_MAP.get(pr.probe_family, "")
        if fam_key and grp in group_probe_counts:
            group_probe_counts[grp][fam_key] += 1

    # Average per (family, role).
    _FAMILY_ABBREV = {
        "direct_visual": "DV",
        "image_plus_name": "IPN",
        "wrong_name": "WN",
        "visual_text_conflict": "VTC",
    }

    def _avg_deltas(accum_key: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for family, roles in delta_accum.items():
            vals = roles.get(accum_key, [])
            if vals:
                short = _FAMILY_ABBREV.get(family, family)
                out[short] = sum(vals) / len(vals)
        return out

    delta_target = _avg_deltas("delta_target")
    delta_retain = _avg_deltas("delta_retain")
    delta_control = _avg_deltas("delta_control")
    delta_untargeted = _avg_deltas("delta_untargeted")

    # P0-8: Compute name_only deltas separately.
    def _avg_name_only(accum_key: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for roles in name_only_accum.values():
            vals = roles.get(accum_key, [])
            if vals:
                out["normalized_exact_match"] = sum(vals) / len(vals)
        return out

    name_only_delta = {
        "target": _avg_name_only("delta_target"),
        "retain": _avg_name_only("delta_retain"),
        "control": _avg_name_only("delta_control"),
        "untargeted": _avg_name_only("delta_untargeted"),
    }

    # -- P0-13: Generate the post-eval manifest BEFORE computing its SHA ---- #
    evaluator.generate_post_eval_manifest()

    # -- Manifest SHA --------------------------------------------------------- #
    manifest_sha = ""
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest_sha = _sha256_file(manifest_path)

    # -- P0-16: Compute DV accuracy per identity-based group ----------------- #
    dv_accuracy: dict[str, float] = {}
    all_dv_corrects = []
    for grp_label, corrects in dv_correct_per_group.items():
        if corrects:
            dv_accuracy[grp_label] = sum(corrects) / len(corrects)
            all_dv_corrects.extend(corrects)
    if all_dv_corrects:
        dv_accuracy["global"] = sum(all_dv_corrects) / len(all_dv_corrects)
    else:
        dv_accuracy["global"] = 0.0

    # -- P1-30: Build intervention provenance -------------------------------- #
    intervention_provenance: dict[str, Any] = {
        "method": method_name,
    }
    if adapter_metadata:
        # Adapter methods: record adapter SHA and config SHA.
        intervention_provenance["adapter_checkpoint_sha"] = (
            adapter_metadata.get("adapter_checkpoint_sha", "")
        )
        intervention_provenance["adapter_config_sha"] = (
            adapter_metadata.get("adapter_config_sha", "")
        )
    # Callers may supply extra method-specific provenance via the
    # backend_override (e.g. MANU prune fraction, prompting prompt hash).
    # These are picked up from the result dict additions below.

    # -- Build result dict ---------------------------------------------------- #
    # P0-8: name_only separated; P0-16: DV accuracy; P0-25: group provenance.
    # P0-10: objective_name records the training objective; method is the
    # canonical suite ID.
    result: dict[str, Any] = {
        "method": method_name,
        "objective_name": objective_name or method_name,
        # P0-16: Contract metadata for new evaluations.
        "evidence_mode": "new_evaluation",
        "validation_contract_version": "mllmu-baseline-suite-v1",
        "evaluation_scope": {
            "mode": "full",
            "expected_probe_count": 500,
        },
        "delta_target": delta_target,
        "delta_retain": delta_retain,
        "delta_control": delta_control,
        "delta_untargeted": delta_untargeted,
        # P0-8: name_only fully separated from signed-margin deltas.
        "name_only_delta": name_only_delta,
        "exact_pair_count": n_pairs,
        "inference_errors": n_inference_errors,
        "manifest_sha256": manifest_sha,
        "per_family_post": summary.get("per_family", {}),
        "summary": summary,
        "eval_output_dir": str(output_dir),
        "results_path": str(results_path),
        "adapter_path": str(adapter_path) if adapter_path else None,
        # Strict validation fields (P0-2).
        "strict_validation_pass": strict_report.get("pass", False),
        "exact_pairing_pass": pair_report.get("exact_match", False),
        "expected_pair_count": 500,
        "actual_pair_count": n_pairs,
        # P0-19/20: Provenance fields for suite-level verification.
        "model_revision": config.model_revision,
        "route_probe_sha256": getattr(config, "route_probe_sha256", ""),
        # P0-4: Selection manifest SHA for identity-grouping provenance.
        "selection_manifest_sha256": selection_manifest_sha,
        # P0-16: DV accuracy per identity-based group for preservation gate.
        "dv_accuracy": dv_accuracy,
        # P0-6: Identity group counts for frozen-contract enforcement.
        "group_identity_counts": group_identity_counts,
        "identity_counts_valid": identity_counts_valid,
        # P0-10/11: Per-family per-group probe counts.
        "group_probe_counts": group_probe_counts,
        # P0-25: Experimental-group provenance.
        "group_definition": {
            "selection_manifest_path": selection_manifest_path,
            "selection_manifest_sha256": selection_manifest_sha,
            "target_identity_ids": sorted(target_ids),
            "retain_identity_ids": sorted(retain_ids),
            "control_identity_ids": sorted(control_ids),
            "untargeted_identity_count": group_identity_counts.get("untargeted", 0),
        },
        # P1-30: Intervention provenance for traceability.
        "intervention_provenance": intervention_provenance,
    }

    # -- Persist as the canonical suite artifact (P0-1) ---------------------- #
    eval_results_path = output_dir / "eval_results.json"
    with open(eval_results_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
        f.write("\n")
    logger.info(f"Wrote eval_results.json: {eval_results_path}")

    logger.info(
        f"evaluate_intervention({method_name}): "
        f"{n_pairs} pairs, {n_inference_errors} errors, "
        f"target_families={list(delta_target)}, "
        f"retain_families={list(delta_retain)}, "
        f"control_families={list(delta_control)}, "
        f"untargeted_families={list(delta_untargeted)}"
    )
    return result
