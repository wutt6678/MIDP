#!/usr/bin/env python3
"""Canonical smoke runner for the Stage 3 unlearning pilot (P1-8).

This script provides a single entrypoint for the full 17-step unlearning
pilot pipeline, with an optional ``--smoke`` mode that exercises the
production code path with 1 optimizer step and 10 post-eval probes.

Usage::

    # Full production pilot
    python scripts/run_unlearning_pilot.py \\
        --config configs/experiments/unlearning_pilot_v1.yaml

    # Smoke mode (1 optimizer step, 10 probes)
    python scripts/run_unlearning_pilot.py \\
        --config configs/experiments/unlearning_pilot_v1.yaml \\
        --smoke

    # Resume a previous run
    python scripts/run_unlearning_pilot.py \\
        --config configs/experiments/unlearning_pilot_v1.yaml \\
        --resume

The 17 orchestration steps
--------------------------
1.  Frozen-input preflight
2.  Identity selection
3.  Leakage validation
4.  Intervention manifest
5.  Model load
6.  LoRA attach
7.  Trainable-parameter validation
8.  Training
9.  Checkpoint hashing
10. Adapter reload
11. Post-eval research preflight
12. Post-eval
13. Strict validation
14. Exact pairing
15. Paired analysis
16. Preservation report
17. GO/NO-GO report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_unlearning_pilot")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _sha256_file(path: str | Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _git_dirty() -> bool:
    """Return True if the git working tree is dirty."""
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=False,
        )
        return bool(r.stdout.strip()) if r.returncode == 0 else False
    except FileNotFoundError:
        return False


# --------------------------------------------------------------------------- #
# Step runner
# --------------------------------------------------------------------------- #

class StepTimer:
    """Context manager that logs elapsed time for a pipeline step."""

    def __init__(self, step_num: int, name: str) -> None:
        self.step_num = step_num
        self.name = name
        self._t0: float = 0.0

    def __enter__(self) -> "StepTimer":
        logger.info("Step %d/17: %s ...", self.step_num, self.name)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        elapsed = time.perf_counter() - self._t0
        logger.info(
            "Step %d/17: %s done (%.1fs)", self.step_num, self.name, elapsed,
        )


def _write_step_evidence(
    output_dir: Path, step_num: int, step_name: str, data: dict[str, Any],
) -> None:
    """Write per-step evidence JSON."""
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / f"step_{step_num:02d}_{step_name}.json"
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


# --------------------------------------------------------------------------- #
# Smoke-mode config overrides
# --------------------------------------------------------------------------- #

def apply_smoke_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Override config for smoke mode (1 step, 10 probes).

    Returns the mutated config (same object).
    """
    cfg.setdefault("method", {}).setdefault("hyperparameters", {})
    cfg["method"]["hyperparameters"]["num_optimizer_steps"] = 1
    cfg["method"]["hyperparameters"]["train_batch_size"] = 1
    cfg["method"]["hyperparameters"]["gradient_accumulation_steps"] = 1
    # Mark as smoke for downstream consumers.
    cfg.setdefault("runtime", {})["smoke_mode"] = True
    return cfg


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def run_pipeline(
    config_path: str,
    *,
    smoke: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute the full 17-step unlearning pilot pipeline.

    Parameters
    ----------
    config_path:
        Path to the experiment YAML config.
    smoke:
        If True, override to 1 optimizer step and 10 post-eval probes.
    resume:
        If True, skip steps that have already produced evidence.

    Returns
    -------
    final_report : dict
        The GO/NO-GO validation report.
    """
    from route_data.eval.run_pilot import (
        PilotRunner,
        validate_experiment_config,
        load_experiment_config,
        _sha256_file as _rp_sha256,
        _git_commit as _rp_git,
    )

    # -- Load & validate config ----------------------------------------- #
    cfg = load_experiment_config(config_path)
    validate_experiment_config(cfg)

    if smoke:
        logger.info("SMOKE MODE: overriding to 1 optimizer step, 10 probes")
        apply_smoke_overrides(cfg)

    base_dir = Path(config_path).resolve().parent.parent.parent
    runner = PilotRunner(config_path, base_dir=base_dir)
    # Inject overridden config so PilotRunner uses smoke values.
    runner._config = cfg

    output_dir = runner.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_state: dict[str, Any] = {
        "config_path": config_path,
        "smoke": smoke,
        "resume": resume,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "steps_completed": [],
    }

    def _already_done(step_name: str) -> bool:
        """Check if a step's evidence already exists (for resume)."""
        if not resume:
            return False
        evidence = output_dir / "evidence"
        # Check for any evidence file matching this step name.
        matches = list(evidence.glob(f"*_{step_name}.json"))
        return len(matches) > 0

    # =================================================================== #
    # Steps 1–4: Selection (CPU-only, delegated to PilotRunner)
    # =================================================================== #

    selection_manifest: dict[str, Any] = {}

    # Step 1: Frozen-input preflight
    # Step 2: Identity selection
    # Step 3: Leakage validation
    # Step 4: Intervention manifest
    if not _already_done("selection"):
        with StepTimer(1, "Frozen-input preflight + selection + leakage + manifest"):
            selection_manifest = runner.run_selection()
            _write_step_evidence(
                output_dir, 1, "selection",
                {
                    "target_identities": selection_manifest.get("target_identities", []),
                    "retain_identities": selection_manifest.get("retain_identities", []),
                    "control_identities": selection_manifest.get("control_identities", []),
                },
            )
        pipeline_state["steps_completed"].append("selection")
    else:
        logger.info("Steps 1-4: selection already complete (resume)")
        sel_path = output_dir / "selection" / "pilot_identity_selection.json"
        if sel_path.exists():
            selection_manifest = json.loads(sel_path.read_text())

    # =================================================================== #
    # Steps 5–8: Training (requires GPU)
    # =================================================================== #

    checkpoint_path = ""
    training_summary: dict[str, Any] = {}

    if not _already_done("training"):
        # Step 5: Model load
        # Step 6: LoRA attach
        # Step 7: Trainable-parameter validation
        # Step 8: Training
        with StepTimer(5, "Model load + LoRA attach + trainable-param validation + training"):
            training_summary = _run_training_phase(
                runner, cfg, selection_manifest, smoke=smoke,
            )
            checkpoint_path = training_summary.get("final_checkpoint_path", "")
            _write_step_evidence(
                output_dir, 5, "training",
                {
                    "training_summary": training_summary,
                    "checkpoint_path": checkpoint_path,
                },
            )
        pipeline_state["steps_completed"].append("training")
    else:
        logger.info("Steps 5-8: training already complete (resume)")
        # Find the latest checkpoint.
        ckpt_dir = output_dir / "checkpoints"
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.iterdir())
            if ckpts:
                checkpoint_path = str(ckpts[-1])

    # =================================================================== #
    # Step 9: Checkpoint hashing
    # =================================================================== #

    checkpoint_sha = ""
    if checkpoint_path:
        with StepTimer(9, "Checkpoint hashing"):
            ckpt_path = Path(checkpoint_path)
            for fname in ("adapter_model.safetensors", "adapter_model.bin"):
                fpath = ckpt_path / fname
                if fpath.exists():
                    checkpoint_sha = _sha256_file(fpath)
                    break
            _write_step_evidence(
                output_dir, 9, "checkpoint_hash",
                {"checkpoint_path": checkpoint_path, "sha256": checkpoint_sha},
            )
        pipeline_state["steps_completed"].append("checkpoint_hash")

    # =================================================================== #
    # Steps 10–16: Post-evaluation (requires GPU)
    # =================================================================== #

    post_eval_summary: dict[str, Any] = {}
    analysis_results: dict[str, Any] = {}

    if not _already_done("post_eval"):
        # Step 10: Adapter reload
        # Step 11: Post-eval research preflight
        # Step 12: Post-eval
        # Step 13: Strict validation
        # Step 14: Exact pairing
        with StepTimer(10, "Adapter reload + preflight + post-eval + validation + pairing"):
            post_eval_results = _run_post_eval_phase(
                runner, cfg, checkpoint_path, smoke=smoke,
            )
            post_eval_summary = post_eval_results.get("summary", {})
            _write_step_evidence(
                output_dir, 10, "post_eval",
                {
                    "post_eval_summary": post_eval_summary,
                    "validation": post_eval_results.get("validation", {}),
                },
            )
        pipeline_state["steps_completed"].append("post_eval")

        # Step 15: Paired analysis
        # Step 16: Preservation report
        with StepTimer(15, "Paired analysis + preservation report"):
            post_results_path = post_eval_results.get("results_path", "")
            if post_results_path:
                try:
                    analysis_results = runner.run_paired_analysis(post_results_path)
                except RuntimeError as exc:
                    if smoke:
                        logger.warning("Step 15: Paired analysis failed (expected in smoke mode): %s", exc)
                        analysis_results = {"smoke_mode": True, "error": str(exc)}
                    else:
                        raise
                else:
                    _write_step_evidence(
                        output_dir, 15, "paired_analysis",
                        {"analysis_keys": sorted(analysis_results.keys())},
                    )
        pipeline_state["steps_completed"].append("paired_analysis")
    else:
        logger.info("Steps 10-16: post-eval already complete (resume)")

    # =================================================================== #
    # Step 17: GO/NO-GO report
    # =================================================================== #

    with StepTimer(17, "GO/NO-GO validation report"):
        final_report = runner.generate_validation_report(
            training_summary=training_summary,
            post_eval_summary=post_eval_summary,
            analysis_results=analysis_results,
        )
        pipeline_state["final_report"] = final_report

    # -- Write pipeline provenance -------------------------------------- #
    pipeline_state["end_time"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
    )
    provenance_path = output_dir / "evidence" / "pipeline_provenance.json"
    provenance_path.write_text(
        json.dumps(pipeline_state, indent=2, default=str) + "\n"
    )
    logger.info("Pipeline provenance written to %s", provenance_path)

    # -- Summary -------------------------------------------------------- #
    gates = final_report.get("gates", {})
    all_pass = all(
        v is True for k, v in gates.items() if not k.startswith("_")
    )
    verdict = "GO" if all_pass else "NO-GO"
    logger.info("=" * 60)
    logger.info("PILOT VERDICT: %s", verdict)
    logger.info("=" * 60)
    for gate_name, gate_val in gates.items():
        if gate_name.startswith("_"):
            continue
        status = "PASS" if gate_val is True else "FAIL"
        logger.info("  %-50s %s", gate_name, status)

    return final_report


# --------------------------------------------------------------------------- #
# Training phase (steps 5–8)
# --------------------------------------------------------------------------- #

def _run_training_phase(
    runner: Any,
    cfg: dict[str, Any],
    selection_manifest: dict[str, Any],
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Execute steps 5-8: model load, LoRA attach, validation, training.

    Returns a training summary dict with ``final_checkpoint_path``.
    """
    from route_data.eval.unlearning_harness import (
        UnlearningConfig,
        UnlearningTrainer,
        apply_lora,
        build_forget_dataset,
        build_retain_dataset,
        generate_run_manifest,
        generate_trainable_parameter_report,
        load_base_model,
    )

    train_cfg_dict = runner.get_training_config(selection_manifest)

    # Step 5: Model load
    logger.info("Step 5: Loading base model %s ...", train_cfg_dict["model_id"])
    model, processor = load_base_model(
        model_id=train_cfg_dict["model_id"],
        revision=train_cfg_dict["model_revision"],
        dtype=train_cfg_dict["dtype"],
    )

    # Step 6: LoRA attach
    logger.info("Step 6: Attaching LoRA adapters ...")
    hp = cfg["method"]["hyperparameters"]
    model = apply_lora(
        model,
        r=hp["lora_rank"],
        lora_alpha=hp["lora_alpha"],
    )

    # Step 7: Trainable-parameter validation
    logger.info("Step 7: Validating trainable parameters ...")
    param_report = generate_trainable_parameter_report(model)
    logger.info(
        "  Trainable: %d / %d (%.2f%%)",
        param_report["trainable_parameters"],
        param_report["total_parameters"],
        param_report["trainable_percentage"],
    )

    # Step 8: Training
    logger.info("Step 8: Starting training ...")
    uc = UnlearningConfig(
        model_id=train_cfg_dict["model_id"],
        model_revision=train_cfg_dict["model_revision"],
        dtype=train_cfg_dict["dtype"],
        seed=train_cfg_dict["seed"],
        lora_rank=hp["lora_rank"],
        lora_alpha=hp["lora_alpha"],
        learning_rate=hp["learning_rate"],
        num_optimizer_steps=hp["num_optimizer_steps"],
        retain_weight=hp["retain_weight"],
        batch_size=hp["train_batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        forget_identity_ids=train_cfg_dict["forget_identity_ids"],
        retain_identity_ids=train_cfg_dict["retain_identity_ids"],
        processed_dataset_path=train_cfg_dict["processed_dataset_path"],
        route_probe_path=train_cfg_dict["route_probe_path"],
        output_dir=train_cfg_dict["output_dir"],
        selection_manifest_sha256=train_cfg_dict["selection_manifest_sha256"],
        code_commit=train_cfg_dict["code_commit"],
    )

    # Build datasets
    forget_ds = build_forget_dataset(
        uc.processed_dataset_path,
        uc.forget_identity_ids,
        processor,
        seed=uc.seed,
    )
    retain_ds = build_retain_dataset(
        uc.processed_dataset_path,
        uc.retain_identity_ids,
        processor,
        seed=uc.seed,
    )

    trainer = UnlearningTrainer(
        config=uc,
        model=model,
        processor=processor,
        forget_dataset=forget_ds,
        retain_dataset=retain_ds,
        reference_model=None,
    )
    training_summary = trainer.train()

    # Find final checkpoint
    ckpt_dir = Path(uc.output_dir) / "checkpoints"
    ckpts = sorted(ckpt_dir.iterdir()) if ckpt_dir.exists() else []
    final_ckpt = str(ckpts[-1]) if ckpts else ""
    training_summary["final_checkpoint_path"] = final_ckpt

    # Write run manifest
    git_dirty = _git_dirty()
    run_manifest = generate_run_manifest(
        uc, training_summary, param_report,
        code_commit=_git_commit(), git_dirty=git_dirty,
    )
    manifest_path = Path(uc.output_dir) / "unlearning_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
    logger.info("Run manifest written to %s", manifest_path)

    return training_summary


# --------------------------------------------------------------------------- #
# Post-eval phase (steps 10–16)
# --------------------------------------------------------------------------- #

def _run_post_eval_phase(
    runner: Any,
    cfg: dict[str, Any],
    checkpoint_path: str,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Execute steps 10-14: adapter reload, preflight, post-eval, validation.

    Returns a dict with ``summary``, ``validation``, and ``results_path``.
    """
    from route_data.eval.post_unlearning_eval import (
        PostEvalConfig,
        PostUnlearningEvaluator,
        load_lora_checkpoint,
    )

    post_cfg_dict = runner.get_post_eval_config(
        checkpoint_path=checkpoint_path,
        checkpoint_name=Path(checkpoint_path).name if checkpoint_path else "final",
    )

    # Step 10: Adapter reload
    logger.info("Step 10: Reloading adapter from %s ...", checkpoint_path)
    model, processor, adapter_metadata = load_lora_checkpoint(
        base_model_id=post_cfg_dict["model_id"],
        base_revision=post_cfg_dict["model_revision"],
        checkpoint_path=checkpoint_path,
        dtype=post_cfg_dict["dtype"],
    )

    # Build a QwenHFBackend wrapping the pre-loaded model + processor.
    # This provides score_candidates() and generate() for BaselineRunner,
    # and fingerprint() with all required P0-5 fields.
    from route_data.config import GenerationConfig, ModelConfig
    from route_data.models.qwen import QwenHFBackend

    qwen_config = ModelConfig(
        backend="qwen_hf",
        model_id=post_cfg_dict["model_id"],
        revision=post_cfg_dict["model_revision"],
        dtype=post_cfg_dict["dtype"],
        generation=GenerationConfig(do_sample=False),
    )
    backend = QwenHFBackend.from_loaded_model(
        config=qwen_config,
        model=model,
        processor=processor,
        adapter_metadata=adapter_metadata,
        resolved_revision=post_cfg_dict["model_revision"],
    )

    # Build model_config for BaselineRunner attribute access (e.g. .revision).
    # model_config_path is set in the config, so _compute_model_config_sha
    # hashes the YAML file directly (no asdict() fallback needed).
    import types
    _fp = backend.fingerprint()
    model_config = types.SimpleNamespace(
        model_id=post_cfg_dict["model_id"],
        revision=post_cfg_dict["model_revision"],
        dtype=post_cfg_dict["dtype"],
        backend="qwen_lora_pilot",
        fingerprint_id=_fp.get("fingerprint_id", ""),
        **adapter_metadata,
    )

    # Steps 11-14: Post-eval
    pe_config = PostEvalConfig(
        model_id=post_cfg_dict["model_id"],
        model_revision=post_cfg_dict["model_revision"],
        dtype=post_cfg_dict["dtype"],
        seed=post_cfg_dict["seed"],
        checkpoint_path=post_cfg_dict["checkpoint_path"],
        checkpoint_name=post_cfg_dict["checkpoint_name"],
        probe_path=post_cfg_dict["probe_path"],
        baseline_results_path=post_cfg_dict["baseline_results_path"],
        baseline_manifest_path=post_cfg_dict["baseline_manifest_path"],
        output_dir=post_cfg_dict["output_dir"],
        selection_manifest_sha256=post_cfg_dict["selection_manifest_sha256"],
        code_commit=post_cfg_dict["code_commit"],
        dataset_manifest_path=post_cfg_dict["dataset_manifest_path"],
        freeze_verification_path=post_cfg_dict["freeze_verification_path"],
        processed_dataset_path=post_cfg_dict["processed_dataset_path"],
        model_config_path=post_cfg_dict["model_config_path"],
    )

    evaluator = PostUnlearningEvaluator(
        config=pe_config, backend=backend, model_config=model_config,
    )

    # Step 11-12: Preflight + inference
    limit = 10 if smoke else None
    logger.info("Steps 11-12: Running post-eval (limit=%s) ...", limit)
    evaluator.run_evaluation(limit=limit)
    results_path = evaluator.save_results()

    # Step 13: Strict validation
    logger.info("Step 13: Running strict validation ...")
    try:
        validation_report = evaluator.validate_results()
    except RuntimeError as exc:
        if smoke:
            logger.warning("Step 13: Validation failed (expected in smoke mode): %s", exc)
            validation_report = {"pass": False, "smoke_mode": True, "error": str(exc)}
        else:
            raise

    # Step 14: Exact pairing
    logger.info("Step 14: Validating exact probe matching ...")
    try:
        pairing = evaluator.validate_against_baseline()
    except RuntimeError as exc:
        if smoke:
            logger.warning("Step 14: Pairing validation failed (expected in smoke mode): %s", exc)
            pairing = {"pass": False, "smoke_mode": True, "error": str(exc)}
        else:
            raise

    # Summary
    summary = evaluator.generate_summary()

    # Post-eval manifest
    evaluator.generate_post_eval_manifest()

    return {
        "summary": summary,
        "validation": validation_report,
        "pairing": pairing,
        "results_path": str(results_path),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Canonical smoke runner for the Stage 3 unlearning pilot.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Smoke mode: 1 optimizer step, 10 post-eval probes.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume: skip steps with existing evidence.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error("Config file not found: %s", config_path)
        return 1

    try:
        report = run_pipeline(
            str(config_path),
            smoke=args.smoke,
            resume=args.resume,
        )
    except Exception:
        logger.exception("Pipeline failed with exception")
        return 1

    gates = report.get("gates", {})
    all_pass = all(
        v is True for k, v in gates.items() if not k.startswith("_")
    )
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
