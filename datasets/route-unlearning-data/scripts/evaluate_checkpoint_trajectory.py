#!/usr/bin/env python3
"""Checkpoint trajectory diagnostic evaluation.

Evaluates saved adapter checkpoints on a diagnostic subset containing
target, retain, and control identities across the four binary route families.
Determines whether an early "selective forgetting window" exists before collapse.

Usage:
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python scripts/evaluate_checkpoint_trajectory.py \\
        --pilot-dir outputs/experiments/unlearning_pilot/Qwen_Qwen3.5-9B/pilot_v1 \\
        --config configs/experiments/unlearning_pilot_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("checkpoint_trajectory")


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

BINARY_FAMILIES = [
    "direct_visual",
    "image_plus_name",
    "wrong_name",
    "visual_text_conflict",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_config(config_path: Path) -> dict[str, Any]:
    """Load experiment config."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_baseline_results(baseline_path: Path) -> list[dict]:
    """Load baseline results."""
    results = []
    with open(baseline_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def get_diagnostic_probe_ids(
    baseline_results: list[dict],
    target_ids: list[str],
    retain_ids: list[str],
    control_ids: list[str],
) -> set[str]:
    """Get probe IDs for diagnostic subset (binary families only)."""
    all_ids = set(target_ids + retain_ids + control_ids)
    probe_ids = set()
    for row in baseline_results:
        if row["identity_id"] in all_ids and row["probe_family"] in BINARY_FAMILIES:
            probe_ids.add(row["probe_id"])
    return probe_ids


def evaluate_checkpoint(
    checkpoint_path: Path,
    checkpoint_name: str,
    config: dict[str, Any],
    diagnostic_probe_ids: set[str],
) -> dict[str, Any]:
    """Evaluate a single checkpoint on the diagnostic subset."""
    import types
    from route_data.eval.post_unlearning_eval import (
        PostEvalConfig,
        PostUnlearningEvaluator,
        load_lora_checkpoint,
    )
    from route_data.config import GenerationConfig, ModelConfig
    from route_data.models.qwen import QwenHFBackend
    
    model_id = config["base_model"]["model_id"]
    model_revision = config["base_model"]["revision"]
    dtype = config["runtime"].get("dtype", "bfloat16")
    baseline_path = Path(config["baseline"]["results_path"])
    probe_path = Path(config["dataset"]["route_probe_path"])
    dataset_manifest_path = config["dataset"].get("research_manifest_path")
    freeze_verification_path = config["dataset"].get("freeze_verification_path")
    processed_dataset_path = config["dataset"].get("processed_dataset_path")
    model_config_path = config["base_model"].get("model_config_path")
    
    # Load adapter
    model, processor, adapter_metadata = load_lora_checkpoint(
        base_model_id=model_id,
        base_revision=model_revision,
        checkpoint_path=str(checkpoint_path),
        dtype=dtype,
    )
    
    # Create backend
    qwen_config = ModelConfig(
        backend="qwen_hf",
        model_id=model_id,
        revision=model_revision,
        dtype=dtype,
        generation=GenerationConfig(do_sample=False),
    )
    backend = QwenHFBackend.from_loaded_model(
        config=qwen_config,
        model=model,
        processor=processor,
        adapter_metadata=adapter_metadata,
        resolved_revision=model_revision,
    )
    
    # Build model_config
    _fp = backend.fingerprint()
    model_config = types.SimpleNamespace(
        model_id=model_id,
        revision=model_revision,
        dtype=dtype,
        backend="qwen_lora_pilot",
        fingerprint_id=_fp.get("fingerprint_id", ""),
        **adapter_metadata,
    )
    
    # Create PostEvalConfig
    output_dir = checkpoint_path.parent.parent / "diagnostic_eval" / checkpoint_name
    pe_config = PostEvalConfig(
        model_id=model_id,
        model_revision=model_revision,
        dtype=dtype,
        seed=config["selection"].get("seed", 17),
        checkpoint_path=str(checkpoint_path),
        checkpoint_name=checkpoint_name,
        probe_path=str(probe_path),
        baseline_results_path=str(baseline_path),
        baseline_manifest_path=str(baseline_path.parent / "manifest.json"),
        output_dir=str(output_dir),
        selection_manifest_sha256="",
        code_commit="",
        dataset_manifest_path=dataset_manifest_path,
        freeze_verification_path=freeze_verification_path,
        processed_dataset_path=processed_dataset_path,
        model_config_path=model_config_path,
    )
    
    # Create evaluator
    evaluator = PostUnlearningEvaluator(
        config=pe_config, backend=backend, model_config=model_config,
    )
    
    # Filter probes to diagnostic subset
    all_probes = evaluator._runner.probes
    diagnostic_probes = [p for p in all_probes if p.probe_id in diagnostic_probe_ids]
    
    logger.info(f"Running {len(diagnostic_probes)} diagnostic probes")
    
    # Run evaluation
    evaluator.run_evaluation(smoke_probes=diagnostic_probes)
    results_path = evaluator.save_results()
    
    # Load results and compute metrics
    results = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))
    
    return results


def compute_metrics(
    results: list[dict],
    target_ids: list[str],
    retain_ids: list[str],
    control_ids: list[str],
) -> dict[str, Any]:
    """Compute per-group, per-family metrics."""
    metrics = {}
    
    for group_name, group_ids in [
        ("target", target_ids),
        ("retain", retain_ids),
        ("control", control_ids),
    ]:
        group_results = [r for r in results if r["identity_id"] in group_ids]
        
        group_metrics = {
            "count": len(group_results),
            "accuracy": sum(1 for r in group_results if r.get("correct", False)) / max(len(group_results), 1),
            "mean_signed_margin": sum(r.get("signed_answer_margin", 0.0) for r in group_results) / max(len(group_results), 1),
        }
        
        # Per-family breakdown
        family_metrics = {}
        for family in BINARY_FAMILIES:
            family_results = [r for r in group_results if r["probe_family"] == family]
            if family_results:
                family_metrics[family] = {
                    "count": len(family_results),
                    "accuracy": sum(1 for r in family_results if r.get("correct", False)) / max(len(family_results), 1),
                    "mean_signed_margin": sum(r.get("signed_answer_margin", 0.0) for r in family_results) / max(len(family_results), 1),
                }
        
        group_metrics["per_family"] = family_metrics
        metrics[group_name] = group_metrics
    
    return metrics


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #

def _compute_delta_metrics(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compute ΔM = M_t - M_0 for all groups and families."""
    delta = {}
    for group in ["target", "retain", "control"]:
        m = metrics[group]
        b = baseline[group]
        group_delta = {
            "accuracy": m["accuracy"] - b["accuracy"],
            "mean_signed_margin": m["mean_signed_margin"] - b["mean_signed_margin"],
        }
        fam_delta = {}
        for fam, fm in m.get("per_family", {}).items():
            fb = b.get("per_family", {}).get(fam, {})
            if fb:
                fam_delta[fam] = {
                    "accuracy": fm["accuracy"] - fb["accuracy"],
                    "mean_signed_margin": fm["mean_signed_margin"] - fb["mean_signed_margin"],
                }
        group_delta["per_family"] = fam_delta
        delta[group] = group_delta
    return delta


def _print_step_summary(
    step: int,
    metrics: dict[str, Any],
    delta: dict[str, Any] | None,
) -> None:
    """Print per-checkpoint summary with M_t and ΔM."""
    logger.info(f"\nStep {step}:")
    for group in ["target", "retain", "control"]:
        m = metrics[group]
        logger.info(
            f"  {group:8s}: acc={m['accuracy']:.3f}  "
            f"M={m['mean_signed_margin']:+.2f}",
        )
        if delta:
            d = delta[group]
            logger.info(
                f"           Δacc={d['accuracy']:+.3f}  "
                f"ΔM={d['mean_signed_margin']:+.2f}",
            )
        # direct_visual detail
        dv = m["per_family"].get("direct_visual", {})
        if dv:
            logger.info(
                f"           DV: acc={dv['accuracy']:.3f}  "
                f"M={dv['mean_signed_margin']:+.2f}",
            )
            if delta:
                ddv = delta[group]["per_family"].get("direct_visual", {})
                if ddv:
                    logger.info(
                        f"           DV Δacc={ddv['accuracy']:+.3f}  "
                        f"ΔM={ddv['mean_signed_margin']:+.2f}",
                    )


def _print_decision_analysis(trajectory: list[dict]) -> None:
    """Print decision analysis across the full trajectory."""
    logger.info("\n" + "="*60)
    logger.info("DECISION ANALYSIS")
    logger.info("="*60)

    # Find the first step where target DV accuracy drops below 0.5
    # while retain DV accuracy remains >= 0.98
    selective_steps = []
    for entry in trajectory:
        step = entry["step"]
        if step == 0:
            continue
        m = entry["metrics"]
        target_dv = m["target"]["per_family"].get("direct_visual", {})
        retain_dv = m["retain"]["per_family"].get("direct_visual", {})
        control_dv = m["control"]["per_family"].get("direct_visual", {})
        t_acc = target_dv.get("accuracy", 0)
        r_acc = retain_dv.get("accuracy", 0)
        c_acc = control_dv.get("accuracy", 0)
        if t_acc < 0.5 and r_acc >= 0.98:
            selective_steps.append(step)
            logger.info(
                f"\nStep {step}: SELECTIVE — "
                f"target_dv_acc={t_acc:.3f}, retain_dv_acc={r_acc:.3f}, "
                f"control_dv_acc={c_acc:.3f}"
            )

    if selective_steps:
        logger.info(
            f"\n✓ Selective window found at steps: {selective_steps}"
        )
    else:
        logger.info("\n✗ No selective window found (target DV < 0.5 while retain DV >= 0.98)")

    # Check route-family coupling: do IPN/WN/VTC move together across groups?
    logger.info("\n--- Route-family coupling analysis ---")
    for entry in trajectory:
        step = entry["step"]
        if step == 0:
            continue
        d = entry.get("delta_metrics")
        if d is None:
            continue
        logger.info(f"\nStep {step} ΔM by group × family:")
        for group in ["target", "retain", "control"]:
            fams = d[group].get("per_family", {})
            parts = []
            for fam in BINARY_FAMILIES:
                fd = fams.get(fam, {})
                dm = fd.get("mean_signed_margin", float("nan"))
                parts.append(f"{fam[:3]:>3s}={dm:+6.2f}")
            logger.info(f"  {group:8s}: {'  '.join(parts)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Checkpoint trajectory evaluation")
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        required=True,
        help="Path to pilot_v1 directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment config",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma-separated list of checkpoint steps to evaluate (e.g., '0,1,5,10,15,20,25,35,50'). If not provided, evaluates all checkpoints found in the directory.",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    baseline_path = Path(config["baseline"]["results_path"])
    
    # Load selection
    selection_path = args.pilot_dir / "selection" / "pilot_identity_selection.json"
    with open(selection_path) as f:
        selection = json.load(f)
    
    target_ids = selection["target_identities"]
    retain_ids = selection["retain_identities"]
    control_ids = selection["control_identities"]
    
    logger.info(f"Target identities: {target_ids}")
    logger.info(f"Retain identities: {retain_ids}")
    logger.info(f"Control identities: {control_ids}")
    
    # Load baseline and get diagnostic probe IDs
    baseline_results = load_baseline_results(baseline_path)
    diagnostic_probe_ids = get_diagnostic_probe_ids(
        baseline_results, target_ids, retain_ids, control_ids,
    )
    logger.info(f"Diagnostic subset: {len(diagnostic_probe_ids)} probes")
    
    # Find checkpoints
    checkpoints_dir = args.pilot_dir / "checkpoints"
    checkpoint_dirs = sorted(checkpoints_dir.glob("optimizer_step_*"))
    
    if not checkpoint_dirs:
        logger.error("No checkpoints found")
        sys.exit(1)
    
    # Filter to requested checkpoints if specified
    if args.checkpoints:
        requested_steps = {int(s.strip()) for s in args.checkpoints.split(",")}
        checkpoint_dirs = [
            d for d in checkpoint_dirs
            if int(d.name.split("_")[-1]) in requested_steps
        ]
        logger.info(f"Filtered to {len(checkpoint_dirs)} requested checkpoints: {[d.name for d in checkpoint_dirs]}")
    else:
        logger.info(f"Found {len(checkpoint_dirs)} checkpoints: {[d.name for d in checkpoint_dirs]}")
    
    # Evaluate each checkpoint
    trajectory = []
    baseline_metrics = None  # step-0 metrics for ΔM computation

    for ckpt_dir in checkpoint_dirs:
        step = int(ckpt_dir.name.split("_")[-1])
        checkpoint_name = f"step_{step:03d}"

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating checkpoint: step {step}")
        logger.info(f"{'='*60}")

        results = evaluate_checkpoint(
            ckpt_dir, checkpoint_name, config, diagnostic_probe_ids,
        )

        metrics = compute_metrics(results, target_ids, retain_ids, control_ids)

        # Compute ΔM = M_t - M_0
        delta_metrics = None
        if step == 0:
            baseline_metrics = metrics
        elif baseline_metrics is not None:
            delta_metrics = _compute_delta_metrics(metrics, baseline_metrics)

        trajectory.append({
            "step": step,
            "metrics": metrics,
            "delta_metrics": delta_metrics,
        })

        # Print summary for this checkpoint
        _print_step_summary(step, metrics, delta_metrics)
    
    # Save trajectory
    output_path = args.pilot_dir / "analysis" / "checkpoint_trajectory.json"
    with open(output_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    
    logger.info(f"\nTrajectory saved to {output_path}")
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("CHECKPOINT TRAJECTORY SUMMARY")
    logger.info("="*60)

    for entry in trajectory:
        step = entry["step"]
        metrics = entry["metrics"]
        delta = entry.get("delta_metrics")
        _print_step_summary(step, metrics, delta)

    # Decision analysis
    _print_decision_analysis(trajectory)


if __name__ == "__main__":
    main()
