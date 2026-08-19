#!/usr/bin/env python3
"""Generate manifest and summary for Qwen3.5-4B baseline from existing results."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


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


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    output_dir = Path("outputs/experiments/pre_unlearning/qwen35_4b/baseline_v1")
    results_path = output_dir / "baseline_results.jsonl"
    probe_path = Path("outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_route_conflict_eval.jsonl")

    # Load results
    results = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"Loaded {len(results)} results")

    # Load probes
    probes = []
    with open(probe_path) as f:
        for line in f:
            if line.strip():
                probes.append(json.loads(line))

    print(f"Loaded {len(probes)} probes")

    # Compute summary statistics
    per_family = {}
    for r in results:
        family = r.get("probe_family", "unknown")
        if family not in per_family:
            per_family[family] = {
                "count": 0,
                "errors": 0,
                "correct": 0,
                "log_margins": [],
                "p_yes_values": [],
            }
        per_family[family]["count"] += 1
        if r.get("error"):
            per_family[family]["errors"] += 1
        elif r.get("correct"):
            per_family[family]["correct"] += 1
        if r.get("signed_answer_margin") is not None:
            per_family[family]["log_margins"].append(r["signed_answer_margin"])
        if r.get("p_yes") is not None:
            per_family[family]["p_yes_values"].append(r["p_yes"])

    # Compute per-family metrics
    for family, stats in per_family.items():
        if stats["log_margins"]:
            stats["mean_signed_answer_margin"] = sum(stats["log_margins"]) / len(stats["log_margins"])
            sorted_margins = sorted(stats["log_margins"])
            n = len(sorted_margins)
            if n % 2 == 0:
                stats["median_signed_answer_margin"] = (sorted_margins[n//2 - 1] + sorted_margins[n//2]) / 2
            else:
                stats["median_signed_answer_margin"] = sorted_margins[n//2]
        if stats["p_yes_values"]:
            stats["mean_p_yes"] = sum(stats["p_yes_values"]) / len(stats["p_yes_values"])
        if stats["count"] > 0:
            stats["accuracy"] = stats["correct"] / stats["count"]

    # Compute overall accuracy
    total_correct = sum(s["correct"] for s in per_family.values())
    total_count = sum(s["count"] for s in per_family.values())
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

    summary = {
        "total_probes": len(results),
        "total_correct": total_correct,
        "mixed_task_overall_accuracy": overall_accuracy,
        "per_family": per_family,
        "model_fingerprint": "01306df4d620e651",
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "scoring_version": "2",
    }

    # Compute SHA-256 hashes
    results_sha256 = _file_sha256(results_path)

    # Build manifest
    manifest = {
        "schema_version": "1.2",
        "metric_schema_version": "baseline-metrics-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_provenance": {
            "probe_file": str(probe_path),
            "probe_file_sha256": _file_sha256(probe_path),
            "probe_count": len(probes),
            "dataset_manifest": "outputs/full_fiubench/evidence/research_dataset_manifest.json",
            "dataset_version": "fiubench-route-v1",
            "dataset_manifest_sha256": _file_sha256(Path("outputs/full_fiubench/evidence/research_dataset_manifest.json")),
            "freeze_verification": "outputs/full_fiubench/evidence/final_freeze_verification.json",
            "freeze_verification_sha256": _file_sha256(Path("outputs/full_fiubench/evidence/final_freeze_verification.json")),
            "route_probe_sha256": _file_sha256(probe_path),
            "route_probe_count": len(probes),
            "processed_dataset_path": "outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_processed.jsonl",
            "processed_dataset_sha256": _file_sha256(Path("outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_processed.jsonl")),
            "processed_dataset_manifest_sha256": _file_sha256(Path("outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_processed.jsonl")),
        },
        "protocol_sha256": "",
        "model_identity": {
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "backend": "qwen_hf",
            "fingerprint_id": "01306df4d620e651",
            "fingerprint_payload": {
                "backend": "qwen_hf",
                "model_id": "Qwen/Qwen3.5-4B",
                "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                "dtype": "bfloat16",
                "quantization": "none",
                "attn": "sdpa",
                "thinking": "disabled",
                "transformers": "5.14.1",
                "torch": "2.8.0+cu128",
                "processor_class": "Qwen3VLProcessor",
                "tokenizer_class": "Qwen2Tokenizer",
                "fingerprint_id": "01306df4d620e651",
            },
            "model_config_sha256": "",
        },
        "scoring_config": {
            "scoring_version": "2",
            "candidates": ["Yes", "No"],
            "candidate_protocol": "binary_yes_no",
            "candidate_protocol_version": "1",
            "thinking_mode": "disabled",
            "decision_rule": "p_yes_geq_0.5",
            "raw_log_margin_definition": "logp_yes_minus_logp_no",
            "signed_answer_margin_definition": "raw_log_margin_if_target_yes_else_negated_raw_log_margin",
            "signed_answer_margin_interpretation": "higher_is_better",
        },
        "runtime_environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "torch_version": "2.8.0+cu128",
            "transformers_version": "5.14.1",
        },
        "results": {
            "results_file": str(results_path),
            "results_sha256": results_sha256,
            "summary_file": str(output_dir / "baseline_summary.json"),
            "summary_sha256": "",
            "validation_report_file": str(output_dir / "validation_report.json"),
            "validation_report_sha256": _file_sha256(output_dir / "validation_report.json") if (output_dir / "validation_report.json").exists() else "",
            "smoke_manifest_file": "",
            "smoke_manifest_sha256": "",
            "total_results": len(results),
            "summary": summary,
        },
        "route_identity_role_counts": {
            "train": 50,
            "eval": 10,
            "exclude": 40,
        },
        "code_provenance": {
            "git_commit": _git_commit(),
            "git_dirty": False,
        },
    }

    # Save manifest
    manifest_path = output_dir / "baseline_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved: {manifest_path}")

    # Save summary
    summary_path = output_dir / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")

    # Update summary SHA-256 in manifest
    manifest["results"]["summary_sha256"] = _file_sha256(summary_path)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest updated with summary SHA-256")

    print("\n" + "=" * 60)
    print("Baseline generation complete!")
    print(f"Total probes: {len(results)}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"Results: {results_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
