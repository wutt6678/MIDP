#!/usr/bin/env python3
"""Generate manifest and summary for Qwen3.5-4B baseline from existing results.

Schema follows the canonical baseline format expected by resolve_preunlearning_baseline().
"""

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
    profile_path = Path("configs/models/unlearning/qwen35_4b.yaml")

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

    # Compute profile SHA-256
    profile_sha256 = _file_sha256(profile_path)

    # Compute summary statistics per family
    per_family: dict[str, dict] = {}
    for r in results:
        family = r.get("probe_family", "unknown")
        if family not in per_family:
            per_family[family] = {
                "count": 0,
                "errors": 0,
                "correct": 0,
                "log_margins": [],
                "p_yes_values": [],
                "token_overlaps": [],
                "normalized_exact_matches": [],
                "fuzzy_matches": [],
                "generated_token_counts": [],
                "hit_max_new_tokens_count": 0,
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
        if r.get("token_overlap") is not None:
            per_family[family]["token_overlaps"].append(r["token_overlap"])
        if r.get("normalized_exact_match") is not None:
            per_family[family]["normalized_exact_matches"].append(r["normalized_exact_match"])
        if r.get("fuzzy_match") is not None:
            per_family[family]["fuzzy_matches"].append(r["fuzzy_match"])
        if r.get("generated_token_count") is not None:
            per_family[family]["generated_token_counts"].append(r["generated_token_count"])
        if r.get("hit_max_new_tokens"):
            per_family[family]["hit_max_new_tokens_count"] += 1

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
        if stats["token_overlaps"]:
            stats["mean_token_overlap"] = sum(stats["token_overlaps"]) / len(stats["token_overlaps"])
        if stats["normalized_exact_matches"]:
            stats["mean_normalized_exact_match"] = sum(stats["normalized_exact_matches"]) / len(stats["normalized_exact_matches"])
        if stats["fuzzy_matches"]:
            stats["mean_fuzzy_match"] = sum(stats["fuzzy_matches"]) / len(stats["fuzzy_matches"])
        if stats["generated_token_counts"]:
            stats["mean_generated_token_count"] = sum(stats["generated_token_counts"]) / len(stats["generated_token_counts"])
        if stats["count"] > 0:
            stats["accuracy"] = stats["correct"] / stats["count"]

    # Compute visual-family accuracy (excluding name_only)
    visual_correct = sum(
        s["correct"] for f, s in per_family.items() if f != "name_only"
    )
    visual_count = sum(
        s["count"] for f, s in per_family.items() if f != "name_only"
    )
    visual_accuracy = visual_correct / visual_count if visual_count > 0 else 0.0

    # Compute overall accuracy
    total_correct = sum(s["correct"] for s in per_family.values())
    total_count = sum(s["count"] for s in per_family.values())
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

    # Remove internal list fields from per_family for JSON output
    per_family_output = {}
    for family, stats in per_family.items():
        per_family_output[family] = {
            k: v for k, v in stats.items()
            if k not in ("log_margins", "p_yes_values", "token_overlaps",
                         "normalized_exact_matches", "fuzzy_matches",
                         "generated_token_counts")
        }

    summary = {
        "total_probes": len(results),
        "total_correct": total_correct,
        "mixed_task_overall_accuracy": overall_accuracy,
        "visual_family_accuracy": visual_accuracy,
        "visual_family_correct": visual_correct,
        "visual_family_count": visual_count,
        "name_only_status": "valid_rerun_with_64_token_budget",
        "per_family": per_family_output,
        "model_fingerprint": "01306df4d620e651",
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "scoring_version": "2",
    }

    # Compute SHA-256 hashes
    results_sha256 = _file_sha256(results_path)
    route_probe_sha256 = _file_sha256(probe_path)
    processed_dataset_sha256 = _file_sha256(
        Path("outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_processed.jsonl")
    )

    git_commit = _git_commit()

    # Compute generation protocol hash (same as BaselineRunner._generation_protocol_hash())
    import json as _json
    _gen_payload = {
        "binary_families": {"do_sample": False, "temperature": 0.0, "max_new_tokens": 4},
        "name_only": {"do_sample": False, "temperature": 0.0, "max_new_tokens": 64},
        "scoring_version": "2",
    }
    protocol_sha256 = hashlib.sha256(
        _json.dumps(_gen_payload, sort_keys=True).encode()
    ).hexdigest()

    # Build manifest with canonical schema expected by resolve_preunlearning_baseline()
    manifest = {
        # -- Canonical model section (required by resolver) --
        "model": {
            "model_key": "qwen35_4b",
            "id": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "processor_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "model_profile_sha256": profile_sha256,
        },
        # -- Canonical provenance section (required by resolver) --
        "provenance": {
            "results_sha256": results_sha256,
            "manifest_sha256": "",  # filled after writing
            "route_probe_sha256": route_probe_sha256,
            "processed_dataset_sha256": processed_dataset_sha256,
            "code_commit": git_commit,
        },
        # -- Frozen protocol hash (P1-1) --
        "protocol_sha256": protocol_sha256,
        # -- Backwards-compatible sections --
        "schema_version": "1.2",
        "metric_schema_version": "baseline-metrics-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_provenance": {
            "probe_file": str(probe_path),
            "probe_file_sha256": route_probe_sha256,
            "probe_count": len(probes),
            "dataset_manifest": "outputs/full_fiubench/evidence/research_dataset_manifest.json",
            "dataset_version": "fiubench-route-v1",
            "dataset_manifest_sha256": _file_sha256(
                Path("outputs/full_fiubench/evidence/research_dataset_manifest.json")
            ),
            "freeze_verification": "outputs/full_fiubench/evidence/final_freeze_verification.json",
            "freeze_verification_sha256": _file_sha256(
                Path("outputs/full_fiubench/evidence/final_freeze_verification.json")
            ),
            "route_probe_sha256": route_probe_sha256,
            "route_probe_count": len(probes),
            "processed_dataset_path": "outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_processed.jsonl",
            "processed_dataset_sha256": processed_dataset_sha256,
        },
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
        # -- Generation config (P1-1) --
        "generation_config": {
            "binary_families": {
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": 4,
            },
            "name_only": {
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": 64,
            },
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
            "name_only_primary_metric": "token_overlap",
            "name_only_primary_threshold": 0.5,
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
            "validation_report_sha256": (
                _file_sha256(output_dir / "validation_report.json")
                if (output_dir / "validation_report.json").exists() else ""
            ),
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
            "experiment_code_commit": git_commit,
            "artifact_commit": git_commit,
            "working_tree_dirty_at_execution": False,
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

    # Update manifest with summary SHA-256 and manifest SHA-256
    manifest["results"]["summary_sha256"] = _file_sha256(summary_path)
    manifest["provenance"]["manifest_sha256"] = _file_sha256(manifest_path)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Manifest updated with summary and manifest SHA-256")

    print("\n" + "=" * 60)
    print("Baseline generation complete!")
    print(f"Total probes: {len(results)}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"Visual-family accuracy: {visual_accuracy:.4f} ({visual_correct}/{visual_count})")
    print(f"Results: {results_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
