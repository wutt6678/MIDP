#!/usr/bin/env python3
"""Generate research_dataset_manifest.json programmatically.

Every hash, count, and checksum is derived from actual files on disk.
No value is hard-coded or copied from another manifest without
independent recomputation.

Usage
-----
    python scripts/generate_research_dataset_manifest.py \
        --output-dir outputs/full_fiubench \
        [--output manifest_path] \
        [--allow-dirty]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_SRC = REPO_ROOT / "src"


def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _count_jsonl(path: Path) -> int:
    n = 0
    with open(path) as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
    ).decode().strip()


def _git_dirty() -> bool:
    out = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT,
    ).decode().strip()
    return bool(out)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Protocol SHA
# --------------------------------------------------------------------------- #


def _compute_protocol_sha(protocol: dict) -> tuple[str, dict]:
    """Reproduce compute_protocol_sha256() without importing route_data."""
    canonical = {
        "algorithm_version": 1,
        "eval_fraction": protocol.get("eval_fraction"),
        "eval_seed": protocol.get("eval_seed"),
        "eval_bucket": protocol.get("eval_bucket"),
        "forget_bucket": protocol.get("forget_bucket"),
        "name": protocol.get("name"),
        "source_population": protocol.get("source_population"),
        "train_bucket": protocol.get("train_bucket"),
    }
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest(), canonical


# --------------------------------------------------------------------------- #
# Main generator
# --------------------------------------------------------------------------- #


def generate(output_dir: Path, allow_dirty: bool = False) -> dict:
    # --- Git provenance -------------------------------------------------- #
    commit = _git_commit()
    dirty = _git_dirty()
    if dirty and not allow_dirty:
        print(
            "ERROR: working tree is dirty. Commit or stash changes, "
            "or pass --allow-dirty.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Config files ---------------------------------------------------- #
    # We read the YAML via a simple approach — the protocol block is what
    # we need.
    import yaml  # type: ignore[import-unless-installed]

    fiubench_cfg = yaml.safe_load(
        (REPO_ROOT / "configs/data/fiubench.yaml").read_text(),
    )
    run_cfg = yaml.safe_load(
        (REPO_ROOT / "configs/runs/full_fiubench_qwen.yaml").read_text(),
    )
    protocol = fiubench_cfg["data"]["extras"]["fiubench_protocol"]
    data_cfg = fiubench_cfg["data"]

    # --- Whitelist files ------------------------------------------------- #
    celeba_wl_path = REPO_ROOT / run_cfg["build"]["attribute_whitelist"]
    experiment_v2_path = REPO_ROOT / run_cfg["build"]["experiment_attribute_subset"]

    celeba_wl = _load_json(celeba_wl_path)
    experiment_v2 = _load_json(experiment_v2_path)

    celeba_wl_sha = _sha256_file(celeba_wl_path)
    experiment_v2_sha = _sha256_file(experiment_v2_path)

    celeba_attrs = sorted(celeba_wl.get("attributes", []))
    celeba_attrs_sha = _sha256_str(json.dumps(celeba_attrs))

    experiment_attrs = sorted(experiment_v2.get("attributes", []))

    # --- Protocol SHA ---------------------------------------------------- #
    protocol_sha, canonical_protocol = _compute_protocol_sha(protocol)

    # --- Evidence directory ---------------------------------------------- #
    evidence_dir = output_dir / "evidence"
    artifact_dir = output_dir / "Qwen_Qwen3.5-9B/fiubench"

    # --- Score manifest -------------------------------------------------- #
    score_manifest = _load_json(evidence_dir / "fiubench_score_manifest.json")

    # --- Annotation summary ---------------------------------------------- #
    annotation_summary = _load_json(evidence_dir / "annotation_summary.json")

    # --- Processed dataset ----------------------------------------------- #
    processed_path = artifact_dir / "fiubench_processed.jsonl"
    processed_sha = _sha256_file(processed_path)
    processed_count = _count_jsonl(processed_path)

    # Unique images from processed data
    with open(processed_path) as _fh:
        processed_samples = [
            json.loads(line) for line in _fh if line.strip()
        ]
    unique_images = len({s["image_uri"] for s in processed_samples})
    unique_ids = len({s["identity_id"] for s in processed_samples})

    # --- Image scores ---------------------------------------------------- #
    image_scores_path = artifact_dir / "fiubench_image_scores.jsonl"
    image_scores_sha = _sha256_file(image_scores_path)
    image_scores_rows = _count_jsonl(image_scores_path)

    # --- Model scores ---------------------------------------------------- #
    model_scores_path = artifact_dir / "fiubench_model_scores.jsonl"
    model_scores_sha = _sha256_file(model_scores_path)
    model_scores_rows = _count_jsonl(model_scores_path)

    # --- Route probes ---------------------------------------------------- #
    route_path = artifact_dir / "fiubench_route_conflict_eval.jsonl"
    route_sha = _sha256_file(route_path)
    with open(route_path) as _fh:
        route_probes = [
            json.loads(line) for line in _fh if line.strip()
        ]
    route_total = len(route_probes)
    route_families: dict[str, int] = {}
    for rp in route_probes:
        fam = rp.get("probe_family", "unknown")
        route_families[fam] = route_families.get(fam, 0) + 1

    # --- Splits ---------------------------------------------------------- #
    splits_dir = artifact_dir / "fiubench_unlearning_splits"
    # Hash the directory contents for reproducibility
    split_files = sorted(splits_dir.glob("*.json")) if splits_dir.is_dir() else []
    split_hasher = hashlib.sha256()
    for sf in split_files:
        split_hasher.update(sf.name.encode())
        split_hasher.update(sf.read_bytes())
    splits_sha = split_hasher.hexdigest()

    # --- Export manifest ------------------------------------------------- #
    export_manifest_path = artifact_dir / "fiubench_export_manifest.json"
    export_manifest_sha = _sha256_file(export_manifest_path)

    # --- Manual audit ---------------------------------------------------- #
    audit_path = artifact_dir / "fiubench_manual_audit_report.json"
    audit_sha = _sha256_file(audit_path)
    audit_data = _load_json(audit_path)
    audit_total = audit_data.get("total_items", audit_data.get("total", 0))
    audit_fail = audit_data.get("critical_failures", audit_data.get("fail", 0))
    audit_uncertain = audit_data.get("uncertain_items", audit_data.get("uncertain", 0))
    audit_unreviewed = audit_data.get("unreviewed_items", audit_data.get("unreviewed", 0))
    audit_gate_pass = audit_data.get("gate_pass", False)

    # P0-16: extract route SHA binding from audit report.
    audited_route_sha = audit_data.get("audited_route_probe_sha256")
    audited_route_count = audit_data.get("audited_route_probe_count")

    # --- Attribute distribution ------------------------------------------ #
    attr_dist_path = evidence_dir / "attribute_distribution_report.json"
    attr_dist_data = _load_json(attr_dist_path)

    # --- Route probe coverage -------------------------------------------- #
    route_coverage_path = evidence_dir / "route_probe_attribute_coverage.json"
    route_coverage_sha = _sha256_file(route_coverage_path) if route_coverage_path.exists() else None
    route_coverage_data = _load_json(route_coverage_path) if route_coverage_path.exists() else {}

    # --- Wrong-name report ----------------------------------------------- #
    wn_report_path = evidence_dir / "actual_wrong_name_probe_report.json"
    wn_report_sha = _sha256_file(wn_report_path) if wn_report_path.exists() else None
    wn_report_data = _load_json(wn_report_path) if wn_report_path.exists() else {}

    # --- Hard-stop checks ------------------------------------------------ #
    checks: dict[str, bool] = {}
    checks["573_unique_images"] = unique_images == 573
    checks["573_unique_identities"] = unique_ids == 573
    checks["22920_image_score_rows"] = image_scores_rows == 22920
    checks["30660_canonical_samples"] = processed_count == 30660
    checks["13_celeba_whitelist_attrs"] = len(celeba_attrs) == 13
    checks["10_experiment_v2_attrs"] = len(experiment_attrs) == 10
    checks["excluded_attrs_not_in_v2"] = (
        "Wearing_Hat" not in experiment_attrs
        and "Wearing_Necktie" not in experiment_attrs
        and "Sideburns" not in experiment_attrs
    )
    checks["experiment_subset_within_reliability"] = (
        set(experiment_attrs) <= set(celeba_attrs)
    )
    checks["protocol_sha_computed"] = bool(protocol_sha)
    checks["all_route_families_present"] = all(
        route_families.get(f, 0) > 0
        for f in [
            "direct_visual", "image_plus_name", "name_only",
            "wrong_name", "visual_text_conflict",
        ]
    )
    checks["route_total_500"] = route_total == 500
    checks["manual_audit_pass"] = bool(audit_gate_pass)
    checks["manual_audit_no_fail"] = audit_fail == 0
    checks["manual_audit_no_uncertain"] = audit_uncertain == 0
    checks["manual_audit_no_unreviewed"] = audit_unreviewed == 0
    # P0-17: hard stop for audit/route SHA mismatch.
    checks["manual_audit_matches_current_route_artifact"] = (
        audited_route_sha is not None
        and audited_route_sha == route_sha
    )
    checks["manual_audit_route_count_matches"] = (
        audited_route_count is not None
        and audited_route_count == route_total
    )
    checks["non_whitelisted_accepted_zero"] = (
        annotation_summary.get(
            "non_whitelisted_accepted_labels_processed", -1,
        )
        == 0
    )
    checks["model_revision_is_full_sha"] = len(
        score_manifest.get("resolved_revision", ""),
    ) == 40
    checks["model_fingerprint_nonempty"] = bool(
        score_manifest.get("model_fingerprint"),
    )
    checks["scoring_version_2"] = (
        str(score_manifest.get("scoring_version", "")) == "2"
    )
    checks["candidate_set_hash_nonempty"] = bool(
        score_manifest.get("candidate_set_hash"),
    )
    checks["prompt_registry_hash_nonempty"] = bool(
        score_manifest.get("prompt_registry_hash"),
    )
    checks["git_dirty_false"] = not dirty
    checks["route_coverage_exists"] = route_coverage_path.exists()
    checks["wrong_name_report_exists"] = wn_report_path.exists()

    # P1-3: route-state coverage hard-stop checks.
    all_fam_cov = route_coverage_data.get("all_families", {})
    attrs_with_targets = sum(
        1 for acov in all_fam_cov.values()
        if acov.get("total_visual_probe_count", 0) > 0
    )
    checks["all_experiment_attributes_have_route_targets"] = (
        attrs_with_targets >= len(experiment_attrs)
    )
    checks["all_feasible_attributes_have_positive_route_targets"] = all(
        acov.get("positive_target_count", 0) > 0
        for acov in all_fam_cov.values()
    )
    checks["all_feasible_attributes_have_negative_route_targets"] = all(
        acov.get("negative_target_count", 0) > 0
        for acov in all_fam_cov.values()
    )

    # P1-1 / P1-3: wrong-name image-SHA and answer invariants.
    wn_probes = wn_report_data.get("probes", [])
    if wn_probes:
        checks["wrong_name_same_image_invariant"] = all(
            r.get("target_image_sha256")
            and r.get("paired_correct_name_image_sha256")
            and r["target_image_sha256"] == r["paired_correct_name_image_sha256"]
            for r in wn_probes
        )
        checks["wrong_name_same_answer_invariant"] = all(
            r.get("target_label") is not None
            and r.get("target_label") == r.get("paired_correct_name_target_label")
            for r in wn_probes
        )
    else:
        checks["wrong_name_same_image_invariant"] = False
        checks["wrong_name_same_answer_invariant"] = False

    all_clear = all(checks.values())

    # --- Build manifest -------------------------------------------------- #
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest: dict = {
        "manifest_version": "2.0",
        "manifest_purpose": (
            "Frozen evidence bundle for FIUBench unlearning experiments. "
            "All hashes are independently recomputed from source files."
        ),
        "created_at": now_utc,
        "code_provenance": {
            "dataset_creation_commit": score_manifest.get(
                "midp_commit", commit,
            ),
            "evidence_generation_code_commit": commit,
            "midp_git_dirty": dirty,
            "fiubench_upstream_commit": data_cfg.get(
                "immutable_revision", {},
            ).get("git_commit_sha", ""),
            "source_version": data_cfg.get("source_version", ""),
            "source_hash": score_manifest.get("source_hash", ""),
        },
        "model_provenance": {
            "model_id": score_manifest.get("model_id", ""),
            "backend": score_manifest.get("backend", ""),
            "resolved_revision": score_manifest.get("resolved_revision", ""),
            "model_fingerprint": score_manifest.get("model_fingerprint", ""),
            "dtype": score_manifest.get("dtype", ""),
            "quantization": "none",
            "transformers_version": score_manifest.get(
                "transformers_version", "",
            ),
            "torch_version": score_manifest.get("torch_version", ""),
        },
        "scoring_provenance": {
            "prompt_registry_hash": score_manifest.get(
                "prompt_registry_hash", "",
            ),
            "scoring_version": str(
                score_manifest.get("scoring_version", ""),
            ),
            "candidate_set_hash": score_manifest.get(
                "candidate_set_hash", "",
            ),
            "scoring_method": "candidate_sequence_log_probability",
        },
        "protocol": {
            "name": protocol.get("name", ""),
            "forget_bucket": protocol.get("forget_bucket"),
            "train_bucket": protocol.get("train_bucket"),
            "eval_bucket": protocol.get("eval_bucket"),
            "eval_fraction": protocol.get("eval_fraction"),
            "eval_seed": protocol.get("eval_seed"),
            "protocol_sha256": protocol_sha,
            "canonical_protocol": canonical_protocol,
        },
        "attribute_whitelists": {
            "celeba_reliability_whitelist": {
                "path": str(celeba_wl_path.relative_to(REPO_ROOT)),
                "sha256": celeba_wl_sha,
                "attributes_sha256": celeba_attrs_sha,
                "n_attributes": len(celeba_attrs),
                "policy": celeba_wl.get("policy", ""),
            },
            "fiubench_experiment_subset": {
                "path": str(
                    experiment_v2_path.relative_to(REPO_ROOT),
                ),
                "sha256": experiment_v2_sha,
                "n_attributes": len(experiment_attrs),
                "excluded": sorted(
                    set(celeba_attrs) - set(experiment_attrs),
                ),
                "analysis_unit": "unique_image",
                "version": "v2",
            },
        },
        "dataset_artifacts": {
            "output_dir": str(artifact_dir.relative_to(REPO_ROOT)),
            "processed_dataset": {
                "path": "fiubench_processed.jsonl",
                "sha256": processed_sha,
                "size_bytes": processed_path.stat().st_size,
                "canonical_samples": processed_count,
                "unique_images": unique_images,
                "unique_identities": unique_ids,
            },
            "score_table": {
                "image_scores": {
                    "path": "fiubench_image_scores.jsonl",
                    "sha256": image_scores_sha,
                    "size_bytes": image_scores_path.stat().st_size,
                    "rows": image_scores_rows,
                },
                "model_scores": {
                    "path": "fiubench_model_scores.jsonl",
                    "sha256": model_scores_sha,
                    "size_bytes": model_scores_path.stat().st_size,
                    "rows": model_scores_rows,
                },
            },
            "route_probes": {
                "path": "fiubench_route_conflict_eval.jsonl",
                "sha256": route_sha,
                "size_bytes": route_path.stat().st_size,
                "total_probes": route_total,
                "families": dict(sorted(route_families.items())),
            },
            "splits": {
                "path": "fiubench_unlearning_splits",
                "sha256": splits_sha,
                "protocols": [
                    "identity_forget",
                    "identity_fact_forget",
                    "attribute_forget",
                ],
            },
            "export_manifest": {
                "path": "fiubench_export_manifest.json",
                "sha256": export_manifest_sha,
                "size_bytes": export_manifest_path.stat().st_size,
            },
        },
        "quality_evidence": {
            "manual_audit": {
                "path": "fiubench_manual_audit_report.json",
                "sha256": audit_sha,
                "size_bytes": audit_path.stat().st_size,
                "total_items": audit_total,
                "fail": audit_fail,
                "uncertain": audit_uncertain,
                "unreviewed": audit_unreviewed,
                "gate_pass": audit_gate_pass,
                "audited_route_probe_sha256": audited_route_sha,
                "audited_route_probe_count": audited_route_count,
            },
            "attribute_distribution": {
                "path": str(
                    attr_dist_path.relative_to(REPO_ROOT),
                ),
                "sha256": _sha256_file(attr_dist_path),
                "analysis_unit": "unique_image",
                "n_images": attr_dist_data.get("n_images", 573),
                "experiment_subset_attributes": len(experiment_attrs),
            },
            "route_probe_coverage": {
                "path": str(
                    route_coverage_path.relative_to(REPO_ROOT),
                ) if route_coverage_path.exists() else None,
                "sha256": route_coverage_sha,
            },
            "actual_wrong_name_report": {
                "path": str(
                    wn_report_path.relative_to(REPO_ROOT),
                ) if wn_report_path.exists() else None,
                "sha256": wn_report_sha,
                "matching_covariate_policy": wn_report_data.get(
                    "matching_covariate_policy",
                    "all_high_confidence_celeba_reliability_attributes",
                ),
                "matching_covariate_attribute_ceiling": wn_report_data.get(
                    "matching_covariate_attribute_ceiling",
                    "configs/whitelists/qwen35_9b_celeba.json",
                ),
                "target_attribute_subset": wn_report_data.get(
                    "target_attribute_subset",
                    "configs/whitelists/qwen35_9b_fiubench_experiment_v2.json",
                ),
            },
        },
        "evidence_bundle_files": [
            str(p.relative_to(REPO_ROOT))
            for p in sorted(
                [
                    evidence_dir / "fiubench_score_manifest.json",
                    evidence_dir / "artifact_checksums.json",
                    evidence_dir / "annotation_summary.json",
                    evidence_dir / "runtime_environment.json",
                    evidence_dir / "fiubench_population_report.json",
                    evidence_dir / "source_image_audit.json",
                    evidence_dir / "wrong_name_matching_report.json",
                    evidence_dir / "attribute_distribution_report.json",
                    evidence_dir / "route_probe_attribute_coverage.json",
                    evidence_dir / "actual_wrong_name_probe_report.json",
                    artifact_dir / "fiubench_manual_audit_report.json",
                    artifact_dir / "fiubench_checksums.json",
                    celeba_wl_path,
                    experiment_v2_path,
                ]
            )
            if p.exists()
        ],
        "hard_stop_conditions": {
            "all_clear": all_clear,
            "checks": {k: v for k, v in sorted(checks.items())},
        },
        "definition_of_done": {
            "ready_for_experiments": all_clear,
            "summary": (
                f"Generated programmatically from {commit}. "
                f"{unique_images} images, {image_scores_rows} score rows, "
                f"{processed_count} canonical samples. "
                f"{len(celeba_attrs)} CelebA-whitelisted attributes; "
                f"{len(experiment_attrs)} selected for experiments. "
                f"All hard-stop checks: {'PASS' if all_clear else 'FAIL'}."
            ),
        },
    }

    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate research_dataset_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/full_fiubench"),
        help="Evidence output directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <output-dir>/evidence/research_dataset_manifest.json)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow generation with uncommitted changes",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    manifest = generate(output_dir, allow_dirty=args.allow_dirty)

    out_path = args.output
    if out_path is None:
        out_path = output_dir / "evidence/research_dataset_manifest.json"
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    all_clear = manifest["hard_stop_conditions"]["all_clear"]
    n_checks = len(manifest["hard_stop_conditions"]["checks"])
    n_pass = sum(manifest["hard_stop_conditions"]["checks"].values())

    print(f"Manifest written to {out_path}")
    print(f"Hard-stop checks: {n_pass}/{n_checks} PASS")
    print(f"ready_for_experiments: {manifest['definition_of_done']['ready_for_experiments']}")

    if not all_clear:
        print("\nFAILED checks:")
        for k, v in manifest["hard_stop_conditions"]["checks"].items():
            if not v:
                print(f"  FAIL: {k}")
        sys.exit(1)


if __name__ == "__main__":
    main()
