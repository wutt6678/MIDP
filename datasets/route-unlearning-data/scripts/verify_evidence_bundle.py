#!/usr/bin/env python3
"""Verify internal consistency of the FIUBench compact evidence bundle.

This script cross-checks the six evidence files in
``outputs/full_fiubench/evidence/`` against each other and against the
actual artifacts in ``outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/``.

Checks performed:

1. ``annotation_summary.accepted_labels_processed`` matches the count
   recomputed from ``fiubench_processed.jsonl``.
2. ``population_report.accepted_labels_processed`` matches
   ``annotation_summary.accepted_labels_processed``.
3. ``score_manifest.midp_commit`` matches the commit recorded in all
   other evidence files.
4. Artifact checksum records match freshly recomputed SHA-256 values.
5. Key structural invariants (573 images, 22 920 score rows, etc.).

Exit code 0 on success, 1 on any failure.

Usage:
    python scripts/verify_evidence_bundle.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = REPO / "outputs" / "full_fiubench" / "Qwen_Qwen3.5-9B" / "fiubench"
EVIDENCE_DIR = REPO / "outputs" / "full_fiubench" / "evidence"

# ── helpers ──────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _recount_accepted_labels(processed_path: Path) -> int:
    """Recount non-null labels directly from the processed JSONL."""
    count = 0
    for line in processed_path.read_text().splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        for obs in doc.get("visual_attributes", {}).values():
            if obs.get("label") is not None:
                count += 1
    return count


# ── main verification ────────────────────────────────────────────────

def verify() -> int:
    failures: list[str] = []
    passes: list[str] = []

    # Load evidence files.
    evidence_files = [
        "annotation_summary.json",
        "fiubench_population_report.json",
        "fiubench_score_manifest.json",
        "source_image_audit.json",
        "runtime_environment.json",
        "artifact_checksums.json",
    ]
    missing = [f for f in evidence_files if not (EVIDENCE_DIR / f).exists()]
    if missing:
        for m in missing:
            failures.append(f"MISSING evidence file: {m}")
        _report(passes, failures)
        return 1

    summary = _load_json(EVIDENCE_DIR / "annotation_summary.json")
    pop_report = _load_json(EVIDENCE_DIR / "fiubench_population_report.json")
    score_manifest = _load_json(EVIDENCE_DIR / "fiubench_score_manifest.json")
    image_audit = _load_json(EVIDENCE_DIR / "source_image_audit.json")
    runtime_env = _load_json(EVIDENCE_DIR / "runtime_environment.json")
    checksums = _load_json(EVIDENCE_DIR / "artifact_checksums.json")

    # ── Check 1: accepted_labels_processed matches recomputed count ──
    processed_path = ARTIFACT_DIR / "fiubench_processed.jsonl"
    if processed_path.exists():
        recomputed = _recount_accepted_labels(processed_path)
        declared = summary.get("accepted_labels_processed")
        if recomputed == declared:
            passes.append(f"accepted_labels_processed matches recomputed ({declared})")
        else:
            failures.append(
                f"accepted_labels_processed mismatch: "
                f"summary declares {declared}, recomputed {recomputed}"
            )
    else:
        failures.append(f"Cannot verify accepted_labels: {processed_path} missing")

    # ── Check 2: population_report matches annotation_summary ──
    pop_accepted = pop_report.get("accepted_labels_processed")
    sum_accepted = summary.get("accepted_labels_processed")
    if pop_accepted == sum_accepted:
        passes.append(f"population_report.accepted_labels_processed == annotation_summary ({sum_accepted})")
    else:
        failures.append(
            f"accepted_labels mismatch between reports: "
            f"population_report={pop_accepted}, annotation_summary={sum_accepted}"
        )

    # ── Check 3: all evidence files share the same midp_commit ──
    commits = {
        "annotation_summary": summary.get("midp_commit"),
        "population_report": pop_report.get("midp_commit"),
        "score_manifest": score_manifest.get("midp_commit"),
        "source_image_audit": image_audit.get("midp_commit"),
        "runtime_environment": runtime_env.get("midp_commit"),
        "artifact_checksums": checksums.get("computed_at_commit"),
    }
    unique_commits = set(commits.values())
    if len(unique_commits) == 1:
        passes.append(f"All evidence files share midp_commit={commits['annotation_summary'][:12]}…")
    else:
        for name, sha in commits.items():
            failures.append(f"midp_commit inconsistency: {name} = {sha}")

    # Also check score_manifest's configured/resolved revision.
    manifest_commit = score_manifest.get("midp_commit")
    configured = score_manifest.get("configured_revision")
    resolved = score_manifest.get("resolved_revision")
    if resolved and manifest_commit and not resolved.startswith(manifest_commit[:12]):
        # The resolved_revision is a Qwen model revision, not the MIDP code commit.
        # These are different things — only flag if they look like they should match.
        pass  # Qwen revision ≠ MIDP commit; no check needed.

    # ── Check 4: artifact checksums verify ──
    recorded_artifacts = checksums.get("artifacts", {})
    for name, info in recorded_artifacts.items():
        artifact_path = ARTIFACT_DIR / name
        if not artifact_path.exists():
            failures.append(f"Checksum artifact missing: {name}")
            continue
        actual_sha = _sha256_file(artifact_path)
        recorded_sha = info.get("sha256", "")
        if actual_sha == recorded_sha:
            passes.append(f"SHA-256 verified: {name}")
        else:
            failures.append(
                f"SHA-256 MISMATCH for {name}: "
                f"recorded={recorded_sha[:16]}… actual={actual_sha[:16]}…"
            )
        actual_size = artifact_path.stat().st_size
        recorded_size = info.get("size_bytes")
        if actual_size == recorded_size:
            passes.append(f"Size verified: {name} ({actual_size} bytes)")
        else:
            failures.append(
                f"Size mismatch for {name}: "
                f"recorded={recorded_size}, actual={actual_size}"
            )

    # ── Check 5: structural invariants ──
    checks_573 = [
        ("source_identity_rows", summary.get("source_identity_rows")),
        ("unique_images (summary)", summary.get("unique_images")),
        ("unique_image_uris (pop)", pop_report.get("unique_image_uris")),
        ("unique_image_sha256 (pop)", pop_report.get("unique_image_sha256")),
        ("source_identities (audit)", image_audit.get("source_identities")),
        ("unique_images (audit)", image_audit.get("unique_images")),
    ]
    for label, val in checks_573:
        if val == 573:
            passes.append(f"{label} = 573")
        else:
            failures.append(f"{label} expected 573, got {val}")

    checks_22920 = [
        ("raw_score_rows (summary)", summary.get("raw_score_rows")),
        ("raw_score_rows (pop)", pop_report.get("raw_score_rows")),
        ("score_rows (audit)", image_audit.get("score_rows")),
    ]
    for label, val in checks_22920:
        if val == 22920:
            passes.append(f"{label} = 22920")
        else:
            failures.append(f"{label} expected 22920, got {val}")

    checks_30660 = [
        ("canonical_samples (summary)", summary.get("canonical_samples")),
        ("canonical_samples_total (pop)", pop_report.get("canonical_samples_total")),
    ]
    for label, val in checks_30660:
        if val == 30660:
            passes.append(f"{label} = 30660")
        else:
            failures.append(f"{label} expected 30660, got {val}")

    checks_1226400 = [
        ("observations (summary)", summary.get("observations")),
        ("observations (pop)", pop_report.get("observations")),
    ]
    for label, val in checks_1226400:
        if val == 1226400:
            passes.append(f"{label} = 1226400")
        else:
            failures.append(f"{label} expected 1226400, got {val}")

    # Score completeness.
    if image_audit.get("score_completeness_ok") is True:
        passes.append("score_completeness_ok = true")
    else:
        failures.append("score_completeness_ok is not true")

    # Non-whitelisted accepted labels must be 0.
    nwl = summary.get("non_whitelisted_accepted_labels_processed")
    if nwl == 0:
        passes.append("non_whitelisted_accepted_labels_processed = 0")
    else:
        failures.append(f"non_whitelisted_accepted_labels_processed expected 0, got {nwl}")

    # Sample type breakdown sums to canonical total.
    orig = pop_report.get("canonical_original_samples", 0)
    para = pop_report.get("canonical_paraphrase_samples", 0)
    pert = pop_report.get("canonical_perturbed_samples", 0)
    sample_sum = orig + para + pert
    if sample_sum == pop_report.get("canonical_samples_total", 0):
        passes.append(f"sample type breakdown sums to canonical total ({sample_sum})")
    else:
        failures.append(
            f"sample type breakdown mismatch: "
            f"{orig}+{para}+{pert}={sample_sum} ≠ {pop_report.get('canonical_samples_total')}"
        )

    # Split counts sum to canonical total.
    split_sum = sum(pop_report.get("split_counts", {}).values())
    if split_sum == pop_report.get("canonical_samples_total", 0):
        passes.append(f"split counts sum to canonical total ({split_sum})")
    else:
        failures.append(
            f"split counts mismatch: "
            f"{split_sum} ≠ {pop_report.get('canonical_samples_total')}"
        )

    # ── Report ──
    _report(passes, failures)
    return 1 if failures else 0


def _report(passes: list[str], failures: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"Evidence Bundle Consistency Check")
    print(f"{'=' * 60}")
    print(f"\nPassed: {len(passes)}")
    for p in passes:
        print(f"  ✓ {p}")
    if failures:
        print(f"\nFailed: {len(failures)}")
        for f in failures:
            print(f"  ✗ {f}")
        print(f"\n*** EVIDENCE BUNDLE VERIFICATION FAILED ***")
    else:
        print(f"\n*** ALL CHECKS PASSED — EVIDENCE BUNDLE IS CONSISTENT ***")


if __name__ == "__main__":
    sys.exit(verify())
