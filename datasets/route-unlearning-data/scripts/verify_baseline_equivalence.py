#!/usr/bin/env python3
"""Verify baseline equivalence for evidence migration (P0-GEMMA-01).

Compares old and current baseline results JSONL files to prove that
the scientific content is identical despite repackaging.

Usage::

    python scripts/verify_baseline_equivalence.py \\
      --model gemma3_12b \\
      --old-commit ac5200c
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Scientific fields for binary probes (exact comparison).
BINARY_SCIENTIFIC_FIELDS = [
    "probe_id", "target_label", "predicted_label",
    "logp_yes", "logp_no",
    "raw_margin", "signed_answer_margin",
    "p_yes", "correct",
]

# Name-only fields (exact comparison).
NAME_ONLY_FIELDS = [
    "probe_id", "generated_response", "normalized_response",
    "token_overlap", "fuzzy_match", "cap_hit",
    "generated_token_count",
]

# Fields to ignore (non-scientific).
IGNORED_FIELDS = {
    "latency", "cache_key", "filesystem_path",
    "artifact_commit", "packaging_timestamp",
}


def _get_old_file(commit: str, rel_path: str) -> str | None:
    """Retrieve a file from a git commit."""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:datasets/route-unlearning-data/{rel_path}"],
            capture_output=True, text=True, check=True,
            cwd=PROJECT_ROOT,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def _parse_jsonl(text: str) -> list[dict]:
    """Parse JSONL text into a list of dicts."""
    return [json.loads(line) for line in text.strip().split("\n") if line.strip()]


def _compare_results(
    old_results: list[dict],
    new_results: list[dict],
) -> dict[str, Any]:
    """Compare two sets of baseline results scientifically."""
    report: dict[str, Any] = {
        "old_count": len(old_results),
        "new_count": len(new_results),
        "probe_ids_match": False,
        "scientific_fields_exact": True,
        "numerical_tolerance_pass": True,
        "differences": [],
    }

    if len(old_results) != len(new_results):
        report["scientific_fields_exact"] = False
        report["differences"].append(
            f"Count mismatch: old={len(old_results)}, new={len(new_results)}"
        )
        return report

    # Index by probe_id
    old_by_id = {r["probe_id"]: r for r in old_results}
    new_by_id = {r["probe_id"]: r for r in new_results}

    old_ids = set(old_by_id.keys())
    new_ids = set(new_by_id.keys())

    report["probe_ids_match"] = old_ids == new_ids
    if old_ids != new_ids:
        missing = old_ids - new_ids
        extra = new_ids - old_ids
        if missing:
            report["differences"].append(f"Missing probe IDs: {len(missing)}")
        if extra:
            report["differences"].append(f"Extra probe IDs: {len(extra)}")

    # Compare scientific fields
    n_compared = 0
    n_exact = 0
    max_logp_diff = 0.0
    tolerance = 1e-6

    for pid in sorted(old_ids & new_ids):
        old_r = old_by_id[pid]
        new_r = new_by_id[pid]

        for field in BINARY_SCIENTIFIC_FIELDS:
            if field in old_r and field in new_r:
                n_compared += 1
                old_val = old_r[field]
                new_val = new_r[field]

                if isinstance(old_val, float):
                    diff = abs(old_val - new_val)
                    max_logp_diff = max(max_logp_diff, diff)
                    if diff > tolerance:
                        report["numerical_tolerance_pass"] = False
                        report["differences"].append(
                            f"{pid}.{field}: old={old_val}, new={new_val}, diff={diff}"
                        )
                    else:
                        n_exact += 1
                else:
                    if old_val == new_val:
                        n_exact += 1
                    else:
                        report["scientific_fields_exact"] = False
                        report["differences"].append(
                            f"{pid}.{field}: old={old_val!r}, new={new_val!r}"
                        )

    report["n_compared"] = n_compared
    report["n_exact"] = n_exact
    report["max_numerical_diff"] = max_logp_diff
    report["overall_pass"] = (
        report["probe_ids_match"]
        and report["scientific_fields_exact"]
        and report["numerical_tolerance_pass"]
        and len(old_results) == len(new_results)
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify baseline equivalence")
    parser.add_argument("--model", required=True, help="Model key")
    parser.add_argument("--old-commit", required=True, help="Git commit with old baseline")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for migration artifacts",
    )
    args = parser.parse_args()

    model_key = args.model
    rel_path = f"outputs/experiments/pre_unlearning/{model_key}/baseline_v1/baseline_results.jsonl"

    # Get old baseline from git
    old_text = _get_old_file(args.old_commit, rel_path)
    if old_text is None:
        print(f"ERROR: Cannot retrieve old baseline from commit {args.old_commit}")
        sys.exit(1)

    old_results = _parse_jsonl(old_text)
    print(f"Old baseline: {len(old_results)} probes (from {args.old_commit})")

    # Load current baseline
    current_path = PROJECT_ROOT / rel_path
    if not current_path.is_file():
        print(f"ERROR: Current baseline not found: {current_path}")
        sys.exit(1)

    with open(current_path) as f:
        new_results = [json.loads(line) for line in f if line.strip()]
    print(f"Current baseline: {len(new_results)} probes")

    # Compare
    comparison = _compare_results(old_results, new_results)
    print(f"Comparison: {'PASS' if comparison['overall_pass'] else 'FAIL'}")
    print(f"  Probes: {comparison['old_count']} old, {comparison['new_count']} new")
    print(f"  IDs match: {comparison['probe_ids_match']}")
    print(f"  Exact fields: {comparison['scientific_fields_exact']}")
    print(f"  Numerical tolerance: {comparison['numerical_tolerance_pass']}")
    print(f"  Max numerical diff: {comparison.get('max_numerical_diff', 'N/A')}")
    if comparison["differences"]:
        print(f"  Differences ({len(comparison['differences'])}):")
        for d in comparison["differences"][:5]:
            print(f"    - {d}")

    # Write migration artifact
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs/experiments/pre_unlearning" / model_key / "baseline_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    migration_path = output_dir / "baseline_equivalence_migration.json"
    migration = {
        "schema_version": "baseline-equivalence-migration-v1",
        "model_key": model_key,
        "old_commit": args.old_commit,
        "old_results_count": len(old_results),
        "new_results_count": len(new_results),
        "comparison": comparison,
        "migration_pass": comparison["overall_pass"],
    }
    with open(migration_path, "w") as f:
        json.dump(migration, f, indent=2)
        f.write("\n")
    print(f"\nMigration artifact: {migration_path}")

    if not comparison["overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
