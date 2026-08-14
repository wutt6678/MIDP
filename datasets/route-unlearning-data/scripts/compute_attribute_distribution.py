#!/usr/bin/env python3
"""Compute FIUBench attribute distribution at the unique-image / identity level.

Produces:
  outputs/full_fiubench/evidence/attribute_distribution_report.json

Analysis unit: unique image (= unique identity).  Each identity has exactly
one source image, so image-level and identity-level statistics coincide.

For each of the 13 CelebA reliability-whitelist attributes the report
contains:
  • image-level positive / negative / uncertain counts and fractions
  • per-role (train / eval / exclude / out_of_protocol) breakdown
  • positive prevalence among accepted (non-uncertain) images

Usage:
    python scripts/compute_attribute_distribution.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO / "outputs" / "full_fiubench" / "Qwen_Qwen3.5-9B" / "fiubench"
EVIDENCE_DIR = REPO / "outputs" / "full_fiubench" / "evidence"

WHITELIST_ATTRS = [
    "Bald", "Bangs", "Blond_Hair", "Eyeglasses", "Gray_Hair",
    "Male", "Mustache", "Sideburns", "Smiling",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necktie",
]


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO,
    ).decode().strip()


def _attr_key(short_name: str) -> str:
    return f"extended_attributes.celeba40.{short_name}"


def compute() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    code_sha = _git_sha()
    processed_path = ARTIFACT_DIR / "fiubench_processed.jsonl"

    # ── Build one record per identity (first canonical row per identity) ──
    # identity_id -> {split, identity_name, labels: {attr: True/False/None}}
    identities: dict[str, dict] = {}
    for line in processed_path.read_text().splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        iid = doc.get("identity_id", "")
        if not iid or iid in identities:
            continue
        split = doc.get("split", "unknown")
        iname = doc.get("identity_name", "")
        labels: dict[str, object] = {}
        va = doc.get("visual_attributes", {})
        for attr_short in WHITELIST_ATTRS:
            obs = va.get(_attr_key(attr_short), {})
            labels[attr_short] = obs.get("label")  # True / False / None
        identities[iid] = {
            "split": split,
            "identity_name": iname,
            "labels": labels,
        }

    n_identities = len(identities)
    print(f"Total identities: {n_identities}")

    # ── Per-attribute image-level statistics ──
    attributes_report: dict[str, dict] = {}

    for attr in WHITELIST_ATTRS:
        pos_ids: list[str] = []
        neg_ids: list[str] = []
        unc_ids: list[str] = []

        role_stats: dict[str, dict[str, int]] = {
            "train": {"positive": 0, "negative": 0, "uncertain": 0},
            "eval": {"positive": 0, "negative": 0, "uncertain": 0},
            "exclude": {"positive": 0, "negative": 0, "uncertain": 0},
            "out_of_protocol": {"positive": 0, "negative": 0, "uncertain": 0},
        }

        for iid, info in identities.items():
            label = info["labels"].get(attr)
            split = info["split"]
            if label is True:
                pos_ids.append(iid)
                if split in role_stats:
                    role_stats[split]["positive"] += 1
            elif label is False:
                neg_ids.append(iid)
                if split in role_stats:
                    role_stats[split]["negative"] += 1
            else:
                unc_ids.append(iid)
                if split in role_stats:
                    role_stats[split]["uncertain"] += 1

        n_pos = len(pos_ids)
        n_neg = len(neg_ids)
        n_unc = len(unc_ids)
        n_accepted = n_pos + n_neg
        accepted_fraction = n_accepted / n_identities if n_identities else 0
        uncertainty_fraction = n_unc / n_identities if n_identities else 0
        pos_prevalence = n_pos / n_accepted if n_accepted else 0.0
        neg_prevalence = n_neg / n_accepted if n_accepted else 0.0

        attributes_report[attr] = {
            "unique_images_total": n_identities,
            "accepted_positive_images": n_pos,
            "accepted_negative_images": n_neg,
            "uncertain_images": n_unc,
            "accepted_fraction": round(accepted_fraction, 4),
            "uncertainty_fraction": round(uncertainty_fraction, 4),
            "positive_prevalence_among_accepted": round(pos_prevalence, 4),
            "negative_prevalence_among_accepted": round(neg_prevalence, 4),
            "roles": role_stats,
            "positive_identity_ids": sorted(pos_ids),
            "negative_identity_ids": sorted(neg_ids),
            "uncertain_identity_ids": sorted(unc_ids),
        }

    # ── Summary ──
    report = {
        "analysis_unit": "unique_image",
        "n_images": n_identities,
        "n_identities": n_identities,
        "whitelist_attributes": WHITELIST_ATTRS,
        "n_whitelist_attributes": len(WHITELIST_ATTRS),
        "attributes": attributes_report,
        "midp_commit": code_sha,
    }

    out_path = EVIDENCE_DIR / "attribute_distribution_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out_path}")

    # ── Print summary table ──
    print(f"\n{'Attribute':<20} {'Pos':>5} {'Neg':>5} {'Unc':>5} {'Acc%':>7} {'Unc%':>7} {'PosPrev':>8}")
    print("-" * 65)
    for attr in WHITELIST_ATTRS:
        a = attributes_report[attr]
        print(
            f"{attr:<20} "
            f"{a['accepted_positive_images']:>5} "
            f"{a['accepted_negative_images']:>5} "
            f"{a['uncertain_images']:>5} "
            f"{a['accepted_fraction']:>7.2%} "
            f"{a['uncertainty_fraction']:>7.2%} "
            f"{a['positive_prevalence_among_accepted']:>8.4f}"
        )


if __name__ == "__main__":
    compute()
