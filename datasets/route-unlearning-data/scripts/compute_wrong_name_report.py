#!/usr/bin/env python3
"""Generate the wrong-name matching report for FIUBench.

Produces:
  outputs/full_fiubench/evidence/wrong_name_matching_report.json

Uses the same eligibility logic as production route-probe generation:
  • An attribute is eligible when label is not None AND confidence_band == "high".
  • Similarity is the Jaccard agreement-over-union on signed attribute states.

The report includes:
  • Number of structurally eligible target/control pairs
  • Number of visually matched pairs (similarity > 0)
  • Matching similarity distribution
  • Attributes represented in matched pairs
  • Target identities with at least one valid control
  • Control identities used
  • Protocol-role distribution

Usage:
    PYTHONPATH=src python scripts/compute_wrong_name_report.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

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


def _accepted_visible_attrs(sample: dict) -> dict[str, bool]:
    """Extract eligible visual attributes from a processed JSONL row."""
    prefix = "extended_attributes.celeba40."
    va = sample.get("visual_attributes", {})
    out: dict[str, bool] = {}
    for key, obs in va.items():
        if not key.startswith(prefix):
            continue
        if not isinstance(obs, dict):
            continue
        label = obs.get("label")
        band = obs.get("confidence_band")
        if label is not None and band == "high":
            short = key[len(prefix):]
            out[short] = bool(label)
    return out


def _jaccard(attrs_a: dict[str, bool], attrs_b: dict[str, bool]) -> float:
    """Agreement-over-union on signed attribute states."""
    shared = set(attrs_a) & set(attrs_b)
    if not shared:
        return 0.0
    agreeing = sum(1 for k in shared if attrs_a[k] == attrs_b[k])
    union = set(attrs_a) | set(attrs_b)
    return agreeing / len(union)


def compute() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    code_sha = _git_sha()
    processed_path = ARTIFACT_DIR / "fiubench_processed.jsonl"

    # ── Group by identity, find eligible visual attrs ──
    # identity_id -> {split, identity_name, anchor_attrs, row_count}
    identity_info: dict[str, dict] = {}
    for line in processed_path.read_text().splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        iid = doc.get("identity_id", "")
        if not iid:
            continue
        if iid not in identity_info:
            identity_info[iid] = {
                "split": doc.get("split", "unknown"),
                "identity_name": doc.get("identity_name", ""),
                "anchor_attrs": {},
                "row_count": 0,
            }
        identity_info[iid]["row_count"] += 1
        # Try to get anchor attrs from first eligible row.
        if not identity_info[iid]["anchor_attrs"]:
            attrs = _accepted_visible_attrs(doc)
            if attrs:
                identity_info[iid]["anchor_attrs"] = attrs

    n_identities = len(identity_info)
    identities_with_anchors = {
        iid: info for iid, info in identity_info.items()
        if info["anchor_attrs"]
    }
    n_with_anchors = len(identities_with_anchors)
    print(f"Total identities: {n_identities}")
    print(f"Identities with eligible visual anchors: {n_with_anchors}")

    # ── Compute all pairs with Jaccard similarity ──
    eligible_ids = sorted(identities_with_anchors)
    all_pairs: list[tuple[str, str, float]] = []
    for i, tgt in enumerate(eligible_ids):
        for ctrl in eligible_ids[i + 1:]:
            # At least one side must have >= 2 samples.
            if (identity_info[tgt]["row_count"] < 2
                    and identity_info[ctrl]["row_count"] < 2):
                continue
            sim = _jaccard(
                identities_with_anchors[tgt]["anchor_attrs"],
                identities_with_anchors[ctrl]["anchor_attrs"],
            )
            all_pairs.append((tgt, ctrl, sim))

    # Sort by descending similarity.
    all_pairs.sort(key=lambda t: (-t[2], t[0], t[1]))

    # ── Statistics ──
    n_total_pairs = len(all_pairs)
    matched_pairs = [(t, c, s) for t, c, s in all_pairs if s > 0]
    n_matched = len(matched_pairs)

    # Similarity distribution.
    sim_bins = {"0.0": 0, "(0.0, 0.2]": 0, "(0.2, 0.4]": 0,
                "(0.4, 0.6]": 0, "(0.6, 0.8]": 0, "(0.8, 1.0]": 0}
    for _, _, s in all_pairs:
        if s == 0.0:
            sim_bins["0.0"] += 1
        elif s <= 0.2:
            sim_bins["(0.0, 0.2]"] += 1
        elif s <= 0.4:
            sim_bins["(0.2, 0.4]"] += 1
        elif s <= 0.6:
            sim_bins["(0.4, 0.6]"] += 1
        elif s <= 0.8:
            sim_bins["(0.6, 0.8]"] += 1
        else:
            sim_bins["(0.8, 1.0]"] += 1

    # Target/control usage.
    target_ids_with_match: set[str] = set()
    control_ids_used: set[str] = set()
    for tgt, ctrl, s in matched_pairs:
        target_ids_with_match.add(tgt)
        target_ids_with_match.add(ctrl)  # Each is both a potential target and control.
        control_ids_used.add(tgt)
        control_ids_used.add(ctrl)

    # Attributes in matched pairs.
    attrs_in_matched: Counter = Counter()
    for tgt, ctrl, s in matched_pairs:
        tgt_attrs = identities_with_anchors[tgt]["anchor_attrs"]
        ctrl_attrs = identities_with_anchors[ctrl]["anchor_attrs"]
        for attr in set(tgt_attrs) | set(ctrl_attrs):
            attrs_in_matched[attr] += 1

    # Protocol-role distribution of matched pairs.
    role_dist: Counter = Counter()
    for tgt, ctrl, s in matched_pairs:
        tgt_role = identity_info[tgt]["split"]
        ctrl_role = identity_info[ctrl]["split"]
        role_dist[f"{tgt_role}↔{ctrl_role}"] += 1

    # Top-20 matched pairs.
    top_pairs = [
        {
            "target_id": t,
            "control_id": c,
            "similarity": round(s, 4),
            "target_name": identity_info[t]["identity_name"],
            "control_name": identity_info[c]["identity_name"],
            "target_split": identity_info[t]["split"],
            "control_split": identity_info[c]["split"],
        }
        for t, c, s in all_pairs[:20]
    ]

    # ── Build report ──
    report = {
        "analysis_unit": "unique_image",
        "n_identities": n_identities,
        "n_identities_with_visual_anchors": n_with_anchors,
        "n_total_candidate_pairs": n_total_pairs,
        "n_visually_matched_pairs": n_matched,
        "n_unmatched_pairs": n_total_pairs - n_matched,
        "similarity_distribution": sim_bins,
        "target_identities_with_at_least_one_match": len(target_ids_with_match),
        "control_identities_used": len(control_ids_used),
        "attributes_in_matched_pairs": dict(attrs_in_matched.most_common()),
        "protocol_role_distribution": dict(role_dist.most_common()),
        "top_20_pairs": top_pairs,
        "midp_commit": code_sha,
    }

    out_path = EVIDENCE_DIR / "wrong_name_matching_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    print(f"Total pairs: {n_total_pairs}")
    print(f"Matched pairs (sim > 0): {n_matched}")
    print(f"Similarity distribution: {sim_bins}")
    print(f"Identities with ≥1 match: {len(target_ids_with_match)}")


if __name__ == "__main__":
    compute()
