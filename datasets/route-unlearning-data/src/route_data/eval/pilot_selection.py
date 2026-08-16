"""Pilot identity selection for the Stage 3 unlearning experiment.

This module deterministically selects *target*, *retain*, and *control*
identity groups from the frozen baseline results, matching on protocol role,
baseline margins, and attribute diversity for retain/control selection.

The selection is seeded (default ``seed=17``) so that the manifest is
reproducible and can be frozen *before* any training begins.

Public API
----------
.. autoclass:: IdentityStats
.. autofunction:: build_identity_stats
.. autofunction:: build_identity_attribute_map
.. autofunction:: select_pilot_identities
.. autofunction:: write_selection_manifest
"""

from __future__ import annotations

import hashlib
import json
import math
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class IdentityStats:
    """Per-identity summary statistics drawn from the frozen baseline."""

    identity_id: str
    protocol_role: str
    target_attribute: str
    attribute_positive: bool
    mean_dv_margin: float
    mean_ipn_margin: float
    mean_wn_margin: float
    mean_vtc_margin: float
    mean_overall_margin: float
    probe_count: int
    unique_images: int

    # -- derived helpers -------------------------------------------------- #

    @property
    def attr_key(self) -> tuple[str, bool]:
        """ ``(attribute_name, is_positive_state)`` tuple. """
        return (self.target_attribute, self.attribute_positive)


@dataclass
class PilotSelection:
    """The three identity groups plus provenance metadata."""

    selection_version: str = "pilot-selection-v1"
    seed: int = 17
    target_identities: list[str] = field(default_factory=list)
    retain_identities: list[str] = field(default_factory=list)
    control_identities: list[str] = field(default_factory=list)
    matching_criteria: list[str] = field(default_factory=list)
    baseline_manifest_sha256: str = ""
    baseline_results_sha256: str = ""
    route_probe_sha256: str = ""
    processed_dataset_sha256: str = ""
    code_commit: str = ""
    identity_details: dict[str, dict[str, Any]] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Statistics builders
# --------------------------------------------------------------------------- #

def _sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build_identity_attribute_map(
    route_probe_path: str | Path,
) -> dict[str, tuple[str, bool]]:
    """Map each identity to ``(target_attribute, is_positive)``.

    The route-probe file carries one ``target_attribute`` per identity.
    The *positive* flag is ``True`` when any image-bearing probe for that
    identity has ``answer_label == True`` for that attribute.
    """
    attr_map: dict[str, dict[str, set[bool]]] = defaultdict(
        lambda: defaultdict(set)
    )
    with open(route_probe_path) as fh:
        for line in fh:
            row = json.loads(line)
            iid = row["identity_id"]
            ta = row.get("target_attribute")
            al = row.get("answer_label")
            if ta is not None and al is not None:
                attr_map[iid][ta].add(bool(al))

    result: dict[str, tuple[str, bool]] = {}
    for iid, attrs in attr_map.items():
        # Each identity has exactly one target attribute in the route probes.
        for attr, states in attrs.items():
            result[iid] = (attr, True in states)
            break  # only one attribute per identity
    return result


def build_identity_stats(
    baseline_results_path: str | Path,
    route_probe_path: str | Path,
    processed_dataset_path: str | Path | None = None,
) -> dict[str, IdentityStats]:
    """Compute per-identity statistics from the frozen baseline.

    Parameters
    ----------
    baseline_results_path:
        Path to ``baseline_results.jsonl``.
    route_probe_path:
        Path to the route-conflict eval JSONL (for attribute info).
    processed_dataset_path:
        Optional path to the processed dataset (for unique-image counts).
    """
    attr_map = build_identity_attribute_map(route_probe_path)

    # -- accumulate per-identity margins ---------------------------------- #
    fam_margins: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    protocol_roles: dict[str, str] = {}
    probe_ids: dict[str, set[str]] = defaultdict(set)

    with open(baseline_results_path) as fh:
        for line in fh:
            row = json.loads(line)
            iid = row["identity_id"]
            fam = row["probe_family"]
            protocol_roles[iid] = row["protocol_role"]
            probe_ids[iid].add(row["probe_id"])
            if fam != "name_only" and row.get("signed_answer_margin") is not None:
                fam_margins[iid][fam].append(float(row["signed_answer_margin"]))

    # -- optional unique-image counts ------------------------------------ #
    unique_images: dict[str, int] = {}
    if processed_dataset_path is not None:
        img_sets: dict[str, set[str]] = defaultdict(set)
        with open(processed_dataset_path) as fh:
            for line in fh:
                row = json.loads(line)
                iid = row["identity_id"]
                sha = row.get("image_sha256")
                if sha:
                    img_sets[iid].add(sha)
        unique_images = {iid: len(s) for iid, s in img_sets.items()}

    # -- build IdentityStats objects ------------------------------------- #
    stats: dict[str, IdentityStats] = {}
    for iid in sorted(protocol_roles):
        fams = fam_margins.get(iid, {})
        dv = fams.get("direct_visual", [])
        ipn = fams.get("image_plus_name", [])
        wn = fams.get("wrong_name", [])
        vtc = fams.get("visual_text_conflict", [])

        mean_dv = sum(dv) / len(dv) if dv else 0.0
        mean_ipn = sum(ipn) / len(ipn) if ipn else 0.0
        mean_wn = sum(wn) / len(wn) if wn else 0.0
        mean_vtc = sum(vtc) / len(vtc) if vtc else 0.0
        mean_overall = (mean_dv + mean_ipn + mean_wn + mean_vtc) / 4.0

        attr, is_pos = attr_map.get(iid, ("Unknown", False))

        stats[iid] = IdentityStats(
            identity_id=iid,
            protocol_role=protocol_roles[iid],
            target_attribute=attr,
            attribute_positive=is_pos,
            mean_dv_margin=mean_dv,
            mean_ipn_margin=mean_ipn,
            mean_wn_margin=mean_wn,
            mean_vtc_margin=mean_vtc,
            mean_overall_margin=mean_overall,
            probe_count=len(probe_ids[iid]),
            unique_images=unique_images.get(iid, 0),
        )
    return stats


# --------------------------------------------------------------------------- #
# Eligibility filter
# --------------------------------------------------------------------------- #

def _eligible(
    stats: dict[str, IdentityStats],
    role: str,
) -> list[IdentityStats]:
    """Return identities with the given *protocol_role* and finite margins."""
    out: list[IdentityStats] = []
    for s in stats.values():
        if s.protocol_role != role:
            continue
        if not all(
            math.isfinite(v)
            for v in [
                s.mean_dv_margin,
                s.mean_ipn_margin,
                s.mean_wn_margin,
                s.mean_vtc_margin,
            ]
        ):
            continue
        if s.probe_count < 5:
            continue
        out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Matching / selection
# --------------------------------------------------------------------------- #

def _margin_distance(a: IdentityStats, b: IdentityStats) -> float:
    """Weighted L2 distance on baseline margins."""
    ddv = a.mean_dv_margin - b.mean_dv_margin
    dipn = a.mean_ipn_margin - b.mean_ipn_margin
    return (ddv ** 2 + dipn ** 2) ** 0.5


def select_pilot_identities(
    stats: dict[str, IdentityStats],
    *,
    target_count: int = 2,
    retain_count: int = 2,
    control_count: int = 2,
    seed: int = 17,
    preferred_role: str = "train",
) -> PilotSelection:
    """Deterministically select target / retain / control identity groups.

    Algorithm
    ---------
    1. Filter to *eligible* identities with ``preferred_role`` and finite
       baseline margins.
    2. Group eligible identities by ``(attribute, positive_state)``.
    3. Among attribute groups with ≥ ``target_count`` members, pick the
       group whose members have the highest mean overall margin (break
       ties by sorted identity-id).  This is the *target attribute*.
    4. Select ``target_count`` identities from that group (seeded shuffle,
       take the first *n*).
    5. For *retain* and *control*, gather candidates from *different*
       attribute groups (not the target attribute), sorted by proximity
       to the mean target margin.  Greedily pick the closest, ensuring
       no identity overlap.
    """
    rng = random.Random(seed)

    eligible = _eligible(stats, preferred_role)
    if len(eligible) < target_count + retain_count + control_count:
        raise ValueError(
            f"Not enough eligible {preferred_role} identities: "
            f"have {len(eligible)}, need "
            f"{target_count + retain_count + control_count}"
        )

    # -- group by attribute ----------------------------------------------- #
    attr_groups: dict[tuple[str, bool], list[IdentityStats]] = defaultdict(list)
    for s in eligible:
        attr_groups[s.attr_key].append(s)

    # -- pick target attribute group -------------------------------------- #
    # Among groups with ≥ target_count members, pick the one with highest
    # mean overall margin (deterministic tie-break on sorted ids).
    candidate_groups = {
        k: v for k, v in attr_groups.items() if len(v) >= target_count
    }
    if not candidate_groups:
        raise ValueError(
            "No attribute group has enough identities for the target set."
        )

    def _group_sort_key(
        item: tuple[tuple[str, bool], list[IdentityStats]],
    ) -> tuple[float, str]:
        _key, members = item
        mean_margin = sum(s.mean_overall_margin for s in members) / len(members)
        sorted_ids = sorted(s.identity_id for s in members)
        return (-mean_margin, sorted_ids[0])

    target_attr_key = min(candidate_groups.items(), key=_group_sort_key)[0]
    target_candidates = sorted(
        candidate_groups[target_attr_key],
        key=lambda s: s.identity_id,
    )
    rng.shuffle(target_candidates)
    targets = target_candidates[:target_count]

    # -- build candidate pool for retain / control ------------------------ #
    target_ids = {s.identity_id for s in targets}
    target_mean_dv = sum(s.mean_dv_margin for s in targets) / len(targets)
    target_mean_ipn = sum(s.mean_ipn_margin for s in targets) / len(targets)
    target_mean_overall = sum(s.mean_overall_margin for s in targets) / len(targets)

    # Candidates from non-target attribute groups
    other_candidates: list[IdentityStats] = []
    for key, members in attr_groups.items():
        if key == target_attr_key:
            continue
        for s in members:
            if s.identity_id not in target_ids:
                other_candidates.append(s)

    # Sort by proximity to target mean margins
    def _proximity(s: IdentityStats) -> tuple[float, str]:
        dist = (
            (s.mean_dv_margin - target_mean_dv) ** 2
            + (s.mean_ipn_margin - target_mean_ipn) ** 2
            + (s.mean_overall_margin - target_mean_overall) ** 2
        ) ** 0.5
        return (dist, s.identity_id)

    other_candidates.sort(key=_proximity)

    # -- greedily assign retain / control --------------------------------- #
    used_ids = set(target_ids)
    retain: list[IdentityStats] = []
    control: list[IdentityStats] = []

    # Track which attributes are used for retain vs control to ensure
    # diversity when possible.
    retain_attrs: set[tuple[str, bool]] = set()
    control_attrs: set[tuple[str, bool]] = set()

    for cand in other_candidates:
        if len(retain) >= retain_count and len(control) >= control_count:
            break
        if cand.identity_id in used_ids:
            continue

        # Prefer attribute diversity: if retain already has this attr,
        # skip for retain (but allow for control if needed).
        if len(retain) < retain_count and cand.attr_key not in retain_attrs:
            retain.append(cand)
            used_ids.add(cand.identity_id)
            retain_attrs.add(cand.attr_key)
            continue

        if len(control) < control_count and cand.attr_key not in control_attrs:
            control.append(cand)
            used_ids.add(cand.identity_id)
            control_attrs.add(cand.attr_key)
            continue

        # Fallback: allow same-attribute if necessary
        if len(retain) < retain_count:
            retain.append(cand)
            used_ids.add(cand.identity_id)
            retain_attrs.add(cand.attr_key)
        elif len(control) < control_count:
            control.append(cand)
            used_ids.add(cand.identity_id)
            control_attrs.add(cand.attr_key)

    if len(retain) < retain_count or len(control) < control_count:
        raise ValueError(
            f"Could not find enough non-overlapping identities: "
            f"retain={len(retain)}/{retain_count}, "
            f"control={len(control)}/{control_count}"
        )

    # -- build result ----------------------------------------------------- #
    sel = PilotSelection(seed=seed)
    sel.target_identities = sorted(s.identity_id for s in targets)
    sel.retain_identities = sorted(s.identity_id for s in retain)
    sel.control_identities = sorted(s.identity_id for s in control)
    sel.matching_criteria = [
        "same-protocol-role",
        "baseline-margin-matched",
        "attribute-diverse-retain-control",
    ]

    # Attach per-identity detail for the manifest
    all_selected = targets + retain + control
    group_map: dict[str, str] = {}
    for s in targets:
        group_map[s.identity_id] = "target"
    for s in retain:
        group_map[s.identity_id] = "retain"
    for s in control:
        group_map[s.identity_id] = "control"

    for s in all_selected:
        sel.identity_details[s.identity_id] = {
            "group": group_map[s.identity_id],
            "protocol_role": s.protocol_role,
            "target_attribute": s.target_attribute,
            "attribute_positive": s.attribute_positive,
            "mean_dv_margin": round(s.mean_dv_margin, 6),
            "mean_ipn_margin": round(s.mean_ipn_margin, 6),
            "mean_wn_margin": round(s.mean_wn_margin, 6),
            "mean_vtc_margin": round(s.mean_vtc_margin, 6),
            "mean_overall_margin": round(s.mean_overall_margin, 6),
            "probe_count": s.probe_count,
            "unique_images": s.unique_images,
        }

    return sel


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Frozen SHA preflight (P0-5)
# --------------------------------------------------------------------------- #

def validate_pilot_frozen_inputs(
    *,
    baseline_manifest_path: str | Path,
    baseline_results_path: str | Path,
    route_probe_path: str | Path,
    processed_dataset_path: str | Path | None,
    expected_manifest_sha: str,
    expected_results_sha: str,
    expected_route_probe_sha: str,
    expected_processed_ds_sha: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Verify frozen input SHAs before selection (P0-5).

    Recomputes SHA-256 of each frozen artifact and compares against the
    configured expected values.  Raises ``RuntimeError`` on any mismatch.

    Returns a preflight report dict.
    """
    actual_manifest_sha = _sha256_file(baseline_manifest_path)
    actual_results_sha = _sha256_file(baseline_results_path)
    actual_route_sha = _sha256_file(route_probe_path)
    actual_processed_sha = (
        _sha256_file(processed_dataset_path)
        if processed_dataset_path is not None
        else ""
    )

    manifest_match = actual_manifest_sha == expected_manifest_sha
    results_match = actual_results_sha == expected_results_sha
    route_match = actual_route_sha == expected_route_probe_sha
    processed_match = (
        actual_processed_sha == expected_processed_ds_sha
        if expected_processed_ds_sha
        else True
    )

    report: dict[str, Any] = {
        "pass": all([manifest_match, results_match, route_match, processed_match]),
        "baseline_manifest_sha_match": manifest_match,
        "baseline_results_sha_match": results_match,
        "route_probe_sha_match": route_match,
        "processed_dataset_sha_match": processed_match,
        "actual_baseline_manifest_sha256": actual_manifest_sha,
        "actual_baseline_results_sha256": actual_results_sha,
        "actual_route_probe_sha256": actual_route_sha,
        "actual_processed_dataset_sha256": actual_processed_sha,
        "expected_baseline_manifest_sha256": expected_manifest_sha,
        "expected_baseline_results_sha256": expected_results_sha,
        "expected_route_probe_sha256": expected_route_probe_sha,
        "expected_processed_dataset_sha256": expected_processed_ds_sha,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    if not report["pass"]:
        raise RuntimeError(
            "Frozen SHA preflight FAILED. "
            f"manifest={manifest_match}, results={results_match}, "
            f"route={route_match}, processed={processed_match}. "
            "See evidence/pilot_preflight_report.json for details."
        )

    logger.info("Frozen SHA preflight PASSED.")
    return report


# --------------------------------------------------------------------------- #
# Leakage detection (P0-3)
# --------------------------------------------------------------------------- #

def run_leakage_detection(
    processed_dataset_path: str | Path,
    route_probe_path: str | Path,
    target_identity_ids: list[str],
    retain_identity_ids: list[str],
    output_path: str | Path,
) -> dict[str, Any]:
    """Detect leakage between training data and frozen eval probes (P0-3).

    Compares all intervention training examples against the frozen route
    probes on sample_id, source_sample_id, probe_id, question text,
    normalised question text, identity_id, and image_sha256.

    Writes ``leakage_report.json`` and returns the report dict.
    Raises ``RuntimeError`` on hard-stop violations.
    """
    # -- Build frozen probe index ---------------------------------------- #
    probe_sample_ids: set[str] = set()
    probe_source_ids: set[str] = set()
    probe_probe_ids: set[str] = set()
    probe_questions: set[str] = set()
    probe_norm_questions: set[str] = set()
    probe_image_shas: set[str] = set()
    probe_identity_ids: set[str] = set()

    with open(route_probe_path) as fh:
        for line in fh:
            row = json.loads(line)
            if sid := row.get("sample_id"):
                probe_sample_ids.add(str(sid))
            if ssid := row.get("source_sample_id"):
                probe_source_ids.add(str(ssid))
            if pid := row.get("probe_id"):
                probe_probe_ids.add(str(pid))
            if q := row.get("question_text", ""):
                probe_questions.add(q)
                probe_norm_questions.add(q.strip().lower())
            if sha := row.get("image_sha256"):
                probe_image_shas.add(sha)
            if iid := row.get("identity_id"):
                probe_identity_ids.add(str(iid))

    # -- Scan training samples ------------------------------------------- #
    train_identity_ids: set[str] = set()
    overlap_sample_ids: list[str] = []
    overlap_source_ids: list[str] = []
    overlap_probe_ids: list[str] = []
    overlap_questions: list[str] = []
    overlap_norm_questions: list[str] = []
    overlap_image_shas: list[str] = []

    with open(processed_dataset_path) as fh:
        for line in fh:
            row = json.loads(line)
            iid = str(row.get("identity_id", ""))
            if iid not in set(target_identity_ids) | set(retain_identity_ids):
                continue
            train_identity_ids.add(iid)

            if sid := row.get("sample_id"):
                if str(sid) in probe_sample_ids:
                    overlap_sample_ids.append(str(sid))
            if ssid := row.get("source_sample_id"):
                if str(ssid) in probe_source_ids:
                    overlap_source_ids.append(str(ssid))
            if pid := row.get("probe_id"):
                if str(pid) in probe_probe_ids:
                    overlap_probe_ids.append(str(pid))
            if q := row.get("question_text", ""):
                if q in probe_questions:
                    overlap_questions.append(q[:200])
                if q.strip().lower() in probe_norm_questions:
                    overlap_norm_questions.append(q[:200])
            if sha := row.get("image_sha256"):
                if sha in probe_image_shas:
                    overlap_image_shas.append(sha)

    report: dict[str, Any] = {
        "pass": (
            len(overlap_sample_ids) == 0
            and len(overlap_probe_ids) == 0
            and len(overlap_questions) == 0
        ),
        "exact_probe_id_overlap": len(overlap_probe_ids),
        "exact_sample_id_overlap": len(overlap_sample_ids),
        "exact_source_id_overlap": len(overlap_source_ids),
        "exact_question_overlap": len(overlap_questions),
        "normalized_question_overlap": len(overlap_norm_questions),
        "image_sha_overlap_count": len(overlap_image_shas),
        "image_overlap_policy": "disallowed",
        "training_identity_count": len(train_identity_ids),
        "probe_identity_count": len(probe_identity_ids),
        "identity_id_overlap": sorted(train_identity_ids & probe_identity_ids),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    logger.info(
        "Leakage detection: probe_id=%d, sample_id=%d, question=%d, image_sha=%d",
        len(overlap_probe_ids),
        len(overlap_sample_ids),
        len(overlap_questions),
        len(overlap_image_shas),
    )

    if not report["pass"]:
        raise RuntimeError(
            "LEAKAGE DETECTED between training data and frozen eval probes. "
            f"probe_id_overlap={len(overlap_probe_ids)}, "
            f"sample_id_overlap={len(overlap_sample_ids)}, "
            f"question_overlap={len(overlap_questions)}. "
            "See selection/leakage_report.json for details."
        )

    return report


# --------------------------------------------------------------------------- #
# Intervention dataset manifest (P0-4)
# --------------------------------------------------------------------------- #

def generate_intervention_manifest(
    *,
    processed_dataset_path: str | Path,
    target_identity_ids: list[str],
    retain_identity_ids: list[str],
    selection_manifest_sha256: str,
    leakage_report_sha256: str,
    experiment_config: dict[str, Any],
    output_path: str | Path,
    seed: int = 17,
    code_commit: str = "",
) -> Path:
    """Generate the intervention dataset manifest (P0-4).

    Records full provenance of every training example used in the pilot
    unlearning run.  The manifest is self-hashing: the final SHA-256 is
    computed over the full content (with ``manifest_sha256`` set to
    ``"pending"``) and then appended.
    """
    import subprocess as _sp

    # -- Git dirty state ------------------------------------------------- #
    try:
        _r = _sp.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=False,
        )
        git_dirty = bool(_r.stdout.strip()) if _r.returncode == 0 else None
    except FileNotFoundError:
        git_dirty = None

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # -- Load processed dataset and partition ---------------------------- #
    target_set = set(target_identity_ids)
    retain_set = set(retain_identity_ids)

    forget_samples: list[dict[str, Any]] = []
    retain_samples: list[dict[str, Any]] = []

    with open(processed_dataset_path) as fh:
        for line in fh:
            row = json.loads(line)
            iid = row.get("identity_id", "")
            entry = {
                "sample_id": row.get("sample_id", ""),
                "identity_id": iid,
                "image_sha256": row.get("image_sha256", ""),
                "question_hash": hashlib.sha256(
                    row.get("question_text", "").encode()
                ).hexdigest()[:16],
                "answer_label": row.get("answer_label"),
                "target_attribute": row.get("target_attribute", ""),
            }
            if iid in target_set:
                forget_samples.append(entry)
            elif iid in retain_set:
                retain_samples.append(entry)

    forget_sample_ids = sorted(s["sample_id"] for s in forget_samples)
    retain_sample_ids = sorted(s["sample_id"] for s in retain_samples)

    forget_samples_manifest = {
        "count": len(forget_samples),
        "identity_ids": sorted(target_identity_ids),
        "sample_ids": forget_sample_ids,
        "entries": forget_samples,
    }
    retain_samples_manifest = {
        "count": len(retain_samples),
        "identity_ids": sorted(retain_identity_ids),
        "sample_ids": retain_sample_ids,
        "entries": retain_samples,
    }

    forget_sha = hashlib.sha256(
        json.dumps(forget_samples_manifest, sort_keys=True).encode()
    ).hexdigest()
    retain_sha = hashlib.sha256(
        json.dumps(retain_samples_manifest, sort_keys=True).encode()
    ).hexdigest()

    processed_ds_sha = _sha256_file(processed_dataset_path)

    # -- Assemble manifest ----------------------------------------------- #
    manifest: dict[str, Any] = {
        "experiment_id": experiment_config.get("experiment_id", ""),
        "selection_version": "pilot-selection-v1",
        "selection_manifest_sha256": selection_manifest_sha256,
        "target_identities": sorted(target_identity_ids),
        "retain_identities": sorted(retain_identity_ids),
        "forget_sample_ids": forget_sample_ids,
        "retain_sample_ids": retain_sample_ids,
        "forget_sample_count": len(forget_samples),
        "retain_sample_count": len(retain_samples),
        "forget_sample_manifest_sha256": forget_sha,
        "retain_sample_manifest_sha256": retain_sha,
        "processed_dataset_sha256": processed_ds_sha,
        "route_probe_sha256": experiment_config.get("dataset", {}).get(
            "route_probe_sha256", ""
        ),
        "leakage_report_sha256": leakage_report_sha256,
        "source_dataset": experiment_config.get("dataset", {}).get(
            "source_dataset", ""
        ),
        "seed": seed,
        "code_commit": code_commit,
        "git_dirty": git_dirty,
    }

    # Self-hash: compute SHA over content with placeholder
    manifest["manifest_sha256"] = "pending"
    content = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    manifest["manifest_sha256"] = hashlib.sha256(content.encode()).hexdigest()

    out.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    )
    logger.info(
        "Intervention manifest: forget=%d, retain=%d, sha=%s",
        len(forget_samples),
        len(retain_samples),
        manifest["manifest_sha256"][:16],
    )
    return out


# --------------------------------------------------------------------------- #
# Manifest serialisation
# --------------------------------------------------------------------------- #

def write_selection_manifest(
    selection: PilotSelection,
    output_path: str | Path,
    *,
    baseline_manifest_sha256: str = "",
    baseline_results_sha256: str = "",
    route_probe_sha256: str = "",
    processed_dataset_sha256: str = "",
    code_commit: str = "",
) -> Path:
    """Write the selection manifest to *output_path* and return the path."""
    selection.baseline_manifest_sha256 = baseline_manifest_sha256
    selection.baseline_results_sha256 = baseline_results_sha256
    selection.route_probe_sha256 = route_probe_sha256
    selection.processed_dataset_sha256 = processed_dataset_sha256
    selection.code_commit = code_commit

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_version": selection.selection_version,
        "seed": selection.seed,
        "target_identities": selection.target_identities,
        "retain_identities": selection.retain_identities,
        "control_identities": selection.control_identities,
        "matching_criteria": selection.matching_criteria,
        "baseline_manifest_sha256": selection.baseline_manifest_sha256,
        "baseline_results_sha256": selection.baseline_results_sha256,
        "route_probe_sha256": selection.route_probe_sha256,
        "processed_dataset_sha256": selection.processed_dataset_sha256,
        "code_commit": selection.code_commit,
        "identity_details": selection.identity_details,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return out


def load_selection_manifest(path: str | Path) -> dict[str, Any]:
    """Load a previously written selection manifest."""
    return json.loads(Path(path).read_text())


def selection_manifest_sha256(path: str | Path) -> str:
    """Compute SHA-256 of the manifest file on disk."""
    return _sha256_file(path)
