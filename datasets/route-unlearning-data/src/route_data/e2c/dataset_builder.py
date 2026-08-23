"""E2C training dataset builder.

Generates condition-specific training records for M, D, and M-shuffled
from frozen manifests. Enforces hard invariants:

- M has zero image_to_attribute target-fact samples
- D has zero name_to_attribute target-fact samples
- M-shuffled uses only shuffled mapping labels
- M and D image populations are identical
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .synthetic_manifest import (
    PROMPT_REGISTRY,
    get_prompt,
)

# --------------------------------------------------------------------------- #
# Record builders
# --------------------------------------------------------------------------- #

def _build_image_to_identity_records(
    identity_id: str,
    alias: str,
    train_image_ids: list[str],
    *,
    condition: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """M1 — image to identity records."""
    prompt_ids = [
        pid for pid, v in PROMPT_REGISTRY.items()
        if v["family"] == "image_to_identity" and "train" in pid
    ]
    records = []
    for img_id in train_image_ids:
        prompt_id = rng.choice(prompt_ids)
        prompt_text = get_prompt(prompt_id)
        img_path = f"e2c/data/processed/{identity_id}/{img_id}.png"
        records.append({
            "sample_id": f"e2c_{condition.lower()}_{identity_id}_{img_id}_i2n",
            "condition": condition,
            "task": "image_to_identity",
            "identity_id": identity_id,
            "alias": alias,
            "image_id": img_id,
            "image_path": img_path,  # filled from split manifest
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "answer": alias,
            "split": "train",
        })
    return records


def _build_name_to_attribute_records(
    identity_id: str,
    alias: str,
    label: str,
    *,
    condition: str,
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """M2 — identity to target fact records (text-only)."""
    prompt_ids = [
        pid for pid, v in PROMPT_REGISTRY.items()
        if v["family"] == "name_to_attribute" and "train" in pid
    ]
    records = []
    for i in range(count):
        prompt_id = rng.choice(prompt_ids)
        prompt_text = get_prompt(prompt_id, alias=alias)
        records.append({
            "sample_id": f"e2c_{condition.lower()}_{identity_id}_name_fact_{i:03d}",
            "condition": condition,
            "task": "name_to_attribute",
            "identity_id": identity_id,
            "alias": alias,
            "image_id": None,
            "image_path": None,
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "answer": label,
            "split": "train",
        })
    return records


def _build_image_to_attribute_records(
    identity_id: str,
    train_image_ids: list[str],
    label: str,
    *,
    condition: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """D — image to target fact records."""
    prompt_ids = [
        pid for pid, v in PROMPT_REGISTRY.items()
        if v["family"] == "image_to_attribute" and "train" in pid
    ]
    records = []
    for img_id in train_image_ids:
        prompt_id = rng.choice(prompt_ids)
        prompt_text = get_prompt(prompt_id)
        img_path = f"e2c/data/processed/{identity_id}/{img_id}.png"
        records.append({
            "sample_id": f"e2c_{condition.lower()}_{identity_id}_{img_id}_direct_fact",
            "condition": condition,
            "task": "image_to_attribute",
            "identity_id": identity_id,
            "image_id": img_id,
            "image_path": img_path,
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "answer": label,
            "split": "train",
        })
    return records


def _build_text_exposure_records(
    alias: str,
    identity_id: str,
    *,
    condition: str,
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """D text-exposure controls — alias-neutral deterministic text examples.

    These match M's text-only alias exposure without revealing the target mapping.
    """
    records = []
    for i in range(count):
        records.append({
            "sample_id": f"e2c_{condition.lower()}_{identity_id}_text_ctrl_{i:03d}",
            "condition": condition,
            "task": "text_exposure_control",
            "identity_id": identity_id,
            "alias": alias,
            "image_id": None,
            "image_path": None,
            "prompt_id": "e2c_text_control_v1",
            "prompt": f"Which synthetic identifier is being referenced: {alias}?",
            "answer": alias,
            "split": "train",
        })
    return records


# --------------------------------------------------------------------------- #
# Condition builders
# --------------------------------------------------------------------------- #

def build_condition_m(
    *,
    alias_map: dict[str, str],
    true_mapping: dict[str, str],
    image_splits: dict[str, dict[str, list[int]]],
    identity_ids: list[str] | None = None,
    experimental_ids: list[str] | None = None,
    population: str = "experimental",
    seed: int,
    name_to_attr_count: int = 10,
) -> list[dict[str, Any]]:
    """Build condition M training records (P0-4: population-aware).

    M contains:
    - M1: image_to_identity (image -> alias)
    - M2: name_to_attribute (alias -> fact) [text-only]

    Hard invariant: M has zero image_to_attribute records.

    Parameters
    ----------
    identity_ids : explicit list of identity IDs to use.
    experimental_ids : deprecated alias for identity_ids.
    population : "experimental" or "calibration".
    """
    ids = identity_ids or experimental_ids or []
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for id_ in sorted(ids):
        alias = alias_map[id_]
        label = true_mapping[id_]
        train_indices = image_splits[id_]["train"]
        train_image_ids = [f"{id_}_img_{idx:03d}" for idx in train_indices]

        records.extend(_build_image_to_identity_records(
            id_, alias, train_image_ids,
            condition="M", rng=rng,
        ))
        records.extend(_build_name_to_attribute_records(
            id_, alias, label,
            condition="M", count=name_to_attr_count, rng=rng,
        ))

    return records


def build_condition_d(
    *,
    alias_map: dict[str, str],
    true_mapping: dict[str, str],
    image_splits: dict[str, dict[str, list[int]]],
    identity_ids: list[str] | None = None,
    experimental_ids: list[str] | None = None,
    population: str = "experimental",
    seed: int,
    text_exposure_count: int = 10,
) -> list[dict[str, Any]]:
    """Build condition D training records (P0-4: population-aware).

    D contains:
    - image_to_attribute (image -> fact) using same train images as M
    - text_exposure_control (alias-neutral text) to match M's text exposure

    Hard invariant: D has zero name_to_attribute records.
    """
    ids = identity_ids or experimental_ids or []
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for id_ in sorted(ids):
        alias = alias_map[id_]
        label = true_mapping[id_]
        train_indices = image_splits[id_]["train"]
        train_image_ids = [f"{id_}_img_{idx:03d}" for idx in train_indices]

        records.extend(_build_image_to_attribute_records(
            id_, train_image_ids, label,
            condition="D", rng=rng,
        ))
        records.extend(_build_text_exposure_records(
            alias, id_,
            condition="D", count=text_exposure_count, rng=rng,
        ))

    return records


def build_condition_m_shuffled(
    *,
    alias_map: dict[str, str],
    shuffled_mapping: dict[str, str],
    image_splits: dict[str, dict[str, list[int]]],
    identity_ids: list[str] | None = None,
    experimental_ids: list[str] | None = None,
    population: str = "experimental",
    seed: int,
    name_to_attr_count: int = 10,
) -> list[dict[str, Any]]:
    """Build condition M-shuffled training records (P0-4: population-aware).

    Same generation path as M, but uses the shuffled mapping.
    """
    ids = identity_ids or experimental_ids or []
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for id_ in sorted(ids):
        alias = alias_map[id_]
        label = shuffled_mapping[id_]
        train_indices = image_splits[id_]["train"]
        train_image_ids = [f"{id_}_img_{idx:03d}" for idx in train_indices]

        records.extend(_build_image_to_identity_records(
            id_, alias, train_image_ids,
            condition="M_shuffled", rng=rng,
        ))
        records.extend(_build_name_to_attribute_records(
            id_, alias, label,
            condition="M_shuffled", count=name_to_attr_count, rng=rng,
        ))

    return records


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_condition_invariants(
    m_records: list[dict],
    d_records: list[dict],
    ms_records: list[dict],
    *,
    true_mapping: dict[str, str],
    shuffled_mapping: dict[str, str],
) -> dict[str, Any]:
    """Validate hard invariants across conditions.

    Returns a report dict. Raises ValueError on invariant violations.
    """
    errors: list[str] = []

    # M has zero image_to_attribute target-fact samples
    m_direct = [r for r in m_records if r["task"] == "image_to_attribute"]
    if m_direct:
        errors.append(f"M has {len(m_direct)} image_to_attribute samples (must be 0)")

    # D has zero name_to_attribute samples
    d_name = [r for r in d_records if r["task"] == "name_to_attribute"]
    if d_name:
        errors.append(f"D has {len(d_name)} name_to_attribute samples (must be 0)")

    # M-shuffled uses only shuffled labels
    for r in ms_records:
        if r["task"] == "name_to_attribute":
            id_ = r["identity_id"]
            expected = shuffled_mapping[id_]
            if r["answer"] != expected:
                errors.append(
                    f"M-shuffled record {r['sample_id']} has answer "
                    f"{r['answer']!r} but shuffled mapping says {expected!r}"
                )

    # M and D image populations identical
    m_images = {
        r["image_id"] for r in m_records
        if r["image_id"] is not None
    }
    d_images = {
        r["image_id"] for r in d_records
        if r["image_id"] is not None and r["task"] == "image_to_attribute"
    }
    if m_images != d_images:
        errors.append(
            f"M/D image population mismatch: "
            f"M has {len(m_images)} images, D has {len(d_images)} images"
        )

    # M and D image-conditioned exposure counts identical
    m_img_count = len([r for r in m_records if r["image_id"] is not None])
    d_img_count = len([r for r in d_records if r["task"] == "image_to_attribute"])
    if m_img_count != d_img_count:
        errors.append(
            f"M/D image-conditioned count mismatch: "
            f"M={m_img_count}, D={d_img_count}"
        )

    # Unique sample IDs within each condition
    for name, recs in [("M", m_records), ("D", d_records), ("M_shuffled", ms_records)]:
        ids = [r["sample_id"] for r in recs]
        if len(ids) != len(set(ids)):
            errors.append(f"{name} has duplicate sample IDs")

    # Cross-condition sample ID uniqueness
    all_ids = (
        [r["sample_id"] for r in m_records]
        + [r["sample_id"] for r in d_records]
        + [r["sample_id"] for r in ms_records]
    )
    if len(all_ids) != len(set(all_ids)):
        errors.append("Cross-condition duplicate sample IDs detected")

    report = {
        "pass": len(errors) == 0,
        "errors": errors,
        "m_total": len(m_records),
        "d_total": len(d_records),
        "ms_total": len(ms_records),
        "m_image_count": m_img_count,
        "d_image_count": d_img_count,
        "m_task_counts": _task_counts(m_records),
        "d_task_counts": _task_counts(d_records),
        "ms_task_counts": _task_counts(ms_records),
    }

    if errors:
        raise ValueError(
            f"Condition invariant violations ({len(errors)}):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return report


def _task_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        t = r["task"]
        counts[t] = counts.get(t, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Condition matching report
# --------------------------------------------------------------------------- #

def build_condition_matching_report(
    m_records: list[dict],
    d_records: list[dict],
    ms_records: list[dict],
    *,
    alias_map: dict[str, str],
    optimizer_steps: int,
) -> dict[str, Any]:
    """Build machine-readable condition matching report.

    Hard-fail if image exposures differ between M and D.
    """
    m_images = {r["image_id"] for r in m_records if r["image_id"] is not None}
    d_images = {
        r["image_id"] for r in d_records
        if r["image_id"] is not None and r["task"] == "image_to_attribute"
    }

    # Alias token exposure (for documentation)
    _m_alias_tokens = sum(
        len(r["alias"]) for r in m_records if r["task"] == "name_to_attribute"
    )
    _d_alias_tokens = sum(
        len(r["alias"]) for r in d_records if r["task"] == "text_exposure_control"
    )

    report = {
        "M": {
            "image_examples": len([r for r in m_records if r["image_id"] is not None]),
            "text_examples": len([r for r in m_records if r["image_id"] is None]),
            "total": len(m_records),
            "unique_train_images": len(m_images),
            "optimizer_steps": optimizer_steps,
        },
        "D": {
            "image_examples": len([r for r in d_records if r["task"] == "image_to_attribute"]),
            "text_examples": len([r for r in d_records if r["task"] == "text_exposure_control"]),
            "total": len(d_records),
            "unique_train_images": len(d_images),
            "optimizer_steps": optimizer_steps,
        },
        "M_shuffled": {
            "image_examples": len([r for r in ms_records if r["image_id"] is not None]),
            "text_examples": len([r for r in ms_records if r["image_id"] is None]),
            "total": len(ms_records),
            "unique_train_images": len({
                r["image_id"] for r in ms_records if r["image_id"] is not None
            }),
            "optimizer_steps": optimizer_steps,
        },
        "image_population_match": m_images == d_images,
    }

    if not report["image_population_match"]:
        raise ValueError("M and D image populations differ — hard fail")

    return report


# --------------------------------------------------------------------------- #
# Population isolation validation (P0-4)
# --------------------------------------------------------------------------- #

def validate_population_isolation(
    *,
    calibration_records: list[dict],
    experimental_records: list[dict],
    calibration_ids: list[str],
    experimental_ids: list[str],
) -> dict[str, Any]:
    """P0-4: Verify calibration and experimental populations are disjoint.

    Hard-fail on any identity overlap.
    """
    errors: list[str] = []
    cal_set = set(calibration_ids)
    exp_set = set(experimental_ids)

    # No overlap between populations
    overlap = cal_set & exp_set
    if overlap:
        errors.append(f"Identity overlap between populations: {sorted(overlap)}")

    # Calibration records only contain calibration identities
    cal_id_set = {r["identity_id"] for r in calibration_records}
    unexpected_cal = cal_id_set - cal_set
    if unexpected_cal:
        errors.append(
            f"Calibration dataset contains non-calibration IDs: {sorted(unexpected_cal)}"
        )

    # Experimental records only contain experimental identities
    exp_id_set = {r["identity_id"] for r in experimental_records}
    unexpected_exp = exp_id_set - exp_set
    if unexpected_exp:
        errors.append(
            f"Experimental dataset contains non-experimental IDs: {sorted(unexpected_exp)}"
        )

    return {
        "pass": len(errors) == 0,
        "errors": errors,
        "calibration_identity_count": len(cal_id_set),
        "experimental_identity_count": len(exp_id_set),
    }


# --------------------------------------------------------------------------- #
# Audit validation (P0-6)
# --------------------------------------------------------------------------- #

def validate_audit_completeness(
    audit_records: list[dict],
    *,
    expected_image_ids: list[str] | None = None,
) -> dict[str, Any]:
    """P0-6 (hardened): Validate identity audit is research-valid.

    Checks:
    - audit_status must be exactly 'pass', 'fail', or 'pending'
    - 'pass' requires all semantic fields to have correct values
    - coverage: audit image IDs must match expected_image_ids
    - no duplicate audit entries
    """
    errors: list[str] = []
    valid_statuses = {"pass", "fail", "pending"}

    pending_count = 0
    fail_count = 0
    pass_count = 0
    seen_ids: list[str] = []

    for rec in audit_records:
        img_id = rec.get("image_id", "?")
        seen_ids.append(img_id)
        status = rec.get("audit_status", "pending")

        # Reject unknown statuses
        if status not in valid_statuses:
            errors.append(
                f"Image {img_id}: unknown audit_status={status!r}")
            continue

        if status == "pending":
            pending_count += 1
            errors.append(
                f"Image {img_id}: audit_status is pending")
            continue

        if status == "fail":
            fail_count += 1
            errors.append(
                f"Image {img_id}: audit_status is fail")
            continue

        # status == "pass" — check semantic values
        pass_count += 1

        # identity_consistent must be True
        if rec.get("identity_consistent") is not True:
            errors.append(
                f"Image {img_id}: identity_consistent is "
                f"{rec.get('identity_consistent')!r} (must be True)")

        # duplicate must be False
        if rec.get("duplicate") is not False:
            errors.append(
                f"Image {img_id}: duplicate is "
                f"{rec.get('duplicate')!r} (must be False)")

        # corrupted must be False
        if rec.get("corrupted") is not False:
            errors.append(
                f"Image {img_id}: corrupted is "
                f"{rec.get('corrupted')!r} (must be False)")

        # watermark must be False
        if rec.get("watermark") is not False:
            errors.append(
                f"Image {img_id}: watermark is "
                f"{rec.get('watermark')!r} (must be False)")

        # alias_leakage must be False
        if rec.get("alias_leakage") is not False:
            errors.append(
                f"Image {img_id}: alias_leakage is "
                f"{rec.get('alias_leakage')!r} (must be False)")

        # target_fact_leakage must be False
        if rec.get("target_fact_leakage") is not False:
            errors.append(
                f"Image {img_id}: target_fact_leakage is "
                f"{rec.get('target_fact_leakage')!r} (must be False)")

    # Check for duplicate audit entries
    if len(seen_ids) != len(set(seen_ids)):
        dupes = [x for x in seen_ids if seen_ids.count(x) > 1]
        errors.append(
            f"Duplicate audit entries for: {sorted(set(dupes))[:5]}")

    # Coverage check: audit image IDs must match expected
    if expected_image_ids is not None:
        expected_set = set(expected_image_ids)
        audit_set = set(seen_ids)
        missing = expected_set - audit_set
        extra = audit_set - expected_set
        if missing:
            errors.append(
                f"Audit missing {len(missing)} images: "
                f"{sorted(missing)[:5]}")
        if extra:
            errors.append(
                f"Audit has {len(extra)} unexpected images: "
                f"{sorted(extra)[:5]}")

    return {
        "pass": len(errors) == 0,
        "n_errors": len(errors),
        "errors": errors[:30],  # cap for readability
        "total_images": len(audit_records),
        "pass_count": pass_count,
        "pending_count": pending_count,
        "fail_count": fail_count,
    }


# --------------------------------------------------------------------------- #
# JSONL writer
# --------------------------------------------------------------------------- #

def write_training_jsonl(
    records: list[dict[str, Any]],
    path: str | Path,
) -> str:
    """Write training records as JSONL and return SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(record, sort_keys=True) + "\n" for record in records)
    from .synthetic_manifest import sha256_file
    return sha256_file(path)
