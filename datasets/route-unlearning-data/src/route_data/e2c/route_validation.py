"""E2C route validation — leakage checks and R1–R7 gate logic.

Leakage validation:
    Hard-fail if test/val images appear in training, wrong-name alias
    equals true alias, M contains direct samples, etc.

Gate criteria (R1–R7):
    R1: I2N accuracy M >= 0.90
    R2: NAME accuracy M >= 0.90 and NAME_D <= 0.65 (or M-D >= 0.25)
    R3: DV_M >= 0.80 and DV_D >= 0.80
    R4: |WrongNameEffect_M| > |WrongNameEffect_D| and CI excludes 0
    R5: |ConflictEffect_M| > |ConflictEffect_D| and CI excludes 0
    R6: M-shuffled agreement_with_shuffled > agreement_with_true
    R7: Visual controls preserved (accuracy decrease <= 0.05)

Aggregate:
    ROUTE_ESTABLISHED = all gates PASS.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# Leakage validation
# --------------------------------------------------------------------------- #

def validate_leakage(
    *,
    train_records: dict[str, list[dict[str, Any]]],
    test_images: list[dict[str, Any]],
    image_splits: list[dict[str, Any]],
    wn_pairs: list[dict[str, Any]],
    alias_map: dict[str, str],
    true_mapping: dict[str, str],
    shuffled_mapping: dict[str, str],
    experimental_ids: list[str],
    calibration_ids: list[str],
) -> dict[str, Any]:
    """Run all leakage checks. Hard-fail on any violation.

    Returns report dict. Raises ValueError on violations.
    """
    errors: list[str] = []

    # Collect train image IDs per condition
    train_image_ids: dict[str, set[str]] = {}
    for condition, records in train_records.items():
        ids = {
            r["image_id"] for r in records
            if r.get("image_id") is not None
        }
        train_image_ids[condition] = ids

    test_image_ids = {img["image_id"] for img in test_images}

    # Test image not in training
    for condition, ids in train_image_ids.items():
        overlap = test_image_ids & ids
        if overlap:
            errors.append(
                f"{condition}: {len(overlap)} test images appear in training"
            )

    # Validation images not in training
    val_image_ids = {
        rec["image_id"] for rec in image_splits
        if rec["split"] == "validation"
    }
    for condition, ids in train_image_ids.items():
        overlap = val_image_ids & ids
        if overlap:
            errors.append(
                f"{condition}: {len(overlap)} validation images in training"
            )

    # Wrong-name alias != true alias
    for pair in wn_pairs:
        if pair["wrong_alias"] == pair["correct_alias"]:
            errors.append(
                f"Wrong-name alias equals true alias for {pair['identity_id']}"
            )

    # Wrong-name alias has opposite label
    for pair in wn_pairs:
        if pair["wrong_label"] == pair["true_label"]:
            errors.append(
                f"Wrong-name label same as true label for {pair['identity_id']}"
            )

    # M has zero image_to_attribute target-fact samples
    m_records = train_records.get("M", [])
    m_direct = [r for r in m_records if r["task"] == "image_to_attribute"]
    if m_direct:
        errors.append(f"M has {len(m_direct)} image_to_attribute samples")

    # D has zero name_to_attribute samples
    d_records = train_records.get("D", [])
    d_name = [r for r in d_records if r["task"] == "name_to_attribute"]
    if d_name:
        errors.append(f"D has {len(d_name)} name_to_attribute samples")

    # M-shuffled uses shuffled mapping
    ms_records = train_records.get("M_shuffled", [])
    for r in ms_records:
        if r["task"] == "name_to_attribute":
            id_ = r["identity_id"]
            expected = shuffled_mapping.get(id_)
            if r["answer"] != expected:
                errors.append(
                    f"M-shuffled {r['sample_id']}: answer {r['answer']!r} "
                    f"!= shuffled {expected!r}"
                )

    # Experimental and calibration identities don't overlap
    exp_set = set(experimental_ids)
    cal_set = set(calibration_ids)
    if exp_set & cal_set:
        errors.append("Experimental and calibration identities overlap")

    # Unique sample IDs within each condition
    for condition, records in train_records.items():
        ids = [r["sample_id"] for r in records]
        if len(ids) != len(set(ids)):
            errors.append(f"{condition}: duplicate sample IDs")

    # M and D image populations identical
    m_images = {r["image_id"] for r in m_records if r.get("image_id")}
    d_images = {
        r["image_id"] for r in d_records
        if r.get("image_id") and r["task"] == "image_to_attribute"
    }
    if m_images != d_images:
        errors.append("M and D image populations differ")

    report = {
        "pass": len(errors) == 0,
        "n_errors": len(errors),
        "errors": errors,
    }

    if errors:
        raise ValueError(
            f"Leakage validation failed ({len(errors)} errors):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return report


# --------------------------------------------------------------------------- #
# R1–R7 gate logic
# --------------------------------------------------------------------------- #

def evaluate_r1(
    i2n_accuracy_m: float,
    *,
    threshold: float = 0.90,
    preferred_threshold: float = 0.95,
) -> dict[str, Any]:
    """R1 — mediated model learns image-to-identity."""
    return {
        "gate": "R1",
        "metric": "I2N_accuracy_M",
        "value": i2n_accuracy_m,
        "threshold": threshold,
        "preferred_threshold": preferred_threshold,
        "status": "PASS" if i2n_accuracy_m >= threshold else "FAIL",
    }


def evaluate_r2(
    name_accuracy_m: float,
    name_accuracy_d: float,
    *,
    name_threshold: float = 0.90,
    d_separation_threshold: float = 0.65,
    difference_threshold: float = 0.25,
) -> dict[str, Any]:
    """R2 — mediated model learns identity-to-fact."""
    m_pass = name_accuracy_m >= name_threshold
    d_separation = (
        name_accuracy_d <= d_separation_threshold
        or (name_accuracy_m - name_accuracy_d) >= difference_threshold
    )
    return {
        "gate": "R2",
        "metric": "NAME_accuracy",
        "value_m": name_accuracy_m,
        "value_d": name_accuracy_d,
        "name_threshold": name_threshold,
        "d_separation_threshold": d_separation_threshold,
        "difference_threshold": difference_threshold,
        "m_pass": m_pass,
        "d_separation_pass": d_separation,
        "status": "PASS" if (m_pass and d_separation) else "FAIL",
    }


def evaluate_r3(
    dv_m: float,
    dv_d: float,
    *,
    threshold: float = 0.80,
) -> dict[str, Any]:
    """R3 — both conditions solve the final image task."""
    return {
        "gate": "R3",
        "metric": "DV_syn_accuracy",
        "value_m": dv_m,
        "value_d": dv_d,
        "threshold": threshold,
        "status": "PASS" if (dv_m >= threshold and dv_d >= threshold) else "FAIL",
    }


def evaluate_r4(
    abs_wn_effect_m: float,
    abs_wn_effect_d: float,
    ci_excludes_zero: bool,
    *,
    bootstrap_ci: dict[str, float] | None = None,
) -> dict[str, Any]:
    """R4 — M more sensitive to wrong-name intervention."""
    magnitude = abs_wn_effect_m > abs_wn_effect_d
    return {
        "gate": "R4",
        "metric": "abs_WrongNameEffect",
        "value_m": abs_wn_effect_m,
        "value_d": abs_wn_effect_d,
        "magnitude_pass": magnitude,
        "ci_excludes_zero": ci_excludes_zero,
        "bootstrap_ci": bootstrap_ci or {},
        "status": "PASS" if (magnitude and ci_excludes_zero) else "FAIL",
    }


def evaluate_r5(
    abs_conflict_effect_m: float,
    abs_conflict_effect_d: float,
    ci_excludes_zero: bool,
    *,
    bootstrap_ci: dict[str, float] | None = None,
) -> dict[str, Any]:
    """R5 — M more sensitive to visual/text conflict."""
    magnitude = abs_conflict_effect_m > abs_conflict_effect_d
    return {
        "gate": "R5",
        "metric": "abs_ConflictEffect",
        "value_m": abs_conflict_effect_m,
        "value_d": abs_conflict_effect_d,
        "magnitude_pass": magnitude,
        "ci_excludes_zero": ci_excludes_zero,
        "bootstrap_ci": bootstrap_ci or {},
        "status": "PASS" if (magnitude and ci_excludes_zero) else "FAIL",
    }


def evaluate_r6(
    agreement_shuffled: float,
    agreement_true: float,
    *,
    preferred_threshold: float = 0.80,
) -> dict[str, Any]:
    """R6 — M-shuffled follows the shuffled association."""
    follows_shuffled = agreement_shuffled > agreement_true
    meets_threshold = agreement_shuffled >= preferred_threshold
    return {
        "gate": "R6",
        "metric": "M_shuffled_mapping_agreement",
        "agreement_with_shuffled": agreement_shuffled,
        "agreement_with_true": agreement_true,
        "follows_shuffled": follows_shuffled,
        "meets_preferred_threshold": meets_threshold,
        "preferred_threshold": preferred_threshold,
        "status": "PASS" if follows_shuffled else "FAIL",
    }


def evaluate_r7(
    visual_control_results: dict[str, dict[str, float]],
    *,
    max_absolute_decrease: float = 0.05,
) -> dict[str, Any]:
    """R7 — ordinary visual controls preserved.

    Parameters
    ----------
    visual_control_results:
        Dict mapping attribute_name -> {"base_accuracy": float, "trained_accuracy": float}.
    """
    failures: list[str] = []
    for attr, scores in visual_control_results.items():
        base = scores.get("base_accuracy", 1.0)
        trained = scores.get("trained_accuracy", 0.0)
        decrease = base - trained
        if decrease > max_absolute_decrease:
            failures.append(
                f"{attr}: decrease {decrease:.3f} > {max_absolute_decrease}"
            )

    return {
        "gate": "R7",
        "metric": "visual_control_preservation",
        "results": visual_control_results,
        "max_absolute_decrease": max_absolute_decrease,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


# --------------------------------------------------------------------------- #
# Aggregate decision
# --------------------------------------------------------------------------- #

def aggregate_route_decision(
    gates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute ROUTE_ESTABLISHED from R1–R7 gate results.

    Never averages gates. All must PASS.
    """
    mandatory = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    all_pass = all(
        gates.get(g, {}).get("status") == "PASS" for g in mandatory
    )

    return {
        "route_established": all_pass,
        "controlled_unlearning_allowed": all_pass,
        "gates": {g: gates.get(g, {"status": "MISSING"}) for g in mandatory},
        "all_mandatory_pass": all_pass,
    }


# --------------------------------------------------------------------------- #
# Failure taxonomy
# --------------------------------------------------------------------------- #

def classify_failure(
    gates: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """Classify route-establishment failures.

    Returns a list of failure descriptions.
    """
    failures: list[dict[str, str]] = []

    r1 = gates.get("R1", {})
    if r1.get("status") == "FAIL":
        failures.append({
            "code": "A",
            "pattern": "I2N weak",
            "interpretation": "Image -> Identity was not learned robustly",
            "action": "Inspect identity consistency; adjust training via calibration",
        })

    r2 = gates.get("R2", {})
    if r2.get("status") == "FAIL":
        failures.append({
            "code": "B",
            "pattern": "NAME_M weak",
            "interpretation": "Identity -> Fact association not learned sufficiently",
            "action": "Increase or rebalance M2 exposure using calibration identities",
        })

    r3 = gates.get("R3", {})
    if r3.get("status") == "FAIL":
        i2n_m = gates.get("R1", {}).get("value", 0)
        name_m = gates.get("R2", {}).get("value_m", 0)
        if i2n_m >= 0.90 and name_m >= 0.90:
            failures.append({
                "code": "C",
                "pattern": "Composition failure",
                "interpretation": "Model learns each mapping separately but does not compose",
                "action": "Investigate prompt-supported composition condition",
            })
        else:
            failures.append({
                "code": "C",
                "pattern": "DV accuracy below threshold",
                "interpretation": "One or both conditions fail to solve the final image task",
                "action": "Adjust training budget or learning rate",
            })

    r4 = gates.get("R4", {})
    r5 = gates.get("R5", {})
    if r4.get("status") == "FAIL" or r5.get("status") == "FAIL":
        failures.append({
            "code": "D",
            "pattern": "M and D show similar WN/VTC sensitivity",
            "interpretation": "Behavioral probes do not establish distinct route dependence",
            "action": "Strengthen mediated training; re-examine D alias associations",
        })

    r6 = gates.get("R6", {})
    if r6.get("status") == "FAIL":
        failures.append({
            "code": "E",
            "pattern": "M-shuffled does not follow shuffled mapping",
            "interpretation": "Behavior is controlled by another route",
            "action": "Do not claim a mediated route",
        })

    r7 = gates.get("R7", {})
    if r7.get("status") == "FAIL":
        failures.append({
            "code": "F",
            "pattern": "Visual controls collapse",
            "interpretation": "Route establishment training broadly damages perception",
            "action": "Return to calibration and reduce training strength",
        })

    return failures
