"""E2C-v3 causal metrics.

Computes the intervention and alignment metrics for the factorial audit.
All metrics operate on prediction records, not raw model outputs.
"""
from collections import defaultdict


def intervention_change_rate(
    correct_code_preds: dict[str, str],
    wrong_code_preds: dict[str, str],
) -> float:
    """P(Y(X_i, C_j) ≠ Y(X_i, C_i)).

    Args:
        correct_code_preds: {probe_key: prediction} for correct code condition
        wrong_code_preds: {probe_key: prediction} for wrong code condition
            where probe_key = "iid__image_id"
    """
    changes = 0
    total = 0
    for key, correct_pred in correct_code_preds.items():
        if key in wrong_code_preds:
            total += 1
            if correct_pred != wrong_code_preds[key]:
                changes += 1
    return changes / total if total > 0 else 0.0


def code_target_alignment(
    wrong_code_preds: dict[str, str],
    wrong_code_expected: dict[str, str],
) -> float:
    """Align_C = P(Y(X_i, C_j) = target(C_j)).

    Fraction of wrong-code predictions that follow the wrong code's target.
    """
    aligns = 0
    total = 0
    for key, pred in wrong_code_preds.items():
        if key in wrong_code_expected:
            total += 1
            if pred.lower() == wrong_code_expected[key].lower():
                aligns += 1
    return aligns / total if total > 0 else 0.0


def image_target_alignment(
    wrong_code_preds: dict[str, str],
    image_expected: dict[str, str],
) -> float:
    """Align_X = P(Y(X_i, C_j) = target(X_i)).

    Fraction of wrong-code predictions that persist with the image's target.
    """
    aligns = 0
    total = 0
    for key, pred in wrong_code_preds.items():
        if key in image_expected:
            total += 1
            if pred.lower() == image_expected[key].lower():
                aligns += 1
    return aligns / total if total > 0 else 0.0


def causal_control_contrast(
    m_code_follow: float,
    d_code_follow: float,
) -> float:
    """M code-following minus D code-following.

    Positive = M more sensitive to code than D.
    """
    return m_code_follow - d_code_follow


def shuffled_mapping_agreement(
    preds: dict[str, str],
    shuffled_expected: dict[str, str],
) -> float:
    """Fraction of predictions following the shuffled mapping."""
    agrees = 0
    total = 0
    for key, pred in preds.items():
        if key in shuffled_expected:
            total += 1
            if pred.lower() == shuffled_expected[key].lower():
                agrees += 1
    return agrees / total if total > 0 else 0.0


def compute_per_identity_metrics(
    probe_results: list[dict],
    identity_ids: list[str],
) -> dict[str, dict]:
    """Compute all metrics broken down by identity."""
    by_id = defaultdict(lambda: {
        "bare_correct": 0, "bare_total": 0,
        "correct_code_correct": 0, "correct_code_total": 0,
        "wrong_code_follow_code": 0, "wrong_code_follow_image": 0,
        "wrong_code_total": 0,
        "wrong_code_change": 0, "wrong_code_change_total": 0,
    })

    for r in probe_results:
        iid = r["identity_id"]
        ptype = r["probe_type"]
        pred = r.get("prediction", "").strip()

        if ptype == "bare_image":
            by_id[iid]["bare_total"] += 1
            if pred.lower() == r["expected_alias"].lower():
                by_id[iid]["bare_correct"] += 1

        elif ptype == "image_correct_code":
            by_id[iid]["correct_code_total"] += 1
            if pred.lower() == r["expected_alias"].lower():
                by_id[iid]["correct_code_correct"] += 1

        elif ptype == "image_wrong_code":
            by_id[iid]["wrong_code_total"] += 1
            exp_code = r.get("expected_alias_if_follow_code", "")
            exp_img = r.get("expected_alias_if_follow_image", "")
            if pred.lower() == exp_code.lower():
                by_id[iid]["wrong_code_follow_code"] += 1
            if pred.lower() == exp_img.lower():
                by_id[iid]["wrong_code_follow_image"] += 1

    per_id = {}
    for iid in identity_ids:
        m = by_id[iid]
        per_id[iid] = {
            "bare_acc": (
                m["bare_correct"] / m["bare_total"]
                if m["bare_total"] > 0 else 0.0
            ),
            "correct_code_acc": (
                m["correct_code_correct"] / m["correct_code_total"]
                if m["correct_code_total"] > 0 else 0.0
            ),
            "wrong_code_follow_code_rate": (
                m["wrong_code_follow_code"] / m["wrong_code_total"]
                if m["wrong_code_total"] > 0 else 0.0
            ),
            "wrong_code_follow_image_rate": (
                m["wrong_code_follow_image"] / m["wrong_code_total"]
                if m["wrong_code_total"] > 0 else 0.0
            ),
            "n_bare": m["bare_total"],
            "n_correct_code": m["correct_code_total"],
            "n_wrong_code": m["wrong_code_total"],
        }
    return per_id
