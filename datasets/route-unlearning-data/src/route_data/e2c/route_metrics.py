"""E2C route metrics and identity-clustered bootstrap.

Computes:
    NameEffect      = M_IPN - M_DV
    WrongNameEffect = M_WN - M_IPN
    ConflictEffect  = M_VTC - M_DV

Between-condition contrasts:
    Delta_NAME       = NAME_accuracy_M - NAME_accuracy_D
    Delta_WN_route   = |WrongNameEffect_M| - |WrongNameEffect_D|
    Delta_VTC_route  = |ConflictEffect_M| - |ConflictEffect_D|

Identity-clustered bootstrap for uncertainty estimation.
"""

from __future__ import annotations

import random
from typing import Any

# --------------------------------------------------------------------------- #
# Signed margin computation
# --------------------------------------------------------------------------- #

def compute_signed_margin(
    score_yes: float,
    score_no: float,
    expected_answer: str,
) -> float:
    """Signed margin relative to the correct answer.

    Positive margin means the expected answer is preferred.
    """
    if expected_answer == "Yes":
        return score_yes - score_no
    elif expected_answer == "No":
        return score_no - score_yes
    else:
        raise ValueError(f"expected_answer must be 'Yes' or 'No', got {expected_answer!r}")


def compute_accuracy_from_probes(
    probe_results: list[dict[str, Any]],
) -> float:
    """Compute accuracy from probe results with signed margins.

    A probe is correct if signed_margin > 0 (ties count as incorrect).
    """
    if not probe_results:
        return 0.0
    correct = sum(1 for p in probe_results if p["signed_answer_margin"] > 0)
    return correct / len(probe_results)


def compute_i2n_accuracy(
    probe_results: list[dict[str, Any]],
) -> float:
    """Compute normalized exact-match accuracy for I2N probes."""
    if not probe_results:
        return 0.0
    correct = 0
    for p in probe_results:
        predicted = _normalize_text(p.get("predicted_answer", ""))
        expected = _normalize_text(p.get("expected_answer", ""))
        if predicted == expected and predicted:
            correct += 1
    return correct / len(probe_results)


def _normalize_text(text: str) -> str:
    """Conservative normalization: trim, case-fold, strip punctuation."""
    text = text.strip().lower()
    for ch in ".,;:!?\"'()[]{}":
        text = text.replace(ch, "")
    return text.strip()


# --------------------------------------------------------------------------- #
# Route effect computation
# --------------------------------------------------------------------------- #

def compute_route_effects(
    *,
    dv_results: list[dict[str, Any]],
    ipn_results: list[dict[str, Any]],
    wn_results: list[dict[str, Any]],
    vtc_results: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute per-image route effects using signed answer margins.

    Returns dict with NameEffect, WrongNameEffect, ConflictEffect,
    and their absolute values.
    """
    # Build per-image lookup
    dv_by_image = {r["image_id"]: r for r in dv_results}
    ipn_by_image = {r["image_id"]: r for r in ipn_results}
    wn_by_image = {r["image_id"]: r for r in wn_results}
    vtc_by_image = {r["image_id"]: r for r in vtc_results}

    # Matched images
    common_images = sorted(
        set(dv_by_image.keys())
        & set(ipn_by_image.keys())
        & set(wn_by_image.keys())
        & set(vtc_by_image.keys())
    )

    name_effects: list[float] = []
    wrong_name_effects: list[float] = []
    conflict_effects: list[float] = []

    for img_id in common_images:
        m_dv = dv_by_image[img_id]["signed_answer_margin"]
        m_ipn = ipn_by_image[img_id]["signed_answer_margin"]
        m_wn = wn_by_image[img_id]["signed_answer_margin"]
        m_vtc = vtc_by_image[img_id]["signed_answer_margin"]

        name_effects.append(m_ipn - m_dv)
        wrong_name_effects.append(m_wn - m_ipn)
        conflict_effects.append(m_vtc - m_dv)

    n = len(common_images) if common_images else 1
    mean_ne = sum(name_effects) / n
    mean_wne = sum(wrong_name_effects) / n
    mean_ce = sum(conflict_effects) / n

    return {
        "NameEffect": mean_ne,
        "WrongNameEffect": mean_wne,
        "ConflictEffect": mean_ce,
        "abs_WrongNameEffect": abs(mean_wne),
        "abs_ConflictEffect": abs(mean_ce),
        "n_images": len(common_images),
        "per_image": {
            "NameEffect": name_effects,
            "WrongNameEffect": wrong_name_effects,
            "ConflictEffect": conflict_effects,
        },
    }


# --------------------------------------------------------------------------- #
# Identity-clustered bootstrap
# --------------------------------------------------------------------------- #

def identity_clustered_bootstrap(
    *,
    probe_results_m: list[dict[str, Any]],
    probe_results_d: list[dict[str, Any]],
    experimental_ids: list[str],
    n_resamples: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    """Identity-clustered bootstrap for M-D route differences.

    Samples experimental identities with replacement, includes all
    relevant held-out probes for each sampled identity, recomputes
    the route statistic, and reports mean + 95% CI.
    """
    rng = random.Random(seed)

    # Build per-identity probe lookup
    m_by_id = _group_by_identity(probe_results_m)
    d_by_id = _group_by_identity(probe_results_d)

    # Compute the actual (point estimate) statistics
    point_stats = _compute_bootstrap_statistics(
        probe_results_m, probe_results_d,
    )

    # Bootstrap resamples
    bootstrap_samples: dict[str, list[float]] = {
        key: [] for key in point_stats
    }

    for _ in range(n_resamples):
        # Sample identities with replacement
        sampled_ids = [
            rng.choice(experimental_ids)
            for _ in range(len(experimental_ids))
        ]

        # Collect all probes for sampled identities
        m_resampled: list[dict] = []
        d_resampled: list[dict] = []
        for id_ in sampled_ids:
            m_resampled.extend(m_by_id.get(id_, []))
            d_resampled.extend(d_by_id.get(id_, []))

        stats = _compute_bootstrap_statistics(m_resampled, d_resampled)
        for key, val in stats.items():
            bootstrap_samples[key].append(val)

    # Compute mean and 95% CI
    result: dict[str, Any] = {
        "n_resamples": n_resamples,
        "point_estimates": point_stats,
        "bootstrap": {},
    }

    for key in point_stats:
        samples = sorted(bootstrap_samples[key])
        n = len(samples)
        mean = sum(samples) / n
        lo = samples[max(0, int(0.025 * n))]
        hi = samples[min(n - 1, int(0.975 * n))]
        result["bootstrap"][key] = {
            "mean": mean,
            "ci_lower": lo,
            "ci_upper": hi,
            "ci_excludes_zero": (lo > 0 and hi > 0) or (lo < 0 and hi < 0),
        }

    return result


def _group_by_identity(
    probe_results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict]] = {}
    for p in probe_results:
        id_ = p.get("identity_id", "unknown")
        groups.setdefault(id_, []).append(p)
    return groups


def _compute_bootstrap_statistics(
    m_results: list[dict[str, Any]],
    d_results: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute route-difference statistics for a bootstrap sample."""
    # NAME accuracy difference
    m_name = [r for r in m_results if r.get("family") == "NAME"]
    d_name = [r for r in d_results if r.get("family") == "NAME"]
    m_name_acc = compute_accuracy_from_probes(m_name) if m_name else 0.0
    d_name_acc = compute_accuracy_from_probes(d_name) if d_name else 0.0
    name_diff = m_name_acc - d_name_acc

    # Absolute WrongNameEffect difference
    m_dv = [r for r in m_results if r.get("family") == "DV_syn"]
    m_ipn = [r for r in m_results if r.get("family") == "IPN_syn"]
    m_wn = [r for r in m_results if r.get("family") == "WN"]
    m_vtc = [r for r in m_results if r.get("family") == "VTC"]

    d_dv = [r for r in d_results if r.get("family") == "DV_syn"]
    d_ipn = [r for r in d_results if r.get("family") == "IPN_syn"]
    d_wn = [r for r in d_results if r.get("family") == "WN"]
    d_vtc = [r for r in d_results if r.get("family") == "VTC"]

    m_effects = compute_route_effects(
        dv_results=m_dv, ipn_results=m_ipn,
        wn_results=m_wn, vtc_results=m_vtc,
    ) if m_dv and m_ipn and m_wn and m_vtc else {
        "abs_WrongNameEffect": 0.0, "abs_ConflictEffect": 0.0,
    }

    d_effects = compute_route_effects(
        dv_results=d_dv, ipn_results=d_ipn,
        wn_results=d_wn, vtc_results=d_vtc,
    ) if d_dv and d_ipn and d_wn and d_vtc else {
        "abs_WrongNameEffect": 0.0, "abs_ConflictEffect": 0.0,
    }

    wn_diff = m_effects["abs_WrongNameEffect"] - d_effects["abs_WrongNameEffect"]
    vtc_diff = m_effects["abs_ConflictEffect"] - d_effects["abs_ConflictEffect"]

    return {
        "NAME_accuracy_M_minus_D": name_diff,
        "abs_WrongNameEffect_M_minus_D": wn_diff,
        "abs_ConflictEffect_M_minus_D": vtc_diff,
    }


# --------------------------------------------------------------------------- #
# M-shuffled analysis
# --------------------------------------------------------------------------- #

def compute_shuffled_analysis(
    probe_results: list[dict[str, Any]],
    true_mapping: dict[str, str],
    shuffled_mapping: dict[str, str],
) -> dict[str, Any]:
    """Analyze M-shuffled against both true and shuffled mappings.

    Returns agreement_with_true_mapping and agreement_with_shuffled_mapping
    for NAME, DV-syn, and IPN-syn families.
    """
    result: dict[str, Any] = {}

    for family in ("NAME", "DV_syn", "IPN_syn"):
        family_results = [
            r for r in probe_results if r.get("family") == family
        ]

        agree_true = 0
        agree_shuffled = 0
        total = len(family_results)

        for r in family_results:
            id_ = r["identity_id"]
            predicted = r.get("predicted_answer", "")

            if predicted == true_mapping.get(id_):
                agree_true += 1
            if predicted == shuffled_mapping.get(id_):
                agree_shuffled += 1

        result[family] = {
            "agreement_with_true_mapping": agree_true / total if total else 0.0,
            "agreement_with_shuffled_mapping": agree_shuffled / total if total else 0.0,
            "n_probes": total,
        }

    return result
