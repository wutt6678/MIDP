"""P2-12 / P2-13: explicit audit status and strict-mode validation.

P2-12: Uncertain manual-audit items must fail under pilot/strict mode.
P2-13: Explicit audit status values with validation rules.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from typing import Any


class AuditStatus(str, enum.Enum):
    """P2-13: explicit audit status for manual review items.

    Each item that goes through manual audit receives exactly one of these
    statuses.  The pilot gate requires:
    - no ``UNREVIEWED`` items (everything must be looked at)
    - no ``FAIL`` items (failed items block the pilot)
    - no ``UNCERTAIN`` items (uncertain items need second review)
    """

    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"
    RESOLVED_AFTER_SECOND_REVIEW = "resolved_after_second_review"

    @classmethod
    def from_str(cls, value: str) -> AuditStatus:
        """Parse a string into an AuditStatus, rejecting unknown values."""
        try:
            return cls(value.lower())
        except ValueError:
            valid = [s.value for s in cls]
            raise ValueError(
                f"Unknown audit status '{value}'; expected one of {valid}"
            ) from None

    def is_acceptable_for_pilot(self) -> bool:
        """Whether this status allows the pilot to proceed.

        Only ``PASS`` and ``RESOLVED_AFTER_SECOND_REVIEW`` are acceptable.
        ``FAIL`` and ``UNCERTAIN`` block the pilot.
        """
        return self in (
            AuditStatus.PASS,
            AuditStatus.RESOLVED_AFTER_SECOND_REVIEW,
        )


def validate_audit_statuses(
    statuses: Sequence[AuditStatus | str],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """P2-12: validate a collection of audit statuses for pilot readiness.

    Returns a report dict with counts and any blocking issues.  In strict
    mode, raises ``AuditGateError`` if any items are unreviewed, failed,
    or uncertain.

    Parameters
    ----------
    statuses:
        Sequence of audit statuses (enum values or strings).
    strict:
        If True, raise on any blocking status.  If False, just report.

    Returns
    -------
    dict with keys:
        - ``total``: number of items
        - ``counts``: {status_value: count}
        - ``blocking``: list of blocking status values found
        - ``pilot_ready``: bool
    """
    parsed: list[AuditStatus] = []
    for s in statuses:
        if isinstance(s, AuditStatus):
            parsed.append(s)
        else:
            parsed.append(AuditStatus.from_str(str(s)))

    counts: dict[str, int] = {}
    for status in parsed:
        counts[status.value] = counts.get(status.value, 0) + 1

    # Blocking statuses for the pilot.
    blocking: list[str] = []
    if counts.get("fail", 0) > 0:
        blocking.append("fail")
    if counts.get("uncertain", 0) > 0:
        blocking.append("uncertain")
    # Note: "unreviewed" is not a status — items must have an explicit
    # status.  If the total count of statuses is less than expected,
    # that's caught elsewhere (missing items check).

    pilot_ready = len(blocking) == 0

    report: dict[str, Any] = {
        "total": len(parsed),
        "counts": counts,
        "blocking": blocking,
        "pilot_ready": pilot_ready,
    }

    if strict and blocking:
        raise AuditGateError(
            f"P2-12: audit gate failed — blocking statuses: {blocking}. "
            f"Counts: {counts}. "
            "Resolve all uncertain/fail items before the pilot."
        )

    return report


class AuditGateError(ValueError):
    """Raised when audit statuses do not meet pilot requirements."""


# --------------------------------------------------------------------------- #
# P2-14: route-family minimums
# --------------------------------------------------------------------------- #

# The six probe families required for route analysis.
REQUIRED_ROUTE_FAMILIES = frozenset({
    "direct_visual",
    "image_plus_name",
    "wrong_name",
    "visual_text_conflict",
    "name_only",
    "cross_image",
})

# Default minimum counts for pilot verification.
DEFAULT_MINIMUM_ROUTE_COUNTS: dict[str, int] = {
    "direct_visual": 20,
    "image_plus_name": 20,
    "wrong_name": 20,
    "visual_text_conflict": 20,
    "name_only": 20,
    "cross_image": 10,
}


def check_route_family_minimums(
    actual_counts: Mapping[str, int],
    minimum_counts: Mapping[str, int] | None = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """P2-14: verify that probe counts meet route-family minimums.

    Parameters
    ----------
    actual_counts:
        Mapping of ``{probe_family: count}`` from the current selection.
    minimum_counts:
        Required minimums.  Defaults to ``DEFAULT_MINIMUM_ROUTE_COUNTS``.
    strict:
        If True, raise ``RouteFamilyError`` on any shortfall.

    Returns
    -------
    dict with keys:
        - ``actual``: the input counts
        - ``required``: the minimums checked
        - ``shortfalls``: {family: {"actual": n, "required": m}}
        - ``pilot_ready``: bool
    """
    required = dict(minimum_counts or DEFAULT_MINIMUM_ROUTE_COUNTS)
    shortfalls: dict[str, dict[str, int]] = {}

    for family, min_count in sorted(required.items()):
        actual = actual_counts.get(family, 0)
        if actual < min_count:
            shortfalls[family] = {"actual": actual, "required": min_count}

    pilot_ready = len(shortfalls) == 0
    report: dict[str, Any] = {
        "actual": dict(actual_counts),
        "required": required,
        "shortfalls": shortfalls,
        "pilot_ready": pilot_ready,
    }

    if strict and shortfalls:
        details = "; ".join(
            f"{fam}: {info['actual']}/{info['required']}"
            for fam, info in shortfalls.items()
        )
        raise RouteFamilyError(
            f"P2-14: route-family minimums not met — {details}. "
            "Increase the selection or lower the minimums in config."
        )

    return report


class RouteFamilyError(ValueError):
    """Raised when route-family counts do not meet pilot minimums."""


# --------------------------------------------------------------------------- #
# P2-15: polarity balance reporting
# --------------------------------------------------------------------------- #


def report_polarity_balance(
    attribute_states: Mapping[str, Sequence[bool]],
    *,
    min_positive: int = 1,
    min_negative: int = 1,
    min_cross_state_pairs: int = 1,
) -> dict[str, Any]:
    """P2-15: report polarity balance for visual attributes.

    For each visual attribute used in causal analysis, report:
    - positive identities (label=True count)
    - negative identities (label=False count)
    - cross-state pairs (positive * negative)

    Attributes that do not have usable polarity coverage (e.g. all positive
    or all negative) are flagged as excluded.

    Parameters
    ----------
    attribute_states:
        Mapping of ``{attribute_name: [label, ...]}`` where each label is
        True (positive) or False (negative).
    min_positive, min_negative, min_cross_state_pairs:
        Minimum thresholds for usable polarity coverage.

    Returns
    -------
    dict with keys:
        - ``per_attribute``: {attr: {positive, negative, cross_state_pairs, usable}}
        - ``usable_attributes``: list of attributes with usable polarity
        - ``excluded_attributes``: list of attributes without usable polarity
        - ``pilot_ready``: bool (all attributes usable)
    """
    per_attribute: dict[str, dict[str, Any]] = {}
    usable: list[str] = []
    excluded: list[str] = []

    for attr in sorted(attribute_states):
        labels = list(attribute_states[attr])
        n_positive = sum(1 for l in labels if l)
        n_negative = len(labels) - n_positive
        cross_state_pairs = n_positive * n_negative

        is_usable = (
            n_positive >= min_positive
            and n_negative >= min_negative
            and cross_state_pairs >= min_cross_state_pairs
        )

        per_attribute[attr] = {
            "positive": n_positive,
            "negative": n_negative,
            "cross_state_pairs": cross_state_pairs,
            "usable": is_usable,
        }

        if is_usable:
            usable.append(attr)
        else:
            excluded.append(attr)

    return {
        "per_attribute": per_attribute,
        "usable_attributes": usable,
        "excluded_attributes": excluded,
        "pilot_ready": len(excluded) == 0,
    }


# --------------------------------------------------------------------------- #
# P2-16: benchmark provenance freeze
# --------------------------------------------------------------------------- #


class ProvenanceFreezeError(ValueError):
    """Raised when benchmark provenance is not frozen before the pilot."""


def check_benchmark_provenance_frozen(
    benchmarks: Mapping[str, Mapping[str, Any]],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """P2-16: verify all benchmark provenance is frozen before the pilot.

    Each benchmark entry should declare:
    - ``source_version``: exact upstream commit/release identifier
    - ``metadata_hash``: SHA-256 of the metadata file
    - ``split_hash``: SHA-256 of the split file
    - ``source_verification``: ``"PASS"`` or ``"FAIL"``

    Parameters
    ----------
    benchmarks:
        Mapping of ``{benchmark_name: {source_version, metadata_hash, ...}}``.
    strict:
        If True, raise ``ProvenanceFreezeError`` on any unfrozen benchmark.

    Returns
    -------
    dict with keys:
        - ``per_benchmark``: {name: {frozen, issues}}
        - ``all_frozen``: bool
    """
    per_benchmark: dict[str, dict[str, Any]] = {}
    all_frozen = True

    for name in sorted(benchmarks):
        info = dict(benchmarks[name])
        issues: list[str] = []

        # Check source_version is not PENDING or missing.
        sv = info.get("source_version")
        if not sv or sv == "PENDING":
            issues.append("source_version is missing or PENDING")

        # Check metadata_hash.
        mh = info.get("metadata_hash")
        if not mh or mh == "PENDING":
            issues.append("metadata_hash is missing or PENDING")

        # Check split_hash.
        sh = info.get("split_hash")
        if not sh or sh == "PENDING":
            issues.append("split_hash is missing or PENDING")

        # Check source_verification status.
        sv_status = info.get("source_verification")
        if sv_status != "PASS":
            issues.append(
                f"source_verification is '{sv_status}', expected 'PASS'"
            )

        frozen = len(issues) == 0
        per_benchmark[name] = {"frozen": frozen, "issues": issues}
        if not frozen:
            all_frozen = False

    report: dict[str, Any] = {
        "per_benchmark": per_benchmark,
        "all_frozen": all_frozen,
    }

    if strict and not all_frozen:
        details = "; ".join(
            f"{name}: {', '.join(info['issues'])}"
            for name, info in per_benchmark.items()
            if not info["frozen"]
        )
        raise ProvenanceFreezeError(
            f"P2-16: benchmark provenance not frozen — {details}. "
            "Freeze all benchmark provenance before the combined pilot."
        )

    return report
