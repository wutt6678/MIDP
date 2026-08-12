"""P2-12 / P2-13 / P2-14 / P2-15: audit strict mode, route-family minimums, polarity.

P2-12 — Uncertain manual-audit items fail under pilot/strict mode.
P2-13 — Explicit audit status values (pass/fail/uncertain/resolved).
P2-14 — Route-family minimums for pilot mode.
P2-15 — Polarity balance checks for visual attributes.
"""

from __future__ import annotations

import pytest

from route_data.build.audit import (
    DEFAULT_MINIMUM_ROUTE_COUNTS,
    REQUIRED_ROUTE_FAMILIES,
    AuditGateError,
    AuditStatus,
    ProvenanceFreezeError,
    RouteFamilyError,
    check_benchmark_provenance_frozen,
    check_route_family_minimums,
    report_polarity_balance,
    validate_audit_statuses,
)

# --------------------------------------------------------------------------- #
# P2-13: AuditStatus enum
# --------------------------------------------------------------------------- #


class TestAuditStatus:
    def test_enum_values(self):
        assert AuditStatus.PASS.value == "pass"
        assert AuditStatus.FAIL.value == "fail"
        assert AuditStatus.UNCERTAIN.value == "uncertain"
        assert AuditStatus.RESOLVED_AFTER_SECOND_REVIEW.value == (
            "resolved_after_second_review"
        )

    def test_from_str_valid(self):
        assert AuditStatus.from_str("pass") == AuditStatus.PASS
        assert AuditStatus.from_str("FAIL") == AuditStatus.FAIL
        assert AuditStatus.from_str("Uncertain") == AuditStatus.UNCERTAIN
        assert AuditStatus.from_str("resolved_after_second_review") == (
            AuditStatus.RESOLVED_AFTER_SECOND_REVIEW
        )

    def test_from_str_invalid(self):
        with pytest.raises(ValueError, match="Unknown audit status"):
            AuditStatus.from_str("bogus")

    def test_is_acceptable_for_pilot(self):
        assert AuditStatus.PASS.is_acceptable_for_pilot()
        assert AuditStatus.RESOLVED_AFTER_SECOND_REVIEW.is_acceptable_for_pilot()
        assert not AuditStatus.FAIL.is_acceptable_for_pilot()
        assert not AuditStatus.UNCERTAIN.is_acceptable_for_pilot()


# --------------------------------------------------------------------------- #
# P2-12: validate_audit_statuses
# --------------------------------------------------------------------------- #


class TestValidateAuditStatuses:
    def test_all_pass(self):
        report = validate_audit_statuses(["pass", "pass", "pass"])
        assert report["total"] == 3
        assert report["counts"] == {"pass": 3}
        assert report["blocking"] == []
        assert report["pilot_ready"] is True

    def test_uncertain_blocks_pilot(self):
        report = validate_audit_statuses(
            ["pass", "uncertain", "pass"], strict=False,
        )
        assert report["total"] == 3
        assert "uncertain" in report["blocking"]
        assert report["pilot_ready"] is False

    def test_fail_blocks_pilot(self):
        report = validate_audit_statuses(["pass", "fail"], strict=False)
        assert "fail" in report["blocking"]
        assert report["pilot_ready"] is False

    def test_strict_raises_on_uncertain(self):
        with pytest.raises(AuditGateError, match="blocking statuses"):
            validate_audit_statuses(["pass", "uncertain"], strict=True)

    def test_strict_raises_on_fail(self):
        with pytest.raises(AuditGateError, match="blocking statuses"):
            validate_audit_statuses(["fail"], strict=True)

    def test_non_strict_reports_without_raising(self):
        report = validate_audit_statuses(["fail", "uncertain"], strict=False)
        assert report["pilot_ready"] is False
        assert "fail" in report["blocking"]
        assert "uncertain" in report["blocking"]

    def test_resolved_is_acceptable(self):
        report = validate_audit_statuses([
            "pass", "resolved_after_second_review",
        ])
        assert report["pilot_ready"] is True
        assert report["blocking"] == []

    def test_accepts_enum_values(self):
        report = validate_audit_statuses(
            [AuditStatus.PASS, AuditStatus.UNCERTAIN], strict=False,
        )
        assert report["total"] == 2
        assert "uncertain" in report["blocking"]


# --------------------------------------------------------------------------- #
# P2-14: route-family minimums
# --------------------------------------------------------------------------- #


class TestRouteFamilyMinimums:
    def test_default_families_match_spec(self):
        """P2-14: the six required families from the fix list."""
        assert REQUIRED_ROUTE_FAMILIES == {
            "direct_visual",
            "image_plus_name",
            "wrong_name",
            "visual_text_conflict",
            "name_only",
            "cross_image",
        }

    def test_default_minimums(self):
        assert DEFAULT_MINIMUM_ROUTE_COUNTS["direct_visual"] == 20
        assert DEFAULT_MINIMUM_ROUTE_COUNTS["cross_image"] == 10

    def test_meets_minimums(self):
        counts = {
            "direct_visual": 25,
            "image_plus_name": 20,
            "wrong_name": 30,
            "visual_text_conflict": 20,
            "name_only": 22,
            "cross_image": 15,
        }
        report = check_route_family_minimums(counts, strict=False)
        assert report["pilot_ready"] is True
        assert report["shortfalls"] == {}

    def test_falls_short(self):
        counts = {"direct_visual": 5, "name_only": 3}
        report = check_route_family_minimums(counts, strict=False)
        assert report["pilot_ready"] is False
        assert "direct_visual" in report["shortfalls"]
        assert "name_only" in report["shortfalls"]
        assert report["shortfalls"]["direct_visual"]["actual"] == 5
        assert report["shortfalls"]["direct_visual"]["required"] == 20

    def test_missing_family_is_zero(self):
        counts = {"direct_visual": 25}
        report = check_route_family_minimums(counts, strict=False)
        # All other families are 0.
        assert report["shortfalls"]["cross_image"]["actual"] == 0

    def test_strict_raises_on_shortfall(self):
        with pytest.raises(RouteFamilyError, match="route-family minimums"):
            check_route_family_minimums({"direct_visual": 5}, strict=True)

    def test_custom_minimums(self):
        custom = {"direct_visual": 3, "name_only": 2}
        counts = {"direct_visual": 5, "name_only": 3}
        report = check_route_family_minimums(counts, custom)
        assert report["pilot_ready"] is True

    def test_custom_minimums_fail(self):
        custom = {"direct_visual": 10}
        counts = {"direct_visual": 5}
        with pytest.raises(RouteFamilyError):
            check_route_family_minimums(counts, custom, strict=True)


# --------------------------------------------------------------------------- #
# P2-15: polarity balance
# --------------------------------------------------------------------------- #


class TestPolarityBalance:
    def test_balanced_attribute(self):
        states = {"Eyeglasses": [True, True, False, False]}
        report = report_polarity_balance(states)
        assert report["pilot_ready"] is True
        info = report["per_attribute"]["Eyeglasses"]
        assert info["positive"] == 2
        assert info["negative"] == 2
        assert info["cross_state_pairs"] == 4
        assert info["usable"] is True

    def test_all_positive_excluded(self):
        states = {"Smiling": [True, True, True]}
        report = report_polarity_balance(states)
        assert report["pilot_ready"] is False
        assert "Smiling" in report["excluded_attributes"]
        info = report["per_attribute"]["Smiling"]
        assert info["positive"] == 3
        assert info["negative"] == 0
        assert info["cross_state_pairs"] == 0
        assert info["usable"] is False

    def test_all_negative_excluded(self):
        states = {"Bald": [False, False]}
        report = report_polarity_balance(states)
        assert "Bald" in report["excluded_attributes"]
        assert report["per_attribute"]["Bald"]["usable"] is False

    def test_mixed_attributes(self):
        states = {
            "Eyeglasses": [True, False, True, False],  # balanced
            "Smiling": [True, True, True],  # all positive
            "Bald": [True, True, False],  # usable
        }
        report = report_polarity_balance(states)
        assert "Eyeglasses" in report["usable_attributes"]
        assert "Bald" in report["usable_attributes"]
        assert "Smiling" in report["excluded_attributes"]
        assert report["pilot_ready"] is False

    def test_single_identity_per_state(self):
        states = {"Hat": [True, False]}
        report = report_polarity_balance(states)
        info = report["per_attribute"]["Hat"]
        assert info["positive"] == 1
        assert info["negative"] == 1
        assert info["cross_state_pairs"] == 1
        assert info["usable"] is True

    def test_empty_input(self):
        report = report_polarity_balance({})
        assert report["pilot_ready"] is True
        assert report["usable_attributes"] == []
        assert report["excluded_attributes"] == []

    def test_custom_thresholds(self):
        states = {"Eyeglasses": [True, False]}
        # Require at least 3 positive — won't be met.
        report = report_polarity_balance(states, min_positive=3)
        assert report["per_attribute"]["Eyeglasses"]["usable"] is False

    def test_cross_state_pairs_calculation(self):
        states = {"Attr": [True, True, True, False]}
        info = report_polarity_balance(states)["per_attribute"]["Attr"]
        assert info["positive"] == 3
        assert info["negative"] == 1
        assert info["cross_state_pairs"] == 3  # 3 * 1


# --------------------------------------------------------------------------- #
# P2-16: benchmark provenance freeze
# --------------------------------------------------------------------------- #


class TestBenchmarkProvenanceFreeze:
    def test_all_frozen(self):
        benchmarks = {
            "FIUBench": {
                "source_version": "v1.0",
                "metadata_hash": "abc123",
                "split_hash": "def456",
                "source_verification": "PASS",
            },
            "FAIRGET": {
                "source_version": "commit:abc123",
                "metadata_hash": "111222",
                "split_hash": "333444",
                "source_verification": "PASS",
            },
        }
        report = check_benchmark_provenance_frozen(benchmarks)
        assert report["all_frozen"] is True
        for info in report["per_benchmark"].values():
            assert info["frozen"] is True
            assert info["issues"] == []

    def test_pending_source_version(self):
        benchmarks = {
            "MLLMU-Bench": {
                "source_version": "PENDING",
                "metadata_hash": "abc",
                "split_hash": "def",
                "source_verification": "PASS",
            },
        }
        report = check_benchmark_provenance_frozen(benchmarks, strict=False)
        assert report["all_frozen"] is False
        info = report["per_benchmark"]["MLLMU-Bench"]
        assert info["frozen"] is False
        assert any("PENDING" in i for i in info["issues"])

    def test_missing_metadata_hash(self):
        benchmarks = {
            "PPU-Bench": {
                "source_version": "v2.0",
                "metadata_hash": None,
                "split_hash": "def",
                "source_verification": "PASS",
            },
        }
        report = check_benchmark_provenance_frozen(benchmarks, strict=False)
        assert report["all_frozen"] is False

    def test_verification_not_pass(self):
        benchmarks = {
            "FIUBench": {
                "source_version": "v1.0",
                "metadata_hash": "abc",
                "split_hash": "def",
                "source_verification": "FAIL",
            },
        }
        report = check_benchmark_provenance_frozen(benchmarks, strict=False)
        assert report["all_frozen"] is False
        assert any("FAIL" in i for i in report["per_benchmark"]["FIUBench"]["issues"])

    def test_strict_raises_on_unfrozen(self):
        benchmarks = {
            "FIUBench": {
                "source_version": "PENDING",
                "metadata_hash": "abc",
                "split_hash": "def",
                "source_verification": "PASS",
            },
        }
        with pytest.raises(ProvenanceFreezeError, match="provenance not frozen"):
            check_benchmark_provenance_frozen(benchmarks, strict=True)

    def test_empty_benchmarks(self):
        report = check_benchmark_provenance_frozen({})
        assert report["all_frozen"] is True
        assert report["per_benchmark"] == {}

    def test_multiple_benchmarks_mixed(self):
        benchmarks = {
            "FIUBench": {
                "source_version": "v1.0",
                "metadata_hash": "abc",
                "split_hash": "def",
                "source_verification": "PASS",
            },
            "FAIRGET": {
                "source_version": "PENDING",
                "metadata_hash": "PENDING",
                "split_hash": "PENDING",
                "source_verification": "PENDING",
            },
        }
        report = check_benchmark_provenance_frozen(benchmarks, strict=False)
        assert report["all_frozen"] is False
        assert report["per_benchmark"]["FIUBench"]["frozen"] is True
        assert report["per_benchmark"]["FAIRGET"]["frozen"] is False
