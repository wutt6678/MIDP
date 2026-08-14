"""Tests for identity-level protocol verification helpers (Item 2 repair).

Exercises the production helpers ``collect_identity_roles_from_processed``
and ``compute_identity_counts_from_roles`` in ``scripts/final_verify.py``,
plus the full ``_verify_protocol_identity_counts`` check function.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Import final_verify as a module (it lives in scripts/, not a package).
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


@pytest.fixture(scope="module")
def fv():
    """Import final_verify.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "final_verify", SCRIPTS_DIR / "final_verify.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["final_verify"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


# --------------------------------------------------------------------------- #
# Tests for collect_identity_roles_from_processed
# --------------------------------------------------------------------------- #


class TestCollectIdentityRoles:
    """Tests for ``collect_identity_roles_from_processed``."""

    def test_counts_unique_identities_not_rows(self, fv, tmp_path: Path) -> None:
        """Each identity counted once even if it has multiple canonical rows."""
        p = tmp_path / "processed.jsonl"
        _write_jsonl(p, [
            {"identity_id": "id_A", "split": "train"},
            {"identity_id": "id_A", "split": "train"},  # duplicate row
            {"identity_id": "id_A", "split": "train"},  # another variant
            {"identity_id": "id_B", "split": "eval"},
            {"identity_id": "id_B", "split": "eval"},
            {"identity_id": "id_C", "split": "exclude"},
        ])
        identity_roles, sample_counts, issues = fv.collect_identity_roles_from_processed(p)

        # 3 unique identities, each with one role.
        assert len(identity_roles) == 3
        assert identity_roles["id_A"] == "train"
        assert identity_roles["id_B"] == "eval"
        assert identity_roles["id_C"] == "exclude"

        # Sample counts reflect all rows (6 total).
        assert sample_counts["train"] == 3
        assert sample_counts["eval"] == 2
        assert sample_counts["exclude"] == 1

        # No inconsistencies.
        assert issues == []

    def test_detects_inconsistent_role_for_same_identity(
        self, fv, tmp_path: Path
    ) -> None:
        """An identity appearing with two different splits is flagged."""
        p = tmp_path / "processed.jsonl"
        _write_jsonl(p, [
            {"identity_id": "id_X", "split": "train"},
            {"identity_id": "id_X", "split": "eval"},  # conflict!
            {"identity_id": "id_Y", "split": "exclude"},
        ])
        identity_roles, sample_counts, issues = fv.collect_identity_roles_from_processed(p)

        assert len(issues) == 1
        assert "id_X" in issues[0]
        assert "inconsistent roles" in issues[0]

    def test_empty_file(self, fv, tmp_path: Path) -> None:
        """An empty processed file yields empty results."""
        p = tmp_path / "processed.jsonl"
        p.write_text("")
        identity_roles, sample_counts, issues = fv.collect_identity_roles_from_processed(p)
        assert identity_roles == {}
        assert sample_counts == {}
        assert issues == []

    def test_blank_lines_skipped(self, fv, tmp_path: Path) -> None:
        """Blank lines in the JSONL are silently skipped."""
        p = tmp_path / "processed.jsonl"
        p.write_text(
            '{"identity_id": "id_1", "split": "train"}\n'
            "\n"
            '{"identity_id": "id_2", "split": "eval"}\n'
            "\n"
        )
        identity_roles, sample_counts, issues = fv.collect_identity_roles_from_processed(p)
        assert len(identity_roles) == 2
        assert issues == []


# --------------------------------------------------------------------------- #
# Tests for compute_identity_counts_from_roles
# --------------------------------------------------------------------------- #


class TestComputeIdentityCounts:
    """Tests for ``compute_identity_counts_from_roles``."""

    def test_basic_counting(self, fv) -> None:
        roles = {"a": "train", "b": "train", "c": "eval", "d": "exclude"}
        counts = fv.compute_identity_counts_from_roles(roles)
        assert counts["train"] == 2
        assert counts["eval"] == 1
        assert counts["exclude"] == 1

    def test_empty(self, fv) -> None:
        assert fv.compute_identity_counts_from_roles({}) == {}

    def test_includes_oop_and_unassigned(self, fv) -> None:
        roles = {"a": "out_of_protocol", "b": "unassigned", "c": "hash"}
        counts = fv.compute_identity_counts_from_roles(roles)
        assert counts["out_of_protocol"] == 1
        assert counts["unassigned"] == 1
        assert counts["hash"] == 1


# --------------------------------------------------------------------------- #
# Tests for _verify_protocol_identity_counts (integration-level)
# --------------------------------------------------------------------------- #


class TestVerifyProtocolIdentityCounts:
    """Integration tests for the full verifier check function."""

    def _make_export_dir(
        self,
        tmp_path: Path,
        rows: list[dict],
        *,
        benchmark: str = "fiubench",
        protocol_report: dict | None = None,
    ) -> Path:
        """Create a minimal export directory with a processed JSONL."""
        export_dir = tmp_path / "evidence"
        export_dir.mkdir()
        _write_jsonl(export_dir / f"{benchmark}_processed.jsonl", rows)
        if protocol_report is not None:
            _write_json(
                export_dir / f"{benchmark}_protocol_report.json",
                protocol_report,
            )
        return export_dir

    def test_rejects_unassigned_identity(self, fv, tmp_path: Path) -> None:
        """Identities with 'unassigned' role cause a FAIL."""
        rows = [
            {"identity_id": "a", "split": "train"},
            {"identity_id": "b", "split": "eval"},
            {"identity_id": "c", "split": "exclude"},
            {"identity_id": "bad", "split": "unassigned"},
        ]
        export_dir = self._make_export_dir(tmp_path, rows)
        failures: list[str] = []
        rec = fv._verify_protocol_identity_counts(
            export_dir, "fiubench", str(SCRIPTS_DIR / "../configs/runs/full_fiubench_qwen.yaml"),
            failures,
        )
        assert rec.result == fv.CheckResult.FAIL
        assert "unassigned" in rec.detail.lower() or any("unassigned" in f for f in failures)

    def test_rejects_hash_identity(self, fv, tmp_path: Path) -> None:
        """Identities with 'hash' role cause a FAIL."""
        rows = [
            {"identity_id": "a", "split": "train"},
            {"identity_id": "b", "split": "eval"},
            {"identity_id": "c", "split": "exclude"},
            {"identity_id": "h1", "split": "hash"},
        ]
        export_dir = self._make_export_dir(tmp_path, rows)
        failures: list[str] = []
        rec = fv._verify_protocol_identity_counts(
            export_dir, "fiubench", str(SCRIPTS_DIR / "../configs/runs/full_fiubench_qwen.yaml"),
            failures,
        )
        assert rec.result == fv.CheckResult.FAIL
        assert "hash" in rec.detail.lower() or any("hash" in f for f in failures)

    def test_identity_sets_match_frozen_protocol(self, fv, tmp_path: Path) -> None:
        """When protocol report has ID lists, exact set equality is checked."""
        rows = [
            {"identity_id": "a", "split": "train"},
            {"identity_id": "b", "split": "eval"},
            {"identity_id": "c", "split": "exclude"},
            {"identity_id": "d", "split": "out_of_protocol"},
        ]
        protocol_report = {
            "train_identity_count": 1,
            "eval_identity_count": 1,
            "forget_identity_count": 1,
            "out_of_protocol_identity_count": 1,
            "train_identity_ids": ["a"],
            "eval_identity_ids": ["b"],
            "forget_identity_ids": ["c"],
            "oop_identity_ids": ["d"],
        }
        export_dir = self._make_export_dir(
            tmp_path, rows, protocol_report=protocol_report,
        )
        failures: list[str] = []
        rec = fv._verify_protocol_identity_counts(
            export_dir, "fiubench", str(SCRIPTS_DIR / "../configs/runs/full_fiubench_qwen.yaml"),
            failures,
        )
        assert rec.result == fv.CheckResult.PASS

    def test_identity_sets_mismatch_fails(self, fv, tmp_path: Path) -> None:
        """When identity sets don't match protocol report, FAIL."""
        rows = [
            {"identity_id": "a", "split": "train"},
            {"identity_id": "b", "split": "eval"},
            {"identity_id": "c", "split": "exclude"},
            {"identity_id": "d", "split": "out_of_protocol"},
        ]
        # Protocol report says train should be ["z"] but processed has ["a"].
        protocol_report = {
            "train_identity_count": 1,
            "eval_identity_count": 1,
            "forget_identity_count": 1,
            "out_of_protocol_identity_count": 1,
            "train_identity_ids": ["z"],  # mismatch!
            "eval_identity_ids": ["b"],
            "forget_identity_ids": ["c"],
            "oop_identity_ids": ["d"],
        }
        export_dir = self._make_export_dir(
            tmp_path, rows, protocol_report=protocol_report,
        )
        failures: list[str] = []
        rec = fv._verify_protocol_identity_counts(
            export_dir, "fiubench", str(SCRIPTS_DIR / "../configs/runs/full_fiubench_qwen.yaml"),
            failures,
        )
        assert rec.result == fv.CheckResult.FAIL

    def test_oop_allowed_in_full_processed(self, fv, tmp_path: Path) -> None:
        """OOP identities are valid in the full-source processed artifact."""
        rows = [
            {"identity_id": "a", "split": "train"},
            {"identity_id": "b", "split": "eval"},
            {"identity_id": "c", "split": "exclude"},
            {"identity_id": "d", "split": "out_of_protocol"},
            {"identity_id": "e", "split": "out_of_protocol"},
        ]
        export_dir = self._make_export_dir(tmp_path, rows)
        failures: list[str] = []
        rec = fv._verify_protocol_identity_counts(
            export_dir, "fiubench", str(SCRIPTS_DIR / "../configs/runs/full_fiubench_qwen.yaml"),
            failures,
        )
        # OOP > 0 is fine in full processed source; should not fail.
        assert rec.result == fv.CheckResult.PASS
        assert "oop=2" in rec.detail
