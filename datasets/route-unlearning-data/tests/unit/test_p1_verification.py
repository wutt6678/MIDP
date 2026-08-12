"""Tests for P1-5 through P1-15 strict verification checks in final_verify.py.

These tests exercise the individual verification functions by creating
temporary artifact directories with controlled content and asserting
PASS / FAIL / NOT_APPLICABLE outcomes.
"""

from __future__ import annotations

import hashlib
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

BENCHMARK = "testbench"


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# P1-5: Read actual score_manifest.json
# --------------------------------------------------------------------------- #


class TestVerifyScoreManifest:
    """P1-5: _verify_score_manifest reads <dataset>_score_manifest.json."""

    def test_pass_with_all_required_fields(self, tmp_path, fv):
        manifest = {
            "model_id": "Qwen/Qwen3.5-9B",
            "backend": "transformers",
            "resolved_revision": "abc123def456",
            "fingerprint_id": "fp-001",
            "model_fingerprint_payload": {"dtype": "float16"},
            "prompt_registry_hash": "pr-hash",
            "candidate_set_hash": "cs-hash",
            "scoring_version": "1.0",
        }
        _write_json(tmp_path / f"{BENCHMARK}_score_manifest.json", manifest)
        failures: list[str] = []
        rec = fv._verify_score_manifest(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.PASS
        assert not failures

    def test_fail_when_manifest_missing(self, tmp_path, fv):
        failures: list[str] = []
        rec = fv._verify_score_manifest(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL
        assert any("MISSING" in f for f in failures)

    def test_fail_when_required_fields_missing(self, tmp_path, fv):
        manifest = {"model_id": "Qwen/Qwen3.5-9B"}  # missing many fields
        _write_json(tmp_path / f"{BENCHMARK}_score_manifest.json", manifest)
        failures: list[str] = []
        rec = fv._verify_score_manifest(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL
        assert any("missing fields" in f for f in failures)

    def test_fail_when_resolved_revision_pending(self, tmp_path, fv):
        manifest = {
            "model_id": "Qwen/Qwen3.5-9B",
            "backend": "transformers",
            "resolved_revision": "PENDING",
            "fingerprint_id": "fp-001",
            "model_fingerprint_payload": {},
            "prompt_registry_hash": "pr-hash",
            "candidate_set_hash": "cs-hash",
            "scoring_version": "1.0",
        }
        _write_json(tmp_path / f"{BENCHMARK}_score_manifest.json", manifest)
        failures: list[str] = []
        rec = fv._verify_score_manifest(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL
        assert any("resolved_revision" in f for f in failures)

    def test_source_version_cannot_replace_model_revision(self, tmp_path, fv):
        """source_version must not satisfy the resolved_revision check."""
        manifest = {
            "model_id": "Qwen/Qwen3.5-9B",
            "backend": "transformers",
            "resolved_revision": None,
            "fingerprint_id": "fp-001",
            "model_fingerprint_payload": {},
            "prompt_registry_hash": "pr-hash",
            "candidate_set_hash": "cs-hash",
            "scoring_version": "1.0",
            "source_version": "v2.0",  # should NOT satisfy revision check
        }
        _write_json(tmp_path / f"{BENCHMARK}_score_manifest.json", manifest)
        failures: list[str] = []
        rec = fv._verify_score_manifest(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL


# --------------------------------------------------------------------------- #
# P1-6: Require exactly 40 CelebA extension attributes per image
# --------------------------------------------------------------------------- #


class TestVerifyScoresPerImage:
    """P1-6: _verify_scores_per_image requires exactly 40 CelebA attrs."""

    def _make_processed(self, tmp_path, n_attrs=40, extra_attrs=None, missing_attrs=None):
        """Create a processed JSONL with controlled CelebA attribute count."""
        from route_data.constants.celeba_attributes import CELEBA_ATTRIBUTES

        attrs = list(CELEBA_ATTRIBUTES)
        if missing_attrs:
            attrs = [a for a in attrs if a not in missing_attrs]
        if extra_attrs:
            attrs.extend(extra_attrs)

        prefix = "extended_attributes.celeba40."
        va = {}
        for a in attrs:
            va[f"{prefix}{a}"] = {"label": True, "source": "source_model"}

        sample = {
            "source_sample_id": "s1",
            "identity_id": "id1",
            "visual_attributes": va,
        }
        _write_jsonl(tmp_path / f"{BENCHMARK}_processed.jsonl", [sample])

    def test_pass_with_exactly_40_attrs(self, tmp_path, fv):
        self._make_processed(tmp_path)
        failures: list[str] = []
        rec = fv._verify_scores_per_image(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.PASS

    def test_fail_with_39_attrs(self, tmp_path, fv):
        self._make_processed(tmp_path, missing_attrs={"Eyeglasses"})
        failures: list[str] = []
        rec = fv._verify_scores_per_image(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL

    def test_fail_with_unknown_extension_attr(self, tmp_path, fv):
        self._make_processed(tmp_path, extra_attrs=["UnknownAttr"])
        failures: list[str] = []
        rec = fv._verify_scores_per_image(tmp_path, BENCHMARK, failures)
        # 41 attrs total → not equal to CELEBA_ATTRIBUTE_SET → FAIL
        assert rec.result == fv.CheckResult.FAIL

    def test_fail_when_no_artifact(self, tmp_path, fv):
        failures: list[str] = []
        rec = fv._verify_scores_per_image(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL


# --------------------------------------------------------------------------- #
# P1-7: Verify *_processed.jsonl directly
# --------------------------------------------------------------------------- #


class TestVerifyProcessedArtifact:
    """P1-7: _verify_processed_artifact checks *_processed.jsonl."""

    def test_pass_with_valid_processed(self, tmp_path, fv):
        rows = [
            {"source_sample_id": "s1", "identity_id": "id1", "visual_attributes": {}},
            {"source_sample_id": "s2", "identity_id": "id2", "visual_attributes": {}},
        ]
        _write_jsonl(tmp_path / f"{BENCHMARK}_processed.jsonl", rows)
        failures: list[str] = []
        rec = fv._verify_processed_artifact(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.PASS

    def test_fail_when_missing(self, tmp_path, fv):
        failures: list[str] = []
        rec = fv._verify_processed_artifact(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL

    def test_fail_when_empty(self, tmp_path, fv):
        (tmp_path / f"{BENCHMARK}_processed.jsonl").write_text("")
        failures: list[str] = []
        rec = fv._verify_processed_artifact(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL

    def test_fail_when_missing_fields(self, tmp_path, fv):
        rows = [{"source_sample_id": "s1"}]  # missing identity_id, visual_attributes
        _write_jsonl(tmp_path / f"{BENCHMARK}_processed.jsonl", rows)
        failures: list[str] = []
        rec = fv._verify_processed_artifact(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL


# --------------------------------------------------------------------------- #
# P1-8: Verify actual whitelist invariant
# --------------------------------------------------------------------------- #


class TestVerifyWhitelistInvariant:
    """P1-8: _verify_whitelist_invariant uses score_manifest whitelist."""

    def _setup(self, tmp_path, wl_attrs=None, wl_sha=None, processed_rows=None):
        sm = {}
        if wl_attrs is not None:
            sm["whitelist_attributes"] = wl_attrs
        if wl_sha is not None:
            sm["whitelist_file_sha256"] = wl_sha
        _write_json(tmp_path / f"{BENCHMARK}_score_manifest.json", sm)
        if processed_rows is not None:
            _write_jsonl(tmp_path / f"{BENCHMARK}_processed.jsonl", processed_rows)

    def test_pass_when_all_labels_whitelisted(self, tmp_path, fv):
        prefix = "extended_attributes.celeba40."
        rows = [{
            "source_sample_id": "s1",
            "identity_id": "id1",
            "visual_attributes": {
                f"{prefix}Eyeglasses": {"label": True, "source": "source_model"},
                f"{prefix}Smiling": {"label": False, "source": "source_model"},
            },
        }]
        self._setup(tmp_path, wl_attrs=["Eyeglasses", "Smiling"], processed_rows=rows)
        failures: list[str] = []
        rec = fv._verify_whitelist_invariant(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.PASS

    def test_fail_when_nonwhitelisted_label_present(self, tmp_path, fv):
        prefix = "extended_attributes.celeba40."
        rows = [{
            "source_sample_id": "s1",
            "identity_id": "id1",
            "visual_attributes": {
                f"{prefix}Eyeglasses": {"label": True, "source": "source_model"},
                f"{prefix}Young": {"label": True, "source": "source_model"},  # NOT in whitelist
            },
        }]
        self._setup(tmp_path, wl_attrs=["Eyeglasses"], processed_rows=rows)
        failures: list[str] = []
        rec = fv._verify_whitelist_invariant(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL
        assert any("non-whitelisted" in f for f in failures)

    def test_not_applicable_when_no_whitelist(self, tmp_path, fv):
        self._setup(tmp_path)
        failures: list[str] = []
        rec = fv._verify_whitelist_invariant(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.NOT_APPLICABLE

    def test_whitelist_file_hash_mismatch(self, tmp_path, fv):
        # Create a whitelist file with known content.
        wl_file = tmp_path / "whitelist.txt"
        wl_file.write_text("Eyeglasses\nSmiling\n")
        wrong_sha = "0" * 64

        sm = {
            "whitelist_attributes": ["Eyeglasses", "Smiling"],
            "whitelist_file_sha256": wrong_sha,
            "whitelist_path": str(wl_file),
        }
        _write_json(tmp_path / f"{BENCHMARK}_score_manifest.json", sm)
        failures: list[str] = []
        rec = fv._verify_whitelist_invariant(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL
        assert any("SHA mismatch" in f or "SHA" in f for f in failures)


# --------------------------------------------------------------------------- #
# P1-9: Verify answer_label / answer_text in route rows
# --------------------------------------------------------------------------- #


class TestVerifyRouteExpectedAnswers:
    """P1-9: _verify_route_expected_answers checks answer schema."""

    def test_pass_with_valid_visual_rows(self, tmp_path, fv):
        rows = [
            {
                "probe_family": "direct_visual",
                "target_attribute": "Eyeglasses",
                "answer_label": True,
                "answer_text": "yes",
            },
            {
                "probe_family": "image_plus_name",
                "target_attribute": "Smiling",
                "answer_label": False,
                "answer_text": "no",
            },
        ]
        _write_jsonl(tmp_path / f"{BENCHMARK}_route_probes.jsonl", rows)
        failures: list[str] = []
        rec = fv._verify_route_expected_answers(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.PASS

    def test_pass_with_valid_name_only_rows(self, tmp_path, fv):
        rows = [{
            "probe_family": "name_only",
            "target_attribute": None,
            "target_fact_id": "fact1",
            "answer_text": "Alice",
        }]
        _write_jsonl(tmp_path / f"{BENCHMARK}_route_probes.jsonl", rows)
        failures: list[str] = []
        rec = fv._verify_route_expected_answers(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.PASS

    def test_fail_when_visual_missing_target_attribute(self, tmp_path, fv):
        rows = [{
            "probe_family": "direct_visual",
            "target_attribute": None,  # should be set
            "answer_label": True,
            "answer_text": "yes",
        }]
        _write_jsonl(tmp_path / f"{BENCHMARK}_route_probes.jsonl", rows)
        failures: list[str] = []
        rec = fv._verify_route_expected_answers(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL

    def test_fail_when_answer_text_invalid(self, tmp_path, fv):
        rows = [{
            "probe_family": "direct_visual",
            "target_attribute": "Eyeglasses",
            "answer_label": True,
            "answer_text": "maybe",  # not "yes" or "no"
        }]
        _write_jsonl(tmp_path / f"{BENCHMARK}_route_probes.jsonl", rows)
        failures: list[str] = []
        rec = fv._verify_route_expected_answers(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL

    def test_fail_when_no_route_file(self, tmp_path, fv):
        failures: list[str] = []
        rec = fv._verify_route_expected_answers(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL


# --------------------------------------------------------------------------- #
# P1-10: Verify name_only image absence in route probes
# --------------------------------------------------------------------------- #


class TestVerifyTextOnlyImageAbsence:
    """P1-10: name_only probes must have image_uri=null, modality=text_only."""

    def test_pass_when_name_only_has_no_image(self, tmp_path, fv):
        rows = [
            {
                "probe_family": "name_only",
                "image_uri": None,
                "modality": "text_only",
            },
            {
                "probe_family": "direct_visual",
                "image_uri": "img.jpg",
                "modality": "visual",
            },
        ]
        _write_jsonl(tmp_path / f"{BENCHMARK}_route_probes.jsonl", rows)
        failures: list[str] = []
        rec = fv._verify_text_only_image_absence(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.PASS

    def test_fail_when_name_only_has_image(self, tmp_path, fv):
        rows = [{
            "probe_family": "name_only",
            "image_uri": "img.jpg",  # should be null
            "modality": "text_only",
        }]
        _write_jsonl(tmp_path / f"{BENCHMARK}_route_probes.jsonl", rows)
        failures: list[str] = []
        rec = fv._verify_text_only_image_absence(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL

    def test_fail_when_name_only_wrong_modality(self, tmp_path, fv):
        rows = [{
            "probe_family": "name_only",
            "image_uri": None,
            "modality": "visual",  # should be text_only
        }]
        _write_jsonl(tmp_path / f"{BENCHMARK}_route_probes.jsonl", rows)
        failures: list[str] = []
        rec = fv._verify_text_only_image_absence(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL


# --------------------------------------------------------------------------- #
# P1-11: Verify pair semantics from pair manifest
# --------------------------------------------------------------------------- #


class TestVerifyPairSemantics:
    """P1-11: _verify_pair_semantics reads pair_manifest.json."""

    def test_pass_with_valid_pairs(self, tmp_path, fv):
        pairs = [{
            "pair_type": "visual_vs_fact_same_image",
            "left_sample_id": "s1",
            "right_sample_id": "s2",
            "expected_route_effect": "conflict",
            "controlled": ["image"],
            "changed": ["fact"],
        }]
        _write_json(tmp_path / f"{BENCHMARK}_pair_manifest.json", pairs)
        failures: list[str] = []
        rec = fv._verify_pair_semantics(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.PASS

    def test_not_applicable_when_no_pair_manifest(self, tmp_path, fv):
        failures: list[str] = []
        rec = fv._verify_pair_semantics(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.NOT_APPLICABLE

    def test_fail_with_invalid_pair_type(self, tmp_path, fv):
        pairs = [{
            "pair_type": "nonexistent_type",
            "expected_route_effect": "conflict",
            "controlled": [],
            "changed": [],
        }]
        _write_json(tmp_path / f"{BENCHMARK}_pair_manifest.json", pairs)
        failures: list[str] = []
        rec = fv._verify_pair_semantics(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL

    def test_fail_when_missing_expected_route_effect(self, tmp_path, fv):
        pairs = [{
            "pair_type": "visual_vs_fact_same_image",
            "expected_route_effect": None,
            "controlled": ["image"],
            "changed": ["fact"],
        }]
        _write_json(tmp_path / f"{BENCHMARK}_pair_manifest.json", pairs)
        failures: list[str] = []
        rec = fv._verify_pair_semantics(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL


# --------------------------------------------------------------------------- #
# P1-12: Verify split invariants from split manifest
# --------------------------------------------------------------------------- #


class TestVerifySplitInvariants:
    """P1-12: _verify_split_invariants reads split_manifest.json."""

    def test_pass_with_clean_splits(self, tmp_path, fv):
        data = {
            "dataset": BENCHMARK,
            "splits": [{
                "name": "default",
                "counts": {"retain_train": 10, "retain_eval": 5, "forget": 3},
                "invariant_issues": [],
            }],
        }
        _write_json(tmp_path / f"{BENCHMARK}_split_manifest.json", data)
        failures: list[str] = []
        rec = fv._verify_split_invariants(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.PASS

    def test_fail_when_invariant_issues_present(self, tmp_path, fv):
        data = {
            "dataset": BENCHMARK,
            "splits": [{
                "name": "default",
                "counts": {"retain_train": 10, "retain_eval": 5, "forget": 3},
                "invariant_issues": ["forget overlaps retain"],
            }],
        }
        _write_json(tmp_path / f"{BENCHMARK}_split_manifest.json", data)
        failures: list[str] = []
        rec = fv._verify_split_invariants(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL

    def test_fail_when_no_split_manifest(self, tmp_path, fv):
        failures: list[str] = []
        rec = fv._verify_split_invariants(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL

    def test_fail_when_forget_without_retain(self, tmp_path, fv):
        data = {
            "dataset": BENCHMARK,
            "splits": [{
                "name": "default",
                "counts": {"retain_train": 0, "retain_eval": 0, "forget": 5},
                "invariant_issues": [],
            }],
        }
        _write_json(tmp_path / f"{BENCHMARK}_split_manifest.json", data)
        failures: list[str] = []
        rec = fv._verify_split_invariants(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL


# --------------------------------------------------------------------------- #
# P1-13: Recompute every checksum independently
# --------------------------------------------------------------------------- #


class TestVerifyChecksums:
    """P1-13: _verify_checksums recomputes SHA-256 independently."""

    def test_pass_with_valid_checksums(self, tmp_path, fv):
        # Create some artifacts.
        art1 = tmp_path / "artifact1.json"
        art1.write_text('{"key": "value"}')
        art2 = tmp_path / "artifact2.jsonl"
        art2.write_text('{"row": 1}\n')

        ckdata = {
            "artifact1.json": _sha256(art1.read_bytes()),
            "artifact2.jsonl": _sha256(art2.read_bytes()),
        }
        _write_json(tmp_path / f"{BENCHMARK}_checksums.json", ckdata)

        # Export manifest referencing checksums.
        _write_json(tmp_path / f"{BENCHMARK}_export_manifest.json", {
            "paths": {"artifact1.json": True, "artifact2.jsonl": True,
                       "checksums": True, f"{BENCHMARK}_checksums.json": True},
        })

        failures: list[str] = []
        rec = fv._verify_checksums(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.PASS

    def test_fail_when_checksum_mismatch(self, tmp_path, fv):
        art1 = tmp_path / "artifact1.json"
        art1.write_text('{"key": "value"}')

        ckdata = {"artifact1.json": "0" * 64}  # wrong hash
        _write_json(tmp_path / f"{BENCHMARK}_checksums.json", ckdata)

        failures: list[str] = []
        rec = fv._verify_checksums(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL
        assert any("mismatch" in f for f in failures)

    def test_fail_when_artifact_missing(self, tmp_path, fv):
        ckdata = {"nonexistent.json": "abc123"}
        _write_json(tmp_path / f"{BENCHMARK}_checksums.json", ckdata)

        failures: list[str] = []
        rec = fv._verify_checksums(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL

    def test_fail_when_self_reference(self, tmp_path, fv):
        ckdata = {f"{BENCHMARK}_checksums.json": "abc"}
        _write_json(tmp_path / f"{BENCHMARK}_checksums.json", ckdata)

        failures: list[str] = []
        rec = fv._verify_checksums(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL

    def test_modified_artifact_fails(self, tmp_path, fv):
        """Tampering with an artifact after checksum computation must fail."""
        art1 = tmp_path / "artifact1.json"
        art1.write_text('{"key": "value"}')
        original_hash = _sha256(art1.read_bytes())

        ckdata = {"artifact1.json": original_hash}
        _write_json(tmp_path / f"{BENCHMARK}_checksums.json", ckdata)

        # Tamper with the artifact.
        art1.write_text('{"key": "TAMPERED"}')

        failures: list[str] = []
        rec = fv._verify_checksums(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL


# --------------------------------------------------------------------------- #
# P1-14: Strict-mode SKIPs fail for required checks
# --------------------------------------------------------------------------- #


class TestCheckResultModel:
    """P1-14: PASS/FAIL/NOT_APPLICABLE model."""

    def test_check_result_enum_values(self, fv):
        assert fv.CheckResult.PASS.value == "PASS"
        assert fv.CheckResult.FAIL.value == "FAIL"
        assert fv.CheckResult.NOT_APPLICABLE.value == "NOT_APPLICABLE"

    def test_check_record_required_flag(self, fv):
        rec = fv.CheckRecord("test", fv.CheckResult.NOT_APPLICABLE, required=True)
        assert rec.required is True
        rec2 = fv.CheckRecord("test", fv.CheckResult.NOT_APPLICABLE, required=False)
        assert rec2.required is False

    def test_strict_mode_fails_required_not_applicable(self, fv):
        """Under --strict, required NOT_APPLICABLE checks should be failures."""
        # Simulate the strict check logic from main_check.
        records = [
            fv.CheckRecord("required_check", fv.CheckResult.NOT_APPLICABLE, required=True),
            fv.CheckRecord("optional_check", fv.CheckResult.NOT_APPLICABLE, required=False),
        ]
        strict_failures = [
            r for r in records
            if r.required and r.result == fv.CheckResult.NOT_APPLICABLE
        ]
        assert len(strict_failures) == 1
        assert strict_failures[0].name == "required_check"


# --------------------------------------------------------------------------- #
# P1-15: Non-zero coverage for required smoke paths
# --------------------------------------------------------------------------- #


class TestVerifyCoverage:
    """P1-15: _verify_coverage requires non-zero train/eval rows."""

    def test_pass_with_nonzero_coverage(self, tmp_path, fv):
        train_qa = tmp_path / f"{BENCHMARK}_celeba40_visual_qa_train.jsonl"
        eval_qa = tmp_path / f"{BENCHMARK}_celeba40_visual_qa_eval.jsonl"
        _write_jsonl(train_qa, [{"q": "test"}])
        _write_jsonl(eval_qa, [{"q": "test"}])
        # P2-22: need all required families present.
        route_rows = [
            {"probe_family": "direct_visual", "target_attribute": "X",
             "answer_label": True, "answer_text": "yes"},
            {"probe_family": "name_only", "target_attribute": None,
             "target_fact_id": "f1", "answer_text": "Alice"},
            {"probe_family": "wrong_name", "target_attribute": "Y",
             "answer_label": False, "answer_text": "no"},
            {"probe_family": "visual_text_conflict", "target_attribute": "Z",
             "answer_label": True, "answer_text": "yes"},
        ]
        _write_jsonl(tmp_path / f"{BENCHMARK}_route_probes.jsonl", route_rows)

        failures: list[str] = []
        rec = fv._verify_coverage(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.PASS

    def test_fail_when_train_empty(self, tmp_path, fv):
        eval_qa = tmp_path / f"{BENCHMARK}_celeba40_visual_qa_eval.jsonl"
        _write_jsonl(eval_qa, [{"q": "test"}])
        # No train QA file → train_rows == 0

        failures: list[str] = []
        rec = fv._verify_coverage(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL

    def test_fail_when_eval_empty(self, tmp_path, fv):
        train_qa = tmp_path / f"{BENCHMARK}_celeba40_visual_qa_train.jsonl"
        _write_jsonl(train_qa, [{"q": "test"}])
        # No eval QA file → eval_rows == 0

        failures: list[str] = []
        rec = fv._verify_coverage(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL


# --------------------------------------------------------------------------- #
# P0-3: Coverage-aware smoke sampling
# --------------------------------------------------------------------------- #


class TestSelectSmokeSubset:
    """P0-3: select_smoke_subset provides coverage-aware selection."""

    def _make_samples(self):
        return [
            {
                "source_sample_id": "s1",
                "identity_id": "alice",
                "image_uri": "alice.jpg",
                "source_split": "train",
                "profile_facts": [{"fact": "likes cats"}],
                "source_metadata": {"is_multiview": False},
            },
            {
                "source_sample_id": "s2",
                "identity_id": "bob",
                "image_uri": "bob.jpg",
                "source_split": "eval",
                "profile_facts": [],
                "source_metadata": {"is_multiview": False},
            },
            {
                "source_sample_id": "s3",
                "identity_id": "carol",
                "image_uri": None,
                "source_split": "exclude",
                "profile_facts": [],
                "source_metadata": {"is_multiview": False},
            },
            {
                "source_sample_id": "s4",
                "identity_id": "dave",
                "image_uri": "dave.jpg",
                "source_split": "train",
                "profile_facts": [],
                "source_metadata": {"is_multiview": True},
            },
        ]

    def test_selects_multiple_identities(self, fv):
        samples = self._make_samples()
        result = fv.select_smoke_subset(samples, min_identities=3)
        assert len(result["coverage"]["identities"]) >= 3

    def test_requires_image_bearing(self, fv):
        samples = self._make_samples()
        result = fv.select_smoke_subset(samples, min_image_bearing=2)
        assert len(result["coverage"]["image_bearing_identities"]) >= 2

    def test_reports_issues_when_coverage_unmet(self, fv):
        samples = [
            {"source_sample_id": "s1", "identity_id": "alice",
             "image_uri": "a.jpg", "source_split": "train"},
        ]
        result = fv.select_smoke_subset(samples, min_identities=3)
        assert len(result["issues"]) > 0

    def test_coverage_counts_populated(self, fv):
        samples = self._make_samples()
        result = fv.select_smoke_subset(samples, min_identities=2)
        cov = result["coverage"]
        assert "selected_samples" in cov
        assert "identities" in cov
        assert "image_bearing_identities" in cov
        assert "splits_seen" in cov


# --------------------------------------------------------------------------- #
# P1-11: FIUBench smoke-selection integration test
# --------------------------------------------------------------------------- #


class TestFIUBenchSmokeSelection:
    """P1-11: integration test with a realistic FIUBench-like fixture.

    Fixture:
    - identity A = retain (→ train), 2 distinct images (multiview)
    - identity B = evaluation (→ eval), 1 image
    - identity C = forget (→ exclude), 1 image
    - identity D = retain (→ train), 2 samples, similar visual attrs to A
    """

    _NS = "extended_attributes.celeba40"

    def _va(self, **attrs: bool) -> dict:
        """Build visual_attributes dict with high-confidence celeba40 entries."""
        return {
            f"{self._NS}.{k}": {"label": v, "confidence_band": "high"}
            for k, v in attrs.items()
        }

    def _make_fixture(self) -> list[dict]:
        return [
            # Identity A: retain → train, image 1
            {
                "source_sample_id": "A-1",
                "identity_id": "identity_A",
                "image_uri": "A_img1.jpg",
                "source_split": "retain",
                "profile_facts": [{"fact": "wears glasses"}],
                "visual_attributes": self._va(Eyeglasses=True, Smiling=False),
            },
            # Identity A: second distinct image (multiview)
            {
                "source_sample_id": "A-2",
                "identity_id": "identity_A",
                "image_uri": "A_img2.jpg",
                "source_split": "retain",
                "profile_facts": [],
                "visual_attributes": self._va(Eyeglasses=True, Smiling=False),
            },
            # Identity B: evaluation → eval
            {
                "source_sample_id": "B-1",
                "identity_id": "identity_B",
                "image_uri": "B_img1.jpg",
                "source_split": "evaluation",
                "profile_facts": [],
                "visual_attributes": self._va(Eyeglasses=False, Smiling=True),
            },
            # Identity C: forget → exclude
            {
                "source_sample_id": "C-1",
                "identity_id": "identity_C",
                "image_uri": "C_img1.jpg",
                "source_split": "forget",
                "profile_facts": [],
                "visual_attributes": self._va(Eyeglasses=False, Smiling=False),
            },
            # Identity D: retain → train, wrong-name candidate for A
            {
                "source_sample_id": "D-1",
                "identity_id": "identity_D",
                "image_uri": "D_img1.jpg",
                "source_split": "retain",
                "profile_facts": [],
                "visual_attributes": self._va(Eyeglasses=True, Smiling=True),
            },
            # Identity D: second sample (needed for wrong-name eligibility)
            {
                "source_sample_id": "D-2",
                "identity_id": "identity_D",
                "image_uri": "D_img2.jpg",
                "source_split": "retain",
                "profile_facts": [],
                "visual_attributes": self._va(Eyeglasses=True, Smiling=True),
            },
        ]

    def test_fiubench_smoke_all_splits(self, fv):
        """Selector covers train, eval, and exclude splits."""
        samples = self._make_fixture()
        result = fv.select_smoke_subset(
            samples, min_identities=3, require_multiview=False,
        )
        cov = result["coverage"]
        assert "train" in cov["splits_seen"]
        assert "eval" in cov["splits_seen"]
        assert "exclude" in cov["splits_seen"]

    def test_fiubench_smoke_identity_minimum(self, fv):
        """Selector selects >= 3 identities."""
        samples = self._make_fixture()
        result = fv.select_smoke_subset(samples, min_identities=3)
        assert len(result["coverage"]["identities"]) >= 3

    def test_fiubench_smoke_profile_fact(self, fv):
        """At least one selected identity has profile facts."""
        samples = self._make_fixture()
        result = fv.select_smoke_subset(samples, min_identities=3)
        assert result["coverage"]["has_profile_facts"] is True

    def test_fiubench_smoke_wrong_name(self, fv):
        """Wrong-name candidate pair exists in the fixture data.

        The production ``find_wrong_name_candidates`` helper operates on
        identity groups.  We verify the fixture provides valid pairs from
        the full data (selector coverage is a separate concern).
        """
        from route_data.build.conflict_generation import find_wrong_name_candidates
        samples = self._make_fixture()
        by_identity: dict[str, list] = {}
        for s in samples:
            by_identity.setdefault(s["identity_id"], []).append(s)
        pairs = find_wrong_name_candidates(by_identity)
        assert len(pairs) > 0, "fixture should have valid wrong-name pairs"
        # At least one pair should involve identity_A or identity_D.
        ids_in_pairs = {t for t, _, _ in pairs} | {c for _, c, _ in pairs}
        assert "identity_A" in ids_in_pairs or "identity_D" in ids_in_pairs

    def test_fiubench_smoke_multiview(self, fv):
        """Multiview is satisfied when requested (identity A has 2 images)."""
        samples = self._make_fixture()
        result = fv.select_smoke_subset(
            samples, min_identities=3, require_multiview=True,
        )
        assert result["coverage"]["has_multiview"] is True
        # No multiview issue should be raised.
        assert not any("multiview" in iss for iss in result["issues"])

    def test_fiubench_smoke_no_issues(self, fv):
        """Full coverage: no issues for splits, multiview, identities, facts."""
        samples = self._make_fixture()
        result = fv.select_smoke_subset(
            samples,
            min_identities=3,
            require_multiview=True,
        )
        # All core coverage conditions are met.
        assert not any("train" in iss for iss in result["issues"])
        assert not any("eval" in iss for iss in result["issues"])
        assert not any("exclude" in iss for iss in result["issues"])
        assert not any("multiview" in iss for iss in result["issues"])
        assert not any("identities" in iss for iss in result["issues"])
        assert not any("profile facts" in iss for iss in result["issues"])


# --------------------------------------------------------------------------- #
# P0-2: Immutable-revision bypass restricted to golden fixture
# --------------------------------------------------------------------------- #


class TestImmutableBypassRestriction:
    """P0-2: ROUTE_DATA_SKIP_IMMUTABLE_CHECK only for golden fixture."""

    def test_final_verify_references_golden_fixture_bypass(self, fv):
        content = (SCRIPTS_DIR / "final_verify.py").read_text()
        # The bypass should be conditional on golden fixture, not unconditional.
        assert "is_golden_fixture" in content or "use_golden_fixture" in content or "golden" in content
        # The os.environ for SKIP_IMMUTABLE should be inside a conditional block
        # (indented), not at module top level (column 0).
        lines = content.splitlines()
        for line in lines:
            if "os.environ" in line and "SKIP_IMMUTABLE" in line and "=" in line:
                # Must be indented (inside a function/conditional), not at column 0.
                assert line[0] in (" ", "\t"), (
                    f"Unconditional os.environ for SKIP_IMMUTABLE at module level: {line!r}"
                )


# --------------------------------------------------------------------------- #
# P2-22: Strengthen route-probe coverage gates
# --------------------------------------------------------------------------- #


class TestCoverageGates:
    """P2-22: per-family coverage minimums."""

    def test_fail_when_required_family_missing(self, tmp_path, fv):
        """Missing required families should cause FAIL."""
        train_qa = tmp_path / f"{BENCHMARK}_celeba40_visual_qa_train.jsonl"
        eval_qa = tmp_path / f"{BENCHMARK}_celeba40_visual_qa_eval.jsonl"
        _write_jsonl(train_qa, [{"q": "test"}])
        _write_jsonl(eval_qa, [{"q": "test"}])
        # Only direct_visual — missing name_only, wrong_name, visual_text_conflict.
        route_rows = [
            {"probe_family": "direct_visual", "target_attribute": "X",
             "answer_label": True, "answer_text": "yes"},
        ]
        _write_jsonl(tmp_path / f"{BENCHMARK}_route_probes.jsonl", route_rows)

        failures: list[str] = []
        rec = fv._verify_coverage(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL
        assert any("missing" in f for f in failures)

    def test_pass_when_all_required_families_present(self, tmp_path, fv):
        train_qa = tmp_path / f"{BENCHMARK}_celeba40_visual_qa_train.jsonl"
        eval_qa = tmp_path / f"{BENCHMARK}_celeba40_visual_qa_eval.jsonl"
        _write_jsonl(train_qa, [{"q": "test"}])
        _write_jsonl(eval_qa, [{"q": "test"}])
        route_rows = [
            {"probe_family": "direct_visual", "target_attribute": "A",
             "answer_label": True, "answer_text": "yes"},
            {"probe_family": "name_only", "target_attribute": None,
             "target_fact_id": "f1", "answer_text": "Alice"},
            {"probe_family": "wrong_name", "target_attribute": "B",
             "answer_label": False, "answer_text": "no"},
            {"probe_family": "visual_text_conflict", "target_attribute": "C",
             "answer_label": True, "answer_text": "yes"},
        ]
        _write_jsonl(tmp_path / f"{BENCHMARK}_route_probes.jsonl", route_rows)

        failures: list[str] = []
        rec = fv._verify_coverage(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.PASS


# --------------------------------------------------------------------------- #
# P2-23: Per-attribute route balance reporting
# --------------------------------------------------------------------------- #


class TestRouteBalance:
    """P2-23: _verify_route_balance reports per-attribute stats."""

    def test_not_applicable_when_no_route_file(self, tmp_path, fv):
        failures: list[str] = []
        rec = fv._verify_route_balance(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.NOT_APPLICABLE

    def test_pass_with_balanced_attributes(self, tmp_path, fv):
        route_rows = [
            {"probe_family": "direct_visual", "target_attribute": "Eyeglasses",
             "answer_label": True, "answer_text": "yes"},
            {"probe_family": "direct_visual", "target_attribute": "Eyeglasses",
             "answer_label": False, "answer_text": "no"},
            {"probe_family": "wrong_name", "target_attribute": "Smiling",
             "answer_label": True, "answer_text": "yes"},
            {"probe_family": "wrong_name", "target_attribute": "Smiling",
             "answer_label": False, "answer_text": "no"},
        ]
        _write_jsonl(tmp_path / f"{BENCHMARK}_route_probes.jsonl", route_rows)

        failures: list[str] = []
        rec = fv._verify_route_balance(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.PASS
        # Balance report should be persisted.
        balance_path = tmp_path / f"{BENCHMARK}_route_balance_report.json"
        assert balance_path.exists()
        report = json.loads(balance_path.read_text())
        assert "Eyeglasses" in report["attributes"]
        assert report["attributes"]["Eyeglasses"]["positive"] == 1
        assert report["attributes"]["Eyeglasses"]["negative"] == 1
        assert len(report["unbalanced_attributes"]) == 0

    def test_flags_unbalanced_attribute(self, tmp_path, fv):
        route_rows = [
            {"probe_family": "direct_visual", "target_attribute": "Young",
             "answer_label": True, "answer_text": "yes"},
            {"probe_family": "direct_visual", "target_attribute": "Young",
             "answer_label": True, "answer_text": "yes"},
        ]
        _write_jsonl(tmp_path / f"{BENCHMARK}_route_probes.jsonl", route_rows)

        failures: list[str] = []
        rec = fv._verify_route_balance(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.PASS  # reporting only, not a failure
        balance_path = tmp_path / f"{BENCHMARK}_route_balance_report.json"
        report = json.loads(balance_path.read_text())
        assert any("Young" in a for a in report["unbalanced_attributes"])


# --------------------------------------------------------------------------- #
# P2-24: Persist and verify manual audit report
# --------------------------------------------------------------------------- #


class TestManualAuditReport:
    """P2-24: _verify_manual_audit_report checks audit report schema."""

    def test_not_applicable_when_report_missing(self, tmp_path, fv):
        failures: list[str] = []
        rec = fv._verify_manual_audit_report(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.NOT_APPLICABLE

    def test_pass_with_valid_report(self, tmp_path, fv):
        report = {
            "audit_report_version": "v1",
            "dataset": BENCHMARK,
            "total_items": 2,
            "unreviewed_items": 0,
            "critical_failures": 0,
            "gate_pass": True,
            "items": [
                {
                    "audit_id": "src-0001",
                    "category": "source_mapping",
                    "sample_id": "s1",
                    "identity_id": "id1",
                    "image_uri": "img1.jpg",
                    "attribute_or_fact": "Eyeglasses",
                    "automatic_checks": {},
                    "review_outcome": "pass",
                    "review_note": "OK",
                },
                {
                    "audit_id": "probe-0001",
                    "category": "route_probe",
                    "sample_id": "s2",
                    "identity_id": "id2",
                    "image_uri": None,
                    "attribute_or_fact": "fact1",
                    "automatic_checks": {},
                    "review_outcome": "pass",
                    "review_note": "OK",
                },
            ],
            "summary": {},
        }
        _write_json(tmp_path / f"{BENCHMARK}_manual_audit_report.json", report)

        failures: list[str] = []
        rec = fv._verify_manual_audit_report(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.PASS
        assert not failures

    def test_fail_with_critical_failures(self, tmp_path, fv):
        report = {
            "audit_report_version": "v1",
            "dataset": BENCHMARK,
            "total_items": 1,
            "unreviewed_items": 0,
            "critical_failures": 1,
            "gate_pass": False,
            "items": [
                {
                    "audit_id": "src-0001",
                    "category": "source_mapping",
                    "sample_id": "s1",
                    "identity_id": "id1",
                    "image_uri": "img1.jpg",
                    "attribute_or_fact": "Eyeglasses",
                    "automatic_checks": {},
                    "review_outcome": "fail",
                    "review_note": "bad",
                },
            ],
            "summary": {},
        }
        _write_json(tmp_path / f"{BENCHMARK}_manual_audit_report.json", report)

        failures: list[str] = []
        rec = fv._verify_manual_audit_report(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL
        assert any("critical failures" in f for f in failures)

    def test_fail_with_invalid_item_schema(self, tmp_path, fv):
        report = {
            "audit_report_version": "v1",
            "dataset": BENCHMARK,
            "total_items": 1,
            "critical_failures": 0,
            "items": [
                {"sample_id": "s1"},  # missing required keys
            ],
            "summary": {},
        }
        _write_json(tmp_path / f"{BENCHMARK}_manual_audit_report.json", report)

        failures: list[str] = []
        rec = fv._verify_manual_audit_report(tmp_path, BENCHMARK, failures)
        assert rec.result == fv.CheckResult.FAIL
        assert any("schema" in f or "invalid" in f for f in failures)


@pytest.fixture(scope="module")
def ag():
    """Import audit_gate.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "audit_gate", SCRIPTS_DIR / "audit_gate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestAuditGateBuildItems:
    """P2-6: audit_gate._build_audit_items produces structured items."""

    def test_build_items_from_source_mappings(self, ag):
        items = ag._build_audit_items(
            source_mappings=[{"identity_id": "id1", "raw_split": "train", "target_split": "train"}],
            pos_labels=[], neg_labels=[], probes=[], pairs=[], facts=[], failures=[],
        )
        assert len(items) == 1
        assert items[0]["category"] == "source_mapping"
        assert items[0]["review_outcome"] == "unreviewed"

    def test_build_items_pair_schema_valid(self, ag):
        pairs = [{
            "pair_id": "p1", "pair_type": "cross_image_attribute_state",
            "left_sample_id": "s1", "right_sample_id": "s2",
            "attribute": "Eyeglasses",
            "controlled": ["attribute"], "changed": ["state"],
            "expected_route_effect": "flip",
            "left_label": True, "right_label": False,
        }]
        items = ag._build_audit_items([], [], [], [], pairs, [], failures=[])
        pair_item = next(it for it in items if it["category"] == "pair")
        assert pair_item["automatic_checks"]["schema_valid"] is True
        assert pair_item["review_outcome"] == "unreviewed"

    def test_persist_audit_report_writes_json(self, tmp_path, ag):
        # P2-9: unreviewed items block the gate.
        items = [{
            "audit_id": "src-0001", "category": "source_mapping",
            "sample_id": "s1", "identity_id": "id1", "image_uri": None,
            "attribute_or_fact": "split=train",
            "automatic_checks": {},
            "review_outcome": "unreviewed", "review_note": "ok",
        }]
        path = ag._persist_audit_report("testbench", tmp_path, items, failures=[])
        assert path is not None
        assert path.exists()
        report = json.loads(path.read_text())
        assert report["total_items"] == 1
        assert report["unreviewed_items"] == 1
        assert report["gate_pass"] is False

    def test_persist_audit_report_gate_pass_when_all_reviewed(self, tmp_path, ag):
        """P2-9: gate passes when zero failures, zero unreviewed, zero critical."""
        items = [{
            "audit_id": "src-0001", "category": "source_mapping",
            "sample_id": "s1", "identity_id": "id1", "image_uri": None,
            "attribute_or_fact": "split=train",
            "automatic_checks": {},
            "review_outcome": "pass", "review_note": "ok",
        }]
        path = ag._persist_audit_report("testbench", tmp_path, items, failures=[])
        report = json.loads(path.read_text())
        assert report["gate_pass"] is True
        assert report["unreviewed_items"] == 0


# --------------------------------------------------------------------------- #
# P3-28: Pre-generation gate
# --------------------------------------------------------------------------- #


class TestPreGenerationGate:
    """P3-28: pre-generation gate requires zero unresolved warnings."""

    def test_gate_passes_when_all_zero(self, fv, tmp_path):
        """Gate passes when all conditions are zero."""
        gate = fv.check_pregeneration_gate(
            tmp_path,
            "test",
            pending_revisions=0,
            strict_verification_failures=0,
            critical_skips=0,
            checksum_mismatches=0,
            source_split_violations=0,
            route_semantic_violations=0,
            manual_audit_critical_failures=0,
        )
        assert gate["gate_passed"] is True
        assert "failing_conditions" not in gate

    def test_gate_fails_when_any_non_zero(self, fv, tmp_path):
        """Gate fails when any condition is non-zero."""
        gate = fv.check_pregeneration_gate(
            tmp_path,
            "test",
            pending_revisions=1,
            strict_verification_failures=0,
            critical_skips=0,
            checksum_mismatches=0,
            source_split_violations=0,
            route_semantic_violations=0,
            manual_audit_critical_failures=0,
        )
        assert gate["gate_passed"] is False
        assert "pending_revisions" in gate["failing_conditions"]

    def test_gate_reports_all_failing_conditions(self, fv, tmp_path):
        """Gate reports all failing conditions."""
        gate = fv.check_pregeneration_gate(
            tmp_path,
            "test",
            pending_revisions=2,
            strict_verification_failures=1,
            critical_skips=0,
            checksum_mismatches=3,
            source_split_violations=0,
            route_semantic_violations=0,
            manual_audit_critical_failures=1,
        )
        assert gate["gate_passed"] is False
        assert len(gate["failing_conditions"]) == 4
        assert "pending_revisions" in gate["failing_conditions"]
        assert "strict_verification_failures" in gate["failing_conditions"]
        assert "checksum_mismatches" in gate["failing_conditions"]
        assert "manual_audit_critical_failures" in gate["failing_conditions"]


# --------------------------------------------------------------------------- #
# P1-12: smoke manifest conformance verification
# --------------------------------------------------------------------------- #


class TestSmokeManifestConformance:
    """P1-12: _verify_smoke_manifest_conformance checks."""

    @pytest.fixture(scope="class")
    def fv(self):
        spec = importlib.util.spec_from_file_location(
            "final_verify", SCRIPTS_DIR / "final_verify.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _make_manifest(self, tmp_path: Path, sample_ids: list[str], image_ids: list[str | None] | None = None) -> Path:
        """Create a minimal smoke manifest JSON.

        ``image_ids`` is aligned with ``sample_ids``; ``None`` entries (or a
        ``None`` list) mean no image for that sample.
        """
        samples = []
        for i, sid in enumerate(sample_ids):
            img = image_ids[i] if image_ids is not None and i < len(image_ids) else None
            samples.append({
                "sample_id": sid,
                "identity_id": f"id_{sid}",
                "image_uri": img,
            })
        manifest = {
            "dataset": "test",
            "selection_version": "smoke_v1",
            "selected_source_sample_ids": sample_ids,
            "selected_identity_ids": sorted({f"id_{s}" for s in sample_ids}),
            "samples": samples,
            "coverage": {},
        }
        path = tmp_path / "smoke_manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def _make_export_dir(self, tmp_path: Path, output_ids: list[str], image_ids: list[str | None] | None = None) -> Path:
        """Create a minimal export directory with processed.jsonl.

        ``image_ids`` is aligned with ``output_ids``; ``None`` entries mean no image.
        """
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        lines = []
        for i, sid in enumerate(output_ids):
            img = image_ids[i] if image_ids is not None and i < len(image_ids) else None
            lines.append(json.dumps({
                "source_sample_id": sid,
                "identity_id": f"id_{sid}",
                "image_uri": img,
            }))
        (export_dir / "test_processed.jsonl").write_text("\n".join(lines) + "\n")
        return export_dir

    def test_not_applicable_without_manifest(self, fv, tmp_path):
        """Without a smoke manifest, the check is NOT_APPLICABLE."""
        failures: list[str] = []
        rec = fv._verify_smoke_manifest_conformance(
            tmp_path, "test", failures, smoke_manifest_path=None,
        )
        assert rec.result == fv.CheckResult.NOT_APPLICABLE
        assert failures == []

    def test_fail_when_manifest_missing(self, fv, tmp_path):
        """Fail when the manifest file does not exist."""
        failures: list[str] = []
        rec = fv._verify_smoke_manifest_conformance(
            tmp_path, "test", failures,
            smoke_manifest_path=tmp_path / "nonexistent.json",
        )
        assert rec.result == fv.CheckResult.FAIL
        assert len(failures) == 1

    def test_fail_on_empty_allowlist(self, fv, tmp_path):
        """Fail when manifest has empty selected_source_sample_ids."""
        manifest_path = tmp_path / "empty_manifest.json"
        manifest_path.write_text(json.dumps({
            "dataset": "test",
            "selected_source_sample_ids": [],
            "samples": [],
        }))
        failures: list[str] = []
        rec = fv._verify_smoke_manifest_conformance(
            tmp_path, "test", failures, smoke_manifest_path=manifest_path,
        )
        assert rec.result == fv.CheckResult.FAIL
        assert "empty" in failures[0].lower()

    def test_pass_when_output_subset_of_manifest(self, fv, tmp_path):
        """Pass when all output IDs are in the manifest."""
        manifest_path = self._make_manifest(tmp_path, ["s1", "s2", "s3"])
        export_dir = self._make_export_dir(tmp_path, ["s1", "s2"])
        failures: list[str] = []
        rec = fv._verify_smoke_manifest_conformance(
            export_dir, "test", failures, smoke_manifest_path=manifest_path,
        )
        assert rec.result == fv.CheckResult.PASS
        assert failures == []

    def test_fail_on_unexpected_output_ids(self, fv, tmp_path):
        """Fail when output contains IDs not in the manifest."""
        manifest_path = self._make_manifest(tmp_path, ["s1", "s2"])
        export_dir = self._make_export_dir(tmp_path, ["s1", "s2", "s99"])
        failures: list[str] = []
        rec = fv._verify_smoke_manifest_conformance(
            export_dir, "test", failures, smoke_manifest_path=manifest_path,
        )
        assert rec.result == fv.CheckResult.FAIL
        assert "unexpected" in failures[0].lower()

    def test_fail_when_image_bearing_sample_unscored(self, fv, tmp_path):
        """Fail when a manifest image-bearing sample is not in the output."""
        # Manifest has s1 (image), s2 (image), s3 (image).
        manifest_path = self._make_manifest(
            tmp_path, ["s1", "s2", "s3"],
            image_ids=["img1.jpg", "img2.jpg", "img3.jpg"],
        )
        # Output only has s1 and s2 — s3 image-bearing sample was not scored.
        export_dir = self._make_export_dir(
            tmp_path, ["s1", "s2"],
            image_ids=["img1.jpg", "img2.jpg"],
        )
        failures: list[str] = []
        rec = fv._verify_smoke_manifest_conformance(
            export_dir, "test", failures, smoke_manifest_path=manifest_path,
        )
        assert rec.result == fv.CheckResult.FAIL
        assert "not scored" in failures[0].lower()

    def test_sha_mismatch_with_score_manifest(self, fv, tmp_path):
        """Fail when manifest SHA does not match score manifest provenance."""
        manifest_path = self._make_manifest(tmp_path, ["s1"])
        export_dir = self._make_export_dir(tmp_path, ["s1"])
        # Write a score manifest with a wrong SHA.
        score_m = {
            "selection_manifest_sha256": "deadbeef" * 8,
        }
        (export_dir / "test_score_manifest.json").write_text(json.dumps(score_m))
        failures: list[str] = []
        rec = fv._verify_smoke_manifest_conformance(
            export_dir, "test", failures, smoke_manifest_path=manifest_path,
        )
        assert rec.result == fv.CheckResult.FAIL
        assert "SHA" in failures[0]

    def test_sha_match_passes(self, fv, tmp_path):
        """Pass when manifest SHA matches score manifest provenance."""
        manifest_path = self._make_manifest(tmp_path, ["s1"])
        export_dir = self._make_export_dir(tmp_path, ["s1"])
        actual_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        score_m = {
            "selection_manifest_sha256": actual_sha,
        }
        (export_dir / "test_score_manifest.json").write_text(json.dumps(score_m))
        failures: list[str] = []
        rec = fv._verify_smoke_manifest_conformance(
            export_dir, "test", failures, smoke_manifest_path=manifest_path,
        )
        assert rec.result == fv.CheckResult.PASS
        assert failures == []

