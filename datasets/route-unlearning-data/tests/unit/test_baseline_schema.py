"""Schema tests for FIUBench baseline results (Stage 2.4).

These tests exercise the :class:`BaselineRunner` with the *stub* backend so
that no GPU is required.  They verify the structural invariants of the
result schema that downstream analysis stages depend on.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from route_data.config import ModelConfig
from route_data.eval.baseline_runner import (
    ALL_FAMILIES,
    BaselineProbe,
    BaselineResult,
    BaselineRunner,
    select_smoke_probes,
    write_smoke_manifest,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

# One probe per family (5 probes total) extracted from the frozen JSONL.
_PROBES_RAW: list[dict] = [
    {
        "probe_id": "probe_dv_test01",
        "sample_id": "fiubench:aaaa:qa:0:original",
        "identity_id": "aaaa",
        "benchmark": "fiubench",
        "probe_family": "direct_visual",
        "modality": "image_only",
        "question": "Based only on the current image, Is the person bald?",
        "expected_evidence_source": "visual",
        "paired_sample_id": None,
        "controlled_variables": ["image"],
        "image_uri": "/fake/aaaa_dv.jpg",
        "image_sha256": "aaa111",
        "registry_hash": "28481d2ebb85a6fb",
        "target_attribute": "Bald",
        "target_fact_id": None,
        "target_fact_relation": None,
        "target_fact_value": None,
        "answer_label": False,
        "answer_text": "no",
    },
    {
        "probe_id": "probe_ipn_test01",
        "sample_id": "fiubench:aaaa:qa:0:original",
        "identity_id": "aaaa",
        "benchmark": "fiubench",
        "probe_family": "image_plus_name",
        "modality": "image_text",
        "question": "The person is Alice. Is the person bald?",
        "expected_evidence_source": "visual",
        "paired_sample_id": None,
        "controlled_variables": ["image"],
        "image_uri": "/fake/aaaa_ipn.jpg",
        "image_sha256": "aaa111",
        "registry_hash": "28481d2ebb85a6fb",
        "target_attribute": "Bald",
        "answer_label": False,
        "answer_text": "no",
    },
    {
        "probe_id": "probe_wn_test01",
        "sample_id": "fiubench:aaaa:qa:0:original",
        "identity_id": "aaaa",
        "benchmark": "fiubench",
        "probe_family": "wrong_name",
        "modality": "image_text",
        "question": "The person is Bob. Is the person bald?",
        "expected_evidence_source": "visual",
        "paired_sample_id": None,
        "controlled_variables": ["image"],
        "image_uri": "/fake/aaaa_wn.jpg",
        "image_sha256": "aaa111",
        "registry_hash": "28481d2ebb85a6fb",
        "target_attribute": "Bald",
        "answer_label": False,
        "answer_text": "no",
        "matched_wrong_identity_id": "bbbb",
        "matching_similarity": 0.9,
        "matching_attributes": ["Bald"],
        "candidate_rank": 1,
        "matching_strategy": "visual_attribute_signed_state_jaccard",
    },
    {
        "probe_id": "probe_vtc_test01",
        "sample_id": "fiubench:aaaa:qa:0:original",
        "identity_id": "aaaa",
        "benchmark": "fiubench",
        "probe_family": "visual_text_conflict",
        "modality": "image_text",
        "question": "Alice always shows bald. Is the person bald?",
        "expected_evidence_source": "conflict",
        "paired_sample_id": None,
        "controlled_variables": ["image"],
        "image_uri": "/fake/aaaa_vtc.jpg",
        "image_sha256": "aaa111",
        "registry_hash": "28481d2ebb85a6fb",
        "target_attribute": "Bald",
        "answer_label": False,
        "answer_text": "no",
    },
    {
        "probe_id": "probe_no_test01",
        "sample_id": "fiubench:aaaa:qa:0:original",
        "identity_id": "aaaa",
        "benchmark": "fiubench",
        "probe_family": "name_only",
        "modality": "text_only",
        "question": "Alice: What is the person's full name?",
        "expected_evidence_source": "identity_fact",
        "paired_sample_id": None,
        "controlled_variables": ["image"],
        "image_uri": None,
        "image_sha256": None,
        "registry_hash": "28481d2ebb85a6fb",
        "target_attribute": None,
        "target_fact_id": "fiubench_qa_00",
        "target_fact_relation": "What is the person's full name?",
        "target_fact_value": "Alice Smith",
        "source_qa_index": 0,
        "original_question": "What is the person's full name?",
        "original_answer": "Alice Smith",
        "question_variant": "canonical",
        "answer_label": None,
        "answer_text": "Alice Smith",
    },
    # Second identity "bbbb" — all 5 families (for smoke selector).
    {
        "probe_id": "probe_dv_test02",
        "sample_id": "fiubench:bbbb:qa:0:original",
        "identity_id": "bbbb",
        "benchmark": "fiubench",
        "probe_family": "direct_visual",
        "modality": "image_only",
        "question": "Based only on the current image, Is the person male?",
        "expected_evidence_source": "visual",
        "paired_sample_id": None,
        "controlled_variables": ["image"],
        "image_uri": "/fake/bbbb_dv.jpg",
        "image_sha256": "bbb222",
        "registry_hash": "28481d2ebb85a6fb",
        "target_attribute": "Male",
        "target_fact_id": None,
        "target_fact_relation": None,
        "target_fact_value": None,
        "answer_label": True,
        "answer_text": "yes",
    },
    {
        "probe_id": "probe_ipn_test02",
        "sample_id": "fiubench:bbbb:qa:0:original",
        "identity_id": "bbbb",
        "benchmark": "fiubench",
        "probe_family": "image_plus_name",
        "modality": "image_text",
        "question": "The person is Bob. Is the person male?",
        "expected_evidence_source": "visual",
        "paired_sample_id": None,
        "controlled_variables": ["image"],
        "image_uri": "/fake/bbbb_ipn.jpg",
        "image_sha256": "bbb222",
        "registry_hash": "28481d2ebb85a6fb",
        "target_attribute": "Male",
        "answer_label": True,
        "answer_text": "yes",
    },
    {
        "probe_id": "probe_wn_test02",
        "sample_id": "fiubench:bbbb:qa:0:original",
        "identity_id": "bbbb",
        "benchmark": "fiubench",
        "probe_family": "wrong_name",
        "modality": "image_text",
        "question": "The person is Alice. Is the person male?",
        "expected_evidence_source": "visual",
        "paired_sample_id": None,
        "controlled_variables": ["image"],
        "image_uri": "/fake/bbbb_wn.jpg",
        "image_sha256": "bbb222",
        "registry_hash": "28481d2ebb85a6fb",
        "target_attribute": "Male",
        "answer_label": True,
        "answer_text": "yes",
        "matched_wrong_identity_id": "aaaa",
        "matching_similarity": 0.8,
        "matching_attributes": ["Male"],
        "candidate_rank": 1,
        "matching_strategy": "visual_attribute_signed_state_jaccard",
    },
    {
        "probe_id": "probe_vtc_test02",
        "sample_id": "fiubench:bbbb:qa:0:original",
        "identity_id": "bbbb",
        "benchmark": "fiubench",
        "probe_family": "visual_text_conflict",
        "modality": "image_text",
        "question": "Bob always shows male attire. Is the person male?",
        "expected_evidence_source": "conflict",
        "paired_sample_id": None,
        "controlled_variables": ["image"],
        "image_uri": "/fake/bbbb_vtc.jpg",
        "image_sha256": "bbb222",
        "registry_hash": "28481d2ebb85a6fb",
        "target_attribute": "Male",
        "answer_label": True,
        "answer_text": "yes",
    },
    {
        "probe_id": "probe_no_test02",
        "sample_id": "fiubench:bbbb:qa:0:original",
        "identity_id": "bbbb",
        "benchmark": "fiubench",
        "probe_family": "name_only",
        "modality": "text_only",
        "question": "Bob: What is the person's full name?",
        "expected_evidence_source": "identity_fact",
        "paired_sample_id": None,
        "controlled_variables": ["image"],
        "image_uri": None,
        "image_sha256": None,
        "registry_hash": "28481d2ebb85a6fb",
        "target_attribute": None,
        "target_fact_id": "fiubench_qa_01",
        "target_fact_relation": "What is the person's full name?",
        "target_fact_value": "Bob Jones",
        "source_qa_index": 0,
        "original_question": "What is the person's full name?",
        "original_answer": "Bob Jones",
        "question_variant": "canonical",
        "answer_label": None,
        "answer_text": "Bob Jones",
    },
]


@pytest.fixture(autouse=True)
def _mock_image_loading():
    """Patch _load_image so stub tests never touch the filesystem."""
    with patch(
        "route_data.eval.baseline_runner._load_image",
        return_value=MagicMock(),
    ):
        yield


@pytest.fixture()
def probe_jsonl(tmp_path: Path) -> Path:
    """Write the 5 test probes to a temporary JSONL file."""
    path = tmp_path / "test_probes.jsonl"
    with open(path, "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in _PROBES_RAW)
    return path


@pytest.fixture()
def stub_model_config() -> ModelConfig:
    return ModelConfig(backend="stub", model_id="test-model", revision="abc123")


@pytest.fixture()
def stub_backend(stub_model_config: ModelConfig):
    from route_data.models.registry import create_backend

    return create_backend(stub_model_config)


# Protocol config for the basic runner fixture (Commit A: P0-3).
_RUNNER_PROTO: dict = {
    "forget_bucket": "forget10",
    "train_bucket": "retain15",
    "eval_bucket": None,
    "eval_fraction": 0.2,
    "eval_seed": 17,
}


@pytest.fixture()
def processed_jsonl_for_runner(tmp_path: Path) -> Path:
    """Processed JSONL with probe identities for the basic runner fixture."""
    path = tmp_path / "runner_processed.jsonl"
    rows = [
        {"identity_id": "aaaa", "source_metadata": {
            "source_subject_id": "S001",
            "official_memberships": ["forget10"],
        }},
        {"identity_id": "bbbb", "source_metadata": {
            "source_subject_id": "S002",
            "official_memberships": ["retain15"],
        }},
    ]
    with open(path, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    return path


@pytest.fixture()
def manifest_for_runner(
    tmp_path: Path,
    probe_jsonl: Path,
    processed_jsonl_for_runner: Path,
) -> Path:
    """Manifest with protocol config and correct SHAs for the runner fixture."""
    from route_data.eval.baseline_runner import _sha256_file

    path = tmp_path / "runner_manifest.json"
    data = {
        "protocol": {"canonical_protocol": dict(_RUNNER_PROTO)},
        "dataset_artifacts": {
            "route_probes": {"sha256": _sha256_file(probe_jsonl)},
            "processed_dataset": {
                "sha256": _sha256_file(processed_jsonl_for_runner),
            },
        },
    }
    path.write_text(json.dumps(data))
    return path


@pytest.fixture()
def runner(
    stub_backend,
    probe_jsonl: Path,
    tmp_path: Path,
    stub_model_config: ModelConfig,
    processed_jsonl_for_runner: Path,
    manifest_for_runner: Path,
) -> BaselineRunner:
    return BaselineRunner(
        backend=stub_backend,
        probe_path=probe_jsonl,
        output_dir=tmp_path / "output",
        model_config=stub_model_config,
        resume=False,
        dataset_manifest_path=manifest_for_runner,
        processed_dataset_path=processed_jsonl_for_runner,
    )


@pytest.fixture()
def runner_no_map(
    stub_backend,
    probe_jsonl: Path,
    tmp_path: Path,
    stub_model_config: ModelConfig,
) -> BaselineRunner:
    """Runner without processed dataset or manifest (for negative tests)."""
    return BaselineRunner(
        backend=stub_backend,
        probe_path=probe_jsonl,
        output_dir=tmp_path / "output_no_map",
        model_config=stub_model_config,
        resume=False,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestBaselineProbeSchema:
    """Verify the probe dataclass mirrors the frozen JSONL schema."""

    def test_from_dict_known_fields(self):
        probe = BaselineProbe.from_dict(_PROBES_RAW[0])
        assert probe.probe_id == "probe_dv_test01"
        assert probe.probe_family == "direct_visual"
        assert probe.answer_label is False

    def test_from_dict_ignores_unknown(self):
        d = dict(_PROBES_RAW[0])
        d["unknown_future_field"] = 42
        probe = BaselineProbe.from_dict(d)
        assert probe.probe_id == "probe_dv_test01"

    def test_wrong_name_extras(self):
        probe = BaselineProbe.from_dict(_PROBES_RAW[2])
        assert probe.matched_wrong_identity_id == "bbbb"
        assert probe.matching_similarity == pytest.approx(0.9)

    def test_name_only_extras(self):
        probe = BaselineProbe.from_dict(_PROBES_RAW[4])
        assert probe.target_fact_id == "fiubench_qa_00"
        assert probe.question_variant == "canonical"
        assert probe.image_sha256 is None

    def test_has_image_property(self):
        # _PROBES_RAW[0] has image_uri set → has_image is True
        assert BaselineProbe.from_dict(_PROBES_RAW[0]).has_image is True
        # name_only probe has image_uri=None → has_image is False
        assert BaselineProbe.from_dict(_PROBES_RAW[4]).has_image is False


class TestBaselineResultSchema:
    """Verify the result dataclass has all required fields."""

    REQUIRED_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "probe_id",
        "sample_id",
        "identity_id",
        "probe_family",
        "modality",
        "model_fingerprint",
        "model_revision",
        "question",
        "image_sha256",
        "generated_answer",
        "parsed_answer",
        "correct",
        "scoring_version",
        "prompt_hash",
        "latency_ms",
    })

    def test_all_required_fields_present(self):
        result = BaselineResult(
            probe_id="p1",
            sample_id="s1",
            identity_id="i1",
            probe_family="direct_visual",
            modality="image_only",
            model_fingerprint="fp",
            model_revision="rev",
            question="q?",
            image_sha256="sha",
            generated_answer="Yes",
            parsed_answer="Yes",
            correct=True,
        )
        d = asdict(result)
        for field_name in self.REQUIRED_FIELDS:
            assert field_name in d, f"Missing required field: {field_name}"

    def test_serializable_to_json(self):
        result = BaselineResult(
            probe_id="p1",
            sample_id="s1",
            identity_id="i1",
            probe_family="direct_visual",
            modality="image_only",
            model_fingerprint="fp",
            model_revision="rev",
            question="q?",
            image_sha256="sha",
            generated_answer="Yes",
            parsed_answer="Yes",
            correct=True,
            logp_yes=-0.1,
            logp_no=-2.3,
            p_yes=0.9,
            raw_log_margin=2.2,
            signed_answer_margin=2.2,
        )
        serialized = json.dumps(asdict(result), default=str)
        restored = json.loads(serialized)
        assert restored["probe_id"] == "p1"
        assert restored["p_yes"] == pytest.approx(0.9)


class TestBaselineRunnerProbeLoading:
    """Verify probe loading and family coverage."""

    def test_loads_all_probes(self, runner: BaselineRunner):
        assert len(runner.probes) == 10

    def test_all_five_families_present(self, runner: BaselineRunner):
        families = {p.probe_family for p in runner.probes}
        assert families == ALL_FAMILIES

    def test_no_duplicate_probe_ids(self, runner: BaselineRunner):
        ids = [p.probe_id for p in runner.probes]
        assert len(ids) == len(set(ids))


class TestBaselineRunnerExecution:
    """Verify the runner produces valid results with the stub backend."""

    def test_run_all_produces_results(self, runner: BaselineRunner):
        results = runner.run_all()
        assert len(results) == 10

    def test_candidate_scores_are_finite(self, runner: BaselineRunner):
        results = runner.run_all()
        for r in results:
            if r.logp_yes is not None:
                assert math.isfinite(r.logp_yes), f"logp_yes not finite for {r.probe_id}"
            if r.logp_no is not None:
                assert math.isfinite(r.logp_no), f"logp_no not finite for {r.probe_id}"
            if r.p_yes is not None:
                assert math.isfinite(r.p_yes), f"p_yes not finite for {r.probe_id}"
                assert 0.0 <= r.p_yes <= 1.0, f"p_yes out of range for {r.probe_id}"

    def test_image_shas_match_frozen_probes(self, runner: BaselineRunner):
        results = runner.run_all()
        probe_shas = {p.probe_id: p.image_sha256 for p in runner.probes}
        for r in results:
            assert r.image_sha256 == probe_shas[r.probe_id], (
                f"image_sha256 mismatch for {r.probe_id}"
            )

    def test_scoring_version_populated(self, runner: BaselineRunner):
        results = runner.run_all()
        for r in results:
            assert r.scoring_version, f"scoring_version empty for {r.probe_id}"

    def test_prompt_hash_populated(self, runner: BaselineRunner):
        results = runner.run_all()
        for r in results:
            assert r.prompt_hash, f"prompt_hash empty for {r.probe_id}"

    def test_latency_non_negative(self, runner: BaselineRunner):
        results = runner.run_all()
        for r in results:
            assert r.latency_ms >= 0, f"negative latency for {r.probe_id}"


class TestBaselineRunnerPersistence:
    """Verify save / resume round-trip."""

    def test_save_results_writes_jsonl(self, runner: BaselineRunner):
        runner.run_all()
        path = runner.save_results()
        assert path.is_file()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 10

    def test_generate_summary_writes_json(self, runner: BaselineRunner):
        runner.run_all()
        summary = runner.generate_summary()
        summary_path = runner.output_dir / "baseline_summary.json"
        assert summary_path.is_file()
        assert summary["total_probes"] == 10
        assert "per_family" in summary

    def test_resume_skips_cached(self, runner: BaselineRunner, tmp_path: Path):
        # First run: all 10 probes
        runner.run_all()
        runner.save_results()
        assert len(runner._results) == 10

        # Second run with resume=True: should reuse cache
        runner2 = BaselineRunner(
            backend=runner.backend,
            probe_path=runner.probe_path,
            output_dir=runner.output_dir,
            model_config=runner.model_config,
            resume=True,
            dataset_manifest_path=runner.dataset_manifest_path,
            processed_dataset_path=runner.processed_dataset_path,
        )
        assert len(runner2._results) == 10  # loaded from cache
        new_results = runner2.run_all()
        # No new results — all were cached
        assert len(new_results) == 10


class TestBaselineRunnerHashVerification:
    """Verify fail-closed hash checking."""

    def test_hash_mismatch_raises(self, runner: BaselineRunner):
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            runner.verify_input_hashes("0" * 64)

    def test_hash_match_passes(self, runner: BaselineRunner):
        from route_data.eval.baseline_runner import _sha256_file

        actual = _sha256_file(runner.probe_path)
        assert runner.verify_input_hashes(actual) is True


class TestTextMatch:
    """Verify the text-match helper for name-only probes."""

    def test_exact_match(self):
        from route_data.eval.baseline_runner import _text_match

        assert _text_match("Alice Smith", "Alice Smith") == 1.0

    def test_case_insensitive(self):
        from route_data.eval.baseline_runner import _text_match

        assert _text_match("alice smith", "Alice Smith") == 1.0

    def test_substring_match(self):
        from route_data.eval.baseline_runner import _text_match

        score = _text_match("The answer is Alice Smith.", "Alice Smith")
        assert score == 0.8

    def test_no_match(self):
        from route_data.eval.baseline_runner import _text_match

        assert _text_match("Bob Jones", "Alice Smith") == 0.0


class TestNewTextMetrics:
    """Verify the new text-metric helpers (Commit A)."""

    def test_compute_exact_match_equal(self):
        from route_data.eval.baseline_runner import _compute_exact_match

        assert _compute_exact_match("Alice Smith", "Alice Smith") == 1.0

    def test_compute_exact_match_case_sensitive(self):
        from route_data.eval.baseline_runner import _compute_exact_match

        assert _compute_exact_match("alice smith", "Alice Smith") == 0.0

    def test_compute_normalized_exact_match_equal(self):
        from route_data.eval.baseline_runner import _compute_normalized_exact_match

        assert _compute_normalized_exact_match("Alice Smith", "Alice Smith") == 1.0

    def test_compute_normalized_exact_match_case_insensitive(self):
        from route_data.eval.baseline_runner import _compute_normalized_exact_match

        assert _compute_normalized_exact_match("alice smith", "Alice Smith") == 1.0

    def test_compute_normalized_exact_match_punctuation(self):
        from route_data.eval.baseline_runner import _compute_normalized_exact_match

        assert _compute_normalized_exact_match("Alice Smith!", "Alice Smith") == 1.0

    def test_compute_normalized_exact_match_whitespace(self):
        from route_data.eval.baseline_runner import _compute_normalized_exact_match

        assert _compute_normalized_exact_match("  Alice   Smith  ", "Alice Smith") == 1.0

    def test_compute_normalized_exact_match_mismatch(self):
        from route_data.eval.baseline_runner import _compute_normalized_exact_match

        assert _compute_normalized_exact_match("Bob Jones", "Alice Smith") == 0.0

    def test_compute_token_overlap_full(self):
        from route_data.eval.baseline_runner import _compute_token_overlap

        assert _compute_token_overlap("Alice Smith", "Alice Smith") == pytest.approx(1.0)

    def test_compute_token_overlap_partial(self):
        from route_data.eval.baseline_runner import _compute_token_overlap

        score = _compute_token_overlap("Alice Jones", "Alice Smith")
        # tokens: {alice, jones} vs {alice, smith} → overlap={alice}
        # precision=1/2, recall=1/2 → F1=0.5
        assert score == pytest.approx(0.5)

    def test_compute_token_overlap_no_match(self):
        from route_data.eval.baseline_runner import _compute_token_overlap

        assert _compute_token_overlap("Bob Jones", "Alice Smith") == 0.0

    def test_compute_token_overlap_empty(self):
        from route_data.eval.baseline_runner import _compute_token_overlap

        assert _compute_token_overlap("", "Alice Smith") == 0.0


class TestSignedMarginSemantics:
    """Verify signed_answer_margin semantics (Commit A)."""

    def test_positive_label_positive_margin(self, runner: BaselineRunner):
        """Positive label + model prefers Yes → positive signed margin."""
        probe = BaselineProbe(
            probe_id="sm_pos",
            sample_id="s1",
            identity_id="aaaa",
            benchmark="fiubench",
            probe_family="direct_visual",
            modality="image_only",
            question="Is the person smiling?",
            expected_evidence_source="visual",
            controlled_variables=["image"],
            image_uri="/fake.jpg",
            image_sha256="abc",
            registry_hash="rh",
            target_attribute="Smiling",
            answer_label=True,
            answer_text="yes",
        )
        result = runner.run_probe(probe)
        if result.raw_log_margin is not None:
            # positive label → signed == raw
            assert result.signed_answer_margin == pytest.approx(result.raw_log_margin)

    def test_negative_label_sign_flip(self, runner: BaselineRunner):
        """Negative label → signed margin is negated raw margin."""
        probe = BaselineProbe(
            probe_id="sm_neg",
            sample_id="s1",
            identity_id="aaaa",
            benchmark="fiubench",
            probe_family="direct_visual",
            modality="image_only",
            question="Is the person bald?",
            expected_evidence_source="visual",
            controlled_variables=["image"],
            image_uri="/fake.jpg",
            image_sha256="abc",
            registry_hash="rh",
            target_attribute="Bald",
            answer_label=False,
            answer_text="no",
        )
        result = runner.run_probe(probe)
        if result.raw_log_margin is not None:
            assert result.signed_answer_margin == pytest.approx(-result.raw_log_margin)

    def test_centered_p_yes_and_raw_margin(self, runner: BaselineRunner):
        """centered_p_yes = p_yes - 0.5; raw_log_margin = logp_yes - logp_no."""
        probe = BaselineProbe(
            probe_id="sm_ctr",
            sample_id="s1",
            identity_id="aaaa",
            benchmark="fiubench",
            probe_family="direct_visual",
            modality="image_only",
            question="Is the person wearing a hat?",
            expected_evidence_source="visual",
            controlled_variables=["image"],
            image_uri="/fake.jpg",
            image_sha256="abc",
            registry_hash="rh",
            target_attribute="Hat",
            answer_label=True,
            answer_text="yes",
        )
        result = runner.run_probe(probe)
        if result.p_yes is not None:
            assert result.centered_p_yes == pytest.approx(result.p_yes - 0.5)
            assert result.raw_log_margin == pytest.approx(
                result.logp_yes - result.logp_no
            )


class TestMetadataCarryThrough:
    """Verify all probe metadata fields are carried through to BaselineResult."""

    def test_all_probe_fields_carried(self, runner: BaselineRunner):
        probe = BaselineProbe(
            probe_id="meta1",
            sample_id="s1",
            identity_id="aaaa",
            benchmark="fiubench",
            probe_family="wrong_name",
            modality="image_text",
            question="Is the person Bob?",
            expected_evidence_source="visual",
            controlled_variables=["image"],
            image_uri="/fake.jpg",
            image_sha256="sha123",
            registry_hash="reg_hash",
            target_attribute="Bald",
            answer_label=False,
            answer_text="no",
            matched_wrong_identity_id="wrong_id",
            matching_similarity=0.85,
            matching_attributes=["Bald", "Male"],
            candidate_rank=2,
            matching_strategy="jaccard",
            paired_sample_id="pair1",
        )
        result = runner.run_probe(probe)
        assert result.probe_id == "meta1"
        assert result.identity_id == "aaaa"
        assert result.probe_family == "wrong_name"
        assert result.registry_hash == "reg_hash"
        assert result.paired_sample_id == "pair1"
        assert result.target_attribute == "Bald"
        assert result.answer_label is False
        assert result.matched_wrong_identity_id == "wrong_id"
        assert result.matching_similarity == pytest.approx(0.85)
        assert result.matching_attributes == ["Bald", "Male"]
        assert result.candidate_rank == 2
        assert result.matching_strategy == "jaccard"
        assert result.image_sha256 == "sha123"
        assert result.model_fingerprint != ""
        assert result.scoring_version != ""
        assert result.prompt_hash != ""
        assert result.latency_ms >= 0


class TestModelConfigSha:
    """Verify model-config SHA computation (Commit B)."""

    def test_model_config_sha_computed(self, runner: BaselineRunner):
        """Even without a YAML file, the runner computes a deterministic SHA."""
        assert runner._model_config_sha
        assert len(runner._model_config_sha) == 64  # SHA-256 hex

    def test_model_config_sha_deterministic(
        self, stub_backend, probe_jsonl, tmp_path, stub_model_config
    ):
        r1 = BaselineRunner(
            stub_backend, probe_jsonl, tmp_path / "o1", stub_model_config, resume=False
        )
        r2 = BaselineRunner(
            stub_backend, probe_jsonl, tmp_path / "o2", stub_model_config, resume=False
        )
        assert r1._model_config_sha == r2._model_config_sha

    def test_model_config_sha_from_yaml(
        self, stub_backend, probe_jsonl, tmp_path, stub_model_config
    ):
        """When a YAML path is given, the SHA is of the file content."""
        yaml_path = tmp_path / "model.yaml"
        yaml_path.write_text("model:\n  backend: stub\n")
        runner = BaselineRunner(
            stub_backend,
            probe_jsonl,
            tmp_path / "out",
            stub_model_config,
            resume=False,
            model_config_path=yaml_path,
        )
        from route_data.eval.baseline_runner import _sha256_file

        expected = _sha256_file(yaml_path)
        assert runner._model_config_sha == expected


class TestManifestDrivenVerification:
    """Verify manifest-driven SHA verification (Commit B)."""

    def test_no_manifest_raises(self, runner_no_map: BaselineRunner):
        with pytest.raises(RuntimeError, match="No dataset manifest"):
            runner_no_map.verify_input_hashes_from_manifest()

    def test_manifest_missing_field_raises(
        self, stub_backend, probe_jsonl, tmp_path, stub_model_config
    ):
        bad_manifest = tmp_path / "bad_manifest.json"
        bad_manifest.write_text(json.dumps({"unrelated": True}))
        runner = BaselineRunner(
            stub_backend,
            probe_jsonl,
            tmp_path / "out",
            stub_model_config,
            resume=False,
            dataset_manifest_path=bad_manifest,
        )
        with pytest.raises(RuntimeError, match="does not contain"):
            runner.verify_input_hashes_from_manifest()

    def test_manifest_verification_passes(
        self, stub_backend, probe_jsonl, tmp_path, stub_model_config
    ):
        from route_data.eval.baseline_runner import _sha256_file

        actual_sha = _sha256_file(probe_jsonl)
        manifest = {
            "dataset_artifacts": {
                "route_probes": {"sha256": actual_sha}
            }
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        runner = BaselineRunner(
            stub_backend,
            probe_jsonl,
            tmp_path / "out",
            stub_model_config,
            resume=False,
            dataset_manifest_path=manifest_path,
        )
        assert runner.verify_input_hashes_from_manifest() is True

    def test_manifest_verification_fails_on_mismatch(
        self, stub_backend, probe_jsonl, tmp_path, stub_model_config
    ):
        manifest = {
            "dataset_artifacts": {
                "route_probes": {"sha256": "0" * 64}
            }
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        runner = BaselineRunner(
            stub_backend,
            probe_jsonl,
            tmp_path / "out",
            stub_model_config,
            resume=False,
            dataset_manifest_path=manifest_path,
        )
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            runner.verify_input_hashes_from_manifest()


class TestStrongerCacheKey:
    """Verify the cache key includes model-config provenance (Commit B)."""

    def test_cache_key_includes_model_config_sha(self, runner: BaselineRunner):
        """The cache key should be a valid SHA-256 hex digest."""
        probe = runner.probes[0]
        key = runner._cache_key(probe)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_different_config_different_key(
        self, stub_backend, probe_jsonl, tmp_path
    ):
        from route_data.models.registry import create_backend

        cfg1 = ModelConfig(backend="stub", model_id="model-a", revision="r1")
        cfg2 = ModelConfig(backend="stub", model_id="model-b", revision="r1")
        b1 = create_backend(cfg1)
        b2 = create_backend(cfg2)
        r1 = BaselineRunner(b1, probe_jsonl, tmp_path / "o1", cfg1, resume=False)
        r2 = BaselineRunner(b2, probe_jsonl, tmp_path / "o2", cfg2, resume=False)
        probe = r1.probes[0]
        assert r1._cache_key(probe) != r2._cache_key(probe)


class TestStrongerFingerprint:
    """Verify the stub fingerprint includes richer metadata (Commit B)."""

    def test_fingerprint_has_dtype(self, stub_backend):
        fp = stub_backend.fingerprint()
        assert "dtype" in fp
        assert fp["dtype"] != ""

    def test_fingerprint_has_quantization(self, stub_backend):
        fp = stub_backend.fingerprint()
        assert "quantization" in fp

    def test_fingerprint_has_attn_implementation(self, stub_backend):
        fp = stub_backend.fingerprint()
        assert "attn_implementation" in fp

    def test_fingerprint_has_processor_id(self, stub_backend):
        fp = stub_backend.fingerprint()
        assert "processor_id" in fp

    def test_fingerprint_id_deterministic(self, stub_model_config):
        from route_data.models.registry import create_backend

        b1 = create_backend(stub_model_config)
        b2 = create_backend(stub_model_config)
        assert b1.fingerprint()["fingerprint_id"] == b2.fingerprint()["fingerprint_id"]

    def test_different_dtype_different_fingerprint(self, stub_model_config):
        from route_data.models.registry import create_backend

        cfg2 = ModelConfig(
            backend="stub", model_id="test-model",
            revision="abc123", dtype="float32",
        )
        b1 = create_backend(stub_model_config)
        b2 = create_backend(cfg2)
        assert b1.fingerprint()["fingerprint_id"] != b2.fingerprint()["fingerprint_id"]


class TestValidateResults:
    """Verify the P0-4 strict baseline validator (8 named checks)."""

    _ALL_CHECK_NAMES: ClassVar[list[str]] = [
        "exact_probe_id_set",
        "family_counts_match",
        "binary_scores_complete",
        "name_only_scores_complete",
        "source_metadata_match",
        "run_provenance_consistent",
        "zero_inference_errors",
        "protocol_role_complete",
        "processed_dataset_sha_match",
        "route_identity_role_counts",
    ]

    def test_validate_passes_with_complete_results(self, runner: BaselineRunner):
        runner.run_all()
        report = runner.validate_results()
        assert report["pass"] is True
        for name in self._ALL_CHECK_NAMES:
            assert report["checks"][name]["pass"] is True, (
                f"check {name!r} should pass"
            )

    def test_validate_writes_report(self, runner: BaselineRunner):
        runner.run_all()
        runner.validate_results()
        report_path = runner.output_dir / "validation_report.json"
        assert report_path.is_file()
        with open(report_path) as f:
            data = json.load(f)
        assert data["pass"] is True
        assert set(data["checks"].keys()) == set(self._ALL_CHECK_NAMES)

    def test_validate_fails_on_missing_results(self, runner: BaselineRunner):
        with pytest.raises(RuntimeError, match="exact_probe_id_set"):
            runner.validate_results()

    def test_validate_fails_on_partial_results(self, runner: BaselineRunner):
        """Running only a subset of probes fails the ID-set check."""
        runner.run_all(limit=3)
        with pytest.raises(RuntimeError, match="exact_probe_id_set"):
            runner.validate_results()

    def test_validate_fails_on_duplicate_probe_id(self, runner: BaselineRunner):
        """Injecting a duplicate result fails the no-duplicates check."""
        runner.run_all()
        dup = runner._results[0]
        runner._results.append(dup)
        with pytest.raises(RuntimeError, match="exact_probe_id_set"):
            runner.validate_results()

    def test_validate_fails_on_metadata_corruption(self, runner: BaselineRunner):
        """Corrupting a result's sample_id fails source_metadata_match."""
        runner.run_all()
        runner._results[0].sample_id = "CORRUPTED"
        with pytest.raises(RuntimeError, match="source_metadata_match"):
            runner.validate_results()

    def test_validate_fails_on_error_row(self, runner: BaselineRunner):
        """A result with an error fails zero_inference_errors."""
        runner.run_all()
        runner._results[0].error = "simulated failure"
        with pytest.raises(RuntimeError, match="zero_inference_errors"):
            runner.validate_results()

    def test_validate_binary_scores_complete(self, runner: BaselineRunner):
        runner.run_all()
        report = runner.validate_results()
        assert report["checks"]["binary_scores_complete"]["pass"] is True
        assert report["checks"]["binary_scores_complete"]["failure_count"] == 0

    def test_validate_name_only_scores_complete(self, runner: BaselineRunner):
        runner.run_all()
        report = runner.validate_results()
        assert report["checks"]["name_only_scores_complete"]["pass"] is True
        assert report["checks"]["name_only_scores_complete"]["failure_count"] == 0

    def test_validate_family_counts_exact(self, runner: BaselineRunner):
        """Family counts match the source probes exactly (2 per family)."""
        runner.run_all()
        report = runner.validate_results()
        expected = {fam: 2 for fam in ALL_FAMILIES}
        assert report["checks"]["family_counts_match"]["expected"] == expected
        assert report["checks"]["family_counts_match"]["actual"] == expected

    def test_validate_provenance_consistent(self, runner: BaselineRunner):
        runner.run_all()
        report = runner.validate_results()
        assert report["checks"]["run_provenance_consistent"]["pass"] is True
        assert report["checks"]["run_provenance_consistent"]["issues"] == {}

    def test_validate_fails_on_provenance_mismatch(self, runner: BaselineRunner):
        """Changing one result's model_revision breaks consistency."""
        runner.run_all()
        runner._results[0].model_revision = "DIFFERENT_REVISION"
        with pytest.raises(RuntimeError, match="run_provenance_consistent"):
            runner.validate_results()

    def test_validate_binary_fails_on_null_logp(self, runner: BaselineRunner):
        """A binary probe with null logp_yes fails binary_scores_complete."""
        runner.run_all()
        # Find a binary-family result and null out logp_yes
        for r in runner._results:
            if r.probe_family in {"direct_visual", "image_plus_name",
                                  "wrong_name", "visual_text_conflict"}:
                r.logp_yes = None
                break
        with pytest.raises(RuntimeError, match="binary_scores_complete"):
            runner.validate_results()

    def test_validate_name_only_fails_on_empty_answer(self, runner: BaselineRunner):
        """A name_only probe with empty generated_answer fails."""
        runner.run_all()
        for r in runner._results:
            if r.probe_family == "name_only":
                r.generated_answer = ""
                break
        with pytest.raises(RuntimeError, match="name_only_scores_complete"):
            runner.validate_results()


class TestBaselineManifest:
    """Verify baseline_manifest.json generation (Commit C)."""

    @pytest.fixture(autouse=True)
    def _mock_clean_git(self):
        """Patch _get_git_state to simulate a clean Git tree."""
        with patch.object(
            BaselineRunner,
            "_get_git_state",
            staticmethod(lambda: {
                "git_commit": "a" * 40,
                "git_dirty": False,
            }),
        ):
            yield

    def test_manifest_written(self, runner: BaselineRunner):
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        runner.generate_baseline_manifest()
        manifest_path = runner.output_dir / "baseline_manifest.json"
        assert manifest_path.is_file()

    def test_manifest_has_dataset_provenance(self, runner: BaselineRunner):
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
        assert "dataset_provenance" in manifest
        dp = manifest["dataset_provenance"]
        assert "probe_file_sha256" in dp
        assert dp["probe_count"] == 10

    def test_manifest_has_model_identity(self, runner: BaselineRunner):
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
        assert "model_identity" in manifest
        mi = manifest["model_identity"]
        assert "fingerprint_id" in mi
        assert "model_config_sha256" in mi

    def test_manifest_has_results_sha(self, runner: BaselineRunner):
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
        assert "results" in manifest
        assert manifest["results"]["results_sha256"] != ""
        assert len(manifest["results"]["results_sha256"]) == 64

    def test_manifest_has_runtime_environment(self, runner: BaselineRunner):
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
        assert "runtime_environment" in manifest
        assert "python_version" in manifest["runtime_environment"]

    def test_manifest_has_scoring_config(self, runner: BaselineRunner):
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
        assert "scoring_config" in manifest
        assert manifest["scoring_config"]["scoring_version"] != ""


class TestSmokeSelector:
    """Verify the deterministic smoke probe selector (Commit D / P0-3)."""

    def test_selects_all_families(self, runner: BaselineRunner):
        selected = select_smoke_probes(runner.probes)
        families = {p.probe_family for p in selected}
        assert families == ALL_FAMILIES

    def test_returns_exactly_10_probes(self, runner: BaselineRunner):
        selected = select_smoke_probes(runner.probes, n_identities=2)
        assert len(selected) == 10

    def test_returns_exactly_2_identities(self, runner: BaselineRunner):
        selected = select_smoke_probes(runner.probes, n_identities=2)
        identities = {p.identity_id for p in selected}
        assert len(identities) == 2
        assert identities == {"aaaa", "bbbb"}

    def test_two_probes_per_family(self, runner: BaselineRunner):
        from collections import Counter

        selected = select_smoke_probes(runner.probes, n_identities=2)
        family_counts = Counter(p.probe_family for p in selected)
        for fam in ALL_FAMILIES:
            assert family_counts[fam] == 2, f"family {fam} has {family_counts[fam]} probes"

    def test_each_identity_has_all_families(self, runner: BaselineRunner):
        selected = select_smoke_probes(runner.probes, n_identities=2)
        by_id: dict[str, set[str]] = {}
        for p in selected:
            by_id.setdefault(p.identity_id, set()).add(p.probe_family)
        for iid, fams in by_id.items():
            assert fams == ALL_FAMILIES, (
                f"identity {iid} missing families: {ALL_FAMILIES - fams}"
            )

    def test_deterministic(self, runner: BaselineRunner):
        s1 = select_smoke_probes(runner.probes)
        s2 = select_smoke_probes(runner.probes)
        assert [p.probe_id for p in s1] == [p.probe_id for p in s2]

    def test_respects_n_identities(self, runner: BaselineRunner):
        selected = select_smoke_probes(runner.probes, n_identities=1)
        identities = {p.identity_id for p in selected}
        assert len(identities) == 1
        assert len(selected) == 5

    def test_no_duplicate_probe_ids(self, runner: BaselineRunner):
        selected = select_smoke_probes(runner.probes)
        ids = [p.probe_id for p in selected]
        assert len(ids) == len(set(ids))

    def test_write_smoke_manifest(self, runner: BaselineRunner, tmp_path: Path):
        selected = select_smoke_probes(runner.probes)
        manifest_path = write_smoke_manifest(
            selected, tmp_path / "smoke_manifest.json"
        )
        assert manifest_path.is_file()
        with open(manifest_path) as f:
            data = json.load(f)
        assert data["probe_count"] == len(selected)
        assert data["identity_count"] == 2
        assert set(data["family_counts"].keys()) == ALL_FAMILIES
        assert set(data["selected_identity_ids"]) == {"aaaa", "bbbb"}
        assert set(data["selected_probe_ids"]) == {p.probe_id for p in selected}
        assert data["route_probe_sha256"] == ""
        assert data["dataset_version"] == "fiubench-route-v1"

    def test_write_smoke_manifest_with_sha(
        self, runner: BaselineRunner, tmp_path: Path
    ):
        selected = select_smoke_probes(runner.probes)
        manifest_path = write_smoke_manifest(
            selected, tmp_path / "smoke.json", probe_file_sha256="abc123"
        )
        with open(manifest_path) as f:
            data = json.load(f)
        assert data["route_probe_sha256"] == "abc123"

    def test_insufficient_identities_raises(
        self, stub_backend, tmp_path, stub_model_config
    ):
        """Fewer than n_identities eligible identities raises ValueError."""
        probes = [BaselineProbe.from_dict(d) for d in _PROBES_RAW[:5]]
        with pytest.raises(ValueError, match="Need 2 eligible"):
            select_smoke_probes(probes, n_identities=2)


class TestSpyCallPaths:
    """Verify the runner calls backend methods correctly (Commit D)."""

    def test_image_probe_calls_score_candidates(self, runner: BaselineRunner):
        """Image-bearing probes should call score_candidates."""
        from unittest.mock import MagicMock, patch

        probe = BaselineProbe(
            probe_id="spy_img",
            sample_id="s1",
            identity_id="aaaa",
            benchmark="fiubench",
            probe_family="direct_visual",
            modality="image_only",
            question="Is the person bald?",
            expected_evidence_source="visual",
            controlled_variables=["image"],
            image_uri="/fake.jpg",
            image_sha256="abc",
            registry_hash="rh",
            target_attribute="Bald",
            answer_label=False,
            answer_text="no",
        )
        fake_image = MagicMock()
        with patch("route_data.eval.baseline_runner._load_image", return_value=fake_image), \
             patch.object(runner.backend, "score_candidates", wraps=runner.backend.score_candidates) as mock:
            runner.run_probe(probe)
            mock.assert_called_once()

    def test_name_only_probe_calls_generate(self, runner: BaselineRunner):
        """Name-only probes should call generate, not score_candidates."""
        from unittest.mock import patch

        probe = BaselineProbe(
            probe_id="spy_txt",
            sample_id="s1",
            identity_id="aaaa",
            benchmark="fiubench",
            probe_family="name_only",
            modality="text_only",
            question="What is the person's name?",
            expected_evidence_source="identity_fact",
            controlled_variables=["image"],
            image_uri=None,
            image_sha256=None,
            registry_hash="rh",
            target_attribute=None,
            answer_label=None,
            answer_text="Alice Smith",
            target_fact_id="q1",
            target_fact_relation="name",
            target_fact_value="Alice Smith",
        )
        with patch.object(runner.backend, "generate", wraps=runner.backend.generate) as mock_gen, \
             patch.object(runner.backend, "score_candidates") as mock_score:
            runner.run_probe(probe)
            mock_gen.assert_called_once()
            mock_score.assert_not_called()


class TestCommit5MetadataEnrichment:
    """Verify Commit 5 metadata enrichment (P1-1 through P1-6)."""

    @pytest.fixture(autouse=True)
    def _mock_clean_git(self):
        """Patch _get_git_state to simulate a clean Git tree."""
        with patch.object(
            BaselineRunner,
            "_get_git_state",
            staticmethod(lambda: {
                "git_commit": "a" * 40,
                "git_dirty": False,
            }),
        ):
            yield

    def test_protocol_role_field_exists(self):
        """P1-2: BaselineResult has protocol_role field."""
        result = BaselineResult(probe_id="test", sample_id="s", identity_id="i",
                                probe_family="name_only", modality="text_only",
                                question="q")
        assert hasattr(result, "protocol_role")
        assert result.protocol_role == ""

    def test_mixed_task_overall_accuracy_in_summary(self, runner: BaselineRunner):
        """P1-4: Summary uses mixed_task_overall_accuracy instead of overall_accuracy."""
        runner.run_all()
        summary = runner.generate_summary()
        assert "mixed_task_overall_accuracy" in summary
        assert "overall_accuracy" not in summary

    def test_enriched_manifest_has_dataset_version(self, runner: BaselineRunner):
        """P1-3: Manifest includes dataset_version and dataset_manifest_sha256."""
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
        dp = manifest["dataset_provenance"]
        assert "dataset_version" in dp
        assert "dataset_manifest_sha256" in dp

    def test_enriched_manifest_has_output_shas(self, runner: BaselineRunner):
        """P1-3: Manifest includes SHA-256 of all output artifacts."""
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        runner.validate_results()
        manifest = runner.generate_baseline_manifest()
        results = manifest["results"]
        assert "summary_sha256" in results
        assert "validation_report_sha256" in results
        assert "smoke_manifest_sha256" in results

    def test_enriched_manifest_has_library_versions(self, runner: BaselineRunner):
        """P1-3: Manifest includes torch/transformers/accelerate versions."""
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
        runtime = manifest["runtime_environment"]
        assert "torch_version" in runtime
        assert "transformers_version" in runtime
        assert "accelerate_version" in runtime

    def test_enriched_manifest_has_fingerprint_payload(self, runner: BaselineRunner):
        """P1-3: Manifest includes full fingerprint payload."""
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
        mi = manifest["model_identity"]
        assert "fingerprint_payload" in mi
        assert isinstance(mi["fingerprint_payload"], dict)

    def test_enriched_manifest_has_scoring_provenance(self, runner: BaselineRunner):
        """P1-3: Manifest includes scoring provenance details."""
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
        sc = manifest["scoring_config"]
        assert "candidate_protocol" in sc
        assert "candidate_protocol_version" in sc
        assert "decision_rule" in sc

    def test_validate_dataset_manifest_requires_manifest(self, runner_no_map: BaselineRunner):
        """P0-1: validate_dataset_manifest raises without manifest."""
        with pytest.raises(RuntimeError, match="requires --dataset-manifest"):
            runner_no_map.validate_dataset_manifest()

    def test_validate_dataset_manifest_with_valid_manifest(
        self, stub_backend, tmp_path, stub_model_config
    ):
        """P0-1: validate_dataset_manifest passes with actual frozen schema."""
        import json
        probe_file = tmp_path / "probes.jsonl"
        probe_file.write_text("\n".join(json.dumps(p) for p in _PROBES_RAW))
        from route_data.eval.baseline_runner import _sha256_file
        sha = _sha256_file(probe_file)
        # Use the *actual* frozen schema paths.
        manifest_data = {
            "definition_of_done": {
                "ready_for_experiments": True,
            },
            "dataset_artifacts": {
                "route_probes": {
                    "sha256": sha,
                    "total_probes": 500,
                    "families": {
                        "direct_visual": 100,
                        "image_plus_name": 100,
                        "wrong_name": 100,
                        "visual_text_conflict": 100,
                        "name_only": 100,
                    },
                }
            },
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest_data))
        # Freeze verification file provides dataset_version.
        freeze_data = {
            "dataset_version": "fiubench-route-v1",
            "ready_for_experiments": True,
        }
        freeze_file = tmp_path / "freeze_verification.json"
        freeze_file.write_text(json.dumps(freeze_data))
        runner = BaselineRunner(
            backend=stub_backend,
            probe_path=str(probe_file),
            output_dir=tmp_path / "out",
            model_config=stub_model_config,
            dataset_manifest_path=str(manifest_file),
            freeze_verification_path=str(freeze_file),
        )
        # Mock _read_jsonl to return 500 rows for the JSONL row-count check.
        with patch(
            "route_data.eval.baseline_runner._read_jsonl",
            return_value=[{}] * 500,
        ):
            checks = runner.validate_dataset_manifest()
        assert checks["ready_for_experiments"] is True
        assert checks["dataset_version"] == "fiubench-route-v1"
        assert checks["route_probe_count"] == 500

    def test_validate_dataset_manifest_fails_on_wrong_version(
        self, stub_backend, tmp_path, stub_model_config
    ):
        """P0-1: validate_dataset_manifest fails on wrong dataset_version."""
        import json
        probe_file = tmp_path / "probes.jsonl"
        probe_file.write_text("\n".join(json.dumps(p) for p in _PROBES_RAW))
        from route_data.eval.baseline_runner import _sha256_file
        sha = _sha256_file(probe_file)
        manifest_data = {
            "definition_of_done": {
                "ready_for_experiments": True,
            },
            "dataset_artifacts": {
                "route_probes": {
                    "sha256": sha,
                    "total_probes": 500,
                    "families": {
                        "direct_visual": 100,
                        "image_plus_name": 100,
                        "wrong_name": 100,
                        "visual_text_conflict": 100,
                        "name_only": 100,
                    },
                }
            },
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest_data))
        # Wrong dataset_version in freeze verification.
        freeze_data = {
            "dataset_version": "wrong-version",
            "ready_for_experiments": True,
        }
        freeze_file = tmp_path / "freeze_verification.json"
        freeze_file.write_text(json.dumps(freeze_data))
        runner = BaselineRunner(
            backend=stub_backend,
            probe_path=str(probe_file),
            output_dir=tmp_path / "out",
            model_config=stub_model_config,
            dataset_manifest_path=str(manifest_file),
            freeze_verification_path=str(freeze_file),
        )
        with patch(
            "route_data.eval.baseline_runner._read_jsonl",
            return_value=[{}] * 500,
        ), pytest.raises(RuntimeError, match="validation failed"):
            runner.validate_dataset_manifest()


class TestFrozenEvidenceIntegration:
    """Integration tests using the actual committed freeze files (P0-4).

    These tests load the real frozen evidence bundles from
    ``outputs/full_fiubench/evidence/`` and verify that field paths,
    cross-file hash relations, and schema locations match what the
    :class:`BaselineRunner` expects.
    """

    EVIDENCE_DIR: Path = (
        Path(__file__).resolve().parent.parent.parent
        / "outputs"
        / "full_fiubench"
        / "evidence"
    )
    MANIFEST_PATH: Path = EVIDENCE_DIR / "research_dataset_manifest.json"
    FREEZE_PATH: Path = EVIDENCE_DIR / "final_freeze_verification.json"

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        """Skip when the frozen evidence files are not present."""
        if not self.MANIFEST_PATH.is_file() or not self.FREEZE_PATH.is_file():
            pytest.skip("Frozen evidence files not available")

    # -- helpers ---------------------------------------------------------- #

    @staticmethod
    def _load_json(path: Path) -> dict:
        return json.loads(path.read_text())

    # -- tests ------------------------------------------------------------ #

    def test_actual_field_paths(self):
        """Verify the actual field paths in committed freeze files."""
        manifest = self._load_json(self.MANIFEST_PATH)
        freeze = self._load_json(self.FREEZE_PATH)

        # Manifest field paths
        assert manifest["definition_of_done"]["ready_for_experiments"] is True
        rp = manifest["dataset_artifacts"]["route_probes"]
        assert rp["total_probes"] == 500
        assert "sha256" in rp
        assert "families" in rp

        # Freeze verification field paths
        assert freeze["dataset_version"] == "fiubench-route-v1"
        assert freeze["ready_for_experiments"] is True
        assert "dataset_manifest_sha256" in freeze
        assert "route_probe_sha256" in freeze

    def test_manifest_sha_relation(self):
        """Freeze ``dataset_manifest_sha256`` matches actual manifest SHA."""
        from route_data.eval.baseline_runner import _sha256_file

        freeze = self._load_json(self.FREEZE_PATH)
        expected_sha = freeze["dataset_manifest_sha256"]
        actual_sha = _sha256_file(self.MANIFEST_PATH)
        assert actual_sha == expected_sha

    def test_dataset_version_location(self):
        """``dataset_version`` lives in freeze verification, not manifest."""
        manifest = self._load_json(self.MANIFEST_PATH)
        freeze = self._load_json(self.FREEZE_PATH)

        # Freeze verification has dataset_version
        assert freeze["dataset_version"] == "fiubench-route-v1"

        # Manifest does NOT have a top-level dataset_version
        assert "dataset_version" not in manifest

    def test_ready_for_experiments_location(self):
        """``ready_for_experiments`` in ``definition_of_done`` and freeze."""
        manifest = self._load_json(self.MANIFEST_PATH)
        freeze = self._load_json(self.FREEZE_PATH)

        # In manifest: nested under definition_of_done
        assert manifest["definition_of_done"]["ready_for_experiments"] is True

        # In freeze: top-level
        assert freeze["ready_for_experiments"] is True

    def test_route_total_probes_location(self):
        """Route count is at ``dataset_artifacts.route_probes.total_probes``."""
        manifest = self._load_json(self.MANIFEST_PATH)
        total = manifest["dataset_artifacts"]["route_probes"]["total_probes"]
        assert total == 500

    def test_route_sha_cross_reference(self):
        """Route SHA agrees across freeze verification and manifest."""
        manifest = self._load_json(self.MANIFEST_PATH)
        freeze = self._load_json(self.FREEZE_PATH)

        manifest_sha = manifest["dataset_artifacts"]["route_probes"]["sha256"]
        freeze_sha = freeze["route_probe_sha256"]
        assert manifest_sha == freeze_sha
        assert len(manifest_sha) == 64  # SHA-256 hex digest length


class TestProtocolRolePopulation:
    """Verify protocol-role population (Commit B: P0-5 to P0-9).

    Tests the identity-to-protocol-role mapping pipeline:
    processed JSONL + manifest protocol config → identity_role_map →
    BaselineResult.protocol_role → validation → summary counts.
    """

    # Protocol config matching the real frozen manifest.
    _PROTO: ClassVar[dict] = {
        "forget_bucket": "forget10",
        "train_bucket": "retain15",
        "eval_bucket": None,
        "eval_fraction": 0.2,
        "eval_seed": 17,
    }

    @staticmethod
    def _find_holdout_id(target_role: str, seed: int = 17) -> str:
        """Find a subject ID that hashes to *target_role* in holdout."""
        import hashlib as _hl

        for i in range(10000):
            sid = f"SUBJ_{i:05d}"
            h = _hl.sha256(f"{seed}|{sid}".encode()).digest()
            x = int.from_bytes(h[:8], "big") / (2**64)
            role = "eval" if x < 0.2 else "train"
            if role == target_role:
                return sid
        raise RuntimeError(f"Could not find holdout ID for {target_role}")

    # -- fixtures --------------------------------------------------------- #

    @pytest.fixture()
    def processed_jsonl(self, tmp_path: Path) -> Path:
        """Synthetic processed JSONL with known role assignments."""
        train_id = self._find_holdout_id("train")
        path = tmp_path / "processed.jsonl"
        rows = [
            {"identity_id": "id_a", "source_metadata": {
                "source_subject_id": "S001",
                "official_memberships": ["forget10"],
            }},
            {"identity_id": "id_b", "source_metadata": {
                "source_subject_id": train_id,
                "official_memberships": ["retain15"],
            }},
            {"identity_id": "id_c", "source_metadata": {
                "source_subject_id": "S003",
                "official_memberships": ["other_bucket"],
            }},
        ]
        with open(path, "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in rows)
        return path

    @pytest.fixture()
    def manifest_jsonl(self, tmp_path: Path, processed_jsonl: Path) -> Path:
        """Synthetic manifest with protocol config and probe SHA."""
        from route_data.eval.baseline_runner import _sha256_file

        probe_file = tmp_path / "test_probes.jsonl"
        probe_file.write_text("\n".join(json.dumps(p) for p in _PROBES_RAW))
        path = tmp_path / "manifest.json"
        data = {
            "protocol": {"canonical_protocol": dict(self._PROTO)},
            "dataset_artifacts": {
                "route_probes": {"sha256": _sha256_file(probe_file)},
            },
        }
        path.write_text(json.dumps(data))
        return path

    @pytest.fixture()
    def runner_with_roles(
        self,
        stub_backend,
        probe_jsonl: Path,
        tmp_path: Path,
        stub_model_config: ModelConfig,
        processed_jsonl: Path,
        manifest_jsonl: Path,
    ) -> BaselineRunner:
        """Runner with identity_role_map populated.

        Note: the probe identities (aaaa, bbbb, …) do NOT overlap with
        the processed JSONL identities (id_a, id_b, id_c).  This is
        intentional — tests that call ``run_probe`` on the loaded probes
        will trigger the missing-identity hard-fail, while tests that
        inspect the map directly work correctly.
        """
        return BaselineRunner(
            backend=stub_backend,
            probe_path=probe_jsonl,
            output_dir=tmp_path / "output",
            model_config=stub_model_config,
            resume=False,
            dataset_manifest_path=manifest_jsonl,
            processed_dataset_path=processed_jsonl,
        )

    @pytest.fixture()
    def runner_full_overlap(
        self,
        stub_backend,
        tmp_path: Path,
        stub_model_config: ModelConfig,
    ) -> BaselineRunner:
        """Runner where probe identities match processed JSONL identities.

        Probes use identity_ids ``id_a``, ``id_b``, ``id_c`` matching the
        synthetic processed JSONL, so ``run_probe`` succeeds.
        """
        from route_data.eval.baseline_runner import _sha256_file

        # Build probes that use the same identity_ids as processed JSONL.
        probes = []
        for idx, iid in enumerate(["id_a", "id_b", "id_c"]):
            for fam in ["direct_visual", "name_only"]:
                base = dict(_PROBES_RAW[0] if fam == "direct_visual"
                            else _PROBES_RAW[4])
                base = dict(base)
                base["probe_id"] = f"p_{iid}_{fam}"
                base["identity_id"] = iid
                base["probe_family"] = fam
                base["sample_id"] = f"s_{iid}_{fam}"
                probes.append(base)
        probe_file = tmp_path / "probes.jsonl"
        probe_file.write_text("\n".join(json.dumps(p) for p in probes))

        # Processed JSONL (shared across fixtures via tmp_path).
        train_id = self._find_holdout_id("train")
        processed_file = tmp_path / "processed.jsonl"
        rows = [
            {"identity_id": "id_a", "source_metadata": {
                "source_subject_id": "S001",
                "official_memberships": ["forget10"],
            }},
            {"identity_id": "id_b", "source_metadata": {
                "source_subject_id": train_id,
                "official_memberships": ["retain15"],
            }},
            {"identity_id": "id_c", "source_metadata": {
                "source_subject_id": "S003",
                "official_memberships": ["other_bucket"],
            }},
        ]
        processed_file.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )

        # Manifest with correct probe SHA and processed dataset SHA.
        manifest_data = {
            "protocol": {"canonical_protocol": dict(self._PROTO)},
            "dataset_artifacts": {
                "route_probes": {"sha256": _sha256_file(probe_file)},
                "processed_dataset": {
                    "sha256": _sha256_file(processed_file),
                },
            },
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest_data))

        return BaselineRunner(
            backend=stub_backend,
            probe_path=probe_file,
            output_dir=tmp_path / "output",
            model_config=stub_model_config,
            resume=False,
            dataset_manifest_path=manifest_file,
            processed_dataset_path=processed_file,
        )

    # -- _extract_protocol ------------------------------------------------ #

    def test_extract_protocol_from_canonical(self, runner_with_roles):
        proto = runner_with_roles._extract_protocol()
        assert proto is not None
        assert proto["forget_bucket"] == "forget10"
        assert proto["train_bucket"] == "retain15"
        assert proto["eval_bucket"] is None
        assert proto["eval_fraction"] == pytest.approx(0.2)
        assert proto["eval_seed"] == 17

    def test_extract_protocol_returns_none_without_manifest(self, runner_no_map):
        """runner_no_map has no manifest → _extract_protocol returns None."""
        assert runner_no_map._extract_protocol() is None

    # -- _build_identity_role_map ----------------------------------------- #

    def test_build_map_exclude_role(self, runner_with_roles):
        assert runner_with_roles.identity_role_map["id_a"] == "exclude"

    def test_build_map_train_role(self, runner_with_roles):
        assert runner_with_roles.identity_role_map["id_b"] == "train"

    def test_build_map_out_of_protocol_role(self, runner_with_roles):
        assert runner_with_roles.identity_role_map["id_c"] == "out_of_protocol"

    def test_build_map_empty_without_processed_path(self, runner_no_map):
        assert runner_no_map.identity_role_map == {}

    def test_build_map_missing_file_raises(
        self, stub_backend, probe_jsonl, tmp_path, stub_model_config, manifest_jsonl
    ):
        with pytest.raises(FileNotFoundError, match="Processed dataset not found"):
            BaselineRunner(
                backend=stub_backend,
                probe_path=probe_jsonl,
                output_dir=tmp_path / "out",
                model_config=stub_model_config,
                dataset_manifest_path=manifest_jsonl,
                processed_dataset_path=tmp_path / "nonexistent.jsonl",
            )

    def test_build_map_without_manifest_raises(
        self, stub_backend, probe_jsonl, tmp_path, stub_model_config, processed_jsonl
    ):
        with pytest.raises(RuntimeError, match="no protocol configuration"):
            BaselineRunner(
                backend=stub_backend,
                probe_path=probe_jsonl,
                output_dir=tmp_path / "out",
                model_config=stub_model_config,
                processed_dataset_path=processed_jsonl,
                # no dataset_manifest_path
            )

    # -- run_probe protocol_role population (P0-7) ------------------------ #

    def test_run_probe_populates_exclude(self, runner_full_overlap):
        probe = runner_full_overlap.probes[0]
        assert probe.identity_id == "id_a"
        result = runner_full_overlap.run_probe(probe)
        assert result.protocol_role == "exclude"

    def test_run_probe_populates_train(self, runner_full_overlap):
        for p in runner_full_overlap.probes:
            if p.identity_id == "id_b":
                result = runner_full_overlap.run_probe(p)
                assert result.protocol_role == "train"
                break

    def test_run_probe_missing_identity_raises(self, runner_with_roles):
        """Probe identity not in processed JSONL → hard fail."""
        probe = runner_with_roles.probes[0]  # identity "aaaa"
        with pytest.raises(RuntimeError, match="not found in identity_role_map"):
            runner_with_roles.run_probe(probe)

    def test_run_probe_invalid_role_raises(
        self, stub_backend, tmp_path, stub_model_config
    ):
        """Identity with out_of_protocol role → hard fail."""
        from route_data.eval.baseline_runner import _sha256_file

        # Single probe with identity "id_c" (out_of_protocol).
        probe_data = dict(_PROBES_RAW[0])
        probe_data["identity_id"] = "id_c"
        probe_data["probe_id"] = "p_oop"
        probe_file = tmp_path / "probes.jsonl"
        probe_file.write_text(json.dumps(probe_data))

        manifest_data = {
            "protocol": {"canonical_protocol": dict(self._PROTO)},
            "dataset_artifacts": {
                "route_probes": {"sha256": _sha256_file(probe_file)},
            },
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest_data))

        processed_file = tmp_path / "processed.jsonl"
        processed_file.write_text(json.dumps({
            "identity_id": "id_c",
            "source_metadata": {
                "source_subject_id": "S003",
                "official_memberships": ["other_bucket"],
            },
        }) + "\n")

        runner = BaselineRunner(
            backend=stub_backend,
            probe_path=probe_file,
            output_dir=tmp_path / "out",
            model_config=stub_model_config,
            resume=False,
            dataset_manifest_path=manifest_file,
            processed_dataset_path=processed_file,
        )
        with pytest.raises(RuntimeError, match="invalid role"):
            runner.run_probe(runner.probes[0])

    def test_run_probe_no_map_gives_empty_role(self, runner_no_map):
        """Without identity_role_map, protocol_role is empty string."""
        probe = runner_no_map.probes[0]
        result = runner_no_map.run_probe(probe)
        assert result.protocol_role == ""

    # -- validate_results protocol_role_complete (P0-8) ------------------- #

    def test_validate_passes_with_valid_roles(self, runner_full_overlap):
        """All probes have valid roles → protocol_role_complete passes."""
        # id_c has out_of_protocol role which will cause run_probe to raise.
        # We need only id_a (exclude) and id_b (train) probes.
        # Filter to only valid-role probes.
        valid_probes = [
            p for p in runner_full_overlap.probes
            if p.identity_id in ("id_a", "id_b")
        ]
        results = runner_full_overlap.run_selected(valid_probes)
        runner_full_overlap._results = results
        # Override probes to match the filtered set for validation.
        runner_full_overlap.probes = valid_probes
        report = runner_full_overlap.validate_results()
        check = report["checks"]["protocol_role_complete"]
        assert check["pass"] is True
        assert check["invalid_role_count"] == 0
        assert check["inconsistent_identity_count"] == 0

    def test_validate_fails_on_empty_role(self, runner_full_overlap):
        """Corrupting a result's role to empty fails the check."""
        valid_probes = [
            p for p in runner_full_overlap.probes
            if p.identity_id in ("id_a", "id_b")
        ]
        results = runner_full_overlap.run_selected(valid_probes)
        runner_full_overlap._results = results
        runner_full_overlap.probes = valid_probes
        # Corrupt one result's role.
        runner_full_overlap._results[0].protocol_role = ""
        with pytest.raises(RuntimeError, match="protocol_role_complete"):
            runner_full_overlap.validate_results()

    def test_validate_raises_without_identity_role_map(self, runner):
        """Without identity_role_map the runner raises RuntimeError (P0-3)."""
        runner.run_all()
        # Clear the identity_role_map to simulate a missing processed dataset.
        runner.identity_role_map = {}
        with pytest.raises(RuntimeError, match="populated protocol roles"):
            runner.validate_results()

    # -- generate_summary per_protocol_role (P0-9) ------------------------ #

    def test_summary_per_protocol_role_with_roles(self, runner_full_overlap):
        """Summary includes per_protocol_role counts when roles populated."""
        valid_probes = [
            p for p in runner_full_overlap.probes
            if p.identity_id in ("id_a", "id_b")
        ]
        results = runner_full_overlap.run_selected(valid_probes)
        runner_full_overlap._results = results
        summary = runner_full_overlap.generate_summary()
        ppr = summary["per_protocol_role"]
        assert "exclude" in ppr
        assert "train" in ppr
        assert "eval" in ppr
        # 2 probes for id_a (exclude) + 2 for id_b (train) = 4 total
        assert ppr["exclude"]["n"] == 2
        assert ppr["train"]["n"] == 2
        assert ppr["eval"]["n"] == 0

    def test_summary_per_protocol_role_without_map(self, runner_no_map):
        """Summary includes per_protocol_role even without map (all zero)."""
        runner_no_map.run_all()
        summary = runner_no_map.generate_summary()
        assert "per_protocol_role" in summary
        ppr = summary["per_protocol_role"]
        assert ppr["exclude"]["n"] == 0
        assert ppr["train"]["n"] == 0
        assert ppr["eval"]["n"] == 0


# --------------------------------------------------------------------------- #
# Commit C — Research freeze hardening tests
# --------------------------------------------------------------------------- #


class TestRequireCleanGit:
    """P0-10: require_clean_git() enforces clean Git tree."""

    def test_clean_tree_returns_state(self, runner):
        with patch.object(
            BaselineRunner,
            "_get_git_state",
            staticmethod(lambda: {"git_commit": "b" * 40, "git_dirty": False}),
        ):
            state = runner.require_clean_git()
        assert state["git_commit"] == "b" * 40
        assert state["git_dirty"] is False

    def test_dirty_tree_raises(self, runner):
        with patch.object(
            BaselineRunner,
            "_get_git_state",
            staticmethod(lambda: {"git_commit": "c" * 40, "git_dirty": True}),
        ), pytest.raises(RuntimeError, match="Git tree is dirty"):
            runner.require_clean_git()

    def test_no_git_commit_raises(self, runner):
        with patch.object(
            BaselineRunner,
            "_get_git_state",
            staticmethod(lambda: {"git_commit": "", "git_dirty": False}),
        ), pytest.raises(RuntimeError, match="Cannot determine Git commit"):
            runner.require_clean_git()


class TestManifestDirtyTree:
    """P0-11: generate_baseline_manifest() refuses dirty tree."""

    def test_dirty_tree_raises(self, runner):
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        with patch.object(
            BaselineRunner,
            "_get_git_state",
            staticmethod(lambda: {"git_commit": "d" * 40, "git_dirty": True}),
        ), pytest.raises(RuntimeError, match="Git tree is dirty"):
            runner.generate_baseline_manifest()

    def test_no_git_commit_raises(self, runner):
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        with patch.object(
            BaselineRunner,
            "_get_git_state",
            staticmethod(lambda: {"git_commit": "", "git_dirty": False}),
        ), pytest.raises(RuntimeError, match="Cannot determine Git commit"):
            runner.generate_baseline_manifest()


class TestScoringProvenanceP11:
    """P1-1: Signed-margin provenance fields in manifest."""

    @pytest.fixture(autouse=True)
    def _mock_clean_git(self):
        with patch.object(
            BaselineRunner,
            "_get_git_state",
            staticmethod(lambda: {"git_commit": "a" * 40, "git_dirty": False}),
        ):
            yield

    def test_raw_log_margin_definition(self, runner):
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
        sc = manifest["scoring_config"]
        assert sc["raw_log_margin_definition"] == "logp_yes_minus_logp_no"

    def test_signed_answer_margin_definition(self, runner):
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
        sc = manifest["scoring_config"]
        assert "signed_answer_margin_definition" in sc
        assert "signed_answer_margin_interpretation" in sc
        assert sc["signed_answer_margin_interpretation"] == "higher_is_better"


class TestMetricSchemaVersion:
    """P1-2: metric_schema_version in manifest."""

    @pytest.fixture(autouse=True)
    def _mock_clean_git(self):
        with patch.object(
            BaselineRunner,
            "_get_git_state",
            staticmethod(lambda: {"git_commit": "a" * 40, "git_dirty": False}),
        ):
            yield

    def test_schema_version_is_1_2(self, runner):
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
        assert manifest["schema_version"] == "1.2"

    def test_metric_schema_version_present(self, runner):
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
        assert manifest["metric_schema_version"] == "baseline-metrics-v1"


class TestFreezeProvenanceInManifest:
    """P1-8: Freeze verification SHA and route probe SHA in manifest."""

    @pytest.fixture(autouse=True)
    def _mock_clean_git(self):
        with patch.object(
            BaselineRunner,
            "_get_git_state",
            staticmethod(lambda: {"git_commit": "a" * 40, "git_dirty": False}),
        ):
            yield

    def test_route_probe_sha256_in_provenance(self, runner):
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
        dp = manifest["dataset_provenance"]
        assert "route_probe_sha256" in dp

    def test_route_probe_count_in_provenance(self, runner):
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
        dp = manifest["dataset_provenance"]
        assert "route_probe_count" in dp
        assert dp["route_probe_count"] == 10

    def test_freeze_verification_sha256_present(self, runner):
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
        dp = manifest["dataset_provenance"]
        assert "freeze_verification_sha256" in dp


class TestValidateFingerprint:
    """P0-4/P0-5: validate_fingerprint() before inference."""

    def test_valid_fingerprint_passes(self, runner):
        runner._fingerprint = {
            "fingerprint_id": "test-model-v1",
            "revision": "abc123",
            "processor_class": "AutoProcessor",
            "tokenizer_class": "AutoTokenizer",
            "chat_template_hash": "ct" + "a" * 62,
        }
        runner._fingerprint_id = "test-model-v1"
        runner._model_config_sha = "x" * 64
        checks = runner.validate_fingerprint()
        assert checks["revision_match"] is True
        assert checks["fingerprint_non_empty"] is True
        assert checks["model_config_sha_non_empty"] is True
        assert checks["processor_tokenizer_available"] is True
        assert checks["missing_fingerprint_fields"] == []

    def test_revision_mismatch_raises(self, runner):
        runner._fingerprint = {
            "fingerprint_id": "test-model-v1",
            "revision": "rev999",
            "processor_class": "AutoProcessor",
            "tokenizer_class": "AutoTokenizer",
            "chat_template_hash": "ct" + "a" * 62,
        }
        runner._fingerprint_id = "test-model-v1"
        runner._model_config_sha = "x" * 64
        # model_config.revision is "abc123" from the fixture
        with pytest.raises(RuntimeError, match="Fingerprint validation failed"):
            runner.validate_fingerprint()

    def test_unknown_fingerprint_raises(self, runner):
        runner._fingerprint = {
            "revision": "abc123",
            "processor_class": "AutoProcessor",
            "tokenizer_class": "AutoTokenizer",
            "chat_template_hash": "ct" + "a" * 62,
        }
        runner._fingerprint_id = "unknown"
        runner._model_config_sha = "x" * 64
        with pytest.raises(RuntimeError, match="Fingerprint validation failed"):
            runner.validate_fingerprint()

    def test_empty_model_config_sha_raises(self, runner):
        runner._fingerprint = {
            "fingerprint_id": "test-model-v1",
            "revision": "abc123",
            "processor_class": "AutoProcessor",
            "tokenizer_class": "AutoTokenizer",
            "chat_template_hash": "ct" + "a" * 62,
        }
        runner._fingerprint_id = "test-model-v1"
        runner._model_config_sha = ""
        with pytest.raises(RuntimeError, match="Fingerprint validation failed"):
            runner.validate_fingerprint()

    def test_no_processor_tokenizer_raises(self, runner):
        runner._fingerprint = {
            "fingerprint_id": "test-model-v1",
            "revision": "abc123",
        }
        runner._fingerprint_id = "test-model-v1"
        runner._model_config_sha = "x" * 64
        with pytest.raises(RuntimeError, match="Fingerprint validation failed"):
            runner.validate_fingerprint()

    def test_missing_processor_class_raises(self, runner):
        runner._fingerprint = {
            "fingerprint_id": "test-model-v1",
            "revision": "abc123",
            "tokenizer_class": "AutoTokenizer",
            "chat_template_hash": "ct" + "a" * 62,
        }
        runner._fingerprint_id = "test-model-v1"
        runner._model_config_sha = "x" * 64
        with pytest.raises(RuntimeError, match="Fingerprint validation failed"):
            runner.validate_fingerprint()

    def test_missing_tokenizer_class_raises(self, runner):
        runner._fingerprint = {
            "fingerprint_id": "test-model-v1",
            "revision": "abc123",
            "processor_class": "AutoProcessor",
            "chat_template_hash": "ct" + "a" * 62,
        }
        runner._fingerprint_id = "test-model-v1"
        runner._model_config_sha = "x" * 64
        with pytest.raises(RuntimeError, match="Fingerprint validation failed"):
            runner.validate_fingerprint()

    def test_missing_chat_template_hash_raises(self, runner):
        runner._fingerprint = {
            "fingerprint_id": "test-model-v1",
            "revision": "abc123",
            "processor_class": "AutoProcessor",
            "tokenizer_class": "AutoTokenizer",
        }
        runner._fingerprint_id = "test-model-v1"
        runner._model_config_sha = "x" * 64
        with pytest.raises(RuntimeError, match="Fingerprint validation failed"):
            runner.validate_fingerprint()

    def test_empty_backend_revision_raises(self, runner):
        """P0-4: empty backend revision must fail when config revision is set."""
        runner._fingerprint = {
            "fingerprint_id": "test-model-v1",
            "revision": "",
            "processor_class": "AutoProcessor",
            "tokenizer_class": "AutoTokenizer",
            "chat_template_hash": "ct" + "a" * 62,
        }
        runner._fingerprint_id = "test-model-v1"
        runner._model_config_sha = "x" * 64
        # model_config.revision is "abc123" — backend revision empty → fail
        with pytest.raises(RuntimeError, match="Fingerprint validation failed"):
            runner.validate_fingerprint()


class TestProcessedDatasetShaBinding:
    """P1-6: Real-schema tests for processed dataset SHA binding.

    Verifies that ``validate_processed_dataset()`` correctly binds the
    processed dataset file to the frozen SHA in the research manifest.
    Uses the actual manifest schema
    (``dataset_artifacts.processed_dataset.sha256``).
    """

    def test_correct_sha_passes(self, runner: BaselineRunner):
        """When the processed dataset SHA matches the manifest, validation passes."""
        checks = runner.validate_processed_dataset()
        assert checks["processed_dataset_exists"] is True
        assert checks["processed_dataset_sha_match"] is True
        assert len(checks["processed_dataset_sha_actual"]) == 64
        assert checks["processed_dataset_sha_expected"] == checks["processed_dataset_sha_actual"]

    def test_wrong_sha_raises(self, runner: BaselineRunner, tmp_path: Path):
        """When the manifest records a different SHA, validation hard-fails."""
        from route_data.eval.baseline_runner import _sha256_file

        # Rewrite the manifest with an incorrect processed-dataset SHA.
        bad_manifest_path = tmp_path / "bad_manifest.json"
        bad_data = {
            "protocol": {"canonical_protocol": dict(_RUNNER_PROTO)},
            "dataset_artifacts": {
                "route_probes": {"sha256": _sha256_file(runner.probe_path)},
                "processed_dataset": {"sha256": "0" * 64},
            },
        }
        bad_manifest_path.write_text(json.dumps(bad_data))

        # Build a runner that uses the corrupted manifest.
        bad_runner = BaselineRunner(
            backend=runner.backend,
            probe_path=runner.probe_path,
            output_dir=tmp_path / "output_bad_sha",
            model_config=runner.model_config,
            resume=False,
            dataset_manifest_path=bad_manifest_path,
            processed_dataset_path=runner.processed_dataset_path,
        )
        with pytest.raises(RuntimeError, match="Processed dataset validation failed"):
            bad_runner.validate_processed_dataset()

    def test_missing_file_raises(
        self, stub_backend, probe_jsonl, tmp_path, stub_model_config
    ):
        """When the processed dataset file does not exist, construction fails."""
        from route_data.eval.baseline_runner import _sha256_file

        processed_path = tmp_path / "nonexistent.jsonl"
        manifest_path = tmp_path / "manifest_missing.json"
        manifest_data = {
            "protocol": {"canonical_protocol": dict(_RUNNER_PROTO)},
            "dataset_artifacts": {
                "route_probes": {"sha256": _sha256_file(probe_jsonl)},
                "processed_dataset": {"sha256": "0" * 64},
            },
        }
        manifest_path.write_text(json.dumps(manifest_data))

        with pytest.raises(FileNotFoundError, match="Processed dataset not found"):
            BaselineRunner(
                backend=stub_backend,
                probe_path=probe_jsonl,
                output_dir=tmp_path / "output_missing",
                model_config=stub_model_config,
                resume=False,
                dataset_manifest_path=manifest_path,
                processed_dataset_path=processed_path,
            )

    def test_no_processed_dataset_path_raises(self, runner_no_map: BaselineRunner):
        """When no processed dataset path was provided, validation fails."""
        with pytest.raises(RuntimeError, match="requires --processed-dataset"):
            runner_no_map.validate_processed_dataset()

    def test_no_manifest_raises(
        self, stub_backend, probe_jsonl, tmp_path, stub_model_config
    ):
        """When no protocol config in manifest, construction fails."""
        from route_data.eval.baseline_runner import _sha256_file

        processed_path = tmp_path / "proc.jsonl"
        processed_path.write_text('{"identity_id": "x"}\n')
        manifest_path = tmp_path / "manifest_no_proto.json"
        manifest_data = {
            "dataset_artifacts": {
                "route_probes": {"sha256": _sha256_file(probe_jsonl)},
                "processed_dataset": {"sha256": _sha256_file(processed_path)},
            },
        }
        manifest_path.write_text(json.dumps(manifest_data))

        with pytest.raises(RuntimeError, match="no protocol configuration"):
            BaselineRunner(
                backend=stub_backend,
                probe_path=probe_jsonl,
                output_dir=tmp_path / "output_no_manifest",
                model_config=stub_model_config,
                resume=False,
                dataset_manifest_path=manifest_path,
                processed_dataset_path=processed_path,
            )


class TestResearchPreflight:
    """P1-7/P1-8/P1-9: validate_research_preflight() integration tests."""

    def _prepare_runner_for_preflight(
        self, runner: BaselineRunner,
    ) -> patch:
        """Configure runner state so gates 1-4 pass.

        Sets a valid fingerprint, patches the manifest with fields needed
        by ``validate_dataset_manifest`` / ``validate_cross_file``, loads
        a passing freeze-verification dict, and mocks ``_read_jsonl`` to
        return 500 rows.

        Returns the ``_read_jsonl`` patcher — the caller **must** call
        ``patcher.stop()`` after all assertions are complete.
        """
        from route_data.eval.baseline_runner import _sha256_file

        # -- valid fingerprint (gates 5) ------------------------------------
        runner._fingerprint = {
            "fingerprint_id": "test-model-v1",
            "revision": "abc123",
            "processor_class": "AutoProcessor",
            "tokenizer_class": "AutoTokenizer",
            "chat_template_hash": "ct" + "a" * 62,
        }
        runner._fingerprint_id = "test-model-v1"
        runner._model_config_sha = "x" * 64

        # -- patch manifest with fields needed by gates 2 & 3 ---------------
        runner._dataset_manifest["dataset_artifacts"]["route_probes"][
            "total_probes"
        ] = 500
        runner._dataset_manifest["dataset_artifacts"]["route_probes"][
            "families"
        ] = {
            "direct_visual": 100,
            "image_plus_name": 100,
            "wrong_name": 100,
            "visual_text_conflict": 100,
            "name_only": 100,
        }

        # -- freeze verification with correct SHAs (gates 1 & 3) -----------
        manifest_sha = _sha256_file(runner.dataset_manifest_path)
        probe_sha = _sha256_file(runner.probe_path)
        runner._freeze_verification = {
            "dataset_version": "fiubench-route-v1",
            "ready_for_experiments": True,
            "bundle_verifier_pass": True,
            "strict_final_verify_pass": True,
            "manual_audit_pass": True,
            "exact_ci_pass": True,
            "hard_stop_conditions": {
                "manual_audit_matches_current_route_artifact": True,
                "manual_audit_route_count_matches": True,
                "all_artifact_hashes_verified": True,
                "all_commits_reachable": True,
                "git_dirty_false": True,
            },
            "dataset_manifest_sha256": manifest_sha,
            "route_probe_sha256": probe_sha,
        }

        # -- mock _read_jsonl to return 500 rows (gate 2) -------------------
        patcher = patch(
            "route_data.eval.baseline_runner._read_jsonl",
            return_value=[{}] * 500,
        )
        patcher.start()
        return patcher

    def test_preflight_passes_writes_report(
        self, runner: BaselineRunner, tmp_path: Path,
    ):
        """All gates pass → report has pass=true and is written to disk."""
        read_patcher = self._prepare_runner_for_preflight(runner)
        try:
            with patch(
                "route_data.eval.baseline_runner.BaselineRunner._get_git_state",
                staticmethod(lambda: {"git_commit": "a" * 40, "git_dirty": False}),
            ):
                report = runner.validate_research_preflight()
            assert report["pass"] is True
            assert "freeze" in report["gates"]
            assert "dataset" in report["gates"]
            assert "cross_file" in report["gates"]
            assert "processed_dataset" in report["gates"]
            assert "fingerprint" in report["gates"]
            assert "git" in report["gates"]
            # Report written to disk.
            report_path = runner.output_dir / "preflight_report.json"
            assert report_path.is_file()
            with open(report_path) as f:
                data = json.load(f)
            assert data["pass"] is True
        finally:
            read_patcher.stop()

    def test_preflight_dirty_git_raises(
        self, runner: BaselineRunner,
    ):
        """Dirty Git tree → preflight fails at the git gate."""
        read_patcher = self._prepare_runner_for_preflight(runner)
        try:
            with (
                patch(
                    "route_data.eval.baseline_runner.BaselineRunner._get_git_state",
                    staticmethod(lambda: {"git_commit": "a" * 40, "git_dirty": True}),
                ),
                pytest.raises(RuntimeError, match="Git tree is dirty"),
            ):
                runner.validate_research_preflight()
            # Report still written (with pass=false).
            report_path = runner.output_dir / "preflight_report.json"
            assert report_path.is_file()
            with open(report_path) as f:
                data = json.load(f)
            assert data["pass"] is False
        finally:
            read_patcher.stop()

    def test_preflight_fingerprint_mismatch_raises(
        self, runner: BaselineRunner,
    ):
        """Fingerprint revision mismatch → preflight fails at fingerprint gate."""
        read_patcher = self._prepare_runner_for_preflight(runner)
        try:
            # Override revision to a wrong value (gate 5).
            runner._fingerprint["revision"] = "WRONG_REV"
            with (
                patch(
                    "route_data.eval.baseline_runner.BaselineRunner._get_git_state",
                    staticmethod(lambda: {"git_commit": "a" * 40, "git_dirty": False}),
                ),
                pytest.raises(
                    RuntimeError, match="Fingerprint validation failed"
                ),
            ):
                runner.validate_research_preflight()
            # Report still written.
            report_path = runner.output_dir / "preflight_report.json"
            assert report_path.is_file()
        finally:
            read_patcher.stop()

    def test_preflight_no_git_commit_raises(
        self, runner: BaselineRunner,
    ):
        """Missing Git commit → preflight fails at git gate."""
        read_patcher = self._prepare_runner_for_preflight(runner)
        try:
            with (
                patch(
                    "route_data.eval.baseline_runner.BaselineRunner._get_git_state",
                    staticmethod(lambda: {"git_commit": "", "git_dirty": False}),
                ),
                pytest.raises(
                    RuntimeError, match="Cannot determine Git commit"
                ),
            ):
                runner.validate_research_preflight()
        finally:
            read_patcher.stop()
