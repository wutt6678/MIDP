"""Schema tests for FIUBench baseline results (Stage 2.4).

These tests exercise the :class:`BaselineRunner` with the *stub* backend so
that no GPU is required.  They verify the structural invariants of the
result schema that downstream analysis stages depend on.
"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path

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
        "image_uri": None,
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
        "image_uri": None,
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
        "image_uri": None,
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
        "image_uri": None,
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
]


@pytest.fixture()
def probe_jsonl(tmp_path: Path) -> Path:
    """Write the 5 test probes to a temporary JSONL file."""
    path = tmp_path / "test_probes.jsonl"
    with open(path, "w") as f:
        for p in _PROBES_RAW:
            f.write(json.dumps(p) + "\n")
    return path


@pytest.fixture()
def stub_model_config() -> ModelConfig:
    return ModelConfig(backend="stub", model_id="test-model", revision="abc123")


@pytest.fixture()
def stub_backend(stub_model_config: ModelConfig):
    from route_data.models.registry import create_backend

    return create_backend(stub_model_config)


@pytest.fixture()
def runner(
    stub_backend,
    probe_jsonl: Path,
    tmp_path: Path,
    stub_model_config: ModelConfig,
) -> BaselineRunner:
    return BaselineRunner(
        backend=stub_backend,
        probe_path=probe_jsonl,
        output_dir=tmp_path / "output",
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
        assert BaselineProbe.from_dict(_PROBES_RAW[0]).has_image is False  # image_uri is None
        # With a non-None image_uri
        d = dict(_PROBES_RAW[0])
        d["image_uri"] = "/some/path.jpg"
        assert BaselineProbe.from_dict(d).has_image is True


class TestBaselineResultSchema:
    """Verify the result dataclass has all required fields."""

    REQUIRED_FIELDS = {
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
    }

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
        assert len(runner.probes) == 5

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
        assert len(results) == 5

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
        assert len(lines) == 5

    def test_generate_summary_writes_json(self, runner: BaselineRunner):
        runner.run_all()
        summary = runner.generate_summary()
        summary_path = runner.output_dir / "baseline_summary.json"
        assert summary_path.is_file()
        assert summary["total_probes"] == 5
        assert "per_family" in summary

    def test_resume_skips_cached(self, runner: BaselineRunner, tmp_path: Path):
        # First run: all 5 probes
        runner.run_all()
        runner.save_results()
        assert len(runner._results) == 5

        # Second run with resume=True: should reuse cache
        runner2 = BaselineRunner(
            backend=runner.backend,
            probe_path=runner.probe_path,
            output_dir=runner.output_dir,
            model_config=runner.model_config,
            resume=True,
        )
        assert len(runner2._results) == 5  # loaded from cache
        new_results = runner2.run_all()
        # No new results — all were cached
        assert len(new_results) == 5


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
            identity_id="i1",
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
            identity_id="i1",
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
            identity_id="i1",
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
            identity_id="i1",
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
        assert result.identity_id == "i1"
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

    def test_no_manifest_raises(self, runner: BaselineRunner):
        with pytest.raises(RuntimeError, match="No dataset manifest"):
            runner.verify_input_hashes_from_manifest()

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
    """Verify the strict completeness validator (Commit C)."""

    def test_validate_passes_with_complete_results(self, runner: BaselineRunner):
        runner.run_all()
        report = runner.validate_results()
        assert report["pass"] is True
        assert report["checks"]["probe_count"]["pass"] is True
        assert report["checks"]["no_errors"]["pass"] is True
        assert report["checks"]["all_families"]["pass"] is True
        assert report["checks"]["provenance_populated"]["pass"] is True

    def test_validate_writes_report(self, runner: BaselineRunner):
        runner.run_all()
        runner.validate_results()
        report_path = runner.output_dir / "validation_report.json"
        assert report_path.is_file()
        with open(report_path) as f:
            data = json.load(f)
        assert data["pass"] is True

    def test_validate_fails_on_count_mismatch(self, runner: BaselineRunner):
        runner.run_all()
        with pytest.raises(RuntimeError, match="probe_count"):
            runner.validate_results(expected_probe_count=999)

    def test_validate_fails_on_missing_results(self, runner: BaselineRunner):
        # Don't run any probes — results list is empty
        with pytest.raises(RuntimeError, match="probe_count"):
            runner.validate_results()

    def test_validate_custom_expected_count(self, runner: BaselineRunner):
        runner.run_all()
        # We have 5 probes, so expect 5
        report = runner.validate_results(expected_probe_count=5)
        assert report["pass"] is True


class TestBaselineManifest:
    """Verify baseline_manifest.json generation (Commit C)."""

    def test_manifest_written(self, runner: BaselineRunner):
        runner.run_all()
        runner.save_results()
        runner.generate_summary()
        manifest = runner.generate_baseline_manifest()
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
        assert dp["probe_count"] == 5

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
    """Verify the deterministic smoke probe selector (Commit D)."""

    def test_selects_all_families(self, runner: BaselineRunner):
        selected = select_smoke_probes(runner.probes)
        families = {p.probe_family for p in selected}
        assert families == ALL_FAMILIES

    def test_deterministic(self, runner: BaselineRunner):
        s1 = select_smoke_probes(runner.probes)
        s2 = select_smoke_probes(runner.probes)
        assert [p.probe_id for p in s1] == [p.probe_id for p in s2]

    def test_respects_n_identities(self, runner: BaselineRunner):
        selected = select_smoke_probes(runner.probes, n_identities=1)
        identities = {p.identity_id for p in selected}
        assert len(identities) <= 1

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
        assert set(data["families_covered"]) == ALL_FAMILIES

    def test_write_smoke_manifest_with_sha(
        self, runner: BaselineRunner, tmp_path: Path
    ):
        selected = select_smoke_probes(runner.probes)
        manifest_path = write_smoke_manifest(
            selected, tmp_path / "smoke.json", probe_file_sha256="abc123"
        )
        with open(manifest_path) as f:
            data = json.load(f)
        assert data["probe_file_sha256"] == "abc123"


class TestSpyCallPaths:
    """Verify the runner calls backend methods correctly (Commit D)."""

    def test_image_probe_calls_score_candidates(self, runner: BaselineRunner):
        """Image-bearing probes should call score_candidates."""
        from unittest.mock import patch, MagicMock

        probe = BaselineProbe(
            probe_id="spy_img",
            sample_id="s1",
            identity_id="i1",
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
            identity_id="i1",
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
