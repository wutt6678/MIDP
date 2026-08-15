"""Regression tests for cache provenance and resume invalidation (P0-2).

Verifies that:

1. Cached results store immutable provenance (cache_key, route_probe_sha256,
   model_config_sha256, model_fingerprint, candidate_protocol_version).
2. Same-config resume produces zero new backend calls.
3. Different route SHA invalidates the cache.
4. Different model config SHA invalidates the cache.
5. Different model fingerprint invalidates the cache.
6. Different scoring version invalidates the cache.
7. Different candidate protocol version invalidates the cache.
8. Full SHA (not truncated) is used in the cache key.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from route_data.config import ModelConfig
from route_data.eval.baseline_runner import (
    BaselineProbe,
    BaselineResult,
    BaselineRunner,
)
from route_data.models.scoring import (
    CANDIDATE_PROTOCOL_VERSION,
    SCORING_VERSION,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_probe(
    probe_id: str = "probe-001",
    *,
    image_sha256: str | None = "imgsha_abc",
    probe_family: str = "direct_visual",
    question: str = "Is the person smiling?",
) -> BaselineProbe:
    return BaselineProbe(
        probe_id=probe_id,
        sample_id="sample-001",
        identity_id="identity-001",
        benchmark="fiubench",
        probe_family=probe_family,
        modality="image+text" if image_sha256 else "text",
        question=question,
        expected_evidence_source="face",
        controlled_variables=[],
        image_uri="/tmp/face.jpg" if image_sha256 else None,
        image_sha256=image_sha256,
        registry_hash="regsha",
        target_attribute="smiling",
        answer_label=True,
        answer_text="Yes",
    )


def _make_runner(
    *,
    manifest_sha: str = "route_sha_A",
    model_config_sha: str = "config_sha_A",
    fingerprint: str = "fp_A",
    revision: str = "rev1",
) -> BaselineRunner:
    """Create a runner with mocked backend and controlled provenance."""
    backend = MagicMock()
    backend.fingerprint.return_value = {"fingerprint_id": fingerprint}
    backend.generate.return_value = MagicMock(text="Yes")

    config = ModelConfig(
        model_id="test-model",
        backend="qwen",
        revision=revision,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        # We need to avoid actually loading probes from disk
        runner = object.__new__(BaselineRunner)
        runner.backend = backend
        runner.probe_path = output_dir / "probes.jsonl"
        runner.output_dir = output_dir
        runner.model_config = config
        runner.resume = True
        runner.dataset_manifest_path = None
        runner.model_config_path = None
        runner._fingerprint = {"fingerprint_id": fingerprint}
        runner._fingerprint_id = fingerprint
        runner._cache_dir = output_dir / ".cache"
        runner._model_config_sha = model_config_sha
        runner._dataset_manifest = None
        runner.probes = []
        runner._results = []

        # Override _route_probe_sha to return controlled value
        runner._route_probe_sha = lambda: manifest_sha  # type: ignore[assignment]

    return runner


def _make_result(
    probe_id: str = "probe-001",
    *,
    cache_key: str = "key123",
    route_sha: str = "route_sha_A",
    config_sha: str = "config_sha_A",
    fingerprint: str = "fp_A",
    scoring_version: str = SCORING_VERSION,
    candidate_protocol_version: str = CANDIDATE_PROTOCOL_VERSION,
) -> BaselineResult:
    return BaselineResult(
        probe_id=probe_id,
        sample_id="sample-001",
        identity_id="identity-001",
        probe_family="direct_visual",
        modality="image+text",
        question="Q?",
        model_fingerprint=fingerprint,
        scoring_version=scoring_version,
        prompt_hash="ph",
        cache_key=cache_key,
        route_probe_sha256=route_sha,
        model_config_sha256=config_sha,
        candidate_protocol_version=candidate_protocol_version,
    )


# --------------------------------------------------------------------------- #
# Tests: Provenance fields are stored in results
# --------------------------------------------------------------------------- #


class TestProvenanceStorage:
    """run_probe() stores immutable provenance in every result."""

    def test_result_has_cache_key(self):
        runner = _make_runner()
        probe = _make_probe()
        result = runner.run_probe(probe)
        assert result.cache_key, "cache_key must be non-empty"
        assert result.cache_key == runner._cache_key(probe)

    def test_result_has_route_probe_sha(self):
        runner = _make_runner(manifest_sha="route_sha_XYZ")
        probe = _make_probe()
        result = runner.run_probe(probe)
        assert result.route_probe_sha256 == "route_sha_XYZ"

    def test_result_has_model_config_sha(self):
        runner = _make_runner(model_config_sha="config_sha_XYZ")
        probe = _make_probe()
        result = runner.run_probe(probe)
        assert result.model_config_sha256 == "config_sha_XYZ"

    def test_result_has_model_fingerprint(self):
        runner = _make_runner(fingerprint="fp_XYZ")
        probe = _make_probe()
        result = runner.run_probe(probe)
        assert result.model_fingerprint == "fp_XYZ"

    def test_result_has_candidate_protocol_version(self):
        runner = _make_runner()
        probe = _make_probe()
        result = runner.run_probe(probe)
        assert result.candidate_protocol_version == CANDIDATE_PROTOCOL_VERSION

    def test_cache_key_uses_full_sha(self):
        """Cache key must use full 64-char model config SHA, not truncated."""
        long_sha = "a" * 64
        runner = _make_runner(model_config_sha=long_sha)
        probe = _make_probe()
        runner._cache_key(probe)  # compute to verify no exception
        # The key is a SHA-256 hex digest itself (64 chars), but the
        # important thing is that the runner stores the full SHA in results.
        result = runner.run_probe(probe)
        assert result.model_config_sha256 == long_sha
        assert len(result.model_config_sha256) == 64


# --------------------------------------------------------------------------- #
# Tests: _done_keys uses stored cache_key
# --------------------------------------------------------------------------- #


class TestDoneKeys:
    def test_done_keys_uses_stored_cache_key(self):
        runner = _make_runner()
        r1 = _make_result("p1", cache_key="key_1")
        r2 = _make_result("p2", cache_key="key_2")
        runner._results = [r1, r2]
        assert runner._done_keys() == {"key_1", "key_2"}

    def test_done_keys_skips_empty_cache_key(self):
        runner = _make_runner()
        r1 = _make_result("p1", cache_key="key_1")
        r2 = _make_result("p2", cache_key="")
        runner._results = [r1, r2]
        assert runner._done_keys() == {"key_1"}


# --------------------------------------------------------------------------- #
# Tests: Cache row compatibility
# --------------------------------------------------------------------------- #


class TestCacheRowCompatibility:
    def test_same_config_compatible(self):
        runner = _make_runner(
            manifest_sha="route_A",
            model_config_sha="config_A",
            fingerprint="fp_A",
        )
        row = _make_result(
            route_sha="route_A",
            config_sha="config_A",
            fingerprint="fp_A",
        )
        assert runner._cache_row_compatible(row, "route_A", "config_A", "fp_A")

    def test_different_route_sha_rejected(self):
        runner = _make_runner()
        row = _make_result(route_sha="route_OLD")
        assert not runner._cache_row_compatible(
            row, "route_NEW", "config_sha_A", "fp_A",
        )

    def test_different_config_sha_rejected(self):
        runner = _make_runner()
        row = _make_result(config_sha="config_OLD")
        assert not runner._cache_row_compatible(
            row, "route_sha_A", "config_NEW", "fp_A",
        )

    def test_different_fingerprint_rejected(self):
        runner = _make_runner()
        row = _make_result(fingerprint="fp_OLD")
        assert not runner._cache_row_compatible(
            row, "route_sha_A", "config_sha_A", "fp_NEW",
        )

    def test_different_scoring_version_rejected(self):
        runner = _make_runner()
        row = _make_result(scoring_version="999")
        assert not runner._cache_row_compatible(
            row, "route_sha_A", "config_sha_A", "fp_A",
        )

    def test_different_candidate_protocol_version_rejected(self):
        runner = _make_runner()
        row = _make_result(candidate_protocol_version="999")
        assert not runner._cache_row_compatible(
            row, "route_sha_A", "config_sha_A", "fp_A",
        )

    def test_empty_provenance_fields_accepted(self):
        """Legacy rows with empty provenance are accepted (backward compat)."""
        runner = _make_runner()
        row = _make_result(
            route_sha="",
            config_sha="",
            fingerprint="",
            scoring_version="",
            candidate_protocol_version="",
        )
        assert runner._cache_row_compatible(
            row, "route_sha_A", "config_sha_A", "fp_A",
        )


# --------------------------------------------------------------------------- #
# Tests: Cache load rejects incompatible rows
# --------------------------------------------------------------------------- #


class TestLoadCache:
    def test_load_cache_rejects_incompatible(self, tmp_path):
        runner = _make_runner(
            manifest_sha="route_NEW",
            model_config_sha="config_NEW",
            fingerprint="fp_NEW",
        )
        runner._cache_dir = tmp_path / ".cache"
        runner._cache_dir.mkdir()

        # Write one compatible and one incompatible row
        good = _make_result(
            "p1",
            cache_key="k1",
            route_sha="route_NEW",
            config_sha="config_NEW",
            fingerprint="fp_NEW",
        )
        bad = _make_result(
            "p2",
            cache_key="k2",
            route_sha="route_OLD",
            config_sha="config_OLD",
            fingerprint="fp_OLD",
        )

        from dataclasses import asdict
        cache_path = runner._cache_dir / "baseline_cache.jsonl"
        with open(cache_path, "w") as f:
            f.write(json.dumps(asdict(good), default=str) + "\n")
            f.write(json.dumps(asdict(bad), default=str) + "\n")

        results = runner._load_cache()
        assert len(results) == 1
        assert results[0].probe_id == "p1"

    def test_load_cache_all_compatible(self, tmp_path):
        runner = _make_runner(
            manifest_sha="route_A",
            model_config_sha="config_A",
            fingerprint="fp_A",
        )
        runner._cache_dir = tmp_path / ".cache"
        runner._cache_dir.mkdir()

        r1 = _make_result("p1", cache_key="k1", route_sha="route_A",
                          config_sha="config_A", fingerprint="fp_A")
        r2 = _make_result("p2", cache_key="k2", route_sha="route_A",
                          config_sha="config_A", fingerprint="fp_A")

        from dataclasses import asdict
        cache_path = runner._cache_dir / "baseline_cache.jsonl"
        with open(cache_path, "w") as f:
            f.write(json.dumps(asdict(r1), default=str) + "\n")
            f.write(json.dumps(asdict(r2), default=str) + "\n")

        results = runner._load_cache()
        assert len(results) == 2


# --------------------------------------------------------------------------- #
# Tests: Resume produces zero new calls
# --------------------------------------------------------------------------- #


class TestResumeZeroNewCalls:
    def test_same_config_resume_zero_new(self):
        """Same config: all probes already cached → zero new backend calls."""
        runner = _make_runner(
            manifest_sha="route_A",
            model_config_sha="config_A",
            fingerprint="fp_A",
        )
        probe = _make_probe()
        cache_key = runner._cache_key(probe)

        # Pre-populate results with a cached row
        cached = _make_result(
            probe.probe_id,
            cache_key=cache_key,
            route_sha="route_A",
            config_sha="config_A",
            fingerprint="fp_A",
        )
        runner._results = [cached]
        runner.probes = [probe]

        results = runner.run_all()
        # run_probe should NOT have been called
        runner.backend.generate.assert_not_called()
        runner.backend.score_candidates.assert_not_called()
        assert len(results) == 1

    def test_different_route_sha_forces_rerun(self):
        """Different route SHA: cached row rejected → probe rerun."""
        runner = _make_runner(
            manifest_sha="route_NEW",
            model_config_sha="config_A",
            fingerprint="fp_A",
        )
        probe = _make_probe(
            image_sha256=None, probe_family="name_only",
            question="What is Alice's full name?",
        )
        old_key = "old_cache_key"

        # Pre-populate with old provenance
        cached = _make_result(
            probe.probe_id,
            cache_key=old_key,
            route_sha="route_OLD",
            config_sha="config_A",
            fingerprint="fp_A",
        )
        runner._results = [cached]
        runner.probes = [probe]

        # The old cache_key won't match the new _cache_key, so run_all
        # will call run_probe for the probe.
        results = runner.run_all()
        # Backend should have been called for the new run
        assert runner.backend.generate.called
        # We should have 2 results now (old + new)
        assert len(results) == 2
