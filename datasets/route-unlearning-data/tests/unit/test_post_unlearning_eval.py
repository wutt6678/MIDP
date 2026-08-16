"""Unit tests for the post-unlearning evaluator (Stage 3, Commit 3).

These tests verify the structural invariants of the post-evaluation
pipeline: exact probe matching, manifest generation, and validation.
No GPU or real model weights are required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from route_data.eval.post_unlearning_eval import (
    PostEvalConfig,
    PostUnlearningEvaluator,
    validate_exact_probe_matching,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_result(probe_id: str, family: str, margin: float = 1.0) -> dict:
    """Create a minimal result dict for testing."""
    return {
        "probe_id": probe_id,
        "sample_id": f"sample_{probe_id}",
        "identity_id": f"identity_{probe_id[:8]}",
        "probe_family": family,
        "modality": "image_text",
        "question": "Is the person bald?",
        "signed_answer_margin": margin,
        "correct": margin > 0,
        "error": None,
    }


def _write_results(path: Path, results: list[dict]) -> None:
    """Write results to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in results)


# --------------------------------------------------------------------------- #
# Tests – configuration
# --------------------------------------------------------------------------- #

class TestPostEvalConfig:
    """Tests for :class:`PostEvalConfig`."""

    def test_defaults(self) -> None:
        cfg = PostEvalConfig()
        assert cfg.model_id == "Qwen/Qwen3.5-9B"
        assert cfg.model_revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        assert cfg.seed == 17

    def test_experiment_id(self) -> None:
        cfg = PostEvalConfig()
        assert cfg.experiment_id == "fiubench_unlearning_pilot_v1"

    def test_custom_checkpoint(self) -> None:
        cfg = PostEvalConfig(
            checkpoint_path="/path/to/checkpoint",
            checkpoint_name="step_050",
        )
        assert cfg.checkpoint_path == "/path/to/checkpoint"
        assert cfg.checkpoint_name == "step_050"


# --------------------------------------------------------------------------- #
# Tests – probe validation
# --------------------------------------------------------------------------- #

class TestProbeValidation:
    """Tests for :func:`validate_exact_probe_matching`."""

    def test_exact_match_passes(self) -> None:
        """Identical probe ID sets should pass validation."""
        baseline = [_make_result(f"probe_{i:03d}", "direct_visual") for i in range(10)]
        post = [_make_result(f"probe_{i:03d}", "direct_visual") for i in range(10)]

        result = validate_exact_probe_matching(baseline, post)

        assert result["passed"] is True
        assert result["exact_match"] is True
        assert result["missing_probes"] == []
        assert result["extra_probes"] == []
        assert result["duplicate_baseline"] == 0
        assert result["duplicate_post"] == 0

    def test_missing_probe_fails(self) -> None:
        """Missing a probe should fail validation."""
        baseline = [_make_result(f"probe_{i:03d}", "direct_visual") for i in range(10)]
        post = [_make_result(f"probe_{i:03d}", "direct_visual") for i in range(9)]  # Missing probe_009

        result = validate_exact_probe_matching(baseline, post)

        assert result["passed"] is False
        assert result["exact_match"] is False
        assert "probe_009" in result["missing_probes"]

    def test_extra_probe_fails(self) -> None:
        """Extra probe should fail validation."""
        baseline = [_make_result(f"probe_{i:03d}", "direct_visual") for i in range(10)]
        post = [_make_result(f"probe_{i:03d}", "direct_visual") for i in range(11)]  # Extra probe_010

        result = validate_exact_probe_matching(baseline, post)

        assert result["passed"] is False
        assert "probe_010" in result["extra_probes"]

    def test_duplicate_baseline_fails(self) -> None:
        """Duplicate probe IDs in baseline should fail."""
        baseline = [_make_result("probe_000", "direct_visual")] * 2
        post = [_make_result("probe_000", "direct_visual")]

        result = validate_exact_probe_matching(baseline, post)

        assert result["passed"] is False
        assert result["duplicate_baseline"] == 1

    def test_duplicate_post_fails(self) -> None:
        """Duplicate probe IDs in post should fail."""
        baseline = [_make_result("probe_000", "direct_visual")]
        post = [_make_result("probe_000", "direct_visual")] * 2

        result = validate_exact_probe_matching(baseline, post)

        assert result["passed"] is False
        assert result["duplicate_post"] == 1

    def test_counts_reported(self) -> None:
        """Probe counts should be reported."""
        baseline = [_make_result(f"probe_{i:03d}", "direct_visual") for i in range(5)]
        post = [_make_result(f"probe_{i:03d}", "direct_visual") for i in range(5)]

        result = validate_exact_probe_matching(baseline, post)

        assert result["baseline_probe_count"] == 5
        assert result["post_probe_count"] == 5


# --------------------------------------------------------------------------- #
# Tests – post evaluator
# --------------------------------------------------------------------------- #

class TestPostUnlearningEvaluator:
    """Tests for :class:`PostUnlearningEvaluator`."""

    def test_validate_against_baseline_exact_match(self, tmp_path: Path) -> None:
        """Validation should pass when probe IDs match exactly."""
        # Create baseline results
        baseline_results = [
            _make_result(f"probe_{i:03d}", "direct_visual")
            for i in range(10)
        ]
        baseline_path = tmp_path / "baseline_results.jsonl"
        _write_results(baseline_path, baseline_results)

        # Create post results (same IDs)
        post_results = [
            _make_result(f"probe_{i:03d}", "direct_visual")
            for i in range(10)
        ]

        # Mock the evaluator
        cfg = PostEvalConfig(
            baseline_results_path=str(baseline_path),
            output_dir=str(tmp_path / "post_eval"),
        )
        backend = MagicMock()
        model_config = MagicMock()

        evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
        evaluator.config = cfg
        evaluator.backend = backend
        evaluator.model_config = model_config
        evaluator._runner = MagicMock()
        evaluator._results = []

        # Mock the BaselineResult objects
        from unittest.mock import MagicMock as Mock
        mock_results = []
        for r in post_results:
            mock_r = Mock()
            mock_r.probe_id = r["probe_id"]
            mock_r.error = None
            mock_results.append(mock_r)
        evaluator._results = mock_results

        validation = evaluator.validate_against_baseline()

        assert validation["passed"] is True
        assert validation["exact_match"] is True

    def test_validate_against_baseline_mismatch(self, tmp_path: Path) -> None:
        """Validation should fail when probe IDs don't match."""
        baseline_results = [
            _make_result(f"probe_{i:03d}", "direct_visual")
            for i in range(10)
        ]
        baseline_path = tmp_path / "baseline_results.jsonl"
        _write_results(baseline_path, baseline_results)

        # Post results missing one probe
        post_results = [
            _make_result(f"probe_{i:03d}", "direct_visual")
            for i in range(9)
        ]

        cfg = PostEvalConfig(
            baseline_results_path=str(baseline_path),
            output_dir=str(tmp_path / "post_eval"),
        )

        evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
        evaluator.config = cfg
        evaluator._results = []

        from unittest.mock import MagicMock as Mock
        mock_results = []
        for r in post_results:
            mock_r = Mock()
            mock_r.probe_id = r["probe_id"]
            mock_results.append(mock_r)
        evaluator._results = mock_results

        validation = evaluator.validate_against_baseline()

        assert validation["passed"] is False
        assert "probe_009" in validation["missing_probes"]


# --------------------------------------------------------------------------- #
# Tests – manifest generation
# --------------------------------------------------------------------------- #

class TestPostEvalManifest:
    """Tests for post-eval manifest generation."""

    def test_manifest_has_required_fields(self, tmp_path: Path) -> None:
        """Manifest should have all required provenance fields."""
        # Create baseline results
        baseline_results = [
            _make_result(f"probe_{i:03d}", "direct_visual")
            for i in range(5)
        ]
        baseline_path = tmp_path / "baseline_results.jsonl"
        _write_results(baseline_path, baseline_results)

        # Create post results
        output_dir = tmp_path / "post_eval"
        output_dir.mkdir(parents=True, exist_ok=True)
        post_results_path = output_dir / "results.jsonl"
        _write_results(post_results_path, baseline_results)

        cfg = PostEvalConfig(
            checkpoint_path="/fake/checkpoint",
            checkpoint_name="step_050",
            probe_path="/fake/probes.jsonl",
            baseline_results_path=str(baseline_path),
            baseline_manifest_path="/fake/manifest.json",
            output_dir=str(output_dir),
            selection_manifest_sha256="abc123",
            unlearning_run_manifest_sha256="def456",
            code_commit="commit789",
        )

        evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
        evaluator.config = cfg
        evaluator._results = []

        from unittest.mock import MagicMock as Mock
        mock_results = []
        for r in baseline_results:
            mock_r = Mock()
            mock_r.probe_id = r["probe_id"]
            mock_r.probe_family = r["probe_family"]
            mock_r.error = None
            mock_r.signed_answer_margin = r["signed_answer_margin"]
            mock_r.correct = r["correct"]
            mock_results.append(mock_r)
        evaluator._results = mock_results

        # Mock the runner
        mock_runner = Mock()
        mock_runner.generate_summary.return_value = {
            "total_probes": 5,
            "families": {"direct_visual": {"count": 5}},
        }
        evaluator._runner = mock_runner

        manifest = evaluator.generate_post_eval_manifest()

        assert manifest["experiment_id"] == "fiubench_unlearning_pilot_v1"
        assert manifest["evaluation_type"] == "post_unlearning"
        assert manifest["checkpoint"]["name"] == "step_050"
        assert manifest["checkpoint"]["path"] == "/fake/checkpoint"
        assert manifest["base_model"]["model_id"] == "Qwen/Qwen3.5-9B"
        assert manifest["provenance"]["selection_manifest_sha256"] == "abc123"
        assert manifest["provenance"]["unlearning_run_manifest_sha256"] == "def456"
        assert manifest["validation"]["exact_match"] is True
        assert manifest["seed"] == 17

    def test_manifest_written_to_disk(self, tmp_path: Path) -> None:
        """Manifest should be written to disk."""
        baseline_results = [
            _make_result(f"probe_{i:03d}", "direct_visual")
            for i in range(5)
        ]
        baseline_path = tmp_path / "baseline_results.jsonl"
        _write_results(baseline_path, baseline_results)

        output_dir = tmp_path / "post_eval"
        output_dir.mkdir(parents=True, exist_ok=True)
        # P0-11: Results and manifest live in checkpoint-specific subdir.
        checkpoint_dir = output_dir / "step_050"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        post_results_path = checkpoint_dir / "results.jsonl"
        _write_results(post_results_path, baseline_results)

        cfg = PostEvalConfig(
            checkpoint_path="/fake/checkpoint",
            checkpoint_name="step_050",
            probe_path="/fake/probes.jsonl",
            baseline_results_path=str(baseline_path),
            output_dir=str(output_dir),
        )

        evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
        evaluator.config = cfg
        evaluator._results = []

        from unittest.mock import MagicMock as Mock
        mock_results = []
        for r in baseline_results:
            mock_r = Mock()
            mock_r.probe_id = r["probe_id"]
            mock_r.probe_family = r["probe_family"]
            mock_r.error = None
            mock_r.signed_answer_margin = r["signed_answer_margin"]
            mock_results.append(mock_r)
        evaluator._results = mock_results

        mock_runner = Mock()
        mock_runner.generate_summary.return_value = {}
        evaluator._runner = mock_runner

        evaluator.generate_post_eval_manifest()

        # P0-11: Manifest is in checkpoint-specific directory.
        manifest_path = checkpoint_dir / "manifest.json"
        assert manifest_path.exists()

        loaded = json.loads(manifest_path.read_text())
        assert loaded["experiment_id"] == "fiubench_unlearning_pilot_v1"


# --------------------------------------------------------------------------- #
# Tests – validation
# --------------------------------------------------------------------------- #

class TestPostEvalValidation:
    """Tests for post-eval result validation.

    P0-8: ``validate_results()`` delegates to
    ``self._runner.validate_results()`` (BaselineRunner strict
    validation).  These tests verify the delegation behaviour.
    """

    def test_validation_delegates_to_runner(self) -> None:
        """validate_results() should delegate to _runner.validate_results()."""
        cfg = PostEvalConfig(output_dir="/fake")

        evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
        evaluator.config = cfg
        evaluator._results = []

        from unittest.mock import Mock
        mock_runner = Mock()
        mock_runner.validate_results.return_value = {
            "pass": True,
            "checks": {
                "total_results": 500,
                "errors": 0,
                "families_complete": True,
                "no_duplicate_probes": True,
                "all_scores_finite": True,
            },
        }
        evaluator._runner = mock_runner

        report = evaluator.validate_results()

        mock_runner.validate_results.assert_called_once()
        assert report["pass"] is True
        assert report["checks"]["total_results"] == 500

    def test_validation_propagates_runner_failure(self) -> None:
        """validate_results() should propagate runner validation failure."""
        cfg = PostEvalConfig(output_dir="/fake")

        evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
        evaluator.config = cfg
        evaluator._results = []

        from unittest.mock import Mock
        mock_runner = Mock()
        mock_runner.validate_results.return_value = {
            "pass": False,
            "checks": {"errors": 3},
        }
        evaluator._runner = mock_runner

        report = evaluator.validate_results()

        assert report["pass"] is False
        assert report["checks"]["errors"] == 3

    def test_validation_propagates_runtime_error(self) -> None:
        """validate_results() should propagate RuntimeError from runner."""
        cfg = PostEvalConfig(output_dir="/fake")

        evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
        evaluator.config = cfg
        evaluator._results = []

        from unittest.mock import Mock
        mock_runner = Mock()
        mock_runner.validate_results.side_effect = RuntimeError(
            "P0-8: validation failed: 3 errors found"
        )
        evaluator._runner = mock_runner

        with pytest.raises(RuntimeError, match="P0-8"):
            evaluator.validate_results()


# --------------------------------------------------------------------------- #
# Tests – smoke-aware validation (P0-4, P0-5)
# --------------------------------------------------------------------------- #

class TestSmokeAwareValidation:
    """Tests for smoke_probe_ids parameter in validation methods."""

    def test_pairing_smoke_filters_baseline(self, tmp_path: Path) -> None:
        """With smoke_probe_ids, baseline is filtered to the smoke subset."""
        # Baseline has 20 probes
        baseline_results = [
            _make_result(f"probe_{i:03d}", "direct_visual")
            for i in range(20)
        ]
        baseline_path = tmp_path / "baseline_results.jsonl"
        _write_results(baseline_path, baseline_results)

        # Post-eval has only 5 probes (smoke subset)
        smoke_ids = {f"probe_{i:03d}" for i in range(5)}
        from unittest.mock import Mock
        mock_results = []
        for pid in sorted(smoke_ids):
            mock_r = Mock()
            mock_r.probe_id = pid
            mock_results.append(mock_r)

        cfg = PostEvalConfig(
            baseline_results_path=str(baseline_path),
            output_dir=str(tmp_path / "post_eval"),
        )
        evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
        evaluator.config = cfg
        evaluator._results = mock_results

        # Without smoke IDs: should fail (5 vs 20)
        validation_full = evaluator.validate_against_baseline()
        assert validation_full["passed"] is False

        # With smoke IDs: should pass (5 vs filtered 5)
        validation_smoke = evaluator.validate_against_baseline(
            smoke_probe_ids=smoke_ids,
        )
        assert validation_smoke["passed"] is True
        assert validation_smoke["exact_match"] is True
        assert validation_smoke["baseline_probe_count"] == 5
        assert validation_smoke["post_probe_count"] == 5

    def test_pairing_smoke_missing_probe(self, tmp_path: Path) -> None:
        """Smoke pairing should detect missing probes within the subset."""
        baseline_results = [
            _make_result(f"probe_{i:03d}", "direct_visual")
            for i in range(10)
        ]
        baseline_path = tmp_path / "baseline_results.jsonl"
        _write_results(baseline_path, baseline_results)

        # Post-eval missing probe_002 from the smoke subset
        smoke_ids = {f"probe_{i:03d}" for i in range(5)}
        from unittest.mock import Mock
        mock_results = []
        for pid in sorted(smoke_ids - {"probe_002"}):
            mock_r = Mock()
            mock_r.probe_id = pid
            mock_results.append(mock_r)

        cfg = PostEvalConfig(
            baseline_results_path=str(baseline_path),
            output_dir=str(tmp_path / "post_eval"),
        )
        evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
        evaluator.config = cfg
        evaluator._results = mock_results

        validation = evaluator.validate_against_baseline(
            smoke_probe_ids=smoke_ids,
        )
        assert validation["passed"] is False
        assert "probe_002" in validation["missing_probes"]

    def test_validate_results_passes_smoke_ids(self) -> None:
        """validate_results should pass smoke_probe_ids to the runner."""
        from unittest.mock import Mock
        cfg = PostEvalConfig()
        evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
        evaluator.config = cfg
        evaluator._results = []

        mock_runner = Mock()
        mock_runner.validate_results.return_value = {
            "pass": True, "checks": {},
        }
        evaluator._runner = mock_runner

        smoke_ids = {"p001", "p002", "p003"}
        evaluator.validate_results(smoke_probe_ids=smoke_ids)

        mock_runner.validate_results.assert_called_once_with(
            smoke_probe_ids=smoke_ids,
        )


# --------------------------------------------------------------------------- #
# Tests – Fix 4: smoke-aware manifest (F, G)
# --------------------------------------------------------------------------- #

class TestSmokeAwareManifest:
    """Fix 4: generate_post_eval_manifest must be smoke-aware."""

    def _make_evaluator(self, tmp_path: Path, n_probes: int = 5):
        """Helper: build a mock evaluator with *n_probes* results."""
        baseline_results = [
            _make_result(f"probe_{i:03d}", "direct_visual")
            for i in range(n_probes)
        ]
        baseline_path = tmp_path / "baseline_results.jsonl"
        _write_results(baseline_path, baseline_results)

        output_dir = tmp_path / "post_eval"
        output_dir.mkdir(parents=True, exist_ok=True)
        post_results_path = output_dir / "results.jsonl"
        _write_results(post_results_path, baseline_results)

        cfg = PostEvalConfig(
            checkpoint_path="/fake/checkpoint",
            checkpoint_name="step_050",
            probe_path="/fake/probes.jsonl",
            baseline_results_path=str(baseline_path),
            baseline_manifest_path="/fake/manifest.json",
            output_dir=str(output_dir),
            code_commit="commit789",
        )

        evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
        evaluator.config = cfg
        evaluator._results = []

        from unittest.mock import Mock
        mock_results = []
        for r in baseline_results:
            mock_r = Mock()
            mock_r.probe_id = r["probe_id"]
            mock_r.probe_family = r["probe_family"]
            mock_r.error = None
            mock_r.signed_answer_margin = r["signed_answer_margin"]
            mock_r.correct = r["correct"]
            mock_results.append(mock_r)
        evaluator._results = mock_results

        mock_runner = Mock()
        mock_runner.generate_summary.return_value = {
            "total_probes": n_probes,
            "families": {"direct_visual": {"count": n_probes}},
        }
        evaluator._runner = mock_runner
        return evaluator

    def test_f_smoke_manifest_validation_scope(self, tmp_path: Path) -> None:
        """Test F: smoke manifest forwards smoke_probe_ids and has
        evaluation_scope mode=smoke."""
        evaluator = self._make_evaluator(tmp_path, n_probes=10)
        smoke_ids = {f"probe_{i:03d}" for i in range(10)}

        manifest = evaluator.generate_post_eval_manifest(
            smoke_probe_ids=smoke_ids,
        )

        assert manifest["evaluation_scope"]["mode"] == "smoke"
        assert manifest["evaluation_scope"]["expected_probe_count"] == 10
        assert manifest["validation"]["exact_match"] is True

    def test_g_full_manifest_validation_scope(self, tmp_path: Path) -> None:
        """Test G: full manifest has evaluation_scope mode=full,
        expected_probe_count=500."""
        evaluator = self._make_evaluator(tmp_path, n_probes=5)

        manifest = evaluator.generate_post_eval_manifest()

        assert manifest["evaluation_scope"]["mode"] == "full"
        assert manifest["evaluation_scope"]["expected_probe_count"] == 500
        assert manifest["validation"]["exact_match"] is True
