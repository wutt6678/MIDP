"""Unit tests for the post-unlearning evaluator (Stage 3, Commit 3).

These tests verify the structural invariants of the post-evaluation
pipeline: exact probe matching, manifest generation, and validation.
No GPU or real model weights are required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

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
        post_results_path = output_dir / "results.jsonl"
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

        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()

        loaded = json.loads(manifest_path.read_text())
        assert loaded["experiment_id"] == "fiubench_unlearning_pilot_v1"


# --------------------------------------------------------------------------- #
# Tests – validation
# --------------------------------------------------------------------------- #

class TestPostEvalValidation:
    """Tests for post-eval result validation."""

    def test_validation_passes_with_valid_results(self) -> None:
        """Validation should pass with 500 valid results."""
        cfg = PostEvalConfig(output_dir="/fake")

        evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
        evaluator.config = cfg

        from unittest.mock import Mock
        mock_results = []
        families = ["direct_visual", "image_plus_name", "wrong_name",
                    "visual_text_conflict", "name_only"]
        for i in range(500):
            mock_r = Mock()
            mock_r.probe_id = f"probe_{i:03d}"
            mock_r.probe_family = families[i % 5]
            mock_r.error = None
            mock_r.signed_answer_margin = 1.0
            mock_results.append(mock_r)
        evaluator._results = mock_results

        report = evaluator.validate_results()

        assert report["passed"] is True
        assert report["total_results"] == 500
        assert report["errors"] == 0
        assert report["families_complete"] is True
        assert report["no_duplicate_probes"] is True
        assert report["all_scores_finite"] is True

    def test_validation_fails_with_errors(self) -> None:
        """Validation should fail if any results have errors."""
        cfg = PostEvalConfig(output_dir="/fake")

        evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
        evaluator.config = cfg

        from unittest.mock import Mock
        mock_results = []
        families = ["direct_visual", "image_plus_name", "wrong_name",
                    "visual_text_conflict", "name_only"]
        for i in range(500):
            mock_r = Mock()
            mock_r.probe_id = f"probe_{i:03d}"
            mock_r.probe_family = families[i % 5]
            mock_r.error = "some error" if i == 0 else None
            mock_r.signed_answer_margin = 1.0
            mock_results.append(mock_r)
        evaluator._results = mock_results

        report = evaluator.validate_results()

        assert report["passed"] is False
        assert report["errors"] == 1

    def test_validation_fails_with_wrong_count(self) -> None:
        """Validation should fail if not exactly 500 results."""
        cfg = PostEvalConfig(output_dir="/fake")

        evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
        evaluator.config = cfg

        from unittest.mock import Mock
        mock_results = []
        for i in range(499):  # One short
            mock_r = Mock()
            mock_r.probe_id = f"probe_{i:03d}"
            mock_r.probe_family = "direct_visual"
            mock_r.error = None
            mock_r.signed_answer_margin = 1.0
            mock_results.append(mock_r)
        evaluator._results = mock_results

        report = evaluator.validate_results()

        assert report["passed"] is False
        assert report["expected_probe_count"] is False
