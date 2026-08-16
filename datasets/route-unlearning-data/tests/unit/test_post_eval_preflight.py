"""Tests for post-eval research preflight wiring (P0-8, P1-4).

Verifies:
- validate_research_preflight() is called before run_all()
- If preflight raises, run_all() is never called
- Config validation rejects legacy fields and missing paths
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from route_data.eval.post_unlearning_eval import (
    PostEvalConfig,
    PostUnlearningEvaluator,
)
from route_data.eval.run_pilot import validate_experiment_config

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_evaluator() -> PostUnlearningEvaluator:
    """Create a PostUnlearningEvaluator with a mocked runner."""
    cfg = PostEvalConfig(
        checkpoint_path="/fake/checkpoint",
        checkpoint_name="step_050",
        output_dir="/tmp/test_post_eval",
    )
    evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
    evaluator.config = cfg
    evaluator.backend = MagicMock()
    evaluator.model_config = MagicMock()
    evaluator._results = []

    mock_runner = Mock()
    mock_runner.run_all.return_value = []
    evaluator._runner = mock_runner

    return evaluator


def _valid_config() -> dict:
    """Return a minimal valid experiment config."""
    return {
        "experiment_id": "test",
        "base_model": {
            "model_id": "test/model",
            "revision": "abc123",
            "dtype": "float32",
            "model_config_path": "configs/models/test.yaml",
        },
        "dataset": {
            "route_probe_path": "/fake/probes.jsonl",
            "processed_dataset_path": "/fake/processed.jsonl",
            "research_manifest_path": "/fake/research_manifest.json",
            "freeze_verification_path": "/fake/freeze_verification.json",
        },
        "method": {
            "name": "lora_targeted_candidate_margin",
            "hyperparameters": {
                "num_optimizer_steps": 50,
            },
        },
    }


# --------------------------------------------------------------------------- #
# Tests: P0-8 — preflight called before inference
# --------------------------------------------------------------------------- #

class TestResearchPreflight:
    """Tests for research preflight wiring."""

    def test_preflight_called_before_run_all(self) -> None:
        """validate_research_preflight() must be called before run_all()."""
        evaluator = _make_evaluator()

        evaluator.run_evaluation()

        evaluator._runner.validate_research_preflight.assert_called_once()
        evaluator._runner.run_all.assert_called_once()

        # Verify ordering: preflight before run_all
        calls = evaluator._runner.method_calls
        preflight_idx = None
        run_all_idx = None
        for i, call in enumerate(calls):
            if call[0] == "validate_research_preflight":
                preflight_idx = i
            elif call[0] == "run_all":
                run_all_idx = i
        assert preflight_idx is not None
        assert run_all_idx is not None
        assert preflight_idx < run_all_idx

    def test_preflight_failure_prevents_run_all(self) -> None:
        """If preflight raises, run_all() must never be called."""
        evaluator = _make_evaluator()
        evaluator._runner.validate_research_preflight.side_effect = RuntimeError(
            "Preflight failed: freeze SHA mismatch"
        )

        with pytest.raises(RuntimeError, match="Preflight failed"):
            evaluator.run_evaluation()

        evaluator._runner.validate_research_preflight.assert_called_once()
        evaluator._runner.run_all.assert_not_called()


# --------------------------------------------------------------------------- #
# Tests: P0-7 — config validation
# --------------------------------------------------------------------------- #

class TestConfigValidation:
    """Tests for validate_experiment_config."""

    def test_valid_config_passes(self) -> None:
        """A valid config should not raise."""
        cfg = _valid_config()
        validate_experiment_config(cfg)  # Should not raise

    def test_legacy_method_name_fails(self) -> None:
        """Legacy method name 'lora_targeted_update' should fail."""
        cfg = _valid_config()
        cfg["method"]["name"] = "lora_targeted_update"

        with pytest.raises(ValueError, match="lora_targeted_update.*legacy"):
            validate_experiment_config(cfg)

    def test_wrong_method_name_fails(self) -> None:
        """Wrong method name should fail."""
        cfg = _valid_config()
        cfg["method"]["name"] = "some_other_method"

        with pytest.raises(ValueError, match="lora_targeted_candidate_margin"):
            validate_experiment_config(cfg)

    def test_legacy_num_steps_fails(self) -> None:
        """Legacy 'num_steps' without 'num_optimizer_steps' should fail."""
        cfg = _valid_config()
        cfg["method"]["hyperparameters"] = {"num_steps": 50}

        with pytest.raises(ValueError, match="num_steps.*legacy"):
            validate_experiment_config(cfg)

    def test_missing_research_manifest_path_fails(self) -> None:
        """Missing dataset.research_manifest_path should fail."""
        cfg = _valid_config()
        del cfg["dataset"]["research_manifest_path"]

        with pytest.raises(ValueError, match="research_manifest_path"):
            validate_experiment_config(cfg)

    def test_missing_freeze_verification_path_fails(self) -> None:
        """Missing dataset.freeze_verification_path should fail."""
        cfg = _valid_config()
        del cfg["dataset"]["freeze_verification_path"]

        with pytest.raises(ValueError, match="freeze_verification_path"):
            validate_experiment_config(cfg)

    def test_missing_model_config_path_fails(self) -> None:
        """Missing base_model.model_config_path should fail."""
        cfg = _valid_config()
        del cfg["base_model"]["model_config_path"]

        with pytest.raises(ValueError, match="model_config_path"):
            validate_experiment_config(cfg)

    def test_missing_revision_fails(self) -> None:
        """Missing base_model.revision should fail."""
        cfg = _valid_config()
        del cfg["base_model"]["revision"]

        with pytest.raises(ValueError, match="revision"):
            validate_experiment_config(cfg)

    def test_empty_revision_fails(self) -> None:
        """Empty base_model.revision should fail."""
        cfg = _valid_config()
        cfg["base_model"]["revision"] = ""

        with pytest.raises(ValueError, match="revision"):
            validate_experiment_config(cfg)
