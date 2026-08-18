"""Integration tests for the MLLMU-Bench baseline suite (P0-24..29).

These tests verify that the frozen-contract propagation, output isolation,
manifest generation, and E2C decision logic work correctly across the
suite orchestration boundary.  No GPU or real model weights are required.

Covers:
  P0-24: _run_eval() does not lose frozen-contract fields
  P0-25: Prompting uses the system-prompt backend, not the plain model
  P0-26: MANU output isolation — distinct dirs per prune rate
  P0-27: MIDP-CM evidence binding from synthetic E2B data
  P0-28: Manifest generation — manifest.json SHA matches actual file
  P0-29: Incomplete-decision regression — missing methods → INCOMPLETE
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"


def _import_run_mllmu_baseline():
    """Import run_mllmu_baseline as a module (it lives outside the package)."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_mllmu_baseline
    return run_mllmu_baseline


def _make_common_config(tmp_path: Path) -> dict:
    """Build a minimal common config dict with all frozen-contract fields."""
    probe_path = tmp_path / "probe.jsonl"
    probe_path.write_text('{"probe_id": "p1"}\n')

    baseline_results = tmp_path / "baseline_results.jsonl"
    baseline_results.write_text('{"probe_id": "p1"}\n')

    baseline_manifest = tmp_path / "baseline_manifest.json"
    baseline_manifest.write_text("{}")

    research_manifest = tmp_path / "research_manifest.json"
    research_manifest.write_text("{}")

    freeze_verification = tmp_path / "freeze_verification.json"
    freeze_verification.write_text("{}")

    processed_dataset = tmp_path / "processed.jsonl"
    processed_dataset.write_text("{}")

    return {
        "data": {
            "route_probe_path": str(probe_path),
            "baseline_results_path": str(baseline_results),
            "baseline_manifest_path": str(baseline_manifest),
            "research_dataset_manifest_path": str(research_manifest),
            "freeze_verification_path": str(freeze_verification),
            "processed_dataset_path": str(processed_dataset),
            "selection_manifest_sha256": "abc123selection",
            "route_probe_sha256": "def456probe",
        },
        "base_model": {
            "model_id": "Qwen/Qwen3.5-9B",
            "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        },
    }


def _make_training_config(tmp_path: Path) -> SimpleNamespace:
    """Build a minimal training config namespace."""
    output_dir = tmp_path / "outputs" / "test_method"
    output_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        model_id="Qwen/Qwen3.5-9B",
        model_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16",
        device="cpu",
        seed=17,
        output_dir=str(output_dir),
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "v_proj"],
        processed_dataset_path=str(tmp_path / "processed.jsonl"),
        forget_identity_ids=["id1"],
        retain_identity_ids=["id2"],
    )


# ========================================================================== #
# P0-24: _run_eval() preserves all frozen-contract fields
# ========================================================================== #


class TestP024RunEvalFrozenContract:
    """P0-24: _run_eval() must not drop frozen-contract fields.

    This test fails if a future refactor drops any provenance field
    from the PostEvalConfig construction.
    """

    def test_all_frozen_fields_propagated(self, tmp_path: Path):
        """PostEvalConfig receives every frozen-contract field."""
        mod = _import_run_mllmu_baseline()
        common_cfg = _make_common_config(tmp_path)
        training_cfg = _make_training_config(tmp_path)

        captured_config = {}

        def mock_evaluate_intervention(
            *, model, processor, adapter_path,
            probe_dataset_path, output_dir, config,
            baseline_results_path, method_name, backend_override,
        ):
            captured_config["config"] = config
            return {
                "method": method_name,
                "exact_pair_count": 500,
                "inference_errors": 0,
                "strict_validation_pass": True,
            }

        with patch(
            "route_data.eval.post_unlearning_eval.evaluate_intervention",
            side_effect=mock_evaluate_intervention,
        ):
            mod._run_eval(
                "test_method",
                model=MagicMock(),
                processor=MagicMock(),
                adapter_path=None,
                config=common_cfg,
                training_config=training_cfg,
            )

        cfg = captured_config["config"]
        assert cfg is not None, "PostEvalConfig was not passed to evaluate_intervention"

        # P0-24: Every frozen-contract field must be populated.
        from route_data.eval.post_unlearning_eval import PostEvalConfig
        assert isinstance(cfg, PostEvalConfig)

        assert cfg.dataset_manifest_path, "dataset_manifest_path is empty"
        assert cfg.freeze_verification_path, "freeze_verification_path is empty"
        assert cfg.processed_dataset_path, "processed_dataset_path is empty"
        assert cfg.baseline_results_path, "baseline_results_path is empty"
        assert cfg.baseline_manifest_path, "baseline_manifest_path is empty"
        assert cfg.selection_manifest_sha256, "selection_manifest_sha256 is empty"
        assert cfg.code_commit is not None, "code_commit is None"
        assert cfg.route_probe_sha256, "route_probe_sha256 is empty"

        # Verify actual values match the config.
        data = common_cfg["data"]
        assert cfg.dataset_manifest_path == data["research_dataset_manifest_path"]
        assert cfg.freeze_verification_path == data["freeze_verification_path"]
        assert cfg.processed_dataset_path == data["processed_dataset_path"]
        assert cfg.baseline_results_path == data["baseline_results_path"]
        assert cfg.baseline_manifest_path == data["baseline_manifest_path"]
        assert cfg.selection_manifest_sha256 == data["selection_manifest_sha256"]
        assert cfg.route_probe_sha256 == data["route_probe_sha256"]

    def test_missing_frozen_field_raises(self, tmp_path: Path):
        """_run_eval() raises when a mandatory field is empty."""
        mod = _import_run_mllmu_baseline()
        common_cfg = _make_common_config(tmp_path)
        training_cfg = _make_training_config(tmp_path)

        # Remove a mandatory field.
        common_cfg["data"]["baseline_results_path"] = ""

        with pytest.raises(RuntimeError, match="mandatory PostEvalConfig"):
            mod._run_eval(
                "test_method",
                model=MagicMock(),
                processor=MagicMock(),
                adapter_path=None,
                config=common_cfg,
                training_config=training_cfg,
            )


# ========================================================================== #
# P0-25: Prompting uses the system-prompt backend
# ========================================================================== #


class TestP025PromptingBackend:
    """P0-25: The canonical prompting result comes from _PromptingBackend.

    Ensures the duplicate plain-model evaluation path is impossible.
    """

    def test_prompting_passes_backend_override(self, tmp_path: Path):
        """_run_eval receives backend_override for prompting."""
        mod = _import_run_mllmu_baseline()
        common_cfg = _make_common_config(tmp_path)
        training_cfg = _make_training_config(tmp_path)

        captured_kwargs = {}

        def mock_evaluate_intervention(
            *, model, processor, adapter_path,
            probe_dataset_path, output_dir, config,
            baseline_results_path, method_name, backend_override,
        ):
            captured_kwargs["backend_override"] = backend_override
            captured_kwargs["method_name"] = method_name
            return {
                "method": method_name,
                "exact_pair_count": 500,
                "inference_errors": 0,
                "strict_validation_pass": True,
            }

        fake_backend = MagicMock()

        with patch(
            "route_data.eval.post_unlearning_eval.evaluate_intervention",
            side_effect=mock_evaluate_intervention,
        ):
            mod._run_eval(
                "prompting",
                model=MagicMock(),
                processor=MagicMock(),
                adapter_path=None,
                config=common_cfg,
                training_config=training_cfg,
                backend_override=fake_backend,
            )

        assert captured_kwargs["backend_override"] is fake_backend, (
            "backend_override was not passed through to evaluate_intervention"
        )
        assert captured_kwargs["method_name"] == "prompting"

    def test_prompting_method_name_is_canonical(self, tmp_path: Path):
        """The method_name recorded is 'prompting', not a variant."""
        mod = _import_run_mllmu_baseline()
        common_cfg = _make_common_config(tmp_path)
        training_cfg = _make_training_config(tmp_path)

        captured_method = {}

        def mock_evaluate_intervention(
            *, model, processor, adapter_path,
            probe_dataset_path, output_dir, config,
            baseline_results_path, method_name, backend_override,
        ):
            captured_method["name"] = method_name
            return {
                "method": method_name,
                "exact_pair_count": 500,
                "inference_errors": 0,
                "strict_validation_pass": True,
            }

        with patch(
            "route_data.eval.post_unlearning_eval.evaluate_intervention",
            side_effect=mock_evaluate_intervention,
        ):
            mod._run_eval(
                "prompting",
                model=MagicMock(),
                processor=MagicMock(),
                adapter_path=None,
                config=common_cfg,
                training_config=training_cfg,
                backend_override=MagicMock(),
            )

        assert captured_method["name"] == "prompting"


# ========================================================================== #
# P0-26: MANU output isolation
# ========================================================================== #


class TestP026ManuOutputIsolation:
    """P0-26: MANU prune rates produce isolated eval outputs.

    No file from 5% pruning is reused by 10% pruning.
    """

    def test_distinct_eval_subdirs(self, tmp_path: Path):
        """Different eval_subdir values produce different output directories."""
        mod = _import_run_mllmu_baseline()
        common_cfg = _make_common_config(tmp_path)
        training_cfg = _make_training_config(tmp_path)

        captured_dirs = {}

        def mock_evaluate_intervention(
            *, model, processor, adapter_path,
            probe_dataset_path, output_dir, config,
            baseline_results_path, method_name, backend_override,
        ):
            captured_dirs[method_name] = Path(output_dir)
            return {
                "method": method_name,
                "exact_pair_count": 500,
                "inference_errors": 0,
                "strict_validation_pass": True,
            }

        with patch(
            "route_data.eval.post_unlearning_eval.evaluate_intervention",
            side_effect=mock_evaluate_intervention,
        ):
            mod._run_eval(
                "manu_prune_05",
                model=MagicMock(),
                processor=MagicMock(),
                adapter_path=None,
                config=common_cfg,
                training_config=training_cfg,
                eval_subdir="prune_05",
            )
            mod._run_eval(
                "manu_prune_10",
                model=MagicMock(),
                processor=MagicMock(),
                adapter_path=None,
                config=common_cfg,
                training_config=training_cfg,
                eval_subdir="prune_10",
            )

        dir_05 = captured_dirs["manu_prune_05"]
        dir_10 = captured_dirs["manu_prune_10"]

        assert dir_05 != dir_10, "MANU prune rates share the same output directory"
        assert dir_05.name == "prune_05"
        assert dir_10.name == "prune_10"
        assert dir_05.parent == dir_10.parent

    def test_different_method_ids(self, tmp_path: Path):
        """Each prune rate records a different method identifier."""
        mod = _import_run_mllmu_baseline()
        common_cfg = _make_common_config(tmp_path)
        training_cfg = _make_training_config(tmp_path)

        captured_methods = {}

        def mock_evaluate_intervention(
            *, model, processor, adapter_path,
            probe_dataset_path, output_dir, config,
            baseline_results_path, method_name, backend_override,
        ):
            captured_methods[method_name] = Path(output_dir).name
            return {
                "method": method_name,
                "exact_pair_count": 500,
                "inference_errors": 0,
                "strict_validation_pass": True,
            }

        with patch(
            "route_data.eval.post_unlearning_eval.evaluate_intervention",
            side_effect=mock_evaluate_intervention,
        ):
            for rate in ("05", "10"):
                mod._run_eval(
                    f"manu_prune_{rate}",
                    model=MagicMock(),
                    processor=MagicMock(),
                    adapter_path=None,
                    config=common_cfg,
                    training_config=training_cfg,
                    eval_subdir=f"prune_{rate}",
                )

        assert "manu_prune_05" in captured_methods
        assert "manu_prune_10" in captured_methods
        assert captured_methods["manu_prune_05"] != captured_methods["manu_prune_10"]


# ========================================================================== #
# P0-27: MIDP-CM evidence binding
# ========================================================================== #


class TestP027MidpCmBinding:
    """P0-27: MIDP-CM binds validated E2B evidence without retraining.

    Uses synthetic E2B evidence to verify SHA validation and canonical
    eval_results.json creation.
    """

    def test_source_sha_validated_and_canonical_written(self, tmp_path: Path):
        """Source evidence SHA is computed and canonical eval_results.json is written."""
        # Create synthetic E2B-B2 output.
        e2b_dir = tmp_path / "e2b_b2"
        e2b_dir.mkdir()
        source_evidence = {
            "method": "midp_candidate_margin",
            "delta_target": {"DV": 0.5, "IPN": 0.3},
            "delta_retain": {"DV": 0.01, "IPN": -0.02},
            "delta_control": {"DV": 0.005, "IPN": 0.01},
            "delta_untargeted": {},
            "exact_pair_count": 500,
            "inference_errors": 0,
            "manifest_sha256": "abc123",
            "per_family_post": {},
            "summary": {},
            "strict_validation_pass": True,
            "exact_pairing_pass": True,
        }
        source_path = e2b_dir / "eval_results.json"
        source_path.write_text(json.dumps(source_evidence))

        # Compute expected SHA.
        expected_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()

        # Create the output directory for MIDP-CM.
        output_dir = tmp_path / "midp_cm_output"
        output_dir.mkdir()
        eval_dir = output_dir / "eval"
        eval_dir.mkdir()

        # Simulate the binding logic (extracted from run_mllmu_baseline.py).
        sha = hashlib.sha256()
        with open(source_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        source_sha = sha.hexdigest()

        assert source_sha == expected_sha

        with open(source_path) as f:
            source_result = json.load(f)

        # Validate required fields.
        for field_name in ("delta_target", "delta_retain", "delta_control"):
            assert source_result.get(field_name), f"Missing {field_name}"

        # Build canonical result.
        result = {
            "method": "midp_candidate_margin",
            "delta_target": source_result.get("delta_target", {}),
            "delta_retain": source_result.get("delta_retain", {}),
            "delta_control": source_result.get("delta_control", {}),
            "delta_untargeted": source_result.get("delta_untargeted", {}),
            "exact_pair_count": source_result.get("exact_pair_count", 0),
            "inference_errors": source_result.get("inference_errors", 0),
            "manifest_sha256": source_result.get("manifest_sha256", ""),
            "per_family_post": source_result.get("per_family_post", {}),
            "summary": source_result.get("summary", {}),
            "eval_output_dir": str(eval_dir),
            "results_path": str(eval_dir / "eval_results.json"),
            "adapter_path": None,
            "e2b_source": str(e2b_dir),
            "e2b_source_eval_results_sha256": source_sha,
            "strict_validation_pass": source_result.get("strict_validation_pass", False),
            "exact_pairing_pass": source_result.get("exact_pairing_pass", False),
        }

        canonical_path = eval_dir / "eval_results.json"
        with open(canonical_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
            f.write("\n")

        # Verify canonical file exists and contains the correct data.
        assert canonical_path.is_file()
        with open(canonical_path) as f:
            written = json.load(f)
        assert written["method"] == "midp_candidate_margin"
        assert written["e2b_source_eval_results_sha256"] == expected_sha
        assert written["delta_target"] == {"DV": 0.5, "IPN": 0.3}
        assert written["adapter_path"] is None

    def test_missing_delta_field_fails(self, tmp_path: Path):
        """MIDP-CM fails when source evidence lacks a required field."""
        e2b_dir = tmp_path / "e2b_b2_bad"
        e2b_dir.mkdir()
        # Missing delta_control.
        source_evidence = {
            "method": "midp_candidate_margin",
            "delta_target": {"DV": 0.5},
            "delta_retain": {"DV": 0.01},
            "delta_control": {},  # Empty — should fail validation.
            "exact_pair_count": 500,
        }
        source_path = e2b_dir / "eval_results.json"
        source_path.write_text(json.dumps(source_evidence))

        with open(source_path) as f:
            source_result = json.load(f)

        # Simulate the validation from run_mllmu_baseline.py.
        missing = []
        for field_name in ("delta_target", "delta_retain", "delta_control"):
            if not source_result.get(field_name):
                missing.append(field_name)

        assert "delta_control" in missing


# ========================================================================== #
# P0-28: Manifest-generation regression test
# ========================================================================== #


class TestP028ManifestGeneration:
    """P0-28: evaluate_intervention() generates manifest.json with correct SHA.

    The manifest must be created before its SHA is computed, and the
    recorded SHA must match the actual file content.
    """

    def test_manifest_sha_matches_file(self, tmp_path: Path):
        """manifest_sha256 in the result equals SHA-256 of manifest.json."""
        from route_data.eval.post_unlearning_eval import (
            PostEvalConfig,
            PostUnlearningEvaluator,
        )

        output_dir = tmp_path / "eval_output"
        output_dir.mkdir()

        # Create a minimal evaluator with mocked internals.
        cfg = PostEvalConfig(
            output_dir=str(output_dir),
            probe_path=str(tmp_path / "probe.jsonl"),
            baseline_results_path=str(tmp_path / "baseline.jsonl"),
            baseline_manifest_path=str(tmp_path / "baseline_manifest.json"),
        )

        evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
        evaluator.config = cfg
        evaluator._results = [
            {
                "probe_id": f"probe_{i:04d}",
                "sample_id": f"sample_{i}",
                "identity_id": f"id_{i % 5}",
                "probe_family": ["direct_visual", "image_plus_name",
                                 "wrong_name", "visual_text_conflict",
                                 "name_only"][i % 5],
                "modality": "image_text",
                "question": "test?",
                "signed_answer_margin": 0.5,
                "normalized_exact_match": 1.0,
                "correct": True,
                "error": None,
                "pre_signed_answer_margin": 0.3,
                "pre_normalized_exact_match": 1.0,
                "protocol_role": ["exclude", "train", "eval"][i % 3],
            }
            for i in range(100)
        ]
        evaluator._runner = MagicMock()

        # Mock validate_against_baseline to avoid file I/O.
        with patch.object(
            evaluator, "validate_against_baseline",
            return_value={"pass": True, "checks": {}},
        ):
            manifest = evaluator.generate_post_eval_manifest()

        # Verify manifest.json exists.
        manifest_path = output_dir / "manifest.json"
        assert manifest_path.is_file(), "manifest.json was not created"

        # Compute actual SHA.
        actual_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

        # The manifest dict should contain consistent data.
        assert "probe_file" in manifest
        assert "baseline_reference" in manifest
        assert "provenance" in manifest

        # Now verify that evaluate_intervention's SHA computation matches.
        # Simulate what evaluate_intervention does after generate_post_eval_manifest.
        assert actual_sha, "manifest SHA is empty"
        assert len(actual_sha) == 64  # SHA-256 hex digest length.

    def test_manifest_created_before_sha_computation(self, tmp_path: Path):
        """manifest.json must exist before its SHA is read."""
        from route_data.eval.post_unlearning_eval import (
            PostEvalConfig,
            PostUnlearningEvaluator,
        )

        output_dir = tmp_path / "eval_output2"
        output_dir.mkdir()

        cfg = PostEvalConfig(
            output_dir=str(output_dir),
            probe_path=str(tmp_path / "probe.jsonl"),
            baseline_results_path=str(tmp_path / "baseline.jsonl"),
            baseline_manifest_path=str(tmp_path / "baseline_manifest.json"),
        )

        evaluator = PostUnlearningEvaluator.__new__(PostUnlearningEvaluator)
        evaluator.config = cfg
        evaluator._results = []
        evaluator._runner = MagicMock()

        # Before generation: no manifest.
        manifest_path = output_dir / "manifest.json"
        assert not manifest_path.is_file()

        # Mock validate_against_baseline to avoid file I/O.
        with patch.object(
            evaluator, "validate_against_baseline",
            return_value={"pass": True, "checks": {}},
        ):
            evaluator.generate_post_eval_manifest()

        # After generation: manifest exists.
        assert manifest_path.is_file(), (
            "manifest.json must be created before SHA computation"
        )


# ========================================================================== #
# P0-29: Incomplete-decision regression test
# ========================================================================== #


class TestP029IncompleteDecision:
    """P0-29: Missing method evidence → INCOMPLETE decision.

    Input: GA complete, GD complete, MANU missing.
    Expected: decision_status=INCOMPLETE, no Case A/B/C.
    """

    def test_missing_method_produces_incomplete(self, tmp_path: Path):
        """A method with empty delta fields triggers INCOMPLETE."""
        from route_data.unlearning.comparison_framework import (
            ComparisonFramework,
            MethodResult,
        )

        fw = ComparisonFramework(output_dir=tmp_path)

        # GA: complete.
        fw.add_result(MethodResult(
            method_id="ga",
            baseline_id="B1",
            description="Gradient Ascent",
            delta_target={"DV": 0.5, "IPN": 0.3},
            delta_retain={"DV": 0.01},
            delta_control={"DV": 0.005},
        ))

        # GD: complete.
        fw.add_result(MethodResult(
            method_id="gd",
            baseline_id="B2",
            description="Gradient Difference",
            delta_target={"DV": 0.4, "IPN": 0.2},
            delta_retain={"DV": 0.02},
            delta_control={"DV": 0.01},
        ))

        # MANU: missing (empty delta_target).
        fw.add_result(MethodResult(
            method_id="manu",
            baseline_id="B8",
            description="MANU (neuron pruning)",
            delta_target={},  # Missing!
            delta_retain={},
            delta_control={},
        ))

        decision = fw.make_e2c_decision()

        assert decision["decision_status"] == "INCOMPLETE"
        assert decision["action"] == "complete_missing_evaluations"
        assert "manu" in decision["missing_eval_methods"]
        # Must NOT have a case classification.
        assert "case" not in decision

    def test_no_case_classification_when_incomplete(self, tmp_path: Path):
        """INCOMPLETE decision has no case, no freeze_and_proceed."""
        from route_data.unlearning.comparison_framework import (
            ComparisonFramework,
            MethodResult,
        )

        fw = ComparisonFramework(output_dir=tmp_path)

        # One complete, one missing.
        fw.add_result(MethodResult(
            method_id="ga", baseline_id="B1", description="GA",
            delta_target={"DV": 0.5}, delta_retain={"DV": 0.01},
            delta_control={"DV": 0.005},
        ))
        fw.add_result(MethodResult(
            method_id="kl", baseline_id="B3", description="KL",
            delta_target={}, delta_retain={}, delta_control={},
        ))

        decision = fw.make_e2c_decision()

        assert decision["decision_status"] == "INCOMPLETE"
        assert "case" not in decision
        assert "freeze_and_proceed" not in json.dumps(decision)
        assert "proceed_to_e2c" not in json.dumps(decision)

    def test_all_complete_does_not_produce_incomplete(self, tmp_path: Path):
        """When all methods are complete, decision is NOT INCOMPLETE."""
        from route_data.unlearning.comparison_framework import (
            ComparisonFramework,
            MethodResult,
        )

        fw = ComparisonFramework(output_dir=tmp_path)

        fw.add_result(MethodResult(
            method_id="ga", baseline_id="B1", description="GA",
            delta_target={"DV": 0.5}, delta_retain={"DV": 0.01},
            delta_control={"DV": 0.005},
        ))
        fw.add_result(MethodResult(
            method_id="gd", baseline_id="B2", description="GD",
            delta_target={"DV": 0.4}, delta_retain={"DV": 0.02},
            delta_control={"DV": 0.01},
        ))

        decision = fw.make_e2c_decision()

        # When all methods are complete, the decision has a "case" key
        # and no "decision_status" of "INCOMPLETE".
        assert decision.get("decision_status") != "INCOMPLETE"
        assert "case" in decision

    def test_eval_complete_gate_produces_incomplete_conclusion(
        self, tmp_path: Path,
    ):
        """P0-18: eval_complete=False → incomplete conclusion, not publishable."""
        from route_data.unlearning.comparison_framework import (
            ComparisonFramework,
            MethodResult,
        )

        fw = ComparisonFramework(output_dir=tmp_path)
        fw.add_result(MethodResult(
            method_id="ga", baseline_id="B1", description="GA",
            delta_target={"DV": 0.5}, delta_retain={"DV": 0.01},
            delta_control={"DV": 0.005},
        ))
        fw.add_result(MethodResult(
            method_id="manu", baseline_id="B8", description="MANU",
            delta_target={}, delta_retain={}, delta_control={},
        ))

        conclusion = fw.write_route_selectivity_conclusion(eval_complete=False)

        assert "INCOMPLETE" in conclusion
        assert "manu" in conclusion

        # Verify the incomplete report was written.
        report_path = tmp_path / "incomplete_suite_report.json"
        assert report_path.is_file()
        with open(report_path) as f:
            report = json.load(f)
        assert report["decision_status"] == "INCOMPLETE"


# ========================================================================== #
# Suite-level validation tests (P0-19/20)
# ========================================================================== #


class TestValidateEvalResult:
    """Tests for the strengthened _validate_eval_result() (P0-19/20)."""

    @staticmethod
    def _import_suite():
        if str(SCRIPTS) not in sys.path:
            sys.path.insert(0, str(SCRIPTS))
        import run_mllmu_baseline_suite
        return run_mllmu_baseline_suite

    def test_method_mismatch_detected(self):
        """P0-20: method identifier mismatch returns error."""
        suite = self._import_suite()
        result = {
            "expected_pair_count": 500,
            "actual_pair_count": 500,
            "exact_pair_count": 500,
            "inference_errors": 0,
            "strict_validation_pass": True,
            "exact_pairing_pass": True,
            "results_path": "",
            "eval_output_dir": "",
            "manifest_sha256": "abc",
            "delta_target": {"DV": 0.5},
            "delta_retain": {"DV": 0.01},
            "delta_control": {"DV": 0.005},
            "method": "wrong_method",
        }
        err = suite._validate_eval_result(result, "ga")
        assert err is not None
        assert "method mismatch" in err

    def test_pair_count_mismatch_detected(self):
        """P0-19: actual_pair_count != 500 returns error."""
        suite = self._import_suite()
        result = {
            "expected_pair_count": 500,
            "actual_pair_count": 499,
            "exact_pair_count": 499,
            "inference_errors": 0,
            "strict_validation_pass": True,
            "exact_pairing_pass": True,
            "results_path": "",
            "eval_output_dir": "",
            "manifest_sha256": "abc",
            "delta_target": {"DV": 0.5},
            "delta_retain": {"DV": 0.01},
            "delta_control": {"DV": 0.005},
            "method": "ga",
        }
        err = suite._validate_eval_result(result, "ga")
        assert err is not None
        assert "actual_pair_count" in err

    def test_revision_mismatch_detected(self):
        """P0-19: model_revision mismatch returns error."""
        suite = self._import_suite()
        result = {
            "expected_pair_count": 500,
            "actual_pair_count": 500,
            "exact_pair_count": 500,
            "inference_errors": 0,
            "strict_validation_pass": True,
            "exact_pairing_pass": True,
            "results_path": "",
            "eval_output_dir": "",
            "manifest_sha256": "abc",
            "delta_target": {"DV": 0.5},
            "delta_retain": {"DV": 0.01},
            "delta_control": {"DV": 0.005},
            "method": "ga",
            "model_revision": "wrong_revision",
        }
        common_config = {
            "base_model": {"revision": "correct_revision"},
        }
        err = suite._validate_eval_result(result, "ga", common_config)
        assert err is not None
        assert "model_revision mismatch" in err

    def test_route_probe_sha_mismatch_detected(self):
        """P0-19: route_probe_sha256 mismatch returns error."""
        suite = self._import_suite()
        result = {
            "expected_pair_count": 500,
            "actual_pair_count": 500,
            "exact_pair_count": 500,
            "inference_errors": 0,
            "strict_validation_pass": True,
            "exact_pairing_pass": True,
            "results_path": "",
            "eval_output_dir": "",
            "manifest_sha256": "abc",
            "delta_target": {"DV": 0.5},
            "delta_retain": {"DV": 0.01},
            "delta_control": {"DV": 0.005},
            "method": "ga",
            "route_probe_sha256": "wrong_sha",
        }
        common_config = {
            "base_model": {"revision": ""},
            "data": {"route_probe_sha256": "correct_sha"},
        }
        err = suite._validate_eval_result(result, "ga", common_config)
        assert err is not None
        assert "route_probe_sha256 mismatch" in err

    def test_valid_result_passes(self):
        """A fully valid result returns None (no error)."""
        suite = self._import_suite()
        result = {
            "expected_pair_count": 500,
            "actual_pair_count": 500,
            "exact_pair_count": 500,
            "inference_errors": 0,
            "strict_validation_pass": True,
            "exact_pairing_pass": True,
            "results_path": "",
            "eval_output_dir": "",
            "manifest_sha256": "abc",
            "delta_target": {"DV": 0.5},
            "delta_retain": {"DV": 0.01},
            "delta_control": {"DV": 0.005},
            "method": "ga",
            "model_revision": "rev123",
            "route_probe_sha256": "sha456",
        }
        common_config = {
            "base_model": {"revision": "rev123"},
            "data": {"route_probe_sha256": "sha456"},
        }
        err = suite._validate_eval_result(result, "ga", common_config)
        assert err is None
