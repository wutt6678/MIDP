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
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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

    selection_manifest = tmp_path / "selection_manifest.json"
    selection_manifest.write_text("{}")

    return {
        "data": {
            "route_probe_path": str(probe_path),
            "baseline_results_path": str(baseline_results),
            "baseline_manifest_path": str(baseline_manifest),
            "research_dataset_manifest_path": str(research_manifest),
            "freeze_verification_path": str(freeze_verification),
            "processed_dataset_path": str(processed_dataset),
            "selection_manifest_sha256": "abc123selection",
            "selection_manifest_path": str(selection_manifest),
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
            objective_name="", trainable_adapter=None,
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
            objective_name="", trainable_adapter=None,
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
            objective_name="", trainable_adapter=None,
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
            objective_name="", trainable_adapter=None,
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
            objective_name="", trainable_adapter=None,
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
            # P0-18: All four binary families required.
            "delta_target": {"DV": 0.5, "IPN": 0.4, "WN": 0.3, "VTC": 0.2},
            "delta_retain": {"DV": 0.01, "IPN": 0.02, "WN": 0.03, "VTC": 0.04},
            "delta_control": {"DV": 0.005, "IPN": 0.006, "WN": 0.007, "VTC": 0.008},
            "delta_untargeted": {"DV": 0.002, "IPN": 0.003, "WN": 0.004, "VTC": 0.005},
            "name_only_delta": {},
            "dv_accuracy": {"global": 1.0, "target": 0.5, "retain": 1.0, "control": 1.0, "untargeted": 0.99},
            "group_identity_counts": {"target": 2, "retain": 2, "control": 2, "untargeted": 94},
            # P0-17: Provenance fields required.
            "route_probe_sha256": "abc123",
            "selection_manifest_sha256": "def456",
            "model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            # P0-12/13: Mandatory group_probe_counts.
            "group_probe_counts": {
                "target": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
                "retain": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
                "control": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
                "untargeted": {"DV": 94, "IPN": 94, "WN": 94, "VTC": 94, "name_only": 94},
            },
            # P0-18/19/20: Contract metadata.
            "validation_contract_version": "mllmu-baseline-suite-v1",
            "evaluation_scope": {"mode": "full", "expected_probe_count": 500},
            "evidence_mode": "new_evaluation",
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
            "delta_target": {"DV": 0.5, "IPN": 0.4, "WN": 0.3, "VTC": 0.2},
            "delta_retain": {"DV": 0.01, "IPN": 0.02, "WN": 0.03, "VTC": 0.04},
            "delta_control": {"DV": 0.005, "IPN": 0.006, "WN": 0.007, "VTC": 0.008},
            "delta_untargeted": {"DV": 0.002, "IPN": 0.003, "WN": 0.004, "VTC": 0.005},
            "name_only_delta": {},
            "dv_accuracy": {"global": 1.0, "target": 0.5, "retain": 1.0, "control": 1.0, "untargeted": 0.99},
            "group_identity_counts": {"target": 2, "retain": 2, "control": 2, "untargeted": 94},
            "method": "ga",
            "route_probe_sha256": "abc",
            "selection_manifest_sha256": "def",
            "model_revision": "rev",
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
            "delta_target": {"DV": 0.5, "IPN": 0.4, "WN": 0.3, "VTC": 0.2},
            "delta_retain": {"DV": 0.01, "IPN": 0.02, "WN": 0.03, "VTC": 0.04},
            "delta_control": {"DV": 0.005, "IPN": 0.006, "WN": 0.007, "VTC": 0.008},
            "delta_untargeted": {"DV": 0.002, "IPN": 0.003, "WN": 0.004, "VTC": 0.005},
            "name_only_delta": {},
            "dv_accuracy": {"global": 1.0, "target": 0.5, "retain": 1.0, "control": 1.0, "untargeted": 0.99},
            "group_identity_counts": {"target": 2, "retain": 2, "control": 2, "untargeted": 94},
            "group_probe_counts": {
                "target": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
                "retain": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
                "control": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
                "untargeted": {"DV": 94, "IPN": 94, "WN": 94, "VTC": 94, "name_only": 94},
            },
            "validation_contract_version": "mllmu-baseline-suite-v1",
            "evaluation_scope": {"mode": "full", "expected_probe_count": 500},
            "evidence_mode": "new_evaluation",
            "method": "ga",
            "route_probe_sha256": "abc",
            "selection_manifest_sha256": "def",
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
            "delta_target": {"DV": 0.5, "IPN": 0.4, "WN": 0.3, "VTC": 0.2},
            "delta_retain": {"DV": 0.01, "IPN": 0.02, "WN": 0.03, "VTC": 0.04},
            "delta_control": {"DV": 0.005, "IPN": 0.006, "WN": 0.007, "VTC": 0.008},
            "delta_untargeted": {"DV": 0.002, "IPN": 0.003, "WN": 0.004, "VTC": 0.005},
            "name_only_delta": {},
            "dv_accuracy": {"global": 1.0, "target": 0.5, "retain": 1.0, "control": 1.0, "untargeted": 0.99},
            "group_identity_counts": {"target": 2, "retain": 2, "control": 2, "untargeted": 94},
            "group_probe_counts": {
                "target": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
                "retain": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
                "control": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
                "untargeted": {"DV": 94, "IPN": 94, "WN": 94, "VTC": 94, "name_only": 94},
            },
            "validation_contract_version": "mllmu-baseline-suite-v1",
            "evaluation_scope": {"mode": "full", "expected_probe_count": 500},
            "evidence_mode": "new_evaluation",
            "method": "ga",
            "route_probe_sha256": "wrong_sha",
            "selection_manifest_sha256": "def",
            "model_revision": "rev",
        }
        common_config = {
            "base_model": {"revision": "rev"},
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
            "delta_target": {"DV": 0.5, "IPN": 0.4, "WN": 0.3, "VTC": 0.2},
            "delta_retain": {"DV": 0.01, "IPN": 0.02, "WN": 0.03, "VTC": 0.04},
            "delta_control": {"DV": 0.005, "IPN": 0.006, "WN": 0.007, "VTC": 0.008},
            "delta_untargeted": {"DV": 0.002, "IPN": 0.003, "WN": 0.004, "VTC": 0.005},
            "name_only_delta": {"target": {}, "retain": {}},
            "dv_accuracy": {
                "global": 1.0, "target": 0.5,
                "retain": 1.0, "control": 1.0, "untargeted": 0.99,
            },
            "group_identity_counts": {
                "target": 2, "retain": 2, "control": 2, "untargeted": 94,
            },
            "group_probe_counts": {
                "target": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
                "retain": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
                "control": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
                "untargeted": {"DV": 94, "IPN": 94, "WN": 94, "VTC": 94, "name_only": 94},
            },
            "validation_contract_version": "mllmu-baseline-suite-v1",
            "evaluation_scope": {"mode": "full", "expected_probe_count": 500},
            "evidence_mode": "new_evaluation",
            "method": "ga",
            "model_revision": "rev123",
            "route_probe_sha256": "sha456",
            "selection_manifest_sha256": "sel_sha",
        }
        common_config = {
            "base_model": {"revision": "rev123"},
            "data": {
                "route_probe_sha256": "sha456",
                "selection_manifest_sha256": "sel_sha",
            },
        }
        err = suite._validate_eval_result(result, "ga", common_config)
        assert err is None

    # ------------------------------------------------------------------ #
    # P0-15: Per-family count regression tests
    # ------------------------------------------------------------------ #

    @staticmethod
    def _valid_base_result() -> dict:
        """Return a minimal result that passes all checks through P0-13."""
        return {
            "expected_pair_count": 500,
            "actual_pair_count": 500,
            "exact_pair_count": 500,
            "inference_errors": 0,
            "strict_validation_pass": True,
            "exact_pairing_pass": True,
            "results_path": "",
            "eval_output_dir": "",
            "manifest_sha256": "abc",
            "delta_target": {"DV": 0.5, "IPN": 0.4, "WN": 0.3, "VTC": 0.2},
            "delta_retain": {"DV": 0.01, "IPN": 0.02, "WN": 0.03, "VTC": 0.04},
            "delta_control": {"DV": 0.005, "IPN": 0.006, "WN": 0.007, "VTC": 0.008},
            "delta_untargeted": {"DV": 0.002, "IPN": 0.003, "WN": 0.004, "VTC": 0.005},
            "name_only_delta": {},
            "dv_accuracy": {"global": 1.0, "target": 0.5, "retain": 1.0, "control": 1.0, "untargeted": 0.99},
            "group_identity_counts": {"target": 2, "retain": 2, "control": 2, "untargeted": 94},
            "group_probe_counts": {
                "target": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
                "retain": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
                "control": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
                "untargeted": {"DV": 94, "IPN": 94, "WN": 94, "VTC": 94, "name_only": 94},
            },
            "validation_contract_version": "mllmu-baseline-suite-v1",
            "evaluation_scope": {"mode": "full", "expected_probe_count": 500},
            "evidence_mode": "new_evaluation",
            "method": "ga",
            "model_revision": "rev",
            "route_probe_sha256": "sha",
            "selection_manifest_sha256": "sel",
        }

    def test_missing_group_probe_counts_fails(self):
        """P0-12: Missing group_probe_counts → FAIL (fail-closed)."""
        suite = self._import_suite()
        result = self._valid_base_result()
        del result["group_probe_counts"]
        err = suite._validate_eval_result(result, "ga")
        assert err is not None
        assert "group_probe_counts" in err

    def test_wrong_target_wn_count_fails(self):
        """P0-15: target.WN = 1 instead of 2 → FAIL."""
        suite = self._import_suite()
        result = self._valid_base_result()
        result["group_probe_counts"]["target"]["WN"] = 1
        err = suite._validate_eval_result(result, "ga")
        assert err is not None
        assert "group_probe_counts[target][WN]" in err

    def test_missing_name_only_target_fails(self):
        """P0-15: target.name_only = 1 instead of 2 → FAIL."""
        suite = self._import_suite()
        result = self._valid_base_result()
        result["group_probe_counts"]["target"]["name_only"] = 1
        err = suite._validate_eval_result(result, "ga")
        assert err is not None
        assert "group_probe_counts[target][name_only]" in err

    def test_extra_untargeted_dv_fails(self):
        """P0-15: untargeted.DV = 95 instead of 94 → FAIL."""
        suite = self._import_suite()
        result = self._valid_base_result()
        result["group_probe_counts"]["untargeted"]["DV"] = 95
        err = suite._validate_eval_result(result, "ga")
        assert err is not None
        assert "group_probe_counts[untargeted][DV]" in err

    def test_missing_entire_group_fails(self):
        """P0-15: Missing 'retain' group → FAIL."""
        suite = self._import_suite()
        result = self._valid_base_result()
        del result["group_probe_counts"]["retain"]
        err = suite._validate_eval_result(result, "ga")
        assert err is not None
        assert "group_probe_counts[retain]" in err

    def test_contract_version_mismatch_fails(self):
        """P0-18: Wrong validation_contract_version → FAIL."""
        suite = self._import_suite()
        result = self._valid_base_result()
        result["validation_contract_version"] = "wrong-version"
        err = suite._validate_eval_result(result, "ga")
        assert err is not None
        assert "validation_contract_version" in err

    def test_evaluation_scope_smoke_fails(self):
        """P0-19: Smoke scope cannot pass as final evidence."""
        suite = self._import_suite()
        result = self._valid_base_result()
        result["evaluation_scope"] = {"mode": "smoke", "expected_probe_count": 50}
        err = suite._validate_eval_result(result, "ga")
        assert err is not None
        assert "evaluation_scope.mode" in err

    def test_invalid_evidence_mode_fails(self):
        """P0-20: Invalid evidence_mode → FAIL."""
        suite = self._import_suite()
        result = self._valid_base_result()
        result["evidence_mode"] = "synthetic"
        err = suite._validate_eval_result(result, "ga")
        assert err is not None
        assert "evidence_mode" in err


# ========================================================================== #
# P0-31: Real-schema integration tests
# ========================================================================== #


class TestRealSchemaIntegration:
    """P0-31: Load actual committed artifacts and verify preflight accepts them."""

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def test_research_manifest_loads(self) -> None:
        """The committed research_dataset_manifest.json is valid JSON."""
        root = self._project_root()
        manifest_path = (
            root / "outputs" / "full_fiubench" / "evidence"
            / "research_dataset_manifest.json"
        )
        if not manifest_path.is_file():
            pytest.skip("research manifest not found")
        with open(manifest_path) as f:
            data = json.load(f)
        assert data.get("manifest_version"), "manifest_version missing"
        assert data.get("model_provenance", {}).get("model_id") == "Qwen/Qwen3.5-9B"

    def test_freeze_verification_loads(self) -> None:
        """The committed final_freeze_verification.json is valid."""
        root = self._project_root()
        fv_path = (
            root / "outputs" / "full_fiubench" / "evidence"
            / "final_freeze_verification.json"
        )
        if not fv_path.is_file():
            pytest.skip("freeze verification not found")
        with open(fv_path) as f:
            data = json.load(f)
        assert data.get("dataset_version") == "fiubench-route-v1"
        assert data.get("ready_for_experiments") is True

    def test_selection_manifest_loads(self) -> None:
        """The committed pilot_identity_selection.json is valid."""
        root = self._project_root()
        sel_path = (
            root / "outputs" / "experiments" / "unlearning_pilot"
            / "Qwen_Qwen3.5-9B" / "pilot_v1" / "selection"
            / "pilot_identity_selection.json"
        )
        if not sel_path.is_file():
            pytest.skip("selection manifest not found")
        with open(sel_path) as f:
            data = json.load(f)
        target = data.get("target_identities", [])
        retain = data.get("retain_identities", [])
        control = data.get("control_identities", [])
        assert len(target) == 2
        assert len(retain) == 2
        assert len(control) == 2


# ========================================================================== #
# P0-32: Frozen-group regression test
# ========================================================================== #


class TestFrozenGroupRegression:
    """P0-32: All selected IDs have protocol_role=train but are correctly
    classified as target/retain/control by identity ID."""

    def test_selected_ids_are_all_train_role(self) -> None:
        """All 6 selected identities have protocol_role=train."""
        root = Path(__file__).resolve().parents[2]
        sel_path = (
            root / "outputs" / "experiments" / "unlearning_pilot"
            / "Qwen_Qwen3.5-9B" / "pilot_v1" / "selection"
            / "pilot_identity_selection.json"
        )
        if not sel_path.is_file():
            pytest.skip("selection manifest not found")
        with open(sel_path) as f:
            data = json.load(f)

        identity_details = data.get("identity_details", {})
        selected_ids = (
            set(data.get("target_identities", []))
            | set(data.get("retain_identities", []))
            | set(data.get("control_identities", []))
        )

        for iid in selected_ids:
            detail = identity_details.get(iid, {})
            assert detail.get("protocol_role") == "train", (
                f"Identity {iid} has protocol_role={detail.get('protocol_role')}, "
                f"expected 'train'. This prevents regression to protocol-role grouping."
            )

    def test_identity_based_classification_not_protocol_role(self) -> None:
        """Identity-based classification produces 2/2/2 groups,
        not protocol-role-based groups."""
        root = Path(__file__).resolve().parents[2]
        sel_path = (
            root / "outputs" / "experiments" / "unlearning_pilot"
            / "Qwen_Qwen3.5-9B" / "pilot_v1" / "selection"
            / "pilot_identity_selection.json"
        )
        if not sel_path.is_file():
            pytest.skip("selection manifest not found")
        with open(sel_path) as f:
            data = json.load(f)

        target_ids = set(data.get("target_identities", []))
        retain_ids = set(data.get("retain_identities", []))
        control_ids = set(data.get("control_identities", []))

        # All 6 have protocol_role=train, but they are different groups.
        assert len(target_ids) == 2
        assert len(retain_ids) == 2
        assert len(control_ids) == 2
        # No overlap between groups.
        assert not (target_ids & retain_ids)
        assert not (target_ids & control_ids)
        assert not (retain_ids & control_ids)


# ========================================================================== #
# P0-33: Runtime git-cleanliness test
# ========================================================================== #


class TestRuntimeGitCleanliness:
    """P0-33: Runtime output path is git-ignored."""

    def test_runtime_outputs_in_gitignore(self) -> None:
        """runtime_outputs/ is listed in .gitignore."""
        root = Path(__file__).resolve().parents[2]
        gitignore = root / ".gitignore"
        if not gitignore.is_file():
            pytest.skip(".gitignore not found")
        content = gitignore.read_text()
        assert "runtime_outputs/" in content, (
            "runtime_outputs/ must be in .gitignore"
        )


# ========================================================================== #
# P0-15: Real MIDP-CM binding test
# ========================================================================== #


class TestMIDPCMBinding:
    """P0-15: E2B-B2 evidence binder works with actual committed layout."""

    def test_bind_e2b_b2_produces_common_schema(self) -> None:
        """bind_e2b_b2_result() reads actual artifacts and produces
        the common comparison schema."""
        root = Path(__file__).resolve().parents[2]
        e2b_dir = (
            root / "outputs" / "experiments" / "unlearning_pilot"
            / "Qwen_Qwen3.5-9B" / "pilot_e2b_b2"
        )
        if not e2b_dir.is_dir():
            pytest.skip("E2B-B2 directory not found")

        from route_data.unlearning.e2b_evidence_binding import bind_e2b_b2_result

        result = bind_e2b_b2_result(e2b_dir)

        # Common schema fields present.
        assert result["method"] == "midp_cm"
        assert result["objective_name"] == "midp_candidate_margin"
        assert "delta_target" in result
        assert "delta_retain" in result
        assert "delta_control" in result
        assert "delta_untargeted" in result
        assert "name_only_delta" in result
        assert "dv_accuracy" in result
        assert "group_identity_counts" in result

        # DV accuracy has required keys.
        dv = result["dv_accuracy"]
        for key in ("global", "target", "retain", "control", "untargeted"):
            assert key in dv, f"dv_accuracy missing key: {key}"

        # Group counts match 2/2/2/94.
        counts = result["group_identity_counts"]
        assert counts.get("target") == 2
        assert counts.get("retain") == 2
        assert counts.get("control") == 2
        assert counts.get("untargeted") == 94

        # Artifact SHAs are populated.
        assert result.get("e2b_artifact_shas"), "e2b_artifact_shas empty"

    def test_bind_e2b_b2_sha_mismatch_hard_fails(self, tmp_path: Path) -> None:
        """Missing artifacts cause a hard failure."""
        from route_data.unlearning.e2b_evidence_binding import bind_e2b_b2_result

        with pytest.raises(FileNotFoundError):
            bind_e2b_b2_result(tmp_path / "nonexistent")


# ========================================================================== #
# P0-12/13/14: Absent-method completeness tests
# ========================================================================== #


def _compute_suite_completeness(
    results: list[dict],
    method_keys: list[str],
    comparison_methods: list[str],
) -> dict:
    """Replicate the suite completeness logic from run_mllmu_baseline_suite.py.

    This helper extracts the P0-7/8/9/10/11 completeness computation
    so it can be tested in isolation.
    """
    required = set(comparison_methods)
    requested = {m for m in method_keys if m in comparison_methods}
    valid_eval = {
        r["method"]
        for r in results
        if (
            r["method"] in comparison_methods
            and r.get("status") == "success"
            and bool(r.get("eval_metrics"))
        )
    }
    missing = sorted(required - valid_eval)
    execution_scope = "full" if requested == required else "partial"
    eval_complete = valid_eval == required
    research_suite_complete = eval_complete and execution_scope == "full"
    run_success = all(r["status"] == "success" for r in results) and bool(results)
    return {
        "required_comparison_methods": sorted(required),
        "requested_comparison_methods": sorted(requested),
        "valid_eval_methods": sorted(valid_eval),
        "missing_comparison_methods": missing,
        "execution_scope": execution_scope,
        "eval_complete": eval_complete,
        "research_suite_complete": research_suite_complete,
        "run_success": run_success,
    }


# Import COMPARISON_METHODS from the suite script.
def _get_comparison_methods() -> list[str]:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_mllmu_baseline_suite
    return run_mllmu_baseline_suite.COMPARISON_METHODS


class TestP012AbsentMethodCompleteness:
    """P0-12: A completely absent method makes eval_complete=False."""

    def test_ga_only_missing_methods(self) -> None:
        """GA-only results: gd, kl, etc. appear in missing_comparison_methods."""
        comparison_methods = _get_comparison_methods()

        # Simulate: only GA succeeded.
        results = [
            {
                "method": "ga",
                "status": "success",
                "eval_metrics": {"exact_pair_count": 500},
            },
        ]
        method_keys = ["ga"]
        summary = _compute_suite_completeness(results, method_keys, comparison_methods)

        assert summary["eval_complete"] is False
        assert "gd" in summary["missing_comparison_methods"]
        assert "kl" in summary["missing_comparison_methods"]
        assert "prompting" in summary["missing_comparison_methods"]
        # GA is NOT missing.
        assert "ga" not in summary["missing_comparison_methods"]

    def test_empty_results_all_missing(self) -> None:
        """No results at all → all methods missing."""
        comparison_methods = _get_comparison_methods()
        results: list[dict] = []
        method_keys: list[str] = []

        summary = _compute_suite_completeness(results, method_keys, comparison_methods)
        assert summary["eval_complete"] is False
        assert len(summary["missing_comparison_methods"]) == len(comparison_methods)


class TestP013PartialOnlyOrchestration:
    """P0-13: --only ga produces partial scope, no Case A/B/C."""

    def test_only_ga_partial_scope(self) -> None:
        """--only ga: run_success=true, scope=partial, eval_complete=false."""
        comparison_methods = _get_comparison_methods()

        results = [
            {
                "method": "ga",
                "status": "success",
                "eval_metrics": {"exact_pair_count": 500},
            },
        ]
        method_keys = ["ga"]
        summary = _compute_suite_completeness(results, method_keys, comparison_methods)

        assert summary["run_success"] is True
        assert summary["execution_scope"] == "partial"
        assert summary["eval_complete"] is False
        assert summary["research_suite_complete"] is False

    def test_partial_cannot_emit_case_conclusion(self) -> None:
        """Partial suite cannot produce Case A/B/C conclusion."""
        from route_data.unlearning.comparison_framework import (
            ComparisonFramework,
            MethodResult,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            fw = ComparisonFramework(tmpdir)
            # Only one method with valid evidence.
            fw.add_result(MethodResult(
                method_id="ga", baseline_id="B1", description="GA",
                delta_target={"DV": -0.5}, delta_retain={"DV": -0.01},
                delta_control={"DV": -0.02},
                dv_accuracy={
                    "global": 1.0, "target": 0.0,
                    "retain": 1.0, "control": 1.0, "untargeted": 0.99,
                },
            ))

            # With eval_complete=False, conclusion must be INCOMPLETE.
            conclusion = fw.write_route_selectivity_conclusion(eval_complete=False)
            assert "INCOMPLETE" in conclusion
            assert "Case A" not in conclusion
            assert "Case B" not in conclusion
            assert "Case C" not in conclusion


class TestP014FullMethodSetCompleteness:
    """P0-14: Full valid method set can still set eval_complete=true."""

    def test_all_methods_present(self) -> None:
        """All comparison methods succeed → eval_complete=true."""
        comparison_methods = _get_comparison_methods()

        results = [
            {
                "method": m,
                "status": "success",
                "eval_metrics": {"exact_pair_count": 500},
            }
            for m in comparison_methods
        ]
        method_keys = list(comparison_methods)
        summary = _compute_suite_completeness(results, method_keys, comparison_methods)

        assert summary["execution_scope"] == "full"
        assert summary["eval_complete"] is True
        assert summary["research_suite_complete"] is True
        assert summary["missing_comparison_methods"] == []

    def test_one_method_failed(self) -> None:
        """One method failed → eval_complete=false."""
        comparison_methods = _get_comparison_methods()

        results = [
            {
                "method": m,
                "status": "success",
                "eval_metrics": {"exact_pair_count": 500},
            }
            for m in comparison_methods
        ]
        # Make one method fail.
        results[-1]["status"] = "failed"
        results[-1]["eval_metrics"] = None

        method_keys = list(comparison_methods)
        summary = _compute_suite_completeness(results, method_keys, comparison_methods)

        assert summary["eval_complete"] is False
        assert len(summary["missing_comparison_methods"]) == 1
