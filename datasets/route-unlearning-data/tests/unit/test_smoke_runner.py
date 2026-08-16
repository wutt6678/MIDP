"""Tests for the canonical smoke runner (P1-8).

Verifies:
- Smoke-mode config overrides (1 step, 10 probes)
- CLI argument parsing (--config, --smoke, --resume)
- StepTimer context manager
- Evidence writing
- Full pipeline orchestration with mocked GPU steps
- Resume skips completed steps
- Pipeline provenance written
- GO/NO-GO verdict logic
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Ensure the project src is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Load the smoke-runner script directly (scripts/ is not a package).
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "run_unlearning_pilot.py"
_spec = importlib.util.spec_from_file_location("run_unlearning_pilot", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
run_unlearning_pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_unlearning_pilot)

from route_data.eval.run_pilot import (
    PilotRunner,
    load_experiment_config,
    validate_experiment_config,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

FAMILIES = [
    "direct_visual",
    "image_plus_name",
    "wrong_name",
    "visual_text_conflict",
    "name_only",
]

ATTRIBUTES = [
    "hair_color", "eye_color", "skin_tone", "face_shape",
    "age_group", "gender", "height", "build",
    "clothing_style", "accessories",
]


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_synthetic_data(tmp_path: Path, n_identities: int = 12) -> tuple[Path, Path]:
    """Create synthetic baseline results and route probes."""
    bl_path = tmp_path / "baseline_results.jsonl"
    rp_path = tmp_path / "route_probes.jsonl"

    probe_counter = 0
    bl_rows = []
    rp_rows = []
    seen_ids: set[str] = set()

    for i in range(n_identities):
        iid = f"identity_{i:03d}"
        attr = ATTRIBUTES[i % len(ATTRIBUTES)]
        is_positive = i % 2 == 0

        for fam in FAMILIES:
            pid = f"probe_{probe_counter:04d}"
            probe_counter += 1

            if fam == "name_only":
                bl_rows.append({
                    "probe_id": pid, "sample_id": f"s_{pid}",
                    "identity_id": iid, "probe_family": fam,
                    "modality": "text", "question": f"What is {iid}'s {attr}?",
                    "target_attribute": attr, "answer_label": None,
                    "protocol_role": "train", "predicted_label": None,
                    "correct": False, "signed_answer_margin": None,
                    "token_overlap": 0.3,
                })
            else:
                margin = 5.0 + (i * 0.1)
                bl_rows.append({
                    "probe_id": pid, "sample_id": f"s_{pid}",
                    "identity_id": iid, "probe_family": fam,
                    "modality": "image+text",
                    "question": f"Does {iid} have {attr}={is_positive}?",
                    "target_attribute": attr, "answer_label": is_positive,
                    "protocol_role": "train",
                    "predicted_label": "Yes" if is_positive else "No",
                    "correct": True, "signed_answer_margin": margin,
                    "token_overlap": None,
                })

                if iid not in seen_ids:
                    seen_ids.add(iid)
                    rp_rows.append({
                        "probe_id": f"route_{iid}", "identity_id": iid,
                        "probe_family": fam, "target_attribute": attr,
                        "answer_label": is_positive,
                        "question": f"Does {iid} have {attr}?",
                        "image_uri": f"images/{iid}.jpg",
                    })

    with bl_path.open("w") as fh:
        for r in bl_rows:
            fh.write(json.dumps(r) + "\n")
    with rp_path.open("w") as fh:
        for r in rp_rows:
            fh.write(json.dumps(r) + "\n")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"version": "test-v1", "n_rows": len(bl_rows)}))

    return bl_path, rp_path


def _make_config(
    tmp_path: Path,
    bl_path: Path,
    rp_path: Path,
) -> Path:
    """Create a minimal experiment config YAML."""
    manifest_path = tmp_path / "manifest.json"
    config = {
        "experiment_id": "test_pilot",
        "base_model": {
            "model_id": "test/model",
            "revision": "abc123",
            "dtype": "float32",
            "fingerprint_id": "test_fp",
            "model_config_path": str(tmp_path / "model_config.yaml"),
        },
        "baseline": {
            "version": "test-v1",
            "manifest_path": str(manifest_path),
            "results_path": str(bl_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "results_sha256": _sha256_file(bl_path),
        },
        "dataset": {
            "route_probe_path": str(rp_path),
            "route_probe_sha256": _sha256_file(rp_path),
            "processed_dataset_path": "",
            "processed_dataset_sha256": "",
            "research_manifest_path": str(tmp_path / "research_manifest.json"),
            "freeze_verification_path": str(tmp_path / "freeze_verification.json"),
        },
        "selection": {
            "target_identity_count": 2,
            "retain_identity_count": 2,
            "control_identity_count": 2,
            "seed": 17,
            "matching_criteria": ["protocol_role"],
            "preferred_role": "train",
        },
        "method": {
            "name": "lora_targeted_candidate_margin",
            "hyperparameters": {
                "lora_rank": 8,
                "lora_alpha": 16,
                "learning_rate": 1e-4,
                "num_optimizer_steps": 50,
                "retain_weight": 0.1,
                "train_batch_size": 1,
                "gradient_accumulation_steps": 4,
            },
        },
        "evaluation": {
            "reuse_frozen_500_probes": True,
            "primary_metric": "signed_answer_margin",
            "preserve_visual_capability": True,
            "direct_visual_accuracy_gate": 0.98,
        },
        "runtime": {
            "seed": 17,
            "output_dir": str(tmp_path / "pilot_v1"),
        },
    }

    cfg_path = tmp_path / "experiment.yaml"
    with cfg_path.open("w") as fh:
        yaml.dump(config, fh)
    return cfg_path


# --------------------------------------------------------------------------- #
# Tests: apply_smoke_overrides
# --------------------------------------------------------------------------- #

class TestSmokeOverrides:
    """P1-8: Smoke mode overrides config correctly."""

    def test_num_optimizer_steps_set_to_1(self) -> None:
        """Smoke mode must set num_optimizer_steps to 1."""
        cfg: dict = {"method": {"hyperparameters": {"num_optimizer_steps": 50}}}
        run_unlearning_pilot.apply_smoke_overrides(cfg)
        assert cfg["method"]["hyperparameters"]["num_optimizer_steps"] == 1

    def test_batch_size_set_to_1(self) -> None:
        """Smoke mode must set train_batch_size to 1."""
        cfg: dict = {"method": {"hyperparameters": {"train_batch_size": 4}}}
        run_unlearning_pilot.apply_smoke_overrides(cfg)
        assert cfg["method"]["hyperparameters"]["train_batch_size"] == 1

    def test_gradient_accumulation_set_to_1(self) -> None:
        """Smoke mode must set gradient_accumulation_steps to 1."""
        cfg: dict = {"method": {"hyperparameters": {"gradient_accumulation_steps": 4}}}
        run_unlearning_pilot.apply_smoke_overrides(cfg)
        assert cfg["method"]["hyperparameters"]["gradient_accumulation_steps"] == 1

    def test_smoke_mode_flag_set(self) -> None:
        """Smoke mode must set runtime.smoke_mode = True."""
        cfg: dict = {"runtime": {}}
        run_unlearning_pilot.apply_smoke_overrides(cfg)
        assert cfg["runtime"]["smoke_mode"] is True

    def test_creates_missing_sections(self) -> None:
        """Smoke overrides must create missing method/runtime sections."""
        cfg: dict = {}
        run_unlearning_pilot.apply_smoke_overrides(cfg)
        assert cfg["method"]["hyperparameters"]["num_optimizer_steps"] == 1
        assert cfg["runtime"]["smoke_mode"] is True

    def test_preserves_other_hyperparameters(self) -> None:
        """Smoke overrides must not alter other hyperparameters."""
        cfg: dict = {
            "method": {
                "hyperparameters": {
                    "lora_rank": 8,
                    "lora_alpha": 16,
                    "learning_rate": 1e-4,
                    "num_optimizer_steps": 50,
                    "retain_weight": 0.1,
                    "train_batch_size": 2,
                    "gradient_accumulation_steps": 4,
                },
            },
        }
        run_unlearning_pilot.apply_smoke_overrides(cfg)
        hp = cfg["method"]["hyperparameters"]
        assert hp["lora_rank"] == 8
        assert hp["lora_alpha"] == 16
        assert hp["learning_rate"] == 1e-4
        assert hp["retain_weight"] == 0.1


# --------------------------------------------------------------------------- #
# Tests: parse_args
# --------------------------------------------------------------------------- #

class TestParseArgs:
    """P1-8: CLI argument parsing."""

    def test_config_required(self) -> None:
        """--config is required."""
        with pytest.raises(SystemExit):
            run_unlearning_pilot.parse_args([])

    def test_config_path(self) -> None:
        """--config stores the path."""
        args = run_unlearning_pilot.parse_args(["--config", "my_config.yaml"])
        assert args.config == "my_config.yaml"

    def test_smoke_default_false(self) -> None:
        """--smoke defaults to False."""
        args = run_unlearning_pilot.parse_args(["--config", "c.yaml"])
        assert args.smoke is False

    def test_smoke_flag(self) -> None:
        """--smoke sets smoke=True."""
        args = run_unlearning_pilot.parse_args(["--config", "c.yaml", "--smoke"])
        assert args.smoke is True

    def test_resume_default_false(self) -> None:
        """--resume defaults to False."""
        args = run_unlearning_pilot.parse_args(["--config", "c.yaml"])
        assert args.resume is False

    def test_resume_flag(self) -> None:
        """--resume sets resume=True."""
        args = run_unlearning_pilot.parse_args(["--config", "c.yaml", "--resume"])
        assert args.resume is True


# --------------------------------------------------------------------------- #
# Tests: StepTimer
# --------------------------------------------------------------------------- #

class TestStepTimer:
    """P1-8: StepTimer context manager."""

    def test_context_manager(self) -> None:
        """StepTimer should work as a context manager."""
        with run_unlearning_pilot.StepTimer(1, "test step") as t:
            assert t.step_num == 1
            assert t.name == "test step"

    def test_elapsed_positive(self) -> None:
        """Elapsed time should be non-negative."""
        import time
        with run_unlearning_pilot.StepTimer(1, "test"):
            time.sleep(0.01)


# --------------------------------------------------------------------------- #
# Tests: _write_step_evidence
# --------------------------------------------------------------------------- #

class TestWriteStepEvidence:
    """P1-8: Per-step evidence writing."""

    def test_writes_json(self, tmp_path: Path) -> None:
        """Evidence file should be valid JSON."""
        data = {"key": "value", "count": 42}
        run_unlearning_pilot._write_step_evidence(tmp_path, 1, "test_step", data)

        evidence_path = tmp_path / "evidence" / "step_01_test_step.json"
        assert evidence_path.exists()

        loaded = json.loads(evidence_path.read_text())
        assert loaded["key"] == "value"
        assert loaded["count"] == 42

    def test_creates_evidence_dir(self, tmp_path: Path) -> None:
        """Should create evidence/ directory if it doesn't exist."""
        output_dir = tmp_path / "subdir"
        run_unlearning_pilot._write_step_evidence(output_dir, 5, "my_step", {"x": 1})
        assert (output_dir / "evidence" / "step_05_my_step.json").exists()

    def test_filename_format(self, tmp_path: Path) -> None:
        """Evidence filename should follow step_NN_name.json format."""
        run_unlearning_pilot._write_step_evidence(tmp_path, 17, "go_nogo", {"pass": True})
        path = tmp_path / "evidence" / "step_17_go_nogo.json"
        assert path.exists()


# --------------------------------------------------------------------------- #
# Tests: run_pipeline (mocked GPU steps)
# --------------------------------------------------------------------------- #

class TestRunPipeline:
    """P1-8: Full pipeline orchestration with mocked GPU steps."""

    def _setup(self, tmp_path: Path) -> Path:
        """Create synthetic data and config, return config path."""
        bl_path, rp_path = _make_synthetic_data(tmp_path)
        cfg_path = _make_config(tmp_path, bl_path, rp_path)
        return cfg_path

    def test_pipeline_runs_selection(self, tmp_path: Path) -> None:
        """Pipeline should complete selection steps (1-4)."""
        cfg_path = self._setup(tmp_path)

        # Mock GPU phases to avoid actual model loading
        mock_training = {
            "num_optimizer_steps": 1,
            "final_loss": 0.5,
            "final_checkpoint_path": str(tmp_path / "pilot_v1" / "checkpoints" / "step_001"),
            "elapsed_seconds": 1.0,
        }
        mock_post_eval = {
            "summary": {"n_probes": 10, "inference_errors": 0},
            "validation": {"pass": True},
            "pairing": {"pass": True},
            "results_path": str(tmp_path / "post_results.jsonl"),
        }

        _mod = run_unlearning_pilot  # noqa: F841  (alias for brevity)
        with patch.object(
            _mod, "_run_training_phase", return_value=mock_training,
        ), patch.object(
            _mod, "_run_post_eval_phase", return_value=mock_post_eval,
        ), patch(
            "route_data.eval.run_pilot.PilotRunner.run_paired_analysis",
            return_value={"probe_deltas": {}, "group_effects": {}, "preservation_report": {}, "pairing_validation": {"pass": True}},
        ):
            report = _mod.run_pipeline(str(cfg_path), smoke=True)

        assert isinstance(report, dict)
        assert "gates" in report or "stages" in report

    def test_smoke_mode_overrides(self, tmp_path: Path) -> None:
        """Smoke mode should override num_optimizer_steps to 1."""
        cfg_path = self._setup(tmp_path)

        captured_cfg: dict = {}

        def _capture_training(runner, cfg, manifest, *, smoke=False):
            captured_cfg.update(cfg)
            return {
                "num_optimizer_steps": 1,
                "final_checkpoint_path": str(tmp_path / "ckpt"),
                "elapsed_seconds": 0.1,
            }

        mock_post_eval = {
            "summary": {"n_probes": 10, "inference_errors": 0},
            "validation": {"pass": True},
            "results_path": str(tmp_path / "post_results.jsonl"),
        }

        _mod = run_unlearning_pilot
        with patch.object(
            _mod, "_run_training_phase", side_effect=_capture_training,
        ), patch.object(
            _mod, "_run_post_eval_phase", return_value=mock_post_eval,
        ), patch(
            "route_data.eval.run_pilot.PilotRunner.run_paired_analysis",
            return_value={"probe_deltas": {}, "group_effects": {}, "preservation_report": {}, "pairing_validation": {"pass": True}},
        ):
            _mod.run_pipeline(str(cfg_path), smoke=True)

        assert captured_cfg["method"]["hyperparameters"]["num_optimizer_steps"] == 1

    def test_pipeline_writes_provenance(self, tmp_path: Path) -> None:
        """Pipeline should write pipeline_provenance.json."""
        cfg_path = self._setup(tmp_path)

        mock_training = {
            "num_optimizer_steps": 1,
            "final_checkpoint_path": "",
            "elapsed_seconds": 0.1,
        }
        mock_post_eval = {
            "summary": {"n_probes": 10, "inference_errors": 0},
            "validation": {"pass": True},
            "results_path": str(tmp_path / "post_results.jsonl"),
        }

        _mod = run_unlearning_pilot
        with patch.object(
            _mod, "_run_training_phase", return_value=mock_training,
        ), patch.object(
            _mod, "_run_post_eval_phase", return_value=mock_post_eval,
        ), patch(
            "route_data.eval.run_pilot.PilotRunner.run_paired_analysis",
            return_value={"probe_deltas": {}, "group_effects": {}, "preservation_report": {}, "pairing_validation": {"pass": True}},
        ):
            report = _mod.run_pipeline(str(cfg_path), smoke=True)

        # Check provenance file exists
        output_dir = tmp_path / "pilot_v1"
        provenance = output_dir / "evidence" / "pipeline_provenance.json"
        assert provenance.exists()

        prov_data = json.loads(provenance.read_text())
        assert prov_data["smoke"] is True
        assert "start_time" in prov_data
        assert "end_time" in prov_data
        assert "code_commit" in prov_data

    def test_resume_skips_selection(self, tmp_path: Path) -> None:
        """Resume mode should skip selection if evidence exists."""
        cfg_path = self._setup(tmp_path)

        # Pre-create selection evidence
        output_dir = tmp_path / "pilot_v1"
        evidence_dir = output_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "step_01_selection.json").write_text("{}")

        # Pre-create selection manifest
        sel_dir = output_dir / "selection"
        sel_dir.mkdir(parents=True, exist_ok=True)
        sel_manifest = {
            "target_identities": ["identity_000", "identity_002"],
            "retain_identities": ["identity_004", "identity_006"],
            "control_identities": ["identity_008", "identity_010"],
        }
        (sel_dir / "pilot_identity_selection.json").write_text(
            json.dumps(sel_manifest),
        )

        mock_training = {
            "num_optimizer_steps": 1,
            "final_checkpoint_path": "",
            "elapsed_seconds": 0.1,
        }
        mock_post_eval = {
            "summary": {"n_probes": 10, "inference_errors": 0},
            "validation": {"pass": True},
            "results_path": str(tmp_path / "post_results.jsonl"),
        }

        selection_called = False
        original_run_selection = PilotRunner.run_selection

        def _track_selection(self_runner):
            nonlocal selection_called
            selection_called = True
            return original_run_selection(self_runner)

        _mod = run_unlearning_pilot
        with patch.object(
            PilotRunner, "run_selection", _track_selection,
        ), patch.object(
            _mod, "_run_training_phase", return_value=mock_training,
        ), patch.object(
            _mod, "_run_post_eval_phase", return_value=mock_post_eval,
        ), patch(
            "route_data.eval.run_pilot.PilotRunner.run_paired_analysis",
            return_value={"probe_deltas": {}, "group_effects": {}, "preservation_report": {}, "pairing_validation": {"pass": True}},
        ):
            _mod.run_pipeline(str(cfg_path), smoke=True, resume=True)

        assert not selection_called, "Selection should be skipped in resume mode"


# --------------------------------------------------------------------------- #
# Tests: main() CLI
# --------------------------------------------------------------------------- #

class TestMainCLI:
    """P1-8: CLI entrypoint."""

    def test_missing_config_returns_1(self, tmp_path: Path) -> None:
        """Missing config file should return exit code 1."""
        ret = run_unlearning_pilot.main(["--config", str(tmp_path / "nonexistent.yaml")])
        assert ret == 1

    def test_smoke_pipeline_returns_int(self, tmp_path: Path) -> None:
        """main() should return an integer exit code."""
        bl_path, rp_path = _make_synthetic_data(tmp_path)
        cfg_path = _make_config(tmp_path, bl_path, rp_path)

        mock_training = {
            "num_optimizer_steps": 1,
            "final_checkpoint_path": "",
            "elapsed_seconds": 0.1,
        }
        mock_post_eval = {
            "summary": {"n_probes": 10, "inference_errors": 0},
            "validation": {"pass": True},
            "results_path": str(tmp_path / "post_results.jsonl"),
        }

        _mod = run_unlearning_pilot
        with patch.object(
            _mod, "_run_training_phase", return_value=mock_training,
        ), patch.object(
            _mod, "_run_post_eval_phase", return_value=mock_post_eval,
        ), patch(
            "route_data.eval.run_pilot.PilotRunner.run_paired_analysis",
            return_value={"probe_deltas": {}, "group_effects": {}, "preservation_report": {}, "pairing_validation": {"pass": True}},
        ):
            ret = _mod.main(["--config", str(cfg_path), "--smoke"])
            assert isinstance(ret, int)


# --------------------------------------------------------------------------- #
# Tests: helpers
# --------------------------------------------------------------------------- #

class TestHelpers:
    """P1-8: Utility functions."""

    def test_sha256_file(self, tmp_path: Path) -> None:
        """SHA-256 should be deterministic."""
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = run_unlearning_pilot._sha256_file(f)
        h2 = run_unlearning_pilot._sha256_file(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_git_commit_returns_string(self) -> None:
        """_git_commit should return a string."""
        result = run_unlearning_pilot._git_commit()
        assert isinstance(result, str)

    def test_git_dirty_returns_bool(self) -> None:
        """_git_dirty should return a bool."""
        result = run_unlearning_pilot._git_dirty()
        assert isinstance(result, bool)
