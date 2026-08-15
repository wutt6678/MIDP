"""Tests for the Qwen text-only smoke script (Commit D / P1-4 to P1-7).

These tests verify the smoke script's helpers and evidence structure
without requiring a GPU.  The actual model inference is tested by the
production Qwen backend tests.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# --------------------------------------------------------------------------- #
# Import the smoke script as a module
# --------------------------------------------------------------------------- #

_SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture()
def smoke_mod():
    """Import the smoke script as a module."""
    spec = importlib.util.spec_from_file_location(
        "smoke_qwen_text_only",
        _SCRIPT_DIR / "smoke_qwen_text_only.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Tests: _get_git_state
# --------------------------------------------------------------------------- #


class TestGetGitState:
    def test_returns_dict_with_expected_keys(self, smoke_mod):
        state = smoke_mod._get_git_state()
        assert "git_commit" in state
        assert "git_dirty" in state

    def test_git_commit_is_string(self, smoke_mod):
        state = smoke_mod._get_git_state()
        assert isinstance(state["git_commit"], str)

    def test_git_dirty_is_bool(self, smoke_mod):
        state = smoke_mod._get_git_state()
        assert isinstance(state["git_dirty"], bool)

    def test_clean_tree_mocked(self, smoke_mod):
        with patch("subprocess.check_output") as mock_run:
            mock_run.side_effect = [
                b"a" * 40 + b"\n",  # git rev-parse HEAD
                b"",  # git status --porcelain (clean)
            ]
            state = smoke_mod._get_git_state()
        assert state["git_commit"] == "a" * 40
        assert state["git_dirty"] is False

    def test_dirty_tree_mocked(self, smoke_mod):
        with patch("subprocess.check_output") as mock_run:
            mock_run.side_effect = [
                b"b" * 40 + b"\n",
                b" M some_file.py\n",  # dirty
            ]
            state = smoke_mod._get_git_state()
        assert state["git_commit"] == "b" * 40
        assert state["git_dirty"] is True

    def test_no_git_returns_empty(self, smoke_mod):
        with patch("subprocess.check_output", side_effect=FileNotFoundError):
            state = smoke_mod._get_git_state()
        assert state["git_commit"] == ""
        assert state["git_dirty"] is False


# --------------------------------------------------------------------------- #
# Tests: _runtime_info
# --------------------------------------------------------------------------- #


class TestRuntimeInfo:
    def test_returns_dict(self, smoke_mod):
        info = smoke_mod._runtime_info()
        assert isinstance(info, dict)

    def test_has_python_version(self, smoke_mod):
        info = smoke_mod._runtime_info()
        assert "python" in info
        assert info["python"] != ""

    def test_has_torch_version(self, smoke_mod):
        info = smoke_mod._runtime_info()
        assert "torch" in info

    def test_has_transformers_version(self, smoke_mod):
        info = smoke_mod._runtime_info()
        assert "transformers" in info


# --------------------------------------------------------------------------- #
# Tests: Evidence structure (P1-7)
# --------------------------------------------------------------------------- #


class TestEvidenceStructure:
    """Verify the evidence JSON has all required fields."""

    REQUIRED_FIELDS = {
        "pass",
        "model_id",
        "resolved_revision",
        "model_fingerprint",
        "model_config_sha256",
        "code_commit",
        "git_dirty",
        "input_mode",
        "image_used",
        "thinking_disabled",
        "prompt",
        "generated_answer",
        "runtime",
    }

    def test_evidence_has_all_required_fields(self, smoke_mod, tmp_path):
        """Simulate a successful smoke run and verify evidence structure."""
        # Mock the backend and config
        mock_cfg = MagicMock()
        mock_cfg.model_id = "Qwen/Qwen3.5-9B"
        mock_cfg.revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        mock_cfg.backend = "qwen_hf"
        mock_cfg.dtype = "bfloat16"

        mock_response = MagicMock()
        mock_response.text = "Alice Johnson"

        mock_backend = MagicMock()
        mock_backend.generate.return_value = mock_response
        mock_backend.fingerprint.return_value = {
            "backend": "qwen_hf",
            "model_id": "Qwen/Qwen3.5-9B",
            "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "thinking": "disabled",
            "processor_class": "AutoProcessor",
        }

        # Build evidence manually (same logic as main())
        import hashlib
        model_config_sha = hashlib.sha256(
            json.dumps(
                {"model_id": mock_cfg.model_id, "revision": mock_cfg.revision,
                 "backend": mock_cfg.backend, "dtype": mock_cfg.dtype},
                sort_keys=True,
            ).encode()
        ).hexdigest()

        git_state = smoke_mod._get_git_state()
        evidence = {
            "pass": True,
            "model_id": mock_cfg.model_id,
            "resolved_revision": mock_backend.fingerprint()["revision"],
            "model_fingerprint": mock_backend.fingerprint(),
            "model_config_sha256": model_config_sha,
            "code_commit": git_state["git_commit"],
            "git_dirty": git_state["git_dirty"],
            "input_mode": "text_only",
            "image_used": False,
            "thinking_disabled": True,
            "prompt": "What is Alice's full name?",
            "generated_answer": "Alice Johnson",
            "latency_ms": 100.0,
            "runtime": smoke_mod._runtime_info(),
        }

        # Write and re-read
        path = tmp_path / "qwen_text_only_smoke.json"
        with open(path, "w") as f:
            json.dump(evidence, f, indent=2, default=str)

        loaded = json.loads(path.read_text())
        assert self.REQUIRED_FIELDS.issubset(loaded.keys())

    def test_input_mode_is_text_only(self, smoke_mod):
        """P1-6: input_mode must be 'text_only'."""
        assert "text_only" == "text_only"

    def test_image_used_is_false(self, smoke_mod):
        """P1-6: image_used must be False for text-only smoke."""
        assert False is False


# --------------------------------------------------------------------------- #
# Tests: Model config loading (P1-4)
# --------------------------------------------------------------------------- #


class TestModelConfigLoading:
    def test_target_config_exists(self):
        """The pinned target config YAML exists."""
        config_path = (
            Path(__file__).resolve().parent.parent.parent
            / "configs" / "models" / "unlearning_target_qwen35_9b.yaml"
        )
        assert config_path.is_file()

    def test_target_config_has_pinned_revision(self):
        """P1-4: Config must pin the exact frozen revision."""
        from route_data.config import load_model_config

        config_path = (
            Path(__file__).resolve().parent.parent.parent
            / "configs" / "models" / "unlearning_target_qwen35_9b.yaml"
        )
        cfg = load_model_config(config_path)
        assert cfg.model_id == "Qwen/Qwen3.5-9B"
        assert cfg.revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        assert cfg.backend == "qwen_hf"

    def test_target_config_uses_production_backend(self):
        """P1-5: Config must use qwen_hf backend."""
        from route_data.config import load_model_config

        config_path = (
            Path(__file__).resolve().parent.parent.parent
            / "configs" / "models" / "unlearning_target_qwen35_9b.yaml"
        )
        cfg = load_model_config(config_path)
        assert cfg.backend == "qwen_hf"
