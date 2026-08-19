"""Integration tests for multi-model runner dispatch (P0-11).

Covers:
- BaselineTrainer adapter constructor + collator dispatch
- Profile-driven GA dry-run
- Suite --model-profile propagation
- Baseline/model identity mismatch
- Profile provenance persistence
- Structural metadata mismatch
- Qwen composite topology synthetic test
- BaselineBinding resolver
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ #
# Fake adapter for BaselineTrainer tests
# ------------------------------------------------------------------ #

class _FakeProfile:
    """Minimal profile stand-in for adapter tests."""

    def __init__(self):
        self.key = "test_model"
        self.adapter_name = "test_family"
        self.processor_id = "test/proc"
        self.processor_revision = "c" * 40


class _FakeAdapter:
    """Minimal adapter stand-in for BaselineTrainer tests."""

    def __init__(self):
        self.profile = _FakeProfile()
        self.collate_calls = 0

    def collate(self, batch):
        self.collate_calls += 1
        return batch

    def image_indexed_keys(self) -> frozenset:
        return frozenset({"pixel_values", "image_grid_thw"})


# ------------------------------------------------------------------ #
# BaselineTrainer adapter constructor + collator tests (P0-1)
# ------------------------------------------------------------------ #

class TestBaselineTrainerAdapter:
    """P0-11: BaselineTrainer accepts adapter and uses its collator."""

    def _make_config(self, tmp_path):
        from route_data.unlearning.baseline_runner import BaselineTrainingConfig
        cfg = BaselineTrainingConfig()
        cfg.output_dir = str(tmp_path)
        cfg.checkpoint_steps = []
        return cfg

    def test_constructor_with_adapter(self, tmp_path):
        """Constructor succeeds with adapter= and stores it."""
        from route_data.unlearning.baseline_runner import BaselineTrainer

        adapter = _FakeAdapter()
        config = self._make_config(tmp_path)
        objective = MagicMock()
        model = nn.Linear(16, 16)  # Real model with parameters.
        processor = MagicMock()
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=1)
        dataset.__getitem__ = MagicMock(return_value={})

        trainer = BaselineTrainer(
            config=config,
            objective=objective,
            model=model,
            processor=processor,
            forget_dataset=dataset,
            adapter=adapter,
        )
        assert trainer.adapter is adapter

    def test_adapter_collator_used(self, tmp_path):
        """When adapter is provided, its collate is used for DataLoaders."""
        from route_data.unlearning.baseline_runner import BaselineTrainer

        adapter = _FakeAdapter()
        config = self._make_config(tmp_path)
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=2)
        dataset.__getitem__ = MagicMock(return_value={"input_ids": torch.tensor([1])})

        trainer = BaselineTrainer(
            config=config,
            objective=MagicMock(),
            model=nn.Linear(16, 16),
            processor=MagicMock(),
            forget_dataset=dataset,
            adapter=adapter,
        )
        # The forget_loader's collate_fn should be the adapter's collate.
        # Bound methods are not identity-stable, so compare underlying funcs.
        assert trainer.forget_loader.collate_fn.__self__ is adapter

    def test_legacy_collator_without_adapter(self, tmp_path):
        """When adapter=None, qwen_collate_fn is used."""
        from route_data.eval.unlearning_harness import qwen_collate_fn
        from route_data.unlearning.baseline_runner import BaselineTrainer

        config = self._make_config(tmp_path)
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=2)
        dataset.__getitem__ = MagicMock(return_value={})

        trainer = BaselineTrainer(
            config=config,
            objective=MagicMock(),
            model=nn.Linear(16, 16),
            processor=MagicMock(),
            forget_dataset=dataset,
            adapter=None,
        )
        assert trainer.forget_loader.collate_fn is qwen_collate_fn


# ------------------------------------------------------------------ #
# Suite --model-profile propagation test (P0-6)
# ------------------------------------------------------------------ #

class TestSuiteProfilePropagation:
    """P0-11: Suite propagates --model-profile to every subprocess."""

    def test_model_profile_in_subprocess_cmd(self, tmp_path):
        """When model_profile_path is in suite_state, subprocess cmd includes it."""
        if str(REPO_ROOT / "scripts") not in sys.path:
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import run_mllmu_baseline_suite as suite_mod

        profile_path = tmp_path / "test_profile.yaml"
        profile_path.write_text("key: test\n")

        suite_state = {
            "model_profile_path": str(profile_path),
        }
        config = {
            "runtime": {"output_dir": str(tmp_path / "output")},
        }

        # Mock subprocess.run to capture the command.
        captured_cmds = []

        def fake_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            raise subprocess.CalledProcessError(1, cmd)

        import subprocess
        with patch("subprocess.run", side_effect=fake_run):
            result = suite_mod._run_single_method("ga", config, suite_state)

        # The method will fail, but we captured the cmd.
        assert result["status"] == "training_failed"
        assert len(captured_cmds) == 1
        cmd = captured_cmds[0]
        assert "--model-profile" in cmd
        idx = cmd.index("--model-profile")
        assert cmd[idx + 1] == str(profile_path)

    def test_no_model_profile_omits_flag(self, tmp_path):
        """Without model_profile_path, --model-profile is NOT in cmd."""
        import subprocess

        if str(REPO_ROOT / "scripts") not in sys.path:
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import run_mllmu_baseline_suite as suite_mod

        suite_state: dict[str, Any] = {}
        config = {
            "runtime": {"output_dir": str(tmp_path / "output")},
        }

        captured_cmds = []

        def fake_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            raise subprocess.CalledProcessError(1, cmd)

        with patch("subprocess.run", side_effect=fake_run):
            result = suite_mod._run_single_method("ga", config, suite_state)

        assert result["status"] == "training_failed"
        cmd = captured_cmds[0]
        assert "--model-profile" not in cmd


# ------------------------------------------------------------------ #
# Baseline/model identity mismatch test (P0-7)
# ------------------------------------------------------------------ #

class TestBaselineIdentityValidation:
    """P0-11: Baseline/model identity mismatch causes hard failure."""

    def test_mismatch_detected(self):
        """validate_baseline_model_identity catches revision mismatch."""
        from route_data.eval.post_unlearning_eval import (
            BaselineBinding,
            validate_baseline_model_identity,
        )

        binding = BaselineBinding(
            model_id="Qwen/Qwen3.5-9B",
            model_revision="a" * 40,
            processor_revision="b" * 40,
        )
        errors = validate_baseline_model_identity(
            binding,
            model_key="qwen35_9b",
            model_id="Qwen/Qwen3.5-9B",
            model_revision="c" * 40,  # MISMATCH
            processor_revision="b" * 40,
        )
        assert len(errors) == 1
        assert "revision" in errors[0]

    def test_match_passes(self):
        """Matching identity produces no errors."""
        from route_data.eval.post_unlearning_eval import (
            BaselineBinding,
            validate_baseline_model_identity,
        )

        binding = BaselineBinding(
            model_id="Qwen/Qwen3.5-9B",
            model_revision="a" * 40,
            processor_revision="b" * 40,
        )
        errors = validate_baseline_model_identity(
            binding,
            model_key="qwen35_9b",
            model_id="Qwen/Qwen3.5-9B",
            model_revision="a" * 40,
            processor_revision="b" * 40,
        )
        assert errors == []

    def test_cross_model_mismatch(self):
        """GLM baseline vs Qwen profile is caught."""
        from route_data.eval.post_unlearning_eval import (
            BaselineBinding,
            validate_baseline_model_identity,
        )

        binding = BaselineBinding(
            model_id="THUDM/GLM-4.6V-Flash",
            model_revision="d" * 40,
        )
        errors = validate_baseline_model_identity(
            binding,
            model_key="glm46v_flash",
            model_id="Qwen/Qwen3.5-9B",  # Deliberate mismatch
            model_revision="a" * 40,
        )
        assert len(errors) >= 1


# ------------------------------------------------------------------ #
# BaselineBinding resolver test (P0-7)
# ------------------------------------------------------------------ #

class TestBaselineResolver:
    """P0-11: resolve_preunlearning_baseline resolves model-specific paths."""

    def test_resolver_finds_model_specific_baseline(self, tmp_path):
        """Resolver constructs correct paths for a given model_key."""
        from route_data.eval.post_unlearning_eval import (
            resolve_preunlearning_baseline,
        )

        # Create fake baseline files.
        baseline_dir = (
            tmp_path / "outputs" / "experiments" / "pre_unlearning"
            / "qwen35_9b" / "baseline_v1"
        )
        baseline_dir.mkdir(parents=True)
        (baseline_dir / "baseline_results.jsonl").write_text(
            '{"probe_id": "p1"}\n'
        )
        manifest = {
            "model": {
                "id": "Qwen/Qwen3.5-9B",
                "revision": "a" * 40,
                "processor_revision": "b" * 40,
                "model_profile_sha256": "e" * 64,
            },
            "provenance": {
                "results_sha256": "f" * 64,
                "manifest_sha256": "g" * 64,
            },
        }
        (baseline_dir / "baseline_manifest.json").write_text(
            json.dumps(manifest)
        )

        binding = resolve_preunlearning_baseline(
            "qwen35_9b",
            project_root=tmp_path,
        )
        assert binding.model_id == "Qwen/Qwen3.5-9B"
        assert binding.model_revision == "a" * 40
        assert binding.processor_revision == "b" * 40
        assert binding.results_sha256 == "f" * 64

    def test_resolver_missing_file_raises(self, tmp_path):
        """Resolver raises FileNotFoundError when baseline doesn't exist."""
        from route_data.eval.post_unlearning_eval import (
            resolve_preunlearning_baseline,
        )

        with pytest.raises(FileNotFoundError, match="baseline_results"):
            resolve_preunlearning_baseline(
                "nonexistent_model",
                project_root=tmp_path,
            )


# ------------------------------------------------------------------ #
# Profile provenance test (P0-8)
# ------------------------------------------------------------------ #

class TestProfileProvenance:
    """P0-11: Provenance fields are persisted in training config."""

    def test_baseline_training_config_has_provenance(self):
        """BaselineTrainingConfig has all P0-8 provenance fields."""
        from route_data.unlearning.baseline_runner import BaselineTrainingConfig

        cfg = BaselineTrainingConfig()
        assert hasattr(cfg, "model_key")
        assert hasattr(cfg, "processor_id")
        assert hasattr(cfg, "processor_revision")
        assert hasattr(cfg, "model_profile_sha256")
        assert hasattr(cfg, "adapter_family")
        assert hasattr(cfg, "lora_target_inventory_sha256")

    def test_post_eval_config_has_provenance(self):
        """PostEvalConfig has all P0-8 provenance fields."""
        from route_data.eval.post_unlearning_eval import PostEvalConfig

        cfg = PostEvalConfig()
        assert hasattr(cfg, "model_key")
        assert hasattr(cfg, "processor_id")
        assert hasattr(cfg, "processor_revision")
        assert hasattr(cfg, "model_profile_sha256")
        assert hasattr(cfg, "adapter_family")


# ------------------------------------------------------------------ #
# Structural metadata mismatch test (P0-5)
# ------------------------------------------------------------------ #

class TestStructuralMetadataValidation:
    """P0-11: Structural metadata mismatch causes hard failure."""

    def test_layer_count_mismatch(self):
        """validate_structural_metadata catches layer count mismatch."""
        from route_data.models.trainable.registry import (
            validate_structural_metadata,
        )

        adapter = MagicMock()
        adapter.profile.num_language_layers = 28
        adapter.profile.language_hidden_size = 4096
        adapter.profile.intermediate_size = 12288
        adapter.language_layers.return_value = [MagicMock()] * 32  # 32 != 28
        adapter.language_hidden_size.return_value = 4096
        adapter.language_intermediate_size.return_value = 12288

        model = MagicMock()
        errors = validate_structural_metadata(adapter, model)
        assert any("num_language_layers" in e for e in errors)

    def test_hidden_size_mismatch(self):
        """validate_structural_metadata catches hidden size mismatch."""
        from route_data.models.trainable.registry import (
            validate_structural_metadata,
        )

        adapter = MagicMock()
        adapter.profile.num_language_layers = 32
        adapter.profile.language_hidden_size = 3584  # Wrong
        adapter.profile.intermediate_size = 12288
        adapter.language_layers.return_value = [MagicMock()] * 32
        adapter.language_hidden_size.return_value = 4096  # Runtime says 4096
        adapter.language_intermediate_size.return_value = 12288

        model = MagicMock()
        errors = validate_structural_metadata(adapter, model)
        assert any("language_hidden_size" in e for e in errors)

    def test_all_match_passes(self):
        """Matching metadata produces no errors."""
        from route_data.models.trainable.registry import (
            validate_structural_metadata,
        )

        adapter = MagicMock()
        adapter.profile.num_language_layers = 32
        adapter.profile.language_hidden_size = 4096
        adapter.profile.intermediate_size = 12288
        adapter.language_layers.return_value = [MagicMock()] * 32
        adapter.language_hidden_size.return_value = 4096
        adapter.language_intermediate_size.return_value = 12288

        model = MagicMock()
        errors = validate_structural_metadata(adapter, model)
        assert errors == []


# ------------------------------------------------------------------ #
# Qwen composite topology synthetic test (P0-3)
# ------------------------------------------------------------------ #

class TestQwenCompositeTopology:
    """P0-11: Synthetic Qwen composite module tree for LoRA regex."""

    def test_qwen35_lora_regex_selects_language_only(self):
        """Qwen3.5 LoRA regex selects language attention, rejects vision."""
        from route_data.models.trainable.base import ModelFamilyProfile
        from route_data.models.trainable.qwen35 import Qwen35Adapter

        # Build a synthetic module tree matching composite Qwen3.5 structure.
        model = nn.Module()
        # Language model layers.
        lang_layers = nn.ModuleList()
        for i in range(4):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(16, 16)
            layer.self_attn.k_proj = nn.Linear(16, 16)
            layer.self_attn.v_proj = nn.Linear(16, 16)
            layer.self_attn.o_proj = nn.Linear(16, 16)
            layer.mlp = nn.Module()
            layer.mlp.gate_proj = nn.Linear(16, 16)
            lang_layers.append(layer)

        # Vision blocks (should NOT be selected).
        vision_blocks = nn.ModuleList()
        vis_block = nn.Module()
        vis_block.attn = nn.Module()
        vis_block.attn.q_proj = nn.Linear(16, 16)
        vis_block.attn.v_proj = nn.Linear(16, 16)
        vision_blocks.append(vis_block)

        # Assemble composite structure.
        model.language_model = nn.Module()
        model.language_model.layers = lang_layers
        model.visual = nn.Module()
        model.visual.blocks = vision_blocks

        # Build a profile with the expected regex.
        profile = ModelFamilyProfile(
            key="qwen35_test",
            model_id="Qwen/Qwen3.5-9B",
            revision="a" * 40,
            processor_id="Qwen/Qwen3.5-9B",
            processor_revision="b" * 40,
            adapter_name="qwen35",
            trust_remote_code=False,
            dtype="bfloat16",
            attn_implementation="sdpa",
            candidate_positive="Yes",
            candidate_negative="No",
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_scope="language_attention_only",
            lora_target_leaf_names=("q_proj", "v_proj"),
            lora_scope_regex=r"^language_model\.layers\.\d+\.self_attn\.",
            r2mu_candidate_layers=(8, 16, 24, 29),
            r2mu_n_select_layers=2,
        )

        adapter = Qwen35Adapter(profile)
        targets = adapter.resolve_lora_targets(model)

        # Language q_proj and v_proj should be selected.
        lang_targets = [t for t in targets if "language_model" in t]
        vision_targets = [t for t in targets if "visual" in t or "blocks" in t]

        assert len(lang_targets) > 0, "Should select language attention modules"
        assert len(vision_targets) == 0, "Must NOT select vision modules"

        # Verify specific targets.
        target_names = set(targets)
        assert "language_model.layers.0.self_attn.q_proj" in target_names
        assert "language_model.layers.0.self_attn.v_proj" in target_names
        assert "visual.blocks.0.attn.q_proj" not in target_names


# ------------------------------------------------------------------ #
# Profile-driven GA dry-run test (P0-11)
# ------------------------------------------------------------------ #

class TestProfileDrivenGADryRun:
    """P0-11: Profile-driven GA constructs BaselineTrainer with adapter."""

    def test_ga_trainer_construction(self, tmp_path):
        """GA objective + adapter → BaselineTrainer constructs correctly."""
        from route_data.unlearning.baseline_runner import (
            BaselineTrainer,
            BaselineTrainingConfig,
        )
        from route_data.unlearning.objectives import GradientAscent

        adapter = _FakeAdapter()
        config = BaselineTrainingConfig()
        config.output_dir = str(tmp_path)
        config.checkpoint_steps = []
        config.method_name = "mllmu_ga"

        objective = GradientAscent()
        model = nn.Linear(16, 16)
        processor = MagicMock()

        # Minimal dataset.
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=2)
        dataset.__getitem__ = MagicMock(
            return_value={"input_ids": torch.tensor([1, 2, 3])}
        )

        trainer = BaselineTrainer(
            config=config,
            objective=objective,
            model=model,
            processor=processor,
            forget_dataset=dataset,
            adapter=adapter,
        )

        assert trainer.adapter is adapter
        assert trainer.forget_loader.collate_fn.__self__ is adapter
        assert trainer.objective is objective


# ------------------------------------------------------------------ #
# Needed for patch import
# ------------------------------------------------------------------ #
from unittest.mock import patch
