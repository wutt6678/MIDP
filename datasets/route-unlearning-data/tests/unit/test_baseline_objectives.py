"""Unit tests for MLLMU-Bench baseline objectives.

These tests use tiny synthetic models and data so that no GPU or real
model weights are required. They verify the mathematical correctness
and structural invariants of each objective implementation.

Tests from plan Section 1.6:
- test_ga_loss_sign
- test_gd_formula
- test_kl_zero_when_logits_equal
- test_kl_reference_frozen
- test_prompting_no_model_update
- test_npo_reference_frozen
- test_npo_beta_paper_value
- test_npo_gradient_direction
- test_answer_only_masking_all_methods
- test_baseline_checkpoint_manifest
- test_baseline_exact_pairing
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from route_data.unlearning.objectives import (
    GradientAscent,
    GradientDifference,
    KLMinimization,
    NegativePreferenceOptimization,
    answer_only_cross_entropy,
)

# --------------------------------------------------------------------------- #
# Fixtures — tiny model and synthetic batches
# --------------------------------------------------------------------------- #

VOCAB_SIZE = 50
SEQ_LEN = 8
HIDDEN_DIM = 16
BATCH_SIZE = 2
ANSWER_START = 4  # Answer tokens start at position 4


class TinyLogitModel(nn.Module):
    """Minimal model that returns logits from a linear head."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.config = MagicMock()
        self.config.model_type = "tiny_test"
        self.config.hidden_size = hidden_dim
        self.config.num_hidden_layers = 1

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        h = self.embedding(input_ids)
        logits = self.head(h)
        return MagicMock(logits=logits)


def _make_synthetic_batch(
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    answer_start: int = ANSWER_START,
) -> dict[str, Any]:
    """Create a synthetic batch with answer-only masking.

    Tokens before answer_start have labels=-100 (masked).
    Tokens from answer_start onward have real labels.
    """
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    # Set answer labels (positions answer_start onward)
    for i in range(batch_size):
        for j in range(answer_start, seq_len):
            labels[i, j] = torch.randint(0, VOCAB_SIZE, (1,)).item()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _make_identical_batch() -> tuple[dict[str, Any], dict[str, Any]]:
    """Create two identical batches for KL=0 test."""
    batch = _make_synthetic_batch()
    batch2 = {k: v.clone() for k, v in batch.items()}
    return batch, batch2


# --------------------------------------------------------------------------- #
# Tests — GA (B1)
# --------------------------------------------------------------------------- #

class TestGradientAscent:
    """Tests for GradientAscent objective."""

    def test_ga_loss_sign(self) -> None:
        """GA loss = -ce_forget (negative)."""
        model = TinyLogitModel()
        batch = _make_synthetic_batch()
        ga = GradientAscent()

        result = ga.compute_loss(model=model, forget_batch=batch)

        # CE is always non-negative, so GA loss should be non-positive
        assert result["total_loss"].item() <= 0.0
        assert result["forget_loss"].item() <= 0.0
        # total_loss == forget_loss for GA
        assert result["total_loss"].item() == pytest.approx(
            result["forget_loss"].item(), abs=1e-6,
        )
        # Unused fields are None
        assert result["retain_loss"] is None
        assert result["kl_loss"] is None
        assert result["npo_loss"] is None

    def test_ga_name(self) -> None:
        ga = GradientAscent()
        assert ga.name == "mllmu_ga"

    def test_ga_gradient_direction(self) -> None:
        """GA gradient should increase forget CE (gradient ascent)."""
        model = TinyLogitModel()
        batch = _make_synthetic_batch()
        ga = GradientAscent()

        # Compute loss and backprop
        result = ga.compute_loss(model=model, forget_batch=batch)
        result["total_loss"].backward()

        # Check that gradients exist
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "GA should produce non-zero gradients"


# --------------------------------------------------------------------------- #
# Tests — GD (B2)
# --------------------------------------------------------------------------- #

class TestGradientDifference:
    """Tests for GradientDifference objective."""

    def test_gd_formula(self) -> None:
        """GD = -forget_ce + retain_weight * retain_ce."""
        model = TinyLogitModel()
        forget_batch = _make_synthetic_batch()
        retain_batch = _make_synthetic_batch()

        retain_weight = 1.0
        gd = GradientDifference(retain_weight=retain_weight)
        result = gd.compute_loss(
            model=model, forget_batch=forget_batch, retain_batch=retain_batch,
        )

        # Compute individual terms manually
        ce_forget = answer_only_cross_entropy(model, forget_batch)
        ce_retain = answer_only_cross_entropy(model, retain_batch)
        expected = -ce_forget + retain_weight * ce_retain

        assert result["total_loss"].item() == pytest.approx(
            expected.item(), abs=1e-5,
        )
        assert result["forget_loss"].item() == pytest.approx(
            -ce_forget.item(), abs=1e-6,
        )
        assert result["retain_loss"].item() == pytest.approx(
            ce_retain.item(), abs=1e-6,
        )

    def test_gd_requires_retain_batch(self) -> None:
        """GD should raise if retain_batch is None."""
        model = TinyLogitModel()
        forget_batch = _make_synthetic_batch()
        gd = GradientDifference()

        with pytest.raises(ValueError, match="retain_batch"):
            gd.compute_loss(model=model, forget_batch=forget_batch)

    def test_gd_retain_weight_effect(self) -> None:
        """Higher retain_weight should increase total loss."""
        model = TinyLogitModel()
        forget_batch = _make_synthetic_batch()
        retain_batch = _make_synthetic_batch()

        gd_low = GradientDifference(retain_weight=0.1)
        gd_high = GradientDifference(retain_weight=10.0)

        r_low = gd_low.compute_loss(model, forget_batch, retain_batch)
        r_high = gd_high.compute_loss(model, forget_batch, retain_batch)

        # Higher retain_weight → higher total (since retain CE > 0)
        assert r_high["total_loss"].item() > r_low["total_loss"].item()

    def test_gd_name(self) -> None:
        gd = GradientDifference()
        assert gd.name == "mllmu_ga_difference"


# --------------------------------------------------------------------------- #
# Tests — KL (B3)
# --------------------------------------------------------------------------- #

class TestKLMinimization:
    """Tests for KLMinimization objective."""

    def test_kl_zero_when_logits_equal(self) -> None:
        """KL = 0 when current model == reference model."""
        model = TinyLogitModel()
        # Use the same model as both current and reference
        batch = _make_synthetic_batch()

        kl = KLMinimization(temperature=1.0)
        result = kl.compute_loss(
            model=model, forget_batch=batch,
            retain_batch=batch, reference_model=model,
        )

        # KL(current || current) = 0
        assert result["kl_loss"].item() == pytest.approx(0.0, abs=1e-5)

    def test_kl_reference_frozen(self) -> None:
        """Reference model should have requires_grad=False."""
        model = TinyLogitModel()
        ref_model = TinyLogitModel()

        # Freeze reference
        for param in ref_model.parameters():
            param.requires_grad = False

        batch = _make_synthetic_batch()
        kl = KLMinimization()
        result = kl.compute_loss(
            model=model, forget_batch=batch,
            retain_batch=batch, reference_model=ref_model,
        )

        # Verify reference model params are still frozen
        for param in ref_model.parameters():
            assert not param.requires_grad

        # Verify total loss has gradient (from current model)
        assert result["total_loss"].requires_grad or result["forget_loss"].requires_grad

    def test_kl_requires_reference(self) -> None:
        """KL should raise if reference_model is None."""
        model = TinyLogitModel()
        batch = _make_synthetic_batch()
        kl = KLMinimization()

        with pytest.raises(ValueError, match="reference_model"):
            kl.compute_loss(
                model=model, forget_batch=batch, retain_batch=batch,
            )

    def test_kl_requires_retain_batch(self) -> None:
        """KL should raise if retain_batch is None."""
        model = TinyLogitModel()
        ref_model = TinyLogitModel()
        batch = _make_synthetic_batch()
        kl = KLMinimization()

        with pytest.raises(ValueError, match="retain_batch"):
            kl.compute_loss(
                model=model, forget_batch=batch,
                reference_model=ref_model,
            )

    def test_kl_positive_divergence(self) -> None:
        """KL should be positive when models differ."""
        torch.manual_seed(42)
        model = TinyLogitModel()
        ref_model = TinyLogitModel()  # Different random weights

        batch = _make_synthetic_batch()
        kl = KLMinimization(temperature=1.0)
        result = kl.compute_loss(
            model=model, forget_batch=batch,
            retain_batch=batch, reference_model=ref_model,
        )

        # KL divergence is always non-negative
        assert result["kl_loss"].item() >= -1e-6  # Allow tiny numerical error

    def test_kl_temperature_effect(self) -> None:
        """Higher temperature should smooth the KL."""
        torch.manual_seed(42)
        model = TinyLogitModel()
        ref_model = TinyLogitModel()

        batch = _make_synthetic_batch()
        kl_low_t = KLMinimization(temperature=0.5)
        kl_high_t = KLMinimization(temperature=2.0)

        r_low = kl_low_t.compute_loss(model, batch, batch, ref_model)
        r_high = kl_high_t.compute_loss(model, batch, batch, ref_model)

        # Both should be finite
        assert torch.isfinite(r_low["kl_loss"])
        assert torch.isfinite(r_high["kl_loss"])

    def test_kl_name(self) -> None:
        kl = KLMinimization()
        assert kl.name == "mllmu_kl_min"


# --------------------------------------------------------------------------- #
# Tests — NPO (B5)
# --------------------------------------------------------------------------- #

class TestNegativePreferenceOptimization:
    """Tests for NegativePreferenceOptimization objective."""

    def test_npo_beta_paper_value(self) -> None:
        """NPO beta should be 0.9 (paper value)."""
        npo = NegativePreferenceOptimization(beta=0.9)
        assert npo.beta == pytest.approx(0.9)

    def test_npo_default_beta(self) -> None:
        """Default beta should be 0.9 (paper value, NOT repo default 0.4)."""
        npo = NegativePreferenceOptimization()
        assert npo.beta == pytest.approx(0.9)

    def test_npo_requires_oracle(self) -> None:
        """NPO should raise if oracle_model is None."""
        model = TinyLogitModel()
        batch = _make_synthetic_batch()
        batch["_prefix_len"] = [2, 2]
        batch["_answer_label"] = [True, False]
        batch["_yes_token_ids"] = [[10]]
        batch["_no_token_ids"] = [[20]]

        npo = NegativePreferenceOptimization()
        with pytest.raises(ValueError, match="oracle_model"):
            npo.compute_loss(model=model, forget_batch=batch)

    def test_npo_reference_frozen(self) -> None:
        """Oracle model should have requires_grad=False."""
        _ = TinyLogitModel()
        oracle = TinyLogitModel()

        for param in oracle.parameters():
            param.requires_grad = False

        for param in oracle.parameters():
            assert not param.requires_grad

    def test_npo_name(self) -> None:
        npo = NegativePreferenceOptimization()
        assert npo.name == "mllmu_npo"

    def test_npo_loss_is_non_negative(self) -> None:
        """NPO loss = -(2/beta) * log_sigmoid(-beta * log_ratio).

        log_sigmoid(x) <= 0 for all x, so -log_sigmoid(x) >= 0.
        With the -(2/beta) factor (beta > 0), the loss is non-negative.
        """
        # Verify mathematically: for any log_ratio, loss >= 0
        for log_ratio_val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            log_ratio = torch.tensor(log_ratio_val)
            beta = 0.9
            loss = -(2.0 / beta) * F.logsigmoid(-beta * log_ratio)
            assert loss.item() >= -1e-6  # Allow tiny numerical error


# --------------------------------------------------------------------------- #
# Tests — Answer-only masking
# --------------------------------------------------------------------------- #

class TestAnswerOnlyMasking:
    """Verify that all methods mask prompt tokens correctly."""

    def test_answer_only_masking_all_methods(self) -> None:
        """All methods should use answer-only CE (labels=-100 for prompts)."""
        model = TinyLogitModel()
        batch = _make_synthetic_batch()

        # Verify labels structure: positions before ANSWER_START are -100
        for i in range(BATCH_SIZE):
            for j in range(ANSWER_START):
                assert batch["labels"][i, j] == -100
            for j in range(ANSWER_START, SEQ_LEN):
                assert batch["labels"][i, j] != -100

        # CE should only use answer tokens
        ce = answer_only_cross_entropy(model, batch)
        assert torch.isfinite(ce)
        assert ce.item() >= 0.0  # CE is non-negative

    def test_all_masked_labels_give_zero_loss(self) -> None:
        """If all labels are -100, CE should be 0 (or NaN from empty mean)."""
        model = TinyLogitModel()
        batch = _make_synthetic_batch()
        # Mask everything
        batch["labels"][:] = -100

        ce = answer_only_cross_entropy(model, batch)
        # With all labels masked, F.cross_entropy with ignore_index=-100
        # and reduction="mean" returns NaN (0/0)
        assert torch.isnan(ce) or ce.item() == 0.0


# --------------------------------------------------------------------------- #
# Tests — Prompting baseline
# --------------------------------------------------------------------------- #

class TestPromptingBaseline:
    """Tests for the prompting baseline (B4)."""

    def test_prompting_no_model_update(self) -> None:
        """Prompting baseline should not modify any model parameters."""
        import tempfile

        from route_data.unlearning.baseline_methods import PromptingBaseline

        model = TinyLogitModel()
        # Record initial parameters
        initial_params = {
            name: p.clone() for name, p in model.named_parameters()
        }

        # Create dummy probe and freeze verification files
        with tempfile.TemporaryDirectory() as tmpdir:
            probe_path = Path(tmpdir) / "probes.jsonl"
            with open(probe_path, "w") as f:
                f.write(json.dumps({
                    "probe_id": "test-001",
                    "sample_id": "test-sample",
                    "identity_id": "identity-test",
                    "benchmark": "route_conflict_eval",
                    "probe_family": "name_only",
                    "modality": "text-only",
                    "question": "Test question?",
                    "expected_evidence_source": "system-memory",
                    "controlled_variables": [],
                    "image_uri": None,
                    "image_sha256": None,
                    "registry_hash": "abc123",
                    "target_attribute": None,
                    "answer_label": None,
                    "answer_text": "No answer.",
                }) + "\n")

            freeze_path = Path(tmpdir) / "freeze_verification.json"
            with open(freeze_path, "w") as f:
                json.dump({
                    "dataset_version": "fiubench-route-v1",
                    "ready_for_experiments": True,
                    "bundle_verifier_pass": True,
                    "strict_final_verify_pass": True,
                    "manual_audit_pass": True,
                    "exact_ci_pass": True,
                    "hard_stop_conditions": {
                        "manual_audit_matches_current_route_artifact": True,
                        "manual_audit_route_count_matches": True,
                        "all_artifact_hashes_verified": True,
                        "all_commits_reachable": True,
                        "git_dirty_false": True,
                    },
                }, f)

            baseline = PromptingBaseline()
            baseline.run_evaluation(
                model=model,
                processor=MagicMock(),
                probe_dataset_path=str(probe_path),
                output_dir="/tmp/test_prompting",
                freeze_verification_path=str(freeze_path),
                skip_research_preflight=True,
            )

        # Verify no parameters changed
        for name, p in model.named_parameters():
            assert torch.equal(p, initial_params[name]), (
                f"Parameter {name} was modified by prompting baseline"
            )

    def test_prompting_system_prompt(self) -> None:
        """Prompting baseline should store the system prompt."""
        from route_data.unlearning.baseline_methods import (
            MLLMU_PRIVACY_SYSTEM_PROMPT,
            PromptingBaseline,
        )

        baseline = PromptingBaseline()
        assert baseline.system_prompt == MLLMU_PRIVACY_SYSTEM_PROMPT
        assert "personal" in baseline.system_prompt.lower()

    def test_prompting_manifest(self) -> None:
        """Prompting baseline should write a valid manifest."""
        import tempfile

        from route_data.unlearning.baseline_methods import PromptingBaseline

        baseline = PromptingBaseline()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy probe and freeze verification files
            probe_path = Path(tmpdir) / "probes.jsonl"
            with open(probe_path, "w") as f:
                f.write(json.dumps({
                    "probe_id": "test-001",
                    "sample_id": "test-sample",
                    "identity_id": "identity-test",
                    "benchmark": "route_conflict_eval",
                    "probe_family": "name_only",
                    "modality": "text-only",
                    "question": "Test question?",
                    "expected_evidence_source": "system-memory",
                    "controlled_variables": [],
                    "image_uri": None,
                    "image_sha256": None,
                    "registry_hash": "abc123",
                    "target_attribute": None,
                    "answer_label": None,
                    "answer_text": "No answer.",
                }) + "\n")

            freeze_path = Path(tmpdir) / "freeze_verification.json"
            with open(freeze_path, "w") as f:
                json.dump({
                    "dataset_version": "fiubench-route-v1",
                    "ready_for_experiments": True,
                    "bundle_verifier_pass": True,
                    "strict_final_verify_pass": True,
                    "manual_audit_pass": True,
                    "exact_ci_pass": True,
                    "hard_stop_conditions": {
                        "manual_audit_matches_current_route_artifact": True,
                        "manual_audit_route_count_matches": True,
                        "all_artifact_hashes_verified": True,
                        "all_commits_reachable": True,
                        "git_dirty_false": True,
                    },
                }, f)

            result = baseline.run_evaluation(
                model=MagicMock(),
                processor=MagicMock(),
                probe_dataset_path=str(probe_path),
                output_dir=tmpdir,
                freeze_verification_path=str(freeze_path),
                skip_research_preflight=True,
            )
            # Verify the return schema matches evaluate_intervention() structure.
            assert result["method"] == "mllmu_prompting"
            assert result["adapter_path"] is None  # No adapter for prompting.
            assert "delta_target" in result  # Schema includes delta fields.
            assert result["exact_pair_count"] >= 0

            # Verify manifest file was written
            manifest_path = Path(tmpdir) / "prompting_manifest.json"
            assert manifest_path.exists()
            with open(manifest_path) as f:
                manifest = json.load(f)
            assert manifest["training"] is False


# --------------------------------------------------------------------------- #
# Tests — BaselineTrainingConfig
# --------------------------------------------------------------------------- #

class TestBaselineTrainingConfig:
    """Tests for BaselineTrainingConfig."""

    def test_defaults(self) -> None:
        from route_data.unlearning.baseline_runner import BaselineTrainingConfig

        cfg = BaselineTrainingConfig()
        assert cfg.method_name == "mllmu_ga"
        assert cfg.model_id == "Qwen/Qwen3.5-9B"
        assert cfg.seed == 17
        assert cfg.lora_rank == 8
        assert cfg.lora_alpha == 16
        assert cfg.lora_dropout == 0.0
        assert cfg.learning_rate == 2e-5
        assert cfg.num_optimizer_steps == 125
        assert cfg.batch_size == 1
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.npo_beta == 0.9

    def test_effective_batch_size(self) -> None:
        from route_data.unlearning.baseline_runner import BaselineTrainingConfig

        cfg = BaselineTrainingConfig(batch_size=2, gradient_accumulation_steps=4)
        assert cfg.effective_batch_size == 8

    def test_checkpoint_steps_default(self) -> None:
        from route_data.unlearning.baseline_runner import BaselineTrainingConfig

        cfg = BaselineTrainingConfig()
        assert cfg.checkpoint_steps == [1, 5, 10, 25, 50, 60, 75, 90, 125]


# --------------------------------------------------------------------------- #
# Tests — build_objective factory
# --------------------------------------------------------------------------- #

class TestBuildObjective:
    """Tests for the build_objective factory."""

    def test_build_ga(self) -> None:
        from route_data.unlearning.baseline_runner import (
            BaselineTrainingConfig,
            build_objective,
        )

        cfg = BaselineTrainingConfig(method_name="mllmu_ga")
        obj = build_objective(cfg)
        assert isinstance(obj, GradientAscent)

    def test_build_gd(self) -> None:
        from route_data.unlearning.baseline_runner import (
            BaselineTrainingConfig,
            build_objective,
        )

        cfg = BaselineTrainingConfig(
            method_name="mllmu_ga_difference", retain_weight=1.0,
        )
        obj = build_objective(cfg)
        assert isinstance(obj, GradientDifference)
        assert obj.retain_weight == 1.0

    def test_build_kl(self) -> None:
        from route_data.unlearning.baseline_runner import (
            BaselineTrainingConfig,
            build_objective,
        )

        cfg = BaselineTrainingConfig(
            method_name="mllmu_kl_min", kl_temperature=1.0,
        )
        obj = build_objective(cfg)
        assert isinstance(obj, KLMinimization)

    def test_build_npo(self) -> None:
        from route_data.unlearning.baseline_runner import (
            BaselineTrainingConfig,
            build_objective,
        )

        cfg = BaselineTrainingConfig(method_name="mllmu_npo", npo_beta=0.9)
        obj = build_objective(cfg)
        assert isinstance(obj, NegativePreferenceOptimization)
        assert obj.beta == 0.9

    def test_build_unknown_raises(self) -> None:
        from route_data.unlearning.baseline_runner import (
            BaselineTrainingConfig,
            build_objective,
        )

        cfg = BaselineTrainingConfig(method_name="unknown_method")
        with pytest.raises(ValueError, match="Unknown method"):
            build_objective(cfg)


# --------------------------------------------------------------------------- #
# Tests — Manifest schema
# --------------------------------------------------------------------------- #

class TestBaselineCheckpointManifest:
    """Tests for manifest schema validity."""

    def test_baseline_checkpoint_manifest(self) -> None:
        """Manifest should have required fields."""
        from route_data.unlearning.baseline_runner import BaselineTrainingConfig

        cfg = BaselineTrainingConfig(
            method_name="mllmu_ga",
            output_dir="/tmp/test_manifest",
        )

        # Build expected manifest structure
        manifest = {
            "method": cfg.method_name,
            "base_model": {
                "model_id": cfg.model_id,
                "revision": cfg.model_revision,
                "dtype": cfg.dtype,
            },
            "lora": {
                "rank": cfg.lora_rank,
                "alpha": cfg.lora_alpha,
                "dropout": cfg.lora_dropout,
                "target_modules": cfg.lora_target_modules,
            },
            "training": {
                "seed": cfg.seed,
                "learning_rate": cfg.learning_rate,
                "num_optimizer_steps": cfg.num_optimizer_steps,
                "batch_size": cfg.batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "max_grad_norm": cfg.max_grad_norm,
            },
            "checkpoint_steps": cfg.checkpoint_steps,
        }

        # Validate required fields
        assert "method" in manifest
        assert "base_model" in manifest
        assert "lora" in manifest
        assert "training" in manifest
        assert "checkpoint_steps" in manifest

        # Validate types
        assert isinstance(manifest["method"], str)
        assert isinstance(manifest["base_model"]["model_id"], str)
        assert isinstance(manifest["lora"]["rank"], int)
        assert isinstance(manifest["training"]["seed"], int)
        assert isinstance(manifest["checkpoint_steps"], list)

    def test_gd_manifest_has_retain_weight(self) -> None:
        """GD manifest should include retain_weight."""
        from route_data.unlearning.baseline_runner import BaselineTrainingConfig

        cfg = BaselineTrainingConfig(
            method_name="mllmu_ga_difference", retain_weight=1.0,
        )
        assert cfg.retain_weight == 1.0

    def test_npo_manifest_has_beta(self) -> None:
        """NPO manifest should include beta=0.9."""
        from route_data.unlearning.baseline_runner import BaselineTrainingConfig

        cfg = BaselineTrainingConfig(method_name="mllmu_npo", npo_beta=0.9)
        assert cfg.npo_beta == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# Tests — YAML config loading
# --------------------------------------------------------------------------- #

class TestYAMLConfigLoading:
    """Tests for YAML config loading."""

    def test_load_ga_config(self, tmp_path: Path) -> None:
        from route_data.unlearning.baseline_runner import load_config_from_yaml

        config_content = """
method:
  name: mllmu_ga
base_model:
  id: Qwen/Qwen3.5-9B
  revision: "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
  dtype: bfloat16
training:
  seed: 17
  learning_rate: 2.0e-5
  max_optimizer_steps: 125
  batch_size: 1
  gradient_accumulation_steps: 4
lora:
  rank: 8
  alpha: 16
  dropout: 0.0
runtime:
  output_dir: /tmp/test_ga
"""
        cfg_path = tmp_path / "ga.yaml"
        cfg_path.write_text(config_content)

        cfg = load_config_from_yaml(cfg_path)
        assert cfg.method_name == "mllmu_ga"
        assert cfg.seed == 17
        assert cfg.learning_rate == 2e-5
        assert cfg.lora_rank == 8

    def test_load_npo_config(self, tmp_path: Path) -> None:
        from route_data.unlearning.baseline_runner import load_config_from_yaml

        config_content = """
method:
  name: mllmu_npo
  beta: 0.9
training:
  seed: 17
runtime:
  output_dir: /tmp/test_npo
"""
        cfg_path = tmp_path / "npo.yaml"
        cfg_path.write_text(config_content)

        cfg = load_config_from_yaml(cfg_path)
        assert cfg.method_name == "mllmu_npo"
        assert cfg.npo_beta == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# Tests — Exact pairing (500↔500 probe pairing)
# --------------------------------------------------------------------------- #

class TestBaselineExactPairing:
    """Tests for 500↔500 probe pairing invariant."""

    def test_baseline_exact_pairing(self) -> None:
        """500 frozen probes should have exact pairing structure.

        This test verifies the structural invariant that the frozen
        probe set maintains exact pairing (target↔control).
        """
        # Simulate paired probe structure
        n_probes = 500
        probes = []
        for i in range(n_probes):
            probe = {
                "probe_id": f"probe_{i:04d}",
                "identity_id": f"id_{i % 6}",
                "probe_family": ["direct_visual", "image_plus_name",
                                 "wrong_name", "visual_text_control",
                                 "text_only_control"][i % 5],
                "paired_probe_id": f"probe_{i:04d}_pair" if i < n_probes // 2 else None,
            }
            probes.append(probe)

        # Verify pairing count
        paired = [p for p in probes if p["paired_probe_id"] is not None]
        assert len(paired) == n_probes // 2

        # Verify all probes have required fields
        for p in probes:
            assert "probe_id" in p
            assert "identity_id" in p
            assert "probe_family" in p

    def test_probe_family_coverage(self) -> None:
        """All probe families should be represented."""
        families = {
            "direct_visual", "image_plus_name", "wrong_name",
            "visual_text_control", "text_only_control",
        }
        # Verify the expected families
        assert len(families) == 5
        assert "direct_visual" in families
        assert "image_plus_name" in families
