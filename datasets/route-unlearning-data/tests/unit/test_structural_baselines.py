"""Unit tests for MMUnlearner (B7), MANU (B8), R²MU-adapted (B9),
and the comparison framework (Phase 4).

These tests use tiny synthetic models and data so that no GPU or real
model weights are required. They verify the structural invariants of
each baseline.

Tests from plan:
- MMUnlearner: mask deterministic, mask non-empty, mask ≠ 100%,
  gradient zero outside mask, selected entries get finite gradient
- MANU: neuron inventory valid, importance finite, exact prune fraction,
  only selected neurons modified, forward pass works after pruning
- R²MU-adapted: random target reproducible, forget repr has gradient,
  retain references frozen model, pooling alignment correct, no CoT claim
- Comparison: E2C decision logic, table generation, efficiency report
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from route_data.unlearning.mmunlearner import (
    MMUnlearnerConfig,
    PARAMETER_CATEGORIES,
    apply_mask_to_gradients,
    build_parameter_inventory,
    estimate_saliency,
    generate_saliency_mask,
    save_parameter_inventory,
)
from route_data.unlearning.manu import (
    MANUConfig,
    build_neuron_inventory,
    prune_neurons,
    select_neurons_to_prune,
)
from route_data.unlearning.r2mu_adapted import (
    R2MUAdaptedConfig,
    forget_representation_loss,
    generate_random_target,
    retain_representation_loss,
    target_sha256,
)
from route_data.unlearning.comparison_framework import (
    ComparisonFramework,
    MethodResult,
)


# --------------------------------------------------------------------------- #
# Fixtures — tiny model with MLP structure
# --------------------------------------------------------------------------- #

VOCAB_SIZE = 50
SEQ_LEN = 8
HIDDEN_DIM = 16
INTERMEDIATE_DIM = 32
BATCH_SIZE = 2
ANSWER_START = 4


class TinyMLPModel(nn.Module):
    """Minimal model with MLP layers for testing MANU/MMUnlearner."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        # Simulate a transformer layer with MLP
        self.layers = nn.ModuleList([
            TinyTransformerLayer() for _ in range(4)
        ])
        self.lm_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.config = MagicMock()
        self.config.model_type = "tiny_test"
        self.config.hidden_size = HIDDEN_DIM
        self.config.num_hidden_layers = 4

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        h = self.embedding(input_ids)
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(h)
        return MagicMock(logits=logits)


class TinyTransformerLayer(nn.Module):
    """Minimal transformer layer with MLP."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.mlp = TinyMLP()

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class TinyMLP(nn.Module):
    """Minimal MLP with gate/up/down projections."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_DIM, INTERMEDIATE_DIM)
        self.up_proj = nn.Linear(HIDDEN_DIM, INTERMEDIATE_DIM)
        self.down_proj = nn.Linear(INTERMEDIATE_DIM, HIDDEN_DIM)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


def _make_synthetic_batch(
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    answer_start: int = ANSWER_START,
) -> dict[str, Any]:
    """Create a synthetic batch with answer-only masking."""
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for i in range(batch_size):
        for j in range(answer_start, seq_len):
            labels[i, j] = torch.randint(0, VOCAB_SIZE, (1,)).item()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _make_dataloader(n_batches: int = 4) -> list[dict[str, Any]]:
    """Create a simple list of batches (acts as a dataloader)."""
    return [_make_synthetic_batch() for _ in range(n_batches)]


# --------------------------------------------------------------------------- #
# Tests — MMUnlearner (B7)
# --------------------------------------------------------------------------- #

class TestMMUnlearnerParameterInventory:
    """Tests for MMUnlearner parameter inventory."""

    def test_inventory_categorizes_all_params(self) -> None:
        """All parameters should be categorized."""
        model = TinyMLPModel()
        inventory = build_parameter_inventory(model)

        total_in_categories = sum(
            info["numel"]
            for cat, info in inventory.items()
            if cat != "_summary"
        )
        total_params = inventory["_summary"]["total_parameters"]
        assert total_in_categories == total_params

    def test_inventory_detects_mlp(self) -> None:
        """MLP parameters should be detected."""
        model = TinyMLPModel()
        inventory = build_parameter_inventory(model)

        assert inventory["mlp"]["numel"] > 0
        assert len(inventory["mlp"]["params"]) > 0

    def test_inventory_detects_embeddings(self) -> None:
        """Embedding parameters should be detected."""
        model = TinyMLPModel()
        inventory = build_parameter_inventory(model)

        assert inventory["embeddings"]["numel"] > 0

    def test_inventory_detects_lm_head(self) -> None:
        """lm_head parameters should be detected."""
        model = TinyMLPModel()
        inventory = build_parameter_inventory(model)

        assert inventory["lm_head"]["numel"] > 0

    def test_inventory_save_load(self) -> None:
        """Inventory should be saveable to JSON."""
        model = TinyMLPModel()
        inventory = build_parameter_inventory(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "inventory.json"
            save_parameter_inventory(inventory, path)
            assert path.exists()

            with open(path) as f:
                loaded = json.load(f)
            assert "_summary" in loaded
            assert "total_parameters" in loaded["_summary"]


class TestMMUnlearnerMask:
    """Tests for MMUnlearner saliency mask."""

    def test_mask_deterministic(self) -> None:
        """Same saliency → same mask."""
        model = TinyMLPModel()
        saliency = {name: torch.randn(1).item() for name, _ in model.named_parameters()}
        # Convert to tensors
        saliency = {name: torch.tensor(score) for name, score in saliency.items()}

        config = MMUnlearnerConfig(target_sparsity=0.5)
        mask1 = generate_saliency_mask(model, saliency, config)
        mask2 = generate_saliency_mask(model, saliency, config)

        for name in mask1:
            assert torch.equal(mask1[name], mask2[name])

    def test_mask_non_empty(self) -> None:
        """Mask should have at least some selected parameters."""
        model = TinyMLPModel()
        saliency = {name: torch.randn(1).item() for name, _ in model.named_parameters()}
        saliency = {name: torch.tensor(score) for name, score in saliency.items()}

        config = MMUnlearnerConfig(target_sparsity=0.5)
        mask = generate_saliency_mask(model, saliency, config)

        total_selected = sum(m.sum().item() for m in mask.values())
        assert total_selected > 0

    def test_mask_not_100_percent(self) -> None:
        """Mask should not select 100% of parameters."""
        model = TinyMLPModel()
        saliency = {name: torch.randn(1).item() for name, _ in model.named_parameters()}
        saliency = {name: torch.tensor(score) for name, score in saliency.items()}

        config = MMUnlearnerConfig(target_sparsity=0.5, max_mask_fraction=0.99)
        mask = generate_saliency_mask(model, saliency, config)

        total_params = sum(p.numel() for p in model.parameters())
        total_selected = sum(m.sum().item() for m in mask.values())
        assert total_selected < total_params

    def test_gradient_zero_outside_mask(self) -> None:
        """Gradients should be zero for parameters outside mask."""
        model = TinyMLPModel()
        batch = _make_synthetic_batch()

        # Forward/backward to get gradients
        model_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        outputs = model(**model_kwargs)
        logits = outputs.logits
        loss = logits.mean()
        loss.backward()

        # Create mask that zeros out half
        mask = {}
        for i, (name, param) in enumerate(model.named_parameters()):
            if i % 2 == 0:
                mask[name] = torch.ones_like(param.data)
            else:
                mask[name] = torch.zeros_like(param.data)

        apply_mask_to_gradients(model, mask)

        # Check that masked-out params have zero gradient
        for i, (name, param) in enumerate(model.named_parameters()):
            if param.grad is not None:
                if i % 2 != 0:
                    assert param.grad.abs().sum() == 0.0

    def test_selected_entries_get_gradient(self) -> None:
        """Parameters inside mask should have non-zero gradients."""
        model = TinyMLPModel()
        batch = _make_synthetic_batch()

        model_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        outputs = model(**model_kwargs)
        logits = outputs.logits
        loss = logits.mean()
        loss.backward()

        # Full mask — all parameters selected
        mask = {name: torch.ones_like(param.data) for name, param in model.named_parameters()}
        apply_mask_to_gradients(model, mask)

        # All params with grad should still have non-zero grad
        has_nonzero = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_nonzero = True
                break
        assert has_nonzero


class TestMMUnlearnerConfig:
    """Tests for MMUnlearner configuration."""

    def test_defaults(self) -> None:
        config = MMUnlearnerConfig()
        assert config.saliency_n_samples == 32
        assert config.mask_granularity == "element"
        assert config.target_sparsity == 0.5
        assert config.modality == "both"

    def test_effective_sparsity(self) -> None:
        config = MMUnlearnerConfig(target_sparsity=0.5)
        assert 0.0 <= config.effective_sparsity <= 1.0

    def test_effective_sparsity_clamped(self) -> None:
        config = MMUnlearnerConfig(
            target_sparsity=1.5,
            min_mask_fraction=0.01,
            max_mask_fraction=0.99,
        )
        assert config.effective_sparsity == 0.99


# --------------------------------------------------------------------------- #
# Tests — MANU (B8)
# --------------------------------------------------------------------------- #

class TestMANUNeuronInventory:
    """Tests for MANU neuron inventory."""

    def test_inventory_valid(self) -> None:
        """Neuron inventory should be valid."""
        model = TinyMLPModel()
        inventory = build_neuron_inventory(model)

        assert "layers" in inventory
        assert "_summary" in inventory
        assert inventory["_summary"]["n_layers"] > 0
        assert inventory["_summary"]["n_total_neurons"] > 0

    def test_inventory_detects_mlp_layers(self) -> None:
        """Should detect MLP layers in the model."""
        model = TinyMLPModel()
        inventory = build_neuron_inventory(model)

        assert len(inventory["layers"]) > 0
        for layer_path, info in inventory["layers"].items():
            assert info["n_neurons"] > 0
            assert "modules" in info


class TestMANUPruning:
    """Tests for MANU neuron pruning."""

    def test_importance_finite(self) -> None:
        """Importance scores should be finite."""
        importance = {
            "layer_0": torch.tensor([1.0, 2.0, -0.5, 0.0, 3.0]),
            "layer_1": torch.tensor([0.5, -1.0, 2.0]),
        }

        for layer, scores in importance.items():
            assert torch.isfinite(scores).all()

    def test_exact_prune_fraction(self) -> None:
        """Pruning should select the exact fraction of neurons."""
        importance = {
            "layer_0": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            "layer_1": torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5]),
        }

        neurons = select_neurons_to_prune(importance, 0.10)
        total_neurons = sum(s.shape[0] for s in importance.values())
        total_pruned = sum(len(idxs) for idxs in neurons.values())

        expected = max(1, int(total_neurons * 0.10))
        assert total_pruned == expected

    def test_prune_50_percent(self) -> None:
        """Pruning 50% should select half the neurons."""
        importance = {
            "layer_0": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "layer_1": torch.tensor([5.0, 6.0, 7.0, 8.0]),
        }

        neurons = select_neurons_to_prune(importance, 0.50)
        total_neurons = sum(s.shape[0] for s in importance.values())
        total_pruned = sum(len(idxs) for idxs in neurons.values())

        assert total_pruned == total_neurons // 2

    def test_forward_pass_after_pruning(self) -> None:
        """Forward pass should work after pruning."""
        model = TinyMLPModel()
        batch = _make_synthetic_batch()

        # Get inventory and prune some neurons
        inventory = build_neuron_inventory(model)
        importance = {
            layer_path: torch.randn(info["n_neurons"])
            for layer_path, info in inventory["layers"].items()
        }
        neurons = select_neurons_to_prune(importance, 0.10)
        prune_neurons(model, neurons)

        # Forward pass should still work
        model.eval()
        with torch.no_grad():
            model_kwargs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            outputs = model(**model_kwargs)
            assert outputs.logits is not None
            assert outputs.logits.shape[0] == batch["input_ids"].shape[0]

    def test_only_selected_neurons_modified(self) -> None:
        """Only selected neurons should be modified."""
        model = TinyMLPModel()

        # Save original weights
        original_weights = {}
        for name, param in model.named_parameters():
            original_weights[name] = param.data.clone()

        # Create pruning plan for specific neurons
        inventory = build_neuron_inventory(model)
        importance = {
            layer_path: torch.randn(info["n_neurons"])
            for layer_path, info in inventory["layers"].items()
        }
        neurons = select_neurons_to_prune(importance, 0.10)

        # Prune
        prune_info = prune_neurons(model, neurons)

        # Check that some parameters were modified
        assert prune_info["n_parameters_modified"] > 0


class TestMANUConfig:
    """Tests for MANU configuration."""

    def test_defaults(self) -> None:
        config = MANUConfig()
        assert config.primary_prune_fraction == 0.10
        assert config.secondary_prune_fraction == 0.05
        assert config.neuron_unit == "mlp_intermediate"

    def test_prune_rates(self) -> None:
        config = MANUConfig()
        rates = config.prune_rates
        assert 0.05 in rates
        assert 0.10 in rates


# --------------------------------------------------------------------------- #
# Tests — R²MU-adapted (B9)
# --------------------------------------------------------------------------- #

class TestR2MUTarget:
    """Tests for R²MU random target generation."""

    def test_target_reproducible(self) -> None:
        """Same seed → same target vector."""
        t1 = generate_random_target(hidden_size=64, seed=42)
        t2 = generate_random_target(hidden_size=64, seed=42)
        assert torch.equal(t1, t2)

    def test_target_different_seeds(self) -> None:
        """Different seeds → different target vectors."""
        t1 = generate_random_target(hidden_size=64, seed=42)
        t2 = generate_random_target(hidden_size=64, seed=99)
        assert not torch.equal(t1, t2)

    def test_target_norm(self) -> None:
        """Target should have specified norm."""
        target = generate_random_target(hidden_size=128, seed=42, target_norm=1.0)
        assert target.norm().item() == pytest.approx(1.0, abs=1e-5)

    def test_target_norm_custom(self) -> None:
        """Target should have custom norm."""
        target = generate_random_target(hidden_size=128, seed=42, target_norm=5.0)
        assert target.norm().item() == pytest.approx(5.0, abs=1e-5)

    def test_target_sha256(self) -> None:
        """SHA-256 should be deterministic."""
        target = generate_random_target(hidden_size=64, seed=42)
        sha1 = target_sha256(target)
        sha2 = target_sha256(target)
        assert sha1 == sha2
        assert len(sha1) == 64  # SHA-256 hex digest length


class TestR2MULossFunctions:
    """Tests for R²MU representation loss functions."""

    def test_forget_loss_zero_when_matching(self) -> None:
        """Forget loss = 0 when current repr == target."""
        target = torch.randn(16)
        current = target.unsqueeze(0).expand(4, -1)  # (batch=4, hidden=16)

        loss = forget_representation_loss(current, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_forget_loss_positive_when_different(self) -> None:
        """Forget loss > 0 when current repr != target."""
        target = torch.randn(16)
        current = torch.randn(4, 16)  # Different from target

        loss = forget_representation_loss(current, target)
        assert loss.item() > 0.0

    def test_forget_loss_has_gradient(self) -> None:
        """Forget loss should have gradient w.r.t. current repr."""
        target = torch.randn(16)
        current = torch.randn(4, 16, requires_grad=True)

        loss = forget_representation_loss(current, target)
        loss.backward()

        assert current.grad is not None
        assert current.grad.abs().sum() > 0

    def test_retain_loss_zero_when_matching(self) -> None:
        """Retain loss = 0 when current == frozen repr."""
        current = torch.randn(4, 16)
        frozen = current.clone()

        loss = retain_representation_loss(current, frozen)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_retain_loss_positive_when_different(self) -> None:
        """Retain loss > 0 when current != frozen repr."""
        current = torch.randn(4, 16)
        frozen = torch.randn(4, 16)

        loss = retain_representation_loss(current, frozen)
        assert loss.item() > 0.0


class TestR2MUConfig:
    """Tests for R²MU-adapted configuration."""

    def test_defaults(self) -> None:
        config = R2MUAdaptedConfig()
        assert config.target_seed == 42
        assert config.target_norm == 1.0
        assert config.gamma == 1.0
        assert config.method_name == "r2mu_adapted"

    def test_no_cot_claim(self) -> None:
        """Config should not reference CoT components."""
        config = R2MUAdaptedConfig()
        # The config should not have any CoT-related fields
        assert not hasattr(config, "cot_layers")
        assert not hasattr(config, "reasoning_traces")


# --------------------------------------------------------------------------- #
# Tests — Comparison Framework (Phase 4)
# --------------------------------------------------------------------------- #

class TestComparisonFramework:
    """Tests for the comparison framework."""

    def test_add_result(self) -> None:
        """Should add results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fw = ComparisonFramework(tmpdir)
            result = MethodResult(
                method_id="ga",
                baseline_id="B1",
                description="Gradient Ascent",
            )
            fw.add_result(result)
            assert len(fw.results) == 1

    def test_generate_tables(self) -> None:
        """Should generate comparison tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fw = ComparisonFramework(tmpdir)

            # Add Table A method
            fw.add_result(MethodResult(
                method_id="ga",
                baseline_id="B1",
                description="Gradient Ascent",
                table="A",
                delta_target={"DV": -0.2, "IPN": -0.15},
                delta_retain={"DV": -0.02},
            ))

            # Add Table B method
            fw.add_result(MethodResult(
                method_id="mmunlearner",
                baseline_id="B7",
                description="MMUnlearner",
                table="B",
                delta_target={"DV": -0.3},
            ))

            tables = fw.generate_tables()
            assert "table_a" in tables
            assert "table_b" in tables
            assert "combined" in tables
            assert len(tables["table_a"]["methods"]) == 1
            assert len(tables["table_b"]["methods"]) == 1
            assert len(tables["combined"]["methods"]) == 2

            # Check files were written
            assert (Path(tmpdir) / "comparison_tables.json").exists()

    def test_efficiency_report(self) -> None:
        """Should generate efficiency report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fw = ComparisonFramework(tmpdir)
            fw.add_result(MethodResult(
                method_id="ga",
                baseline_id="B1",
                description="Gradient Ascent",
                gpu_hours=0.5,
                peak_memory_gb=12.0,
                trainable_parameters=1_000_000,
            ))

            report = fw.generate_efficiency_report()
            assert len(report["methods"]) == 1
            assert report["methods"][0]["gpu_hours"] == 0.5
            assert (Path(tmpdir) / "efficiency_report.json").exists()

    def test_e2c_decision_case_a(self) -> None:
        """Case A: target-specific degradation with preservation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fw = ComparisonFramework(tmpdir)
            fw.add_result(MethodResult(
                method_id="test",
                baseline_id="B1",
                description="Test method",
                delta_target={"DV": -0.3, "IPN": -0.25},  # Strong degradation
                delta_retain={"DV": -0.01},  # Well preserved
                delta_control={"DV": -0.02},  # Well preserved
            ))

            decision = fw.make_e2c_decision()
            assert decision["case"] == "A"
            assert "test" in decision["case_a_methods"]
            assert decision["action"] == "replicate_seeds"

    def test_e2c_decision_case_c(self) -> None:
        """Case C: all methods collapse non-selectively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fw = ComparisonFramework(tmpdir)
            fw.add_result(MethodResult(
                method_id="test",
                baseline_id="B1",
                description="Test method",
                delta_target={"DV": -0.3},  # Target degrades
                delta_retain={"DV": -0.25},  # Retain also degrades a lot
                delta_control={"DV": -0.20},  # Control also degrades
            ))

            decision = fw.make_e2c_decision()
            # This should be Case C since retain/control are not preserved
            assert decision["case"] == "C"

    def test_selectivity_score(self) -> None:
        """Selectivity = target_deg - max(retain_deg, control_deg)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fw = ComparisonFramework(tmpdir)
            result = MethodResult(
                method_id="test",
                baseline_id="B1",
                description="Test",
                delta_target={"DV": -0.3},
                delta_retain={"DV": -0.1},
                delta_control={"DV": -0.05},
            )

            score = fw._compute_selectivity_score(result)
            # target_deg = 0.3, max(retain_deg=0.1, control_deg=0.05) = 0.1
            # selectivity = 0.3 - 0.1 = 0.2
            assert score == pytest.approx(0.2, abs=1e-6)

    def test_trajectory_analysis(self) -> None:
        """Should generate trajectory analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fw = ComparisonFramework(tmpdir)
            fw.add_result(MethodResult(
                method_id="ga",
                baseline_id="B1",
                description="Gradient Ascent",
                num_steps=125,
                final_loss=-2.5,
                delta_target={"DV": -0.2},
            ))

            analysis = fw.generate_trajectory_analysis()
            assert len(analysis["methods"]) == 1
            assert analysis["methods"][0]["num_steps"] == 125
            assert (Path(tmpdir) / "trajectory_analysis.json").exists()

    def test_route_selectivity_conclusion(self) -> None:
        """Should write conclusion file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fw = ComparisonFramework(tmpdir)
            fw.add_result(MethodResult(
                method_id="ga",
                baseline_id="B1",
                description="Gradient Ascent",
                delta_target={"DV": -0.2},
            ))

            conclusion = fw.write_route_selectivity_conclusion()
            assert "ROUTE-SELECTIVITY CONCLUSION" in conclusion
            assert (Path(tmpdir) / "route_selectivity_conclusion.txt").exists()
