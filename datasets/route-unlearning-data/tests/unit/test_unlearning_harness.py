"""Unit tests for the unlearning harness (Stage 3, Commit 2).

These tests use synthetic data and a tiny model stub so that no GPU
or real model weights are required.  They verify the structural
invariants of the training pipeline before any real intervention.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from route_data.eval.unlearning_harness import (
    UnlearningConfig,
    build_forget_dataset,
    build_retain_dataset,
    generate_run_manifest,
    generate_trainable_parameter_report,
    sha256_file,
)

# --------------------------------------------------------------------------- #
# Helpers – synthetic data
# --------------------------------------------------------------------------- #

def _write_processed_dataset(
    path: Path,
    identities: dict[str, list[dict]],
) -> None:
    """Write a minimal processed-dataset JSONL.

    Parameters
    ----------
    path:
        Output path.
    identities:
        Mapping from identity_id to list of sample dicts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for iid, samples in identities.items():
            for s in samples:
                row = {
                    "identity_id": iid,
                    "identity_name": s.get("name", f"Person_{iid[:8]}"),
                    "image_uri": s.get("image_uri", f"/fake/{iid}.jpg"),
                    "image_sha256": s.get("image_sha256", f"sha_{iid}"),
                    "question": s.get("question", "Is the person bald?"),
                    "answer_text": s.get("answer_text", "no"),
                    "answer_label": s.get("answer_label"),
                    "modality": "image_text",
                    "task_type": "private_profile_vqa",
                    "benchmark": "fiubench",
                }
                fh.write(json.dumps(row) + "\n")


def _make_mock_processor() -> MagicMock:
    """Create a mock processor that returns valid tensors."""
    import torch

    processor = MagicMock()

    def mock_apply_chat_template(messages, **kwargs):
        return "<mock prompt>"

    processor.apply_chat_template = mock_apply_chat_template

    def mock_call(text=None, images=None, return_tensors="pt", **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "pixel_values": torch.zeros(1, 3, 224, 224),
        }

    processor.__call__ = mock_call
    return processor


# --------------------------------------------------------------------------- #
# Tests – configuration
# --------------------------------------------------------------------------- #

class TestUnlearningConfig:
    """Tests for :class:`UnlearningConfig`."""

    def test_defaults(self) -> None:
        cfg = UnlearningConfig()
        assert cfg.model_id == "Qwen/Qwen3.5-9B"
        assert cfg.lora_rank == 8
        assert cfg.lora_alpha == 16
        assert cfg.learning_rate == 1e-4
        assert cfg.num_steps == 50
        assert cfg.retain_weight == 0.1
        assert cfg.seed == 17

    def test_effective_batch_size(self) -> None:
        cfg = UnlearningConfig(batch_size=2, gradient_accumulation_steps=4)
        assert cfg.effective_batch_size == 8

    def test_custom_values(self) -> None:
        cfg = UnlearningConfig(
            lora_rank=16,
            learning_rate=5e-5,
            forget_identity_ids=["a", "b"],
            retain_identity_ids=["c", "d"],
        )
        assert cfg.lora_rank == 16
        assert cfg.learning_rate == 5e-5
        assert len(cfg.forget_identity_ids) == 2


# --------------------------------------------------------------------------- #
# Tests – dataset builders
# --------------------------------------------------------------------------- #

class TestDatasetBuilders:
    """Tests for forget/retain dataset construction."""

    def test_forget_dataset_filters_by_identity(self, tmp_path: Path) -> None:
        ds_path = tmp_path / "processed.jsonl"
        identities = {
            "target_1": [{"question": f"Q{i}"} for i in range(5)],
            "target_2": [{"question": f"Q{i}"} for i in range(3)],
            "other_1": [{"question": f"Q{i}"} for i in range(10)],
        }
        _write_processed_dataset(ds_path, identities)

        processor = _make_mock_processor()
        ds = build_forget_dataset(ds_path, ["target_1", "target_2"], processor)

        assert len(ds) == 8  # 5 + 3 from target identities

    def test_retain_dataset_filters_by_identity(self, tmp_path: Path) -> None:
        ds_path = tmp_path / "processed.jsonl"
        identities = {
            "target_1": [{"question": f"Q{i}"} for i in range(5)],
            "retain_1": [{"question": f"Q{i}"} for i in range(4)],
            "retain_2": [{"question": f"Q{i}"} for i in range(6)],
        }
        _write_processed_dataset(ds_path, identities)

        processor = _make_mock_processor()
        ds = build_retain_dataset(ds_path, ["retain_1", "retain_2"], processor)

        assert len(ds) == 10  # 4 + 6 from retain identities

    def test_forget_retain_no_overlap(self, tmp_path: Path) -> None:
        """Forget and retain datasets must not share identities."""
        ds_path = tmp_path / "processed.jsonl"
        identities = {
            "target_1": [{"question": "Q"} for _ in range(5)],
            "retain_1": [{"question": "Q"} for _ in range(5)],
            "other": [{"question": "Q"} for _ in range(5)],
        }
        _write_processed_dataset(ds_path, identities)

        processor = _make_mock_processor()
        forget_ds = build_forget_dataset(ds_path, ["target_1"], processor)
        retain_ds = build_retain_dataset(ds_path, ["retain_1"], processor)

        forget_ids = {s["identity_id"] for s in forget_ds.samples}
        retain_ids = {s["identity_id"] for s in retain_ds.samples}
        assert forget_ids & retain_ids == set()

    def test_max_samples_cap(self, tmp_path: Path) -> None:
        ds_path = tmp_path / "processed.jsonl"
        identities = {
            "target_1": [{"question": f"Q{i}"} for i in range(20)],
        }
        _write_processed_dataset(ds_path, identities)

        processor = _make_mock_processor()
        ds = build_forget_dataset(ds_path, ["target_1"], processor, max_samples=5)
        assert len(ds) == 5

    def test_deterministic_sampling(self, tmp_path: Path) -> None:
        """Same seed → same dataset ordering."""
        ds_path = tmp_path / "processed.jsonl"
        identities = {
            "target_1": [{"question": f"Q{i}"} for i in range(10)],
        }
        _write_processed_dataset(ds_path, identities)

        processor = _make_mock_processor()
        ds_a = build_forget_dataset(ds_path, ["target_1"], processor, seed=17)
        ds_b = build_forget_dataset(ds_path, ["target_1"], processor, seed=17)

        for a, b in zip(ds_a.samples, ds_b.samples):
            assert a["question"] == b["question"]


# --------------------------------------------------------------------------- #
# Tests – probe leakage rejection
# --------------------------------------------------------------------------- #

class TestProbeLeakage:
    """Tests that evaluation probes do not leak into training data."""

    def test_forget_dataset_excludes_non_target(self, tmp_path: Path) -> None:
        """Only target identities appear in the forget dataset."""
        ds_path = tmp_path / "processed.jsonl"
        identities = {
            "target_1": [{"question": "Q"} for _ in range(5)],
            "control_1": [{"question": "Q"} for _ in range(5)],
            "eval_only": [{"question": "Q"} for _ in range(5)],
        }
        _write_processed_dataset(ds_path, identities)

        processor = _make_mock_processor()
        ds = build_forget_dataset(ds_path, ["target_1"], processor)

        identity_ids = {s["identity_id"] for s in ds.samples}
        assert identity_ids == {"target_1"}
        assert "control_1" not in identity_ids
        assert "eval_only" not in identity_ids


# --------------------------------------------------------------------------- #
# Tests – trainable parameter report
# --------------------------------------------------------------------------- #

class TestTrainableParameterReport:
    """Tests for :func:`generate_trainable_parameter_report`."""

    def test_report_structure(self) -> None:
        """Report has expected keys and types."""
        model = MagicMock()
        # Mock parameters
        param1 = MagicMock()
        param1.numel.return_value = 1000
        param1.requires_grad = True

        param2 = MagicMock()
        param2.numel.return_value = 2000
        param2.requires_grad = False

        param3 = MagicMock()
        param3.numel.return_value = 500
        param3.requires_grad = True

        model.parameters.return_value = [param1, param2, param3]
        model.named_parameters.return_value = [
            ("layer1.weight", param1),
            ("layer2.weight", param2),
            ("lora_A.weight", param3),
        ]

        report = generate_trainable_parameter_report(model)

        assert report["total_parameters"] == 3500
        assert report["trainable_parameters"] == 1500
        assert 0 < report["trainable_percentage"] < 100
        assert "trainable_module_count" in report
        assert "trainable_modules" in report

    def test_all_frozen(self) -> None:
        """Report handles all-frozen model."""
        model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 1000
        param.requires_grad = False

        model.parameters.return_value = [param]
        model.named_parameters.return_value = [("layer.weight", param)]

        report = generate_trainable_parameter_report(model)
        assert report["trainable_parameters"] == 0
        assert report["trainable_percentage"] == 0.0


# --------------------------------------------------------------------------- #
# Tests – run manifest
# --------------------------------------------------------------------------- #

class TestRunManifest:
    """Tests for :func:`generate_run_manifest`."""

    def test_manifest_structure(self) -> None:
        cfg = UnlearningConfig(
            forget_identity_ids=["t1", "t2"],
            retain_identity_ids=["r1", "r2"],
            selection_manifest_sha256="abc123",
        )
        training_summary = {
            "total_steps": 50,
            "final_loss": 2.5,
            "elapsed_seconds": 120.0,
            "checkpoints_saved": 5,
        }
        param_report = {
            "total_parameters": 9_000_000_000,
            "trainable_parameters": 5_000_000,
            "trainable_percentage": 0.056,
            "trainable_module_count": 32,
            "trainable_modules": ["lora_A"],
        }

        manifest = generate_run_manifest(
            cfg, training_summary, param_report,
            code_commit="abc123", git_dirty=False,
        )

        assert manifest["experiment_id"] == "fiubench_unlearning_pilot_v1"
        assert manifest["base_model"]["model_id"] == "Qwen/Qwen3.5-9B"
        assert manifest["method"]["name"] == "lora_targeted_update"
        assert manifest["method"]["hyperparameters"]["lora_rank"] == 8
        assert manifest["seed"] == 17
        assert manifest["forget_identities"] == ["t1", "t2"]
        assert manifest["retain_identities"] == ["r1", "r2"]
        assert manifest["training_summary"]["total_steps"] == 50
        assert manifest["code_provenance"]["git_commit"] == "abc123"
        assert manifest["code_provenance"]["git_dirty"] is False

    def test_manifest_has_hyperparameters(self) -> None:
        cfg = UnlearningConfig()
        manifest = generate_run_manifest(
            cfg, {}, {}, code_commit="x", git_dirty=False,
        )
        hp = manifest["method"]["hyperparameters"]
        assert "lora_rank" in hp
        assert "lora_alpha" in hp
        assert "learning_rate" in hp
        assert "num_steps" in hp
        assert "retain_weight" in hp


# --------------------------------------------------------------------------- #
# Tests – SHA-256 utility
# --------------------------------------------------------------------------- #

class TestSha256:
    """Tests for :func:`sha256_file`."""

    def test_sha256_stable(self, tmp_path: Path) -> None:
        p = tmp_path / "test.txt"
        p.write_text("hello world")
        sha1 = sha256_file(p)
        sha2 = sha256_file(p)
        assert sha1 == sha2
        assert len(sha1) == 64

    def test_sha256_changes_with_content(self, tmp_path: Path) -> None:
        p = tmp_path / "test.txt"
        p.write_text("hello")
        sha1 = sha256_file(p)
        p.write_text("world")
        sha2 = sha256_file(p)
        assert sha1 != sha2


# --------------------------------------------------------------------------- #
# Tests – intervention dataset manifest
# --------------------------------------------------------------------------- #

class TestInterventionManifest:
    """Tests for the intervention dataset manifest."""

    def test_manifest_content(self, tmp_path: Path) -> None:
        """The intervention manifest records target identities and samples."""
        ds_path = tmp_path / "processed.jsonl"
        identities = {
            "target_1": [
                {"question": "Q1", "image_uri": "/img1.jpg"},
                {"question": "Q2", "image_uri": "/img2.jpg"},
            ],
            "target_2": [
                {"question": "Q3", "image_uri": "/img3.jpg"},
            ],
        }
        _write_processed_dataset(ds_path, identities)

        # Build a simple manifest
        manifest = {
            "target_identities": ["target_1", "target_2"],
            "sample_count": 3,
            "data_source": "fiubench_processed",
            "dataset_path": str(ds_path),
            "selection_seed": 17,
        }

        manifest_path = tmp_path / "intervention_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        loaded = json.loads(manifest_path.read_text())
        assert loaded["target_identities"] == ["target_1", "target_2"]
        assert loaded["sample_count"] == 3
        assert loaded["selection_seed"] == 17
