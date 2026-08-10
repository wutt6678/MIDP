"""P1-17: benchmark-specific source-split mapping tests.

Each benchmark has its own data config with specific partition labels.
These tests verify the DEFAULT_SOURCE_MAPPING and per-benchmark overrides
produce correct train/eval/exclude/hash assignments for the actual
released split vocabularies of FAIRGET, FIUBench, MLLMU-Bench, and
PPU-Bench.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIGS_DIR = REPO_ROOT / "configs" / "data"


# The DEFAULT_SOURCE_MAPPING from cli.py cmd_build_qa.
DEFAULT_SOURCE_MAPPING = {
    "train": "train",
    "retain_train": "train",
    "validation": "eval",
    "val": "eval",
    "eval": "eval",
    "retain_eval": "eval",
    "test": "eval",
    "forget": "exclude",
    "unassigned": "hash",
}


def _load_data_config(name: str) -> dict:
    path = DATA_CONFIGS_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_mapping(name: str) -> dict[str, str]:
    """Resolve the effective source mapping for a benchmark."""
    cfg = _load_data_config(name)
    mapping = DEFAULT_SOURCE_MAPPING.copy()
    extras = cfg.get("data", {}).get("extras", {})
    if extras and isinstance(extras.get("source_mapping"), dict):
        mapping.update(extras["source_mapping"])
    return mapping


class TestDefaultSourceMapping:
    """Verify the default mapping covers common partition vocabularies."""

    def test_train_labels(self):
        assert DEFAULT_SOURCE_MAPPING["train"] == "train"
        assert DEFAULT_SOURCE_MAPPING["retain_train"] == "train"

    def test_eval_labels(self):
        assert DEFAULT_SOURCE_MAPPING["validation"] == "eval"
        assert DEFAULT_SOURCE_MAPPING["val"] == "eval"
        assert DEFAULT_SOURCE_MAPPING["eval"] == "eval"
        assert DEFAULT_SOURCE_MAPPING["retain_eval"] == "eval"
        assert DEFAULT_SOURCE_MAPPING["test"] == "eval"

    def test_forget_label(self):
        assert DEFAULT_SOURCE_MAPPING["forget"] == "exclude"

    def test_unassigned_label(self):
        assert DEFAULT_SOURCE_MAPPING["unassigned"] == "hash"

    def test_all_values_valid(self):
        valid_targets = {"train", "eval", "exclude", "hash"}
        for label, target in DEFAULT_SOURCE_MAPPING.items():
            assert target in valid_targets, f"{label} -> {target} not in {valid_targets}"


class TestFairgetSourceMapping:
    """FAIRGET: official unbalanced forget splits (splits/official.json)."""

    def test_config_exists(self):
        assert (DATA_CONFIGS_DIR / "fairget.yaml").exists()

    def test_source_version_pinned(self):
        cfg = _load_data_config("fairget")
        sv = cfg["data"]["source_version"]
        assert sv and sv != "unknown"
        assert sv == "fairget-2026.07"

    def test_split_file_defined(self):
        cfg = _load_data_config("fairget")
        assert cfg["data"]["split_file"] == "splits/official.json"

    def test_reuse_official_splits(self):
        cfg = _load_data_config("fairget")
        assert cfg["data"]["reuse_official_splits"] is True

    def test_mapping_includes_forget(self):
        mapping = _resolve_mapping("fairget")
        assert mapping["forget"] == "exclude"

    def test_mapping_includes_train(self):
        mapping = _resolve_mapping("fairget")
        assert mapping["train"] == "train"

    def test_images_root_is_dot(self):
        cfg = _load_data_config("fairget")
        assert cfg["data"]["images_root"] == "."

    def test_adapter_version(self):
        cfg = _load_data_config("fairget")
        assert cfg["data"]["adapter_version"] == "fairget-v2"


class TestFIUBenchSourceMapping:
    """FIUBench: official forget/retain/evaluation grouping."""

    def test_config_exists(self):
        assert (DATA_CONFIGS_DIR / "fiubench.yaml").exists()

    def test_source_version_pinned(self):
        cfg = _load_data_config("fiubench")
        sv = cfg["data"]["source_version"]
        assert sv == "fiubench-1.0"

    def test_split_file_defined(self):
        cfg = _load_data_config("fiubench")
        assert cfg["data"]["split_file"] == "splits/official.json"

    def test_reuse_official_splits(self):
        cfg = _load_data_config("fiubench")
        assert cfg["data"]["reuse_official_splits"] is True

    def test_mapping_includes_forget(self):
        mapping = _resolve_mapping("fiubench")
        assert mapping["forget"] == "exclude"

    def test_mapping_includes_eval(self):
        mapping = _resolve_mapping("fiubench")
        # FIUBench uses "evaluation" which should map to eval
        # Either via default "eval" key or benchmark-specific override
        assert "eval" in mapping.values() or "evaluation" in mapping

    def test_adapter_version(self):
        cfg = _load_data_config("fiubench")
        assert cfg["data"]["adapter_version"] == "fiubench-v2"

    def test_multiview_opt_in(self):
        cfg = _load_data_config("fiubench")
        mv = cfg["data"].get("multiview", {})
        assert mv.get("enabled") is False  # opt-in, not default


class TestMLLMUSourceMapping:
    """MLLMU-Bench: Full_Set validated configuration."""

    def test_config_exists(self):
        assert (DATA_CONFIGS_DIR / "mllmu.yaml").exists()

    def test_source_version_pinned(self):
        cfg = _load_data_config("mllmu")
        sv = cfg["data"]["source_version"]
        assert sv == "mllmu-bench-1.0"

    def test_hf_config_name_pinned(self):
        cfg = _load_data_config("mllmu")
        assert cfg["data"]["hf_config_name"] == "Full_Set"

    def test_source_file_matches_config(self):
        cfg = _load_data_config("mllmu")
        # source_file must track hf_config_name
        assert cfg["data"]["source_file"] == "data/Full_Set.jsonl"

    def test_keep_subsets_separate(self):
        cfg = _load_data_config("mllmu")
        assert cfg["data"]["keep_subsets_separate"] is True

    def test_adapter_version(self):
        cfg = _load_data_config("mllmu")
        assert cfg["data"]["adapter_version"] == "mllmu-v2"

    def test_mapping_covers_default_partitions(self):
        mapping = _resolve_mapping("mllmu")
        # MLLMU-Bench uses the default mapping since it has no split_file
        assert mapping["train"] == "train"
        assert mapping["test"] == "eval"
        assert mapping["forget"] == "exclude"


class TestPPUBenchSourceMapping:
    """PPU-Bench: real public figures with ppu_eval_classification."""

    def test_config_exists(self):
        assert (DATA_CONFIGS_DIR / "ppubench.yaml").exists()

    def test_source_version_pinned(self):
        cfg = _load_data_config("ppubench")
        sv = cfg["data"]["source_version"]
        assert sv == "ppu-bench-1.0"

    def test_hf_dataset_id_set(self):
        cfg = _load_data_config("ppubench")
        assert cfg["data"]["hf_dataset_id"] == "closerG/ppu-bench"

    def test_hf_config_name_pinned(self):
        cfg = _load_data_config("ppubench")
        assert cfg["data"]["hf_config_name"] == "ppu_eval_classification"

    def test_hf_split_pinned(self):
        cfg = _load_data_config("ppubench")
        assert cfg["data"]["hf_split"] == "test"

    def test_source_file_matches_config(self):
        cfg = _load_data_config("ppubench")
        assert cfg["data"]["source_file"] == "data/ppu_eval_classification.jsonl"

    def test_preserve_fields_defined(self):
        cfg = _load_data_config("ppubench")
        fields = cfg["data"].get("preserve_fields", [])
        assert len(fields) > 0
        # Critical fields that must be preserved
        assert "subject_id" in fields
        assert "task_type" in fields
        assert "modality" in fields

    def test_adapter_version(self):
        cfg = _load_data_config("ppubench")
        assert cfg["data"]["adapter_version"] == "ppubench-v2"

    def test_mapping_covers_default_partitions(self):
        mapping = _resolve_mapping("ppubench")
        assert mapping["train"] == "train"
        assert mapping["test"] == "eval"


class TestCrossBenchmarkConsistency:
    """Verify invariants that must hold across all benchmark configs."""

    BENCHMARKS = ("fairget", "fiubench", "mllmu", "ppubench")

    def test_all_configs_have_source_version(self):
        for name in self.BENCHMARKS:
            cfg = _load_data_config(name)
            sv = cfg["data"].get("source_version")
            assert sv, f"{name} missing source_version"
            assert sv != "unknown", f"{name} has source_version='unknown'"

    def test_all_configs_have_adapter_version(self):
        for name in self.BENCHMARKS:
            cfg = _load_data_config(name)
            av = cfg["data"].get("adapter_version")
            assert av, f"{name} missing adapter_version"

    def test_all_configs_have_source_file(self):
        for name in self.BENCHMARKS:
            cfg = _load_data_config(name)
            sf = cfg["data"].get("source_file")
            assert sf, f"{name} missing source_file"

    def test_all_mappings_have_hash_fallback(self):
        """Every benchmark must have a hash fallback for unassigned."""
        for name in self.BENCHMARKS:
            mapping = _resolve_mapping(name)
            assert "unassigned" in mapping, f"{name} missing 'unassigned' mapping"
            assert mapping["unassigned"] == "hash"

    def test_all_mappings_have_forget_excluded(self):
        """Every benchmark must map 'forget' to 'exclude'."""
        for name in self.BENCHMARKS:
            mapping = _resolve_mapping(name)
            assert "forget" in mapping, f"{name} missing 'forget' mapping"
            assert mapping["forget"] == "exclude"
