"""P1-17: benchmark-specific source-split mapping tests.

Each benchmark has its own data config with specific partition labels.
These tests verify the DEFAULT_SOURCE_MAPPING and per-benchmark overrides
produce correct train/eval/exclude/hash assignments for the actual
released split vocabularies of FAIRGET, FIUBench, MLLMU-Bench, and
PPU-Bench.
"""

from __future__ import annotations

from pathlib import Path

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

    def test_immutable_revision_fields_present(self):
        # R10: immutable_revision block must exist with the required fields.
        cfg = _load_data_config("fairget")
        ir = cfg["data"].get("immutable_revision", {})
        assert "git_commit_sha" in ir
        assert "dataset_json_sha256" in ir
        assert "official_split_sha256" in ir

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

    def test_released_bucket_resolution_exact(self):
        # R9: FAIRGET's released split file maps identity -> bucket and the
        # adapter reuses bucket names verbatim (fairget._split_lookup); the
        # effective resolution of every bucket must therefore be pinned here.
        mapping = _resolve_mapping("fairget")
        assert mapping["forget"] == "exclude"  # official unbalanced forget
        assert mapping["train"] == "train"     # released training partition
        # Identities absent from the split file fall back to "unassigned".
        assert mapping["unassigned"] == "hash"

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

    def test_immutable_revision_fields_present(self):
        # R10: immutable_revision block must exist with the required fields.
        cfg = _load_data_config("fiubench")
        ir = cfg["data"].get("immutable_revision", {})
        assert "git_commit_sha" in ir
        assert "profile_file_sha256" in ir
        assert "split_file_sha256" in ir

    def test_split_file_defined(self):
        cfg = _load_data_config("fiubench")
        assert cfg["data"]["split_file"] == "splits/official.json"

    def test_reuse_official_splits(self):
        cfg = _load_data_config("fiubench")
        assert cfg["data"]["reuse_official_splits"] is True

    def test_mapping_includes_forget(self):
        mapping = _resolve_mapping("fiubench")
        assert mapping["forget"] == "exclude"

    def test_released_split_vocabulary_exact(self):
        # R9: FIUBench's official grouping is exactly forget / retain /
        # evaluation (splits/official.json).  The config must declare the
        # full vocabulary and nothing else.
        cfg = _load_data_config("fiubench")
        extras_mapping = cfg["data"]["extras"]["source_mapping"]
        assert extras_mapping == {
            "forget": "exclude",
            "retain": "train",
            "evaluation": "eval",
        }

    def test_released_labels_resolve_exactly(self):
        # R9: each of the three actual FIUBench labels must resolve to the
        # documented target — forget never enters ordinary QA, retain trains,
        # evaluation evaluates.
        mapping = _resolve_mapping("fiubench")
        assert mapping["forget"] == "exclude"
        assert mapping["retain"] == "train"
        assert mapping["evaluation"] == "eval"

    def test_no_released_label_leaks_to_hash(self):
        # Every official FIUBench label is explicitly mapped; none may fall
        # through to the unassigned/hash fallback (R3/R9).
        mapping = _resolve_mapping("fiubench")
        for label in ("forget", "retain", "evaluation"):
            assert mapping[label] != "hash", f"fiubench {label} leaked to hash"

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

    def test_immutable_revision_fields_present(self):
        # R10: immutable_revision block must exist with the required HF fields.
        cfg = _load_data_config("mllmu")
        ir = cfg["data"].get("immutable_revision", {})
        assert "hf_dataset_id" in ir
        assert "hf_config" in ir
        assert "hf_split" in ir
        assert "hf_revision_sha" in ir

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

    def test_adapter_split_table_exact(self):
        # R9: the adapter's per-configuration split semantics are the real
        # released vocabulary; pin the complete table.
        from route_data.data.adapters.mllmu import _KNOWN_CONFIGS, _SPLIT_BY_CONFIG

        assert _SPLIT_BY_CONFIG == {
            "forget_5": "forget",
            "forget_10": "forget",
            "forget_15": "forget",
            "retain_85": "retain_train",
            "retain_90": "retain_train",
            "retain_95": "retain_train",
            "Retain_Set": "retain_eval",
            "Test_Set": "test",
            "ft_Data": "finetune",
            "Full_Set": "unassigned",
        }
        assert set(_SPLIT_BY_CONFIG) == set(_KNOWN_CONFIGS)

    def test_emitted_labels_resolve_exactly(self):
        # R9: every split label the MLLMU adapter can emit must resolve to
        # the documented target through the effective mapping (cmd_build_qa
        # falls back to "hash" for labels absent from the mapping, which is
        # the documented resolution for "finetune").
        from route_data.data.adapters.mllmu import _SPLIT_BY_CONFIG

        mapping = _resolve_mapping("mllmu")
        expected = {
            "forget": "exclude",        # forget_5/10/15
            "retain_train": "train",    # retain_85/90/95
            "retain_eval": "eval",      # Retain_Set
            "test": "eval",             # Test_Set
            "unassigned": "hash",       # Full_Set
            "finetune": "hash",         # ft_Data: no mapping entry -> hash
        }
        for label in set(_SPLIT_BY_CONFIG.values()):
            resolved = mapping.get(label, "hash")
            assert resolved == expected[label], (
                f"mllmu split label {label!r} resolved to {resolved!r}, "
                f"expected {expected[label]!r}"
            )


class TestPPUBenchSourceMapping:
    """PPU-Bench: real public figures with ppu_eval_classification."""

    def test_config_exists(self):
        assert (DATA_CONFIGS_DIR / "ppubench.yaml").exists()

    def test_source_version_pinned(self):
        cfg = _load_data_config("ppubench")
        sv = cfg["data"]["source_version"]
        assert sv == "ppu-bench-1.0"

    def test_immutable_revision_fields_present(self):
        # R10: immutable_revision block must exist with the required HF fields.
        cfg = _load_data_config("ppubench")
        ir = cfg["data"].get("immutable_revision", {})
        assert "hf_dataset_id" in ir
        assert "hf_config" in ir
        assert "hf_split" in ir
        assert "hf_revision_sha" in ir

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

    def test_released_split_vocabulary_exact(self):
        # R9: PPU-Bench releases a single evaluation split; the adapter tags
        # every row with the pinned hf_split value (ppubench._split_name),
        # so the complete released vocabulary is exactly {"test"} and every
        # row must resolve to eval — never train, exclude, or hash.
        cfg = _load_data_config("ppubench")
        released_labels = {cfg["data"]["hf_split"]}
        assert released_labels == {"test"}
        mapping = _resolve_mapping("ppubench")
        for label in released_labels:
            assert mapping[label] == "eval"


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
