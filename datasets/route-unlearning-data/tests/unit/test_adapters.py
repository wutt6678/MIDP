"""Benchmark adapters: fail-loud mapping from raw rows to canonical samples (plan 9)."""

from __future__ import annotations

import json

import pytest

from route_data.config import DataConfig
from route_data.data.adapters.base import (
    AdapterError,
    available_adapters,
    create_adapter,
)
from route_data.data.adapters.fairget import FairgetAdapter


def _row(**overrides) -> dict:
    row = {
        "sample_id": "gf_0001",
        "identity_id": "golden_alpha",
        "identity_name": "Ava Alpha",
        "image_id": "gf_0001",
        "image_path": "images/gf_0001.png",
        "split": "forget",
        "modality": "image_text",
        "fairface": {"glasses": True},
        "profile": {"nationality": "Fjordmark", "occupation": "Cartographer"},
    }
    row.update(overrides)
    return row


class TestFairgetToSample:
    def test_fields_are_mapped(self):
        sample = FairgetAdapter(DataConfig(name="fairget")).to_sample(_row())
        assert sample.source_sample_id == "gf_0001"
        assert sample.identity_id == "golden_alpha"
        assert sample.identity_name == "Ava Alpha"
        assert sample.image_uri == "images/gf_0001.png"
        assert sample.provenance.source_dataset == "fairget"
        assert "source_attributes.fairface.glasses" in sample.visual_attributes
        fact_ids = {f.fact_id for f in sample.profile_facts}
        assert fact_ids == {"fairget_nationality", "fairget_occupation"}

    def test_missing_required_field_raises(self):
        row = _row()
        del row["sample_id"]
        with pytest.raises(AdapterError, match="source_sample_id"):
            FairgetAdapter(DataConfig(name="fairget")).to_sample(row)

    def test_optional_identity_name_defaults_to_none(self):
        row = _row()
        del row["identity_name"]
        sample = FairgetAdapter(DataConfig(name="fairget")).to_sample(row)
        assert sample.identity_name is None

    def test_field_map_extras_override(self):
        config = DataConfig(
            name="fairget", extras={"field_map": {"source_sample_id": "uid"}}
        )
        row = _row(uid="custom_1")
        del row["sample_id"]
        sample = FairgetAdapter(config).to_sample(row)
        assert sample.source_sample_id == "custom_1"


class TestFairgetLoading:
    def _adapter(self, root) -> FairgetAdapter:
        return FairgetAdapter(DataConfig(name="fairget", root=str(root)))

    def test_iter_rows_reads_local_jsonl(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        with (data_dir / "rows.jsonl").open("w") as fh:
            fh.write(json.dumps(_row()) + "\n")
        samples = list(self._adapter(tmp_path).load())
        assert len(samples) == 1
        assert samples[0].source_sample_id == "gf_0001"

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(AdapterError, match="does not exist"):
            list(self._adapter(tmp_path / "nope").iter_rows())


class TestRegistry:
    def test_builtin_adapters_registered(self):
        names = set(available_adapters())
        assert {"fairget", "fiubench", "mllmu", "ppubench"} <= names

    def test_unknown_dataset_raises(self):
        with pytest.raises(AdapterError, match="No adapter registered"):
            create_adapter(DataConfig(name="nonexistent_benchmark"))
