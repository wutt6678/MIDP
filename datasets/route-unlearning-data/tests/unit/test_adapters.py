"""Benchmark adapter contract tests (repair plan E3) + round-trip (E4).

Each adapter is exercised against its released source shape:

- fairget: the golden fixture materializes the real nested layout
  (``data/dataset.json`` + ``train_images/`` + ``test_images/`` + split file);
- fiubench / mllmu / ppubench: small synthetic rows matching the released
  schemas are fed through ``to_samples`` directly.

Asserted per plan E3: raw count, canonical count, IDs, identity, task,
modality, image view, question, answers, options, facts, metadata, and
provenance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from route_data.config import DataConfig
from route_data.data.adapters.base import (
    AdapterError,
    available_adapters,
    create_adapter,
)
from route_data.data.adapters.fairget import FairgetAdapter
from route_data.data.adapters.fiubench import FiubenchAdapter
from route_data.data.adapters.mllmu import MllmuAdapter
from route_data.data.adapters.ppubench import PpubenchAdapter
from route_data.data.io import read_jsonl, read_parquet_rows, write_jsonl, write_parquet
from route_data.data.schemas import CanonicalSample

FAIRGET_PIN = "fairget-golden-v1"


def _fairget_adapter(root: Path, **extra_extras) -> FairgetAdapter:
    extras = {"split_file": "splits/official.json"}
    extras.update(extra_extras)
    return FairgetAdapter(
        DataConfig(
            name="fairget", root=str(root), source_version=FAIRGET_PIN, extras=extras
        )
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class TestRegistry:
    def test_builtin_adapters_registered(self):
        names = set(available_adapters())
        assert {"fairget", "fiubench", "mllmu", "ppubench"} <= names

    def test_unknown_dataset_raises(self):
        with pytest.raises(AdapterError, match="No adapter registered"):
            create_adapter(DataConfig(name="nonexistent_benchmark"))


# --------------------------------------------------------------------------- #
# FAIRGET (plans B6-B12) — nested released layout via the golden fixture
# --------------------------------------------------------------------------- #


class TestFairgetContract:
    def test_raw_and_canonical_counts(self, golden_root):
        samples = list(_fairget_adapter(golden_root).load())
        # 3 identities x 6 canonical samples each (E5 ground truth).
        assert len(samples) == 18
        identities = {s.identity_id for s in samples}
        assert identities == {"gld_001", "gld_002", "gld_003"}

    def test_source_files_require_full_layout(self, golden_root, tmp_path):
        adapter = _fairget_adapter(golden_root)
        files = adapter.source_files()
        assert files[0] == golden_root / "data" / "dataset.json"
        assert files[1] == golden_root / "splits" / "official.json"
        # Empty directory fails loudly before any model loads (plan C4).
        with pytest.raises(AdapterError, match="incompatible source layout"):
            _fairget_adapter(tmp_path / "empty").source_files()

    def test_deterministic_ids_and_view_expansion(self, golden_root):
        samples = {
            s.source_sample_id: s
            for s in _fairget_adapter(golden_root).load()
        }
        expected = {
            # train image QA expands over BOTH training views (all_views).
            "fairget:gld_001:train:image:glasses:0:frame_0001",
            "fairget:gld_001:train:image:glasses:0:frame_0002",
            # train text QA has no view.
            "fairget:gld_001:train:text:nationality:0:text",
            # eval IDs keep media type, task, attribute, index, view.
            "fairget:gld_001:eval:image:identity_fact:nationality:0:frame_0003",
            "fairget:gld_001:eval:image:visual_attribute:smiling:0:frame_0003",
            "fairget:gld_001:eval:text:knowledge_qa:occupation:0:text",
        }
        assert expected <= set(samples)

        view1 = samples["fairget:gld_001:train:image:glasses:0:frame_0001"]
        assert view1.image_uri == str(
            golden_root / "train_images" / "gld_001" / "frame_0001.png"
        )
        assert view1.image_id == "frame_0001"
        assert view1.question == "Is this person wearing glasses?"
        assert view1.answer_label is True
        assert view1.source_metadata["image_expansion_policy"] == "all_views"
        assert view1.source_metadata["view_assignment_seed"] == 17
        assert view1.source_metadata["q_words"] == ["wearing", "glasses"]

    def test_modality_and_demographics_image_only(self, golden_root):
        samples = {
            s.source_sample_id: s
            for s in _fairget_adapter(golden_root).load()
        }
        text = samples["fairget:gld_001:train:text:nationality:0:text"]
        assert text.modality == "text_only"
        assert text.image_uri is None
        assert text.visual_attributes == {}  # demographics are image-level only
        assert text.task_type == "knowledge_qa"

        image = samples["fairget:gld_001:train:image:glasses:0:frame_0001"]
        assert image.modality == "image_text"
        assert set(image.visual_attributes) == {
            "source_attributes.fairface.smiling",
            "source_attributes.fairface.wearing_hat",
        }
        assert image.visual_attributes["source_attributes.fairface.smiling"].label is True
        assert image.identity_name == "Ava Alpha"

    def test_profile_facts_skip_empty_values(self, golden_root):
        samples = list(_fairget_adapter(golden_root).load())
        fact_ids = {f.fact_id for f in samples[0].profile_facts}
        # health_condition is explicitly empty in the fixture -> skipped.
        assert fact_ids == {"fairget_nationality", "fairget_occupation"}

    def test_splits_come_from_official_file(self, golden_root):
        splits = {
            s.identity_id: s.split for s in _fairget_adapter(golden_root).load()
        }
        assert splits == {
            "gld_001": "forget",
            "gld_002": "retain_train",
            "gld_003": "retain_eval",
        }

    def test_provenance_and_metadata_pinned(self, golden_root):
        sample = next(iter(_fairget_adapter(golden_root).load()))
        prov = sample.provenance
        assert prov.source_dataset == "fairget"
        assert prov.source_version == FAIRGET_PIN
        assert prov.adapter == "fairget"
        assert prov.adapter_version == "fairget-v2"
        assert sample.source_metadata["source_dataset"] == "fairget"
        assert sample.source_metadata["source_revision"] == FAIRGET_PIN
        assert sample.source_metadata["source_row_index"] in (0, 1, 2)

    def test_unpinned_revision_raises(self, golden_root):
        adapter = FairgetAdapter(
            DataConfig(
                name="fairget",
                root=str(golden_root),
                source_version="unknown",
                extras={"split_file": "splits/official.json"},
            )
        )
        with pytest.raises(AdapterError, match="not pinned"):
            adapter.source_revision()

    def test_hash_assigned_policy_emits_one_view(self, golden_root):
        adapter = _fairget_adapter(
            golden_root, image_expansion_policy="hash_assigned"
        )
        samples = [
            s
            for s in adapter.load()
            if s.source_sample_id.startswith("fairget:gld_001:train:image:")
        ]
        assert len(samples) == 1  # one stable hash-assigned view, not two
        assert samples[0].source_metadata["image_expansion_policy"] == "hash_assigned"


# --------------------------------------------------------------------------- #
# FIUBench (plans B13-B18) — profile-level qa_list flattening
# --------------------------------------------------------------------------- #


def _fiubench_adapter(root: Path, **extra_extras) -> FiubenchAdapter:
    extras = {"source_file": "profiles.jsonl"}
    extras.update(extra_extras)
    return FiubenchAdapter(
        DataConfig(
            name="fiubench",
            root=str(root),
            source_version="fiubench-pin-v1",
            extras=extras,
        )
    )


def _fiubench_row(**overrides) -> dict:
    row = {
        "image_path": "images/a.png",
        "name": "Ava Alpha",
        "gender": "female",
        "caption": "Ava is a synthetic test persona.",
        "raw_data": {"city": "Alpha City"},
        "qa_list": [
            {
                "question": "Where does Ava live?",
                "answer": "Alpha City",
                "paraphrased_question": "In which city does Ava live?",
                "paraphrased_answer": "She lives in Alpha City",
                "perturbed_answer": "Beta Town",
                "keywords": ["city", "home"],
            },
            {"question": "What is Ava's caption?", "answer": "test persona"},
        ],
    }
    row.update(overrides)
    return row


class TestFiubenchContract:
    def _context(self, adapter, row_index: int):
        return adapter.base_context(
            source_file="profiles.jsonl", source_row_index=row_index
        )

    def test_original_variants_only_by_default(self, tmp_path):
        adapter = _fiubench_adapter(tmp_path)
        samples = list(
            adapter.to_samples(_fiubench_row(), source_context=self._context(adapter, 0))
        )
        assert len(samples) == 2  # one record per original QA item
        identity_id = samples[0].identity_id
        assert [s.source_sample_id for s in samples] == [
            f"fiubench:{identity_id}:qa:0:original",
            f"fiubench:{identity_id}:qa:1:original",
        ]
        assert samples[0].question == "Where does Ava live?"
        assert samples[0].answer_text == "Alpha City"
        assert samples[0].task_type == "private_profile_vqa"
        assert samples[0].modality == "image_text"
        assert samples[0].source_metadata["variant_type"] == "original"
        assert samples[0].source_metadata["keywords"] == ["city", "home"]
        # Gender is profile metadata only — never a visual label (B14).
        assert samples[0].source_metadata["gender"] == "female"
        assert samples[0].visual_attributes == {}

    def test_paraphrase_and_perturbed_gated_by_extras(self, tmp_path):
        adapter = _fiubench_adapter(
            tmp_path, include_paraphrases=True, include_perturbed=True
        )
        samples = list(
            adapter.to_samples(_fiubench_row(), source_context=self._context(adapter, 0))
        )
        # qa0 carries all variants, qa1 only the original.
        assert len(samples) == 4
        variants = {s.source_subset: s for s in samples}
        assert variants["paraphrase"].question == "In which city does Ava live?"
        assert variants["paraphrase"].answer_text == "She lives in Alpha City"
        assert variants["perturbed"].answer_text == "Beta Town"
        base_id = variants["original"].source_sample_id.replace(
            ":qa:1:original", ":qa:0:original"
        )
        assert variants["paraphrase"].source_metadata["base_qa_id"] == (
            f"fiubench:{samples[0].identity_id}:qa:0:original"
        )
        assert base_id.startswith("fiubench:")

    def test_identity_id_is_stable_per_pinned_row(self, tmp_path):
        adapter = _fiubench_adapter(tmp_path)
        row = _fiubench_row()
        first = next(
            adapter.to_samples(row, source_context=self._context(adapter, 0))
        ).identity_id
        again = next(
            adapter.to_samples(row, source_context=self._context(adapter, 0))
        ).identity_id
        shifted = next(
            adapter.to_samples(row, source_context=self._context(adapter, 1))
        ).identity_id
        assert first == again
        assert first != shifted
        expected = hashlib.sha256(
            b"fiubench-pin-v1|0|images/a.png|Ava Alpha"
        ).hexdigest()[:16]
        assert first == expected

    def test_caption_and_raw_profile_preserved_whole(self, tmp_path):
        adapter = _fiubench_adapter(tmp_path)
        sample = next(
            adapter.to_samples(
                _fiubench_row(), source_context=self._context(adapter, 0)
            )
        )
        facts = {f.fact_id: f.value for f in sample.profile_facts}
        assert facts == {
            "fiubench_caption": "Ava is a synthetic test persona.",
            "fiubench_raw_profile": '{"city": "Alpha City"}',
        }

    def test_split_membership_from_split_file(self, tmp_path):
        import json

        (tmp_path / "splits.json").write_text(json.dumps({"forget": ["Ava Alpha"]}))
        adapter = _fiubench_adapter(tmp_path, split_file="splits.json")
        ava = next(
            adapter.to_samples(
                _fiubench_row(), source_context=self._context(adapter, 0)
            )
        )
        other = next(
            adapter.to_samples(
                _fiubench_row(name="Ben Beta"),
                source_context=self._context(adapter, 1),
            )
        )
        assert ava.split == "forget"
        assert other.split == "unassigned"

    def test_missing_required_field_raises(self, tmp_path):
        adapter = _fiubench_adapter(tmp_path)
        row = _fiubench_row()
        del row["name"]
        with pytest.raises(AdapterError, match="identity_name"):
            list(adapter.to_samples(row, source_context=self._context(adapter, 0)))


# --------------------------------------------------------------------------- #
# MLLMU-Bench (plans B19-B26) — config matrix + task block flattening
# --------------------------------------------------------------------------- #


def _mllmu_adapter(root: Path, config_name: str = "forget_5") -> MllmuAdapter:
    return MllmuAdapter(
        DataConfig(
            name="mllmu",
            root=str(root),
            source_version="mllmu-pin-v1",
            extras={"hf_config_name": config_name, "source_file": "rows.jsonl"},
        )
    )


def _mllmu_row(root: Path, **overrides) -> dict:
    (root / "img1.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    row = {
        "ID": "P001",
        "name": "Person One",
        "biography": "Person One was born in Testville.",
        "image": "img1.png",
        "Classification_Task": {
            "Image_Textual_Questions": [
                {
                    "Question": "Who is this?",
                    "Options": {
                        "A": "Person One",
                        "B": "Person Two",
                        "C": "Person Three",
                        "D": "Person Four",
                    },
                    "Correct_Answer": "B",
                }
            ],
            "Pure_Text_Questions": [
                {
                    "Question": "Where was Person One born?",
                    "Options": ["Testville", "Otherton", "Nowhere", "Somewhere"],
                    "Correct_Answer": "A",
                }
            ],
        },
        "Generation_Task": [
            {"Question": "Describe the person.", "Ground_Truth": "GT text", "Type": "biography"}
        ],
        "Mask_Task": {
            "fill_blank": [
                {
                    "Question": "Person ___ was born in Testville.",
                    "Ground_Truth": "One",
                    "Type": "cloze",
                }
            ]
        },
    }
    row.update(overrides)
    return row


class TestMllmuContract:
    def test_config_matrix_enforced(self, tmp_path):
        no_config = MllmuAdapter(
            DataConfig(
                name="mllmu",
                root=str(tmp_path),
                source_version="mllmu-pin-v1",
                extras={"source_file": "rows.jsonl"},
            )
        )
        with pytest.raises(AdapterError, match="hf_config_name"):
            no_config.config_name()
        bad = _mllmu_adapter(tmp_path, config_name="Not_A_Set")
        with pytest.raises(AdapterError, match="validated configuration matrix"):
            bad.config_name()

    def test_full_row_expansion(self, tmp_path):
        adapter = _mllmu_adapter(tmp_path)
        samples = list(
            adapter.to_samples(
                _mllmu_row(tmp_path),
                source_context=adapter.base_context(source_row_index=0),
            )
        )
        # 2 classification items (1 view each) + 1 generation + 1 mask.
        assert len(samples) == 4
        by_id = {s.source_sample_id: s for s in samples}
        expected_ids = {
            "mllmu:forget_5:forget_5:P001:classification_qa:image_text:0:0",
            "mllmu:forget_5:forget_5:P001:classification_qa:text_only:0:0",
            "mllmu:forget_5:forget_5:P001:generation_qa:image_text:2:0",
            "mllmu:forget_5:forget_5:P001:mask_qa:image_text:3:0",
        }
        assert expected_ids <= set(by_id)

    def test_classification_options_label_and_text(self, tmp_path):
        adapter = _mllmu_adapter(tmp_path)
        samples = list(
            adapter.to_samples(
                _mllmu_row(tmp_path),
                source_context=adapter.base_context(source_row_index=0),
            )
        )
        image_item = samples[0]
        assert image_item.modality == "image_text"
        assert image_item.image_uri == str(tmp_path / "img1.png")
        assert image_item.options == [
            "Person One",
            "Person Two",
            "Person Three",
            "Person Four",
        ]
        # Original label AND resolved text are both preserved (B21).
        assert image_item.answer_label == "B"
        assert image_item.answer_text == "Person Two"
        assert image_item.source_metadata["original_answer_label"] == "B"
        assert image_item.source_metadata["config"] == "forget_5"
        assert image_item.source_subset == "forget_5"

        text_item = next(s for s in samples if s.modality == "text_only")
        assert text_item.image_uri is None
        assert text_item.options == ["Testville", "Otherton", "Nowhere", "Somewhere"]
        assert text_item.answer_text == "Testville"

    def test_generation_and_mask_preserve_type(self, tmp_path):
        adapter = _mllmu_adapter(tmp_path)
        samples = list(
            adapter.to_samples(
                _mllmu_row(tmp_path),
                source_context=adapter.base_context(source_row_index=0),
            )
        )
        generation = next(s for s in samples if s.task_type == "generation_qa")
        assert generation.question == "Describe the person."
        assert generation.answer_text == "GT text"
        assert generation.source_metadata["task_type_source"] == "biography"
        mask = next(s for s in samples if s.task_type == "mask_qa")
        assert mask.answer_text == "One"
        assert mask.source_metadata["task_type_source"] == "cloze"

    def test_biography_fact_and_split_by_config(self, tmp_path):
        adapter = _mllmu_adapter(tmp_path)
        sample = next(
            adapter.to_samples(
                _mllmu_row(tmp_path),
                source_context=adapter.base_context(source_row_index=0),
            )
        )
        facts = {f.fact_id: f.value for f in sample.profile_facts}
        assert facts == {"mllmu_biography": "Person One was born in Testville."}
        assert sample.split == "forget"  # forget_5 -> forget
        assert sample.identity_id == "forget_5:P001"
        assert sample.identity_name == "Person One"
        retain = _mllmu_adapter(tmp_path, config_name="Retain_Set")
        retain_sample = next(
            retain.to_samples(
                _mllmu_row(tmp_path),
                source_context=retain.base_context(source_row_index=0),
            )
        )
        assert retain_sample.split == "retain_eval"

    def test_missing_id_raises(self, tmp_path):
        adapter = _mllmu_adapter(tmp_path)
        row = _mllmu_row(tmp_path)
        del row["ID"]
        with pytest.raises(AdapterError, match="'ID'"):
            list(
                adapter.to_samples(
                    row, source_context=adapter.base_context(source_row_index=0)
                )
            )


# --------------------------------------------------------------------------- #
# PPU-Bench (plans B27-B33) — config/split pinning + multi-image views
# --------------------------------------------------------------------------- #


def _ppubench_adapter(root: Path, **extra_extras) -> PpubenchAdapter:
    extras = {"hf_config_name": "Public_Figures", "hf_split": "test"}
    extras.update(extra_extras)
    return PpubenchAdapter(
        DataConfig(
            name="ppubench",
            root=str(root),
            source_version="ppubench-pin-v1",
            extras=extras,
        )
    )


def _ppubench_row(**overrides) -> dict:
    row = {
        "sample_id": "s1",
        "subject_id": "p1",
        "subject": "Person One",
        "task_type": "attribute_qa",
        "modality": "image-text",
        "question": "What is the person wearing?",
        "answer_text": "hat",
        "answer_label": "A",
        "option_a": "hat",
        "option_b": "glasses",
        "option_c": None,
        "option_d": "scarf",
        "image": "img1.png",
        "image_002": "img2.png",
        "image_003": None,
    }
    row.update(overrides)
    return row


class TestPpubenchContract:
    def test_config_and_split_required(self, tmp_path):
        with pytest.raises(AdapterError, match="hf_config_name"):
            PpubenchAdapter(
                DataConfig(
                    name="ppubench",
                    root=str(tmp_path),
                    source_version="ppubench-pin-v1",
                    extras={"hf_split": "test"},
                )
            )
        with pytest.raises(AdapterError, match="hf_split"):
            PpubenchAdapter(
                DataConfig(
                    name="ppubench",
                    root=str(tmp_path),
                    source_version="ppubench-pin-v1",
                    extras={"hf_config_name": "Public_Figures"},
                )
            )

    def test_one_record_per_non_null_view(self, tmp_path):
        adapter = _ppubench_adapter(tmp_path)
        samples = list(
            adapter.to_samples(
                _ppubench_row(),
                source_context=adapter.base_context(source_row_index=0),
            )
        )
        # image_003 is null -> no record; text_only never emitted alongside views.
        assert [s.source_sample_id for s in samples] == [
            "ppubench:Public_Figures:s1:image",
            "ppubench:Public_Figures:s1:image_002",
        ]
        first = samples[0]
        assert first.identity_id == "p1"
        assert first.identity_name == "Person One"
        assert first.modality == "image_text"
        assert first.task_type == "attribute_qa"
        assert first.options == ["hat", "glasses", "scarf"]  # null option skipped
        assert first.answer_label == "A"
        assert first.answer_text == "hat"
        assert first.split == "test"
        assert first.source_metadata["config"] == "Public_Figures"
        assert first.source_metadata["image_field"] == "image"
        assert first.source_metadata["original_sample_id"] == "s1"
        assert first.source_metadata["original_modality"] == "image-text"
        assert samples[1].source_metadata["image_field"] == "image_002"

    def test_text_only_only_after_all_image_columns_checked(self, tmp_path):
        adapter = _ppubench_adapter(tmp_path)
        row = _ppubench_row(image=None, image_002=None)
        samples = list(
            adapter.to_samples(
                row, source_context=adapter.base_context(source_row_index=0)
            )
        )
        assert len(samples) == 1
        assert samples[0].source_sample_id == "ppubench:Public_Figures:s1:text_only"
        assert samples[0].modality == "text_only"
        assert samples[0].image_uri is None


# --------------------------------------------------------------------------- #
# Round-trip (plan E4): canonical -> JSONL/Parquet -> reload
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    def test_jsonl_round_trip_preserves_records(self, golden_root, tmp_path):
        samples = list(_fairget_adapter(golden_root).load())
        out_path = tmp_path / "canonical.jsonl"
        write_jsonl((s.to_dict() for s in samples), out_path)
        reloaded = [CanonicalSample.from_dict(row) for row in read_jsonl(out_path)]
        assert len(reloaded) == len(samples)
        assert [s.to_dict() for s in reloaded] == [s.to_dict() for s in samples]

    def test_parquet_round_trip_preserves_core_fields(self, golden_root, tmp_path):
        samples = list(_fairget_adapter(golden_root).load())
        out_path = tmp_path / "canonical.parquet"
        write_parquet([s.to_dict() for s in samples], out_path)
        rows = read_parquet_rows(out_path)
        assert len(rows) == len(samples)
        by_id = {row["source_sample_id"]: row for row in rows}
        for sample in samples:
            row = by_id[sample.source_sample_id]
            assert row["identity_id"] == sample.identity_id
            assert row["modality"] == sample.modality
            assert row["task_type"] == sample.task_type
            assert row["question"] == sample.question
            assert row["answer_text"] == sample.answer_text
            assert row["split"] == sample.split
