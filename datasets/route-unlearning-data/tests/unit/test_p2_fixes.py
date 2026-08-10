"""Tests for P2 and late-P1 fixes: matched wrong-name, export finalization,
route-probe coverage report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from route_data.build.conflict_generation import (
    _accepted_visible_attributes,
    _visual_attribute_jaccard,
    select_matched_wrong_name,
    select_multiple_wrong_names,
)
from route_data.data.schemas import (
    AttributeObservation,
    CanonicalSample,
    ProfileFact,
    Provenance,
)

EYEGLASSES_KEY = "extended_attributes.celeba40.Eyeglasses"
SMILING_KEY = "extended_attributes.celeba40.Smiling"
BALD_KEY = "extended_attributes.celeba40.Bald"


def _sample(
    *,
    identity_id: str,
    identity_name: str,
    source_sample_id: str,
    image_uri: str,
    attrs: dict[str, bool],
    facts: list[ProfileFact] | None = None,
) -> CanonicalSample:
    """Helper to build a CanonicalSample with high-confidence visual attrs."""
    va: dict[str, AttributeObservation] = {}
    for name, label in attrs.items():
        key = f"extended_attributes.celeba40.{name}"
        va[key] = AttributeObservation(
            name=key,
            label=label,
            source="source_model",
            confidence_band="high",
        )
    return CanonicalSample(
        benchmark="fairget",
        source_sample_id=source_sample_id,
        identity_id=identity_id,
        identity_name=identity_name,
        provenance=Provenance(source_dataset="fairget"),
        image_uri=image_uri,
        modality="image_text",
        visual_attributes=va,
        profile_facts=facts or [
            ProfileFact(fact_id="f_nationality", relation="nationality", value="X"),
        ],
    )


# --------------------------------------------------------------------------- #
# P2-19: visual-attribute Jaccard similarity
# --------------------------------------------------------------------------- #


class TestVisualAttributeJaccard:
    """Unit tests for _visual_attribute_jaccard."""

    def test_identical_attributes_return_one(self):
        a = {"Eyeglasses": True, "Smiling": False}
        assert _visual_attribute_jaccard(a, a) == 1.0

    def test_completely_different_returns_zero(self):
        a = {"Eyeglasses": True}
        b = {"Eyeglasses": False}
        # Shared key but disagreeing → 0 agreeing / 1 union = 0.0
        assert _visual_attribute_jaccard(a, b) == 0.0

    def test_disjoint_keys(self):
        a = {"Eyeglasses": True}
        b = {"Smiling": False}
        # No shared keys → 0.0
        assert _visual_attribute_jaccard(a, b) == 0.0

    def test_empty_inputs(self):
        assert _visual_attribute_jaccard({}, {}) == 0.0
        assert _visual_attribute_jaccard({"A": True}, {}) == 0.0

    def test_partial_overlap(self):
        a = {"Eyeglasses": True, "Smiling": True, "Bald": False}
        b = {"Eyeglasses": True, "Smiling": False, "Bald": False}
        # Shared: all 3. Agreeing: Eyeglasses + Bald = 2.
        # Union: 3. Jaccard = 2/3.
        assert abs(_visual_attribute_jaccard(a, b) - 2 / 3) < 1e-9

    def test_superset(self):
        a = {"Eyeglasses": True}
        b = {"Eyeglasses": True, "Smiling": True}
        # Shared: Eyeglasses. Agreeing: 1. Union: 2. Jaccard = 0.5
        assert abs(_visual_attribute_jaccard(a, b) - 0.5) < 1e-9


# --------------------------------------------------------------------------- #
# P2-19: select_matched_wrong_name
# --------------------------------------------------------------------------- #


class TestSelectMatchedWrongName:
    """P2-19: select the most visually similar wrong identity."""

    def _build_by_identity(self) -> dict[str, list[CanonicalSample]]:
        """Three identities: id_a (glasses+smile), id_b (glasses+smile),
        id_c (no glasses, no smile).  id_a should match id_b (more similar)."""
        return {
            "id_a": [
                _sample(
                    identity_id="id_a", identity_name="Alice",
                    source_sample_id="a1", image_uri="a1.png",
                    attrs={"Eyeglasses": True, "Smiling": True},
                ),
                _sample(
                    identity_id="id_a", identity_name="Alice",
                    source_sample_id="a2", image_uri="a2.png",
                    attrs={"Eyeglasses": True, "Smiling": True},
                ),
            ],
            "id_b": [
                _sample(
                    identity_id="id_b", identity_name="Bob",
                    source_sample_id="b1", image_uri="b1.png",
                    attrs={"Eyeglasses": True, "Smiling": True},
                ),
                _sample(
                    identity_id="id_b", identity_name="Bob",
                    source_sample_id="b2", image_uri="b2.png",
                    attrs={"Eyeglasses": True, "Smiling": True},
                ),
            ],
            "id_c": [
                _sample(
                    identity_id="id_c", identity_name="Carol",
                    source_sample_id="c1", image_uri="c1.png",
                    attrs={"Eyeglasses": False, "Smiling": False},
                ),
                _sample(
                    identity_id="id_c", identity_name="Carol",
                    source_sample_id="c2", image_uri="c2.png",
                    attrs={"Eyeglasses": False, "Smiling": False},
                ),
            ],
        }

    def test_most_similar_chosen(self):
        by_id = self._build_by_identity()
        result = select_matched_wrong_name("id_a", by_id)
        # id_b is identical visually → should be chosen over id_c.
        assert result == "Bob"

    def test_returns_none_for_unknown_identity(self):
        by_id = self._build_by_identity()
        result = select_matched_wrong_name("nonexistent", by_id)
        assert result is None

    def test_returns_none_when_no_candidates(self):
        by_id = {
            "id_a": [
                _sample(
                    identity_id="id_a", identity_name="Alice",
                    source_sample_id="a1", image_uri="a1.png",
                    attrs={"Eyeglasses": True},
                ),
            ],
        }
        result = select_matched_wrong_name("id_a", by_id)
        assert result is None

    def test_deterministic_on_ties(self):
        """When two candidates have equal similarity, name order breaks ties."""
        by_id = {
            "id_target": [
                _sample(
                    identity_id="id_target", identity_name="Target",
                    source_sample_id="t1", image_uri="t1.png",
                    attrs={"Eyeglasses": True},
                ),
                _sample(
                    identity_id="id_target", identity_name="Target",
                    source_sample_id="t2", image_uri="t2.png",
                    attrs={"Eyeglasses": True},
                ),
            ],
            "id_z": [
                _sample(
                    identity_id="id_z", identity_name="Zara",
                    source_sample_id="z1", image_uri="z1.png",
                    attrs={"Eyeglasses": True},
                ),
                _sample(
                    identity_id="id_z", identity_name="Zara",
                    source_sample_id="z2", image_uri="z2.png",
                    attrs={"Eyeglasses": True},
                ),
            ],
            "id_a": [
                _sample(
                    identity_id="id_a", identity_name="Anna",
                    source_sample_id="a1", image_uri="a1.png",
                    attrs={"Eyeglasses": True},
                ),
                _sample(
                    identity_id="id_a", identity_name="Anna",
                    source_sample_id="a2", image_uri="a2.png",
                    attrs={"Eyeglasses": True},
                ),
            ],
        }
        result = select_matched_wrong_name("id_target", by_id)
        # Both have Jaccard=1.0; "Anna" < "Zara" alphabetically.
        assert result == "Anna"

    def test_skips_identities_with_single_sample(self):
        """Candidates must have >= 2 samples to be considered."""
        by_id = {
            "id_a": [
                _sample(
                    identity_id="id_a", identity_name="Alice",
                    source_sample_id="a1", image_uri="a1.png",
                    attrs={"Eyeglasses": True},
                ),
                _sample(
                    identity_id="id_a", identity_name="Alice",
                    source_sample_id="a2", image_uri="a2.png",
                    attrs={"Eyeglasses": True},
                ),
            ],
            "id_b": [
                _sample(
                    identity_id="id_b", identity_name="Bob",
                    source_sample_id="b1", image_uri="b1.png",
                    attrs={"Eyeglasses": True},
                ),
                # Only one sample — should be skipped.
            ],
        }
        result = select_matched_wrong_name("id_a", by_id)
        assert result is None


# --------------------------------------------------------------------------- #
# P2-19: select_multiple_wrong_names
# --------------------------------------------------------------------------- #


class TestSelectMultipleWrongNames:
    """P2-19: return up to N matched wrong names."""

    def test_returns_up_to_n(self):
        by_id: dict[str, list[CanonicalSample]] = {}
        for i, name in enumerate(["Alice", "Bob", "Carol", "Dave"]):
            iid = f"id_{i}"
            by_id[iid] = [
                _sample(
                    identity_id=iid, identity_name=name,
                    source_sample_id=f"{iid}_1", image_uri=f"{iid}_1.png",
                    attrs={"Eyeglasses": i % 2 == 0, "Smiling": True},
                ),
                _sample(
                    identity_id=iid, identity_name=name,
                    source_sample_id=f"{iid}_2", image_uri=f"{iid}_2.png",
                    attrs={"Eyeglasses": i % 2 == 0, "Smiling": True},
                ),
            ]
        result = select_multiple_wrong_names("id_0", by_id, candidates_per_sample=2)
        assert len(result) == 2
        # All names should be from other identities.
        assert "Alice" not in result

    def test_returns_empty_for_unknown(self):
        result = select_multiple_wrong_names("nonexistent", {})
        assert result == []

    def test_returns_empty_when_no_accepted_attrs(self):
        by_id = {
            "id_a": [
                _sample(
                    identity_id="id_a", identity_name="Alice",
                    source_sample_id="a1", image_uri="a1.png",
                    attrs={},  # no accepted attrs
                ),
                _sample(
                    identity_id="id_a", identity_name="Alice",
                    source_sample_id="a2", image_uri="a2.png",
                    attrs={},
                ),
            ],
        }
        result = select_multiple_wrong_names("id_a", by_id)
        assert result == []


# --------------------------------------------------------------------------- #
# P1-15: export manifest / checksum ordering
# --------------------------------------------------------------------------- #


class TestExportManifestChecksums:
    """P1-15: verify export_all produces correct checksums + manifest."""

    def test_checksums_exclude_self(self, tmp_path):
        """checksums.json must not contain a hash for itself."""
        from route_data.build.export import ExtensionExporter
        from route_data.data.schemas import CanonicalSample

        samples = [
            CanonicalSample(
                benchmark="test",
                source_sample_id="s1",
                identity_id="id1",
                identity_name="Test",
                provenance=Provenance(source_dataset="test"),
                image_uri="img.png",
                modality="image_text",
                visual_attributes={
                    "extended_attributes.celeba40.Eyeglasses": AttributeObservation(
                        name="extended_attributes.celeba40.Eyeglasses",
                        label=True,
                        source="source_model",
                        confidence_band="high",
                    ),
                },
                profile_facts=[
                    ProfileFact(fact_id="f1", relation="nationality", value="X"),
                ],
            ),
        ]
        exporter = ExtensionExporter(tmp_path, "test")
        record = exporter.export_all(samples)

        checksums_path = Path(record.paths["checksums"])
        checksums = json.loads(checksums_path.read_text())

        # checksums.json must not reference itself.
        for key in checksums:
            assert "checksums" not in key.lower() or "checksums.json" not in key

    def test_manifest_references_checksums(self, tmp_path):
        """Export manifest must include a 'checksums' path."""
        from route_data.build.export import ExtensionExporter

        samples = [
            CanonicalSample(
                benchmark="test",
                source_sample_id="s1",
                identity_id="id1",
                identity_name="Test",
                provenance=Provenance(source_dataset="test"),
                image_uri="img.png",
                modality="image_text",
                visual_attributes={
                    "extended_attributes.celeba40.Eyeglasses": AttributeObservation(
                        name="extended_attributes.celeba40.Eyeglasses",
                        label=True,
                        source="source_model",
                        confidence_band="high",
                    ),
                },
                profile_facts=[],
            ),
        ]
        exporter = ExtensionExporter(tmp_path, "test")
        record = exporter.export_all(samples)

        manifest_path = Path(record.paths["manifest"])
        manifest = json.loads(manifest_path.read_text())
        assert "checksums" in manifest.get("paths", {})

    def test_manifest_includes_provenance(self, tmp_path):
        """Export manifest includes provenance when provided."""
        from route_data.build.export import ExtensionExporter

        samples = [
            CanonicalSample(
                benchmark="test",
                source_sample_id="s1",
                identity_id="id1",
                identity_name="Test",
                provenance=Provenance(source_dataset="test"),
                image_uri="img.png",
                modality="image_text",
                visual_attributes={},
                profile_facts=[],
            ),
        ]
        exporter = ExtensionExporter(tmp_path, "test")
        provenance = {"model_id": "test-model", "source_version": "v1"}
        record = exporter.export_all(samples, provenance=provenance)

        manifest = json.loads(Path(record.paths["manifest"]).read_text())
        assert manifest["provenance"]["model_id"] == "test-model"
        assert manifest["provenance"]["source_version"] == "v1"

    def test_checksums_are_relative_paths(self, tmp_path):
        """All checksum keys must be relative (not absolute) paths."""
        from route_data.build.export import ExtensionExporter

        samples = [
            CanonicalSample(
                benchmark="test",
                source_sample_id="s1",
                identity_id="id1",
                identity_name="Test",
                provenance=Provenance(source_dataset="test"),
                image_uri="img.png",
                modality="image_text",
                visual_attributes={},
                profile_facts=[],
            ),
        ]
        exporter = ExtensionExporter(tmp_path, "test")
        record = exporter.export_all(samples)

        checksums = json.loads(Path(record.paths["checksums"]).read_text())
        for key in checksums:
            assert not Path(key).is_absolute(), f"absolute path in checksums: {key}"

    def test_checksums_values_are_sha256(self, tmp_path):
        """All checksum values must be 64-char hex strings."""
        from route_data.build.export import ExtensionExporter

        samples = [
            CanonicalSample(
                benchmark="test",
                source_sample_id="s1",
                identity_id="id1",
                identity_name="Test",
                provenance=Provenance(source_dataset="test"),
                image_uri="img.png",
                modality="image_text",
                visual_attributes={},
                profile_facts=[],
            ),
        ]
        exporter = ExtensionExporter(tmp_path, "test")
        record = exporter.export_all(samples)

        checksums = json.loads(Path(record.paths["checksums"]).read_text())
        for path_key, digest in checksums.items():
            assert len(digest) == 64, f"bad SHA-256 for {path_key}: {digest!r}"
            assert all(c in "0123456789abcdef" for c in digest)

    def test_manifest_hash_appended_to_checksums(self, tmp_path):
        """After export_all, checksums.json must include the manifest hash."""
        from route_data.build.export import ExtensionExporter

        samples = [
            CanonicalSample(
                benchmark="test",
                source_sample_id="s1",
                identity_id="id1",
                identity_name="Test",
                provenance=Provenance(source_dataset="test"),
                image_uri="img.png",
                modality="image_text",
                visual_attributes={},
                profile_facts=[],
            ),
        ]
        exporter = ExtensionExporter(tmp_path, "test")
        record = exporter.export_all(samples)

        checksums = json.loads(Path(record.paths["checksums"]).read_text())
        # Must contain the manifest hash.
        manifest_rel = Path(record.paths["manifest"]).name
        found = any(manifest_rel in key for key in checksums)
        assert found, f"manifest hash not in checksums; keys: {list(checksums)}"


# --------------------------------------------------------------------------- #
# P1-18: route-probe coverage report
# --------------------------------------------------------------------------- #


class TestProbeCoverageReport:
    """P1-18: _build_probe_coverage_report produces correct structure."""

    def test_basic_coverage_report(self):
        """Coverage report has all required top-level keys."""
        from route_data.cli import _build_probe_coverage_report

        s1 = _sample(
            identity_id="id1", identity_name="Alice",
            source_sample_id="s1", image_uri="s1.png",
            attrs={"Eyeglasses": True, "Smiling": True},
        )
        s2 = _sample(
            identity_id="id1", identity_name="Alice",
            source_sample_id="s2", image_uri="s2.png",
            attrs={"Eyeglasses": True, "Smiling": False},
        )
        by_identity = {"id1": [s1, s2]}
        probe_rows = [
            {"probe_family": "direct_visual", "target_attribute": "Eyeglasses",
             "answer_text": "yes"},
            {"probe_family": "direct_visual", "target_attribute": "Smiling",
             "answer_text": "yes"},
            {"probe_family": "name_only", "target_attribute": None,
             "answer_text": "Testland"},
        ]
        pairs = [
            {
                "pair_type": "cross_image_attribute_state",
                "attribute": "Smiling",
                "left_sample_id": "s1",
                "right_sample_id": "s2",
            }
        ]
        skipped = ["id2: no accepted visible attribute"]

        report = _build_probe_coverage_report(
            "fairget", [s1, s2], by_identity, probe_rows, pairs, skipped,
        )

        assert report["dataset"] == "fairget"
        assert report["identities_total"] == 1
        assert report["identities_with_visual_anchors"] == 1
        assert report["identities_with_profile_facts"] == 1
        assert report["identities_with_second_valid_image"] == 1
        assert "direct_visual" in report["probe_families"]
        assert report["probe_families"]["direct_visual"] == 2
        assert report["cross_image_attribute_state_pairs"] == 1
        assert "no accepted visible attribute" in report["skipped_identities_by_reason"]

    def test_per_attribute_coverage(self):
        """Per-attribute positive/negative/state_change counts."""
        from route_data.cli import _build_probe_coverage_report

        s1 = _sample(
            identity_id="id1", identity_name="Alice",
            source_sample_id="s1", image_uri="s1.png",
            attrs={"Eyeglasses": True},
        )
        by_identity = {"id1": [s1]}
        probe_rows = [
            {"probe_family": "direct_visual", "target_attribute": "Eyeglasses",
             "answer_text": "yes"},
            {"probe_family": "direct_visual", "target_attribute": "Eyeglasses",
             "answer_text": "no"},
        ]
        pairs = [
            {
                "pair_type": "cross_image_attribute_state",
                "attribute": "Eyeglasses",
            },
        ]

        report = _build_probe_coverage_report(
            "test", [s1], by_identity, probe_rows, pairs, [],
        )
        cov = report["per_attribute_coverage"]
        assert "Eyeglasses" in cov
        assert cov["Eyeglasses"]["positive"] == 1
        assert cov["Eyeglasses"]["negative"] == 1
        assert cov["Eyeglasses"]["state_change"] == 1

    def test_empty_inputs(self):
        """Report handles empty inputs gracefully."""
        from route_data.cli import _build_probe_coverage_report

        report = _build_probe_coverage_report("test", [], {}, [], [], [])
        assert report["identities_total"] == 0
        assert report["probe_families"] == {}
        assert report["per_attribute_coverage"] == {}

    def test_wrong_name_availability(self):
        """wrong_name_availability counts identities with eligible candidates."""
        from route_data.cli import _build_probe_coverage_report

        s1 = _sample(
            identity_id="id1", identity_name="Alice",
            source_sample_id="s1", image_uri="s1.png",
            attrs={"Eyeglasses": True},
        )
        s2 = _sample(
            identity_id="id2", identity_name="Bob",
            source_sample_id="s2", image_uri="s2.png",
            attrs={"Eyeglasses": False},
        )
        s3 = _sample(
            identity_id="id2", identity_name="Bob",
            source_sample_id="s3", image_uri="s3.png",
            attrs={"Eyeglasses": False},
        )
        by_identity = {"id1": [s1], "id2": [s2, s3]}

        report = _build_probe_coverage_report(
            "test", [s1, s2, s3], by_identity, [], [], [],
        )
        # id1 has candidate id2 (2+ samples, accepted attrs) → 1
        # id2 has no candidate (id1 has only 1 sample) → 0
        assert report["wrong_name_availability"] == 1
