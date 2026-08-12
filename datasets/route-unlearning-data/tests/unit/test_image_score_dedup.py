"""P2-img: image-level score deduplication tests.

Covers:
- ImageScoreTable: building, lookup, serialisation, metrics
- ImageScoreCacheKey: deterministic cache key construction
- annotate_sample_via_image_table: lookup + fallback semantics
- build_sample_to_image_sha: mapping construction
"""

from __future__ import annotations

import pytest

from route_data.build.annotate import (
    AnnotationPolicy,
    BenchmarkAnnotator,
    ImageScoreCacheKey,
    ImageScoreTable,
    annotate_sample_via_image_table,
    build_sample_to_image_sha,
    celeba40_key,
)
from route_data.data.schemas import AttributeObservation, CanonicalSample, Provenance

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _sample(
    sample_id: str,
    image_sha: str = "",
    image_uri: str = "",
    identity_id: str = "id1",
) -> CanonicalSample:
    return CanonicalSample(
        benchmark="fiubench",
        source_sample_id=sample_id,
        identity_id=identity_id,
        provenance=Provenance(source_dataset="test", source_version="1"),
        image_sha256=image_sha or None,
        image_uri=image_uri or None,
    )


def _make_table(shas: list[str], attrs: list[str] | None = None) -> ImageScoreTable:
    """Build a table with deterministic scores for testing."""
    if attrs is None:
        attrs = ["Smiling", "Eyeglasses", "Wearing_Hat"]
    table = ImageScoreTable()
    for i, sha in enumerate(shas):
        for j, attr in enumerate(attrs):
            table.add(sha, attr, 0.5 + 0.01 * (i * 10 + j), image_uri=f"file://{sha}.png")
    return table


# --------------------------------------------------------------------------- #
# ImageScoreCacheKey
# --------------------------------------------------------------------------- #


class TestImageScoreCacheKey:
    def test_cache_key_is_deterministic(self):
        k1 = ImageScoreCacheKey.compute("sha1", "fp1", "pr1", "Smiling", "v1", "cs1")
        k2 = ImageScoreCacheKey.compute("sha1", "fp1", "pr1", "Smiling", "v1", "cs1")
        assert k1.cache_key() == k2.cache_key()

    def test_cache_key_changes_with_image(self):
        k1 = ImageScoreCacheKey.compute("sha1", "fp1", "pr1", "Smiling", "v1")
        k2 = ImageScoreCacheKey.compute("sha2", "fp1", "pr1", "Smiling", "v1")
        assert k1.cache_key() != k2.cache_key()

    def test_cache_key_changes_with_attribute(self):
        k1 = ImageScoreCacheKey.compute("sha1", "fp1", "pr1", "Smiling", "v1")
        k2 = ImageScoreCacheKey.compute("sha1", "fp1", "pr1", "Eyeglasses", "v1")
        assert k1.cache_key() != k2.cache_key()

    def test_cache_key_changes_with_model_fingerprint(self):
        k1 = ImageScoreCacheKey.compute("sha1", "fp1", "pr1", "Smiling", "v1")
        k2 = ImageScoreCacheKey.compute("sha1", "fp2", "pr1", "Smiling", "v1")
        assert k1.cache_key() != k2.cache_key()

    def test_cache_key_changes_with_scoring_version(self):
        k1 = ImageScoreCacheKey.compute("sha1", "fp1", "pr1", "Smiling", "v1")
        k2 = ImageScoreCacheKey.compute("sha1", "fp1", "pr1", "Smiling", "v2")
        assert k1.cache_key() != k2.cache_key()

    def test_cache_key_contains_all_fields(self):
        k = ImageScoreCacheKey.compute("sha1", "fp1", "pr1", "Smiling", "v1", "cs1")
        key_str = k.cache_key()
        assert "sha1" in key_str
        assert "fp1" in key_str
        assert "pr1" in key_str
        assert "Smiling" in key_str
        assert "v1" in key_str
        assert "cs1" in key_str


# --------------------------------------------------------------------------- #
# ImageScoreTable
# --------------------------------------------------------------------------- #


class TestImageScoreTable:
    def test_empty_table(self):
        table = ImageScoreTable()
        assert table.unique_images == 0
        assert table.get_scores("nonexistent") == {}
        assert not table.has_image("nonexistent")

    def test_add_and_lookup(self):
        table = ImageScoreTable()
        table.add("sha1", "Smiling", 0.95)
        table.add("sha1", "Eyeglasses", 0.10)
        assert table.has_image("sha1")
        assert table.unique_images == 1
        scores = table.get_scores("sha1")
        assert scores["Smiling"] == pytest.approx(0.95)
        assert scores["Eyeglasses"] == pytest.approx(0.10)

    def test_multiple_images(self):
        table = _make_table(["sha1", "sha2", "sha3"])
        assert table.unique_images == 3
        assert table.has_image("sha1")
        assert table.has_image("sha2")
        assert table.has_image("sha3")
        # Each image has different scores
        assert table.get_scores("sha1")["Smiling"] != table.get_scores("sha2")["Smiling"]

    def test_attribute_count(self):
        table = _make_table(["sha1"], ["Smiling", "Eyeglasses", "Wearing_Hat"])
        assert table.attribute_count("sha1") == 3
        assert table.attribute_count("nonexistent") == 0

    def test_from_score_rows(self):
        rows = [
            {"sample_id": "s1", "attribute": "Smiling", "p_positive": 0.9},
            {"sample_id": "s1", "attribute": "Eyeglasses", "p_positive": 0.1},
            {"sample_id": "s2", "attribute": "Smiling", "p_positive": 0.8},
            {"sample_id": "s2", "attribute": "Eyeglasses", "p_positive": 0.2},
        ]
        sample_to_image = {"s1": "sha_A", "s2": "sha_B"}
        table = ImageScoreTable.from_score_rows(rows, sample_to_image)
        assert table.unique_images == 2
        assert table.get_scores("sha_A")["Smiling"] == pytest.approx(0.9)
        assert table.get_scores("sha_B")["Eyeglasses"] == pytest.approx(0.2)

    def test_from_score_rows_skips_missing_sha(self):
        rows = [
            {"sample_id": "s1", "attribute": "Smiling", "p_positive": 0.9},
            {"sample_id": "s_no_sha", "attribute": "Smiling", "p_positive": 0.5},
        ]
        sample_to_image = {"s1": "sha_A"}  # s_no_sha not mapped
        table = ImageScoreTable.from_score_rows(rows, sample_to_image)
        assert table.unique_images == 1

    def test_deduplication_shares_scores(self):
        """P2-1: same image → same scores regardless of sample_id."""
        rows = [
            # Both s1 and s2 share sha_A; only s1's scores are stored
            {"sample_id": "s1", "attribute": "Smiling", "p_positive": 0.9},
            {"sample_id": "s2", "attribute": "Smiling", "p_positive": 0.9},
        ]
        sample_to_image = {"s1": "sha_A", "s2": "sha_A"}
        table = ImageScoreTable.from_score_rows(rows, sample_to_image)
        assert table.unique_images == 1
        assert table.get_scores("sha_A")["Smiling"] == pytest.approx(0.9)

    def test_to_image_score_rows(self):
        table = _make_table(["sha1"], ["Smiling", "Eyeglasses"])
        rows = table.to_image_score_rows(model_fingerprint="fp1", prompt_registry_hash="pr1")
        assert len(rows) == 2
        assert all(r["model_fingerprint"] == "fp1" for r in rows)
        assert all(r["prompt_id"] == "pr1" for r in rows)
        assert {r["attribute"] for r in rows} == {"Smiling", "Eyeglasses"}
        assert all(r["image_sha256"] == "sha1" for r in rows)

    def test_deduplication_metrics(self):
        table = _make_table(["sha1", "sha2"], ["Smiling", "Eyeglasses"])
        metrics = table.deduplication_metrics(
            canonical_samples=140,
            image_bearing_samples=140,
            raw_score_rows=4,  # 2 images × 2 attrs
        )
        assert metrics["canonical_samples"] == 140
        assert metrics["image_bearing_samples"] == 140
        assert metrics["unique_images"] == 2
        assert metrics["raw_visual_score_rows"] == 4
        # 140 samples × 2 attrs = 280 without dedup; 4 actual → 276 avoided
        assert metrics["avoided_duplicate_score_requests"] == 276


# --------------------------------------------------------------------------- #
# build_sample_to_image_sha
# --------------------------------------------------------------------------- #


class TestBuildSampleToImageSha:
    def test_maps_samples_with_sha(self):
        samples = [
            _sample("s1", image_sha="sha1"),
            _sample("s2", image_sha="sha2"),
            _sample("s3", image_sha=""),  # no SHA
        ]
        mapping = build_sample_to_image_sha(samples)
        assert mapping == {"s1": "sha1", "s2": "sha2"}

    def test_empty_for_no_images(self):
        samples = [_sample("s1"), _sample("s2")]
        assert build_sample_to_image_sha(samples) == {}


# --------------------------------------------------------------------------- #
# annotate_sample_via_image_table
# --------------------------------------------------------------------------- #


class TestAnnotateSampleViaImageTable:
    def _annotator(self) -> BenchmarkAnnotator:
        return BenchmarkAnnotator(AnnotationPolicy())

    def test_annotates_from_image_sha_field(self):
        table = _make_table(["sha1"], ["Smiling", "Eyeglasses"])
        sample = _sample("s1", image_sha="sha1")
        result = annotate_sample_via_image_table(self._annotator(), sample, table)
        assert celeba40_key("Smiling") in result.visual_attributes
        assert celeba40_key("Eyeglasses") in result.visual_attributes

    def test_annotates_from_sample_to_image_fallback(self):
        table = _make_table(["sha1"], ["Smiling"])
        sample = _sample("s1", image_sha="")  # no SHA on sample
        mapping = {"s1": "sha1"}  # but mapping knows the SHA
        result = annotate_sample_via_image_table(
            self._annotator(), sample, table, sample_to_image=mapping
        )
        assert celeba40_key("Smiling") in result.visual_attributes

    def test_falls_back_when_image_not_in_table(self):
        table = ImageScoreTable()  # empty
        sample = _sample("s1", image_sha="sha_unknown")
        result = annotate_sample_via_image_table(self._annotator(), sample, table)
        assert result.visual_attributes == {}

    def test_falls_back_for_no_image_sample(self):
        table = _make_table(["sha1"])
        sample = _sample("s1", image_sha="")
        result = annotate_sample_via_image_table(self._annotator(), sample, table)
        assert result.visual_attributes == {}

    def test_same_image_same_scores_across_samples(self):
        """P2-1: different samples sharing an image get identical annotations."""
        table = _make_table(["sha_shared"], ["Smiling", "Eyeglasses", "Wearing_Hat"])
        s1 = _sample("s1", image_sha="sha_shared")
        s2 = _sample("s2", image_sha="sha_shared")
        annotator = self._annotator()
        r1 = annotate_sample_via_image_table(annotator, s1, table)
        r2 = annotate_sample_via_image_table(annotator, s2, table)
        for key in r1.visual_attributes:
            assert r1.visual_attributes[key].score == r2.visual_attributes[key].score

    def test_preserves_existing_visual_attributes_when_no_image(self):
        """Non-image samples keep their existing visual_attributes untouched."""
        table = _make_table(["sha1"])
        existing_obs = AttributeObservation(
            name="source_attributes.fairface.skin_tone",
            label="dark",
            score=1.0,
            source="source_human",
        )
        sample = _sample("s1", image_sha="")
        sample.visual_attributes = {"source_attributes.fairface.skin_tone": existing_obs}
        result = annotate_sample_via_image_table(self._annotator(), sample, table)
        assert "source_attributes.fairface.skin_tone" in result.visual_attributes
        assert result.visual_attributes["source_attributes.fairface.skin_tone"].label == "dark"
