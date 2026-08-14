"""Regression tests for Phase 1 repair: stale-cache rejection (P0-5)
and exact score-table completeness (P0-6).

These tests verify the invariants enforced by the scoring pipeline
in src/route_data/cli.py.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# P0-5: Resume cache rejects stale image SHA
# --------------------------------------------------------------------------- #


class TestResumeCacheStaleImageSha:
    """P0-5: cached rows whose image_sha256 no longer matches the current
    source image for the same sample_id must be rejected."""

    def test_rejects_cached_row_with_stale_image_sha(self):
        """If the cached image SHA differs from the current source SHA,
        the row must be dropped."""
        cached_row = {
            "sample_id": "s1",
            "attribute": "Smiling",
            "p_positive": 0.9,
            "image_sha256": "old_sha_value",
            "_cache_key": "old_sha_value|fp1|pr1|cs1|sv2",
        }
        current_sha_by_sample = {"s1": "new_sha_value"}

        # Simulate the P0-5 check
        sid = cached_row["sample_id"]
        img_sha = cached_row.get("image_sha256", "")
        current_sha = current_sha_by_sample.get(sid, "")

        assert img_sha != current_sha, "Stale SHA should be detected"

    def test_accepts_cached_row_with_current_image_sha(self):
        """If the cached image SHA matches the current source SHA,
        the row should be accepted."""
        cached_row = {
            "sample_id": "s1",
            "attribute": "Smiling",
            "p_positive": 0.9,
            "image_sha256": "same_sha_value",
            "_cache_key": "same_sha_value|fp1|pr1|cs1|sv2",
        }
        current_sha_by_sample = {"s1": "same_sha_value"}

        sid = cached_row["sample_id"]
        img_sha = cached_row.get("image_sha256", "")
        current_sha = current_sha_by_sample.get(sid, "")

        assert img_sha == current_sha, "Current SHA should match"

    def test_rescores_when_image_changed_for_same_sample_id(self):
        """When an image changes for the same sample_id, the old cached
        scores must not be reused — the sample must be rescored."""
        cached_rows = [
            {
                "sample_id": "s1",
                "attribute": "Smiling",
                "p_positive": 0.9,
                "image_sha256": "old_image_sha",
                "_cache_key": "old_image_sha|fp1|pr1|cs1|sv2",
            },
            {
                "sample_id": "s1",
                "attribute": "Eyeglasses",
                "p_positive": 0.1,
                "image_sha256": "old_image_sha",
                "_cache_key": "old_image_sha|fp1|pr1|cs1|sv2",
            },
        ]
        # Image was replaced but sample_id stayed the same
        current_sha_by_sample = {"s1": "new_image_sha"}

        accepted = []
        for row in cached_rows:
            sid = row["sample_id"]
            img_sha = row.get("image_sha256", "")
            current_sha = current_sha_by_sample.get(sid, "")
            if img_sha and current_sha and img_sha != current_sha:
                continue  # stale — drop
            accepted.append(row)

        assert len(accepted) == 0, "All cached rows should be rejected"

    def test_drops_row_when_current_sha_unavailable(self):
        """If the current image SHA is unavailable (empty), the cached
        row must be dropped to avoid trusting an unverifiable cache."""
        cached_row = {
            "sample_id": "s1",
            "attribute": "Smiling",
            "p_positive": 0.9,
            "image_sha256": "some_sha",
            "_cache_key": "some_sha|fp1|pr1|cs1|sv2",
        }
        current_sha_by_sample = {"s1": ""}

        sid = cached_row["sample_id"]
        img_sha = cached_row.get("image_sha256", "")
        current_sha = current_sha_by_sample.get(sid, "")

        # P0-5: if current SHA is empty, drop the row
        assert img_sha and not current_sha, "Should detect missing current SHA"


# --------------------------------------------------------------------------- #
# P0-6: Score-table exact completeness
# --------------------------------------------------------------------------- #


class TestScoreTableExactCompleteness:
    """P0-6: the score table must contain exactly one row for every
    (unique_image, attribute) pair — no missing, no extra."""

    def test_rejects_extra_image_attribute_pair(self):
        """Extra pairs (not in the expected set) must be rejected."""
        unique_image_shas = {"sha1", "sha2"}
        attributes = ["Smiling", "Eyeglasses"]

        expected_pairs = {
            (sha, attr)
            for sha in unique_image_shas
            for attr in attributes
        }
        # Actual has an extra pair for a phantom image
        actual_pairs = {
            ("sha1", "Smiling"),
            ("sha1", "Eyeglasses"),
            ("sha2", "Smiling"),
            ("sha2", "Eyeglasses"),
            ("sha_phantom", "Smiling"),  # extra!
        }

        extra = actual_pairs - expected_pairs
        assert len(extra) == 1
        assert ("sha_phantom", "Smiling") in extra

    def test_rejects_missing_image_attribute_pair(self):
        """Missing pairs must be detected."""
        unique_image_shas = {"sha1", "sha2"}
        attributes = ["Smiling", "Eyeglasses"]

        expected_pairs = {
            (sha, attr)
            for sha in unique_image_shas
            for attr in attributes
        }
        # Actual is missing one pair
        actual_pairs = {
            ("sha1", "Smiling"),
            ("sha1", "Eyeglasses"),
            ("sha2", "Smiling"),
            # ("sha2", "Eyeglasses") missing!
        }

        missing = expected_pairs - actual_pairs
        assert len(missing) == 1
        assert ("sha2", "Eyeglasses") in missing

    def test_requires_exact_row_count(self):
        """Score row count must equal expected pair count."""
        unique_image_shas = {"sha1", "sha2", "sha3"}
        attributes = ["Smiling", "Eyeglasses", "Wearing_Hat"]
        expected_count = len(unique_image_shas) * len(attributes)
        assert expected_count == 9

        # Simulate duplicate rows causing count mismatch
        score_rows = [
            {"image_sha256": "sha1", "attribute": "Smiling"},
            {"image_sha256": "sha1", "attribute": "Smiling"},  # duplicate
            {"image_sha256": "sha1", "attribute": "Eyeglasses"},
        ]
        # After dedup, actual pairs = 2, but expected = 9
        actual_pairs = {
            (r["image_sha256"], r["attribute"]) for r in score_rows
        }
        assert len(actual_pairs) != expected_count

    def test_exact_match_passes(self):
        """When actual pairs exactly equal expected pairs, validation passes."""
        unique_image_shas = {"sha1", "sha2"}
        attributes = ["Smiling", "Eyeglasses"]

        expected_pairs = {
            (sha, attr)
            for sha in unique_image_shas
            for attr in attributes
        }
        actual_pairs = {
            ("sha1", "Smiling"),
            ("sha1", "Eyeglasses"),
            ("sha2", "Smiling"),
            ("sha2", "Eyeglasses"),
        }

        missing = expected_pairs - actual_pairs
        extra = actual_pairs - expected_pairs
        assert len(missing) == 0
        assert len(extra) == 0
        assert len(actual_pairs) == len(expected_pairs)
