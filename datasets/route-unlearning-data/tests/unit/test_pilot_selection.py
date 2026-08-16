"""Unit tests for pilot identity selection (Stage 3, Commit 1).

These tests use *synthetic* data so they do not require GPU, model
weights, or the frozen baseline artifacts.  They verify the structural
invariants that the selection manifest must satisfy before any training
begins.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from route_data.eval.pilot_selection import (
    IdentityStats,
    PilotSelection,
    build_identity_attribute_map,
    build_identity_stats,
    load_selection_manifest,
    select_pilot_identities,
    selection_manifest_sha256,
    write_selection_manifest,
)


# --------------------------------------------------------------------------- #
# Helpers – synthetic data generators
# --------------------------------------------------------------------------- #

def _make_stats(
    identity_id: str,
    *,
    role: str = "train",
    attribute: str = "Bald",
    positive: bool = True,
    dv: float = 5.0,
    ipn: float = 4.0,
    wn: float = 3.0,
    vtc: float = 2.0,
    probes: int = 5,
    images: int = 10,
) -> IdentityStats:
    """Create an :class:`IdentityStats` with sensible defaults."""
    overall = (dv + ipn + wn + vtc) / 4.0
    return IdentityStats(
        identity_id=identity_id,
        protocol_role=role,
        target_attribute=attribute,
        attribute_positive=positive,
        mean_dv_margin=dv,
        mean_ipn_margin=ipn,
        mean_wn_margin=wn,
        mean_vtc_margin=vtc,
        mean_overall_margin=overall,
        probe_count=probes,
        unique_images=images,
    )


def _build_synthetic_stats(n_per_group: int = 8) -> dict[str, IdentityStats]:
    """Return a dict of synthetic identities across several attribute groups.

    The layout (all ``role='train'``):
      - 8 × (Bald, POS)     — high margin group
      - 8 × (Male, NEG)     — medium-high margin group
      - 8 × (Smiling, POS)  — medium margin group
      - 8 × (Bangs, NEG)    — low margin group
    """
    configs = [
        ("Bald", True, 8.0, 7.0),
        ("Male", False, 6.0, 5.0),
        ("Smiling", True, 4.0, 3.0),
        ("Bangs", False, 2.0, 1.0),
    ]
    stats: dict[str, IdentityStats] = {}
    idx = 0
    for attr, pos, dv, ipn in configs:
        for j in range(n_per_group):
            iid = f"id_{idx:04d}"
            stats[iid] = _make_stats(
                iid,
                attribute=attr,
                positive=pos,
                dv=dv + j * 0.1,
                ipn=ipn + j * 0.05,
            )
            idx += 1
    return stats


def _write_route_probe(path: Path, identities: dict[str, IdentityStats]) -> None:
    """Write a minimal route-probe JSONL for *identities*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    families = ["direct_visual", "image_plus_name", "wrong_name",
                "visual_text_conflict", "name_only"]
    with open(path, "w") as fh:
        for iid, s in identities.items():
            for fam in families:
                row = {
                    "probe_id": f"probe_{fam}_{iid}",
                    "identity_id": iid,
                    "probe_family": fam,
                    "target_attribute": s.target_attribute,
                    "answer_label": s.attribute_positive,
                    "image_sha256": f"img_{iid}_0",
                }
                fh.write(json.dumps(row) + "\n")


def _write_baseline_results(path: Path, stats: dict[str, IdentityStats]) -> None:
    """Write a minimal baseline-results JSONL matching *stats*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    families = ["direct_visual", "image_plus_name", "wrong_name",
                "visual_text_conflict", "name_only"]
    with open(path, "w") as fh:
        for iid, s in stats.items():
            for fam in families:
                margin = {
                    "direct_visual": s.mean_dv_margin,
                    "image_plus_name": s.mean_ipn_margin,
                    "wrong_name": s.mean_wn_margin,
                    "visual_text_conflict": s.mean_vtc_margin,
                }.get(fam, None)
                row = {
                    "probe_id": f"probe_{fam}_{iid}",
                    "identity_id": iid,
                    "probe_family": fam,
                    "protocol_role": s.protocol_role,
                    "signed_answer_margin": margin,
                }
                fh.write(json.dumps(row) + "\n")


def _write_processed_dataset(path: Path, stats: dict[str, IdentityStats]) -> None:
    """Write a minimal processed-dataset JSONL with image counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for iid, s in stats.items():
            for k in range(s.unique_images):
                row = {
                    "identity_id": iid,
                    "image_sha256": f"img_{iid}_{k}",
                }
                fh.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------- #
# Tests – attribute map
# --------------------------------------------------------------------------- #

class TestBuildIdentityAttributeMap:
    """Tests for :func:`build_identity_attribute_map`."""

    def test_basic_mapping(self, tmp_path: Path) -> None:
        stats = _build_synthetic_stats(n_per_group=2)
        rp = tmp_path / "route_probe.jsonl"
        _write_route_probe(rp, stats)

        attr_map = build_identity_attribute_map(rp)
        assert len(attr_map) == len(stats)
        for iid, s in stats.items():
            attr, is_pos = attr_map[iid]
            assert attr == s.target_attribute
            assert is_pos == s.attribute_positive

    def test_positive_detection(self, tmp_path: Path) -> None:
        """Positive state is detected when any probe has answer_label=True."""
        rp = tmp_path / "route_probe.jsonl"
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w") as fh:
            fh.write(json.dumps({
                "identity_id": "x1",
                "target_attribute": "Bald",
                "answer_label": True,
            }) + "\n")
            fh.write(json.dumps({
                "identity_id": "x1",
                "target_attribute": "Bald",
                "answer_label": False,
            }) + "\n")

        attr_map = build_identity_attribute_map(rp)
        assert attr_map["x1"] == ("Bald", True)

    def test_negative_state(self, tmp_path: Path) -> None:
        """All-False answer_labels → negative state."""
        rp = tmp_path / "route_probe.jsonl"
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w") as fh:
            fh.write(json.dumps({
                "identity_id": "y1",
                "target_attribute": "Male",
                "answer_label": False,
            }) + "\n")

        attr_map = build_identity_attribute_map(rp)
        assert attr_map["y1"] == ("Male", False)


# --------------------------------------------------------------------------- #
# Tests – identity stats
# --------------------------------------------------------------------------- #

class TestBuildIdentityStats:
    """Tests for :func:`build_identity_stats`."""

    def test_stats_from_synthetic_data(self, tmp_path: Path) -> None:
        raw_stats = _build_synthetic_stats(n_per_group=3)
        rp = tmp_path / "route_probe.jsonl"
        br = tmp_path / "baseline_results.jsonl"
        ds = tmp_path / "processed.jsonl"
        _write_route_probe(rp, raw_stats)
        _write_baseline_results(br, raw_stats)
        _write_processed_dataset(ds, raw_stats)

        stats = build_identity_stats(br, rp, ds)
        assert len(stats) == len(raw_stats)

        for iid, s in stats.items():
            assert s.probe_count == 5
            assert s.unique_images == raw_stats[iid].unique_images
            assert s.protocol_role == "train"
            assert math.isfinite(s.mean_dv_margin)

    def test_unique_images_optional(self, tmp_path: Path) -> None:
        """processed_dataset_path is optional; unique_images defaults to 0."""
        raw_stats = _build_synthetic_stats(n_per_group=2)
        rp = tmp_path / "route_probe.jsonl"
        br = tmp_path / "baseline_results.jsonl"
        _write_route_probe(rp, raw_stats)
        _write_baseline_results(br, raw_stats)

        stats = build_identity_stats(br, rp, processed_dataset_path=None)
        for s in stats.values():
            assert s.unique_images == 0


# --------------------------------------------------------------------------- #
# Tests – selection invariants
# --------------------------------------------------------------------------- #

class TestSelectPilotIdentities:
    """Tests for :func:`select_pilot_identities`."""

    def test_correct_group_sizes(self) -> None:
        stats = _build_synthetic_stats(n_per_group=8)
        sel = select_pilot_identities(stats, seed=17)
        assert len(sel.target_identities) == 2
        assert len(sel.retain_identities) == 2
        assert len(sel.control_identities) == 2

    def test_no_overlap(self) -> None:
        stats = _build_synthetic_stats(n_per_group=8)
        sel = select_pilot_identities(stats, seed=17)
        all_ids = (
            set(sel.target_identities)
            | set(sel.retain_identities)
            | set(sel.control_identities)
        )
        total = (
            len(sel.target_identities)
            + len(sel.retain_identities)
            + len(sel.control_identities)
        )
        assert len(all_ids) == total, "Groups must not overlap"

    def test_deterministic(self) -> None:
        """Same seed → identical selection."""
        stats = _build_synthetic_stats(n_per_group=8)
        sel_a = select_pilot_identities(stats, seed=17)
        sel_b = select_pilot_identities(stats, seed=17)
        assert sel_a.target_identities == sel_b.target_identities
        assert sel_a.retain_identities == sel_b.retain_identities
        assert sel_a.control_identities == sel_b.control_identities

    def test_different_seed_differs(self) -> None:
        """Different seeds should (usually) produce different selections."""
        stats = _build_synthetic_stats(n_per_group=8)
        sel_a = select_pilot_identities(stats, seed=17)
        sel_b = select_pilot_identities(stats, seed=42)
        # At least one group should differ
        differs = (
            sel_a.target_identities != sel_b.target_identities
            or sel_a.retain_identities != sel_b.retain_identities
            or sel_a.control_identities != sel_b.control_identities
        )
        assert differs, "Different seeds should produce different selections"

    def test_target_highest_margin_attribute(self) -> None:
        """Target group should come from the highest-margin attribute group."""
        stats = _build_synthetic_stats(n_per_group=8)
        sel = select_pilot_identities(stats, seed=17)

        # The synthetic data has Bald/POS as the highest-margin group.
        for iid in sel.target_identities:
            detail = sel.identity_details[iid]
            assert detail["target_attribute"] == "Bald"
            assert detail["attribute_positive"] is True

    def test_retain_control_different_attribute(self) -> None:
        """Retain and control identities should NOT share the target attribute."""
        stats = _build_synthetic_stats(n_per_group=8)
        sel = select_pilot_identities(stats, seed=17)

        target_attr = sel.identity_details[sel.target_identities[0]]["target_attribute"]
        for iid in sel.retain_identities + sel.control_identities:
            detail = sel.identity_details[iid]
            assert detail["target_attribute"] != target_attr

    def test_all_preferred_role(self) -> None:
        """All selected identities should have the preferred protocol role."""
        stats = _build_synthetic_stats(n_per_group=8)
        sel = select_pilot_identities(stats, seed=17, preferred_role="train")
        for iid, detail in sel.identity_details.items():
            assert detail["protocol_role"] == "train"

    def test_insufficient_identities_raises(self) -> None:
        """Should raise if not enough eligible identities."""
        stats = _build_synthetic_stats(n_per_group=1)  # only 4 identities
        with pytest.raises(ValueError, match="Not enough"):
            select_pilot_identities(stats, seed=17)

    def test_custom_counts(self) -> None:
        """Custom group sizes should be respected."""
        stats = _build_synthetic_stats(n_per_group=10)
        sel = select_pilot_identities(
            stats, target_count=3, retain_count=3, control_count=3, seed=17,
        )
        assert len(sel.target_identities) == 3
        assert len(sel.retain_identities) == 3
        assert len(sel.control_identities) == 3
        all_ids = (
            set(sel.target_identities)
            | set(sel.retain_identities)
            | set(sel.control_identities)
        )
        assert len(all_ids) == 9

    def test_sorted_identity_lists(self) -> None:
        """Identity lists in the selection should be sorted."""
        stats = _build_synthetic_stats(n_per_group=8)
        sel = select_pilot_identities(stats, seed=17)
        assert sel.target_identities == sorted(sel.target_identities)
        assert sel.retain_identities == sorted(sel.retain_identities)
        assert sel.control_identities == sorted(sel.control_identities)

    def test_matching_criteria_populated(self) -> None:
        stats = _build_synthetic_stats(n_per_group=8)
        sel = select_pilot_identities(stats, seed=17)
        assert "protocol_role" in sel.matching_criteria
        assert "baseline_direct_visual_margin" in sel.matching_criteria

    def test_identity_details_complete(self) -> None:
        """Every selected identity should have full detail in the manifest."""
        stats = _build_synthetic_stats(n_per_group=8)
        sel = select_pilot_identities(stats, seed=17)
        all_ids = (
            sel.target_identities + sel.retain_identities + sel.control_identities
        )
        for iid in all_ids:
            detail = sel.identity_details[iid]
            assert detail["group"] in ("target", "retain", "control")
            assert "protocol_role" in detail
            assert "target_attribute" in detail
            assert "attribute_positive" in detail
            assert "mean_dv_margin" in detail
            assert "mean_ipn_margin" in detail
            assert "mean_overall_margin" in detail
            assert "probe_count" in detail


# --------------------------------------------------------------------------- #
# Tests – manifest serialisation
# --------------------------------------------------------------------------- #

class TestManifest:
    """Tests for manifest write / load / SHA-256."""

    def test_write_and_load_roundtrip(self, tmp_path: Path) -> None:
        stats = _build_synthetic_stats(n_per_group=8)
        sel = select_pilot_identities(stats, seed=17)
        manifest_path = tmp_path / "manifest.json"

        write_selection_manifest(
            sel, manifest_path,
            baseline_manifest_sha256="aaa",
            baseline_results_sha256="bbb",
            route_probe_sha256="ccc",
            processed_dataset_sha256="ddd",
            code_commit="eee",
        )

        loaded = load_selection_manifest(manifest_path)
        assert loaded["target_identities"] == sel.target_identities
        assert loaded["retain_identities"] == sel.retain_identities
        assert loaded["control_identities"] == sel.control_identities
        assert loaded["baseline_manifest_sha256"] == "aaa"
        assert loaded["code_commit"] == "eee"

    def test_manifest_sha256_stable(self, tmp_path: Path) -> None:
        """Writing the same selection twice → identical SHA-256."""
        stats = _build_synthetic_stats(n_per_group=8)
        sel = select_pilot_identities(stats, seed=17)

        p1 = tmp_path / "m1.json"
        p2 = tmp_path / "m2.json"
        for p in (p1, p2):
            write_selection_manifest(
                sel, p,
                baseline_manifest_sha256="aaa",
                baseline_results_sha256="bbb",
                route_probe_sha256="ccc",
                processed_dataset_sha256="ddd",
                code_commit="eee",
            )

        sha1 = selection_manifest_sha256(p1)
        sha2 = selection_manifest_sha256(p2)
        assert sha1 == sha2

    def test_manifest_is_valid_json(self, tmp_path: Path) -> None:
        stats = _build_synthetic_stats(n_per_group=8)
        sel = select_pilot_identities(stats, seed=17)
        manifest_path = tmp_path / "manifest.json"
        write_selection_manifest(sel, manifest_path)

        data = json.loads(manifest_path.read_text())
        assert isinstance(data, dict)
        assert "selection_version" in data
        assert data["selection_version"] == "pilot-selection-v1"

    def test_manifest_creates_parent_dirs(self, tmp_path: Path) -> None:
        stats = _build_synthetic_stats(n_per_group=8)
        sel = select_pilot_identities(stats, seed=17)
        nested = tmp_path / "a" / "b" / "c" / "manifest.json"
        write_selection_manifest(sel, nested)
        assert nested.exists()


# --------------------------------------------------------------------------- #
# Tests – eligibility filter
# --------------------------------------------------------------------------- #

class TestEligibility:
    """Tests for the internal eligibility filter."""

    def test_wrong_role_excluded(self) -> None:
        # Build identities across multiple attribute groups so retain/control
        # can be found.  Identity "b" has role='eval' and must not appear.
        stats: dict[str, IdentityStats] = {}
        attrs = [("Bald", True), ("Male", False), ("Smiling", True)]
        for attr, pos in attrs:
            for j in range(5):
                iid = f"{attr}_{j}"
                stats[iid] = _make_stats(iid, role="train", attribute=attr, positive=pos)
        # Add the eval-role identity that must be excluded
        stats["b"] = _make_stats("b", role="eval", attribute="Bangs", positive=False)

        sel = select_pilot_identities(stats, seed=17, preferred_role="train")
        all_ids = (
            set(sel.target_identities)
            | set(sel.retain_identities)
            | set(sel.control_identities)
        )
        assert "b" not in all_ids

    def test_fewer_than_5_probes_excluded(self) -> None:
        # Build identities across multiple attribute groups with diverse attrs.
        stats: dict[str, IdentityStats] = {}
        attrs = [("Bald", True), ("Male", False), ("Smiling", True), ("Bangs", False)]
        idx = 0
        for attr, pos in attrs:
            for j in range(5):
                iid = f"id_{idx}"
                probes = 5  # all eligible by probe count
                stats[iid] = _make_stats(iid, attribute=attr, positive=pos, probes=probes)
                idx += 1
        # Add identities with too few probes — they should be excluded
        for i in range(3):
            iid = f"few_{i}"
            stats[iid] = _make_stats(iid, attribute="Gray_Hair", positive=True, probes=3)

        sel = select_pilot_identities(stats, seed=17)
        all_ids = (
            set(sel.target_identities)
            | set(sel.retain_identities)
            | set(sel.control_identities)
        )
        for iid in all_ids:
            assert sel.identity_details[iid]["probe_count"] >= 5
        # The "few_*" identities should not appear
        for iid in all_ids:
            assert not iid.startswith("few_")
