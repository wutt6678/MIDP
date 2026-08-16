"""Unit tests for paired_analysis module (Commit 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from route_data.eval.paired_analysis import (
    PairedAnalysis,
    PairedAnalysisConfig,
    compute_group_effects,
    compute_identity_effects,
    compute_preservation_report,
    compute_probe_deltas,
    compute_route_effects,
    load_results_jsonl,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_row(
    probe_id: str = "p1",
    identity_id: str = "id_A",
    probe_family: str = "direct_visual",
    signed_answer_margin: float | None = 5.0,
    predicted_label: str | None = "Yes",
    correct: bool | None = True,
    token_overlap: float | None = None,
    target_attribute: str | None = "hair_color",
    answer_label: bool | None = True,
    protocol_role: str = "eval",
) -> dict:
    """Build a minimal result row."""
    return {
        "probe_id": probe_id,
        "identity_id": identity_id,
        "probe_family": probe_family,
        "signed_answer_margin": signed_answer_margin,
        "predicted_label": predicted_label,
        "correct": correct,
        "token_overlap": token_overlap,
        "target_attribute": target_attribute,
        "answer_label": answer_label,
        "protocol_role": protocol_role,
    }


def _make_name_only_row(
    probe_id: str = "p_no1",
    identity_id: str = "id_A",
    token_overlap: float = 0.3,
    correct: bool | None = False,
    target_attribute: str | None = None,
    answer_label: bool | None = None,
    protocol_role: str = "eval",
) -> dict:
    return {
        "probe_id": probe_id,
        "identity_id": identity_id,
        "probe_family": "name_only",
        "signed_answer_margin": None,
        "predicted_label": None,
        "correct": correct,
        "token_overlap": token_overlap,
        "target_attribute": target_attribute,
        "answer_label": answer_label,
        "protocol_role": protocol_role,
    }


# --------------------------------------------------------------------------- #
# Tests: load_results_jsonl
# --------------------------------------------------------------------------- #

class TestLoadResults:
    def test_load_jsonl(self, tmp_path: Path) -> None:
        f = tmp_path / "results.jsonl"
        rows = [{"probe_id": "a"}, {"probe_id": "b"}]
        with f.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        loaded = load_results_jsonl(f)
        assert len(loaded) == 2
        assert loaded[0]["probe_id"] == "a"

    def test_load_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        assert load_results_jsonl(f) == []


# --------------------------------------------------------------------------- #
# Tests: compute_probe_deltas
# --------------------------------------------------------------------------- #

class TestComputeProbeDeltas:
    def test_basic_delta(self) -> None:
        bl = [_make_row(probe_id="p1", signed_answer_margin=6.0)]
        po = [_make_row(probe_id="p1", signed_answer_margin=4.0)]
        deltas = compute_probe_deltas(bl, po)
        assert len(deltas) == 1
        d = deltas[0]
        assert d["pre_signed_margin"] == 6.0
        assert d["post_signed_margin"] == 4.0
        assert d["delta_signed_margin"] == pytest.approx(-2.0)
        assert d["prediction_changed"] is False

    def test_prediction_changed(self) -> None:
        bl = [_make_row(probe_id="p1", predicted_label="Yes")]
        po = [_make_row(probe_id="p1", predicted_label="No")]
        deltas = compute_probe_deltas(bl, po)
        assert deltas[0]["prediction_changed"] is True

    def test_missing_post(self) -> None:
        bl = [_make_row(probe_id="p1"), _make_row(probe_id="p2")]
        po = [_make_row(probe_id="p1")]
        deltas = compute_probe_deltas(bl, po)
        assert len(deltas) == 1

    def test_name_only_token_overlap(self) -> None:
        bl = [_make_name_only_row(probe_id="no1", token_overlap=0.4)]
        po = [_make_name_only_row(probe_id="no1", token_overlap=0.2)]
        deltas = compute_probe_deltas(bl, po)
        assert len(deltas) == 1
        d = deltas[0]
        assert d["pre_token_overlap"] == pytest.approx(0.4)
        assert d["post_token_overlap"] == pytest.approx(0.2)
        assert d["delta_token_overlap"] == pytest.approx(-0.2)
        assert "delta_signed_margin" not in d

    def test_identity_groups(self) -> None:
        bl = [_make_row(probe_id="p1", identity_id="tgt_1")]
        po = [_make_row(probe_id="p1", identity_id="tgt_1")]
        groups = {"tgt_1": "target"}
        deltas = compute_probe_deltas(bl, po, identity_groups=groups)
        assert deltas[0]["group"] == "target"

    def test_untargeted_default(self) -> None:
        bl = [_make_row(probe_id="p1", identity_id="unknown")]
        po = [_make_row(probe_id="p1", identity_id="unknown")]
        deltas = compute_probe_deltas(bl, po)
        assert deltas[0]["group"] == "untargeted"

    def test_none_margins(self) -> None:
        bl = [_make_row(probe_id="p1", signed_answer_margin=None)]
        po = [_make_row(probe_id="p1", signed_answer_margin=3.0)]
        deltas = compute_probe_deltas(bl, po)
        assert deltas[0]["delta_signed_margin"] is None


# --------------------------------------------------------------------------- #
# Tests: compute_identity_effects
# --------------------------------------------------------------------------- #

class TestComputeIdentityEffects:
    def test_single_identity(self) -> None:
        deltas = [
            {"identity_id": "A", "family": "direct_visual", "delta_signed_margin": -1.0},
            {"identity_id": "A", "family": "direct_visual", "delta_signed_margin": -3.0},
            {"identity_id": "A", "family": "image_plus_name", "delta_signed_margin": 0.5},
        ]
        effects = compute_identity_effects(deltas)
        assert "A" in effects
        assert effects["A"]["direct_visual_dM"] == pytest.approx(-2.0)
        assert effects["A"]["image_plus_name_dM"] == pytest.approx(0.5)
        assert effects["A"]["wrong_name_dM"] is None

    def test_overall_visual(self) -> None:
        deltas = [
            {"identity_id": "B", "family": "direct_visual", "delta_signed_margin": 2.0},
            {"identity_id": "B", "family": "wrong_name", "delta_signed_margin": -2.0},
        ]
        effects = compute_identity_effects(deltas)
        # overall = mean of [2.0, -2.0] = 0.0
        assert effects["B"]["overall_visual_dM"] == pytest.approx(0.0)

    def test_name_only_skipped(self) -> None:
        deltas = [
            {"identity_id": "C", "family": "name_only", "delta_token_overlap": -0.1},
        ]
        effects = compute_identity_effects(deltas)
        # name_only is skipped, so C should not appear
        assert "C" not in effects

    def test_probe_count(self) -> None:
        deltas = [
            {"identity_id": "D", "family": "direct_visual", "delta_signed_margin": 1.0},
            {"identity_id": "D", "family": "visual_text_conflict", "delta_signed_margin": -0.5},
        ]
        effects = compute_identity_effects(deltas)
        assert effects["D"]["probe_count"] == 2


# --------------------------------------------------------------------------- #
# Tests: compute_group_effects
# --------------------------------------------------------------------------- #

class TestComputeGroupEffects:
    def test_target_group(self) -> None:
        deltas = [
            {"group": "target", "family": "direct_visual", "delta_signed_margin": -2.0},
            {"group": "target", "family": "direct_visual", "delta_signed_margin": -4.0},
        ]
        ge = compute_group_effects(deltas)
        assert ge["target"]["overall"]["mean"] == pytest.approx(-3.0)
        assert ge["target"]["overall"]["count"] == 2
        assert ge["target"]["overall"]["median"] == pytest.approx(-3.0)

    def test_empty_group(self) -> None:
        deltas = [
            {"group": "target", "family": "direct_visual", "delta_signed_margin": 1.0},
        ]
        ge = compute_group_effects(deltas)
        assert ge["retain"]["overall"]["count"] == 0
        assert ge["retain"]["overall"]["mean"] is None

    def test_per_family(self) -> None:
        deltas = [
            {"group": "retain", "family": "wrong_name", "delta_signed_margin": 0.5},
            {"group": "retain", "family": "wrong_name", "delta_signed_margin": 1.5},
        ]
        ge = compute_group_effects(deltas)
        assert ge["retain"]["per_family"]["wrong_name"]["mean"] == pytest.approx(1.0)

    def test_name_only_uses_token_overlap(self) -> None:
        deltas = [
            {"group": "control", "family": "name_only", "delta_token_overlap": -0.1},
        ]
        ge = compute_group_effects(deltas)
        assert ge["control"]["overall"]["count"] == 1

    def test_std_single_value(self) -> None:
        deltas = [
            {"group": "target", "family": "direct_visual", "delta_signed_margin": 5.0},
        ]
        ge = compute_group_effects(deltas)
        assert ge["target"]["overall"]["std"] == 0.0


# --------------------------------------------------------------------------- #
# Tests: compute_preservation_report
# --------------------------------------------------------------------------- #

class TestComputePreservationReport:
    def test_global_direct_visual(self) -> None:
        bl = [
            _make_row(probe_id="p1", probe_family="direct_visual", correct=True, signed_answer_margin=5.0),
            _make_row(probe_id="p2", probe_family="direct_visual", correct=False, signed_answer_margin=-1.0),
        ]
        po = [
            _make_row(probe_id="p1", probe_family="direct_visual", correct=True, signed_answer_margin=4.0),
            _make_row(probe_id="p2", probe_family="direct_visual", correct=True, signed_answer_margin=2.0),
        ]
        report = compute_preservation_report(bl, po)
        assert report["global_direct_visual"]["pre_accuracy"] == pytest.approx(0.5)
        assert report["global_direct_visual"]["post_accuracy"] == pytest.approx(1.0)
        assert report["global_direct_visual"]["count"] == 2

    def test_positive_negative_state(self) -> None:
        bl = [
            _make_row(probe_id="p1", probe_family="direct_visual", answer_label=True, correct=True),
            _make_row(probe_id="p2", probe_family="direct_visual", answer_label=False, correct=True),
        ]
        po = [
            _make_row(probe_id="p1", probe_family="direct_visual", answer_label=True, correct=False),
            _make_row(probe_id="p2", probe_family="direct_visual", answer_label=False, correct=True),
        ]
        report = compute_preservation_report(bl, po)
        assert report["positive_state"]["pre_accuracy"] == pytest.approx(1.0)
        assert report["positive_state"]["post_accuracy"] == pytest.approx(0.0)
        assert report["negative_state"]["pre_accuracy"] == pytest.approx(1.0)
        assert report["negative_state"]["post_accuracy"] == pytest.approx(1.0)

    def test_retain_control_groups(self) -> None:
        bl = [
            _make_row(probe_id="p1", identity_id="r1", probe_family="direct_visual", correct=True),
            _make_row(probe_id="p2", identity_id="c1", probe_family="direct_visual", correct=True),
        ]
        po = [
            _make_row(probe_id="p1", identity_id="r1", probe_family="direct_visual", correct=False),
            _make_row(probe_id="p2", identity_id="c1", probe_family="direct_visual", correct=True),
        ]
        groups = {"r1": "retain", "c1": "control"}
        report = compute_preservation_report(bl, po, identity_groups=groups)
        assert report["retain_group"]["pre_accuracy"] == pytest.approx(1.0)
        assert report["retain_group"]["post_accuracy"] == pytest.approx(0.0)
        assert report["control_group"]["post_accuracy"] == pytest.approx(1.0)

    def test_per_attribute(self) -> None:
        bl = [
            _make_row(probe_id="p1", probe_family="direct_visual", target_attribute="hair_color", correct=True),
            _make_row(probe_id="p2", probe_family="image_plus_name", target_attribute="hair_color", correct=True),
        ]
        po = [
            _make_row(probe_id="p1", probe_family="direct_visual", target_attribute="hair_color", correct=False),
            _make_row(probe_id="p2", probe_family="image_plus_name", target_attribute="hair_color", correct=True),
        ]
        report = compute_preservation_report(bl, po)
        assert "hair_color" in report["per_attribute"]
        assert report["per_attribute"]["hair_color"]["pre_accuracy"] == pytest.approx(1.0)
        assert report["per_attribute"]["hair_color"]["post_accuracy"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Tests: compute_route_effects
# --------------------------------------------------------------------------- #

class TestComputeRouteEffects:
    def _make_rows_for_group(
        self,
        group: str,
        identity_id: str,
        margins: dict[str, tuple[float, float]],
    ) -> tuple[list[dict], list[dict]]:
        """Create baseline/post rows for a single identity.

        ``margins`` maps family → (pre_margin, post_margin).
        """
        bl, po = [], []
        for i, (fam, (pre_m, post_m)) in enumerate(margins.items()):
            pid = f"{group}_{fam}_{i}"
            bl.append(_make_row(
                probe_id=pid, identity_id=identity_id,
                probe_family=fam, signed_answer_margin=pre_m,
            ))
            po.append(_make_row(
                probe_id=pid, identity_id=identity_id,
                probe_family=fam, signed_answer_margin=post_m,
            ))
        return bl, po

    def test_target_route_effects(self) -> None:
        margins = {
            "direct_visual": (6.0, 4.0),
            "image_plus_name": (5.0, 3.0),
            "wrong_name": (5.5, 3.5),
            "visual_text_conflict": (4.0, 2.0),
        }
        bl, po = self._make_rows_for_group("target", "t1", margins)
        groups = {"t1": "target"}

        re = compute_route_effects(bl, po, identity_groups=groups)

        # Pre: Δ_name = 5.0 - 6.0 = -1.0
        assert re["target"]["delta_name"]["pre"] == pytest.approx(-1.0)
        # Post: Δ_name = 3.0 - 4.0 = -1.0
        assert re["target"]["delta_name"]["post"] == pytest.approx(-1.0)
        # Change: -1.0 - (-1.0) = 0.0
        assert re["target"]["delta_name"]["change"] == pytest.approx(0.0)

        # Δ_conflict pre = 4.0 - 6.0 = -2.0
        assert re["target"]["delta_conflict"]["pre"] == pytest.approx(-2.0)

    def test_empty_group(self) -> None:
        bl = [_make_row(probe_id="p1", identity_id="t1", probe_family="direct_visual")]
        po = [_make_row(probe_id="p1", identity_id="t1", probe_family="direct_visual")]
        groups = {"t1": "target"}
        re = compute_route_effects(bl, po, identity_groups=groups)
        # retain and control should have None values
        assert re["retain"]["delta_name"]["pre"] is None
        assert re["control"]["delta_conflict"]["post"] is None

    def test_name_only_excluded(self) -> None:
        bl = [
            _make_row(probe_id="p1", identity_id="t1", probe_family="direct_visual"),
            _make_name_only_row(probe_id="no1", identity_id="t1"),
        ]
        po = [
            _make_row(probe_id="p1", identity_id="t1", probe_family="direct_visual"),
            _make_name_only_row(probe_id="no1", identity_id="t1"),
        ]
        groups = {"t1": "target"}
        re = compute_route_effects(bl, po, identity_groups=groups)
        # Should not crash; name_only rows are excluded from route effects
        assert re["target"]["delta_name"]["pre"] is None  # only dv has data


# --------------------------------------------------------------------------- #
# Tests: PairedAnalysis orchestrator
# --------------------------------------------------------------------------- #

class TestPairedAnalysis:
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    def test_full_pipeline(self, tmp_path: Path) -> None:
        bl_path = tmp_path / "baseline.jsonl"
        po_path = tmp_path / "post.jsonl"
        sel_path = tmp_path / "selection.json"
        out_dir = tmp_path / "analysis"

        bl = [
            _make_row(probe_id="p1", identity_id="t1", probe_family="direct_visual",
                       signed_answer_margin=6.0, correct=True),
            _make_row(probe_id="p2", identity_id="t1", probe_family="image_plus_name",
                       signed_answer_margin=5.0, correct=True),
            _make_row(probe_id="p3", identity_id="r1", probe_family="direct_visual",
                       signed_answer_margin=7.0, correct=True),
        ]
        po = [
            _make_row(probe_id="p1", identity_id="t1", probe_family="direct_visual",
                       signed_answer_margin=3.0, correct=False),
            _make_row(probe_id="p2", identity_id="t1", probe_family="image_plus_name",
                       signed_answer_margin=4.0, correct=True),
            _make_row(probe_id="p3", identity_id="r1", probe_family="direct_visual",
                       signed_answer_margin=6.5, correct=True),
        ]
        selection = {
            "target_identities": ["t1"],
            "retain_identities": ["r1"],
            "control_identities": [],
        }

        self._write_jsonl(bl_path, bl)
        self._write_jsonl(po_path, po)
        with sel_path.open("w") as fh:
            json.dump(selection, fh)

        config = PairedAnalysisConfig(
            baseline_results_path=str(bl_path),
            post_results_path=str(po_path),
            selection_manifest_path=str(sel_path),
            output_dir=str(out_dir),
        )
        pa = PairedAnalysis(config)
        pa.load_data()
        results = pa.run_all()

        assert len(results["probe_deltas"]) == 3
        assert "t1" in results["identity_effects"]
        assert "target" in results["group_effects"]
        assert "global_direct_visual" in results["preservation_report"]
        assert "target" in results["route_effects"]

        # Write artifacts
        pa.write_artifacts(results)
        assert (out_dir / "paired_probe_deltas.jsonl").exists()
        assert (out_dir / "identity_effects.json").exists()
        assert (out_dir / "group_effects.json").exists()
        assert (out_dir / "preservation_report.json").exists()
        assert (out_dir / "route_effects_post.json").exists()

    def test_load_data_no_manifest(self, tmp_path: Path) -> None:
        bl_path = tmp_path / "baseline.jsonl"
        po_path = tmp_path / "post.jsonl"
        out_dir = tmp_path / "analysis"

        bl = [_make_row(probe_id="p1")]
        po = [_make_row(probe_id="p1")]
        self._write_jsonl(bl_path, bl)
        self._write_jsonl(po_path, po)

        config = PairedAnalysisConfig(
            baseline_results_path=str(bl_path),
            post_results_path=str(po_path),
            selection_manifest_path=str(tmp_path / "nonexistent.json"),
            output_dir=str(out_dir),
        )
        pa = PairedAnalysis(config)
        pa.load_data()
        deltas = pa.run_probe_deltas()
        assert len(deltas) == 1
        assert deltas[0]["group"] == "untargeted"
