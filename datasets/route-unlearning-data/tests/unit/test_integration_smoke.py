"""Stub integration smoke test for the full Stage 3 pipeline.

Wires together selection → (stub training) → evaluation → paired analysis
without any GPU. Verifies that all modules interoperate correctly and
produce well-formed artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from route_data.eval.paired_analysis import PairedAnalysis, PairedAnalysisConfig
from route_data.eval.pilot_selection import (
    build_identity_stats,
    select_pilot_identities,
    write_selection_manifest,
)

# --------------------------------------------------------------------------- #
# Fixtures: synthetic data
# --------------------------------------------------------------------------- #

FAMILIES = [
    "direct_visual",
    "image_plus_name",
    "wrong_name",
    "visual_text_conflict",
    "name_only",
]

ATTRIBUTES = [
    "hair_color",
    "eye_color",
    "skin_tone",
    "face_shape",
    "age_group",
    "gender",
    "height",
    "build",
    "clothing_style",
    "accessories",
]


def _make_baseline_results(
    n_identities: int = 12,
    probes_per_identity_per_family: int = 1,
) -> list[dict]:
    """Create synthetic baseline results for *n_identities* identities.

    Each identity gets ``probes_per_identity_per_family`` probes per family,
    with protocol_role='train' for all (to ensure eligibility).
    """
    rows: list[dict] = []
    probe_counter = 0

    for i in range(n_identities):
        iid = f"identity_{i:03d}"
        attr = ATTRIBUTES[i % len(ATTRIBUTES)]
        is_positive = i % 2 == 0

        for fam in FAMILIES:
            for _ in range(probes_per_identity_per_family):
                pid = f"probe_{probe_counter:04d}"
                probe_counter += 1

                if fam == "name_only":
                    row = {
                        "probe_id": pid,
                        "sample_id": f"sample_{pid}",
                        "identity_id": iid,
                        "probe_family": fam,
                        "modality": "text",
                        "question": f"What is {iid}'s {attr}?",
                        "target_attribute": attr,
                        "answer_label": None,
                        "protocol_role": "train",
                        "predicted_label": None,
                        "correct": False,
                        "signed_answer_margin": None,
                        "token_overlap": 0.3,
                        "generated_answer": "unknown",
                    }
                else:
                    label = is_positive
                    margin = 5.0 + (i * 0.1)
                    if fam == "image_plus_name":
                        margin -= 0.5
                    elif fam == "wrong_name":
                        margin -= 0.3
                    elif fam == "visual_text_conflict":
                        margin -= 1.0

                    row = {
                        "probe_id": pid,
                        "sample_id": f"sample_{pid}",
                        "identity_id": iid,
                        "probe_family": fam,
                        "modality": "image+text",
                        "question": f"Does {iid} have {attr}={label}?",
                        "target_attribute": attr,
                        "answer_label": label,
                        "protocol_role": "train",
                        "predicted_label": "Yes" if label else "No",
                        "correct": True,
                        "signed_answer_margin": margin,
                        "token_overlap": None,
                        "logp_yes": 0.1,
                        "logp_no": -0.1,
                        "p_yes": 0.55,
                    }
                rows.append(row)

    return rows


def _make_route_probes(baseline_rows: list[dict]) -> list[dict]:
    """Extract one route probe row per identity from baseline results."""
    seen: set[str] = set()
    probes: list[dict] = []
    for row in baseline_rows:
        iid = row["identity_id"]
        if iid in seen or row["probe_family"] == "name_only":
            continue
        seen.add(iid)
        probes.append({
            "probe_id": f"route_{iid}",
            "identity_id": iid,
            "probe_family": row["probe_family"],
            "target_attribute": row["target_attribute"],
            "answer_label": row["answer_label"],
            "question": row["question"],
            "image_uri": f"images/{iid}.jpg",
        })
    return probes


def _make_post_results(
    baseline_rows: list[dict],
    target_ids: list[str],
    margin_shift: float = -2.0,
) -> list[dict]:
    """Create post-eval results by copying baseline and shifting target margins."""
    import copy
    post = []
    for row in baseline_rows:
        new_row = copy.deepcopy(row)
        if (
            row["identity_id"] in target_ids
            and row["probe_family"] != "name_only"
            and row.get("signed_answer_margin") is not None
        ):
            new_row["signed_answer_margin"] = row["signed_answer_margin"] + margin_shift
            # flip correctness if margin goes negative
            if new_row["signed_answer_margin"] < 0:
                new_row["correct"] = False
                new_row["predicted_label"] = (
                    "No" if row["predicted_label"] == "Yes" else "Yes"
                )
        if row["probe_family"] == "name_only" and row["identity_id"] in target_ids:
            new_row["token_overlap"] = max(0.0, (row.get("token_overlap") or 0.3) - 0.1)
        post.append(new_row)
    return post


# --------------------------------------------------------------------------- #
# Integration test
# --------------------------------------------------------------------------- #

class TestStubIntegrationSmoke:
    """End-to-end smoke test: selection → stub training → eval → analysis."""

    def test_full_pipeline_no_gpu(self, tmp_path: Path) -> None:
        """Run the full pipeline with synthetic data, no GPU required."""
        # -- Step 1: Create synthetic data files --
        baseline_rows = _make_baseline_results(n_identities=12)
        route_probes = _make_route_probes(baseline_rows)

        bl_path = tmp_path / "baseline_results.jsonl"
        rp_path = tmp_path / "route_probes.jsonl"
        sel_path = tmp_path / "selection" / "pilot_identity_selection.json"
        post_path = tmp_path / "post_results.jsonl"
        analysis_dir = tmp_path / "analysis"

        with bl_path.open("w") as fh:
            for r in baseline_rows:
                fh.write(json.dumps(r) + "\n")
        with rp_path.open("w") as fh:
            for r in route_probes:
                fh.write(json.dumps(r) + "\n")

        # -- Step 2: Run identity selection --
        stats = build_identity_stats(bl_path, rp_path)
        assert len(stats) == 12

        selection = select_pilot_identities(
            stats, target_count=2, retain_count=2, control_count=2, seed=17,
        )
        assert len(selection.target_identities) == 2
        assert len(selection.retain_identities) == 2
        assert len(selection.control_identities) == 2

        # No overlap
        all_ids = (
            set(selection.target_identities)
            | set(selection.retain_identities)
            | set(selection.control_identities)
        )
        assert len(all_ids) == 6

        # -- Step 3: Write selection manifest --
        write_selection_manifest(selection, sel_path, code_commit="smoke_test")
        assert sel_path.exists()
        manifest = json.loads(sel_path.read_text())
        assert manifest["target_identities"] == selection.target_identities

        # -- Step 4: Stub "training" — just simulate post-eval results --
        # (In a real run, we would train LoRA and evaluate the checkpoint.)
        post_rows = _make_post_results(
            baseline_rows,
            target_ids=selection.target_identities,
            margin_shift=-2.5,
        )
        with post_path.open("w") as fh:
            for r in post_rows:
                fh.write(json.dumps(r) + "\n")

        # -- Step 5: Run paired analysis --
        config = PairedAnalysisConfig(
            baseline_results_path=str(bl_path),
            post_results_path=str(post_path),
            selection_manifest_path=str(sel_path),
            output_dir=str(analysis_dir),
        )
        pa = PairedAnalysis(config)
        pa.load_data()
        results = pa.run_all()
        pa.write_artifacts(results)

        # -- Step 6: Verify all artifacts exist and are well-formed --
        assert (analysis_dir / "paired_probe_deltas.jsonl").exists()
        assert (analysis_dir / "identity_effects.json").exists()
        assert (analysis_dir / "group_effects.json").exists()
        assert (analysis_dir / "preservation_report.json").exists()
        assert (analysis_dir / "route_effects_post.json").exists()

        # Probe deltas: one per baseline probe
        deltas = results["probe_deltas"]
        assert len(deltas) == len(baseline_rows)

        # Check that target identities have negative delta
        target_deltas = [
            d for d in deltas if d["group"] == "target" and d["family"] != "name_only"
        ]
        assert len(target_deltas) > 0
        mean_target_delta = sum(
            d["delta_signed_margin"] for d in target_deltas
        ) / len(target_deltas)
        assert mean_target_delta < 0  # target should have decreased

        # Check that retain/control identities have zero delta
        retain_deltas = [
            d for d in deltas if d["group"] == "retain" and d["family"] != "name_only"
        ]
        for d in retain_deltas:
            assert d["delta_signed_margin"] == pytest.approx(0.0)

        # Identity effects
        ie = results["identity_effects"]
        for tid in selection.target_identities:
            assert tid in ie
            assert ie[tid]["overall_visual_dM"] is not None
            assert ie[tid]["overall_visual_dM"] < 0

        # Group effects
        ge = results["group_effects"]
        assert ge["target"]["overall"]["mean"] < 0
        assert ge["retain"]["overall"]["mean"] == pytest.approx(0.0)

        # Preservation report
        pr = results["preservation_report"]
        assert "global_direct_visual" in pr
        assert "per_attribute" in pr
        assert "positive_state" in pr
        assert "negative_state" in pr
        assert "retain_group" in pr
        assert "control_group" in pr

        # Route effects
        re = results["route_effects"]
        assert "target" in re
        assert "retain" in re
        assert "control" in re
        # Target delta_conflict should show change
        assert re["target"]["delta_conflict"]["change"] is not None

    def test_selection_determinism(self, tmp_path: Path) -> None:
        """Verify that selection is deterministic across runs."""
        baseline_rows = _make_baseline_results(n_identities=12)
        route_probes = _make_route_probes(baseline_rows)

        bl_path = tmp_path / "baseline.jsonl"
        rp_path = tmp_path / "route.jsonl"
        with bl_path.open("w") as fh:
            for r in baseline_rows:
                fh.write(json.dumps(r) + "\n")
        with rp_path.open("w") as fh:
            for r in route_probes:
                fh.write(json.dumps(r) + "\n")

        stats1 = build_identity_stats(bl_path, rp_path)
        sel1 = select_pilot_identities(stats1, seed=17)

        stats2 = build_identity_stats(bl_path, rp_path)
        sel2 = select_pilot_identities(stats2, seed=17)

        assert sel1.target_identities == sel2.target_identities
        assert sel1.retain_identities == sel2.retain_identities
        assert sel1.control_identities == sel2.control_identities
