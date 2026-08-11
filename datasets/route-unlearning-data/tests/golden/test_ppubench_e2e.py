"""PPU-Bench fixture-backed golden end-to-end test (Fix List P2-18).

Exercises the full build chain against the synthetic PPU-Bench fixture:

    adapter → stub annotate → processed → QA → route probes → splits
    → export → verification

Asserts that:
- multi-image expansion works (image + image_002 columns);
- text-only fallback works when no image columns are present;
- options (option_a..option_d) are preserved as ordered list;
- answer_label and answer_text survive;
- subject identity (subject_id / subject) is preserved;
- route probes are generated;
- export artifacts are produced.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fixtures.ppubench_fixture import build_ppubench_fixture

from route_data.cli import main
from route_data.data.io import read_jsonl

STAGES = ("annotate", "qa", "route-probes", "splits", "export")
# ``splits`` may return non-zero on tiny fixtures because the
# attribute_forget split cannot guarantee both positive and negative CelebA
# cases in every bucket with so few samples.  The manifest is still written.
_STAGES_STRICT = {"annotate", "qa", "route-probes", "export"}
_STUB_MODEL_DIR = "local_stub-vlm-v1"


@pytest.fixture(scope="module")
def ppubench_run(tmp_path_factory, repo_root: Path) -> dict:
    base = tmp_path_factory.mktemp("golden_ppubench")
    fixture_root = base / "ppubench_root"
    gt = build_ppubench_fixture(fixture_root)
    out = base / "out"
    cfg = str(repo_root / "configs/runs/golden_ppubench.yaml")

    previous = os.environ.get("PPUBENCH_ROOT")
    os.environ["PPUBENCH_ROOT"] = str(fixture_root)
    stage_rcs: dict[str, int] = {}
    try:
        for stage in STAGES:
            rc = main(
                [
                    "build",
                    stage,
                    "--dataset",
                    "ppubench",
                    "--config",
                    cfg,
                    "--output-dir",
                    str(out),
                ]
            )
            stage_rcs[stage] = rc
            if stage in _STAGES_STRICT:
                assert rc == 0, f"build {stage} failed with rc={rc}"
        validate_rc = main(
            [
                "validate",
                "dataset",
                "--dataset",
                "ppubench_celeba40",
                "--config",
                cfg,
                "--output-dir",
                str(out),
            ]
        )
    finally:
        if previous is None:
            os.environ.pop("PPUBENCH_ROOT", None)
        else:
            os.environ["PPUBENCH_ROOT"] = previous

    return {"out": out, "validate_rc": validate_rc, "stage_rcs": stage_rcs, "gt": gt}


class TestPpubenchGoldenEndToEnd:
    def test_stages_produced_manifests(self, ppubench_run):
        """All build stages ran; split manifest exists even if rc != 0."""
        dataset_dir = ppubench_run["out"] / _STUB_MODEL_DIR / "ppubench"
        assert (dataset_dir / "ppubench_split_manifest.json").exists()
        assert ppubench_run["stage_rcs"].get("splits") in (0, 1)

    def test_annotated_samples_exist(self, ppubench_run):
        dataset_dir = ppubench_run["out"] / _STUB_MODEL_DIR / "ppubench"
        annotated = list(read_jsonl(dataset_dir / "ppubench_annotated.jsonl"))
        assert len(annotated) > 0, "no annotated samples produced"

    def test_multi_image_expansion(self, ppubench_run):
        """Items with multiple image columns produce one record per view."""
        dataset_dir = ppubench_run["out"] / _STUB_MODEL_DIR / "ppubench"
        annotated = list(read_jsonl(dataset_dir / "ppubench_annotated.jsonl"))
        # sub_001 has sample ppu_s001_q1 with image + image_002 (2 views),
        # and ppu_s001_q2 with image (1 view) → at least 3 image records.
        image_records = [r for r in annotated if r.get("image_uri")]
        assert len(image_records) >= 3, (
            f"expected >= 3 image records from multi-image expansion, got {len(image_records)}"
        )

    def test_text_only_fallback(self, ppubench_run):
        """Items with no image columns produce a text_only record."""
        dataset_dir = ppubench_run["out"] / _STUB_MODEL_DIR / "ppubench"
        annotated = list(read_jsonl(dataset_dir / "ppubench_annotated.jsonl"))
        text_only = [r for r in annotated if r.get("modality") == "text_only"]
        assert len(text_only) >= 1, "expected at least one text_only record"

    def test_options_preserved(self, ppubench_run):
        """Classification items should preserve ordered options list."""
        dataset_dir = ppubench_run["out"] / _STUB_MODEL_DIR / "ppubench"
        annotated = list(read_jsonl(dataset_dir / "ppubench_annotated.jsonl"))
        has_options = any(
            isinstance(r.get("options"), list) and len(r["options"]) >= 2
            for r in annotated
        )
        assert has_options, "no records with ordered options list"

    def test_answer_label_and_text(self, ppubench_run):
        """answer_label and answer_text should survive from source rows."""
        dataset_dir = ppubench_run["out"] / _STUB_MODEL_DIR / "ppubench"
        annotated = list(read_jsonl(dataset_dir / "ppubench_annotated.jsonl"))
        has_label = any(r.get("answer_label") is not None for r in annotated)
        has_text = any(r.get("answer_text") is not None for r in annotated)
        assert has_label, "no records with answer_label"
        assert has_text, "no records with answer_text"

    def test_subject_identity_preserved(self, ppubench_run):
        """subject_id and subject name should be preserved."""
        dataset_dir = ppubench_run["out"] / _STUB_MODEL_DIR / "ppubench"
        annotated = list(read_jsonl(dataset_dir / "ppubench_annotated.jsonl"))
        identity_ids = {r.get("identity_id") for r in annotated if r.get("identity_id")}
        assert len(identity_ids) >= 2, "expected multiple subject identities"
        has_name = any(r.get("identity_name") for r in annotated)
        assert has_name, "no records with identity_name"

    def test_route_probes_generated(self, ppubench_run):
        dataset_dir = ppubench_run["out"] / _STUB_MODEL_DIR / "ppubench"
        probes = list(read_jsonl(dataset_dir / "ppubench_route_probes.jsonl"))
        assert len(probes) > 0, "no route probes generated"

    def test_flat_export_artifacts(self, ppubench_run):
        dataset_dir = ppubench_run["out"] / _STUB_MODEL_DIR / "ppubench"
        for rel in (
            "ppubench_celeba40_image_annotations.parquet",
            "ppubench_celeba40_visual_qa_train.jsonl",
            "ppubench_celeba40_visual_qa_eval.jsonl",
            "ppubench_route_conflict_eval.jsonl",
            "ppubench_extension_card.md",
            "ppubench_export_manifest.json",
            "ppubench_checksums.json",
        ):
            assert (dataset_dir / rel).exists(), f"missing export artifact {rel}"
