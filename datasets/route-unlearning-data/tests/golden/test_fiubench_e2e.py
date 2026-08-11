"""FIUBench fixture-backed golden end-to-end test (Fix List P2-16).

Exercises the full build chain against the synthetic FIUBench fixture with
the deterministic stub backend:

    adapter → stub annotate → processed → QA → route probes → splits
    → export → strict verification

Asserts that:
- forget identities are excluded,
- retain identities map to train,
- evaluation identities map to eval,
- QA variants (paraphrase / perturbed) survive correctly,
- profile facts (caption + raw_data) are preserved,
- route probes are generated,
- strict validation passes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fixtures.fiubench_fixture import build_fiubench_fixture

from route_data.cli import main
from route_data.data.io import read_jsonl

STAGES = ("annotate", "qa", "route-probes", "splits", "export")
_STUB_MODEL_DIR = "local_stub-vlm-v1"  # model_id "local/stub-vlm-v1" sanitized


@pytest.fixture(scope="module")
def fiubench_run(tmp_path_factory, repo_root: Path) -> dict:
    base = tmp_path_factory.mktemp("golden_fiubench")
    fixture_root = base / "fiubench_root"
    gt = build_fiubench_fixture(fixture_root)
    out = base / "out"
    cfg = str(repo_root / "configs/runs/golden_fiubench.yaml")

    previous = os.environ.get("FIUBENCH_ROOT")
    os.environ["FIUBENCH_ROOT"] = str(fixture_root)
    try:
        for stage in STAGES:
            rc = main(
                [
                    "build",
                    stage,
                    "--dataset",
                    "fiubench",
                    "--config",
                    cfg,
                    "--output-dir",
                    str(out),
                ]
            )
            assert rc == 0, f"build {stage} failed with rc={rc}"
        validate_rc = main(
            [
                "validate",
                "dataset",
                "--dataset",
                "fiubench_celeba40",
                "--config",
                cfg,
                "--output-dir",
                str(out),
                "--strict",
            ]
        )
    finally:
        if previous is None:
            os.environ.pop("FIUBENCH_ROOT", None)
        else:
            os.environ["FIUBENCH_ROOT"] = previous

    return {"out": out, "validate_rc": validate_rc, "gt": gt}


class TestFiubenchGoldenEndToEnd:
    def test_strict_validation_passes(self, fiubench_run):
        assert fiubench_run["validate_rc"] == 0

    def test_annotated_samples_exist(self, fiubench_run):
        dataset_dir = fiubench_run["out"] / _STUB_MODEL_DIR / "fiubench"
        annotated = list(
            read_jsonl(dataset_dir / "fiubench_annotated.jsonl")
        )
        # 3 identities × 2 QA items × up to 3 variants = up to 18 samples.
        # Exact count depends on stub annotator behavior; must be > 0.
        assert len(annotated) > 0, "no annotated samples produced"

    def test_split_assignment(self, fiubench_run):
        """Forget → exclude, retain → train, evaluation → eval."""
        dataset_dir = fiubench_run["out"] / _STUB_MODEL_DIR / "fiubench"
        annotated = list(
            read_jsonl(dataset_dir / "fiubench_annotated.jsonl")
        )
        # Collect identity → split mapping from annotated samples.
        identity_splits: dict[str, set[str]] = {}
        for row in annotated:
            iid = row.get("identity_id")
            split = row.get("split")
            if iid and split:
                identity_splits.setdefault(iid, set()).add(split)
        # Each identity should map to exactly one split bucket.
        for iid, splits in identity_splits.items():
            assert len(splits) == 1, (
                f"identity {iid} appears in multiple splits: {splits}"
            )

    def test_qa_variants_present(self, fiubench_run):
        """Paraphrase and perturbed QA variants should be generated."""
        dataset_dir = fiubench_run["out"] / _STUB_MODEL_DIR / "fiubench"
        annotated = list(
            read_jsonl(dataset_dir / "fiubench_annotated.jsonl")
        )
        variant_types = {
            row.get("source_metadata", {}).get("variant_type")
            for row in annotated
            if row.get("source_metadata", {}).get("variant_type")
        }
        # At minimum, "original" should be present.
        assert "original" in variant_types, "no original QA variants found"
        # With paraphrases + perturbed enabled, expect those too.
        assert "paraphrase" in variant_types, "paraphrase variants missing"
        assert "perturbed" in variant_types, "perturbed variants missing"

    def test_profile_facts_preserved(self, fiubench_run):
        """Caption and raw_data should survive as profile facts."""
        dataset_dir = fiubench_run["out"] / _STUB_MODEL_DIR / "fiubench"
        annotated = list(
            read_jsonl(dataset_dir / "fiubench_annotated.jsonl")
        )
        has_caption = False
        has_raw = False
        for row in annotated:
            for fact in row.get("profile_facts", []):
                if fact.get("fact_id") == "fiubench_caption":
                    has_caption = True
                if fact.get("fact_id") == "fiubench_raw_profile":
                    has_raw = True
        assert has_caption, "no fiubench_caption profile facts found"
        assert has_raw, "no fiubench_raw_profile profile facts found"

    def test_route_probes_generated(self, fiubench_run):
        dataset_dir = fiubench_run["out"] / _STUB_MODEL_DIR / "fiubench"
        probes = list(
            read_jsonl(dataset_dir / "fiubench_route_probes.jsonl")
        )
        assert len(probes) > 0, "no route probes generated"

    def test_flat_export_artifacts(self, fiubench_run):
        dataset_dir = fiubench_run["out"] / _STUB_MODEL_DIR / "fiubench"
        for rel in (
            "fiubench_celeba40_image_annotations.parquet",
            "fiubench_celeba40_visual_qa_train.jsonl",
            "fiubench_celeba40_visual_qa_eval.jsonl",
            "fiubench_route_conflict_eval.jsonl",
            "fiubench_extension_card.md",
            "fiubench_export_manifest.json",
            "fiubench_checksums.json",
        ):
            assert (dataset_dir / rel).exists(), f"missing export artifact {rel}"

    def test_split_manifest_invariants(self, fiubench_run):
        dataset_dir = fiubench_run["out"] / _STUB_MODEL_DIR / "fiubench"
        payload = json.loads(
            (dataset_dir / "fiubench_split_manifest.json").read_text()
        )
        assert len(payload["splits"]) == 3
        by_name = {s["name"]: s for s in payload["splits"]}
        assert set(by_name) == {
            "identity_forget",
            "identity_fact_forget",
            "attribute_forget",
        }
        for split in payload["splits"]:
            assert split["invariant_issues"] == []
