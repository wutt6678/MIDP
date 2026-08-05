"""Golden end-to-end build: annotate → qa → probes → splits → export → validate."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from fixtures.golden_fixture import build_golden_fixture
from route_data.cli import main
from route_data.data.io import read_jsonl

STAGES = ("annotate", "qa", "route-probes", "splits", "export")


@pytest.fixture(scope="module")
def golden_run(tmp_path_factory, repo_root: Path) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("golden")
    fixture_root = base / "golden_root"
    build_golden_fixture(fixture_root)
    out = base / "out"
    cfg = str(repo_root / "configs/runs/golden_stub.yaml")

    previous = os.environ.get("FAIRGET_ROOT")
    os.environ["FAIRGET_ROOT"] = str(fixture_root)
    try:
        for stage in STAGES:
            rc = main(
                [
                    "build",
                    stage,
                    "--dataset",
                    "fairget",
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
                "fairget_celeba40",
                "--config",
                cfg,
                "--output-dir",
                str(out),
                "--strict",
            ]
        )
    finally:
        if previous is None:
            os.environ.pop("FAIRGET_ROOT", None)
        else:
            os.environ["FAIRGET_ROOT"] = previous

    return {"out": out, "validate_rc": validate_rc}


def _accepted_labels(rows: list[dict], namespace: str | None = None) -> int:
    count = 0
    for row in rows:
        for key, obs in (row.get("visual_attributes") or {}).items():
            if namespace and not key.startswith(namespace):
                continue
            if obs.get("label") is not None:
                count += 1
    return count


class TestGoldenEndToEnd:
    def test_strict_validation_passes(self, golden_run):
        assert golden_run["validate_rc"] == 0

    def test_annotated_samples_and_labels(self, golden_run):
        rows = list(
            read_jsonl(golden_run["out"] / "fairget" / "fairget_annotated.jsonl")
        )
        assert len(rows) == 6
        # 48 accepted labels overall (24 model celeba40 + 24 source fairface).
        assert _accepted_labels(rows) == 48
        assert (
            _accepted_labels(rows, namespace="extended_attributes.celeba40.")
            == 24
        )

    def test_visual_qa_rows(self, golden_run):
        dataset_dir = golden_run["out"] / "fairget"
        train = list(read_jsonl(dataset_dir / "fairget_visual_qa_train.jsonl"))
        eval_rows = list(read_jsonl(dataset_dir / "fairget_visual_qa_eval.jsonl"))
        assert len(train) == 24
        assert len(eval_rows) == 24

    def test_route_probes(self, golden_run):
        probes = list(
            read_jsonl(
                golden_run["out"] / "fairget" / "fairget_route_probes.jsonl"
            )
        )
        assert len(probes) == 18  # 3 identities x 6 probe families

    def test_flat_export_artifacts(self, golden_run):
        dataset_dir = golden_run["out"] / "fairget"
        for rel in (
            "fairget_celeba40_image_annotations.parquet",
            "fairget_celeba40_visual_qa_train.jsonl",
            "fairget_celeba40_visual_qa_eval.jsonl",
            "fairget_route_conflict_eval.jsonl",
            "fairget_extension_card.md",
            "fairget_export_manifest.json",
        ):
            assert (dataset_dir / rel).exists(), f"missing export artifact {rel}"
        splits = sorted(
            (dataset_dir / "fairget_unlearning_splits").glob("*.json")
        )
        assert len(splits) == 3

    def test_split_manifest_invariants_clean(self, golden_run):
        payload = json.loads(
            (
                golden_run["out"] / "fairget" / "fairget_split_manifest.json"
            ).read_text()
        )
        assert len(payload["splits"]) == 3
        for split in payload["splits"]:
            assert split["invariant_issues"] == []
