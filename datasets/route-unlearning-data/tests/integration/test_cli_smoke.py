"""CLI smoke tests: every top-level command runs against the stub backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fixtures.golden_fixture import build_golden_fixture

from route_data.cli import main
from route_data.data.io import read_jsonl


@pytest.fixture()
def cfg(repo_root: Path) -> str:
    return str(repo_root / "configs/runs/golden_stub.yaml")


@pytest.fixture()
def fairget_data_cfg(repo_root: Path) -> str:
    return str(repo_root / "configs/data/fairget.yaml")


@pytest.fixture()
def fairget_env(monkeypatch, golden_root: Path) -> None:
    monkeypatch.setenv("FAIRGET_ROOT", str(golden_root))


class TestCliSmoke:
    def test_model_inspect(self, cfg):
        assert main(["model", "inspect", "--config", cfg]) == 0

    def test_model_smoke_test(self, cfg):
        assert main(["model", "smoke-test", "--config", cfg]) == 0

    def test_source_inspect(self, cfg, fairget_data_cfg, fairget_env):
        assert (
            main(
                [
                    "source",
                    "inspect",
                    "--dataset",
                    "fairget",
                    "--config",
                    fairget_data_cfg,
                ]
            )
            == 0
        )

    def test_build_annotate_dry_run(self, cfg, fairget_env, tmp_path):
        rc = main(
            [
                "build",
                "annotate",
                "--dataset",
                "fairget",
                "--config",
                cfg,
                "--output-dir",
                str(tmp_path / "out"),
                "--dry-run",
            ]
        )
        assert rc == 0


def _run_annotate(cfg: str, out: Path, *extra: str) -> int:
    return main(
        [
            "build",
            "annotate",
            "--dataset",
            "fairget",
            "--config",
            cfg,
            "--output-dir",
            str(out),
            *extra,
        ]
    )


_STUB_MODEL_DIR = "local_stub-vlm-v1"  # model_id "local/stub-vlm-v1" sanitized


class TestBuildCli:
    def test_annotate_limit_writes_capped_output(self, cfg, fairget_env, tmp_path):
        out = tmp_path / "out"
        assert _run_annotate(cfg, out, "--limit", "3") == 0
        rows = list(read_jsonl(out / _STUB_MODEL_DIR / "fairget" / "fairget_annotated.jsonl"))
        assert len(rows) == 3

    def test_annotate_resume_second_run_succeeds(self, cfg, fairget_env, tmp_path):
        out = tmp_path / "out"
        assert _run_annotate(cfg, out) == 0
        scores = out / _STUB_MODEL_DIR / "fairget" / "fairget_model_scores.jsonl"
        assert scores.exists()
        first_count = len(list(read_jsonl(scores)))
        assert first_count > 0
        assert _run_annotate(cfg, out, "--resume") == 0
        assert len(list(read_jsonl(scores))) == first_count

    def test_custom_output_dir_honored(self, cfg, fairget_env, tmp_path):
        out = tmp_path / "custom" / "location"
        assert _run_annotate(cfg, out) == 0
        assert (out / _STUB_MODEL_DIR / "fairget" / "fairget_annotated.jsonl").exists()

    def test_downstream_dry_runs_after_annotate(self, cfg, fairget_env, tmp_path):
        out = tmp_path / "out"
        assert _run_annotate(cfg, out) == 0
        for stage in ("qa", "route-probes", "splits", "export"):
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
                    "--dry-run",
                ]
            )
            assert rc == 0, f"build {stage} --dry-run returned rc={rc}"

    def test_qa_requires_annotate_prerequisite(self, cfg, fairget_env, tmp_path):
        rc = main(
            [
                "build",
                "qa",
                "--dataset",
                "fairget",
                "--config",
                cfg,
                "--output-dir",
                str(tmp_path / "fresh"),
            ]
        )
        assert rc == 2

    def test_missing_source_layout_fails(self, cfg, monkeypatch, tmp_path):
        empty = tmp_path / "empty_source"
        empty.mkdir()
        monkeypatch.setenv("FAIRGET_ROOT", str(empty))
        rc = _run_annotate(cfg, tmp_path / "out")
        assert rc == 2

    def test_malformed_source_row_fails(self, cfg, monkeypatch, tmp_path):
        root = tmp_path / "bad_source"
        build_golden_fixture(root)
        dataset = root / "data" / "dataset.json"
        payload = json.loads(dataset.read_text())
        payload[0].pop("ID", None)
        dataset.write_text(json.dumps(payload))
        monkeypatch.setenv("FAIRGET_ROOT", str(root))
        rc = _run_annotate(cfg, tmp_path / "out")
        assert rc == 2
