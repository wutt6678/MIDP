"""CLI smoke tests: every top-level command runs against the stub backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from route_data.cli import main


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
