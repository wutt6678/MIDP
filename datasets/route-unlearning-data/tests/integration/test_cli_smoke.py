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


def _make_image(path: Path, size: tuple[int, int] = (224, 224)) -> None:
    """Create a tiny solid-colour PNG for smoke testing."""
    from PIL import Image

    img = Image.new("RGB", size, (50, 100, 150))
    img.save(path)


class TestMultiImageSmoke:
    """P2-19: multi-image smoke test with --image (repeatable) and --image-list."""

    def test_multiple_images_repeatable_flag(self, cfg, tmp_path):
        """--image can be passed multiple times; per-image results are emitted."""
        img_a = tmp_path / "a.png"
        img_b = tmp_path / "b.png"
        _make_image(img_a, (100, 100))
        _make_image(img_b, (200, 200))

        out_dir = tmp_path / "smoke_out"
        rc = main(
            [
                "model",
                "smoke-test",
                "--config",
                cfg,
                "--image",
                str(img_a),
                "--image",
                str(img_b),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc == 0

        artifact = out_dir / "smoke_test" / "golden_stub_model_smoke.json"
        assert artifact.exists()
        payload = json.loads(artifact.read_text())
        assert payload["n_images"] == 2
        assert len(payload["per_image"]) == 2
        # Each per-image entry has its own results
        for entry in payload["per_image"]:
            assert len(entry["smoke_results"]) == 3
            assert entry["image_sha256"] != "placeholder_gray_image"
        # Cross-image variation should be populated
        assert "cross_image_variation" in payload
        for variation in payload["cross_image_variation"].values():
            assert "spread" in variation
            assert variation["spread"] > 0

    def test_image_list_file(self, cfg, tmp_path):
        """--image-list reads paths from a text file."""
        img_a = tmp_path / "x.png"
        img_b = tmp_path / "y.png"
        img_c = tmp_path / "z.png"
        _make_image(img_a, (64, 64))
        _make_image(img_b, (128, 256))
        _make_image(img_c, (32, 32))

        list_file = tmp_path / "images.txt"
        list_file.write_text(
            f"{img_a}\n# comment\n{img_b}\n\n{img_c}\n"
        )

        out_dir = tmp_path / "smoke_out"
        rc = main(
            [
                "model",
                "smoke-test",
                "--config",
                cfg,
                "--image-list",
                str(list_file),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc == 0

        artifact = out_dir / "smoke_test" / "golden_stub_model_smoke.json"
        payload = json.loads(artifact.read_text())
        assert payload["n_images"] == 3

    def test_combined_image_and_image_list(self, cfg, tmp_path):
        """--image and --image-list can be combined."""
        img_a = tmp_path / "a.png"
        img_b = tmp_path / "b.png"
        _make_image(img_a, (100, 100))
        _make_image(img_b, (200, 200))

        list_file = tmp_path / "extra.txt"
        img_c = tmp_path / "c.png"
        _make_image(img_c, (300, 300))
        list_file.write_text(f"{img_c}\n")

        out_dir = tmp_path / "smoke_out"
        rc = main(
            [
                "model",
                "smoke-test",
                "--config",
                cfg,
                "--image",
                str(img_a),
                "--image",
                str(img_b),
                "--image-list",
                str(list_file),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        artifact = out_dir / "smoke_test" / "golden_stub_model_smoke.json"
        payload = json.loads(artifact.read_text())
        assert payload["n_images"] == 3

    def test_backward_compatible_single_image(self, cfg, tmp_path):
        """Single --image still produces backward-compatible top-level fields."""
        img = tmp_path / "single.png"
        _make_image(img, (224, 224))

        out_dir = tmp_path / "smoke_out"
        rc = main(
            [
                "model",
                "smoke-test",
                "--config",
                cfg,
                "--image",
                str(img),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        artifact = out_dir / "smoke_test" / "golden_stub_model_smoke.json"
        payload = json.loads(artifact.read_text())
        assert payload["n_images"] == 1
        # Backward-compatible top-level fields
        assert payload["image_path"] != "placeholder"
        assert payload["image_sha256"] != "placeholder_gray_image"
        assert len(payload["smoke_results"]) == 3

    def test_stub_no_images_uses_placeholder(self, cfg, tmp_path):
        """Stub backend with no --image uses a placeholder (backward compat)."""
        out_dir = tmp_path / "smoke_out"
        rc = main(
            [
                "model",
                "smoke-test",
                "--config",
                cfg,
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        artifact = out_dir / "smoke_test" / "golden_stub_model_smoke.json"
        payload = json.loads(artifact.read_text())
        assert payload["n_images"] == 1
        assert payload["image_path"] == "placeholder"
        assert payload["image_sha256"] == "placeholder_gray_image"

    def test_identical_images_fail_variation_check(self, cfg, tmp_path):
        """Multiple images with identical size produce identical stub scores,
        which should fail the cross-image variation check."""
        img_a = tmp_path / "a.png"
        img_b = tmp_path / "b.png"
        # Same size → same stub hash → same scores
        _make_image(img_a, (224, 224))
        _make_image(img_b, (224, 224))

        out_dir = tmp_path / "smoke_out"
        rc = main(
            [
                "model",
                "smoke-test",
                "--config",
                cfg,
                "--image",
                str(img_a),
                "--image",
                str(img_b),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc == 2  # ConfigError: identical scores across images


class TestSanityCheck:
    """P2-20: image-conditioned sanity cases with --smoke-expected."""

    def test_sanity_check_with_expected_labels(self, cfg, tmp_path):
        """--smoke-expected compares predictions against known labels."""
        img = tmp_path / "test.png"
        _make_image(img, (150, 150))

        # The stub backend produces deterministic scores based on image size.
        # We don't know the exact prediction, but we can provide an expectation
        # and verify the sanity_check section is populated.
        expected_file = tmp_path / "expected.json"
        expected_file.write_text(
            json.dumps(
                [
                    {
                        "image": str(img),
                        "expected": {"Eyeglasses": True, "Smiling": False},
                    }
                ]
            )
        )

        out_dir = tmp_path / "smoke_out"
        rc = main(
            [
                "model",
                "smoke-test",
                "--config",
                cfg,
                "--image",
                str(img),
                "--smoke-expected",
                str(expected_file),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc == 0

        artifact = out_dir / "smoke_test" / "golden_stub_model_smoke.json"
        payload = json.loads(artifact.read_text())
        assert "sanity_check" in payload
        sc = payload["sanity_check"]
        assert sc["n_images_with_expectations"] == 1
        assert sc["total_matches"] + sc["total_mismatches"] == 2
        assert len(sc["results"]) == 1
        assert len(sc["results"][0]["details"]) == 2
        for detail in sc["results"][0]["details"]:
            assert "attribute" in detail
            assert "expected" in detail
            assert "predicted" in detail
            assert "match" in detail

    def test_sanity_check_no_expectations(self, cfg, tmp_path):
        """Without --smoke-expected, sanity_check is empty."""
        img = tmp_path / "test.png"
        _make_image(img, (100, 100))

        out_dir = tmp_path / "smoke_out"
        rc = main(
            [
                "model",
                "smoke-test",
                "--config",
                cfg,
                "--image",
                str(img),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        artifact = out_dir / "smoke_test" / "golden_stub_model_smoke.json"
        payload = json.loads(artifact.read_text())
        assert payload["sanity_check"] == {}
