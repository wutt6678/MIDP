"""Fail-fast configuration validation tests (plan section 6)."""

from __future__ import annotations

import pytest
import yaml

from route_data.config import (
    ConfigError,
    DataConfig,
    RunConfig,
    expand_env,
    load_run_config,
    validate_run_config,
)


def _write_yaml(tmp_path, payload: dict) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload))
    return str(path)


class TestValidateRunConfig:
    def test_defaults_validate(self):
        validate_run_config(RunConfig())

    def test_invalid_model_role_fails(self):
        cfg = RunConfig()
        cfg.run.model_role = "oracle"
        with pytest.raises(ConfigError, match="model_role"):
            validate_run_config(cfg)

    def test_invalid_backend_fails(self):
        cfg = RunConfig()
        cfg.model.backend = "totally_real_backend"
        with pytest.raises(ConfigError, match="backend"):
            validate_run_config(cfg)

    def test_invalid_dtype_fails(self):
        cfg = RunConfig()
        cfg.model.dtype = "int2"
        with pytest.raises(ConfigError, match="dtype"):
            validate_run_config(cfg)

    def test_bad_confidence_band_ordering_fails(self):
        cfg = RunConfig()
        cfg.build.confidence_bands = {"high": 0.5, "medium": 0.9}
        with pytest.raises(ConfigError, match="confidence_bands"):
            validate_run_config(cfg)

    def test_missing_band_key_fails(self):
        cfg = RunConfig()
        cfg.build.confidence_bands = {"high": 0.9}
        with pytest.raises(ConfigError, match="medium"):
            validate_run_config(cfg)


class TestLoadRunConfig:
    def test_unknown_model_key_fails_fast(self, tmp_path):
        path = _write_yaml(tmp_path, {"model": {"backend": "stub", "bogus_key": 1}})
        with pytest.raises(ConfigError, match="bogus_key"):
            load_run_config(path)

    def test_missing_file_fails(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_run_config(tmp_path / "nope.yaml")

    def test_nested_dataclass_fill_and_defaults(self, tmp_path):
        path = _write_yaml(tmp_path, {"model": {"backend": "stub", "generation": {"max_new_tokens": 2}}})
        cfg = load_run_config(path)
        assert cfg.model.backend == "stub"
        assert cfg.model.generation.max_new_tokens == 2
        # Untouched nested fields keep their defaults.
        assert cfg.model.generation.do_sample is False
        assert cfg.run.model_role == "evaluator"

    def test_prompt_paths_resolved_to_existing_files(self, repo_root):
        cfg = load_run_config(repo_root / "configs/runs/golden_stub.yaml")
        from pathlib import Path

        assert Path(cfg.prompts.binary).is_absolute()
        assert Path(cfg.prompts.binary).is_file()
        assert Path(cfg.prompts.route_conflict).is_file()
        assert Path(cfg.thresholds).is_file()

    def test_env_expansion(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MIDP_TEST_ROOT", "/data/somewhere")
        monkeypatch.delenv("MIDP_TEST_MISSING", raising=False)
        assert expand_env("${MIDP_TEST_ROOT}/x") == "/data/somewhere/x"
        # Missing variables expand to an empty string (and fail later at use).
        assert expand_env("${MIDP_TEST_MISSING}/x") == "/x"
        assert expand_env({"a": ["${MIDP_TEST_ROOT}"]}) == {"a": ["/data/somewhere"]}


class TestDataConfig:
    def test_require_root_prefers_root(self):
        cfg = DataConfig(root="/a", extras={"local_root": "/b"})
        assert cfg.require_root().as_posix() == "/a"

    def test_require_root_falls_back_to_extras(self):
        cfg = DataConfig(root=None, extras={"local_root": "/b"})
        assert cfg.require_root().as_posix() == "/b"

    def test_require_root_missing_raises(self):
        with pytest.raises(ConfigError, match="data.root"):
            DataConfig().require_root()
