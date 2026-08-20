"""Validate the Qwen3.5-4B pre-unlearning baseline resolver (P0-2).

This test ensures that the committed baseline manifest uses the canonical
schema expected by ``resolve_preunlearning_baseline()`` and that all
identity fields are populated and consistent.

Run before any unlearning experiment::

    pytest tests/test_baseline_resolver.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def binding():
    """Resolve the qwen35_4b baseline binding."""
    from route_data.eval.post_unlearning_eval import (
        resolve_preunlearning_baseline,
    )

    return resolve_preunlearning_baseline(
        "qwen35_4b",
        protocol_version="baseline_v1",
        project_root=PROJECT_ROOT,
    )


class TestBaselineResolver:
    """P0-2: manifest schema must be compatible with the resolver."""

    def test_resolve_does_not_raise(self, binding):
        """Resolver must find both results and manifest files."""
        assert binding.results_path
        assert binding.manifest_path

    def test_model_key_populated(self, binding):
        assert binding.model_key == "qwen35_4b"

    def test_model_id_populated(self, binding):
        assert binding.model_id == "Qwen/Qwen3.5-4B"

    def test_model_revision_populated(self, binding):
        assert binding.model_revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

    def test_processor_revision_populated(self, binding):
        assert binding.processor_revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

    def test_model_profile_sha256_nonempty(self, binding):
        assert binding.model_profile_sha256, "model_profile_sha256 must not be empty"

    def test_results_sha256_nonempty(self, binding):
        assert binding.results_sha256, "results_sha256 must not be empty"

    def test_manifest_sha256_nonempty(self, binding):
        assert binding.manifest_sha256, "manifest_sha256 must not be empty"


class TestBaselineIdentityValidation:
    """Run validate_baseline_model_identity and assert zero errors."""

    def test_identity_validation_passes(self, binding):
        from route_data.eval.post_unlearning_eval import (
            validate_baseline_model_identity,
        )

        errors = validate_baseline_model_identity(
            binding,
            model_key="qwen35_4b",
            model_id="Qwen/Qwen3.5-4B",
            model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            processor_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            model_profile_sha256=binding.model_profile_sha256,
        )
        assert errors == [], f"Baseline identity validation errors: {errors}"


class TestManifestSchema:
    """Verify the manifest has all required canonical sections."""

    def test_canonical_sections_exist(self):
        import json

        manifest_path = (
            PROJECT_ROOT
            / "outputs/experiments/pre_unlearning/qwen35_4b/baseline_v1/baseline_manifest.json"
        )
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Required top-level sections
        assert "model" in manifest, "Missing 'model' section"
        assert "provenance" in manifest, "Missing 'provenance' section"

        # Required model fields
        model = manifest["model"]
        for key in ("model_key", "id", "revision", "processor_revision", "model_profile_sha256"):
            assert key in model, f"Missing model.{key}"
            assert model[key], f"model.{key} must not be empty"

        # Required provenance fields
        prov = manifest["provenance"]
        for key in ("results_sha256", "manifest_sha256", "route_probe_sha256", "processed_dataset_sha256", "code_commit"):
            assert key in prov, f"Missing provenance.{key}"
            assert prov[key], f"provenance.{key} must not be empty"

    def test_generation_config_present(self):
        """P1-1: generation_config must be part of the frozen protocol."""
        import json

        manifest_path = (
            PROJECT_ROOT
            / "outputs/experiments/pre_unlearning/qwen35_4b/baseline_v1/baseline_manifest.json"
        )
        with open(manifest_path) as f:
            manifest = json.load(f)

        gen_cfg = manifest.get("generation_config", {})
        assert "name_only" in gen_cfg, "Missing generation_config.name_only"
        assert gen_cfg["name_only"]["max_new_tokens"] == 64, (
            "name_only max_new_tokens must be 64"
        )

    def test_name_only_primary_metric_is_token_overlap(self):
        """P1-3: token_overlap is the primary name_only metric."""
        import json

        manifest_path = (
            PROJECT_ROOT
            / "outputs/experiments/pre_unlearning/qwen35_4b/baseline_v1/baseline_manifest.json"
        )
        with open(manifest_path) as f:
            manifest = json.load(f)

        scoring = manifest.get("scoring_config", {})
        assert scoring.get("name_only_primary_metric") == "token_overlap"
        assert scoring.get("name_only_primary_threshold") == 0.5
