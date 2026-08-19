"""Tests for the trainable VLM adapter registry and profile system.

Covers:
- Profile parsing from YAML
- Adapter family registration and model-key dispatch
- Profile-driven adapter creation (adapter.profile is profile)
- Duplicate registration rejection
- Unknown adapter rejection
- Revision validation (exact 40-hex SHA, placeholder rejection)
- Research-profile validation (capability-consistent checks)
- Profile field type validation
- Qwen3.5 adapter basic properties
- NeuronSpec
- compute_profile_sha256
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from route_data.models.trainable.base import ModelFamilyProfile


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_profile(**overrides) -> ModelFamilyProfile:
    """Build a minimal valid ModelFamilyProfile for testing."""
    from route_data.models.trainable.base import ModelFamilyProfile

    defaults = {
        "key": "test_model",
        "model_id": "test/model",
        "revision": "a" * 40,
        "processor_id": "test/model",
        "processor_revision": "b" * 40,
        "adapter_name": "test",
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "candidate_positive": "Yes",
        "candidate_negative": "No",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_scope": "language_attention_only",
        "lora_target_leaf_names": ("q_proj", "v_proj"),
        "lora_scope_regex": r"^model\.layers\.\d+\.self_attn\.",
        "r2mu_candidate_layers": (7, 14),
        "r2mu_n_select_layers": 2,
    }
    defaults.update(overrides)
    return ModelFamilyProfile(**defaults)


# ------------------------------------------------------------------ #
# ModelFamilyProfile
# ------------------------------------------------------------------ #

class TestModelFamilyProfile:
    """Tests for the ModelFamilyProfile frozen dataclass."""

    def test_profile_creation(self):
        profile = _make_profile()
        assert profile.key == "test_model"
        assert profile.lora_rank == 8
        assert profile.supports_prompting is True  # default

    def test_validate_revision_immutable_rejects_main(self):
        profile = _make_profile(revision="main")
        with pytest.raises(ValueError, match="mutable branch"):
            profile.validate_revision_immutable()

    def test_validate_revision_immutable_rejects_empty(self):
        profile = _make_profile(revision="")
        with pytest.raises(ValueError, match="non-empty"):
            profile.validate_revision_immutable()

    def test_validate_revision_immutable_accepts_sha(self):
        sha = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        profile = _make_profile(revision=sha, processor_revision=sha)
        profile.validate_revision_immutable()  # should not raise


# ------------------------------------------------------------------ #
# Revision validation (P0-E)
# ------------------------------------------------------------------ #

class TestRevisionValidation:
    """Tests for strict 40-hex SHA revision validation."""

    def test_placeholder_rejected(self):
        from route_data.models.trainable.registry import (
            validate_research_profile,
        )
        profile = _make_profile(
            revision="<PIN_EXACT_HF_COMMIT_SHA>",
            processor_revision="<PIN_EXACT_HF_COMMIT_SHA>",
        )
        errors = validate_research_profile(profile)
        assert any("placeholder" in e for e in errors)

    def test_short_sha_rejected(self):
        from route_data.models.trainable.registry import (
            validate_research_profile,
        )
        profile = _make_profile(revision="abc123", processor_revision="abc123")
        errors = validate_research_profile(profile)
        assert any("40-char" in e for e in errors)

    def test_non_hex_40char_rejected(self):
        from route_data.models.trainable.registry import (
            validate_research_profile,
        )
        bad = "z" * 40  # non-hex
        profile = _make_profile(revision=bad, processor_revision=bad)
        errors = validate_research_profile(profile)
        # Should be rejected because it's not hex
        assert any("40-char hex" in e for e in errors)

    def test_valid_lowercase_sha_accepted(self):
        from route_data.models.trainable.registry import (
            validate_research_profile,
        )
        sha = "a" * 40
        profile = _make_profile(revision=sha, processor_revision=sha)
        errors = validate_research_profile(profile)
        sha_errors = [e for e in errors if "40-char" in e or "placeholder" in e]
        assert not sha_errors

    def test_valid_uppercase_sha_accepted(self):
        from route_data.models.trainable.registry import (
            validate_research_profile,
        )
        sha = "A" * 40
        profile = _make_profile(revision=sha, processor_revision=sha)
        errors = validate_research_profile(profile)
        sha_errors = [e for e in errors if "40-char" in e or "placeholder" in e]
        assert not sha_errors

    def test_empty_revision_rejected(self):
        from route_data.models.trainable.registry import (
            validate_research_profile,
        )
        profile = _make_profile(revision="", processor_revision="a" * 40)
        errors = validate_research_profile(profile)
        assert any("empty" in e for e in errors)


# ------------------------------------------------------------------ #
# Research-profile validation
# ------------------------------------------------------------------ #

class TestResearchProfileValidation:
    """Tests for capability-consistent profile validation."""

    def test_methods_enabled_but_revision_placeholder(self):
        from route_data.models.trainable.registry import (
            validate_research_profile,
        )
        profile = _make_profile(
            revision="<PIN>",
            processor_revision="a" * 40,
            supports_ga=True,
        )
        errors = validate_research_profile(profile)
        assert any("revision" in e.lower() for e in errors)

    def test_unresolved_lora_scope_regex(self):
        from route_data.models.trainable.registry import (
            validate_research_profile,
        )
        sha = "a" * 40
        profile = _make_profile(
            revision=sha,
            processor_revision=sha,
            lora_scope_regex="<DISCOVER_ON_REAL_MODEL>",
            supports_ga=True,
        )
        errors = validate_research_profile(profile)
        assert any("lora_scope_regex" in e for e in errors)

    def test_r2mu_enabled_but_no_layers(self):
        from route_data.models.trainable.registry import (
            validate_research_profile,
        )
        sha = "a" * 40
        profile = _make_profile(
            revision=sha,
            processor_revision=sha,
            supports_r2mu=True,
            r2mu_candidate_layers=(),
        )
        errors = validate_research_profile(profile)
        assert any("r2mu_candidate_layers" in e for e in errors)

    def test_identical_candidates_rejected(self):
        from route_data.models.trainable.registry import (
            validate_research_profile,
        )
        sha = "a" * 40
        profile = _make_profile(
            revision=sha,
            processor_revision=sha,
            candidate_positive="Yes",
            candidate_negative="Yes",
        )
        errors = validate_research_profile(profile)
        assert any("identical" in e for e in errors)


# ------------------------------------------------------------------ #
# Adapter registry
# ------------------------------------------------------------------ #

class TestAdapterRegistry:
    """Tests for the adapter family registry and model-key dispatch."""

    def test_register_family_and_model_key(self):
        from route_data.models.trainable.base import (
            TrainableVLMAdapter,
        )
        from route_data.models.trainable.registry import (
            _ADAPTER_CACHE,
            _ADAPTER_FAMILIES,
            _MODEL_KEY_TO_FAMILY,
            _ensure_builtin_adapters_loaded,
            create_adapter,
            register_adapter_family,
            register_model_key,
        )

        # Ensure built-in adapters are loaded before saving state
        _ensure_builtin_adapters_loaded()

        # Save state
        saved_families = dict(_ADAPTER_FAMILIES)
        saved_keys = dict(_MODEL_KEY_TO_FAMILY)
        saved_cache = dict(_ADAPTER_CACHE)
        _ADAPTER_FAMILIES.clear()
        _MODEL_KEY_TO_FAMILY.clear()
        _ADAPTER_CACHE.clear()

        try:
            @register_adapter_family("test_fam")
            class _TestAdapter(TrainableVLMAdapter):
                def __init__(self, profile):
                    self._profile = profile
                @property
                def profile(self):
                    return self._profile
                def load_model_processor(self, **kw): raise NotImplementedError
                def build_prefix(self, *a, **kw): raise NotImplementedError
                def build_supervised_example(self, *a, **kw): raise NotImplementedError
                def candidate_token_ids(self, *a): raise NotImplementedError
                def collate(self, batch): raise NotImplementedError
                def append_candidate(self, *a): raise NotImplementedError
                def resolve_lora_targets(self, model): raise NotImplementedError
                def language_layers(self, model): raise NotImplementedError
                def language_hidden_size(self, model): raise NotImplementedError
                def to_eval_backend(self, **kw): raise NotImplementedError

            register_model_key("test_key", "test_fam")

            profile = _make_profile(key="test_key", adapter_name="test_fam")
            adapter = create_adapter("test_key", profile=profile)

            assert adapter.profile is profile
            assert adapter.profile.key == "test_key"
        finally:
            _ADAPTER_FAMILIES.clear()
            _ADAPTER_FAMILIES.update(saved_families)
            _MODEL_KEY_TO_FAMILY.clear()
            _MODEL_KEY_TO_FAMILY.update(saved_keys)
            _ADAPTER_CACHE.clear()
            _ADAPTER_CACHE.update(saved_cache)

    def test_create_adapter_requires_profile(self):
        from route_data.models.trainable.registry import create_adapter
        with pytest.raises(ValueError, match="profile is required"):
            create_adapter("qwen35_9b")

    def test_unknown_model_key_raises(self):
        from route_data.models.trainable.registry import create_adapter
        profile = _make_profile(key="nonexistent_xyz")
        with pytest.raises(KeyError, match="Unknown model key"):
            create_adapter("nonexistent_xyz", profile=profile)

    def test_duplicate_family_registration_raises(self):
        from route_data.models.trainable.registry import (
            _ADAPTER_FAMILIES,
            register_adapter_family,
        )
        saved = dict(_ADAPTER_FAMILIES)
        _ADAPTER_FAMILIES.clear()
        try:
            @register_adapter_family("dup_fam")
            class _A:
                pass

            with pytest.raises(ValueError, match="already registered"):
                @register_adapter_family("dup_fam")
                class _B:
                    pass
        finally:
            _ADAPTER_FAMILIES.clear()
            _ADAPTER_FAMILIES.update(saved)

    def test_adapter_profile_identity(self):
        """Verify adapter.profile is the exact YAML-loaded profile object."""
        import tempfile
        import textwrap

        from route_data.models.trainable.registry import (
            create_adapter,
            load_profile_from_yaml,
        )

        yaml_content = textwrap.dedent("""\
            key: qwen35_9b
            model:
              id: Qwen/Qwen3.5-9B
              revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a
              processor_id: Qwen/Qwen3.5-9B
              processor_revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a
              adapter: qwen35
              trust_remote_code: true
              dtype: bfloat16
              attn_implementation: sdpa
            candidate_protocol:
              positive: "Yes"
              negative: "No"
            lora:
              rank: 8
              alpha: 16
              dropout: 0.05
              scope: language_attention_only
              target_leaf_names: [q_proj, k_proj, v_proj, o_proj]
              scope_regex: "^model\\\\.layers\\\\.\\\\d+\\\\.self_attn\\\\."
            structural:
              r2mu_candidate_layers: [7, 14, 21, 25]
              r2mu_n_select_layers: 4
            compatibility:
              min_transformers: "5.0.0rc0"
              tested_transformers: "5.14.1"
            access:
              requires_hf_auth: false
            supports_prompting: true
            supports_candidate_margin: true
            supports_ga: true
            supports_gd: true
            supports_kl: true
            supports_npo: true
            supports_mmunlearner: true
            supports_manu: true
            supports_r2mu: true
        """)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        ) as f:
            f.write(yaml_content)
            f.flush()
            profile = load_profile_from_yaml(f.name)

        adapter = create_adapter("qwen35_9b", profile=profile)
        assert adapter.profile is profile


# ------------------------------------------------------------------ #
# Profile YAML loading
# ------------------------------------------------------------------ #

class TestProfileYAMLLoading:
    """Tests for loading profiles from YAML."""

    def test_load_qwen35_9b_profile(self, tmp_path):
        from route_data.models.trainable.registry import load_profile_from_yaml

        yaml_content = """\
key: test_yaml_model
model:
  id: TestModel/Test-1B
  revision: abc123def456
  processor_id: TestModel/Test-1B
  processor_revision: abc123def456
  adapter: test_adapter
  trust_remote_code: false
  dtype: bfloat16
  attn_implementation: sdpa
candidate_protocol:
  positive: "Yes"
  negative: "No"
lora:
  rank: 8
  alpha: 16
  dropout: 0.05
  scope: language_attention_only
  target_leaf_names: [q_proj, v_proj]
  scope_regex: "^model\\\\.layers\\\\."
structural:
  r2mu_candidate_layers: [5, 10]
  r2mu_n_select_layers: 2
compatibility:
  min_transformers: "4.50.0"
  tested_transformers: "5.14.1"
access:
  requires_hf_auth: false
"""
        yaml_path = tmp_path / "test_profile.yaml"
        yaml_path.write_text(yaml_content)

        profile = load_profile_from_yaml(yaml_path)
        assert profile.key == "test_yaml_model"
        assert profile.model_id == "TestModel/Test-1B"
        assert profile.revision == "abc123def456"
        assert profile.lora_rank == 8
        assert profile.lora_target_leaf_names == ("q_proj", "v_proj")
        assert profile.r2mu_candidate_layers == (5, 10)
        assert profile.min_transformers_version == "4.50.0"
        assert profile.requires_hf_auth is False

    def test_string_r2mu_layers_rejected(self, tmp_path):
        """r2mu_candidate_layers as a string must fail, not silently split."""
        from route_data.models.trainable.registry import load_profile_from_yaml

        yaml_content = """\
key: bad_model
model:
  id: Test/Bad
  revision: abc
  processor_id: Test/Bad
  processor_revision: abc
  adapter: test
  trust_remote_code: false
  dtype: bfloat16
structural:
  r2mu_candidate_layers: "<DISCOVER_AND_FREEZE>"
  r2mu_n_select_layers: 0
"""
        yaml_path = tmp_path / "bad_profile.yaml"
        yaml_path.write_text(yaml_content)

        with pytest.raises(TypeError, match="r2mu_candidate_layers must be a list"):
            load_profile_from_yaml(yaml_path)

    def test_non_bool_support_flag_rejected(self, tmp_path):
        from route_data.models.trainable.registry import load_profile_from_yaml

        yaml_content = """\
key: bad_model
model:
  id: Test/Bad
  revision: abc
  processor_id: Test/Bad
  processor_revision: abc
  adapter: test
  trust_remote_code: false
  dtype: bfloat16
supports_ga: "yes_please"
"""
        yaml_path = tmp_path / "bad_flag.yaml"
        yaml_path.write_text(yaml_content)

        with pytest.raises(ValueError, match="must be bool"):
            load_profile_from_yaml(yaml_path)


# ------------------------------------------------------------------ #
# Profile SHA256
# ------------------------------------------------------------------ #

class TestProfileSHA256:
    def test_compute_profile_sha256(self, tmp_path):
        from route_data.models.trainable.registry import compute_profile_sha256

        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text("key: test\n")
        sha = compute_profile_sha256(yaml_path)
        assert len(sha) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in sha)

    def test_sha256_deterministic(self, tmp_path):
        from route_data.models.trainable.registry import compute_profile_sha256

        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text("key: test\nmodel:\n  id: x\n")
        sha1 = compute_profile_sha256(yaml_path)
        sha2 = compute_profile_sha256(yaml_path)
        assert sha1 == sha2


# ------------------------------------------------------------------ #
# Qwen3.5 adapter
# ------------------------------------------------------------------ #

class TestQwen35Adapter:
    """Tests for the Qwen3.5 adapter basic properties."""

    def _make_adapter(self):
        from route_data.models.trainable.qwen35 import Qwen35Adapter
        profile = _make_profile(
            key="qwen35_9b",
            model_id="Qwen/Qwen3.5-9B",
            adapter_name="qwen35",
            trust_remote_code=True,
        )
        return Qwen35Adapter(profile)

    def test_profile_properties(self):
        adapter = self._make_adapter()
        p = adapter.profile
        assert p.key == "qwen35_9b"
        assert p.model_id == "Qwen/Qwen3.5-9B"
        assert p.lora_rank == 8

    def test_chat_template_kwargs(self):
        adapter = self._make_adapter()
        assert adapter.chat_template_kwargs() == {"enable_thinking": False}

    def test_image_indexed_keys(self):
        adapter = self._make_adapter()
        keys = adapter.image_indexed_keys()
        assert "pixel_values" in keys
        assert "image_grid_thw" in keys
        assert "image_sizes" in keys

    def test_pad_token_id(self):
        adapter = self._make_adapter()
        assert adapter.pad_token_id(None) == 0

    def test_required_multimodal_keys(self):
        adapter = self._make_adapter()
        assert "pixel_values" in adapter.required_multimodal_keys()

    def test_language_config_standalone(self):
        """Test language_config with a model that has config directly."""
        import types
        adapter = self._make_adapter()
        model = types.SimpleNamespace()
        model.config = types.SimpleNamespace(hidden_size=4096)
        cfg = adapter.language_config(model)
        assert cfg.hidden_size == 4096

    def test_language_config_composite(self):
        """Test language_config with a composite model (text_config)."""
        import types
        adapter = self._make_adapter()
        model = types.SimpleNamespace()
        model.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(hidden_size=3584),
        )
        cfg = adapter.language_config(model)
        assert cfg.hidden_size == 3584


# ------------------------------------------------------------------ #
# NeuronSpec
# ------------------------------------------------------------------ #

class TestNeuronSpec:
    """Tests for the NeuronSpec frozen dataclass."""

    def test_creation(self):
        from route_data.models.trainable.base import NeuronSpec

        spec = NeuronSpec(
            layer_name="model.layers.0.mlp",
            neuron_count=18944,
            input_projection_names=("gate_proj", "up_proj"),
            output_projection_name="down_proj",
            input_axis=0,
            output_axis=1,
        )
        assert spec.layer_name == "model.layers.0.mlp"
        assert spec.neuron_count == 18944
        assert spec.input_axis == 0

    def test_frozen(self):
        from route_data.models.trainable.base import NeuronSpec

        spec = NeuronSpec(
            layer_name="test", neuron_count=100,
            input_projection_names=("a",), output_projection_name="b",
            input_axis=0, output_axis=1,
        )
        with pytest.raises(AttributeError):
            spec.neuron_count = 200  # type: ignore


# ------------------------------------------------------------------ #
# LoRA scope regex (P0 item 10)
# ------------------------------------------------------------------ #

class TestLoraScopeRegex:
    """Tests for the Qwen LoRA scope regex matching."""

    def test_qwen_regex_matches_language_attn(self):
        import re
        regex = r"^model\.layers\.\d+\.self_attn\."
        pattern = re.compile(regex)

        assert pattern.match("model.layers.0.self_attn.q_proj")
        assert pattern.match("model.layers.31.self_attn.o_proj")
        assert not pattern.match("visual.layers.0.self_attn.q_proj")
        assert not pattern.match("model.layers.0.mlp.up_proj")

    def test_regex_selects_correct_modules(self):
        """Build synthetic module tree and verify selection."""
        import torch

        class FakeLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Module()
                self.self_attn.q_proj = torch.nn.Linear(10, 10)
                self.self_attn.k_proj = torch.nn.Linear(10, 10)
                self.self_attn.v_proj = torch.nn.Linear(10, 10)
                self.self_attn.o_proj = torch.nn.Linear(10, 10)
                self.mlp = torch.nn.Module()
                self.mlp.up_proj = torch.nn.Linear(10, 10)

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList(
                    [FakeLayer() for _ in range(4)]
                )

        model = FakeModel()
        from route_data.models.trainable.qwen35 import Qwen35Adapter
        profile = _make_profile(
            key="qwen35_9b",
            adapter_name="qwen35",
            lora_target_leaf_names=("q_proj", "k_proj", "v_proj", "o_proj"),
            lora_scope_regex=r"^model\.layers\.\d+\.self_attn\.",
        )
        adapter = Qwen35Adapter(profile)
        targets = adapter.resolve_lora_targets(model)

        assert len(targets) > 0
        # All targets should be self_attn projections
        for t in targets:
            assert "self_attn" in t
            assert "mlp" not in t
