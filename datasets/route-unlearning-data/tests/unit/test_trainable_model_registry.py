"""Tests for the trainable VLM adapter registry and profile system.

Covers:
- Profile parsing from YAML
- Adapter registration and creation
- Duplicate registration rejection
- Unknown adapter rejection
- Mutable revision rejection in research mode
- Capability flags
- Qwen3.5 adapter basic properties
"""

from __future__ import annotations

import pytest


class TestModelFamilyProfile:
    """Tests for the ModelFamilyProfile frozen dataclass."""

    def test_profile_creation(self):
        from route_data.models.trainable.base import ModelFamilyProfile

        profile = ModelFamilyProfile(
            key="test_model",
            model_id="test/model",
            revision="abc123",
            processor_id="test/model",
            processor_revision="abc123",
            adapter_name="test",
            trust_remote_code=False,
            dtype="bfloat16",
            attn_implementation="sdpa",
            candidate_positive="Yes",
            candidate_negative="No",
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_scope="language_attention_only",
            lora_target_leaf_names=("q_proj", "v_proj"),
            lora_scope_regex=r"^model\.layers\.\d+\.self_attn\.",
            r2mu_candidate_layers=(7, 14),
            r2mu_n_select_layers=2,
        )
        assert profile.key == "test_model"
        assert profile.lora_rank == 8
        assert profile.supports_prompting is True  # default

    def test_validate_revision_immutable_rejects_main(self):
        from route_data.models.trainable.base import ModelFamilyProfile

        profile = ModelFamilyProfile(
            key="test",
            model_id="test/model",
            revision="main",  # mutable branch
            processor_id="test/model",
            processor_revision="abc123",
            adapter_name="test",
            trust_remote_code=False,
            dtype="bfloat16",
            attn_implementation=None,
            candidate_positive="Yes",
            candidate_negative="No",
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_scope="all",
            lora_target_leaf_names=(),
            lora_scope_regex="",
            r2mu_candidate_layers=(),
            r2mu_n_select_layers=0,
        )
        with pytest.raises(ValueError, match="mutable branch"):
            profile.validate_revision_immutable()

    def test_validate_revision_immutable_rejects_empty(self):
        from route_data.models.trainable.base import ModelFamilyProfile

        profile = ModelFamilyProfile(
            key="test",
            model_id="test/model",
            revision="",  # empty
            processor_id="test/model",
            processor_revision="abc123",
            adapter_name="test",
            trust_remote_code=False,
            dtype="bfloat16",
            attn_implementation=None,
            candidate_positive="Yes",
            candidate_negative="No",
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_scope="all",
            lora_target_leaf_names=(),
            lora_scope_regex="",
            r2mu_candidate_layers=(),
            r2mu_n_select_layers=0,
        )
        with pytest.raises(ValueError, match="non-empty"):
            profile.validate_revision_immutable()

    def test_validate_revision_immutable_accepts_sha(self):
        from route_data.models.trainable.base import ModelFamilyProfile

        sha = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        profile = ModelFamilyProfile(
            key="test",
            model_id="test/model",
            revision=sha,
            processor_id="test/model",
            processor_revision=sha,
            adapter_name="test",
            trust_remote_code=False,
            dtype="bfloat16",
            attn_implementation=None,
            candidate_positive="Yes",
            candidate_negative="No",
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_scope="all",
            lora_target_leaf_names=(),
            lora_scope_regex="",
            r2mu_candidate_layers=(),
            r2mu_n_select_layers=0,
        )
        # Should not raise
        profile.validate_revision_immutable()


class TestAdapterRegistry:
    """Tests for the trainable adapter registry."""

    def test_register_and_create(self):
        from route_data.models.trainable.base import (
            ModelFamilyProfile,
            TrainableVLMAdapter,
        )
        from route_data.models.trainable.registry import (
            _ADAPTER_CACHE,
            _ADAPTER_FACTORIES,
            create_adapter,
            register_adapter,
        )

        # Save state
        saved_factories = dict(_ADAPTER_FACTORIES)
        saved_cache = dict(_ADAPTER_CACHE)
        _ADAPTER_FACTORIES.clear()
        _ADAPTER_CACHE.clear()

        try:
            # Create a minimal adapter for testing
            class _TestAdapter(TrainableVLMAdapter):
                @property
                def profile(self):
                    return ModelFamilyProfile(
                        key="test_reg", model_id="t/m", revision="sha",
                        processor_id="t/m", processor_revision="sha",
                        adapter_name="test", trust_remote_code=False,
                        dtype="bfloat16", attn_implementation=None,
                        candidate_positive="Yes", candidate_negative="No",
                        lora_rank=8, lora_alpha=16, lora_dropout=0.05,
                        lora_scope="all", lora_target_leaf_names=(),
                        lora_scope_regex="", r2mu_candidate_layers=(),
                        r2mu_n_select_layers=0,
                    )
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

            @register_adapter("test_reg")
            def _factory():
                return _TestAdapter()

            adapter = create_adapter("test_reg")
            assert adapter.profile.key == "test_reg"

            # Cached
            adapter2 = create_adapter("test_reg")
            assert adapter is adapter2
        finally:
            # Restore state
            _ADAPTER_FACTORIES.clear()
            _ADAPTER_FACTORIES.update(saved_factories)
            _ADAPTER_CACHE.clear()
            _ADAPTER_CACHE.update(saved_cache)

    def test_duplicate_registration_raises(self):
        from route_data.models.trainable.registry import (
            _ADAPTER_FACTORIES,
            register_adapter,
        )

        # Save state
        saved = dict(_ADAPTER_FACTORIES)
        _ADAPTER_FACTORIES.clear()

        try:
            @register_adapter("dup_test")
            def _factory1():
                pass

            with pytest.raises(ValueError, match="already registered"):
                @register_adapter("dup_test")
                def _factory2():
                    pass
        finally:
            _ADAPTER_FACTORIES.clear()
            _ADAPTER_FACTORIES.update(saved)

    def test_unknown_adapter_raises(self):
        from route_data.models.trainable.registry import create_adapter

        with pytest.raises(KeyError, match="Unknown trainable adapter"):
            create_adapter("nonexistent_model_key_xyz")


class TestQwen35Adapter:
    """Tests for the Qwen3.5 adapter basic properties."""

    def test_profile_properties(self):
        from route_data.models.trainable.qwen35 import Qwen35Adapter

        adapter = Qwen35Adapter()
        p = adapter.profile

        assert p.key == "qwen35_9b"
        assert p.model_id == "Qwen/Qwen3.5-9B"
        assert p.lora_rank == 8
        assert p.lora_alpha == 16
        assert p.candidate_positive == "Yes"
        assert p.candidate_negative == "No"
        assert p.supports_prompting is True
        assert p.supports_manu is True

    def test_chat_template_kwargs(self):
        from route_data.models.trainable.qwen35 import Qwen35Adapter

        adapter = Qwen35Adapter()
        kw = adapter.chat_template_kwargs()
        assert kw == {"enable_thinking": False}

    def test_image_indexed_keys(self):
        from route_data.models.trainable.qwen35 import Qwen35Adapter

        adapter = Qwen35Adapter()
        keys = adapter.image_indexed_keys()
        assert "pixel_values" in keys
        assert "image_grid_thw" in keys
        assert "image_sizes" in keys

    def test_pad_token_id(self):
        from route_data.models.trainable.qwen35 import Qwen35Adapter

        adapter = Qwen35Adapter()
        assert adapter.pad_token_id(None) == 0

    def test_required_multimodal_keys(self):
        from route_data.models.trainable.qwen35 import Qwen35Adapter

        adapter = Qwen35Adapter()
        keys = adapter.required_multimodal_keys()
        assert "pixel_values" in keys


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
