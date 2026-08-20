"""Phi-4-MM adapter unit tests.

Tests structural and logic aspects of the Phi adapter without requiring
the actual model checkpoint (which needs a separate conda environment).

Covers:
- P0-1: ``build_prefix(image=None)`` for name_only probes
- P0-2: ``language_hidden_size()`` uses correct profile field
- P0-3: NeuronSpec construction with fused up projection
- P0-4: ``collate()`` does not double-unsqueeze image-indexed tensors
- P0-6: Environment validation with ``max_transformers_exclusive``
- P0-7: Fingerprint records ``phi_sdpa_shim_v1``
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_phi_profile(**overrides):
    """Build a minimal ModelFamilyProfile for Phi testing."""
    from route_data.models.trainable.base import ModelFamilyProfile

    defaults = {
        "key": "phi4_mm",
        "model_id": "microsoft/Phi-4-multimodal-instruct",
        "revision": "93f923e1a7727d1c4f446756212d9d3e8fcc5d81",
        "processor_id": "microsoft/Phi-4-multimodal-instruct",
        "processor_revision": "93f923e1a7727d1c4f446756212d9d3e8fcc5d81",
        "adapter_name": "phi4mm",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "attn_implementation": "phi_sdpa_shim_v1",
        "candidate_positive": "Yes",
        "candidate_negative": "No",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_scope": "language_attention_only",
        "lora_target_leaf_names": ("qkv_proj", "o_proj"),
        "lora_scope_regex": r"^model\.layers\.\d+\.(self_attn)\.(qkv_proj|o_proj)$",
        "r2mu_candidate_layers": (8, 16, 24, 30),
        "r2mu_n_select_layers": 4,
        "language_layer_path": "model.layers",
        "language_hidden_size": 3072,
        "intermediate_size": 8192,
        "num_language_layers": 32,
        "supports_manu": False,
        "supports_r2mu": False,
        "min_transformers_version": "4.47.0",
        "max_transformers_version_exclusive": "4.49.0",
        "tested_transformers_version": "4.48.3",
    }
    defaults.update(overrides)
    return ModelFamilyProfile(**defaults)


def _make_phi_adapter():
    """Create a Phi4MMAdapter with a test profile."""
    from route_data.models.trainable.phi4mm import Phi4MMAdapter

    profile = _make_phi_profile()
    return Phi4MMAdapter(profile)


def _make_mock_processor_text_only():
    """Mock processor for text-only (name_only) prefix building."""
    processor = MagicMock()
    tokenizer = MagicMock()

    # apply_chat_template returns a simple text string
    def _apply_template(chat, tokenize=False, add_generation_prompt=True):
        parts = []
        for msg in chat:
            parts.append(f"<|{msg['role']}|>\\n{msg['content']}")
        if add_generation_prompt:
            return "".join(parts) + "<|assistant|>\\n"
        return "".join(parts)

    tokenizer.apply_chat_template = _apply_template
    tokenizer.encode = MagicMock(return_value=[100, 200, 300])
    processor.tokenizer = tokenizer

    # When called without images, return text-only tensors
    def _processor_call(text, images=None, return_tensors="pt"):
        result = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        if images is not None:
            # Simulate Phi image tensors (with batch dim for image-indexed)
            result["input_image_embeds"] = torch.randn(1, 3, 448, 448)
            result["image_attention_mask"] = torch.ones(1, 256, dtype=torch.long)
            result["image_sizes"] = torch.tensor([[448, 448]])
            # InputMode.VISION = 1
            result["input_mode"] = torch.tensor([1], dtype=torch.long)
        else:
            # InputMode.LANGUAGE = 0
            result["input_mode"] = torch.tensor([0], dtype=torch.long)
        return result

    processor.side_effect = _processor_call
    return processor


# ------------------------------------------------------------------ #
# P0-1: name_only (image=None)
# ------------------------------------------------------------------ #

class TestPhiNameOnly:
    """P0-1: build_prefix(image=None) must produce text-only output."""

    def test_build_prefix_no_image_token(self):
        """When image=None, the image token must not appear."""
        adapter = _make_phi_adapter()
        processor = _make_mock_processor_text_only()

        adapter.build_prefix(
            processor, image=None, prompt="What is the capital of France?",
        )

        # Verify processor was called WITHOUT images=
        call_kwargs = processor.call_args
        assert call_kwargs is not None
        # images should not be in the call kwargs
        if call_kwargs.kwargs:
            assert "images" not in call_kwargs.kwargs
        elif len(call_kwargs.args) > 1:
            pytest.fail("processor called with positional images arg")

    def test_build_prefix_no_image_tensors(self):
        """When image=None, no image tensors should be in the result."""
        adapter = _make_phi_adapter()
        processor = _make_mock_processor_text_only()

        prefix = adapter.build_prefix(
            processor, image=None, prompt="What is the capital of France?",
        )

        # No image-indexed keys should be present
        assert "input_image_embeds" not in prefix
        assert "image_attention_mask" not in prefix
        assert "image_sizes" not in prefix
        # Text tensors should be present
        assert "input_ids" in prefix
        assert "attention_mask" in prefix

    def test_build_prefix_preserves_input_mode(self):
        """input_mode must be PRESERVED for the outer model (P0-1)."""
        adapter = _make_phi_adapter()
        processor = _make_mock_processor_text_only()

        # Text-only: input_mode = LANGUAGE (0)
        prefix = adapter.build_prefix(
            processor, image=None, prompt="Hello",
        )
        assert "input_mode" in prefix
        assert prefix["input_mode"].item() == 0  # LANGUAGE

    def test_build_supervised_example_no_image(self):
        """build_supervised_example(image=None) must work for name_only."""
        adapter = _make_phi_adapter()
        processor = _make_mock_processor_text_only()

        example = adapter.build_supervised_example(
            processor, image=None,
            prompt="What is the capital of France?",
            answer_text="Paris",
        )

        assert "input_ids" in example
        assert "labels" in example
        assert "input_image_embeds" not in example


# ------------------------------------------------------------------ #
# P0-2: language_hidden_size uses correct profile field
# ------------------------------------------------------------------ #

class TestPhiLanguageHiddenSize:
    """P0-2: language_hidden_size() must use profile.language_hidden_size."""

    def test_returns_profile_language_hidden_size(self):
        adapter = _make_phi_adapter()
        model = MagicMock()  # not used by Phi's implementation

        result = adapter.language_hidden_size(model)
        assert result == 3072  # from profile.language_hidden_size

    def test_does_not_use_nonexistent_field(self):
        """Verify the adapter doesn't reference profile.hidden_size."""
        adapter = _make_phi_adapter()
        # The profile should NOT have a 'hidden_size' attribute
        assert not hasattr(adapter._profile, "hidden_size")
        # It should have 'language_hidden_size'
        assert hasattr(adapter._profile, "language_hidden_size")
        assert adapter._profile.language_hidden_size == 3072


# ------------------------------------------------------------------ #
# P0-3: NeuronSpec with fused up projection
# ------------------------------------------------------------------ #

class TestPhiNeuronSpec:
    """P0-3: NeuronSpec must support fused up projections."""

    def test_fused_up_projection_fields(self):
        from route_data.models.trainable.base import NeuronSpec

        spec = NeuronSpec(
            layer_name="model.layers.0",
            neuron_count=8192,
            input_projection_names=("gate_up_proj",),
            output_projection_name="down_proj",
            input_axis=1,   # down_proj: intermediate -> hidden
            output_axis=0,  # down_proj: intermediate -> hidden
            is_fused_up=True,
            fused_up_input_axis=0,   # gate_up_proj: hidden -> intermediate
            fused_up_output_axis=1,  # gate_up_proj: hidden -> intermediate
        )
        assert spec.is_fused_up is True
        assert spec.fused_up_input_axis == 0
        assert spec.fused_up_output_axis == 1
        assert spec.input_projection_names == ("gate_up_proj",)

    def test_default_not_fused(self):
        from route_data.models.trainable.base import NeuronSpec

        spec = NeuronSpec(
            layer_name="test", neuron_count=100,
            input_projection_names=("gate_proj", "up_proj"),
            output_projection_name="down_proj",
            input_axis=0, output_axis=1,
        )
        assert spec.is_fused_up is False
        assert spec.fused_up_input_axis == 0
        assert spec.fused_up_output_axis == 0


# ------------------------------------------------------------------ #
# P0-4: collate() does not double-unsqueeze image-indexed tensors
# ------------------------------------------------------------------ #

class TestPhiCollate:
    """P0-4: collate() must not unsqueeze image-indexed tensors."""

    def test_text_tensors_get_batch_dim(self):
        """Text tensors should be unsqueezed from 1D to 2D."""
        adapter = _make_phi_adapter()

        example = {
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        }
        batch = adapter.collate([example])

        assert batch["input_ids"].dim() == 2
        assert batch["input_ids"].shape == (1, 5)
        assert batch["attention_mask"].dim() == 2
        assert batch["attention_mask"].shape == (1, 5)

    def test_image_indexed_tensors_not_unsqueezed(self):
        """Image-indexed tensors must NOT get an extra batch dim."""
        adapter = _make_phi_adapter()

        example = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            # These already have the correct leading dim (num_images=1)
            "input_image_embeds": torch.randn(1, 3, 448, 448),
            "image_attention_mask": torch.ones(1, 256, dtype=torch.long),
            "image_sizes": torch.tensor([[448, 448]]),
        }
        batch = adapter.collate([example])

        # Text tensors: unsqueezed
        assert batch["input_ids"].dim() == 2
        assert batch["input_ids"].shape == (1, 3)

        # Image-indexed tensors: NOT unsqueezed (keep original shape)
        assert batch["input_image_embeds"].shape == (1, 3, 448, 448)
        assert batch["image_attention_mask"].shape == (1, 256)
        assert batch["image_sizes"].shape == (1, 2)

    def test_batch_gt_1_raises(self):
        """Batching > 1 is not yet supported for Phi."""
        adapter = _make_phi_adapter()
        example = {"input_ids": torch.tensor([1, 2, 3])}
        with pytest.raises(NotImplementedError):
            adapter.collate([example, example])


# ------------------------------------------------------------------ #
# P0-6: Environment validation with max_transformers_exclusive
# ------------------------------------------------------------------ #

class TestEnvironmentValidation:
    """P0-6: max_transformers_exclusive must be enforced."""

    def test_profile_has_max_transformers_field(self):
        profile = _make_phi_profile()
        assert hasattr(profile, "max_transformers_version_exclusive")
        assert profile.max_transformers_version_exclusive == "4.49.0"

    def test_validate_environment_compatible(self):
        """validate_environment_compatibility returns errors for mismatch."""
        from route_data.models.trainable.registry import (
            validate_environment_compatibility,
        )

        profile = _make_phi_profile()
        # In the midp-qwen35 environment, transformers is 5.x which is
        # >= 4.49.0 (max_transformers_exclusive). This should produce an
        # error for Phi.
        errors = validate_environment_compatibility(profile)
        # We're running in midp-qwen35 with transformers 5.x
        import transformers
        from packaging.version import Version
        current = Version(transformers.__version__.split(".dev")[0])
        if current >= Version("4.49.0"):
            assert len(errors) > 0
            assert any("transformers" in e and ">=" in e for e in errors)
        else:
            assert errors == []

    def test_yaml_loads_max_transformers(self):
        """Verify the YAML loader reads max_transformers_exclusive."""
        from pathlib import Path

        from route_data.models.trainable.registry import load_profile_from_yaml

        yaml_path = (
            Path(__file__).parent.parent.parent
            / "configs/models/unlearning/phi4_mm.yaml"
        )
        if not yaml_path.exists():
            pytest.skip("phi4_mm.yaml not found")

        profile = load_profile_from_yaml(str(yaml_path))
        assert profile.max_transformers_version_exclusive == "4.49.0"
        assert profile.min_transformers_version == "4.47.0"


# ------------------------------------------------------------------ #
# P0-7: Fingerprint records phi_sdpa_shim_v1
# ------------------------------------------------------------------ #

class TestPhiAttentionFingerprint:
    """P0-7: attn_implementation must be phi_sdpa_shim_v1."""

    def test_profile_attn_field(self):
        adapter = _make_phi_adapter()
        assert adapter.profile.attn_implementation == "phi_sdpa_shim_v1"

    def test_yaml_attn_field(self):
        """Verify the YAML has the correct attn_implementation."""
        from pathlib import Path

        from route_data.models.trainable.registry import load_profile_from_yaml

        yaml_path = (
            Path(__file__).parent.parent.parent
            / "configs/models/unlearning/phi4_mm.yaml"
        )
        if not yaml_path.exists():
            pytest.skip("phi4_mm.yaml not found")

        profile = load_profile_from_yaml(str(yaml_path))
        assert profile.attn_implementation == "phi_sdpa_shim_v1"
        assert profile.attn_implementation != "eager"


# ------------------------------------------------------------------ #
# P1: target_leaf_names restored
# ------------------------------------------------------------------ #

class TestPhiTargetLeafNames:
    """P1: target_leaf_names must be present in the profile."""

    def test_profile_has_target_leaf_names(self):
        adapter = _make_phi_adapter()
        assert adapter.profile.lora_target_leaf_names == ("qkv_proj", "o_proj")

    def test_yaml_has_target_leaf_names(self):
        from pathlib import Path

        from route_data.models.trainable.registry import load_profile_from_yaml

        yaml_path = (
            Path(__file__).parent.parent.parent
            / "configs/models/unlearning/phi4_mm.yaml"
        )
        if not yaml_path.exists():
            pytest.skip("phi4_mm.yaml not found")

        profile = load_profile_from_yaml(str(yaml_path))
        assert "qkv_proj" in profile.lora_target_leaf_names
        assert "o_proj" in profile.lora_target_leaf_names


# ------------------------------------------------------------------ #
# P0-1 (new): _PhiInnerModelWrapper consumes input_mode
# ------------------------------------------------------------------ #

class TestPhiInnerModelWrapperInputMode:
    """P0-1: _PhiInnerModelWrapper must consume input_mode."""

    def test_wrapper_strips_input_mode(self):
        """The wrapper must remove input_mode before calling inner model."""
        from route_data.models.trainable.phi4mm import _PhiInnerModelWrapper

        inner_model = MagicMock()
        lm_head = MagicMock()

        # Mock inner model output
        mock_output = MagicMock()
        mock_output.__getitem__ = lambda self, idx: torch.randn(1, 5, 3072)
        mock_output.past_key_values = None
        mock_output.hidden_states = None
        mock_output.attentions = None
        inner_model.return_value = mock_output

        lm_head.return_value = torch.randn(1, 5, 100)

        wrapper = _PhiInnerModelWrapper(inner_model, lm_head)

        # Call with input_mode
        wrapper(
            input_ids=torch.tensor([[1, 2, 3]]),
            input_mode=torch.tensor([1]),  # VISION
        )

        # Verify inner_model was called WITHOUT input_mode
        call_kwargs = inner_model.call_args.kwargs
        assert "input_mode" not in call_kwargs

    def test_wrapper_sets_audio_projection_vision(self):
        """VISION input_mode should set audio_projection_mode='vision'."""
        from route_data.models.trainable.phi4mm import _PhiInnerModelWrapper

        inner_model = MagicMock()
        lm_head = MagicMock()
        mock_output = MagicMock()
        mock_output.__getitem__ = lambda self, idx: torch.randn(1, 5, 3072)
        mock_output.past_key_values = None
        mock_output.hidden_states = None
        mock_output.attentions = None
        inner_model.return_value = mock_output
        lm_head.return_value = torch.randn(1, 5, 100)

        wrapper = _PhiInnerModelWrapper(inner_model, lm_head)
        wrapper(
            input_ids=torch.tensor([[1, 2, 3]]),
            input_mode=torch.tensor([1]),  # VISION
        )

        call_kwargs = inner_model.call_args.kwargs
        assert call_kwargs.get("audio_projection_mode") == "vision"

    def test_wrapper_sets_audio_projection_language(self):
        """LANGUAGE input_mode should set audio_projection_mode='speech'."""
        from route_data.models.trainable.phi4mm import _PhiInnerModelWrapper

        inner_model = MagicMock()
        lm_head = MagicMock()
        mock_output = MagicMock()
        mock_output.__getitem__ = lambda self, idx: torch.randn(1, 5, 3072)
        mock_output.past_key_values = None
        mock_output.hidden_states = None
        mock_output.attentions = None
        inner_model.return_value = mock_output
        lm_head.return_value = torch.randn(1, 5, 100)

        wrapper = _PhiInnerModelWrapper(inner_model, lm_head)
        wrapper(
            input_ids=torch.tensor([[1, 2, 3]]),
            input_mode=torch.tensor([0]),  # LANGUAGE
        )

        call_kwargs = inner_model.call_args.kwargs
        assert call_kwargs.get("audio_projection_mode") == "speech"

    def test_wrapper_calls_outer_model_set_adapter(self):
        """VISION input_mode should call outer_model.set_lora_adapter('vision')."""
        from route_data.models.trainable.phi4mm import _PhiInnerModelWrapper

        inner_model = MagicMock()
        lm_head = MagicMock()
        outer_model = MagicMock()
        mock_output = MagicMock()
        mock_output.__getitem__ = lambda self, idx: torch.randn(1, 5, 3072)
        mock_output.past_key_values = None
        mock_output.hidden_states = None
        mock_output.attentions = None
        inner_model.return_value = mock_output
        lm_head.return_value = torch.randn(1, 5, 100)

        wrapper = _PhiInnerModelWrapper(inner_model, lm_head, outer_model=outer_model)
        wrapper(
            input_ids=torch.tensor([[1, 2, 3]]),
            input_mode=torch.tensor([1]),  # VISION
        )

        outer_model.set_lora_adapter.assert_called_once_with("vision")

    def test_wrapper_calls_outer_model_unset_adapter(self):
        """LANGUAGE input_mode should call outer_model.unset_lora_adapter()."""
        from route_data.models.trainable.phi4mm import _PhiInnerModelWrapper

        inner_model = MagicMock()
        lm_head = MagicMock()
        outer_model = MagicMock()
        mock_output = MagicMock()
        mock_output.__getitem__ = lambda self, idx: torch.randn(1, 5, 3072)
        mock_output.past_key_values = None
        mock_output.hidden_states = None
        mock_output.attentions = None
        inner_model.return_value = mock_output
        lm_head.return_value = torch.randn(1, 5, 100)

        wrapper = _PhiInnerModelWrapper(inner_model, lm_head, outer_model=outer_model)
        wrapper(
            input_ids=torch.tensor([[1, 2, 3]]),
            input_mode=torch.tensor([0]),  # LANGUAGE
        )

        outer_model.unset_lora_adapter.assert_called_once()
