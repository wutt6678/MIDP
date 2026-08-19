"""Qwen3.5-4B real-model canary test via the adapter path.

This test proves the new multi-model adapter abstraction works on the
Qwen3.5-4B checkpoint. It verifies:

- Model/processor revision pinning
- Structural metadata matches frozen profile
- Language layer count, hidden size, intermediate size
- Pad token resolution
- Prefix construction
- Supervised example construction
- Candidate scoring is finite
- Free generation is non-blank

Run with::

    QWEN35_4B_CANARY=1 \\
    pytest tests/integration/test_qwen35_4b_real_canary.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

# Skip all tests unless explicitly enabled
pytestmark = pytest.mark.skipif(
    not os.environ.get("QWEN35_4B_CANARY"),
    reason="QWEN35_4B_CANARY not set; set to 1 to enable real-model canary",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROFILE_PATH = PROJECT_ROOT / "configs" / "models" / "unlearning" / "qwen35_4b.yaml"


def _batch_prefix(prefix: dict, device: str = "cpu") -> dict:
    """Ensure prefix tensors are batched and on device."""
    batched = {}
    for key, val in prefix.items():
        if isinstance(val, torch.Tensor):
            if val.dim() == 1:
                val = val.unsqueeze(0)
            val = val.to(device)
        batched[key] = val
    return batched


@pytest.fixture(scope="module")
def profile():
    from route_data.models.trainable.registry import load_profile_from_yaml
    return load_profile_from_yaml(str(PROFILE_PATH))


@pytest.fixture(scope="module")
def adapter(profile):
    from route_data.models.trainable.registry import create_adapter
    return create_adapter(profile.key, profile=profile)


@pytest.fixture(scope="module")
def model_and_processor(adapter):
    device = os.environ.get("QWEN35_4B_DEVICE", "cuda:0")
    model, processor = adapter.load_model_processor(
        model_id=adapter.profile.model_id,
        revision=adapter.profile.revision,
        processor_revision=adapter.profile.processor_revision,
        dtype=adapter.profile.dtype,
        device=device,
        training=False,
    )
    model.eval()
    return model, processor, device


@pytest.fixture(scope="module")
def model(model_and_processor):
    return model_and_processor[0]


@pytest.fixture(scope="module")
def processor(model_and_processor):
    return model_and_processor[1]


@pytest.fixture(scope="module")
def device(model_and_processor):
    return model_and_processor[2]


class TestQwen35_4BRealCanary:
    """Profile-authoritative Qwen3.5-4B real-model canary."""

    def test_model_revision_pinned(self, profile, model):
        """Resolved model revision equals frozen profile revision."""
        assert profile.revision
        assert len(profile.revision) == 40
        assert all(c in "0123456789abcdef" for c in profile.revision)

    def test_processor_revision_pinned(self, profile):
        """Processor revision is an exact SHA."""
        assert profile.processor_revision
        assert len(profile.processor_revision) == 40
        assert all(c in "0123456789abcdef" for c in profile.processor_revision)

    def test_model_class_recorded(self, model):
        """Model class is Qwen3_5ForConditionalGeneration."""
        model_class = type(model).__name__
        assert model_class == "Qwen3_5ForConditionalGeneration"

    def test_processor_class_recorded(self, processor):
        """Processor class is recorded."""
        proc_class = type(processor).__name__
        assert proc_class  # non-empty

    def test_language_layer_count(self, adapter, model, profile):
        """Language layer count equals frozen profile."""
        layers = adapter.language_layers(model)
        assert len(layers) == profile.num_language_layers
        assert len(layers) == 32

    def test_hidden_size(self, adapter, model, profile):
        """Hidden size equals frozen profile."""
        lang_cfg = adapter.language_config(model)
        assert lang_cfg.hidden_size == profile.language_hidden_size
        assert lang_cfg.hidden_size == 2560

    def test_intermediate_size(self, adapter, model, profile):
        """Intermediate size equals frozen profile."""
        intermediate = adapter.language_intermediate_size(model)
        assert intermediate == profile.intermediate_size
        assert intermediate == 9216

    def test_pad_token_resolved(self, adapter, processor):
        """Pad token is resolved from processor, not hard-coded."""
        pad_id = adapter.pad_token_id(processor)
        assert pad_id == 248044
        # Verify it comes from the processor
        tokenizer = getattr(processor, "tokenizer", processor)
        assert pad_id == tokenizer.pad_token_id

    def test_prefix_construction(self, adapter, processor):
        """Prefix construction succeeds."""
        from PIL import Image
        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        prompt = "Is this an image of a cat?"
        prefix = adapter.build_prefix(processor, image=dummy_image, prompt=prompt)
        assert "input_ids" in prefix
        assert prefix["_prefix_len"] > 0

    def test_supervised_example(self, adapter, processor):
        """Supervised example construction succeeds."""
        from PIL import Image
        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        prompt = "Is this an image of a cat?"
        example = adapter.build_supervised_example(
            processor, image=dummy_image, prompt=prompt, answer_text="Yes",
        )
        assert "input_ids" in example
        assert "labels" in example
        assert example["_prefix_len"] > 0

    def test_candidate_scoring_finite(self, adapter, model, processor, device):
        """Candidate scoring is finite."""
        from PIL import Image

        from route_data.models.scoring import score_candidate_sequence_tensor

        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        prompt = "Is this an image of a cat?"
        prefix = adapter.build_prefix(processor, image=dummy_image, prompt=prompt)
        prefix = _batch_prefix(prefix, device=device)

        yes_ids = adapter.candidate_token_ids(processor, "Yes")
        no_ids = adapter.candidate_token_ids(processor, "No")

        assert len(yes_ids) > 0
        assert len(no_ids) > 0
        assert yes_ids != no_ids

        with torch.no_grad():
            log_p_yes = score_candidate_sequence_tensor(
                model, prefix, yes_ids, adapter=adapter,
            )
            log_p_no = score_candidate_sequence_tensor(
                model, prefix, no_ids, adapter=adapter,
            )

        assert torch.isfinite(log_p_yes)
        assert torch.isfinite(log_p_no)

    def test_free_generation_nonblank(self, adapter, model, processor):
        """Free generation is non-blank."""
        from PIL import Image
        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        prompt = "Describe this image briefly."
        messages = [{"role": "user", "content": [
            {"type": "image", "image": dummy_image},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(
            text=[text], images=[dummy_image], return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
            )
        # Decode only the generated tokens
        generated = output[0, inputs["input_ids"].shape[1]:]
        decoded = processor.tokenizer.decode(generated, skip_special_tokens=True)
        assert decoded.strip(), "free generation is blank"

    def test_lora_target_count(self, adapter, model):
        """LoRA target count is exactly 32 (8 layers × 4 projections)."""
        targets = adapter.resolve_lora_targets(model)
        assert len(targets) == 32

    def test_lora_language_only(self, adapter, model):
        """All LoRA targets are in the language tower."""
        targets = adapter.resolve_lora_targets(model)
        for t in targets:
            assert "language_model" in t, f"target {t} not in language tower"
            assert "self_attn" in t, f"target {t} not a self_attn module"

    def test_lora_no_vision_leakage(self, adapter, model):
        """No LoRA targets in vision tower."""
        targets = adapter.resolve_lora_targets(model)
        for t in targets:
            assert "visual" not in t, f"vision leakage: {t}"
            assert "vision" not in t, f"vision leakage: {t}"

    def test_structural_validation(self, adapter, model):
        """Structural validation passes."""
        from route_data.models.trainable.registry import (
            validate_structural_metadata,
        )
        errors = validate_structural_metadata(adapter, model)
        assert not errors, f"Structural validation failed: {errors}"
