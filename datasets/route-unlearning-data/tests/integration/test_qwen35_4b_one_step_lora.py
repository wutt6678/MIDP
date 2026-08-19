"""One-step LoRA training canary for Qwen3.5-4B.

Verifies:
- Forward loss is finite
- Backward pass completes
- Gradient norm is finite
- At least one trainable gradient is non-zero
- All trainable parameters are LoRA parameters
- No vision/projector/connector parameter has requires_grad=True
- optimizer.step() changes at least one LoRA parameter
- Frozen base parameters remain unchanged

Run with::

    QWEN35_4B_CANARY=1 \\
    pytest tests/integration/test_qwen35_4b_one_step_lora.py -v -s
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.environ.get("QWEN35_4B_CANARY"),
    reason="QWEN35_4B_CANARY not set",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROFILE_PATH = PROJECT_ROOT / "configs" / "models" / "unlearning" / "qwen35_4b.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "qwen35_4b_canary"


@pytest.fixture(scope="module")
def profile():
    from route_data.models.trainable.registry import load_profile_from_yaml
    return load_profile_from_yaml(str(PROFILE_PATH))


@pytest.fixture(scope="module")
def adapter(profile):
    from route_data.models.trainable.registry import create_adapter
    return create_adapter(profile.key, profile=profile)


@pytest.fixture(scope="module")
def training_setup(adapter):
    """Load model in training mode and apply LoRA."""
    from peft import LoraConfig, get_peft_model

    device = os.environ.get("QWEN35_4B_DEVICE", "cuda:0")
    model, processor = adapter.load_model_processor(
        model_id=adapter.profile.model_id,
        revision=adapter.profile.revision,
        processor_revision=adapter.profile.processor_revision,
        dtype=adapter.profile.dtype,
        device=device,
        training=True,
    )

    # Record base parameter fingerprints before LoRA
    base_fingerprints = {}
    for name, param in model.named_parameters():
        base_fingerprints[name] = param.data.clone()

    # Apply LoRA
    targets = adapter.resolve_lora_targets(model)
    lora_config = LoraConfig(
        r=adapter.profile.lora_rank,
        lora_alpha=adapter.profile.lora_alpha,
        lora_dropout=adapter.profile.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type=None,
    )
    lora_model = get_peft_model(model, lora_config)

    # Record LoRA parameter fingerprints
    lora_fingerprints = {}
    for name, param in lora_model.named_parameters():
        if param.requires_grad:
            lora_fingerprints[name] = param.data.clone()

    return {
        "model": lora_model,
        "processor": processor,
        "adapter": adapter,
        "device": device,
        "base_fingerprints": base_fingerprints,
        "lora_fingerprints": lora_fingerprints,
        "targets": targets,
    }


class TestQwen35_4BOneStepLoRA:
    """One-step LoRA training canary."""

    def test_trainable_params_are_lora(self, training_setup):
        """All trainable parameters are LoRA parameters."""
        model = training_setup["model"]
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert "lora" in name.lower(), (
                    f"Trainable param {name} is not a LoRA parameter"
                )

    def test_no_vision_trainable(self, training_setup):
        """No vision parameter has requires_grad=True."""
        model = training_setup["model"]
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert "visual" not in name.lower(), (
                    f"Vision param {name} has requires_grad=True"
                )

    def test_no_projector_trainable(self, training_setup):
        """No projector/connector parameter has requires_grad=True."""
        model = training_setup["model"]
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert "projector" not in name.lower(), (
                    f"Projector param {name} has requires_grad=True"
                )
                assert "connector" not in name.lower(), (
                    f"Connector param {name} has requires_grad=True"
                )

    def test_one_step_backward(self, training_setup):
        """One real multimodal batch: forward, backward, optimizer step."""
        from PIL import Image

        model = training_setup["model"]
        processor = training_setup["processor"]
        adapter = training_setup["adapter"]
        device = training_setup["device"]

        # Build a supervised example
        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        prompt = "Is this an image of a cat?"
        example = adapter.build_supervised_example(
            processor, image=dummy_image, prompt=prompt, answer_text="Yes",
        )

        # Batch it
        batch = {}
        for key, val in example.items():
            if isinstance(val, torch.Tensor):
                if val.dim() == 1:
                    val = val.unsqueeze(0)
                batch[key] = val.to(device)
            else:
                batch[key] = val

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert loss.item() > 0, f"Loss is zero or negative: {loss}"

        # Backward pass
        loss.backward()

        # Check gradient norm is finite
        total_grad_norm = 0.0
        n_nonzero_grads = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                assert not math.isnan(grad_norm), (
                    f"Gradient NaN for {name}"
                )
                total_grad_norm += grad_norm ** 2
                if param.grad.abs().sum().item() > 0:
                    n_nonzero_grads += 1

        total_grad_norm = total_grad_norm ** 0.5
        assert total_grad_norm > 0, "All gradients are zero"
        assert n_nonzero_grads > 0, "No non-zero gradients"

        # Record pre-step LoRA fingerprints
        pre_step = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                pre_step[name] = param.data.clone()

        # Optimizer step
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
        )
        optimizer.step()

        # Check that at least one LoRA parameter changed
        n_changed_lora = 0
        for name, param in model.named_parameters():
            if name in pre_step and not torch.equal(param.data, pre_step[name]):
                n_changed_lora += 1

        assert n_changed_lora > 0, "No LoRA parameters changed after optimizer step"

        # Check that frozen base parameters remain unchanged
        base_fps = training_setup["base_fingerprints"]
        n_changed_frozen = 0
        for name, param in model.named_parameters():
            if name in base_fps and not param.requires_grad and not torch.equal(param.data, base_fps[name]):
                n_changed_frozen += 1

        assert n_changed_frozen == 0, (
            f"{n_changed_frozen} frozen base parameters changed"
        )

        # Write report
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        report = {
            "loss": loss.item(),
            "gradient_norm": total_grad_norm,
            "n_trainable_parameters": sum(
                1 for p in model.parameters() if p.requires_grad
            ),
            "n_trainable_modules": len({
                name.rsplit(".", 1)[0]
                for name, p in model.named_parameters()
                if p.requires_grad
            }),
            "n_changed_lora_tensors": n_changed_lora,
            "n_changed_frozen_base_tensors": n_changed_frozen,
            "n_nonzero_gradients": n_nonzero_grads,
            "overall_pass": True,
        }
        with open(OUTPUT_DIR / "backward_step.json", "w") as f:
            json.dump(report, f, indent=2)
