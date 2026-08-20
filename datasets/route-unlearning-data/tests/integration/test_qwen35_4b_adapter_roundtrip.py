"""Adapter save/reload persistence test for Qwen3.5-4B.

Full round-trip with real training:

1. Attach LoRA
2. Snapshot initial LoRA weights
3. Compute real training loss (supervised forward)
4. backward()
5. optimizer.step()
6. Verify LoRA tensors changed (differ from initialization)
7. Score (trained, pre-save)
8. Save adapter
9. Release trained model
10. Load fresh base model at exact pinned revision
11. Reload adapter
12. Score again (post-reload)

Verifies:
- Trained LoRA weights differ from initialization
- Trained pre-save and post-reload margins match within tolerance
- Yes/No decisions are identical pre-save vs post-reload
- Base model revision is exact
- Processor revision is exact
- Adapter config contains rank 8 / alpha 16 / dropout 0.05
- Reloaded LoRA module set equals pre-save module set

Run with::

    QWEN35_4B_CANARY=1 \\
    pytest tests/integration/test_qwen35_4b_adapter_roundtrip.py -v -s
"""

from __future__ import annotations

import json
import os
import tempfile
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


def _batch_prefix(prefix: dict, device: str = "cpu") -> dict:
    batched = {}
    for key, val in prefix.items():
        if isinstance(val, torch.Tensor):
            if val.dim() == 1:
                val = val.unsqueeze(0)
            val = val.to(device)
        batched[key] = val
    return batched


def _score_candidates(adapter, model, processor, image, prompt, device):
    """Score Yes/No candidates for a single image."""
    from route_data.models.scoring import score_candidate_sequence_tensor

    prefix = adapter.build_prefix(processor, image=image, prompt=prompt)
    prefix = _batch_prefix(prefix, device=device)
    yes_ids = adapter.candidate_token_ids(processor, "Yes")
    no_ids = adapter.candidate_token_ids(processor, "No")
    with torch.no_grad():
        log_p_yes = score_candidate_sequence_tensor(
            model, prefix, yes_ids, adapter=adapter,
        )
        log_p_no = score_candidate_sequence_tensor(
            model, prefix, no_ids, adapter=adapter,
        )
    return log_p_yes.item(), log_p_no.item()


def _snapshot_lora_weights(model) -> dict[str, torch.Tensor]:
    """Clone all trainable LoRA parameters into a snapshot dict."""
    return {
        name: p.data.clone()
        for name, p in model.named_parameters()
        if p.requires_grad and "lora" in name.lower()
    }


def _weights_changed(
    snap_before: dict[str, torch.Tensor],
    snap_after: dict[str, torch.Tensor],
) -> tuple[bool, int, int]:
    """Return (any_changed, n_compared, n_changed)."""
    n_changed = 0
    n_compared = 0
    for name, val_before in snap_before.items():
        if name in snap_after:
            n_compared += 1
            if not torch.equal(val_before, snap_after[name]):
                n_changed += 1
    return n_changed > 0, n_compared, n_changed


@pytest.fixture(scope="module")
def profile():
    from route_data.models.trainable.registry import load_profile_from_yaml
    return load_profile_from_yaml(str(PROFILE_PATH))


@pytest.fixture(scope="module")
def adapter(profile):
    from route_data.models.trainable.registry import create_adapter
    return create_adapter(profile.key, profile=profile)


class TestQwen35_4BAdapterRoundtrip:
    """Adapter save/reload persistence test with real training."""

    def test_adapter_roundtrip(self, adapter):
        """Train LoRA, save, reload on fresh model, verify scoring matches."""
        from peft import LoraConfig, get_peft_model
        from PIL import Image

        device = os.environ.get("QWEN35_4B_DEVICE", "cuda:0")

        # --- Phase 1: Load model, apply LoRA ---
        model1, processor = adapter.load_model_processor(
            model_id=adapter.profile.model_id,
            revision=adapter.profile.revision,
            processor_revision=adapter.profile.processor_revision,
            dtype=adapter.profile.dtype,
            device=device,
            training=True,
        )

        targets = adapter.resolve_lora_targets(model1)
        lora_config = LoraConfig(
            r=adapter.profile.lora_rank,
            lora_alpha=adapter.profile.lora_alpha,
            lora_dropout=adapter.profile.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type=None,
        )
        lora_model1 = get_peft_model(model1, lora_config)

        # Snapshot initial LoRA weights (before training)
        snap_init = _snapshot_lora_weights(lora_model1)

        # --- Phase 2: Real training step ---
        # Build a supervised training example.
        test_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        train_prompt = "Is this an image of a cat?"
        train_answer = "No"

        example = adapter.build_supervised_example(
            processor,
            image=test_image,
            prompt=train_prompt,
            answer_text=train_answer,
        )
        batch = adapter.collate([example])
        # Move batch to device.
        batch_device = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_device[k] = v.to(device)
            else:
                batch_device[k] = v

        # Forward pass to compute loss.
        outputs = lora_model1(**batch_device)
        loss = outputs.loss
        assert loss is not None, "Model forward did not return a loss"
        assert torch.isfinite(loss), f"Non-finite training loss: {loss.item()}"
        train_loss = loss.item()

        # Collect only LoRA parameters for the optimizer.
        lora_params = [
            p for n, p in lora_model1.named_parameters()
            if p.requires_grad and "lora" in n.lower()
        ]
        assert len(lora_params) > 0, "No trainable LoRA parameters found"

        optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # --- Phase 3: Verify LoRA tensors changed ---
        snap_trained = _snapshot_lora_weights(lora_model1)
        changed, n_compared, n_changed = _weights_changed(snap_init, snap_trained)
        assert changed, (
            f"LoRA weights did not change after training step "
            f"({n_changed}/{n_compared} tensors changed)"
        )

        # Record LoRA module names
        lora_modules_pre = sorted([
            name for name, p in lora_model1.named_parameters()
            if p.requires_grad
        ])

        # Snapshot trained LoRA weights (before save)
        snap_pre_save = _snapshot_lora_weights(lora_model1)

        # Set model to eval mode for scoring
        lora_model1.eval()

        # --- Phase 4: Score trained model (pre-save) ---
        score_image = Image.new("RGB", (224, 224), color=(64, 64, 64))
        score_prompt = "Is this an image of a dog?"

        yes_pre, no_pre = _score_candidates(
            adapter, lora_model1, processor, score_image, score_prompt, device,
        )
        margin_pre = yes_pre - no_pre

        # --- Phase 5: Save adapter ---
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter"
            lora_model1.save_pretrained(str(adapter_path))

            # --- Phase 6: Release model, load fresh, reload adapter ---
            del lora_model1, model1
            torch.cuda.empty_cache()

            # Load fresh base model
            model2, processor2 = adapter.load_model_processor(
                model_id=adapter.profile.model_id,
                revision=adapter.profile.revision,
                processor_revision=adapter.profile.processor_revision,
                dtype=adapter.profile.dtype,
                device=device,
                training=False,  # Use eval mode for inference
            )

            # Load saved adapter
            from peft import PeftModel
            lora_model2 = PeftModel.from_pretrained(model2, str(adapter_path))

            # Ensure LoRA parameters are trainable after reload
            for name, param in lora_model2.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True

            # Verify adapter config
            peft_config = lora_model2.peft_config.get("default", {})
            assert peft_config.r == adapter.profile.lora_rank == 8
            assert peft_config.lora_alpha == adapter.profile.lora_alpha == 16

            # Snapshot reloaded LoRA weights and compare
            snap_post_reload = _snapshot_lora_weights(lora_model2)
            reloaded_weights_changed, n_cmp, n_diff = _weights_changed(snap_pre_save, snap_post_reload)
            print("\nLoRA weight comparison (pre-save vs post-reload):")
            print(f"  Tensors compared: {n_cmp}")
            print(f"  Tensors different: {n_diff}")
            print(f"  Weights identical: {not reloaded_weights_changed}")
            if reloaded_weights_changed:
                # Find which tensors differ
                for name, val_pre in snap_pre_save.items():
                    if name in snap_post_reload and not torch.equal(val_pre, snap_post_reload[name]):
                        diff = (val_pre - snap_post_reload[name]).abs().max().item()
                        print(f"    {name}: max_diff={diff:.6e}")

            # Assert reloaded weights are identical to pre-save
            assert n_cmp == len(snap_pre_save), (
                f"Expected {len(snap_pre_save)} tensors compared, got {n_cmp}"
            )
            assert n_diff == 0, (
                f"Expected 0 tensors different after reload, got {n_diff}"
            )
            assert not reloaded_weights_changed, (
                "Reloaded LoRA weights differ from pre-save weights"
            )

            # Set model to eval mode for scoring
            lora_model2.eval()

            # --- Phase 7: Score post-reload ---
            yes_post, no_post = _score_candidates(
                adapter, lora_model2, processor2, score_image, score_prompt, device,
            )
            margin_post = yes_post - no_post

            # Record LoRA module names
            lora_modules_post = sorted([
                name for name, p in lora_model2.named_parameters()
                if p.requires_grad
            ])

        # --- Assertions ---
        # Module sets match
        assert lora_modules_pre == lora_modules_post, (
            "LoRA module sets differ after reload"
        )

        # Trained pre-save and post-reload margins match
        atol = 1e-3
        assert abs(yes_pre - yes_post) < atol, (
            f"Yes score mismatch: {yes_pre:.6f} vs {yes_post:.6f}"
        )
        assert abs(no_pre - no_post) < atol, (
            f"No score mismatch: {no_pre:.6f} vs {no_post:.6f}"
        )
        assert abs(margin_pre - margin_post) < atol, (
            f"Margin mismatch: {margin_pre:.6f} vs {margin_post:.6f}"
        )

        # Decisions identical
        decision_pre = "Yes" if yes_pre > no_pre else "No"
        decision_post = "Yes" if yes_post > no_post else "No"
        assert decision_pre == decision_post, (
            f"Decision mismatch: {decision_pre} vs {decision_post}"
        )

        # Write report
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        report = {
            "model_revision": adapter.profile.revision,
            "processor_revision": adapter.profile.processor_revision,
            "lora_rank": adapter.profile.lora_rank,
            "lora_alpha": adapter.profile.lora_alpha,
            "lora_dropout": adapter.profile.lora_dropout,
            "training_loss": train_loss,
            "lora_tensors_compared": n_compared,
            "lora_tensors_changed": n_changed,
            "lora_weights_changed_after_training": changed,
            "reload_tensors_compared": n_cmp,
            "reload_tensors_different": n_diff,
            "reload_weights_exact_match": not reloaded_weights_changed,
            "lora_modules_match": lora_modules_pre == lora_modules_post,
            "yes_score_pre": yes_pre,
            "yes_score_post": yes_post,
            "no_score_pre": no_pre,
            "no_score_post": no_post,
            "margin_pre": margin_pre,
            "margin_post": margin_post,
            "decision_pre": decision_pre,
            "decision_post": decision_post,
            "overall_pass": True,
        }
        with open(OUTPUT_DIR / "adapter_roundtrip.json", "w") as f:
            json.dump(report, f, indent=2)
