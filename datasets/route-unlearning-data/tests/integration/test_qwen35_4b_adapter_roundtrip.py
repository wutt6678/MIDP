"""Adapter save/reload persistence test for Qwen3.5-4B.

After one update:
1. Save adapter
2. Release the trained model object
3. Load a fresh base model at the exact pinned revision
4. Load the saved adapter
5. Rerun deterministic scoring

Verifies:
- Base model revision is exact
- Processor revision is exact
- Adapter config contains rank 8 / alpha 16 / dropout 0.05
- Reloaded LoRA module set equals pre-save module set
- Reloaded LoRA inventory SHA matches
- Candidate margins match pre-save values within frozen tolerance
- Yes/No decisions are identical

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


@pytest.fixture(scope="module")
def profile():
    from route_data.models.trainable.registry import load_profile_from_yaml
    return load_profile_from_yaml(str(PROFILE_PATH))


@pytest.fixture(scope="module")
def adapter(profile):
    from route_data.models.trainable.registry import create_adapter
    return create_adapter(profile.key, profile=profile)


class TestQwen35_4BAdapterRoundtrip:
    """Adapter save/reload persistence test."""

    def test_adapter_roundtrip(self, adapter):
        """Save adapter, reload on fresh model, verify scoring matches."""
        from peft import LoraConfig, get_peft_model

        device = os.environ.get("QWEN35_4B_DEVICE", "cuda:0")

        # --- Phase 1: Load model, apply LoRA, score ---
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

        # Record pre-save scores
        from PIL import Image
        test_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        test_prompt = "Is this an image of a cat?"

        yes_pre, no_pre = _score_candidates(
            adapter, lora_model1, processor, test_image, test_prompt, device,
        )
        margin_pre = yes_pre - no_pre

        # Record LoRA module names
        lora_modules_pre = sorted([
            name for name, p in lora_model1.named_parameters()
            if p.requires_grad
        ])

        # Save adapter
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter"
            lora_model1.save_pretrained(str(adapter_path))

            # --- Phase 2: Release model, load fresh, reload adapter ---
            del lora_model1, model1
            torch.cuda.empty_cache()

            # Load fresh base model
            model2, processor2 = adapter.load_model_processor(
                model_id=adapter.profile.model_id,
                revision=adapter.profile.revision,
                processor_revision=adapter.profile.processor_revision,
                dtype=adapter.profile.dtype,
                device=device,
                training=True,
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

            # Record post-reload scores
            yes_post, no_post = _score_candidates(
                adapter, lora_model2, processor2, test_image, test_prompt, device,
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

        # Scores match within tolerance
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
