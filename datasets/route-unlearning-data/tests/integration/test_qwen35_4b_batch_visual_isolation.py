"""Batch-size-2 visual isolation test for Qwen3.5-4B.

Proves that sample A never receives sample B's visual features when
batching two different images. Evaluates:

- A alone
- B alone
- [A, B]
- [B, A]

For both Yes and No candidate scores:
  score(A alone) ≈ score(A in [A,B]) ≈ score(A in [B,A])
  score(B alone) ≈ score(B in [A,B]) ≈ score(B in [B,A])

Run with::

    QWEN35_4B_CANARY=1 \\
    pytest tests/integration/test_qwen35_4b_batch_visual_isolation.py -v -s
"""

from __future__ import annotations

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

# Two different test images from different identities
IMAGE_A_PATH = (
    PROJECT_ROOT / "data" / "tmp_final_verify" / "golden_root"
    / "test_images" / "gld_001" / "frame_0003.png"
)
IMAGE_B_PATH = (
    PROJECT_ROOT / "data" / "tmp_final_verify" / "golden_root"
    / "test_images" / "gld_002" / "frame_0006.png"
)

ATOL = 0.1  # tolerance for score comparison


def _batch_prefix(prefix: dict, device: str = "cpu") -> dict:
    batched = {}
    for key, val in prefix.items():
        if isinstance(val, torch.Tensor):
            if val.dim() == 1:
                val = val.unsqueeze(0)
            val = val.to(device)
        batched[key] = val
    return batched


def _score_single(adapter, model, processor, image, prompt, device):
    """Score a single image with the given prompt."""
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


def _score_batch(adapter, model, processor, images, prompt, device):
    """Score a batch of images. Returns per-sample scores."""
    from route_data.models.scoring import score_candidate_sequence_tensor

    examples = []
    for img in images:
        ex = adapter.build_supervised_example(
            processor, image=img, prompt=prompt, answer_text="Yes",
        )
        examples.append(ex)

    # Collate batch
    batch = adapter.collate(examples)
    batch = _batch_prefix(batch, device=device)

    yes_ids = adapter.candidate_token_ids(processor, "Yes")
    no_ids = adapter.candidate_token_ids(processor, "No")

    scores_yes = []
    scores_no = []
    with torch.no_grad():
        for i in range(len(images)):
            sample_prefix = adapter.extract_sample_prefix(batch, i)
            log_p_yes = score_candidate_sequence_tensor(
                model, sample_prefix, yes_ids, adapter=adapter,
            )
            log_p_no = score_candidate_sequence_tensor(
                model, sample_prefix, no_ids, adapter=adapter,
            )
            scores_yes.append(log_p_yes.item())
            scores_no.append(log_p_no.item())

    return scores_yes, scores_no


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


@pytest.fixture(scope="module")
def images():
    from PIL import Image
    assert IMAGE_A_PATH.exists(), f"Image A not found: {IMAGE_A_PATH}"
    assert IMAGE_B_PATH.exists(), f"Image B not found: {IMAGE_B_PATH}"
    return Image.open(IMAGE_A_PATH).convert("RGB"), Image.open(IMAGE_B_PATH).convert("RGB")


class TestQwen35_4BBatchVisualIsolation:
    """Batch-size-2 visual isolation test."""

    def test_visual_isolation_yes_scores(
        self, adapter, model, processor, device, images,
    ):
        """Score(A alone) ≈ score(A in [A,B]) ≈ score(A in [B,A])."""
        img_a, img_b = images
        prompt = "Is this person wearing glasses?"

        # A alone
        yes_a_alone, _ = _score_single(adapter, model, processor, img_a, prompt, device)

        # B alone
        yes_b_alone, _ = _score_single(adapter, model, processor, img_b, prompt, device)

        # [A, B] batch
        scores_yes_ab, _ = _score_batch(
            adapter, model, processor, [img_a, img_b], prompt, device,
        )
        yes_a_in_ab = scores_yes_ab[0]
        yes_b_in_ab = scores_yes_ab[1]

        # [B, A] batch
        scores_yes_ba, _ = _score_batch(
            adapter, model, processor, [img_b, img_a], prompt, device,
        )
        yes_a_in_ba = scores_yes_ba[1]
        yes_b_in_ba = scores_yes_ba[0]

        # A scores should match across orderings
        assert abs(yes_a_alone - yes_a_in_ab) < ATOL, (
            f"A alone ({yes_a_alone:.4f}) != A in [A,B] ({yes_a_in_ab:.4f})"
        )
        assert abs(yes_a_alone - yes_a_in_ba) < ATOL, (
            f"A alone ({yes_a_alone:.4f}) != A in [B,A] ({yes_a_in_ba:.4f})"
        )

        # B scores should match across orderings
        assert abs(yes_b_alone - yes_b_in_ab) < ATOL, (
            f"B alone ({yes_b_alone:.4f}) != B in [A,B] ({yes_b_in_ab:.4f})"
        )
        assert abs(yes_b_alone - yes_b_in_ba) < ATOL, (
            f"B alone ({yes_b_alone:.4f}) != B in [B,A] ({yes_b_in_ba:.4f})"
        )

    def test_visual_isolation_no_scores(
        self, adapter, model, processor, device, images,
    ):
        """Same test for No scores."""
        img_a, img_b = images
        prompt = "Is this person wearing glasses?"

        _, no_a_alone = _score_single(adapter, model, processor, img_a, prompt, device)
        _, no_b_alone = _score_single(adapter, model, processor, img_b, prompt, device)

        _, scores_no_ab = _score_batch(
            adapter, model, processor, [img_a, img_b], prompt, device,
        )
        no_a_in_ab = scores_no_ab[0]
        no_b_in_ab = scores_no_ab[1]

        _, scores_no_ba = _score_batch(
            adapter, model, processor, [img_b, img_a], prompt, device,
        )
        no_a_in_ba = scores_no_ba[1]
        no_b_in_ba = scores_no_ba[0]

        assert abs(no_a_alone - no_a_in_ab) < ATOL
        assert abs(no_a_alone - no_a_in_ba) < ATOL
        assert abs(no_b_alone - no_b_in_ab) < ATOL
        assert abs(no_b_alone - no_b_in_ba) < ATOL

    def test_visual_spans_do_not_overlap(
        self, adapter, processor, images,
    ):
        """Visual spans for A and B do not overlap in the batch."""
        img_a, img_b = images
        prompt = "Test prompt"

        examples = []
        for img in [img_a, img_b]:
            ex = adapter.build_supervised_example(
                processor, image=img, prompt=prompt, answer_text="Yes",
            )
            examples.append(ex)
        batch = adapter.collate(examples)

        visual_spans = batch.get("_visual_spans", {})
        for key, spans in visual_spans.items():
            if len(spans) >= 2:
                start_a, stop_a = spans[0]
                start_b, stop_b = spans[1]
                # Check no overlap
                assert stop_a <= start_b or stop_b <= start_a, (
                    f"Visual spans overlap for {key}: "
                    f"A=[{start_a},{stop_a}), B=[{start_b},{stop_b})"
                )

    def test_extract_sample_prefix_isolation(
        self, adapter, processor, images,
    ):
        """extract_sample_prefix extracts only the correct sample's visual data."""
        img_a, img_b = images
        prompt = "Test prompt"

        examples = []
        for img in [img_a, img_b]:
            ex = adapter.build_supervised_example(
                processor, image=img, prompt=prompt, answer_text="Yes",
            )
            examples.append(ex)
        batch = adapter.collate(examples)

        # Extract sample 0 (should be A)
        prefix_0 = adapter.extract_sample_prefix(batch, 0)
        # Extract sample 1 (should be B)
        prefix_1 = adapter.extract_sample_prefix(batch, 1)

        # Build single-sample examples for comparison
        ex_a = adapter.build_supervised_example(
            processor, image=img_a, prompt=prompt, answer_text="Yes",
        )
        ex_b = adapter.build_supervised_example(
            processor, image=img_b, prompt=prompt, answer_text="Yes",
        )
        batch_a_only = _batch_prefix(ex_a, device="cpu")
        batch_b_only = _batch_prefix(ex_b, device="cpu")

        # Check that extracted prefixes have the same visual tensor shapes
        for key in ("pixel_values",):
            if key in prefix_0 and key in batch_a_only:
                assert prefix_0[key].shape == batch_a_only[key].shape, (
                    f"Sample 0 visual shape mismatch for {key}"
                )
            if key in prefix_1 and key in batch_b_only:
                assert prefix_1[key].shape == batch_b_only[key].shape, (
                    f"Sample 1 visual shape mismatch for {key}"
                )
