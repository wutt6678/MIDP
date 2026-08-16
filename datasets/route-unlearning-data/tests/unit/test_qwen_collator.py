"""Tests for the Qwen-compatible collator (P0-3, P1-2).

Verifies:
- batch_size=1 and batch_size=2
- Different text lengths are padded correctly
- pixel_values, image_grid_thw, mm_token_type_ids present
- Correct model-facing dimensions
- No accidental extra dimensions
- Missing required field hard-fails
"""

from __future__ import annotations

import pytest
import torch

from route_data.eval.unlearning_harness import qwen_collate_fn


def _make_sample(
    seq_len: int = 10,
    vocab_size: int = 100,
    num_tiles: int = 4,
    *,
    with_mm_token_type_ids: bool = True,
    with_image_grid_thw: bool = True,
    with_image_sizes: bool = True,
    prefix_len: int = 5,
) -> dict:
    """Create a minimal sample dict mimicking ForgetDataset.__getitem__."""
    sample = {
        "input_ids": torch.arange(1, seq_len + 1),
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "labels": torch.full((seq_len,), -100, dtype=torch.long),
        "pixel_values": torch.randn(num_tiles, 3, 224, 224),
    }
    if with_mm_token_type_ids:
        mm = torch.zeros(seq_len, dtype=torch.long)
        mm[:prefix_len] = 1  # Simulate image tokens
        sample["mm_token_type_ids"] = mm
    if with_image_grid_thw:
        sample["image_grid_thw"] = torch.tensor([[1, 2, 2]])  # [1, 3]
    if with_image_sizes:
        sample["image_sizes"] = torch.tensor([[448, 448]])
    # Metadata
    sample["_prefix_len"] = prefix_len
    sample["_correct_answer_token_ids"] = [50]
    sample["_answer_label"] = True
    sample["_yes_token_ids"] = [50]
    sample["_no_token_ids"] = [60]
    return sample


class TestQwenCollatorBatchSize1:
    """batch_size=1 tests."""

    def test_single_sample_shapes(self) -> None:
        """Single sample produces correct batch dimensions."""
        sample = _make_sample(seq_len=10, num_tiles=4)
        batch = qwen_collate_fn([sample])

        assert batch["input_ids"].shape == (1, 10)
        assert batch["attention_mask"].shape == (1, 10)
        assert batch["labels"].shape == (1, 10)

    def test_pixel_values_present(self) -> None:
        """pixel_values is present with correct shape."""
        sample = _make_sample(num_tiles=4)
        batch = qwen_collate_fn([sample])

        assert "pixel_values" in batch
        # Single image with 4 tiles: [4, 3, 224, 224]
        assert batch["pixel_values"].shape[0] == 4

    def test_image_grid_thw_present(self) -> None:
        """image_grid_thw is present."""
        sample = _make_sample()
        batch = qwen_collate_fn([sample])

        assert "image_grid_thw" in batch
        assert batch["image_grid_thw"].dim() == 2

    def test_mm_token_type_ids_present(self) -> None:
        """mm_token_type_ids is present and correctly shaped."""
        sample = _make_sample(seq_len=10)
        batch = qwen_collate_fn([sample])

        assert "mm_token_type_ids" in batch
        assert batch["mm_token_type_ids"].shape == (1, 10)

    def test_metadata_preserved(self) -> None:
        """Metadata keys are preserved."""
        sample = _make_sample()
        batch = qwen_collate_fn([sample])

        assert "_prefix_len" in batch
        assert "_answer_label" in batch
        assert "_yes_token_ids" in batch
        assert "_no_token_ids" in batch


class TestQwenCollatorBatchSize2:
    """batch_size=2 tests."""

    def test_same_length_shapes(self) -> None:
        """Two samples with same length produce correct shapes."""
        s1 = _make_sample(seq_len=10, num_tiles=4)
        s2 = _make_sample(seq_len=10, num_tiles=4)
        batch = qwen_collate_fn([s1, s2])

        assert batch["input_ids"].shape == (2, 10)
        assert batch["attention_mask"].shape == (2, 10)
        assert batch["labels"].shape == (2, 10)
        assert batch["mm_token_type_ids"].shape == (2, 10)

    def test_different_text_lengths_padded(self) -> None:
        """Different text lengths are right-padded to max."""
        s1 = _make_sample(seq_len=8, num_tiles=4)
        s2 = _make_sample(seq_len=12, num_tiles=4)
        batch = qwen_collate_fn([s1, s2])

        # Should pad to max_len=12
        assert batch["input_ids"].shape == (2, 12)
        assert batch["attention_mask"].shape == (2, 12)
        assert batch["labels"].shape == (2, 12)
        assert batch["mm_token_type_ids"].shape == (2, 12)

        # First sample should have padding in attention_mask
        assert batch["attention_mask"][0, 8:].sum() == 0
        # Second sample should have no padding
        assert batch["attention_mask"][1].sum() == 12

    def test_pixel_values_concatenated(self) -> None:
        """pixel_values from multiple images are concatenated."""
        s1 = _make_sample(num_tiles=4)
        s2 = _make_sample(num_tiles=6)
        batch = qwen_collate_fn([s1, s2])

        # 4 + 6 = 10 tiles total
        assert batch["pixel_values"].shape[0] == 10

    def test_image_grid_thw_stacked(self) -> None:
        """image_grid_thw rows are concatenated."""
        s1 = _make_sample()
        s2 = _make_sample()
        batch = qwen_collate_fn([s1, s2])

        # 2 images, each with [1, 3] → [2, 3]
        assert batch["image_grid_thw"].shape == (2, 3)

    def test_no_extra_dimensions(self) -> None:
        """No accidental extra dimensions in text-aligned tensors."""
        s1 = _make_sample(seq_len=10)
        s2 = _make_sample(seq_len=10)
        batch = qwen_collate_fn([s1, s2])

        assert batch["input_ids"].dim() == 2
        assert batch["attention_mask"].dim() == 2
        assert batch["labels"].dim() == 2
        assert batch["mm_token_type_ids"].dim() == 2


class TestQwenCollatorHardFail:
    """Hard-fail tests for missing required fields."""

    def test_missing_pixel_values_raises(self) -> None:
        """Missing pixel_values raises RuntimeError."""
        sample = _make_sample()
        del sample["pixel_values"]

        with pytest.raises(RuntimeError, match="missing required keys"):
            qwen_collate_fn([sample])

    def test_missing_input_ids_raises(self) -> None:
        """Missing input_ids raises RuntimeError."""
        sample = _make_sample()
        del sample["input_ids"]

        with pytest.raises(RuntimeError, match="missing required keys"):
            qwen_collate_fn([sample])

    def test_missing_labels_raises(self) -> None:
        """Missing labels raises RuntimeError."""
        sample = _make_sample()
        del sample["labels"]

        with pytest.raises(RuntimeError, match="missing required keys"):
            qwen_collate_fn([sample])
