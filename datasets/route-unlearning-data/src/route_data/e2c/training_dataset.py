"""E2C training dataset — PyTorch Dataset for condition-specific training.

Wraps JSONL training records into a PyTorch Dataset that builds supervised
examples via the existing TrainableVLMAdapter interface (Qwen3.5).

Supports both image-conditioned and text-only samples within the same
dataset, matching the HuggingFaceChatAdapter.build_supervised_example
contract.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class E2CTrainingDataset(Dataset):
    """PyTorch Dataset for E2C condition training.

    Each item is a dict produced by
    ``adapter.build_supervised_example(processor, image=..., prompt=..., answer_text=...)``.

    Parameters
    ----------
    records:
        List of training record dicts from dataset_builder.
    processor:
        The Qwen AutoProcessor.
    adapter:
        A TrainableVLMAdapter (e.g. Qwen35Adapter).
    image_loader:
        Callable that loads an image given a path, or None for text-only.
    image_base_dir:
        Base directory for resolving relative image paths.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        processor: Any,
        adapter: Any,
        image_loader: Any = None,
        image_base_dir: str | Path | None = None,
    ):
        self.records = records
        self.processor = processor
        self.adapter = adapter
        self.image_loader = image_loader
        self.image_base_dir = Path(image_base_dir) if image_base_dir else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        prompt = record["prompt"]
        answer = record["answer"]
        image_path = record.get("image_path")

        # Load image if present
        image = None
        if image_path and self.image_loader:
            full_path = image_path
            if self.image_base_dir and not Path(image_path).is_absolute():
                # Avoid double-prefixing if path already starts with base dir
                base_str = str(self.image_base_dir)
                if not image_path.startswith(base_str):
                    full_path = str(self.image_base_dir / image_path)
            image = self.image_loader(full_path)

        example = self.adapter.build_supervised_example(
            self.processor,
            image=image,
            prompt=prompt,
            answer_text=answer,
        )

        # Attach metadata for training loop
        example["_e2c_sample_id"] = record["sample_id"]
        example["_e2c_task"] = record["task"]
        example["_e2c_condition"] = record["condition"]
        example["_e2c_identity_id"] = record["identity_id"]

        return example


def load_records_from_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load training records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
