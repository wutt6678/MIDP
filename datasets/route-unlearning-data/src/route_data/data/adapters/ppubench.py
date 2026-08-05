"""PPU-Bench adapter (coding plan section 15).

PPU-Bench contains real public figures across several unlearning settings and
is released as a Hugging Face dataset with many configurations. It must be
accessed through this adapter (``datasets.load_dataset``) rather than copied
into a custom layout, and the configuration name is a required setting
(plan section 15.1). Every flattened row records its source configuration.

Public figures may show different attributes across photographs, so images are
annotated independently downstream and never collapsed to one identity-level
value (plan section 15.2).
"""

from __future__ import annotations

from typing import Any

from ..schemas import CanonicalSample
from .base import AdapterError, BenchmarkAdapter, register_adapter


@register_adapter("ppubench")
class PpubenchAdapter(BenchmarkAdapter):
    required_fields = ("source_sample_id", "identity_id")

    def __init__(self, config):
        super().__init__(config)
        if not config.extras.get("hf_config") and not config.extras.get("hf_config_name"):
            raise AdapterError(
                "[ppubench] extras.hf_config (the PPU-Bench HF configuration name) "
                "is required; PPU-Bench exposes many configurations and must not be "
                "flattened without recording the source."
            )

    def _default_field_map(self) -> dict[str, str]:
        return {
            "source_sample_id": "id",
            "identity_id": "subject_id",
            "identity_name": "subject_name",
            "image_uri": "image",
            "split": "source_split",
            "question": "question",
            "answer_text": "answer",
            "options": "options",
            "task_type": "task_type",
            "modality": "modality",
        }

    def to_sample(self, row: dict[str, Any]) -> CanonicalSample:
        source_id = str(self.source_field(row, "source_sample_id"))
        identity_id = str(self.source_field(row, "identity_id"))
        hf_config = self.config.extras.get("hf_config") or self.config.extras.get(
            "hf_config_name"
        )
        # Preserve the HF configuration and split the row came from.
        source_split = self.source_field(row, "split", required=False) or row.get(
            "_source_split"
        )
        modality = self.source_field(row, "modality", required=False)
        if modality not in ("image_text", "text_only", "image_only"):
            modality = (
                "text_only"
                if not self.source_field(row, "image_uri", required=False)
                else "image_text"
            )
        sample = CanonicalSample(
            benchmark="ppubench",
            source_sample_id=source_id,
            identity_id=identity_id,
            provenance=self.provenance(
                source_id,
                source_subset=hf_config,
                notes=f"hf_config_name={hf_config}",
            ),
            source_subset=hf_config,
            identity_name=self.source_field(row, "identity_name", required=False),
            image_uri=self.source_field(row, "image_uri", required=False),
            modality=modality,
            task_type=self.source_field(row, "task_type", required=False) or "unlearning_qa",
            question=self.source_field(row, "question", required=False),
            answer_text=self.source_field(row, "answer_text", required=False),
            options=self.source_field(row, "options", required=False),
            split=source_split or "unassigned",
        )
        return sample.validate()
