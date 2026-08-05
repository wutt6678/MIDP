"""MLLMU-Bench adapter (coding plan section 14).

Keeps the fictitious and public-celebrity subsets separate via
``source_subset`` (plan section 14.1): the celebrity subset may overlap Llama
pretraining knowledge and must not be the only evidence of learned/forgotten
information. Both multimodal and text-only QA are normalized into the same
canonical record.
"""

from __future__ import annotations

from typing import Any

from ..schemas import CanonicalSample, ProfileFact
from .base import BenchmarkAdapter, register_adapter

_FICTITIOUS_SUBSETS = {"fictitious", "fictional", "synthetic"}


@register_adapter("mllmu")
class MllmuAdapter(BenchmarkAdapter):
    required_fields = ("source_sample_id", "identity_id")

    def _default_field_map(self) -> dict[str, str]:
        return {
            "source_sample_id": "sample_id",
            "identity_id": "profile_id",
            "identity_name": "profile_name",
            "image_uri": "image",
            "split": "split",
            "question": "question",
            "answer_text": "answer",
            "options": "options",
            "task_type": "prompt_type",
            "modality": "modality",
            "source_subset": "subset",
        }

    def _normalized_subset(self, row: dict[str, Any]) -> str:
        raw = self.source_field(row, "source_subset", required=False)
        if raw is None:
            # Fall back to a boolean-ish flag if present.
            raw = row.get("is_fictitious")
            if raw is None:
                raise KeyError(
                    "[mllmu] cannot determine fictitious vs celebrity subset; "
                    "provide 'subset' or 'is_fictitious' in the source row"
                )
            return "mllmu_fictitious" if raw else "mllmu_celebrities"
        raw_l = str(raw).lower()
        return "mllmu_fictitious" if raw_l in _FICTITIOUS_SUBSETS else "mllmu_celebrities"

    def to_sample(self, row: dict[str, Any]) -> CanonicalSample:
        source_id = str(self.source_field(row, "source_sample_id"))
        identity_id = str(self.source_field(row, "identity_id"))
        subset = self._normalized_subset(row)
        modality = self.source_field(row, "modality", required=False)
        if modality not in ("image_text", "text_only", "image_only"):
            # MLLMU distinguishes multimodal vs text-only; default text_only
            # when no image is attached.
            modality = "text_only" if not self.source_field(row, "image_uri", required=False) else "image_text"
        facts = []
        fact_id = self.source_field(row, "profile_fact_id", required=False)
        if fact_id is not None:
            facts.append(
                ProfileFact(
                    fact_id=str(fact_id),
                    relation=self.source_field(row, "profile_fact_relation", required=False) or "profile_fact",
                    value=self.source_field(row, "answer_text", required=False) or "",
                    privacy_class="profile",
                    source="source_human",
                    forgettable=True,
                ).validate()
            )
        sample = CanonicalSample(
            benchmark="mllmu",
            source_sample_id=source_id,
            identity_id=identity_id,
            provenance=self.provenance(source_id, source_subset=subset),
            source_subset=subset,
            identity_name=self.source_field(row, "identity_name", required=False),
            image_uri=self.source_field(row, "image_uri", required=False),
            profile_facts=facts,
            modality=modality,
            task_type=self.source_field(row, "task_type", required=False) or "knowledge_qa",
            question=self.source_field(row, "question", required=False),
            answer_text=self.source_field(row, "answer_text", required=False),
            options=self.source_field(row, "options", required=False),
            split=self.source_field(row, "split", required=False) or "unassigned",
        )
        return sample.validate()
