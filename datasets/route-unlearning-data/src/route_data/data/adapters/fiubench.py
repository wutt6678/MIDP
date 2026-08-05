"""FIUBench adapter (coding plan section 13).

FIUBench holds fictitious facial identities with private-profile VQA and an
official forget/retain/evaluation grouping. Originally one primary image per
identity; the optional multi-view stage is handled elsewhere and never mixed
into the original records here (plan section 13.3).
"""

from __future__ import annotations

from typing import Any

from ..schemas import CanonicalSample, ProfileFact
from .base import BenchmarkAdapter, register_adapter


@register_adapter("fiubench")
class FiubenchAdapter(BenchmarkAdapter):
    required_fields = ("source_sample_id", "identity_id")

    def _default_field_map(self) -> dict[str, str]:
        return {
            "source_sample_id": "sample_id",
            "identity_id": "identity_id",
            "identity_name": "identity_name",
            "image_id": "image_id",
            "image_uri": "image",
            "split": "group",          # official forget/retain/evaluation grouping
            "question": "question",
            "answer_text": "answer",
            "task_type": "task_type",
            "forget_scope": "forget_scope",
        }

    def _private_facts(self, row: dict[str, Any]) -> list[ProfileFact]:
        facts: list[ProfileFact] = []
        profile = row.get("profile") or row.get("private_facts") or {}
        if isinstance(profile, dict):
            items = profile.items()
        elif isinstance(profile, list):
            items = enumerate(profile)
        else:
            items = []
        for key, value in items:
            if value in (None, ""):
                continue
            facts.append(
                ProfileFact(
                    fact_id=f"fiubench_{key}",
                    relation=str(key),
                    value=str(value),
                    privacy_class="private_profile",
                    source="source_human",
                    forgettable=True,
                ).validate()
            )
        return facts

    def to_sample(self, row: dict[str, Any]) -> CanonicalSample:
        source_id = str(self.source_field(row, "source_sample_id"))
        identity_id = str(self.source_field(row, "identity_id"))
        sample = CanonicalSample(
            benchmark="fiubench",
            source_sample_id=source_id,
            identity_id=identity_id,
            provenance=self.provenance(source_id, source_subset="original"),
            identity_name=self.source_field(row, "identity_name", required=False),
            image_id=self.source_field(row, "image_id", required=False),
            image_uri=self.source_field(row, "image_uri", required=False),
            profile_facts=self._private_facts(row),
            modality=self.source_field(row, "modality", required=False) or "image_text",
            task_type=self.source_field(row, "task_type", required=False) or "private_profile_vqa",
            question=self.source_field(row, "question", required=False),
            answer_text=self.source_field(row, "answer_text", required=False),
            split=self.source_field(row, "split", required=False) or "unassigned",
            forget_scope=self.source_field(row, "forget_scope", required=False),
        )
        return sample.validate()
