"""FAIRGET adapter (coding plan section 12).

FAIRGET ships multiple synthetic views per identity, FairFace-derived visual
demographic annotations, and ten identity-linked textual attributes. It is the
primary construction target. This adapter:

- preserves official split files;
- normalizes identity and image IDs;
- retains FairGet demographics under ``source_attributes.fairface.*`` (never
  overwritten by CelebA-style predictions, plan section 12.2);
- retains the ten textual profile attributes as :class:`ProfileFact` rows;
- maps existing QAs to canonical records;
- preserves source row identifiers.
"""

from __future__ import annotations

from typing import Any

from ..schemas import AttributeObservation, CanonicalSample, ProfileFact
from .base import BenchmarkAdapter, register_adapter

# Ten FAIRGET textual identity attributes (plan section 2.4 / 12.1).
_TEXTUAL_ATTRIBUTES = (
    "nationality",
    "occupation",
    "education",
    "relationship_status",
    "religion",
    "political_view",
    "gender_identity",
    "sexual_orientation",
    "health_condition",
    "financial_status",
)


@register_adapter("fairget")
class FairgetAdapter(BenchmarkAdapter):
    required_fields = ("source_sample_id", "identity_id")

    def _default_field_map(self) -> dict[str, str]:
        return {
            "source_sample_id": "sample_id",
            "identity_id": "identity_id",
            "identity_name": "identity_name",
            "image_id": "image_id",
            "image_uri": "image_path",
            "split": "split",
            "question": "question",
            "answer_text": "answer",
            "options": "options",
            "task_type": "task_type",
            "modality": "modality",
        }

    def _fairface_observations(self, row: dict[str, Any]) -> dict[str, AttributeObservation]:
        """FairGet visual demographics -> namespaced source observations."""
        obs: dict[str, AttributeObservation] = {}
        demographics = row.get("fairface") or row.get("demographics") or {}
        if isinstance(demographics, dict):
            for key, value in demographics.items():
                name = f"source_attributes.fairface.{key}"
                obs[name] = AttributeObservation(
                    name=name,
                    label=bool(value) if isinstance(value, (bool, int)) else None,
                    source="source_human",
                    confidence_band="high",
                    attribute_class="fairface_demographic",
                ).validate()
        return obs

    def _profile_facts(self, row: dict[str, Any]) -> list[ProfileFact]:
        facts: list[ProfileFact] = []
        profile = row.get("profile") or row.get("textual_attributes") or {}
        container = profile if isinstance(profile, dict) else {}
        for i, attr in enumerate(_TEXTUAL_ATTRIBUTES):
            if attr in container and container[attr] not in (None, ""):
                facts.append(
                    ProfileFact(
                        fact_id=f"fairget_{attr}",
                        relation=attr,
                        value=str(container[attr]),
                        privacy_class="identity_textual",
                        source="source_human",
                        forgettable=True,
                    ).validate()
                )
        return facts

    def to_sample(self, row: dict[str, Any]) -> CanonicalSample:
        source_id = str(self.source_field(row, "source_sample_id"))
        identity_id = str(self.source_field(row, "identity_id"))
        modality = self.source_field(row, "modality", required=False) or "image_text"
        sample = CanonicalSample(
            benchmark="fairget",
            source_sample_id=source_id,
            identity_id=identity_id,
            provenance=self.provenance(source_id),
            source_subset=self.source_field(row, "split", required=False),
            identity_name=self.source_field(row, "identity_name", required=False),
            image_id=self.source_field(row, "image_id", required=False),
            image_uri=self.source_field(row, "image_uri", required=False),
            visual_attributes=self._fairface_observations(row),
            profile_facts=self._profile_facts(row),
            modality=modality,
            task_type=self.source_field(row, "task_type", required=False) or "visual_qa",
            question=self.source_field(row, "question", required=False),
            answer_text=self.source_field(row, "answer_text", required=False),
            options=self.source_field(row, "options", required=False),
            split=self.source_field(row, "split", required=False) or "unassigned",
        )
        return sample.validate()
