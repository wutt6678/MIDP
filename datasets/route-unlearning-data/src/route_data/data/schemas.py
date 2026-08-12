"""Canonical multimodal-unlearning schema (coding plan section 9).

Pydantic is intentionally not required: the runtime environment may not have
it. These dataclasses mirror the plan's ``BaseModel`` records and provide
``validate``/``to_dict``/``from_dict`` so records can round-trip through
Parquet (flat columns) and JSONL (nested objects) without losing fidelity.

Design rules enforced here (plan sections 3.1-3.3):
- visual attributes live at image level, never collapsed into identity-level
  biographies;
- model outputs are weak labels carrying explicit provenance, never silent
  ground truth;
- source annotations and derived predictions occupy separate namespaces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ATTRIBUTE_SOURCES: frozenset[str] = frozenset(
    {"source_human", "source_model", "human_verified_model", "derived"}
)
CONFIDENCE_BANDS: frozenset[str] = frozenset({"high", "medium", "low", "unknown"})
MODALITIES: frozenset[str] = frozenset({"image_text", "text_only", "image_only"})
PROBE_FAMILIES: frozenset[str] = frozenset(
    {
        "direct_visual",
        "name_only",
        "image_plus_name",
        "wrong_name",
        "visual_text_conflict",
        "cross_image",
        "visual_occlusion",
    }
)
EVIDENCE_SOURCES: frozenset[str] = frozenset(
    {"visual", "identity_fact", "conflict", "unknown"}
)


class SchemaError(ValueError):
    """Raised when a canonical record fails structural validation."""


def _check_literal(value: Any, allowed: frozenset[str], field_name: str) -> None:
    if value not in allowed:
        raise SchemaError(
            f"{field_name}={value!r} is not one of the allowed values {sorted(allowed)}"
        )


def _require_nonempty(value: Any, field_name: str) -> None:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise SchemaError(f"{field_name} must be a non-empty value, got {value!r}")


# --------------------------------------------------------------------------- #
# Nested record types
# --------------------------------------------------------------------------- #


@dataclass
class AttributeObservation:
    """A single per-image attribute label with full provenance."""

    name: str
    label: bool | None
    score: float | None = None
    source: str = "derived"
    model_fingerprint: str | None = None
    prompt_id: str | None = None
    confidence_band: str = "unknown"
    attribute_class: str = "unknown"

    def validate(self) -> AttributeObservation:
        _require_nonempty(self.name, "AttributeObservation.name")
        _check_literal(self.source, ATTRIBUTE_SOURCES, "AttributeObservation.source")
        _check_literal(
            self.confidence_band, CONFIDENCE_BANDS, "AttributeObservation.confidence_band"
        )
        if self.label is not None and not isinstance(self.label, bool):
            raise SchemaError("AttributeObservation.label must be bool or None")
        if self.score is not None and not isinstance(self.score, (int, float)):
            raise SchemaError("AttributeObservation.score must be numeric or None")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttributeObservation:
        return cls(
            name=data["name"],
            label=data.get("label"),
            score=data.get("score"),
            source=data.get("source", "derived"),
            model_fingerprint=data.get("model_fingerprint"),
            prompt_id=data.get("prompt_id"),
            confidence_band=data.get("confidence_band", "unknown"),
            attribute_class=data.get("attribute_class", "unknown"),
        ).validate()


@dataclass
class ProfileFact:
    """An identity-linked factual statement (textual knowledge)."""

    fact_id: str
    relation: str
    value: str
    privacy_class: str = "public"
    source: str = "source_human"
    forgettable: bool = False
    # P2-11: fact provenance fields for exact traceability.
    source_qa_index: int | None = None
    original_question: str | None = None
    original_answer: str | None = None
    question_variant: str = "canonical"

    def validate(self) -> ProfileFact:
        _require_nonempty(self.fact_id, "ProfileFact.fact_id")
        _require_nonempty(self.relation, "ProfileFact.relation")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileFact:
        return cls(
            fact_id=data["fact_id"],
            relation=data["relation"],
            value=data.get("value", ""),
            privacy_class=data.get("privacy_class", "public"),
            source=data.get("source", "source_human"),
            forgettable=bool(data.get("forgettable", False)),
            source_qa_index=data.get("source_qa_index"),
            original_question=data.get("original_question"),
            original_answer=data.get("original_answer"),
            question_variant=data.get("question_variant", "canonical"),
        ).validate()


@dataclass
class RouteProbe:
    """Marks a sample as a route / matched-modality probe."""

    probe_family: str
    paired_sample_id: str | None = None
    expected_evidence_source: str = "unknown"
    controlled_variables: list[str] = field(default_factory=list)

    def validate(self) -> RouteProbe:
        _check_literal(self.probe_family, PROBE_FAMILIES, "RouteProbe.probe_family")
        _check_literal(
            self.expected_evidence_source,
            EVIDENCE_SOURCES,
            "RouteProbe.expected_evidence_source",
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RouteProbe:
        return cls(
            probe_family=data["probe_family"],
            paired_sample_id=data.get("paired_sample_id"),
            expected_evidence_source=data.get("expected_evidence_source", "unknown"),
            controlled_variables=list(data.get("controlled_variables", [])),
        ).validate()


@dataclass
class Provenance:
    """Where a record came from; required for every canonical sample."""

    source_dataset: str
    source_version: str = "unknown"
    source_sample_id: str | None = None
    source_subset: str | None = None
    adapter: str | None = None
    adapter_version: str | None = None
    created_utc: str | None = None
    notes: str | None = None

    def validate(self) -> Provenance:
        _require_nonempty(self.source_dataset, "Provenance.source_dataset")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        return cls(
            source_dataset=data["source_dataset"],
            source_version=data.get("source_version", "unknown"),
            source_sample_id=data.get("source_sample_id"),
            source_subset=data.get("source_subset"),
            adapter=data.get("adapter"),
            adapter_version=data.get("adapter_version"),
            created_utc=data.get("created_utc"),
            notes=data.get("notes"),
        ).validate()


# --------------------------------------------------------------------------- #
# Top-level canonical record
# --------------------------------------------------------------------------- #


@dataclass
class CanonicalSample:
    benchmark: str
    source_sample_id: str
    identity_id: str
    provenance: Provenance
    source_subset: str | None = None
    # Raw upstream record identifier (pre-flattening), when one source row
    # expands into many canonical records (repair plan B3).
    source_record_id: str | None = None
    identity_name: str | None = None
    image_id: str | None = None
    image_uri: str | None = None
    image_sha256: str | None = None
    visual_attributes: dict[str, AttributeObservation] = field(default_factory=dict)
    profile_facts: list[ProfileFact] = field(default_factory=list)
    modality: str = "image_only"
    task_type: str = "visual_attribute"
    question: str | None = None
    answer_text: str | None = None
    answer_label: Any = None
    options: list[str] | None = None
    split: str = "unassigned"
    forget_scope: str | None = None
    route_probe: RouteProbe | None = None
    # Source configuration, nested task path/index, image field/view, original
    # answer label, source file, etc. (repair plan B3). JSON-safe values only.
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> CanonicalSample:
        _require_nonempty(self.benchmark, "CanonicalSample.benchmark")
        _require_nonempty(self.source_sample_id, "CanonicalSample.source_sample_id")
        _require_nonempty(self.identity_id, "CanonicalSample.identity_id")
        _check_literal(self.modality, MODALITIES, "CanonicalSample.modality")
        self.provenance.validate()
        for obs in self.visual_attributes.values():
            obs.validate()
        for fact in self.profile_facts:
            fact.validate()
        if self.route_probe is not None:
            self.route_probe.validate()
        if not isinstance(self.source_metadata, dict):
            raise SchemaError("CanonicalSample.source_metadata must be a dict")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "source_subset": self.source_subset,
            "source_record_id": self.source_record_id,
            "source_sample_id": self.source_sample_id,
            "source_metadata": dict(self.source_metadata),
            "identity_id": self.identity_id,
            "identity_name": self.identity_name,
            "image_id": self.image_id,
            "image_uri": self.image_uri,
            "image_sha256": self.image_sha256,
            "visual_attributes": {k: v.to_dict() for k, v in self.visual_attributes.items()},
            "profile_facts": [f.to_dict() for f in self.profile_facts],
            "modality": self.modality,
            "task_type": self.task_type,
            "question": self.question,
            "answer_text": self.answer_text,
            "answer_label": self.answer_label,
            "options": self.options,
            "split": self.split,
            "forget_scope": self.forget_scope,
            "route_probe": self.route_probe.to_dict() if self.route_probe else None,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CanonicalSample:
        visual = {
            k: AttributeObservation.from_dict(v)
            for k, v in (data.get("visual_attributes") or {}).items()
        }
        facts = [ProfileFact.from_dict(f) for f in (data.get("profile_facts") or [])]
        probe = (
            RouteProbe.from_dict(data["route_probe"]) if data.get("route_probe") else None
        )
        return cls(
            benchmark=data["benchmark"],
            source_sample_id=data["source_sample_id"],
            identity_id=data["identity_id"],
            provenance=Provenance.from_dict(data["provenance"]),
            source_subset=data.get("source_subset"),
            source_record_id=data.get("source_record_id"),
            identity_name=data.get("identity_name"),
            image_id=data.get("image_id"),
            image_uri=data.get("image_uri"),
            image_sha256=data.get("image_sha256"),
            visual_attributes=visual,
            profile_facts=facts,
            modality=data.get("modality", "image_only"),
            task_type=data.get("task_type", "visual_attribute"),
            question=data.get("question"),
            answer_text=data.get("answer_text"),
            answer_label=data.get("answer_label"),
            options=data.get("options"),
            split=data.get("split", "unassigned"),
            forget_scope=data.get("forget_scope"),
            route_probe=probe,
            source_metadata=dict(data.get("source_metadata") or {}),
        ).validate()
