"""Static visual-QA generation (plan sections 12.3, 16).

QA is produced from a *static, versioned, reviewable* template registry; the
first release deliberately does not let an LLM freely generate benchmark
questions (plan 16.1). Every generated row embeds the template-registry hash so
a template change is always detectable downstream (plan 16.2).

Leakage controls (plan 16.2):
- train / validation / test question templates must be disjoint;
- route-conflict templates are kept out of ordinary fine-tuning data;
- a unit-testable check asserts no exact (normalized) prompt string appears
  across protected splits.

Answer normalization (plan 16.3) supports exact binary answers, canonical
string answers, and multiple choice; a separate judge model is never used as the
primary metric for binary visual attributes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..config import ConfigError
from ..constants.celeba_attributes import CELEBA_ATTRIBUTE_SET
from ..data.schemas import CanonicalSample
from ..prompts.render import normalize_whitespace

PROTECTED_SPLITS: tuple[str, ...] = ("train", "validation", "test")
ANSWER_TRUE = "yes"
ANSWER_FALSE = "no"

# Canonical surface forms accepted for a binary answer (plan 16.3).
_TRUE_SYNONYMS = frozenset({"yes", "y", "true", "positive", "1"})
_FALSE_SYNONYMS = frozenset({"no", "n", "false", "negative", "0"})


class QaError(ValueError):
    """Raised when QA generation or leakage control fails."""


class QaLeakageError(QaError):
    """Raised when a question string leaks across protected splits."""


# --------------------------------------------------------------------------- #
# Answer normalization
# --------------------------------------------------------------------------- #


def canonical_answer_text(label: bool) -> str:
    return ANSWER_TRUE if label else ANSWER_FALSE


def normalize_binary_answer(text: str) -> bool | None:
    """Map a free-form answer to True/False, or None when unparseable."""
    token = normalize_whitespace(str(text)).lower().strip(" .!?")
    if token in _TRUE_SYNONYMS:
        return True
    if token in _FALSE_SYNONYMS:
        return False
    return None


def normalize_choice_answer(text: str, options: Sequence[str]) -> int | None:
    """Return the index of the matching option, or None when ambiguous."""
    token = normalize_whitespace(str(text)).lower()
    matches = [i for i, opt in enumerate(options) if str(opt).lower() == token]
    if len(matches) == 1:
        return matches[0]
    return None


# --------------------------------------------------------------------------- #
# Template registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QaTemplate:
    template_id: str
    attribute: str
    split: str
    question: str

    def normalized_question(self) -> str:
        return normalize_whitespace(self.question).lower()


class QaTemplateRegistry:
    """Versioned collection of QA templates with leakage guarantees."""

    def __init__(self, templates: Iterable[QaTemplate], version: str):
        self.templates: list[QaTemplate] = list(templates)
        self.version = version
        if not self.templates:
            raise QaError("QaTemplateRegistry requires at least one template")
        for t in self.templates:
            if t.split not in PROTECTED_SPLITS:
                raise QaError(f"Template {t.template_id!r} has unknown split {t.split!r}")
            if t.attribute not in CELEBA_ATTRIBUTE_SET:
                raise QaError(f"Template {t.template_id!r} has unknown attribute {t.attribute!r}")

    # -- construction --------------------------------------------------- #

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "QaTemplateRegistry":
        """Build from a YAML-style mapping (plan 16.1 example format)."""
        version = str(doc.get("version", "qa_v1"))
        templates: list[QaTemplate] = []
        entries = doc.get("templates")
        if not isinstance(entries, list) or not entries:
            raise ConfigError("qa registry document has no 'templates' list")
        for entry in entries:
            attribute = entry.get("attribute")
            template_id = entry.get("id")
            answers = entry.get("answers", {})
            if not attribute or not template_id:
                raise QaError(f"QA template entry missing id/attribute: {entry!r}")
            for split in PROTECTED_SPLITS:
                for question in entry.get("question_templates", {}).get(split, []) or []:
                    templates.append(
                        QaTemplate(
                            template_id=f"{template_id}.{split}",
                            attribute=attribute,
                            split=split,
                            question=question,
                        )
                    )
        return cls(templates, version)

    @classmethod
    def default_for(cls, attributes: Sequence[str], version: str = "qa_v1") -> "QaTemplateRegistry":
        """Deterministic built-in templates (train/val/test are disjoint)."""
        templates: list[QaTemplate] = []
        for attribute in attributes:
            if attribute not in CELEBA_ATTRIBUTE_SET:
                raise QaError(f"Unknown attribute for QA templates: {attribute}")
            nice = attribute.replace("_", " ").lower()
            templates.extend(
                [
                    QaTemplate(
                        f"direct_visual_binary_v1.{attribute}.train",
                        attribute,
                        "train",
                        f"Is the person showing {nice} in this image?",
                    ),
                    QaTemplate(
                        f"direct_visual_binary_v1.{attribute}.validation",
                        attribute,
                        "validation",
                        f"Can you observe {nice} on the person in this photo?",
                    ),
                    QaTemplate(
                        f"direct_visual_binary_v1.{attribute}.test",
                        attribute,
                        "test",
                        f"Based only on this image, does the person have {nice}?",
                    ),
                ]
            )
        return cls(templates, version)

    # -- lookup --------------------------------------------------------- #

    def templates_for(self, attribute: str, split: str) -> list[QaTemplate]:
        return [t for t in self.templates if t.attribute == attribute and t.split == split]

    def attributes(self) -> list[str]:
        return sorted({t.attribute for t in self.templates})

    # -- provenance / leakage ------------------------------------------ #

    def registry_hash(self) -> str:
        payload = json.dumps(
            {
                "version": self.version,
                "templates": [
                    {"id": t.template_id, "attribute": t.attribute, "split": t.split,
                     "question": t.normalized_question()}
                    for t in sorted(self.templates, key=lambda t: t.template_id)
                ],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def assert_split_disjoint(self) -> None:
        """Fail loudly if a normalized question appears in more than one split."""
        seen: dict[str, str] = {}
        for template in self.templates:
            key = template.normalized_question()
            prior = seen.get(key)
            if prior is not None and prior != template.split:
                raise QaLeakageError(
                    f"Question {template.question!r} appears in both "
                    f"'{prior}' and '{template.split}' splits"
                )
            seen.setdefault(key, template.split)


# --------------------------------------------------------------------------- #
# Row generation
# --------------------------------------------------------------------------- #


def _row_id(sample_id: str, template_id: str) -> str:
    return f"qa_{hashlib.sha256(f'{sample_id}|{template_id}'.encode()).hexdigest()[:12]}"


def generate_binary_qa(
    sample: CanonicalSample,
    attribute: str,
    label: bool,
    registry: QaTemplateRegistry,
    *,
    split: str = "train",
    variant_index: int = 0,
    score: float | None = None,
) -> dict[str, Any]:
    """Render one binary visual-QA row for an accepted attribute observation."""
    templates = registry.templates_for(attribute, split)
    if not templates:
        raise QaError(f"No {split} templates for attribute '{attribute}'")
    template = templates[variant_index % len(templates)]
    return {
        "qa_id": _row_id(sample.source_sample_id, template.template_id),
        "sample_id": sample.source_sample_id,
        "identity_id": sample.identity_id,
        "benchmark": sample.benchmark,
        "image_uri": sample.image_uri,
        "image_sha256": sample.image_sha256,
        "attribute": attribute,
        "task_type": "visual_binary",
        "modality": "image_only",
        "split": split,
        "question": template.question,
        "template_id": template.template_id,
        "answer_label": bool(label),
        "answer_text": canonical_answer_text(label),
        "answer_score": score,
        "registry_hash": registry.registry_hash(),
        "registry_version": registry.version,
    }


def generate_qa_rows(
    samples: Iterable[CanonicalSample],
    registry: QaTemplateRegistry,
    *,
    split: str = "train",
    min_confidence_band: str | None = "high",
) -> list[dict[str, Any]]:
    """Generate QA for every accepted CelebA-40 observation in ``samples``.

    Only observations carrying an explicit boolean label are used; uncertain
    observations (``label is None``) are skipped so no forced label leaks into
    training or evaluation data (plan 11.1).
    """
    from .annotate import CELEBA40_NAMESPACE

    registry.assert_split_disjoint()
    band_rank = {"low": 0, "medium": 1, "high": 2}
    min_rank = band_rank[min_confidence_band] if min_confidence_band else -1
    rows: list[dict[str, Any]] = []
    prefix = CELEBA40_NAMESPACE + "."
    for sample in samples:
        for key, obs in sample.visual_attributes.items():
            if not key.startswith(prefix) or obs.label is None:
                continue
            if band_rank.get(obs.confidence_band, -1) < min_rank:
                continue
            attribute = key[len(prefix):]
            rows.append(
                generate_binary_qa(
                    sample,
                    attribute,
                    bool(obs.label),
                    registry,
                    split=split,
                    score=obs.score,
                )
            )
    return rows
