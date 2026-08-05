"""Versioned prompt registry (coding plan sections 8.2, 8.8, 16.2).

Loads the human-readable prompt dictionaries (Mode A single-attribute,
Mode B grouped JSON, route-conflict probes) and exposes deterministic
lookup/rendering. Every generated row must record the registry hash so a
prompt change is always detectable downstream (plan section 16.2).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ConfigError, PromptsConfig, load_yaml
from ..constants.celeba_attributes import CELEBA_ATTRIBUTES
from .render import normalize_whitespace, render


@dataclass(frozen=True)
class BinaryPromptEntry:
    attribute: str
    question: str
    variants: tuple[str, ...]
    attribute_class: str


class PromptRegistry:
    """Single source of truth for all prompt templates.

    Lazily loads each YAML the first time its family is requested so a
    build that only needs Mode A never requires the route-conflict file.
    """

    def __init__(self, config: PromptsConfig):
        self.config = config
        self._binary: dict[str, Any] | None = None
        self._grouped: dict[str, Any] | None = None
        self._route: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _require(self, path: str | None, family: str) -> dict[str, Any]:
        if not path:
            raise ConfigError(f"Prompt file for '{family}' is not configured")
        return load_yaml(path)

    def _binary_doc(self) -> dict[str, Any]:
        if self._binary is None:
            doc = self._require(self.config.binary, "binary")
            attrs = doc.get("attributes")
            if not isinstance(attrs, dict) or not attrs:
                raise ConfigError("binary prompt file has no 'attributes' mapping")
            # Fail loudly if a canonical attribute is missing a prompt.
            missing = [a for a in CELEBA_ATTRIBUTES if a not in attrs]
            if missing:
                raise ConfigError(f"Binary prompts missing for attributes: {missing}")
            self._binary = doc
        return self._binary

    def _grouped_doc(self) -> dict[str, Any]:
        if self._grouped is None:
            doc = self._require(self.config.grouped, "grouped")
            if not isinstance(doc.get("groups"), dict):
                raise ConfigError("grouped prompt file has no 'groups' mapping")
            if not doc.get("prompt_template"):
                raise ConfigError("grouped prompt file has no 'prompt_template'")
            self._grouped = doc
        return self._grouped

    def _route_doc(self) -> dict[str, Any]:
        if self._route is None:
            doc = self._require(self.config.route_conflict, "route_conflict")
            if not isinstance(doc.get("templates"), dict):
                raise ConfigError("route prompt file has no 'templates' mapping")
            self._route = doc
        return self._route

    # ------------------------------------------------------------------ #
    # Mode A: single-attribute binary prompts
    # ------------------------------------------------------------------ #

    @property
    def binary_defaults(self) -> dict[str, str]:
        return dict(self._binary_doc().get("defaults", {}))

    def binary_entry(self, attribute: str) -> BinaryPromptEntry:
        attrs = self._binary_doc()["attributes"]
        if attribute not in attrs:
            raise ConfigError(f"No binary prompt defined for attribute '{attribute}'")
        spec = attrs[attribute]
        question = spec.get("question")
        if not question:
            raise ConfigError(f"Attribute '{attribute}' has no 'question'")
        return BinaryPromptEntry(
            attribute=attribute,
            question=question,
            variants=tuple(spec.get("variants", []) or []),
            attribute_class=spec.get("class", "unknown"),
        )

    def binary_prompt(self, attribute: str, variant_index: int | None = None) -> str:
        """Assemble the full Mode-A prompt for one attribute.

        ``variant_index=None`` uses the primary question; an integer selects a
        prompt-stability variant (plan section 8.8).
        """
        entry = self.binary_entry(attribute)
        defaults = self.binary_defaults
        if variant_index is None:
            question = entry.question
        else:
            if variant_index < 0 or variant_index >= len(entry.variants):
                raise ConfigError(
                    f"Attribute '{attribute}' has {len(entry.variants)} variants; "
                    f"index {variant_index} out of range"
                )
            question = entry.variants[variant_index]
        parts = [
            defaults.get("instruction", ""),
            question,
            defaults.get("answer_constraint", ""),
        ]
        return normalize_whitespace("\n".join(p for p in parts if p))

    def binary_variants_count(self, attribute: str) -> int:
        return len(self.binary_entry(attribute).variants)

    # ------------------------------------------------------------------ #
    # Mode B: grouped JSON prompts
    # ------------------------------------------------------------------ #

    def grouped_names(self) -> list[str]:
        return list(self._grouped_doc()["groups"].keys())

    def grouped_attributes(self, group: str) -> list[str]:
        groups = self._grouped_doc()["groups"]
        if group not in groups:
            raise ConfigError(f"Unknown grouped-prompt group '{group}'")
        return list(groups[group])

    def grouped_prompt(self, group: str) -> tuple[str, list[str]]:
        """Return ``(prompt_text, attribute_keys)`` for a grouped query."""
        keys = self.grouped_attributes(group)
        template = self._grouped_doc()["prompt_template"]
        keys_repr = ", ".join(f'"{k}"' for k in keys)
        text = normalize_whitespace(render(template, keys=keys_repr))
        return text, keys

    # ------------------------------------------------------------------ #
    # Route-conflict / matched-modality probes
    # ------------------------------------------------------------------ #

    def route_families(self) -> list[str]:
        return list(self._route_doc()["templates"].keys())

    def route_template(self, family: str) -> dict[str, Any]:
        templates = self._route_doc()["templates"]
        if family not in templates:
            raise ConfigError(f"Unknown route family '{family}'")
        return dict(templates[family])

    def render_route(self, family: str, **fields: object) -> str:
        """Render a route probe prompt for ``family``.

        All template fields must be supplied; missing fields raise via
        :func:`render`. The preamble (if present) is prepended.
        """
        spec = self.route_template(family)
        chunks: list[str] = []
        if spec.get("preamble"):
            chunks.append(render(spec["preamble"], **fields))
        if spec.get("question"):
            chunks.append(render(spec["question"], **fields))
        return normalize_whitespace(" ".join(chunks))

    # ------------------------------------------------------------------ #
    # Provenance
    # ------------------------------------------------------------------ #

    def _canonical_docs(self) -> dict[str, Any]:
        docs: dict[str, Any] = {}
        if self.config.binary:
            docs["binary"] = self._binary_doc()
        if self.config.grouped:
            docs["grouped"] = self._grouped_doc()
        if self.config.route_conflict:
            docs["route"] = self._route_doc()
        return docs

    def registry_hash(self) -> str:
        """Stable sha256 over all loaded prompt documents (plan 16.2)."""
        payload = json.dumps(self._canonical_docs(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def versions(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, doc in self._canonical_docs().items():
            out[key] = str(doc.get("version", "unknown"))
        return out

    def fingerprint(self) -> dict[str, Any]:
        return {
            "prompt_registry_hash": self.registry_hash(),
            "versions": self.versions(),
        }
