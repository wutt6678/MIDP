"""Route-analysis / matched-modality probe construction (plan sections 12.4, 18).

Builds the paired eval-only probes needed for later mechanistic route analysis.
The first release does not implement activation patching; it only preserves the
matched examples (plan 18). Two artifacts are produced:

- **probe rows**: per-probe eval instances carrying a :class:`RouteProbe` that
  records the probe family, expected evidence source, and controlled variables;
- **pair manifest**: the matched left/right pairs (plan 18.2) with an explicit
  ``pair_type``, ``controlled``, ``changed``, and ``expected_route_effect``.

Rule baked into every visual probe (plan 12.4): the answer must follow the
current-image evidence; injected profile statements are deliberate synthetic
associations, never real-world facts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from ..config import ConfigError
from ..data.schemas import CanonicalSample, ProfileFact, RouteProbe
from ..prompts.registry import PromptRegistry

# probe_family -> (route template family, expected_evidence_source)
_FAMILY_TO_ROUTE: dict[str, tuple[str, str]] = {
    "direct_visual": ("direct_visual", "visual"),
    "name_only": ("name_only", "identity_fact"),
    "image_plus_name": ("image_plus_name", "visual"),
    "wrong_name": ("wrong_name", "visual"),
    "visual_text_conflict": ("visual_text_conflict", "conflict"),
    "cross_image": ("cross_image", "visual"),
}

# Pair types required for route analysis (plan 18.1) and their metadata.
PAIR_TYPES: dict[str, dict[str, Any]] = {
    "visual_vs_fact_same_image": {
        "controlled": ["image"],
        "changed": ["prompt_relation"],
        "expected_route_effect": "route_switch",
    },
    "cross_image_attribute_state": {
        "controlled": ["identity", "attribute"],
        "changed": ["image", "attribute_state"],
        "expected_route_effect": "visual_consistency",
    },
    "correct_name_vs_wrong_name": {
        "controlled": ["image", "question_relation"],
        "changed": ["identity_text"],
        "expected_route_effect": "identity_mediation",
    },
    "neutral_vs_conflict_text": {
        "controlled": ["image", "question"],
        "changed": ["context_text"],
        "expected_route_effect": "conflict_resolution",
    },
    "image_vs_name_conditioned": {
        "controlled": ["fact"],
        "changed": ["conditioning"],
        "expected_route_effect": "identity_mediation",
    },
    "attribute_counterfactual": {
        "controlled": ["identity", "prompt"],
        "changed": ["attribute_state"],
        "expected_route_effect": "visual_binding",
    },
}


class ConflictError(ValueError):
    """Raised when probe construction is missing required inputs."""


def _attr_phrase(attribute: str) -> str:
    return attribute.replace("_", " ").lower()


def conflict_claim_for(attribute: str, visible_label: bool) -> str:
    """Synthetic profile text that contradicts the current-image evidence."""
    phrase = _attr_phrase(attribute)
    return f"never shows {phrase}" if visible_label else f"always shows {phrase}"


def _probe_id(sample_id: str, family: str, attribute: str) -> str:
    raw = f"{sample_id}|{family}|{attribute}"
    return f"probe_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


@dataclass
class RouteProbeBuilder:
    """Builds route probes from a prompt registry + per-identity context."""

    prompt_registry: PromptRegistry
    registry_hash: str | None = None

    def __post_init__(self) -> None:
        if self.registry_hash is None:
            self.registry_hash = self.prompt_registry.registry_hash()

    # -- prompt fragments ------------------------------------------------ #

    def attribute_question(self, attribute: str) -> str:
        return self.prompt_registry.binary_entry(attribute).question

    @staticmethod
    def fact_question(fact: ProfileFact) -> str:
        return f"What is this person's {fact.relation.replace('_', ' ')}?"

    # -- probe rendering ------------------------------------------------- #

    def render_probe(
        self,
        family: str,
        *,
        attribute: str | None = None,
        identity_name: str | None = None,
        fact: ProfileFact | None = None,
        wrong_identity_name: str | None = None,
        visible_label: bool | None = None,
    ) -> str:
        if family not in _FAMILY_TO_ROUTE:
            raise ConflictError(f"Unknown probe family '{family}'")
        route_family, _ = _FAMILY_TO_ROUTE[family]
        fields: dict[str, object] = {}
        if attribute is not None:
            fields["attribute_question"] = self.attribute_question(attribute)
        if identity_name is not None:
            fields["identity_name"] = identity_name
        if wrong_identity_name is not None:
            fields["wrong_identity_name"] = wrong_identity_name
        if fact is not None:
            fields["fact_question"] = self.fact_question(fact)
            # P0-3: use the actual visible label so the conflict claim
            # genuinely contradicts the image evidence.
            vl = visible_label if visible_label is not None else True
            fields["conflict_claim"] = conflict_claim_for(
                attribute or fact.relation, visible_label=vl
            )
        return self.prompt_registry.render_route(route_family, **fields)

    # -- probe sample construction -------------------------------------- #

    def probe_sample(
        self,
        base: CanonicalSample,
        family: str,
        question: str,
        *,
        attribute: str | None = None,
        paired_sample_id: str | None = None,
        controlled_variables: Sequence[str] = (),
    ) -> CanonicalSample:
        _, evidence = _FAMILY_TO_ROUTE[family]
        spec = self.prompt_registry.route_template(_FAMILY_TO_ROUTE[family][0])
        probe = RouteProbe(
            probe_family=family,
            paired_sample_id=paired_sample_id,
            expected_evidence_source=evidence,
            controlled_variables=list(controlled_variables),
        ).validate()
        # P0-5: store the target attribute in source_metadata so probe_row
        # can retrieve it without an explicit parameter.
        new_meta = dict(base.source_metadata)
        if attribute is not None:
            new_meta["target_attribute"] = attribute
        # P0-6: for text_only modality probes, nullify image fields to
        # prevent visual leakage into text-only conditions.
        modality = str(spec.get("modality", base.modality))
        img_fields: dict[str, Any] = {}
        if modality == "text_only":
            img_fields = {"image_id": None, "image_uri": None, "image_sha256": None}
        return replace(
            base,
            modality=modality,
            task_type="route_probe",
            question=question,
            route_probe=probe,
            source_metadata=new_meta,
            **img_fields,
        ).validate()

    def probe_row(self, probe: CanonicalSample, *, attribute: str | None = None) -> dict[str, Any]:
        rp = probe.route_probe
        assert rp is not None
        # P0-5: retrieve attribute from source_metadata if not explicitly given.
        if attribute is None:
            attribute = probe.source_metadata.get("target_attribute")
        row: dict[str, Any] = {
            "probe_id": _probe_id(probe.source_sample_id, rp.probe_family, attribute or ""),
            "sample_id": probe.source_sample_id,
            "identity_id": probe.identity_id,
            "benchmark": probe.benchmark,
            "probe_family": rp.probe_family,
            "modality": probe.modality,
            "question": probe.question,
            "expected_evidence_source": rp.expected_evidence_source,
            "paired_sample_id": rp.paired_sample_id,
            "controlled_variables": rp.controlled_variables,
            "image_uri": probe.image_uri,
            "registry_hash": self.registry_hash,
        }
        # P0-5: add expected answer fields so downstream consumers know the
        # correct answer for each probe without re-deriving it.
        row["target_attribute"] = attribute
        answer_label, answer_text = self._expected_answer(probe, attribute)
        row["answer_label"] = answer_label
        row["answer_text"] = answer_text
        return row

    @staticmethod
    def _expected_answer(
        probe: CanonicalSample, attribute: str | None
    ) -> tuple[Any, str | None]:
        """Derive the expected (answer_label, answer_text) for a route probe.

        Visual probe families follow the image evidence; name_only follows
        the profile fact value.
        """
        from .annotate import CELEBA40_NAMESPACE

        family = probe.route_probe.probe_family if probe.route_probe else None
        if family == "name_only":
            # Text-only: answer comes from the first available profile fact.
            for fact in probe.profile_facts:
                if attribute and fact.relation != attribute:
                    continue
                return None, fact.value
            # Fallback: first fact if attribute didn't match.
            if probe.profile_facts:
                return None, probe.profile_facts[0].value
            return None, None
        # Visual families: answer follows the image evidence.
        if attribute:
            prefix = CELEBA40_NAMESPACE + "."
            key = prefix + attribute
            obs = probe.visual_attributes.get(key)
            if obs is not None and obs.label is not None:
                return obs.label, "yes" if obs.label else "no"
        return None, None


# --------------------------------------------------------------------------- #
# Identity-level probe construction
# --------------------------------------------------------------------------- #


def _accepted_visible_attributes(sample: CanonicalSample) -> dict[str, bool]:
    from .annotate import CELEBA40_NAMESPACE

    prefix = CELEBA40_NAMESPACE + "."
    out: dict[str, bool] = {}
    for key, obs in sample.visual_attributes.items():
        if key.startswith(prefix) and obs.label is not None and obs.confidence_band == "high":
            out[key[len(prefix):]] = bool(obs.label)
    return out


def build_identity_probes(
    identity_samples: Sequence[CanonicalSample],
    builder: RouteProbeBuilder,
    *,
    wrong_identity_name: str | None = None,
) -> list[CanonicalSample]:
    """Generate the six probe types for one identity's samples.

    Requires at least one image with an accepted visible attribute and one
    profile fact; otherwise raises so the caller can decide what to skip.
    """
    if not identity_samples:
        raise ConflictError("build_identity_probes requires at least one sample")

    # P0-10: find an anchor with accepted visible attributes instead of
    # blindly using identity_samples[0].
    anchor = None
    attributes: dict[str, bool] = {}
    for s in identity_samples:
        attrs = _accepted_visible_attributes(s)
        if attrs:
            anchor = s
            attributes = attrs
            break
    if anchor is None:
        raise ConflictError(
            f"Identity {identity_samples[0].identity_id} has no accepted visible attribute"
        )

    # Fix 5: aggregate profile facts across the complete identity group
    # instead of taking only from the visual anchor.  Different samples for
    # the same identity may carry different facts (e.g. one sample has image
    # + visual labels, another has text-only profile facts).
    facts_by_id: dict[str, ProfileFact] = {}
    for s in identity_samples:
        for fact in s.profile_facts:
            facts_by_id[fact.fact_id] = fact
    facts = list(facts_by_id.values())
    if not facts:
        raise ConflictError(f"Identity {anchor.identity_id} has no profile facts")

    attribute = sorted(attributes)[0]
    visible_label = attributes[attribute]
    identity_name = anchor.identity_name or anchor.identity_id
    fact = facts[0]
    probes: list[CanonicalSample] = []

    def make(family: str, **render_fields: Any) -> CanonicalSample:
        question = builder.render_probe(family, attribute=attribute, **render_fields)
        return builder.probe_sample(
            anchor, family, question, attribute=attribute, controlled_variables=["image"]
        )

    probes.append(make("direct_visual"))
    probes.append(make("name_only", identity_name=identity_name, fact=fact))
    probes.append(make("image_plus_name", identity_name=identity_name))
    if wrong_identity_name:
        probes.append(make("wrong_name", wrong_identity_name=wrong_identity_name))
    probes.append(make("visual_text_conflict", identity_name=identity_name, fact=fact,
                        visible_label=visible_label))

    # P0-11: cross_image must use a genuinely different image as the second
    # sample, not the same anchor.
    # P0-4: the target attribute must also be accepted (high-confidence) on
    # the second image so the cross-image probe is meaningful.
    if len(identity_samples) > 1:
        second = None
        for s in identity_samples:
            if s.source_sample_id != anchor.source_sample_id and s.image_uri != anchor.image_uri:
                second_attrs = _accepted_visible_attributes(s)
                if attribute in second_attrs:
                    second = s
                    break
        if second is not None:
            question = builder.render_probe(
                "cross_image", attribute=attribute, identity_name=identity_name
            )
            probes.append(
                builder.probe_sample(
                    second,
                    "cross_image",
                    question,
                    attribute=attribute,
                    paired_sample_id=anchor.source_sample_id,
                    controlled_variables=["identity", "attribute"],
                )
            )
    return probes


# --------------------------------------------------------------------------- #
# Pair manifest (plan 18.2)
# --------------------------------------------------------------------------- #


def make_pair(
    pair_type: str,
    left_sample_id: str,
    right_sample_id: str,
    *,
    index: int = 0,
) -> dict[str, Any]:
    if pair_type not in PAIR_TYPES:
        raise ConflictError(f"Unknown pair_type '{pair_type}'; use one of {sorted(PAIR_TYPES)}")
    meta = PAIR_TYPES[pair_type]
    return {
        "pair_id": f"pair_{index:06d}",
        "left_sample_id": left_sample_id,
        "right_sample_id": right_sample_id,
        "pair_type": pair_type,
        "controlled": list(meta["controlled"]),
        "changed": list(meta["changed"]),
        "expected_route_effect": meta["expected_route_effect"],
    }


def build_pair_manifest(pairs: Iterable[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Assign stable ``pair_id`` values and validate each pair's type.

    Extra fields (e.g. ``attribute``, ``left_label``, ``right_label`` for
    per-attribute cross_image pairs — Fix 4) are preserved in the output.
    """
    manifest: list[dict[str, Any]] = []
    for i, pair in enumerate(pairs):
        entry = make_pair(
            str(pair["pair_type"]),
            str(pair["left_sample_id"]),
            str(pair["right_sample_id"]),
            index=i,
        )
        # Pass through extra fields beyond the standard pair keys.
        for key in ("attribute", "left_label", "right_label"):
            if key in pair:
                entry[key] = pair[key]
        manifest.append(entry)
    return manifest


def validate_pair_manifest(
    pairs: Sequence[Mapping[str, Any]],
    samples_by_id: Mapping[str, CanonicalSample],
) -> list[str]:
    """Validate semantic constraints on a pair manifest.

    Returns a list of human-readable issue strings (empty == all clean).

    Checks performed per pair:

    - both ``left_sample_id`` and ``right_sample_id`` exist in *samples_by_id*;
    - both samples belong to the same identity;
    - the two samples reference different images (``image_uri``);
    - for ``cross_image_attribute_state`` pairs, the target attribute must be
      accepted (high-confidence visible) on both images and the left/right
      labels must differ;
    - ``left_label`` / ``right_label`` are present where the pair type
      carries an ``attribute`` field.
    """
    issues: list[str] = []
    for pair in pairs:
        pid = pair.get("pair_id", "?")
        left_id = pair.get("left_sample_id", "")
        right_id = pair.get("right_sample_id", "")
        pair_type = pair.get("pair_type", "")

        # Both samples must exist.
        left = samples_by_id.get(left_id)
        right = samples_by_id.get(right_id)
        if left is None:
            issues.append(f"{pid}: left_sample_id {left_id} not found")
            continue
        if right is None:
            issues.append(f"{pid}: right_sample_id {right_id} not found")
            continue

        # Same identity.
        if left.identity_id != right.identity_id:
            issues.append(
                f"{pid}: identity mismatch ({left.identity_id} vs {right.identity_id})"
            )

        # Different images.
        if left.image_uri == right.image_uri:
            issues.append(f"{pid}: same image on both sides ({left.image_uri})")

        # Attribute-level pairs: labels must be accepted and differ.
        if pair_type == "cross_image_attribute_state":
            attr = pair.get("attribute")
            if not attr:
                issues.append(f"{pid}: cross_image pair missing 'attribute'")
            else:
                left_attrs = _accepted_visible_attributes(left)
                right_attrs = _accepted_visible_attributes(right)
                if attr not in left_attrs:
                    issues.append(
                        f"{pid}: attribute {attr} not accepted on left image"
                    )
                if attr not in right_attrs:
                    issues.append(
                        f"{pid}: attribute {attr} not accepted on right image"
                    )
                if (
                    attr in left_attrs
                    and attr in right_attrs
                    and left_attrs[attr] == right_attrs[attr]
                ):
                    issues.append(
                        f"{pid}: left/right labels identical for {attr}"
                    )
            # Explicit label fields should be present and differ.
            left_label = pair.get("left_label")
            right_label = pair.get("right_label")
            if left_label is None or right_label is None:
                issues.append(f"{pid}: missing left_label or right_label")
            elif left_label == right_label:
                issues.append(f"{pid}: left_label == right_label ({left_label})")

    return issues
