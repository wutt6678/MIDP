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
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

log = logging.getLogger(__name__)

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
        # P2-8: prefer the original FIUBench question when available so the
        # name_only probe uses the authentic QA wording, not a synthetic one.
        if fact.original_question:
            return fact.original_question
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
        target_fact: ProfileFact | None = None,
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
        # P0-C9: for name_only probes, store the exact target fact so the
        # expected answer is derived from the selected fact, not from
        # whatever profile facts happen to remain on the visual anchor.
        if target_fact is not None:
            new_meta["target_fact_id"] = target_fact.fact_id
            new_meta["target_fact_relation"] = target_fact.relation
            new_meta["target_fact_value"] = target_fact.value
            # P2-11: store fact provenance for exact traceability.
            if target_fact.source_qa_index is not None:
                new_meta["source_qa_index"] = target_fact.source_qa_index
            if target_fact.original_question is not None:
                new_meta["original_question"] = target_fact.original_question
            if target_fact.original_answer is not None:
                new_meta["original_answer"] = target_fact.original_answer
            new_meta["question_variant"] = target_fact.question_variant
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
            "image_sha256": probe.image_sha256,
            "registry_hash": self.registry_hash,
        }
        # P0-5: add expected answer fields so downstream consumers know the
        # correct answer for each probe without re-deriving it.
        # P0-C9: for name_only, target_attribute is null and the answer
        # comes from the stored target fact metadata.
        if rp.probe_family == "name_only":
            row["target_attribute"] = None
            row["target_fact_id"] = probe.source_metadata.get("target_fact_id")
            row["target_fact_relation"] = probe.source_metadata.get("target_fact_relation")
            row["target_fact_value"] = probe.source_metadata.get("target_fact_value")
            # P2-11: fact provenance fields.
            row["source_qa_index"] = probe.source_metadata.get("source_qa_index")
            row["original_question"] = probe.source_metadata.get("original_question")
            row["original_answer"] = probe.source_metadata.get("original_answer")
            row["question_variant"] = probe.source_metadata.get(
                "question_variant", "canonical"
            )
        else:
            row["target_attribute"] = attribute
            row["target_fact_id"] = None
            row["target_fact_relation"] = None
            row["target_fact_value"] = None
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
        the stored target fact value from source_metadata (P0-C9).
        """
        from .annotate import CELEBA40_NAMESPACE

        family = probe.route_probe.probe_family if probe.route_probe else None
        if family == "name_only":
            # P0-C9: use the exact target fact stored in source_metadata,
            # not whatever profile_facts happen to remain on the anchor.
            fact_value = probe.source_metadata.get("target_fact_value")
            if fact_value is not None:
                return None, str(fact_value)
            # Legacy fallback for probes built before P0-C9.
            for fact in probe.profile_facts:
                if attribute and fact.relation != attribute:
                    continue
                return None, fact.value
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


def _visual_attribute_jaccard(
    attrs_a: Mapping[str, bool], attrs_b: Mapping[str, bool],
) -> float:
    """R15: matching similarity metric, documented precisely.

    Metric: agreement-over-union on *signed attribute states*.  Each accepted
    (high-confidence) label is treated as a signed state such as
    ``("Eyeglasses", True)`` or ``("Smiling", False)``.  For the attributes
    accepted on both sides we count states that agree (same attribute *and*
    same polarity), then divide by the total number of attributes accepted on
    *either* side (the union).  Attributes accepted on only one side therefore
    count against similarity, and disagreeing polarity counts against it.

    Consequences:

    - The score lies in ``[0, 1]``; 1.0 means both sides accept exactly the
      same attribute set with identical states.
    - It is symmetric: ``sim(a, b) == sim(b, a)``.
    - It returns 0.0 when either side has no accepted attributes (or the
      accepted attribute sets are disjoint).

    Returns 0.0 when either side has no accepted attributes.
    """
    shared_keys = set(attrs_a) & set(attrs_b)
    if not shared_keys:
        return 0.0
    agreeing = sum(1 for k in shared_keys if attrs_a[k] == attrs_b[k])
    union_keys = set(attrs_a) | set(attrs_b)
    return agreeing / len(union_keys)


def _first_eligible_visual_attrs(
    group: Sequence[CanonicalSample],
) -> dict[str, bool]:
    """R14/R16: first sample in ``group`` with accepted visible attributes.

    Mirrors the visual-anchor selection used by ``build_identity_probes`` so
    wrong-name matching and coverage reporting never depend on ``group[0]``
    (which may be a text-only record with no visual labels).
    """
    for s in group:
        attrs = _accepted_visible_attributes(s)
        if attrs:
            return attrs
    return {}


def matched_wrong_name_details(
    identity_id: str,
    by_identity: Mapping[str, Sequence[CanonicalSample]],
    *,
    candidates_per_sample: int = 3,
) -> dict[str, Any] | None:
    """R14: best matched wrong-name control with research metadata.

    Searches *every* sample of each other identity group for the first
    eligible visual sample (an identity is no longer excluded merely because
    its first canonical record is text-only), ranks candidates by the
    signed-state similarity metric (:func:`_visual_attribute_jaccard`), and
    returns the best match with audit metadata:

    - ``wrong_identity_name`` / ``matched_wrong_identity_id``
    - ``matching_similarity``
    - ``matching_attributes`` (accepted attributes shared by both sides)
    - ``candidate_rank`` (1-based rank among all scored candidates)

    Ties are broken by sorted identity name for determinism.
    """
    group = by_identity.get(identity_id, [])
    if not group:
        return None

    anchor_attrs = _first_eligible_visual_attrs(group)
    if not anchor_attrs:
        return None

    scored: list[tuple[float, str, str, list[str]]] = []
    for other_id, other_group in by_identity.items():
        if other_id == identity_id:
            continue
        if len(other_group) < 2:
            continue
        # R14: search the whole group, not just other_group[0].
        other_attrs = _first_eligible_visual_attrs(other_group)
        if not other_attrs:
            continue
        sim = _visual_attribute_jaccard(anchor_attrs, other_attrs)
        shared = sorted(set(anchor_attrs) & set(other_attrs))
        other_name = other_group[0].identity_name or other_id
        scored.append((sim, other_name, other_id, shared))

    if not scored:
        return None

    scored.sort(key=lambda t: (-t[0], t[1]))
    best_sim, best_name, best_id, best_shared = scored[0]
    return {
        "wrong_identity_name": best_name,
        "matched_wrong_identity_id": best_id,
        "matching_similarity": best_sim,
        "matching_attributes": best_shared,
        "candidate_rank": 1,
        "candidates_considered": len(scored),
        "candidates_per_sample": candidates_per_sample,
        "matching_strategy": "visual_attribute_signed_state_jaccard",
    }


def select_matched_wrong_name(
    identity_id: str,
    by_identity: Mapping[str, Sequence[CanonicalSample]],
    *,
    candidates_per_sample: int = 3,
) -> str | None:
    """P2-19: select the best matched wrong-name control.

    Instead of picking the alphabetically-first candidate, rank other
    identities by visual-attribute similarity (Jaccard overlap of accepted
    high-confidence labels) so the wrong-name control is visually matched.
    Ties are broken by sorted identity name for determinism.

    Returns the single best wrong identity name, or ``None`` when no
    suitable candidate exists.
    """
    details = matched_wrong_name_details(
        identity_id, by_identity, candidates_per_sample=candidates_per_sample
    )
    return details["wrong_identity_name"] if details else None


def select_multiple_wrong_names(
    identity_id: str,
    by_identity: Mapping[str, Sequence[CanonicalSample]],
    *,
    candidates_per_sample: int = 3,
) -> list[str]:
    """P2-19: return up to ``candidates_per_sample`` matched wrong names."""
    group = by_identity.get(identity_id, [])
    if not group:
        return []

    anchor_attrs = _first_eligible_visual_attrs(group)
    if not anchor_attrs:
        return []

    scored: list[tuple[float, str, str]] = []
    for other_id, other_group in by_identity.items():
        if other_id == identity_id:
            continue
        if len(other_group) < 2:
            continue
        # R14: search the whole group, not just other_group[0].
        other_attrs = _first_eligible_visual_attrs(other_group)
        if not other_attrs:
            continue
        sim = _visual_attribute_jaccard(anchor_attrs, other_attrs)
        other_name = other_group[0].identity_name or other_id
        scored.append((sim, other_name, other_id))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [name for _, name, _ in scored[:candidates_per_sample]]


def _select_name_only_fact(facts: list[ProfileFact]) -> ProfileFact:
    """Pick the best fact for the name_only probe (P2-8).

    Preference order:
    1. A QA-derived fact (``source_qa_index is not None``) — these carry
       the original FIUBench question and answer.
    2. The first available fact (caption / raw_profile fallback).

    Raises :class:`ConflictError` when *facts* is empty (caller handles).
    """
    for f in facts:
        if f.source_qa_index is not None:
            return f
    return facts[0]


def build_identity_probes(
    identity_samples: Sequence[CanonicalSample],
    builder: RouteProbeBuilder,
    *,
    wrong_identity_name: str | None = None,
    experiment_attributes: set[str] | None = None,
    target_attribute: str | None = None,
) -> list[CanonicalSample]:
    """Generate the six probe types for one identity's samples.

    Requires at least one image with an accepted visible attribute and one
    profile fact; otherwise raises so the caller can decide what to skip.

    Parameters
    ----------
    experiment_attributes:
        If provided, restrict the target attribute selection to this set
        (the *experiment subset*).  Accepted visible attributes that are
        not in this set are excluded before the deterministic ``min()``
        selection.  This prevents attributes outside the frozen experiment
        plan (e.g. Wearing_Hat, Wearing_Necktie, Sideburns) from becoming
        route-probe targets.
    target_attribute:
        If provided, use this specific attribute as the probe target
        instead of ``min(eligible)``.  The attribute must be present in
        the anchor's eligible set; otherwise :class:`ConflictError` is
        raised.  This enables balanced probe assignment across attributes.
    """
    if not identity_samples:
        raise ConflictError("build_identity_probes requires at least one sample")

    # P0-3 (freeze): find an anchor with accepted visible attributes,
    # optionally restricted to the experiment-attribute subset.
    anchor = None
    attributes: dict[str, bool] = {}
    for s in identity_samples:
        attrs = _accepted_visible_attributes(s)
        if experiment_attributes is not None:
            attrs = {k: v for k, v in attrs.items() if k in experiment_attributes}
        if attrs:
            anchor = s
            attributes = attrs
            break
    if anchor is None:
        raise ConflictError(
            f"Identity {identity_samples[0].identity_id} has no accepted visible attribute"
            + (" in experiment subset" if experiment_attributes is not None else "")
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

    attribute = target_attribute if target_attribute is not None else min(attributes)
    if attribute not in attributes:
        raise ConflictError(
            f"Target attribute '{attribute}' not accepted on identity "
            f"{anchor.identity_id} (eligible: {sorted(attributes)})"
        )
    visible_label = attributes[attribute]
    identity_name = anchor.identity_name or anchor.identity_id
    # P2-8: prefer a QA-derived fact (one with source_qa_index set) for
    # the name_only probe so the question uses the original FIUBench
    # wording and the expected answer is the original answer.
    fact = _select_name_only_fact(facts)
    probes: list[CanonicalSample] = []

    def make(
        family: str,
        *,
        target_fact: ProfileFact | None = None,
        **render_fields: Any,
    ) -> CanonicalSample:
        question = builder.render_probe(family, attribute=attribute, **render_fields)
        return builder.probe_sample(
            anchor,
            family,
            question,
            attribute=attribute,
            controlled_variables=["image"],
            target_fact=target_fact,
        )

    probes.append(make("direct_visual"))
    # P0-C9: pass the exact selected fact as target_fact so the probe
    # stores it in source_metadata and the expected answer is derived
    # from that fact, not from the anchor's profile_facts.
    probes.append(make("name_only", target_fact=fact, identity_name=identity_name, fact=fact))
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
    """Validate semantic constraints on a pair manifest (P0-C10).

    Returns a list of human-readable issue strings (empty == all clean).

    Checks are pair-type-specific:

    - ``cross_image_attribute_state``: same identity, different image, same
      target attribute accepted on both, differing label state.
    - ``correct_name_vs_wrong_name``: same image, same identity, different
      identity text (controlled image + question/relation).
    - ``neutral_vs_conflict_text``: same image, same question, same expected
      answer; only context text changes.
    - ``visual_vs_fact_same_image``: same image, appropriate route/relation
      change.
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

        if pair_type == "cross_image_attribute_state":
            # Same identity, different image, same target attribute
            # accepted on both, differing label state.
            if left.identity_id != right.identity_id:
                issues.append(
                    f"{pid}: identity mismatch ({left.identity_id} vs {right.identity_id})"
                )
            if left.image_uri == right.image_uri:
                issues.append(f"{pid}: same image on both sides ({left.image_uri})")
            attr = pair.get("attribute")
            if not attr:
                issues.append(f"{pid}: cross_image pair missing 'attribute'")
            else:
                left_attrs = _accepted_visible_attributes(left)
                right_attrs = _accepted_visible_attributes(right)
                if attr not in left_attrs:
                    issues.append(f"{pid}: attribute {attr} not accepted on left image")
                if attr not in right_attrs:
                    issues.append(f"{pid}: attribute {attr} not accepted on right image")
                if (
                    attr in left_attrs
                    and attr in right_attrs
                    and left_attrs[attr] == right_attrs[attr]
                ):
                    issues.append(f"{pid}: left/right labels identical for {attr}")
            left_label = pair.get("left_label")
            right_label = pair.get("right_label")
            if left_label is None or right_label is None:
                issues.append(f"{pid}: missing left_label or right_label")
            elif left_label == right_label:
                issues.append(f"{pid}: left_label == right_label ({left_label})")

        elif pair_type == "correct_name_vs_wrong_name":
            # Same image, same identity, different identity text.
            if left.image_uri != right.image_uri:
                issues.append(
                    f"{pid}: correct_name_vs_wrong_name requires same image "
                    f"({left.image_uri} vs {right.image_uri})"
                )
            if left.identity_id != right.identity_id:
                issues.append(
                    f"{pid}: identity mismatch ({left.identity_id} vs {right.identity_id})"
                )

        elif pair_type == "neutral_vs_conflict_text":
            # Same image, same question, same expected answer; only context
            # text changes.
            if left.image_uri != right.image_uri:
                issues.append(
                    f"{pid}: neutral_vs_conflict_text requires same image "
                    f"({left.image_uri} vs {right.image_uri})"
                )
            if left.identity_id != right.identity_id:
                issues.append(
                    f"{pid}: identity mismatch ({left.identity_id} vs {right.identity_id})"
                )

        elif pair_type == "visual_vs_fact_same_image":
            # Same image, route/relation change.
            if left.image_uri != right.image_uri:
                issues.append(
                    f"{pid}: visual_vs_fact_same_image requires same image "
                    f"({left.image_uri} vs {right.image_uri})"
                )
            if left.identity_id != right.identity_id:
                issues.append(
                    f"{pid}: identity mismatch ({left.identity_id} vs {right.identity_id})"
                )

        elif pair_type == "image_vs_name_conditioned":
            # Same fact, different conditioning.
            if left.identity_id != right.identity_id:
                issues.append(
                    f"{pid}: identity mismatch ({left.identity_id} vs {right.identity_id})"
                )

        elif pair_type == "attribute_counterfactual":
            # Same identity + prompt, different attribute state.
            if left.identity_id != right.identity_id:
                issues.append(
                    f"{pid}: identity mismatch ({left.identity_id} vs {right.identity_id})"
                )

        else:
            # Unknown pair type: basic sanity.
            if left.identity_id != right.identity_id:
                issues.append(
                    f"{pid}: identity mismatch ({left.identity_id} vs {right.identity_id})"
                )

    return issues


# --------------------------------------------------------------------------- #
# P1-9: shared wrong-name eligibility (dict-compatible)
# --------------------------------------------------------------------------- #


def _accepted_visible_attrs_dict(sample) -> dict[str, bool]:
    """Dict-compatible version of ``_accepted_visible_attributes``.

    Works on both plain dicts and CanonicalSample objects.
    """
    from .annotate import CELEBA40_NAMESPACE

    prefix = CELEBA40_NAMESPACE + "."
    va = sample.get("visual_attributes", {}) if isinstance(sample, Mapping) else {}
    out: dict[str, bool] = {}
    if isinstance(va, Mapping):
        for key, obs in va.items():
            if not key.startswith(prefix):
                continue
            if isinstance(obs, Mapping):
                label = obs.get("label")
                band = obs.get("confidence_band")
            else:
                label = getattr(obs, "label", None)
                band = getattr(obs, "confidence_band", None)
            if label is not None and band == "high":
                out[key[len(prefix):]] = bool(label)
    return out


def _first_eligible_visual_attrs_dict(
    group: Sequence,
) -> dict[str, bool]:
    """Dict-compatible version of ``_first_eligible_visual_attrs``."""
    for s in group:
        attrs = _accepted_visible_attrs_dict(s)
        if attrs:
            return attrs
    return {}


def find_wrong_name_candidates(
    by_identity: Mapping[str, Sequence],
) -> list[tuple[str, str, float]]:
    """P1-9: return ``(target_id, control_id, similarity)`` triples.

    Uses the *same* eligibility logic as production route-probe generation:

    - target identity must have an eligible visual anchor (accepted
      high-confidence CelebA attributes);
    - control identity must differ from target, have ``>= 2`` samples, and
      also carry an eligible visual anchor;
    - similarity is the Jaccard matching ratio on signed attribute states.

    Returns an empty list when no valid pair exists.  The list is sorted by
    descending similarity, then alphabetically by target/control name.
    """
    # Step 1: find the visual anchor attrs for every identity.
    anchor_attrs: dict[str, dict[str, bool]] = {}
    for iid, group in by_identity.items():
        attrs = _first_eligible_visual_attrs_dict(group)
        if attrs:
            anchor_attrs[iid] = attrs

    # Step 2: build scored candidate pairs.
    triples: list[tuple[float, str, str]] = []
    eligible_ids = sorted(anchor_attrs)
    for i, target_id in enumerate(eligible_ids):
        for control_id in eligible_ids[i + 1:]:
            # At least one side must have >= 2 samples (production rule).
            if len(by_identity[target_id]) < 2 and len(by_identity[control_id]) < 2:
                continue
            sim = _visual_attribute_jaccard(
                anchor_attrs[target_id], anchor_attrs[control_id],
            )
            triples.append((sim, target_id, control_id))

    triples.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [(target, control, sim) for sim, target, control in triples]


# --------------------------------------------------------------------------- #
# Pre-inference structural wrong-name eligibility (P0-3 / P0-6 review 600ea5b)
# --------------------------------------------------------------------------- #


def _identity_name(sample: Mapping[str, Any] | Any) -> str | None:
    """Return the canonical identity name, or *None* if absent/blank.

    Works on both plain dicts (``CanonicalSample.to_dict()`` output) and
    :class:`CanonicalSample` objects.  The canonical field is
    ``identity_name`` — never the top-level ``name`` key.
    """
    if isinstance(sample, Mapping):
        name = sample.get("identity_name")
    else:
        name = getattr(sample, "identity_name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def structural_wrong_name_candidates(
    by_identity: Mapping[str, Sequence],
    *,
    min_control_rows: int = 2,
) -> list[dict[str, Any]]:
    """Pre-inference structural wrong-name candidate pairs.

    Returns every ``(target, control)`` pair that is structurally feasible
    *before* Qwen annotation — i.e. all conditions knowable without visual
    attribute labels.

    Conditions checked:

    * target identity has a non-blank ``identity_name``;
    * target identity has at least one image-bearing selected sample;
    * control identity differs from target;
    * control identity has a non-blank ``identity_name``;
    * control identity has at least one image-bearing selected sample;
    * control identity has ``>= min_control_rows`` selected canonical rows
      (matching the production ``len(other_group) < 2`` guard at
      :func:`matched_wrong_name_details`).

    Does **not** require accepted visual attributes, Jaccard similarity,
    or ``matching_similarity`` — those belong to post-annotation Gate B
    (:func:`find_wrong_name_candidates`).
    """
    # Collect per-identity structural facts.
    id_info: dict[str, dict[str, Any]] = {}
    for iid, group in by_identity.items():
        has_name = any(_identity_name(s) for s in group)
        has_image = any(
            (s.get("image_uri") if isinstance(s, Mapping) else getattr(s, "image_uri", None))
            for s in group
        )
        id_info[iid] = {
            "has_name": has_name,
            "has_image": has_image,
            "row_count": len(group),
        }

    eligible_targets = [
        iid for iid, info in id_info.items()
        if info["has_name"] and info["has_image"]
    ]
    eligible_controls = [
        iid for iid, info in id_info.items()
        if info["has_name"] and info["has_image"] and info["row_count"] >= min_control_rows
    ]

    pairs: list[dict[str, Any]] = []
    for tgt in sorted(eligible_targets):
        for ctrl in sorted(eligible_controls):
            if tgt == ctrl:
                continue
            pairs.append({
                "target_identity_id": tgt,
                "control_identity_id": ctrl,
                "target_selected_records": id_info[tgt]["row_count"],
                "control_selected_records": id_info[ctrl]["row_count"],
                "target_has_name": True,
                "control_has_name": True,
                "target_has_image": True,
                "control_has_image": True,
                "structurally_valid": True,
            })
    return pairs


def assign_balanced_route_attributes(
    eligible_by_identity: Mapping[str, Mapping[str, bool]],
    experiment_attributes: set[str],
    target_quota: int = 10,
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """State-balanced deterministic route-attribute assignment.

    Two-stage assignment ensures every experiment attribute has both
    positive and negative target identities when both states exist in
    the protocol-eligible source population.

    **Stage 1 — seed state coverage.**  For each attribute, if both
    positive-eligible and negative-eligible identities exist, at least
    one of each is assigned before general balancing.

    **Stage 2 — fill remaining quota.**  Unassigned identities are
    allocated to the least-assigned eligible attribute (ties broken
    alphabetically).

    Parameters
    ----------
    eligible_by_identity:
        ``{identity_id: {attr_name: label_bool, ...}}`` — accepted
        high-confidence experiment attributes and their label states
        for each protocol-eligible identity.
    experiment_attributes:
        The frozen experiment-v2 attribute set.
    target_quota:
        Maximum number of identities assigned to each attribute.

    Returns
    -------
    assignment : dict[str, str]
        ``{identity_id: selected_attribute}``.
    stats : dict[str, dict[str, int]]
        Per-attribute ``positive`` and ``negative`` assignment counts.
    """
    experiment_attrs = sorted(experiment_attributes)
    attr_counts: dict[str, int] = {a: 0 for a in experiment_attrs}
    assignment: dict[str, str] = {}

    # Pre-compute eligible positive / negative pools per attribute.
    eligible_pos: dict[str, list[str]] = {a: [] for a in experiment_attrs}
    eligible_neg: dict[str, list[str]] = {a: [] for a in experiment_attrs}
    for identity_id in sorted(eligible_by_identity):
        attrs_labels = eligible_by_identity[identity_id]
        for attr in experiment_attrs:
            if attr in attrs_labels:
                if attrs_labels[attr]:
                    eligible_pos[attr].append(identity_id)
                else:
                    eligible_neg[attr].append(identity_id)

    # Stage 1: seed at least 1 positive + 1 negative per attribute.
    for attr in experiment_attrs:
        if eligible_pos[attr] and eligible_neg[attr]:
            # P1-1: skip candidates already assigned to an earlier attribute.
            pos_id = next(
                (iid for iid in eligible_pos[attr] if iid not in assignment),
                None,
            )
            if pos_id is not None:
                assignment[pos_id] = attr
                attr_counts[attr] += 1
            neg_id = next(
                (iid for iid in eligible_neg[attr] if iid not in assignment),
                None,
            )
            if neg_id is not None:
                assignment[neg_id] = attr
                attr_counts[attr] += 1

    # Stage 2: fill remaining quota with least-assigned eligible attr.
    for identity_id in sorted(eligible_by_identity):
        if identity_id in assignment:
            continue
        eligible = sorted(
            a for a in eligible_by_identity[identity_id]
            if a in experiment_attrs and attr_counts[a] < target_quota
        )
        if not eligible:
            continue
        best = min(eligible, key=lambda a: attr_counts[a])
        assignment[identity_id] = best
        attr_counts[best] += 1

    # Compute per-attribute positive / negative stats.
    stats: dict[str, dict[str, int]] = {
        a: {"positive": 0, "negative": 0} for a in experiment_attrs
    }
    for identity_id, attr in assignment.items():
        label = eligible_by_identity[identity_id].get(attr)
        if label is True:
            stats[attr]["positive"] += 1
        elif label is False:
            stats[attr]["negative"] += 1

    log.info(
        "[experiment-attr] state-balanced assignment: counts=%s, stats=%s",
        {a: attr_counts[a] for a in experiment_attrs},
        stats,
    )
    return assignment, stats
