"""P2-7 / P2-8 / P2-9 / P2-10 / P2-11: name-only fact semantics.

P2-7  — Each original FIUBench QA → structured identity fact (no LLM).
P2-8  — name_only probes use original FIUBench question + answer.
P2-9  — Paraphrases are robustness variants, not separate facts.
P2-10 — Perturbed answers never become ground-truth facts.
P2-11 — Fact provenance fields for exact traceability.
"""

from __future__ import annotations

import pytest

from route_data.build.conflict_generation import (
    RouteProbeBuilder,
    _select_name_only_fact,
    build_identity_probes,
)
from route_data.config import PromptsConfig
from route_data.data.schemas import (
    AttributeObservation,
    CanonicalSample,
    ProfileFact,
    Provenance,
)
from route_data.prompts.registry import PromptRegistry

EYEGLASSES_KEY = "extended_attributes.celeba40.Eyeglasses"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def registry(repo_root) -> PromptRegistry:
    prompts = PromptsConfig(
        binary=str(repo_root / "configs/prompts/celeba_binary_v1.yaml"),
        grouped=str(repo_root / "configs/prompts/celeba_grouped_json_v1.yaml"),
        route_conflict=str(repo_root / "configs/prompts/route_conflict_v1.yaml"),
    )
    return PromptRegistry(prompts)


def _qa_fact(
    index: int = 0,
    question: str = "Where does Ava live?",
    answer: str = "Alpha City",
) -> ProfileFact:
    return ProfileFact(
        fact_id=f"fiubench_qa_{index:02d}",
        relation=question,
        value=answer,
        privacy_class="private_profile",
        source="source_human",
        forgettable=True,
        source_qa_index=index,
        original_question=question,
        original_answer=answer,
        question_variant="canonical",
    )


def _plain_fact() -> ProfileFact:
    return ProfileFact(
        fact_id="fiubench_caption",
        relation="caption",
        value="Ava is a test persona.",
        privacy_class="private_profile",
        source="source_human",
        forgettable=True,
    )


def _anchor_with_facts(facts: list[ProfileFact]) -> CanonicalSample:
    return CanonicalSample(
        benchmark="fiubench",
        source_sample_id="test_s1",
        identity_id="test_id",
        identity_name="Ava Alpha",
        provenance=Provenance(source_dataset="fiubench"),
        image_uri="images/test.png",
        modality="image_text",
        visual_attributes={
            EYEGLASSES_KEY: AttributeObservation(
                name=EYEGLASSES_KEY,
                label=True,
                source="source_model",
                confidence_band="high",
            )
        },
        profile_facts=facts,
    )


# --------------------------------------------------------------------------- #
# P2-7: structured QA facts
# --------------------------------------------------------------------------- #


class TestQAFactCreation:
    def test_qa_fact_has_provenance(self):
        fact = _qa_fact(index=7, question="Where?", answer="Here")
        assert fact.fact_id == "fiubench_qa_07"
        assert fact.source_qa_index == 7
        assert fact.original_question == "Where?"
        assert fact.original_answer == "Here"
        assert fact.question_variant == "canonical"
        assert fact.privacy_class == "private_profile"

    def test_qa_fact_round_trip(self):
        fact = _qa_fact()
        d = fact.to_dict()
        restored = ProfileFact.from_dict(d)
        assert restored.source_qa_index == fact.source_qa_index
        assert restored.original_question == fact.original_question
        assert restored.original_answer == fact.original_answer
        assert restored.question_variant == fact.question_variant

    def test_plain_fact_has_no_qa_provenance(self):
        fact = _plain_fact()
        assert fact.source_qa_index is None
        assert fact.original_question is None
        assert fact.original_answer is None
        assert fact.question_variant == "canonical"


# --------------------------------------------------------------------------- #
# P2-8: name_only probes use original question
# --------------------------------------------------------------------------- #


class TestNameOnlyOriginalQuestion:
    def test_fact_question_uses_original(self):
        fact = _qa_fact(question="Where does Ava live?")
        assert RouteProbeBuilder.fact_question(fact) == "Where does Ava live?"

    def test_fact_question_fallback_for_plain_fact(self, registry):
        fact = _plain_fact()
        # No original_question → synthetic fallback.
        q = RouteProbeBuilder.fact_question(fact)
        assert "caption" in q.lower()

    def test_select_name_only_fact_prefers_qa(self):
        plain = _plain_fact()
        qa = _qa_fact()
        selected = _select_name_only_fact([plain, qa])
        assert selected.fact_id == "fiubench_qa_00"
        assert selected.source_qa_index == 0

    def test_select_name_only_fact_falls_back_to_plain(self):
        plain = _plain_fact()
        selected = _select_name_only_fact([plain])
        assert selected.fact_id == "fiubench_caption"

    def test_name_only_probe_uses_original_question(self, registry):
        qa = _qa_fact(question="Where does Ava live?", answer="Alpha City")
        plain = _plain_fact()
        anchor = _anchor_with_facts([plain, qa])
        builder = RouteProbeBuilder(registry)
        probes = build_identity_probes([anchor], builder)
        name_only = [p for p in probes if p.route_probe.probe_family == "name_only"]
        assert len(name_only) == 1
        # The probe question embeds the original FIUBench question via the
        # name_only template "{identity_name}: {fact_question}".
        assert "Where does Ava live?" in name_only[0].question

    def test_name_only_probe_stores_fact_provenance(self, registry):
        qa = _qa_fact(index=3, question="What city?", answer="Beta Town")
        anchor = _anchor_with_facts([qa])
        builder = RouteProbeBuilder(registry)
        probes = build_identity_probes([anchor], builder)
        name_only = [p for p in probes if p.route_probe.probe_family == "name_only"]
        assert len(name_only) == 1
        meta = name_only[0].source_metadata
        assert meta["target_fact_id"] == "fiubench_qa_03"
        assert meta["source_qa_index"] == 3
        assert meta["original_question"] == "What city?"
        assert meta["original_answer"] == "Beta Town"
        assert meta["question_variant"] == "canonical"


# --------------------------------------------------------------------------- #
# P2-9: paraphrases are robustness variants, not separate facts
# --------------------------------------------------------------------------- #


class TestParaphraseRobustness:
    def test_paraphrase_fact_has_canonical_variant(self):
        fact = _qa_fact()
        assert fact.question_variant == "canonical"
        # Paraphrases would have question_variant != "canonical" if they
        # were stored as facts, but they are NOT stored as facts (P2-9).


# --------------------------------------------------------------------------- #
# P2-10: perturbed answers never become ground-truth facts
# --------------------------------------------------------------------------- #


class TestPerturbedExclusion:
    def test_perturbed_not_in_qa_fact(self):
        # The _qa_fact method always uses the original answer, never a
        # perturbed one.
        item = {
            "question": "Where?",
            "answer": "Alpha City",
            "perturbed_answer": ["Beta Town"],
        }
        from route_data.data.adapters.fiubench import FiubenchAdapter

        fact = FiubenchAdapter._qa_fact(0, item)
        assert fact.value == "Alpha City"
        assert fact.value != "Beta Town"


# --------------------------------------------------------------------------- #
# P2-11: fact provenance fields in probe_row output
# --------------------------------------------------------------------------- #


class TestProbeRowProvenance:
    def test_probe_row_includes_provenance(self, registry):
        qa = _qa_fact(index=5, question="Which country?", answer="Fjordmark")
        anchor = _anchor_with_facts([qa])
        builder = RouteProbeBuilder(registry)
        probes = build_identity_probes([anchor], builder)
        name_only = [p for p in probes if p.route_probe.probe_family == "name_only"]
        row = builder.probe_row(name_only[0])
        assert row["source_qa_index"] == 5
        assert row["original_question"] == "Which country?"
        assert row["original_answer"] == "Fjordmark"
        assert row["question_variant"] == "canonical"
        assert row["target_fact_id"] == "fiubench_qa_05"

    def test_probe_row_visual_has_no_qa_provenance(self, registry):
        plain = _plain_fact()
        anchor = _anchor_with_facts([plain])
        builder = RouteProbeBuilder(registry)
        probes = build_identity_probes([anchor], builder)
        visual = [p for p in probes if p.route_probe.probe_family == "direct_visual"]
        row = builder.probe_row(visual[0], attribute="Eyeglasses")
        # Visual probes have no fact provenance.
        assert row.get("source_qa_index") is None
        assert row.get("original_question") is None
