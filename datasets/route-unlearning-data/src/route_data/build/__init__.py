"""Benchmark extension build pipeline (plan sections 11-18).

Annotate benchmark images with the frozen CelebA protocol, generate static
visual QA and route probes, construct unlearning splits, and export auditable
artifacts + dataset cards.
"""

from .annotate import (
    CELEBA40_NAMESPACE,
    DEFAULT_BANDS,
    AnnotationError,
    AnnotationPolicy,
    BenchmarkAnnotator,
    celeba40_key,
    confidence_band,
    decision_from_probability,
    load_frozen_calibrators,
    predictions_to_scores,
    validate_bands,
)
from .conflict_generation import (
    PAIR_TYPES,
    ConflictError,
    RouteProbeBuilder,
    build_identity_probes,
    build_pair_manifest,
    conflict_claim_for,
    make_pair,
)
from .export import ExportRecord, ExtensionExporter
from .qa_generation import (
    PROTECTED_SPLITS,
    QaError,
    QaLeakageError,
    QaTemplate,
    QaTemplateRegistry,
    canonical_answer_text,
    generate_binary_qa,
    generate_qa_rows,
    normalize_binary_answer,
    normalize_choice_answer,
)
from .split_generation import (
    FORGET_SCOPES,
    SplitBuilder,
    SplitError,
    SplitResult,
    SplitSpec,
    validate_split_invariants,
)

__all__ = [
    # annotate
    "CELEBA40_NAMESPACE",
    "DEFAULT_BANDS",
    # splits
    "FORGET_SCOPES",
    # conflict / route probes
    "PAIR_TYPES",
    # qa generation
    "PROTECTED_SPLITS",
    "AnnotationError",
    "AnnotationPolicy",
    "BenchmarkAnnotator",
    "ConflictError",
    # export
    "ExportRecord",
    "ExtensionExporter",
    "QaError",
    "QaLeakageError",
    "QaTemplate",
    "QaTemplateRegistry",
    "RouteProbeBuilder",
    "SplitBuilder",
    "SplitError",
    "SplitResult",
    "SplitSpec",
    "build_identity_probes",
    "build_pair_manifest",
    "canonical_answer_text",
    "celeba40_key",
    "confidence_band",
    "conflict_claim_for",
    "decision_from_probability",
    "generate_binary_qa",
    "generate_qa_rows",
    "load_frozen_calibrators",
    "make_pair",
    "normalize_binary_answer",
    "normalize_choice_answer",
    "predictions_to_scores",
    "validate_bands",
    "validate_split_invariants",
]
