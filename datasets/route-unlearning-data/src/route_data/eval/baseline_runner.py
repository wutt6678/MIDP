"""FIUBench baseline runner (post-freeze Stages 2-3).

Evaluates the frozen 500-probe route-conflict dataset against the target
model *before* any unlearning.  Results establish the pre-unlearning
behavioural baseline that later stages compare against.

Design guarantees:
- **Deterministic**: ``do_sample=False``, pinned model revision, fixed seed.
- **Cacheable / resumable**: every scored probe is cached with a composite
  key (probe-id, model-revision, prompt-hash, image-hash, scoring-version);
  interrupted runs skip already-completed entries.
- **Fail-closed**: the probe-file SHA-256 is checked against the research
  dataset manifest before any inference; a mismatch aborts the run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import re
import string
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import ModelConfig
from ..models.base import VisionLanguageModel
from ..models.scoring import (
    CANDIDATE_PROTOCOL_VERSION,
    SCORING_VERSION,
    binary_probability,
)

log = logging.getLogger(__name__)

__all__ = [
    "BaselineProbe",
    "BaselineResult",
    "BaselineRunner",
    "select_smoke_probes",
]

# Candidate strings used for binary (Yes/No) probe families.
CANDIDATES: tuple[str, ...] = ("Yes", "No")
_POSITIVE = "Yes"
_NEGATIVE = "No"

# Probe families that carry an image and expect a binary answer.
_IMAGE_FAMILIES = frozenset({
    "direct_visual",
    "image_plus_name",
    "wrong_name",
    "visual_text_conflict",
})
_TEXT_ONLY_FAMILIES = frozenset({"name_only"})
ALL_FAMILIES = _IMAGE_FAMILIES | _TEXT_ONLY_FAMILIES

# Metric schema version for baseline-manifest provenance (P1-2).
METRIC_SCHEMA_VERSION = "baseline-metrics-v1"

# Frozen generation budget for name_only (P0-1).
_NAME_ONLY_MAX_NEW_TOKENS = 128


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BaselineProbe:
    """One frozen route-conflict probe from ``fiubench_route_conflict_eval.jsonl``."""

    probe_id: str
    sample_id: str
    identity_id: str
    benchmark: str
    probe_family: str
    modality: str
    question: str
    expected_evidence_source: str
    controlled_variables: list[str]
    image_uri: str | None
    image_sha256: str | None
    registry_hash: str
    target_attribute: str | None
    answer_label: bool | None
    answer_text: str

    # -- family-specific extras (populated from JSONL, ignored when absent) --
    # wrong_name
    matched_wrong_identity_id: str | None = None
    matching_similarity: float | None = None
    matching_attributes: list[str] | None = None
    candidate_rank: int | None = None
    matching_strategy: str | None = None
    # name_only
    target_fact_id: str | None = None
    target_fact_relation: str | None = None
    target_fact_value: str | None = None
    source_qa_index: int | None = None
    original_question: str | None = None
    original_answer: str | None = None
    question_variant: str | None = None
    paired_sample_id: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BaselineProbe:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def has_image(self) -> bool:
        return self.image_uri is not None


@dataclass
class BaselineResult:
    """One result row per probe (Stage 2.4 result schema).

    For image-bearing binary families the *candidate scoring* path fills
    ``logp_yes``, ``logp_no``, ``p_yes``, ``raw_log_margin``, and
    ``signed_answer_margin``.  For ``name_only`` (text-only, free-form
    answer) the *generation* path fills ``generated_answer`` and the
    text-match metrics; the probability fields stay ``None``.

    All probe metadata fields are carried through so downstream analysis
    can group by attribute, identity, family, or split without joining
    back to the source probe file.
    """

    # -- Probe identity (carried through from BaselineProbe) --
    probe_id: str
    sample_id: str
    identity_id: str
    probe_family: str
    modality: str
    question: str
    expected_evidence_source: str = ""
    controlled_variables: list[str] = field(default_factory=list)
    registry_hash: str = ""
    paired_sample_id: str | None = None

    # -- Route / attribute fields --
    target_attribute: str | None = None
    answer_label: bool | None = None
    answer_text: str = ""

    # -- Wrong-name fields --
    matched_wrong_identity_id: str | None = None
    matching_similarity: float | None = None
    matching_attributes: list[str] | None = None
    candidate_rank: int | None = None
    matching_strategy: str | None = None

    # -- Name-only fields --
    target_fact_id: str | None = None
    target_fact_relation: str | None = None
    target_fact_value: str | None = None
    source_qa_index: int | None = None
    question_variant: str | None = None
    original_question: str | None = None
    original_answer: str | None = None

    # -- Model identity --
    model_fingerprint: str = ""
    model_revision: str | None = None

    # -- Inputs --
    image_sha256: str | None = None

    # -- Outputs --
    generated_answer: str = ""
    parsed_answer: str | None = None
    predicted_label: str | None = None
    correct: bool | None = None

    # -- Candidate scores (binary families only) --
    logp_yes: float | None = None
    logp_no: float | None = None
    p_yes: float | None = None
    centered_p_yes: float | None = None
    raw_log_margin: float | None = None
    signed_answer_margin: float | None = None

    # -- Text-match scores (name_only only) --
    exact_match: float | None = None
    normalized_exact_match: float | None = None
    token_overlap: float | None = None
    fuzzy_match: float | None = None

    # -- Generation provenance (name_only only, P1-1) --
    generated_token_count: int | None = None
    hit_max_new_tokens: bool | None = None
    eos_reached: bool | None = None
    generation_max_new_tokens: int | None = None

    # -- Protocol role (P1-2) --
    protocol_role: str = ""  # train | eval | exclude

    # -- Provenance --
    scoring_version: str = ""
    prompt_hash: str = ""
    latency_ms: float = 0.0
    error: str | None = None
    # Immutable cache provenance (Commit 2 — P0-2).
    cache_key: str = ""
    route_probe_sha256: str = ""
    model_config_sha256: str = ""
    candidate_protocol_version: str = ""


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


class BaselineRunner:
    """Run the frozen 500-probe baseline evaluation.

    Parameters
    ----------
    backend:
        Instantiated :class:`VisionLanguageModel`.
    probe_path:
        Path to ``fiubench_route_conflict_eval.jsonl``.
    output_dir:
        Directory for ``baseline_results.jsonl`` and ``baseline_summary.json``.
    model_config:
        Validated model configuration (provides revision + fingerprint).
    resume:
        When *True*, load the on-disk cache and skip already-scored probes.
    dataset_manifest_path:
        Optional path to the research dataset manifest JSON.  When provided,
        :meth:`verify_input_hashes_from_manifest` reads the expected probe
        SHA-256 from the manifest instead of relying on a hard-coded literal.
    model_config_path:
        Optional path to the model config YAML.  When provided, its SHA-256
        is computed and incorporated into the cache key.
    """

    def __init__(
        self,
        backend: VisionLanguageModel,
        probe_path: str | Path,
        output_dir: str | Path,
        model_config: ModelConfig,
        resume: bool = True,
        dataset_manifest_path: str | Path | None = None,
        model_config_path: str | Path | None = None,
        freeze_verification_path: str | Path | None = None,
        processed_dataset_path: str | Path | None = None,
    ):
        self.backend = backend
        self.probe_path = Path(probe_path)
        self.output_dir = Path(output_dir)
        self.model_config = model_config
        self.resume = resume
        self.dataset_manifest_path = (
            Path(dataset_manifest_path) if dataset_manifest_path else None
        )
        self.model_config_path = (
            Path(model_config_path) if model_config_path else None
        )
        self.freeze_verification_path = (
            Path(freeze_verification_path) if freeze_verification_path else None
        )
        self.processed_dataset_path = (
            Path(processed_dataset_path) if processed_dataset_path else None
        )

        self._fingerprint = backend.fingerprint()
        self._fingerprint_id: str = str(
            self._fingerprint.get("fingerprint_id", "unknown")
        )
        self._cache_dir = self.output_dir / ".cache"

        # Compute model-config SHA from the source YAML (or serialised dataclass).
        self._model_config_sha: str = self._compute_model_config_sha()

        # Load the dataset manifest if provided.
        self._dataset_manifest: dict[str, Any] | None = (
            self._load_dataset_manifest() if self.dataset_manifest_path else None
        )

        # Load the freeze verification file if provided.
        self._freeze_verification: dict[str, Any] | None = (
            self._load_freeze_verification() if self.freeze_verification_path else None
        )

        # Build the identity → protocol-role map (P0-5 / P0-6).
        self.identity_role_map: dict[str, str] = self._build_identity_role_map()

        self.probes: list[BaselineProbe] = self.load_probes()
        self._results: list[BaselineResult] = (
            self._load_cache() if resume else []
        )

    # ------------------------------------------------------------------ #
    # Probe loading & verification
    # ------------------------------------------------------------------ #

    def load_probes(self) -> list[BaselineProbe]:
        """Load frozen probes from the route-conflict JSONL."""
        if not self.probe_path.is_file():
            raise FileNotFoundError(f"Probe file not found: {self.probe_path}")
        probes = [BaselineProbe.from_dict(d) for d in _read_jsonl(self.probe_path)]
        log.info("Loaded %d probes from %s", len(probes), self.probe_path)
        return probes

    def verify_input_hashes(self, expected_sha256: str) -> bool:
        """Check probe-file SHA-256 against the manifest.  Fail closed."""
        actual = _sha256_file(self.probe_path)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Probe file SHA-256 mismatch!\n"
                f"  expected: {expected_sha256}\n"
                f"  actual:   {actual}\n"
                "Refusing to run — the frozen dataset may have been modified."
            )
        log.info("Probe file hash verified: %s", actual[:16])
        return True

    def verify_input_hashes_from_manifest(self) -> bool:
        """Read the expected probe SHA-256 from the dataset manifest.

        The manifest must contain ``dataset_artifacts.route_probes.sha256``.
        Falls back to :meth:`verify_input_hashes` with the manifest value.
        """
        if self._dataset_manifest is None:
            raise RuntimeError(
                "No dataset manifest loaded.  Pass dataset_manifest_path to "
                "the constructor or call verify_input_hashes() directly."
            )
        expected = (
            self._dataset_manifest
            .get("dataset_artifacts", {})
            .get("route_probes", {})
            .get("sha256")
        )
        if not expected:
            raise RuntimeError(
                "Dataset manifest does not contain "
                "dataset_artifacts.route_probes.sha256"
            )
        return self.verify_input_hashes(expected)

    def validate_dataset_manifest(self) -> dict[str, Any]:
        """Validate the dataset manifest for full research baselines (P0-1).

        Uses the *actual* frozen schema:

        - ``definition_of_done.ready_for_experiments`` is true
        - ``dataset_artifacts.route_probes.total_probes`` is 500
        - Route SHA matches the probe file
        - Family counts match the frozen manifest

        When a freeze-verification file is also loaded, ``dataset_version``
        and ``ready_for_experiments`` are read from there instead.

        Returns
        -------
        dict
            Validation details.

        Raises
        ------
        RuntimeError
            If any check fails.
        """
        if self._dataset_manifest is None:
            raise RuntimeError(
                "Full research baseline requires --dataset-manifest."
            )
        m = self._dataset_manifest
        checks: dict[str, Any] = {}
        passed = True

        # -- ready_for_experiments -----------------------------------------
        # Prefer the freeze-verification file when available; fall back to
        # the dataset manifest's definition_of_done section.
        if self._freeze_verification:
            ready = self._freeze_verification.get("ready_for_experiments", False)
        else:
            ready = (
                m.get("definition_of_done", {})
                .get("ready_for_experiments", False)
            )
        checks["ready_for_experiments"] = ready
        if not ready:
            passed = False

        # -- dataset_version (from freeze-verification only) ---------------
        if self._freeze_verification:
            dv = self._freeze_verification.get("dataset_version", "")
        else:
            dv = "fiubench-route-v1"  # assumed when no freeze file
        checks["dataset_version"] = dv
        if dv != "fiubench-route-v1":
            passed = False

        # -- route SHA matches probe file ----------------------------------
        expected_sha = (
            m.get("dataset_artifacts", {})
            .get("route_probes", {})
            .get("sha256", "")
        )
        actual_sha = _sha256_file(self.probe_path)
        checks["route_sha_match"] = expected_sha == actual_sha
        if expected_sha != actual_sha:
            passed = False

        # -- route probe count == 500 (total_probes field) -----------------
        probe_count = (
            m.get("dataset_artifacts", {})
            .get("route_probes", {})
            .get("total_probes", 0)
        )
        checks["route_probe_count"] = probe_count
        if probe_count != 500:
            passed = False

        # -- JSONL row count == 500 ----------------------------------------
        jsonl_rows = len(_read_jsonl(self.probe_path))
        checks["jsonl_row_count"] = jsonl_rows
        if jsonl_rows != 500:
            passed = False

        # -- family counts -------------------------------------------------
        frozen_families = (
            m.get("dataset_artifacts", {})
            .get("route_probes", {})
            .get("families", {})
        )
        expected_families = {
            "direct_visual": 100,
            "image_plus_name": 100,
            "wrong_name": 100,
            "visual_text_conflict": 100,
            "name_only": 100,
        }
        checks["family_counts"] = frozen_families
        if frozen_families != expected_families:
            passed = False

        if not passed:
            raise RuntimeError(
                f"Dataset manifest validation failed: {checks}"
            )
        log.info("Dataset manifest validated: %s", checks)
        return checks

    def validate_freeze_verification(self) -> dict[str, Any]:
        """Validate the freeze-verification file for full research baselines.

        Checks all required gates:

        - ``dataset_version == "fiubench-route-v1"``
        - ``ready_for_experiments == true``
        - ``bundle_verifier_pass == true``
        - ``strict_final_verify_pass == true``
        - ``manual_audit_pass == true``
        - ``exact_ci_pass == true``
        - All ``hard_stop_conditions`` are true

        Returns
        -------
        dict
            Validation details.

        Raises
        ------
        RuntimeError
            If any check fails or no freeze-verification file is loaded.
        """
        if self._freeze_verification is None:
            raise RuntimeError(
                "Full research baseline requires --freeze-verification."
            )
        fv = self._freeze_verification
        checks: dict[str, Any] = {}
        passed = True

        # -- top-level gates -----------------------------------------------
        for gate in (
            "bundle_verifier_pass",
            "strict_final_verify_pass",
            "manual_audit_pass",
            "exact_ci_pass",
            "ready_for_experiments",
        ):
            val = fv.get(gate, False)
            checks[gate] = val
            if not val:
                passed = False

        # -- dataset_version -----------------------------------------------
        dv = fv.get("dataset_version", "")
        checks["dataset_version"] = dv
        if dv != "fiubench-route-v1":
            passed = False

        # -- hard_stop_conditions ------------------------------------------
        hsc = fv.get("hard_stop_conditions", {})
        for key in (
            "manual_audit_matches_current_route_artifact",
            "manual_audit_route_count_matches",
            "all_artifact_hashes_verified",
            "all_commits_reachable",
            "git_dirty_false",
        ):
            val = hsc.get(key, False)
            checks[f"hard_stop.{key}"] = val
            if not val:
                passed = False

        if not passed:
            raise RuntimeError(
                f"Freeze verification failed: {checks}"
            )
        log.info("Freeze verification validated: all gates pass")
        return checks

    def validate_cross_file(self) -> dict[str, Any]:
        """Cross-file hash validation between freeze, manifest, and probe file.

        Validates:

        1. ``dataset_manifest_sha256`` in freeze-verification matches the
           actual SHA-256 of the dataset manifest file.
        2. ``route_probe_sha256`` agrees across both freeze files and the
           actual probe JSONL (three-way agreement).
        3. Route probe count in the manifest is 500.

        Returns
        -------
        dict
            Cross-file validation details.

        Raises
        ------
        RuntimeError
            If any check fails.
        """
        if self._freeze_verification is None:
            raise RuntimeError(
                "Cross-file validation requires --freeze-verification."
            )
        if self._dataset_manifest is None:
            raise RuntimeError(
                "Cross-file validation requires --dataset-manifest."
            )
        fv = self._freeze_verification
        m = self._dataset_manifest
        checks: dict[str, Any] = {}
        passed = True

        # -- dataset manifest SHA ------------------------------------------
        expected_manifest_sha = fv.get("dataset_manifest_sha256", "")
        actual_manifest_sha = _sha256_file(self.dataset_manifest_path)
        checks["dataset_manifest_sha_match"] = (
            expected_manifest_sha == actual_manifest_sha
        )
        checks["dataset_manifest_sha_expected"] = expected_manifest_sha
        checks["dataset_manifest_sha_actual"] = actual_manifest_sha
        if expected_manifest_sha != actual_manifest_sha:
            passed = False

        # -- route SHA three-way agreement ---------------------------------
        fv_route_sha = fv.get("route_probe_sha256", "")
        manifest_route_sha = (
            m.get("dataset_artifacts", {})
            .get("route_probes", {})
            .get("sha256", "")
        )
        actual_route_sha = _sha256_file(self.probe_path)
        three_way = fv_route_sha == manifest_route_sha == actual_route_sha
        checks["route_sha_three_way"] = three_way
        checks["route_sha_freeze"] = fv_route_sha
        checks["route_sha_manifest"] = manifest_route_sha
        checks["route_sha_actual"] = actual_route_sha
        if not three_way:
            passed = False

        # -- route count ---------------------------------------------------
        route_count = (
            m.get("dataset_artifacts", {})
            .get("route_probes", {})
            .get("total_probes", 0)
        )
        checks["route_probe_count"] = route_count
        if route_count != 500:
            passed = False

        if not passed:
            raise RuntimeError(
                f"Cross-file validation failed: {checks}"
            )
        log.info("Cross-file validation passed: freeze ↔ manifest ↔ probe")
        return checks

    # ------------------------------------------------------------------ #
    # Processed-dataset validation (P0-2)
    # ------------------------------------------------------------------ #

    def validate_processed_dataset(self) -> dict[str, Any]:
        """Validate the processed dataset against the frozen manifest (P0-2).

        Computes SHA-256 of the processed dataset file and compares it
        against ``dataset_artifacts.processed_dataset.sha256`` from the
        research dataset manifest.

        Returns
        -------
        dict
            Validation details including SHA match status.

        Raises
        ------
        RuntimeError
            If the processed dataset is not provided, not found, or its
            SHA-256 does not match the frozen evidence.
        """
        if not self.processed_dataset_path:
            raise RuntimeError(
                "Processed dataset validation requires --processed-dataset."
            )
        if not self.processed_dataset_path.is_file():
            raise RuntimeError(
                f"Processed dataset not found: {self.processed_dataset_path}"
            )
        if self._dataset_manifest is None:
            raise RuntimeError(
                "Processed dataset validation requires --dataset-manifest."
            )

        checks: dict[str, Any] = {}
        passed = True

        # -- file existence -----------------------------------------------
        checks["processed_dataset_exists"] = True

        # -- SHA-256 match ------------------------------------------------
        actual_sha = _sha256_file(self.processed_dataset_path)
        expected_sha = (
            self._dataset_manifest
            .get("dataset_artifacts", {})
            .get("processed_dataset", {})
            .get("sha256", "")
        )
        sha_match = actual_sha == expected_sha
        checks["processed_dataset_sha_match"] = sha_match
        checks["processed_dataset_sha_expected"] = expected_sha
        checks["processed_dataset_sha_actual"] = actual_sha
        if not sha_match:
            passed = False

        # -- secondary counts (advisory) ----------------------------------
        proc_section = (
            self._dataset_manifest
            .get("dataset_artifacts", {})
            .get("processed_dataset", {})
        )
        for count_key in ("canonical_samples", "unique_images", "unique_identities"):
            expected_count = proc_section.get(count_key)
            if expected_count is not None:
                checks[f"processed_dataset_{count_key}"] = expected_count

        if not passed:
            raise RuntimeError(
                f"Processed dataset validation failed: {checks}"
            )
        log.info("Processed dataset validation passed: SHA match confirmed")
        return checks

    # ------------------------------------------------------------------ #
    # Identity → protocol-role mapping (P0-5 / P0-6)
    # ------------------------------------------------------------------ #

    def _build_identity_role_map(self) -> dict[str, str]:
        """Build a deterministic ``identity_id → protocol_role`` map.

        The map is constructed by scanning the processed dataset JSONL
        (``fiubench_processed.jsonl``) to extract each identity's
        ``source_subject_id`` and ``official_memberships``, then resolving
        the role via :func:`resolve_protocol_role` using the protocol
        configuration from the dataset manifest.

        Returns
        -------
        dict[str, str]
            Mapping of ``identity_id`` to one of ``"train"``, ``"eval"``,
            or ``"exclude"``.

        Raises
        ------
        RuntimeError
            If the processed dataset path is not provided or the protocol
            configuration is unavailable.
        """
        if not self.processed_dataset_path:
            log.warning(
                "No processed_dataset_path provided; identity_role_map will "
                "be empty.  Pass processed_dataset_path to enable "
                "protocol-role population."
            )
            return {}

        if not self.processed_dataset_path.is_file():
            raise FileNotFoundError(
                f"Processed dataset not found: {self.processed_dataset_path}"
            )

        # Extract the protocol config from the dataset manifest.
        protocol = self._extract_protocol()
        if protocol is None:
            raise RuntimeError(
                "Cannot build identity_role_map: no protocol configuration "
                "available in the dataset manifest."
            )

        from ..data.split_mapping import resolve_protocol_role

        # First pass: collect unique identity_id → (source_subject_id, memberships).
        identity_info: dict[str, tuple[str | None, list[str]]] = {}
        with open(self.processed_dataset_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                iid = row.get("identity_id", "")
                if not iid or iid in identity_info:
                    continue
                smeta = row.get("source_metadata", {})
                subject_id = smeta.get("source_subject_id")
                memberships = smeta.get("official_memberships", [])
                identity_info[iid] = (subject_id, memberships)

        log.info(
            "Extracted %d unique identities from processed dataset %s",
            len(identity_info), self.processed_dataset_path,
        )

        # Second pass: resolve each identity's protocol role.
        role_map: dict[str, str] = {}
        for iid, (subject_id, memberships) in identity_info.items():
            role = resolve_protocol_role(
                memberships, protocol, source_subject_id=subject_id,
            )
            role_map[iid] = role

        # Log role distribution.
        role_counts: dict[str, int] = {}
        for role in role_map.values():
            role_counts[role] = role_counts.get(role, 0) + 1
        log.info("Identity role distribution: %s", role_counts)

        return role_map

    def _extract_protocol(self) -> dict[str, Any] | None:
        """Extract the protocol config dict from the dataset manifest.

        Returns a dict suitable for :func:`resolve_protocol_role` with
        keys ``forget_bucket``, ``train_bucket``, ``eval_bucket``,
        ``eval_fraction``, ``eval_seed``.
        """
        if not self._dataset_manifest:
            return None
        proto_section = self._dataset_manifest.get("protocol")
        if not proto_section:
            return None
        # The manifest may nest under "canonical_protocol".
        canonical = proto_section.get("canonical_protocol", proto_section)
        return {
            "forget_bucket": canonical.get("forget_bucket"),
            "train_bucket": canonical.get("train_bucket"),
            "eval_bucket": canonical.get("eval_bucket"),
            "eval_fraction": canonical.get("eval_fraction", 0.0),
            "eval_seed": canonical.get("eval_seed", 0),
        }

    def _route_identity_role_counts(self) -> dict[str, int]:
        """Count unique frozen route identities by protocol role.

        The count is derived from ``self.probes`` (the frozen route-evaluation
        artifact), *not* from the processed-dataset population.  Each unique
        ``identity_id`` appearing in the probes is looked up in
        ``self.identity_role_map`` and tallied under its protocol role.

        Returns
        -------
        dict[str, int]
            ``{"train": n_train, "eval": n_eval, "exclude": n_exclude}``

        Raises
        ------
        RuntimeError
            If any probe identity has a missing or out-of-protocol role.
        """
        route_identity_ids = {
            probe.identity_id
            for probe in self.probes
        }

        counts: dict[str, int] = {
            "train": 0,
            "eval": 0,
            "exclude": 0,
        }

        for identity_id in route_identity_ids:
            role = self.identity_role_map.get(identity_id)

            if role not in counts:
                raise RuntimeError(
                    f"Route identity {identity_id} has invalid or "
                    f"missing role: {role}"
                )

            counts[role] += 1

        return counts

    # ------------------------------------------------------------------ #
    # Manifest / config helpers
    # ------------------------------------------------------------------ #

    def _load_dataset_manifest(self) -> dict[str, Any] | None:
        """Load and return the dataset manifest JSON, or *None* on failure."""
        if not self.dataset_manifest_path or not self.dataset_manifest_path.is_file():
            log.warning(
                "Dataset manifest not found: %s", self.dataset_manifest_path
            )
            return None
        with open(self.dataset_manifest_path) as f:
            manifest = json.load(f)
        log.info("Loaded dataset manifest from %s", self.dataset_manifest_path)
        return manifest

    def _load_freeze_verification(self) -> dict[str, Any] | None:
        """Load and return the freeze-verification JSON, or *None*."""
        if (
            not self.freeze_verification_path
            or not self.freeze_verification_path.is_file()
        ):
            log.warning(
                "Freeze verification not found: %s",
                self.freeze_verification_path,
            )
            return None
        with open(self.freeze_verification_path) as f:
            fv = json.load(f)
        log.info(
            "Loaded freeze verification from %s",
            self.freeze_verification_path,
        )
        return fv

    def _compute_model_config_sha(self) -> str:
        """SHA-256 of the model config YAML (or serialised dataclass)."""
        if self.model_config_path and self.model_config_path.is_file():
            return _sha256_file(self.model_config_path)
        # Fallback: hash the serialised dataclass.
        raw = json.dumps(asdict(self.model_config), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------ #
    # Git-state helpers (P0-10 / P0-11)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_git_state() -> dict[str, Any]:
        """Return ``{git_commit, git_dirty}`` from the working tree."""
        git_commit = ""
        git_dirty = False
        try:
            import subprocess
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            git_dirty = bool(status)
        except Exception:
            pass
        return {"git_commit": git_commit, "git_dirty": git_dirty}

    def require_clean_git(self) -> dict[str, Any]:
        """Abort if the Git tree is dirty (P0-10).

        For a full research baseline the tree must be clean and the
        commit SHA must be non-empty.

        Returns
        -------
        dict
            ``{git_commit, git_dirty}``.

        Raises
        ------
        RuntimeError
            If the tree is dirty or the commit SHA is empty.
        """
        state = self._get_git_state()
        if not state["git_commit"]:
            raise RuntimeError(
                "Cannot determine Git commit SHA.  "
                "Full research baseline requires a Git repository."
            )
        if state["git_dirty"]:
            raise RuntimeError(
                "Git tree is dirty.  Full research baseline requires "
                "a clean working tree (git_dirty == false)."
            )
        log.info("Git state clean: %s", state["git_commit"][:12])
        return state

    # ------------------------------------------------------------------ #
    # Fingerprint validation (P1-9)
    # ------------------------------------------------------------------ #

    def validate_fingerprint(self) -> dict[str, Any]:
        """Validate the model fingerprint before inference (P0-4, P0-5).

        Requires:

        - Model revision matches the config revision (P0-4: uses ``revision``
          field, not ``model_revision``).
        - Fingerprint is non-empty.
        - Model-config SHA is non-empty.
        - Processor / tokenizer / chat-template fields are present (P0-5:
          ``processor_class``, ``tokenizer_class``, ``chat_template_hash``).

        Returns
        -------
        dict
            Validation details.

        Raises
        ------
        RuntimeError
            If any check fails.
        """
        checks: dict[str, Any] = {}
        passed = True

        # -- revision match (P0-4) ---------------------------------------
        backend_rev = self._fingerprint.get("revision", "")
        config_rev = self.model_config.revision or ""
        # Require both non-empty and equal for production Qwen baseline.
        if config_rev and not backend_rev:
            rev_match = False
        elif backend_rev and config_rev:
            rev_match = backend_rev == config_rev
        else:
            rev_match = False
        checks["revision_match"] = rev_match
        checks["backend_revision"] = backend_rev
        checks["config_revision"] = config_rev
        if not rev_match:
            passed = False

        # -- fingerprint non-empty ---------------------------------------
        fp_ok = self._fingerprint_id and self._fingerprint_id != "unknown"
        checks["fingerprint_non_empty"] = fp_ok
        checks["fingerprint_id"] = self._fingerprint_id
        if not fp_ok:
            passed = False

        # -- model config SHA non-empty ----------------------------------
        sha_ok = bool(self._model_config_sha)
        checks["model_config_sha_non_empty"] = sha_ok
        if not sha_ok:
            passed = False

        # -- processor / tokenizer / chat-template fields (P0-5) ---------
        required_fp_fields = (
            "processor_class",
            "tokenizer_class",
            "chat_template_hash",
        )
        missing_fp_fields = [
            k for k in required_fp_fields if not self._fingerprint.get(k)
        ]
        has_processor = len(missing_fp_fields) == 0
        checks["processor_tokenizer_available"] = has_processor
        checks["missing_fingerprint_fields"] = missing_fp_fields
        if not has_processor:
            passed = False

        if not passed:
            raise RuntimeError(
                f"Fingerprint validation failed: {checks}"
            )
        log.info("Fingerprint validation passed: %s", self._fingerprint_id)
        return checks

    # ------------------------------------------------------------------ #
    # Research preflight (P1-8 / P1-9)
    # ------------------------------------------------------------------ #

    def validate_research_preflight(self) -> dict[str, Any]:
        """Run all pre-inference validation gates in order (P1-8).

        Sequence:
        1. ``validate_freeze_verification()``
        2. ``validate_dataset_manifest()``
        3. ``validate_cross_file()``
        4. ``validate_processed_dataset()``
        5. ``validate_fingerprint()``
        6. ``require_clean_git()``

        Writes ``preflight_report.json`` to the output directory (P1-9).

        Returns
        -------
        dict
            Combined report with per-gate results.

        Raises
        ------
        RuntimeError
            If any gate fails.  No probes will have been evaluated.
        """
        report: dict[str, Any] = {"pass": False}
        gates: list[tuple[str, Any]] = []

        # 1. Freeze verification
        try:
            freeze_result = self.validate_freeze_verification()
            gates.append(("freeze", freeze_result))
        except RuntimeError as exc:
            gates.append(("freeze", {"error": str(exc)}))
            report["gates"] = dict(gates)
            self._write_preflight_report(report)
            raise

        # 2. Dataset manifest
        try:
            manifest_result = self.validate_dataset_manifest()
            gates.append(("dataset", manifest_result))
        except RuntimeError as exc:
            gates.append(("dataset", {"error": str(exc)}))
            report["gates"] = dict(gates)
            self._write_preflight_report(report)
            raise

        # 3. Cross-file
        try:
            cross_result = self.validate_cross_file()
            gates.append(("cross_file", cross_result))
        except RuntimeError as exc:
            gates.append(("cross_file", {"error": str(exc)}))
            report["gates"] = dict(gates)
            self._write_preflight_report(report)
            raise

        # 4. Processed dataset
        try:
            proc_result = self.validate_processed_dataset()
            gates.append(("processed_dataset", proc_result))
        except RuntimeError as exc:
            gates.append(("processed_dataset", {"error": str(exc)}))
            report["gates"] = dict(gates)
            self._write_preflight_report(report)
            raise

        # 5. Fingerprint
        try:
            fp_result = self.validate_fingerprint()
            gates.append(("fingerprint", fp_result))
        except RuntimeError as exc:
            gates.append(("fingerprint", {"error": str(exc)}))
            report["gates"] = dict(gates)
            self._write_preflight_report(report)
            raise

        # 6. Clean Git
        try:
            git_result = self.require_clean_git()
            gates.append(("git", git_result))
        except RuntimeError as exc:
            gates.append(("git", {"error": str(exc)}))
            report["gates"] = dict(gates)
            self._write_preflight_report(report)
            raise

        report["gates"] = dict(gates)
        report["pass"] = True
        self._write_preflight_report(report)
        log.info("Research preflight passed: all 6 gates cleared.")
        return report

    def _write_preflight_report(self, report: dict[str, Any]) -> None:
        """Write ``preflight_report.json`` to the output directory (P1-9)."""
        report_path = self.output_dir / "preflight_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info("Preflight report written: %s", report_path)

    # ------------------------------------------------------------------ #
    # Cache / resume
    # ------------------------------------------------------------------ #

    @property
    def _cache_path(self) -> Path:
        return self._cache_dir / "baseline_cache.jsonl"

    def _generation_protocol_hash(self) -> str:
        """SHA-256 of the frozen generation protocol.

        Incorporates both the binary-family and name_only generation
        settings so that any change (e.g. max_new_tokens 4→64)
        produces a distinct protocol hash and invalidates the cache.
        """
        payload = {
            "binary_families": {
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": self.model_config.generation.max_new_tokens,
            },
            "name_only": {
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": _NAME_ONLY_MAX_NEW_TOKENS,
            },
            "scoring_version": SCORING_VERSION,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _family_generation_hash(self, probe_family: str) -> str:
        """SHA-256 of generation settings for a specific probe family.

        Unlike ``_generation_protocol_hash`` (which covers all families),
        this returns a hash covering only the settings relevant to
        *probe_family*.  This allows changing name_only budget without
        invalidating cached binary results (and vice versa).
        """
        if probe_family in _TEXT_ONLY_FAMILIES:
            payload = {
                "family": probe_family,
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": _NAME_ONLY_MAX_NEW_TOKENS,
                "scoring_version": SCORING_VERSION,
            }
        else:
            payload = {
                "family": probe_family,
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": self.model_config.generation.max_new_tokens,
                "scoring_version": SCORING_VERSION,
            }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_key(self, probe: BaselineProbe) -> str:
        """Composite cache key for deterministic resumption.

        The key includes the probe identity, model revision, prompt hash,
        image hash, scoring version, generation protocol hash, and — when
        available — the route-probe SHA from the dataset manifest and the
        model-config SHA so that any change to the dataset, model config,
        or generation settings invalidates the cache.
        """
        parts = [
            probe.probe_id,
            self.model_config.revision or "none",
            _prompt_hash(probe.question),
            probe.image_sha256 or "none",
            SCORING_VERSION,
        ]
        # Stronger key: include dataset and model config provenance.
        route_sha = self._route_probe_sha()
        parts.append(route_sha)
        parts.append(self._model_config_sha)  # full 64-char SHA
        # P0-1 final: family-specific generation settings prevent old
        # results from being silently reused after a budget change.
        # Using per-family settings (not the full protocol hash) ensures
        # that changing name_only budget does not invalidate binary results.
        parts.append(self._family_generation_hash(probe.probe_family))
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _load_cache(self) -> list[BaselineResult]:
        """Load previously completed results from the cache file.

        Rows whose immutable provenance does not match the current runner
        configuration are rejected (skipped with a warning).
        """
        if not self._cache_path.is_file():
            return []
        results: list[BaselineResult] = []
        rejected = 0
        current_route_sha = self._route_probe_sha()
        current_config_sha = self._model_config_sha
        current_fp = self._fingerprint_id
        for doc in _read_jsonl(self._cache_path):
            row = BaselineResult(**doc)
            if not self._cache_row_compatible(
                row, current_route_sha, current_config_sha, current_fp,
            ):
                rejected += 1
                continue
            results.append(row)
        if rejected:
            log.warning(
                "Rejected %d incompatible cache rows from %s",
                rejected, self._cache_path,
            )
        log.info("Loaded %d cached results from %s", len(results), self._cache_path)
        return results

    def _done_keys(self) -> set[str]:
        return {r.cache_key for r in self._results if r.cache_key}

    def _cache_row_compatible(
        self,
        row: BaselineResult,
        current_route_sha: str,
        current_config_sha: str,
        current_fp: str,
    ) -> bool:
        """Return True if a cached row's immutable provenance matches.

        Rows with missing provenance fields are rejected (P1-3).
        """
        # P1-3: reject rows with missing provenance fields.
        required = (
            row.cache_key,
            row.route_probe_sha256,
            row.model_config_sha256,
            row.model_fingerprint,
            row.scoring_version,
            row.candidate_protocol_version,
        )
        if not all(required):
            return False
        if row.route_probe_sha256 != current_route_sha:
            return False
        if row.model_config_sha256 != current_config_sha:
            return False
        if row.model_fingerprint != current_fp:
            return False
        if row.scoring_version != SCORING_VERSION:
            return False
        return row.candidate_protocol_version == CANDIDATE_PROTOCOL_VERSION

    def _route_probe_sha(self) -> str:
        """Return the route-probe SHA from the dataset manifest (or 'none')."""
        if self._dataset_manifest:
            return (
                self._dataset_manifest
                .get("dataset_artifacts", {})
                .get("route_probes", {})
                .get("sha256", "none")
            )
        return "none"

    # ------------------------------------------------------------------ #
    # Single-probe execution
    # ------------------------------------------------------------------ #

    def run_probe(self, probe: BaselineProbe) -> BaselineResult:
        """Run deterministic inference + scoring for one probe."""
        fp = self._fingerprint_id
        rev = self.model_config.revision
        ph = _prompt_hash(probe.question)
        cache_key = self._cache_key(probe)
        route_sha = self._route_probe_sha()
        started = time.perf_counter()

        generated_answer = ""
        parsed_answer: str | None = None
        predicted_label: str | None = None
        correct: bool | None = None
        logp_yes: float | None = None
        logp_no: float | None = None
        p_yes: float | None = None
        centered_p_yes: float | None = None
        raw_log_margin: float | None = None
        signed_answer_margin: float | None = None
        exact_match: float | None = None
        normalized_exact_match: float | None = None
        token_overlap_score: float | None = None
        fuzzy_match: float | None = None
        # Generation provenance (P1-1)
        generated_token_count: int | None = None
        hit_max_new_tokens: bool | None = None
        eos_reached: bool | None = None
        generation_max_new_tokens: int | None = None
        error: str | None = None

        try:
            if probe.has_image:
                image = _load_image(probe.image_uri)
                resp = self.backend.score_candidates(
                    image, probe.question, list(CANDIDATES)
                )
                generated_answer = resp.text or ""
                scores = {
                    c.candidate: c.log_probability
                    for c in (resp.candidate_scores or [])
                }
                if _POSITIVE in scores and _NEGATIVE in scores:
                    logp_yes = scores[_POSITIVE]
                    logp_no = scores[_NEGATIVE]
                    p_yes = binary_probability(logp_yes, logp_no)
                    centered_p_yes = p_yes - 0.5
                    raw_log_margin = logp_yes - logp_no
                    predicted_label = _POSITIVE if p_yes >= 0.5 else _NEGATIVE
                    # signed_answer_margin: positive = correct side preferred
                    if probe.answer_label is not None:
                        signed_answer_margin = (
                            raw_log_margin
                            if probe.answer_label
                            else -raw_log_margin
                        )
                        correct = (predicted_label == _POSITIVE) == probe.answer_label
                    else:
                        signed_answer_margin = raw_log_margin
                else:
                    error = "backend did not return scores for both candidates"
            else:
                # name_only: text-only generation with larger budget (P0-1)
                generation_max_new_tokens = _NAME_ONLY_MAX_NEW_TOKENS
                resp = self.backend.generate(
                    None, probe.question,
                    max_new_tokens=_NAME_ONLY_MAX_NEW_TOKENS,
                )
                generated_answer = resp.text or ""
                parsed_answer = generated_answer.strip()
                # Extract generation provenance from response metadata (P1-1)
                gen_meta = resp.metadata or {}
                generated_token_count = gen_meta.get("generated_token_count")
                hit_max_new_tokens = gen_meta.get("hit_max_new_tokens")
                eos_reached = gen_meta.get("eos_reached")
                if probe.answer_text:
                    exact_match = _compute_exact_match(
                        generated_answer, probe.answer_text
                    )
                    normalized_exact_match = _compute_normalized_exact_match(
                        generated_answer, probe.answer_text
                    )
                    token_overlap_score = _compute_token_overlap(
                        generated_answer, probe.answer_text
                    )
                    fuzzy_match = _text_match(
                        generated_answer, probe.answer_text
                    )
                    # Primary metric: token_overlap (P1-3)
                    # Use token_overlap > 0.5 as the correctness threshold
                    correct = (token_overlap_score > 0.5)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.warning("Probe %s failed: %s", probe.probe_id, error)

        latency = (time.perf_counter() - started) * 1000.0

        # Protocol role (P0-7).
        if self.identity_role_map:
            role = self.identity_role_map.get(probe.identity_id, "")
            if not role:
                raise RuntimeError(
                    f"Identity {probe.identity_id} not found in "
                    f"identity_role_map.  Cannot populate protocol_role."
                )
            if role not in ("train", "eval", "exclude"):
                raise RuntimeError(
                    f"Identity {probe.identity_id} has invalid role "
                    f"{role!r}.  Expected one of: train, eval, exclude."
                )
        else:
            role = ""

        return BaselineResult(
            # Probe identity
            probe_id=probe.probe_id,
            sample_id=probe.sample_id,
            identity_id=probe.identity_id,
            probe_family=probe.probe_family,
            modality=probe.modality,
            question=probe.question,
            expected_evidence_source=probe.expected_evidence_source,
            controlled_variables=list(probe.controlled_variables),
            registry_hash=probe.registry_hash,
            paired_sample_id=probe.paired_sample_id,
            # Route / attribute
            target_attribute=probe.target_attribute,
            answer_label=probe.answer_label,
            answer_text=probe.answer_text,
            # Wrong-name fields
            matched_wrong_identity_id=probe.matched_wrong_identity_id,
            matching_similarity=probe.matching_similarity,
            matching_attributes=(
                list(probe.matching_attributes) if probe.matching_attributes else None
            ),
            candidate_rank=probe.candidate_rank,
            matching_strategy=probe.matching_strategy,
            # Name-only fields
            target_fact_id=probe.target_fact_id,
            target_fact_relation=probe.target_fact_relation,
            target_fact_value=probe.target_fact_value,
            source_qa_index=probe.source_qa_index,
            question_variant=probe.question_variant,
            original_question=probe.original_question,
            original_answer=probe.original_answer,
            # Model identity
            model_fingerprint=fp,
            model_revision=rev,
            # Inputs
            image_sha256=probe.image_sha256,
            # Outputs
            generated_answer=generated_answer,
            parsed_answer=parsed_answer,
            predicted_label=predicted_label,
            correct=correct,
            # Candidate scores
            logp_yes=logp_yes,
            logp_no=logp_no,
            p_yes=p_yes,
            centered_p_yes=centered_p_yes,
            raw_log_margin=raw_log_margin,
            signed_answer_margin=signed_answer_margin,
            # Text-match scores
            exact_match=exact_match,
            normalized_exact_match=normalized_exact_match,
            token_overlap=token_overlap_score,
            fuzzy_match=fuzzy_match,
            # Generation provenance (P1-1)
            generated_token_count=generated_token_count,
            hit_max_new_tokens=hit_max_new_tokens,
            eos_reached=eos_reached,
            generation_max_new_tokens=generation_max_new_tokens,
            # Protocol role (P0-7)
            protocol_role=role,
            # Provenance
            scoring_version=SCORING_VERSION,
            prompt_hash=ph,
            latency_ms=latency,
            error=error,
            # Immutable cache provenance
            cache_key=cache_key,
            route_probe_sha256=route_sha,
            model_config_sha256=self._model_config_sha,
            candidate_protocol_version=CANDIDATE_PROTOCOL_VERSION,
        )

    # ------------------------------------------------------------------ #
    # Batch execution
    # ------------------------------------------------------------------ #

    def run_all(self, limit: int | None = None) -> list[BaselineResult]:
        """Iterate all probes with progress logging and cache/resume."""
        probes = self.probes if limit is None else self.probes[:limit]
        return self._run_probes(probes)

    def run_selected(self, probes: list[BaselineProbe]) -> list[BaselineResult]:
        """Run exactly the specified probes with cache/resume."""
        return self._run_probes(probes)

    def _run_probes(self, probes: list[BaselineProbe]) -> list[BaselineResult]:
        """Internal: iterate probes with progress logging and cache/resume."""
        done = self._done_keys()
        new_count = 0

        for idx, probe in enumerate(probes):
            key = self._cache_key(probe)
            if key in done:
                continue
            result = self.run_probe(probe)
            self._results.append(result)
            self._append_cache(result)
            done.add(key)
            new_count += 1
            if new_count % 50 == 0:
                log.info(
                    "Baseline progress: %d/%d new results",
                    new_count,
                    len(probes),
                )

        log.info(
            "Baseline complete: %d new, %d total (of %d probes)",
            new_count,
            len(self._results),
            len(probes),
        )
        return self._results

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save_results(self) -> Path:
        """Write ``baseline_results.jsonl`` and return its path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "baseline_results.jsonl"
        rows = [asdict(r) for r in self._results]
        with open(path, "w") as f:
            f.writelines(json.dumps(row, default=str) + "\n" for row in rows)
        log.info("Wrote %d results to %s", len(rows), path)
        return path

    def generate_summary(self) -> dict[str, Any]:
        """Compute and write ``baseline_summary.json``.

        Per-family metrics are reported separately because the five probe
        families are heterogeneous.  A mixed overall accuracy is provided
        only as a secondary convenience metric.
        """
        results = self._results
        by_family: dict[str, list[BaselineResult]] = {}
        for r in results:
            by_family.setdefault(r.probe_family, []).append(r)

        per_family: dict[str, Any] = {}
        for fam in sorted(by_family):
            fam_results = by_family[fam]
            n = len(fam_results)
            n_errors = sum(1 for r in fam_results if r.error is not None)
            entry: dict[str, Any] = {
                "count": n,
                "errors": n_errors,
            }
            if fam in _IMAGE_FAMILIES:
                # Binary route metrics
                n_correct = sum(1 for r in fam_results if r.correct is True)
                margins = [
                    r.signed_answer_margin
                    for r in fam_results
                    if r.signed_answer_margin is not None
                ]
                p_vals = [r.p_yes for r in fam_results if r.p_yes is not None]
                pos_correct = sum(
                    1
                    for r in fam_results
                    if r.answer_label is True and r.correct is True
                )
                neg_correct = sum(
                    1
                    for r in fam_results
                    if r.answer_label is False and r.correct is True
                )
                pos_total = sum(
                    1 for r in fam_results if r.answer_label is True
                )
                neg_total = sum(
                    1 for r in fam_results if r.answer_label is False
                )
                entry["correct"] = n_correct
                entry["accuracy"] = n_correct / n if n else None
                if margins:
                    entry["mean_signed_answer_margin"] = (
                        sum(margins) / len(margins)
                    )
                    sorted_m = sorted(margins)
                    mid = len(sorted_m) // 2
                    entry["median_signed_answer_margin"] = (
                        sorted_m[mid]
                        if len(sorted_m) % 2
                        else (sorted_m[mid - 1] + sorted_m[mid]) / 2
                    )
                if p_vals:
                    entry["mean_p_yes"] = sum(p_vals) / len(p_vals)
                if pos_total:
                    entry["positive_target_accuracy"] = (
                        pos_correct / pos_total
                    )
                if neg_total:
                    entry["negative_target_accuracy"] = (
                        neg_correct / neg_total
                    )
                entry["positive_count"] = pos_total
                entry["negative_count"] = neg_total
            elif fam in _TEXT_ONLY_FAMILIES:
                # Name-only metrics
                nem_vals = [
                    r.normalized_exact_match
                    for r in fam_results
                    if r.normalized_exact_match is not None
                ]
                to_vals = [
                    r.token_overlap
                    for r in fam_results
                    if r.token_overlap is not None
                ]
                fm_vals = [
                    r.fuzzy_match
                    for r in fam_results
                    if r.fuzzy_match is not None
                ]
                n_correct = sum(1 for r in fam_results if r.correct is True)
                non_empty = sum(
                    1
                    for r in fam_results
                    if r.generated_answer.strip()
                )
                # P1-1: cap hit rate (fraction hitting max_new_tokens)
                cap_hits = sum(
                    1 for r in fam_results if r.hit_max_new_tokens is True
                )
                entry["correct"] = n_correct
                entry["normalized_exact_match_rate"] = (
                    sum(nem_vals) / len(nem_vals) if nem_vals else None
                )
                entry["mean_token_overlap"] = (
                    sum(to_vals) / len(to_vals) if to_vals else None
                )
                entry["mean_fuzzy_match"] = (
                    sum(fm_vals) / len(fm_vals) if fm_vals else None
                )
                entry["answer_non_empty_rate"] = (
                    non_empty / n if n else None
                )
                entry["cap_hit_rate"] = (
                    cap_hits / n if n else None
                )
                entry["generation_max_new_tokens"] = _NAME_ONLY_MAX_NEW_TOKENS
            per_family[fam] = entry

        total = len(results)
        total_correct = sum(1 for r in results if r.correct is True)

        # Top-level visual-family accuracy (excluding name_only).
        visual_correct = sum(
            1 for r in results
            if r.probe_family in _IMAGE_FAMILIES and r.correct is True
        )
        visual_total = sum(1 for r in results if r.probe_family in _IMAGE_FAMILIES)
        visual_accuracy = visual_correct / visual_total if visual_total else None

        # Direct-visual accuracy (single family).
        dv_results = by_family.get("direct_visual", [])
        dv_correct = sum(1 for r in dv_results if r.correct is True)
        direct_visual_accuracy = (
            dv_correct / len(dv_results) if dv_results else None
        )

        # Name-only top-level metrics (P1-3: token_overlap is primary).
        no_results = by_family.get("name_only", [])
        no_to = [
            r.token_overlap for r in no_results
            if r.token_overlap is not None
        ]
        no_nem = [
            r.normalized_exact_match for r in no_results
            if r.normalized_exact_match is not None
        ]
        no_fm = [
            r.fuzzy_match for r in no_results
            if r.fuzzy_match is not None
        ]
        no_cap = sum(
            1 for r in no_results if r.hit_max_new_tokens is True
        )

        # Per-protocol-role counts (P0-9).
        role_counts: dict[str, int] = {"train": 0, "eval": 0, "exclude": 0}
        for r in results:
            if r.protocol_role in role_counts:
                role_counts[r.protocol_role] += 1
        per_protocol_role: dict[str, Any] = {
            role: {"n": count} for role, count in sorted(role_counts.items())
        }

        # Route identity role counts (P1-5): unique frozen route identities.
        # Gracefully degrade when role mapping is incomplete; the strict
        # check lives in validate_results().
        try:
            route_identity_role_counts = self._route_identity_role_counts()
        except RuntimeError:
            route_identity_role_counts = None

        summary: dict[str, Any] = {
            "total_probes": total,
            "total_correct": total_correct,
            "mixed_task_overall_accuracy": total_correct / total if total else None,
            "visual_accuracy": visual_accuracy,
            "visual_correct": visual_correct,
            "visual_total": visual_total,
            "direct_visual_accuracy": direct_visual_accuracy,
            "name_only_mean_token_overlap": (
                sum(no_to) / len(no_to) if no_to else None
            ),
            "name_only_normalized_exact_match": (
                sum(no_nem) / len(no_nem) if no_nem else None
            ),
            "name_only_fuzzy_match": (
                sum(no_fm) / len(no_fm) if no_fm else None
            ),
            "name_only_cap_hit_rate": (
                no_cap / len(no_results) if no_results else None
            ),
            "per_family": per_family,
            "per_protocol_role": per_protocol_role,
            "route_identity_role_counts": route_identity_role_counts,
            "model_fingerprint": self._fingerprint_id,
            "model_revision": self.model_config.revision,
            "scoring_version": SCORING_VERSION,
            "generation_protocol_sha256": self._generation_protocol_hash(),
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "baseline_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info("Summary written to %s", path)
        return summary

    # ------------------------------------------------------------------ #
    # Validation & manifest generation
    # ------------------------------------------------------------------ #

    def validate_results(self, smoke_probe_ids: set[str] | None = None) -> dict[str, Any]:
        """Strict P0-4 baseline validation for research-grade results.

        Performs the following checks:

        1. **exact_probe_id_set** — result probe IDs match source probes
           exactly (no duplicates, no missing, no unexpected).
           When *smoke_probe_ids* is provided, checks against that subset.
        2. **family_counts_match** — per-family result counts match the
           source probe artifact exactly.
        3. **binary_scores_complete** — every image-family result has finite
           log-probabilities, p_yes in [0, 1], finite margins, valid
           predicted_label and correct fields.
        4. **name_only_scores_complete** — every name_only result has
           non-empty generated/parsed answers, non-null text-match metrics,
           and non-null target_fact_id.
        5. **source_metadata_match** — every result's carried-through probe
           metadata matches the source probe exactly.
        6. **run_provenance_consistent** — all results share exactly one
           unique value for model revision, fingerprint, config SHA,
           route SHA, scoring version, and candidate protocol version.
        7. **zero_inference_errors** — no result row has an error.
        8. **protocol_role_complete** — every result has a valid protocol
           role in {train, eval, exclude} and all probes for one identity
           share the same role.

        Parameters
        ----------
        smoke_probe_ids:
            If provided, validate against this subset of probe IDs instead
            of all probes. Used for deterministic smoke testing (P0-4).

        Returns
        -------
        dict
            Validation report with ``pass`` (bool) and ``checks`` (dict).

        Raises
        ------
        RuntimeError
            If any check fails.
        """
        results = self._results
        probes_by_id: dict[str, BaselineProbe] = {
            p.probe_id: p for p in self.probes
        }
        checks: dict[str, Any] = {}
        passed = True

        def _record(name: str, ok: bool, **details: Any) -> None:
            nonlocal passed
            checks[name] = {"pass": ok, **details}
            if not ok:
                passed = False

        # -- 4.1  Exact probe-ID equality --------------------------------
        if smoke_probe_ids is not None:
            expected_ids = smoke_probe_ids
        else:
            expected_ids = {p.probe_id for p in self.probes}
        actual_ids = [r.probe_id for r in results]
        actual_id_set = set(actual_ids)
        no_dupes = len(actual_ids) == len(actual_id_set)
        id_set_match = actual_id_set == expected_ids
        _record(
            "exact_probe_id_set",
            id_set_match and no_dupes,
            expected_count=len(expected_ids),
            actual_count=len(actual_ids),
            unique_count=len(actual_id_set),
            no_duplicates=no_dupes,
            missing=sorted(expected_ids - actual_id_set),
            unexpected=sorted(actual_id_set - expected_ids),
        )

        # -- 4.2  Exact family counts ------------------------------------
        if smoke_probe_ids is not None:
            expected_family_counts = dict(Counter(
                p.probe_family for p in self.probes if p.probe_id in smoke_probe_ids
            ))
        else:
            expected_family_counts = dict(Counter(p.probe_family for p in self.probes))
        actual_family_counts = dict(Counter(r.probe_family for r in results))
        family_ok = expected_family_counts == actual_family_counts
        _record(
            "family_counts_match",
            family_ok,
            expected=expected_family_counts,
            actual=actual_family_counts,
        )

        # -- 4.3  Binary-family score completeness ------------------------
        _BINARY_FIELDS = (
            "logp_yes", "logp_no", "p_yes",
            "raw_log_margin", "signed_answer_margin",
        )
        binary_failures: list[dict[str, Any]] = []
        for r in results:
            if r.probe_family not in _IMAGE_FAMILIES:
                continue
            issues: list[str] = []
            if r.image_sha256 is None:
                issues.append("image_sha256_null")
            if r.answer_label is None:
                issues.append("answer_label_null")
            for fname in _BINARY_FIELDS:
                val = getattr(r, fname)
                if val is None:
                    issues.append(f"{fname}_null")
                elif not math.isfinite(val):
                    issues.append(f"{fname}_non_finite")
            if r.p_yes is not None and not (0.0 <= r.p_yes <= 1.0):
                issues.append("p_yes_out_of_range")
            if r.predicted_label not in ("Yes", "No"):
                issues.append(f"predicted_label_invalid={r.predicted_label!r}")
            if r.correct not in (True, False):
                issues.append(f"correct_invalid={r.correct!r}")
            if issues:
                binary_failures.append({"probe_id": r.probe_id, "issues": issues})
        _record(
            "binary_scores_complete",
            len(binary_failures) == 0,
            failure_count=len(binary_failures),
            failures=binary_failures[:10],
        )

        # -- 4.4  Name-only completeness ---------------------------------
        _TEXT_METRIC_FIELDS = (
            "exact_match", "normalized_exact_match",
            "token_overlap", "fuzzy_match",
        )
        name_only_failures: list[dict[str, Any]] = []
        for r in results:
            if r.probe_family != "name_only":
                continue
            issues: list[str] = []
            if not r.generated_answer:
                issues.append("generated_answer_empty")
            if not r.parsed_answer:
                issues.append("parsed_answer_empty")
            for fname in _TEXT_METRIC_FIELDS:
                if getattr(r, fname) is None:
                    issues.append(f"{fname}_null")
            if r.target_fact_id is None:
                issues.append("target_fact_id_null")
            if not r.answer_text:
                issues.append("answer_text_empty")
            if issues:
                name_only_failures.append({"probe_id": r.probe_id, "issues": issues})
        _record(
            "name_only_scores_complete",
            len(name_only_failures) == 0,
            failure_count=len(name_only_failures),
            failures=name_only_failures[:10],
        )

        # -- 4.5  Result / source metadata consistency -------------------
        _COMPARE_FIELDS = (
            "sample_id", "identity_id", "probe_family",
            "target_attribute", "answer_label", "answer_text",
            "image_sha256", "registry_hash", "paired_sample_id",
        )
        _WRONG_NAME_EXTRA = (
            "matched_wrong_identity_id",
            "matching_similarity",
            "matching_strategy",
        )
        metadata_mismatches: list[dict[str, Any]] = []
        for r in results:
            probe = probes_by_id.get(r.probe_id)
            if probe is None:
                continue
            diffs: list[dict[str, Any]] = []
            for fname in _COMPARE_FIELDS:
                exp = getattr(probe, fname, None)
                act = getattr(r, fname, None)
                if exp != act:
                    diffs.append({"field": fname, "expected": exp, "actual": act})
            if r.probe_family == "wrong_name":
                for fname in _WRONG_NAME_EXTRA:
                    exp = getattr(probe, fname, None)
                    act = getattr(r, fname, None)
                    if exp != act:
                        diffs.append({"field": fname, "expected": exp, "actual": act})
            if diffs:
                metadata_mismatches.append({"probe_id": r.probe_id, "mismatches": diffs})
        _record(
            "source_metadata_match",
            len(metadata_mismatches) == 0,
            mismatch_count=len(metadata_mismatches),
            mismatches=metadata_mismatches[:10],
        )

        # -- 4.6  Run-wide provenance consistency ------------------------
        _PROVENANCE_FIELDS = (
            "model_revision", "model_fingerprint",
            "model_config_sha256", "route_probe_sha256",
            "scoring_version", "candidate_protocol_version",
        )
        provenance_issues: dict[str, Any] = {}
        for fname in _PROVENANCE_FIELDS:
            values = {getattr(r, fname) for r in results}
            if len(values) != 1:
                provenance_issues[fname] = {
                    "unique_count": len(values),
                    "values": sorted(str(v) for v in values),
                }
        _record(
            "run_provenance_consistent",
            len(provenance_issues) == 0,
            issues=provenance_issues,
        )

        # -- 4.7  Zero inference errors ----------------------------------
        errors = [(r.probe_id, r.error) for r in results if r.error is not None]
        _record(
            "zero_inference_errors",
            len(errors) == 0,
            error_count=len(errors),
            errors=errors[:10],
        )

        # -- 4.8  Protocol-role completeness (P0-3 / P0-8) -----------------
        _VALID_ROLES = {"train", "eval", "exclude"}
        role_issues: list[dict[str, Any]] = []
        identity_roles: dict[str, set[str]] = {}
        for r in results:
            if r.protocol_role not in _VALID_ROLES:
                role_issues.append({
                    "probe_id": r.probe_id,
                    "identity_id": r.identity_id,
                    "role": r.protocol_role,
                    "issue": "invalid_or_empty_role",
                })
            identity_roles.setdefault(r.identity_id, set()).add(r.protocol_role)
        # Check identity-level consistency: all probes for one identity
        # must share the same role.
        inconsistent_identities: list[dict[str, Any]] = []
        for iid, roles in sorted(identity_roles.items()):
            if len(roles) > 1:
                inconsistent_identities.append({
                    "identity_id": iid,
                    "roles": sorted(roles),
                })
        # P0-3: For full research runs protocol_role_complete is mandatory.
        # The identity_role_map must be populated (processed dataset provided)
        # and every result must carry a valid role.
        role_complete_ok = (
            len(role_issues) == 0
            and len(inconsistent_identities) == 0
            and bool(self.identity_role_map)
        )
        _record(
            "protocol_role_complete",
            role_complete_ok,
            invalid_role_count=len(role_issues),
            inconsistent_identity_count=len(inconsistent_identities),
            identity_role_map_populated=bool(self.identity_role_map),
            invalid_roles=role_issues[:10],
            inconsistent_identities=inconsistent_identities[:10],
        )
        if not self.identity_role_map:
            raise RuntimeError(
                "Research baseline requires populated protocol roles.  "
                "Pass --processed-dataset for frozen protocol-role population."
            )

        # -- 4.9  Processed-dataset provenance (P1-2) ----------------------
        processed_sha_match = False
        if self.processed_dataset_path and self._dataset_manifest:
            try:
                proc_checks = self.validate_processed_dataset()
                processed_sha_match = proc_checks.get(
                    "processed_dataset_sha_match", False
                )
            except RuntimeError:
                pass
        checks["processed_dataset_sha_match"] = {
            "pass": processed_sha_match,
        }
        if not processed_sha_match:
            passed = False

        # -- 4.10  Route identity role counts (P1-5) -----------------------
        route_role_counts = self._route_identity_role_counts()
        checks["route_identity_role_counts"] = {
            "pass": True,
            **route_role_counts,
        }

        # -- 4.11  Write validation report ---------------------------------
        report = {"pass": passed, "checks": checks}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info("Validation report written to %s (pass=%s)", report_path, passed)

        if not passed:
            failed = [k for k, v in checks.items() if not v["pass"]]
            raise RuntimeError(
                f"Baseline validation failed checks: {failed}"
            )
        return report

    def generate_baseline_manifest(self) -> dict[str, Any]:
        """Generate and write ``baseline_manifest.json``.

        Freezes dataset provenance, model identity, scoring config, runtime
        environment, results summary, and code provenance into a single
        manifest file.  Also computes SHA-256 of all output artifacts.

        A clean Git tree is required (P0-11): a full baseline produced
        from a dirty tree is never marked research-valid.

        Returns
        -------
        dict
            The manifest dictionary.
        """
        # P0-11: clean-tree check before writing the manifest.
        git_state = self._get_git_state()
        if git_state["git_dirty"]:
            raise RuntimeError(
                "Git tree is dirty.  Refusing to write a research "
                "baseline manifest from a dirty tree."
            )
        if not git_state["git_commit"]:
            raise RuntimeError(
                "Cannot determine Git commit SHA.  Refusing to write "
                "a research baseline manifest outside a Git repository."
            )

        results_path = self.output_dir / "baseline_results.jsonl"
        summary_path = self.output_dir / "baseline_summary.json"
        validation_report_path = self.output_dir / "validation_report.json"
        smoke_manifest_path = self.output_dir / "smoke_manifest.json"

        # Compute SHA-256 of output files.
        results_sha = _sha256_file(results_path) if results_path.is_file() else ""
        summary_sha = _sha256_file(summary_path) if summary_path.is_file() else ""
        validation_sha = (
            _sha256_file(validation_report_path)
            if validation_report_path.is_file()
            else ""
        )
        smoke_sha = (
            _sha256_file(smoke_manifest_path)
            if smoke_manifest_path.is_file()
            else ""
        )

        # Load summary if available.
        summary_data: dict[str, Any] = {}
        if summary_path.is_file():
            with open(summary_path) as f:
                summary_data = json.load(f)

        # Git provenance (refactored via _get_git_state helper).
        git_commit = git_state["git_commit"]
        git_dirty = git_state["git_dirty"]

        # Dataset provenance (P1-8: full freeze chain).
        dataset_version = ""
        dataset_manifest_sha = ""
        if self._freeze_verification:
            dataset_version = self._freeze_verification.get(
                "dataset_version", ""
            )
            dataset_manifest_sha = self._freeze_verification.get(
                "dataset_manifest_sha256", ""
            )
        if not dataset_manifest_sha and self.dataset_manifest_path:
            dataset_manifest_sha = _sha256_file(self.dataset_manifest_path)
        if not dataset_version and self._dataset_manifest:
            dataset_version = self._dataset_manifest.get("dataset_version", "")

        # Freeze-verification SHA (P1-8).
        freeze_verification_sha = ""
        if self.freeze_verification_path:
            freeze_verification_sha = _sha256_file(
                self.freeze_verification_path
            )

        # Route-probe SHA from freeze verification or manifest.
        route_probe_sha = ""
        route_probe_count = len(self.probes)
        if self._freeze_verification:
            route_probe_sha = self._freeze_verification.get(
                "route_probe_sha256", ""
            )
        if not route_probe_sha:
            route_probe_sha = self._route_probe_sha()
            if route_probe_sha == "none":
                route_probe_sha = ""

        # Processed-dataset provenance (P1-3).
        processed_dataset_sha = ""
        if self.processed_dataset_path and self.processed_dataset_path.is_file():
            processed_dataset_sha = _sha256_file(self.processed_dataset_path)
        processed_dataset_manifest_sha = ""
        if self._dataset_manifest:
            processed_dataset_manifest_sha = (
                self._dataset_manifest
                .get("dataset_artifacts", {})
                .get("processed_dataset", {})
                .get("sha256", "")
            )

        # Protocol SHA: computed from the frozen generation config (P1-1).
        # This incorporates binary + name_only generation settings so that
        # any future change to max_new_tokens etc. produces a distinct hash.
        protocol_sha256 = self._generation_protocol_hash()

        # Route identity role counts (P1-5): unique frozen route identities.
        route_identity_role_counts = self._route_identity_role_counts()

        # Runtime library versions (P1-3).
        torch_version = ""
        transformers_version = ""
        accelerate_version = ""
        try:
            import torch
            torch_version = torch.__version__
        except ImportError:
            pass
        try:
            import transformers
            transformers_version = transformers.__version__
        except ImportError:
            pass
        try:
            import accelerate
            accelerate_version = accelerate.__version__
        except ImportError:
            pass

        # Full fingerprint payload from backend (P1-3 / P1-6).
        fingerprint_payload = dict(self._fingerprint)

        # Scoring provenance (P1-3).
        from ..models.scoring import CANDIDATE_PROTOCOL_VERSION as _cpv
        scoring_provenance = {
            "candidate_protocol": "binary_yes_no",
            "candidate_protocol_version": _cpv,
            "scoring_version": SCORING_VERSION,
            "thinking_mode": "disabled",
            "decision_rule": "p_yes_geq_0.5",
            "raw_log_margin_definition": "logp_yes_minus_logp_no",
            "signed_answer_margin_definition": (
                "raw_log_margin_if_target_yes_else_negated_raw_log_margin"
            ),
            "signed_answer_margin_interpretation": "higher_is_better",
        }

        manifest: dict[str, Any] = {
            "schema_version": "1.2",
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset_provenance": {
                "probe_file": str(self.probe_path),
                "probe_file_sha256": _sha256_file(self.probe_path),
                "probe_count": len(self.probes),
                "dataset_manifest": (
                    str(self.dataset_manifest_path)
                    if self.dataset_manifest_path else None
                ),
                "dataset_version": dataset_version,
                "dataset_manifest_sha256": dataset_manifest_sha,
                "freeze_verification": (
                    str(self.freeze_verification_path)
                    if self.freeze_verification_path else None
                ),
                "freeze_verification_sha256": freeze_verification_sha,
                "route_probe_sha256": route_probe_sha,
                "route_probe_count": route_probe_count,
                "processed_dataset_path": (
                    str(self.processed_dataset_path)
                    if self.processed_dataset_path else None
                ),
                "processed_dataset_sha256": processed_dataset_sha,
                "processed_dataset_manifest_sha256": processed_dataset_manifest_sha,
            },
            "protocol_sha256": protocol_sha256,
            "model_identity": {
                "model_id": self.model_config.model_id,
                "model_revision": self.model_config.revision,
                "backend": self.model_config.backend,
                "fingerprint_id": self._fingerprint_id,
                "fingerprint_payload": fingerprint_payload,
                "model_config_sha256": self._model_config_sha,
            },
            "scoring_config": {
                "scoring_version": SCORING_VERSION,
                "candidates": list(CANDIDATES),
                **scoring_provenance,
            },
            "runtime_environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "cwd": os.getcwd(),
                "torch_version": torch_version,
                "transformers_version": transformers_version,
                "accelerate_version": accelerate_version,
            },
            "results": {
                "results_file": str(results_path),
                "results_sha256": results_sha,
                "summary_file": str(summary_path),
                "summary_sha256": summary_sha,
                "validation_report_file": str(validation_report_path),
                "validation_report_sha256": validation_sha,
                "smoke_manifest_file": str(smoke_manifest_path),
                "smoke_manifest_sha256": smoke_sha,
                "total_results": len(self._results),
                "summary": summary_data,
            },
            "route_identity_role_counts": route_identity_role_counts,
            "code_provenance": {
                "git_commit": git_commit,
                "git_dirty": git_dirty,
            },
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "baseline_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        log.info("Baseline manifest written to %s", manifest_path)
        return manifest

    # ------------------------------------------------------------------ #
    # Internal cache I/O
    # ------------------------------------------------------------------ #

    def _append_cache(self, result: BaselineResult) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "a") as f:
            f.write(json.dumps(asdict(result), default=str) + "\n")


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


def _prompt_hash(text: str) -> str:
    """SHA-256 hex digest of a prompt string (first 16 chars)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_image(path: str | None):
    """Load an RGB image from a filesystem path."""
    from PIL import Image

    if path is None:
        return None
    return Image.open(path).convert("RGB")


def _compute_exact_match(generated: str, expected: str) -> float:
    """1.0 if *generated* exactly matches *expected* (case-sensitive)."""
    return 1.0 if generated == expected else 0.0


def _compute_normalized_exact_match(generated: str, expected: str) -> float:
    """1.0 if match after lowering, stripping punctuation, collapsing whitespace."""
    def _normalize(text: str) -> str:
        text = text.strip().lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    return 1.0 if _normalize(generated) == _normalize(expected) else 0.0


def _compute_token_overlap(generated: str, expected: str) -> float:
    """Token-level F1 overlap between *generated* and *expected*."""
    g_tokens = set(generated.strip().lower().split())
    e_tokens = set(expected.strip().lower().split())
    if not g_tokens or not e_tokens:
        return 0.0
    overlap = g_tokens & e_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(g_tokens)
    recall = len(overlap) / len(e_tokens)
    return 2 * precision * recall / (precision + recall)


def _text_match(generated: str, expected: str) -> float:
    """Normalised text overlap score for name-only probes.

    Returns 1.0 for an exact match (after lowering + stripping), a partial
    score based on token overlap, or 0.0 for a complete mismatch.
    """
    g = generated.strip().lower()
    e = expected.strip().lower()
    if g == e:
        return 1.0
    if e in g or g in e:
        return 0.8
    g_tokens = set(g.split())
    e_tokens = set(e.split())
    if not e_tokens:
        return 0.0
    overlap = len(g_tokens & e_tokens) / len(e_tokens)
    return overlap if overlap > 0.3 else 0.0


def select_smoke_probes(
    probes: list[BaselineProbe],
    n_identities: int = 2,
) -> list[BaselineProbe]:
    """Select a deterministic smoke-test subset of probes.

    Picks exactly *n_identities* eligible identities where each identity
    has all 5 probe families. Returns exactly ``n_identities * 5`` probes.

    An eligible identity is one that has at least one probe in each of the
    5 families: direct_visual, image_plus_name, wrong_name,
    visual_text_conflict, name_only.

    Parameters
    ----------
    probes:
        Full list of baseline probes.
    n_identities:
        Number of distinct identities to include (default 2).

    Returns
    -------
    list[BaselineProbe]
        Selected probes (exactly ``n_identities * 5``).

    Raises
    ------
    ValueError
        If fewer than *n_identities* eligible identities exist.
    """
    # Group by identity and family.
    by_identity: dict[str, dict[str, BaselineProbe]] = {}
    for p in probes:
        by_identity.setdefault(p.identity_id, {})[p.probe_family] = p

    # Find eligible identities (those with all 5 families).
    eligible_ids = sorted(
        iid for iid, fam_map in by_identity.items()
        if set(fam_map) == ALL_FAMILIES
    )

    if len(eligible_ids) < n_identities:
        raise ValueError(
            f"Need {n_identities} eligible identities with all {len(ALL_FAMILIES)} "
            f"families, but only {len(eligible_ids)} found."
        )

    # Select the first n_identities eligible identities.
    selected_ids = eligible_ids[:n_identities]

    # For each selected identity, pick one probe per family (sorted by probe_id).
    selected: list[BaselineProbe] = []
    for iid in selected_ids:
        fam_map = by_identity[iid]
        for fam in sorted(ALL_FAMILIES):
            selected.append(fam_map[fam])

    return selected


def write_smoke_manifest(
    probes: list[BaselineProbe],
    output_path: str | Path,
    probe_file_sha256: str = "",
) -> Path:
    """Write a smoke manifest JSON for the selected probes.

    The manifest records the selected probe IDs, identity count, family
    counts, and (optionally) the SHA-256 of the source probe file for
    reproducibility.

    Parameters
    ----------
    probes:
        Selected smoke-test probes.
    output_path:
        Path to write the manifest JSON.
    probe_file_sha256:
        Optional SHA-256 of the source probe file.

    Returns
    -------
    Path
        The path to the written manifest.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute family counts.
    family_counts: dict[str, int] = {}
    for fam in sorted(ALL_FAMILIES):
        family_counts[fam] = sum(1 for p in probes if p.probe_family == fam)

    manifest = {
        "dataset_version": "fiubench-route-v1",
        "route_probe_sha256": probe_file_sha256,
        "identity_count": len({p.identity_id for p in probes}),
        "probe_count": len(probes),
        "selected_identity_ids": sorted({p.identity_id for p in probes}),
        "selected_probe_ids": sorted(p.probe_id for p in probes),
        "family_counts": family_counts,
    }
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Smoke manifest written to %s (%d probes)", output_path, len(probes))
    return output_path
