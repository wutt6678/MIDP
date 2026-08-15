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
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import ModelConfig
from ..models.base import VisionLanguageModel
from ..models.scoring import SCORING_VERSION, binary_probability

log = logging.getLogger(__name__)

__all__ = [
    "BaselineProbe",
    "BaselineResult",
    "BaselineRunner",
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

    # -- Provenance --
    scoring_version: str = ""
    prompt_hash: str = ""
    latency_ms: float = 0.0
    error: str | None = None


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

    def _compute_model_config_sha(self) -> str:
        """SHA-256 of the model config YAML (or serialised dataclass)."""
        if self.model_config_path and self.model_config_path.is_file():
            return _sha256_file(self.model_config_path)
        # Fallback: hash the serialised dataclass.
        raw = json.dumps(asdict(self.model_config), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------ #
    # Cache / resume
    # ------------------------------------------------------------------ #

    @property
    def _cache_path(self) -> Path:
        return self._cache_dir / "baseline_cache.jsonl"

    def _cache_key(self, probe: BaselineProbe) -> str:
        """Composite cache key for deterministic resumption.

        The key includes the probe identity, model revision, prompt hash,
        image hash, scoring version, and — when available — the route-probe
        SHA from the dataset manifest and the model-config SHA so that any
        change to the dataset or model config invalidates the cache.
        """
        parts = [
            probe.probe_id,
            self.model_config.revision or "none",
            _prompt_hash(probe.question),
            probe.image_sha256 or "none",
            SCORING_VERSION,
        ]
        # Stronger key: include dataset and model config provenance.
        route_sha = "none"
        if self._dataset_manifest:
            route_sha = (
                self._dataset_manifest
                .get("dataset_artifacts", {})
                .get("route_probes", {})
                .get("sha256", "none")
            )
        parts.append(route_sha)
        parts.append(self._model_config_sha[:16])
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _load_cache(self) -> list[BaselineResult]:
        """Load previously completed results from the cache file."""
        if not self._cache_path.is_file():
            return []
        results: list[BaselineResult] = []
        for doc in _read_jsonl(self._cache_path):
            results.append(BaselineResult(**doc))
        log.info("Loaded %d cached results from %s", len(results), self._cache_path)
        return results

    def _done_keys(self) -> set[str]:
        return {self._cache_key_for_result(r) for r in self._results}

    def _cache_key_for_result(self, result: BaselineResult) -> str:
        """Recompute the cache key from a result record.

        Must produce the same key as :meth:`_cache_key` for the same probe.
        """
        parts = [
            result.probe_id,
            result.model_revision or "none",
            result.prompt_hash,
            result.image_sha256 or "none",
            result.scoring_version,
        ]
        route_sha = "none"
        if self._dataset_manifest:
            route_sha = (
                self._dataset_manifest
                .get("dataset_artifacts", {})
                .get("route_probes", {})
                .get("sha256", "none")
            )
        parts.append(route_sha)
        parts.append(self._model_config_sha[:16])
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------ #
    # Single-probe execution
    # ------------------------------------------------------------------ #

    def run_probe(self, probe: BaselineProbe) -> BaselineResult:
        """Run deterministic inference + scoring for one probe."""
        fp = self._fingerprint_id
        rev = self.model_config.revision
        ph = _prompt_hash(probe.question)
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
                # name_only: text-only generation
                resp = self.backend.generate(None, probe.question)
                generated_answer = resp.text or ""
                parsed_answer = generated_answer.strip()
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
                    # Primary metric: normalized exact match
                    correct = (normalized_exact_match == 1.0)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.warning("Probe %s failed: %s", probe.probe_id, error)

        latency = (time.perf_counter() - started) * 1000.0

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
            # Provenance
            scoring_version=SCORING_VERSION,
            prompt_hash=ph,
            latency_ms=latency,
            error=error,
        )

    # ------------------------------------------------------------------ #
    # Batch execution
    # ------------------------------------------------------------------ #

    def run_all(self, limit: int | None = None) -> list[BaselineResult]:
        """Iterate all probes with progress logging and cache/resume."""
        probes = self.probes if limit is None else self.probes[:limit]
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
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
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
                n_correct = sum(1 for r in fam_results if r.correct is True)
                non_empty = sum(
                    1
                    for r in fam_results
                    if r.generated_answer.strip()
                )
                entry["correct"] = n_correct
                entry["normalized_exact_match_rate"] = (
                    sum(nem_vals) / len(nem_vals) if nem_vals else None
                )
                entry["mean_token_overlap"] = (
                    sum(to_vals) / len(to_vals) if to_vals else None
                )
                entry["answer_non_empty_rate"] = (
                    non_empty / n if n else None
                )
            per_family[fam] = entry

        total = len(results)
        total_correct = sum(1 for r in results if r.correct is True)
        summary: dict[str, Any] = {
            "total_probes": total,
            "total_correct": total_correct,
            "overall_accuracy": total_correct / total if total else None,
            "per_family": per_family,
            "model_fingerprint": self._fingerprint_id,
            "model_revision": self.model_config.revision,
            "scoring_version": SCORING_VERSION,
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

    def validate_results(
        self,
        expected_probe_count: int | None = None,
    ) -> dict[str, Any]:
        """Strict completeness validator for baseline results.

        Checks:
        - All probes have been scored (count matches expected or total probes).
        - No errors in any result.
        - All five probe families are represented.
        - Every result has a valid scoring_version and prompt_hash.

        Parameters
        ----------
        expected_probe_count:
            If given, assert that exactly this many results exist.  When
            *None*, the check is against ``len(self.probes)``.

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
        checks: dict[str, Any] = {}
        passed = True

        # 1. Probe count check
        expected = expected_probe_count if expected_probe_count is not None else len(self.probes)
        actual = len(results)
        count_ok = actual == expected
        checks["probe_count"] = {
            "expected": expected,
            "actual": actual,
            "pass": count_ok,
        }
        if not count_ok:
            passed = False

        # 2. No errors
        errors = [(r.probe_id, r.error) for r in results if r.error is not None]
        no_errors = len(errors) == 0
        checks["no_errors"] = {
            "error_count": len(errors),
            "errors": errors[:10],  # first 10 for diagnostics
            "pass": no_errors,
        }
        if not no_errors:
            passed = False

        # 3. All families covered
        families_present = {r.probe_family for r in results}
        families_ok = families_present == ALL_FAMILIES
        checks["all_families"] = {
            "expected": sorted(ALL_FAMILIES),
            "actual": sorted(families_present),
            "pass": families_ok,
        }
        if not families_ok:
            passed = False

        # 4. Provenance fields populated
        missing_scoring = [r.probe_id for r in results if not r.scoring_version]
        missing_prompt = [r.probe_id for r in results if not r.prompt_hash]
        provenance_ok = len(missing_scoring) == 0 and len(missing_prompt) == 0
        checks["provenance_populated"] = {
            "missing_scoring_version": len(missing_scoring),
            "missing_prompt_hash": len(missing_prompt),
            "pass": provenance_ok,
        }
        if not provenance_ok:
            passed = False

        report = {"pass": passed, "checks": checks}

        # Write validation report
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
        manifest file.  Also computes SHA-256 of the results JSONL.

        Returns
        -------
        dict
            The manifest dictionary.
        """
        results_path = self.output_dir / "baseline_results.jsonl"
        summary_path = self.output_dir / "baseline_summary.json"

        # Compute SHA-256 of results file
        results_sha = ""
        if results_path.is_file():
            results_sha = _sha256_file(results_path)

        # Load summary if available
        summary_data: dict[str, Any] = {}
        if summary_path.is_file():
            with open(summary_path) as f:
                summary_data = json.load(f)

        # Git provenance
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

        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset_provenance": {
                "probe_file": str(self.probe_path),
                "probe_file_sha256": _sha256_file(self.probe_path),
                "probe_count": len(self.probes),
                "dataset_manifest": (
                    str(self.dataset_manifest_path) if self.dataset_manifest_path else None
                ),
            },
            "model_identity": {
                "model_id": self.model_config.model_id,
                "model_revision": self.model_config.revision,
                "backend": self.model_config.backend,
                "fingerprint_id": self._fingerprint_id,
                "model_config_sha256": self._model_config_sha,
            },
            "scoring_config": {
                "scoring_version": SCORING_VERSION,
                "candidates": list(CANDIDATES),
            },
            "runtime_environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "cwd": os.getcwd(),
            },
            "results": {
                "results_file": str(results_path),
                "results_sha256": results_sha,
                "total_results": len(self._results),
                "summary": summary_data,
            },
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
