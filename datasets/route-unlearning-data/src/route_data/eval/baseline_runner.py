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
    ``logp_yes``, ``logp_no``, and ``p_yes``.  For ``name_only`` (text-only,
    free-form answer) the *generation* path fills ``generated_answer`` and
    ``text_match_score``; the probability fields stay ``None``.
    """

    # Probe identity
    probe_id: str
    sample_id: str
    identity_id: str
    probe_family: str
    modality: str

    # Model identity
    model_fingerprint: str
    model_revision: str | None

    # Inputs
    question: str
    image_sha256: str | None

    # Outputs
    generated_answer: str
    parsed_answer: str | None
    correct: bool | None

    # Candidate scores (binary families only)
    logp_yes: float | None = None
    logp_no: float | None = None
    p_yes: float | None = None
    margin: float | None = None

    # Text-match score (name_only only)
    text_match_score: float | None = None

    # Ground truth (carried through for downstream analysis)
    answer_label: bool | None = None
    answer_text: str = ""

    # Provenance
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
    """

    def __init__(
        self,
        backend: VisionLanguageModel,
        probe_path: str | Path,
        output_dir: str | Path,
        model_config: ModelConfig,
        resume: bool = True,
    ):
        self.backend = backend
        self.probe_path = Path(probe_path)
        self.output_dir = Path(output_dir)
        self.model_config = model_config
        self.resume = resume

        self._fingerprint = backend.fingerprint()
        self._fingerprint_id: str = str(
            self._fingerprint.get("fingerprint_id", "unknown")
        )
        self._cache_dir = self.output_dir / ".cache"

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

    # ------------------------------------------------------------------ #
    # Cache / resume
    # ------------------------------------------------------------------ #

    @property
    def _cache_path(self) -> Path:
        return self._cache_dir / "baseline_cache.jsonl"

    def _cache_key(self, probe: BaselineProbe) -> str:
        """Composite cache key for deterministic resumption."""
        parts = [
            probe.probe_id,
            self.model_config.revision or "none",
            _prompt_hash(probe.question),
            probe.image_sha256 or "none",
            SCORING_VERSION,
        ]
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
        """Recompute the cache key from a result record."""
        parts = [
            result.probe_id,
            result.model_revision or "none",
            result.prompt_hash,
            result.image_sha256 or "none",
            result.scoring_version,
        ]
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
        correct: bool | None = None
        logp_yes: float | None = None
        logp_no: float | None = None
        p_yes: float | None = None
        margin: float | None = None
        text_match_score: float | None = None
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
                    margin = p_yes - 0.5
                    parsed_answer = _POSITIVE if p_yes >= 0.5 else _NEGATIVE
                    if probe.answer_label is not None:
                        correct = (parsed_answer == _POSITIVE) == probe.answer_label
                else:
                    error = "backend did not return scores for both candidates"
            else:
                # name_only: text-only generation
                resp = self.backend.generate(None, probe.question)
                generated_answer = resp.text or ""
                parsed_answer = generated_answer.strip()
                if probe.answer_text:
                    text_match_score = _text_match(
                        generated_answer, probe.answer_text
                    )
                    correct = text_match_score > 0.5
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.warning("Probe %s failed: %s", probe.probe_id, error)

        latency = (time.perf_counter() - started) * 1000.0

        return BaselineResult(
            probe_id=probe.probe_id,
            sample_id=probe.sample_id,
            identity_id=probe.identity_id,
            probe_family=probe.probe_family,
            modality=probe.modality,
            model_fingerprint=fp,
            model_revision=rev,
            question=probe.question,
            image_sha256=probe.image_sha256,
            generated_answer=generated_answer,
            parsed_answer=parsed_answer,
            correct=correct,
            logp_yes=logp_yes,
            logp_no=logp_no,
            p_yes=p_yes,
            margin=margin,
            text_match_score=text_match_score,
            answer_label=probe.answer_label,
            answer_text=probe.answer_text,
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
        """Compute and write ``baseline_summary.json``."""
        results = self._results
        by_family: dict[str, list[BaselineResult]] = {}
        for r in results:
            by_family.setdefault(r.probe_family, []).append(r)

        per_family: dict[str, Any] = {}
        for fam in sorted(by_family):
            fam_results = by_family[fam]
            n = len(fam_results)
            n_correct = sum(1 for r in fam_results if r.correct is True)
            n_errors = sum(1 for r in fam_results if r.error is not None)
            p_values = [r.p_yes for r in fam_results if r.p_yes is not None]
            entry: dict[str, Any] = {
                "count": n,
                "correct": n_correct,
                "accuracy": n_correct / n if n else None,
                "errors": n_errors,
            }
            if p_values:
                entry["mean_p_yes"] = sum(p_values) / len(p_values)
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
