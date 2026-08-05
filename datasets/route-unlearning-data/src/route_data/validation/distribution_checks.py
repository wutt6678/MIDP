"""Distribution reports for benchmarks and splits (plan section 19.3).

Pure reporting — no pass/fail semantics — so the numbers can be embedded in
extension reports (plan 22.2) and reviewed by humans. For every benchmark and
split it reports: identities, images, QAs, positive rate by attribute, images
per identity, facts per identity, forget/retain ratio, and the share of
high/medium/low-confidence labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from ..build.split_generation import SplitResult
from ..data.schemas import CanonicalSample

CELEBA40_PREFIX = "extended_attributes.celeba40."


def _images_per_identity(samples: Sequence[CanonicalSample]) -> dict[str, float]:
    counts = Counter(s.identity_id for s in samples)
    values = list(counts.values())
    if not values:
        return {"mean": 0.0, "min": 0, "max": 0}
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def _facts_per_identity(samples: Sequence[CanonicalSample]) -> dict[str, float]:
    counts = Counter(
        s.identity_id for s in samples if s.profile_facts
    )
    identities = {s.identity_id for s in samples}
    if not identities:
        return {"mean": 0.0, "min": 0, "max": 0}
    values = [counts.get(i, 0) for i in identities]
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def _attribute_stats(samples: Sequence[CanonicalSample]) -> dict[str, dict[str, float]]:
    """Positive rate + confidence-band share per CelebA-40 attribute."""
    labeled: dict[str, int] = defaultdict(int)
    positive: dict[str, int] = defaultdict(int)
    bands: dict[str, Counter] = defaultdict(Counter)
    for sample in samples:
        for key, obs in sample.visual_attributes.items():
            if not key.startswith(CELEBA40_PREFIX):
                continue
            attribute = key[len(CELEBA40_PREFIX):]
            bands[attribute][obs.confidence_band] += 1
            if obs.label is None:
                continue
            labeled[attribute] += 1
            positive[attribute] += int(obs.label)
    stats: dict[str, dict[str, float]] = {}
    for attribute in sorted(set(labeled) | set(bands)):
        total = labeled.get(attribute, 0)
        entry: dict[str, float] = {
            "labeled": total,
            "positive_rate": (positive[attribute] / total) if total else None,
        }
        band_total = sum(bands[attribute].values())
        for band in ("high", "medium", "low"):
            entry[f"{band}_fraction"] = (
                bands[attribute][band] / band_total if band_total else 0.0
            )
        stats[attribute] = entry
    return stats


def summarize_samples(
    samples: Sequence[CanonicalSample], *, qa_count: int | None = None
) -> dict[str, Any]:
    """Plan-19.3 statistics for one bucket of samples."""
    identities = {s.identity_id for s in samples}
    images = {s.image_sha256 or s.image_uri for s in samples if s.image_uri}
    band_counts = Counter()
    for sample in samples:
        for obs in sample.visual_attributes.values():
            if obs.name.startswith(CELEBA40_PREFIX):
                band_counts[obs.confidence_band] += 1
    band_total = sum(band_counts.values())
    return {
        "num_samples": len(samples),
        "num_identities": len(identities),
        "num_images": len(images),
        "num_qas": qa_count if qa_count is not None else sum(
            1 for s in samples if s.task_type in {"visual_attribute", "identity_fact"}
        ),
        "images_per_identity": _images_per_identity(samples),
        "facts_per_identity": _facts_per_identity(samples),
        "confidence_band_fractions": {
            band: (band_counts[band] / band_total if band_total else 0.0)
            for band in ("high", "medium", "low", "unknown")
        },
        "attributes": _attribute_stats(samples),
        "modality_counts": dict(Counter(s.modality for s in samples)),
        "task_type_counts": dict(Counter(s.task_type for s in samples)),
    }


def summarize_split(result: SplitResult) -> dict[str, Any]:
    """Per-split report including the forget/retain ratio."""
    report = {
        "name": result.spec.name,
        "forget_scope": result.spec.forget_scope,
        "forget": summarize_samples(result.forget),
        "retain_train": summarize_samples(result.retain_train),
        "retain_eval": summarize_samples(result.retain_eval),
        "unassigned": summarize_samples(result.unassigned),
    }
    retain = len(result.retain_train) + len(result.retain_eval)
    report["forget_retain_ratio"] = len(result.forget) / retain if retain else None
    return report


def distribution_report(
    samples: Sequence[CanonicalSample],
    *,
    benchmark: str | None = None,
    split_results: Iterable[SplitResult] = (),
) -> dict[str, Any]:
    """Full plan-19.3 report for one benchmark."""
    return {
        "benchmark": benchmark or (samples[0].benchmark if samples else None),
        "overall": summarize_samples(samples),
        "splits": [summarize_split(result) for result in split_results],
    }


def render_distribution_md(report: Mapping[str, Any]) -> str:
    """Compact Markdown rendering for dataset cards / extension reports."""
    overall = report.get("overall", {})
    lines = [
        f"# Distribution report — {report.get('benchmark', 'unknown')}",
        "",
        f"- Samples: {overall.get('num_samples', 0)}",
        f"- Identities: {overall.get('num_identities', 0)}",
        f"- Images: {overall.get('num_images', 0)}",
        "",
        "## Confidence bands (CelebA-40 observations)",
    ]
    for band, fraction in overall.get("confidence_band_fractions", {}).items():
        lines.append(f"- {band}: {fraction:.3f}")
    lines += ["", "## Attribute positive rates (labeled observations only)", ""]
    lines.append("| attribute | labeled | positive rate |")
    lines.append("|---|---:|---:|")
    for attribute, stats in overall.get("attributes", {}).items():
        rate = stats.get("positive_rate")
        lines.append(
            f"| {attribute} | {stats.get('labeled', 0)} | "
            f"{rate:.3f} |" if rate is not None else f"| {attribute} | 0 | n/a |"
        )
    for split in report.get("splits", []):
        lines += [
            "",
            f"## Split: {split['name']} ({split['forget_scope']})",
            f"- forget: {split['forget']['num_samples']}",
            f"- retain_train: {split['retain_train']['num_samples']}",
            f"- retain_eval: {split['retain_eval']['num_samples']}",
            f"- forget/retain ratio: {split.get('forget_retain_ratio')}",
        ]
    return "\n".join(lines) + "\n"
