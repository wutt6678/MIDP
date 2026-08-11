"""Extension export: Parquet/JSONL artifacts + dataset cards (plan 12.6, 19.2).

Serializes the built extension for one benchmark into the plan's output layout:

- ``{benchmark}_celeba40_image_annotations.parquet`` (one row per observation);
- ``{benchmark}_celeba40_visual_qa_train.jsonl`` / ``..._eval.jsonl``;
- ``{benchmark}_route_conflict_eval.jsonl``;
- ``{benchmark}_unlearning_splits/*.json``;
- ``{benchmark}_extension_card.md``.

Images are never redistributed: cards reference images by URI/checksum and embed
the source license, citation, prompt-registry hash, and model fingerprint so the
export is auditable (plan 19.2). Subjective/sensitive labels carry the CelebA
definition caveat in every card (plan 10.4).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..constants.attribute_taxonomy import LOW_RELIABILITY, SENSITIVE_DATASET_LABELS
from ..data.io import ensure_parent_dir, write_json, write_jsonl, write_parquet
from ..data.schemas import CanonicalSample
from .split_generation import SplitResult


@dataclass
class ExportRecord:
    """Paths + counts for everything written for one benchmark."""

    benchmark: str
    output_dir: Path
    paths: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "output_dir": str(self.output_dir),
            "paths": dict(self.paths),
            "counts": dict(self.counts),
        }


class ExtensionExporter:
    """Writes one benchmark's built extension to disk and renders its card."""

    def __init__(
        self,
        output_dir: str | Path,
        benchmark: str,
        *,
        source_version: str = "unknown",
        registry_hash: str | None = None,
        model_fingerprint: str | None = None,
        license_text: str = "See the upstream dataset license.",
        citation: str = "",
    ):
        if not benchmark:
            raise ValueError("benchmark name must be non-empty")
        self.benchmark = benchmark
        self.output_dir = Path(output_dir) / benchmark
        self.source_version = source_version
        self.registry_hash = registry_hash
        self.model_fingerprint = model_fingerprint
        self.license_text = license_text
        self.citation = citation

    # -- helpers -------------------------------------------------------- #

    def _path(self, filename: str) -> Path:
        return self.output_dir / filename

    @staticmethod
    def _annotation_rows(samples: Iterable[CanonicalSample]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for sample in samples:
            for key, obs in sample.visual_attributes.items():
                rows.append(
                    {
                        "sample_id": sample.source_sample_id,
                        "identity_id": sample.identity_id,
                        "benchmark": sample.benchmark,
                        "image_uri": sample.image_uri,
                        "image_sha256": sample.image_sha256,
                        "attribute": key,
                        "label": obs.label,
                        "score": obs.score,
                        "source": obs.source,
                        "confidence_band": obs.confidence_band,
                        "attribute_class": obs.attribute_class,
                        "model_fingerprint": obs.model_fingerprint,
                        "prompt_id": obs.prompt_id,
                    }
                )
        return rows

    # -- writers -------------------------------------------------------- #

    def write_image_annotations(self, samples: Iterable[CanonicalSample]) -> str:
        rows = self._annotation_rows(samples)
        path = self._path(f"{self.benchmark}_celeba40_image_annotations.parquet")
        write_parquet(rows, path)
        return str(path)

    def write_visual_qa(
        self, train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]]
    ) -> tuple[str, str]:
        train_path = self._path(f"{self.benchmark}_celeba40_visual_qa_train.jsonl")
        eval_path = self._path(f"{self.benchmark}_celeba40_visual_qa_eval.jsonl")
        write_jsonl(list(train_rows), train_path)
        write_jsonl(list(eval_rows), eval_path)
        return str(train_path), str(eval_path)

    def write_route_probes(self, rows: Sequence[Mapping[str, Any]]) -> str:
        path = self._path(f"{self.benchmark}_route_conflict_eval.jsonl")
        write_jsonl(list(rows), path)
        return str(path)

    def write_splits(self, results: Sequence[SplitResult]) -> str:
        split_dir = self._path(f"{self.benchmark}_unlearning_splits")
        split_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            payload = {
                "manifest": result.manifest(),
                "forget_sample_ids": [s.source_sample_id for s in result.forget],
                "retain_train_sample_ids": [s.source_sample_id for s in result.retain_train],
                "retain_eval_sample_ids": [s.source_sample_id for s in result.retain_eval],
            }
            write_json(payload, split_dir / f"{result.spec.name}.json")
        return str(split_dir)

    # -- dataset card --------------------------------------------------- #

    def render_dataset_card(self, counts: Mapping[str, int] | None = None) -> str:
        counts = dict(counts or {})
        lines: list[str] = [
            f"# {self.benchmark} CelebA-40 extension card",
            "",
            f"- Benchmark: `{self.benchmark}`",
            f"- Source version: `{self.source_version}`",
            f"- Prompt registry hash: `{self.registry_hash or 'n/a'}`",
            f"- Annotator model fingerprint: `{self.model_fingerprint or 'n/a'}`",
            "",
            "## Counts",
        ]
        for key in sorted(counts):
            lines.append(f"- {key}: {counts[key]}")
        lines += [
            "",
            "## Label provenance",
            "All CelebA-style predictions are **weak labels** produced by a frozen",
            "protocol and stored under `extended_attributes.celeba40.*`. Source",
            "annotations (e.g. `source_attributes.fairface.*`) are never overwritten.",
            "Observations carry one of three tiers: high-confidence automatic,",
            "human-verified, or unlabeled/uncertain (`label: null`).",
            "",
            "## Sensitive and low-reliability labels",
            f"- Subjective/sensitive (inherit CelebA definitions and limitations): "
            f"{sorted(SENSITIVE_DATASET_LABELS)}",
            f"- Low-reliability source labels: {sorted(LOW_RELIABILITY)}",
            "`Male` is the CelebA binary annotation, not a person's self-identified",
            "gender. These labels must not be interpreted as ground-truth identity.",
            "",
            "## Images",
            "Images are referenced by URI/SHA-256 only and are **not** redistributed.",
            "",
            "## License and citation",
            f"- License: {self.license_text}",
        ]
        if self.citation:
            lines.append(f"- Citation: {self.citation}")
        path = self._path(f"{self.benchmark}_extension_card.md")
        ensure_parent_dir(path)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    # -- orchestration -------------------------------------------------- #

    @staticmethod
    def _sha256_file(path: str | Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        p = Path(path)
        if p.is_file():
            h.update(p.read_bytes())
            return h.hexdigest()
        # For directories, hash sorted file list + contents.
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    h.update(child.name.encode())
                    h.update(child.read_bytes())
            return h.hexdigest()
        return ""

    def export_all(
        self,
        samples: Sequence[CanonicalSample],
        *,
        train_qa: Sequence[Mapping[str, Any]] = (),
        eval_qa: Sequence[Mapping[str, Any]] = (),
        probe_rows: Sequence[Mapping[str, Any]] = (),
        split_results: Sequence[SplitResult] = (),
        provenance: Mapping[str, Any] | None = None,
    ) -> ExportRecord:
        record = ExportRecord(benchmark=self.benchmark, output_dir=self.output_dir)
        record.paths["annotations"] = self.write_image_annotations(samples)
        record.counts["annotation_rows"] = sum(
            len(s.visual_attributes) for s in samples
        )
        train_path, eval_path = self.write_visual_qa(train_qa, eval_qa)
        record.paths["qa_train"] = train_path
        record.paths["qa_eval"] = eval_path
        record.counts["qa_train"] = len(train_qa)
        record.counts["qa_eval"] = len(eval_qa)
        record.paths["route_probes"] = self.write_route_probes(probe_rows)
        record.counts["route_probes"] = len(probe_rows)
        record.paths["splits"] = self.write_splits(split_results)
        record.counts["splits"] = len(split_results)
        record.paths["dataset_card"] = self.render_dataset_card(record.counts)

        # R13: fully self-consistent finalization —
        # 1. Register checksums/manifest paths *before* serialization so the
        #    on-disk manifest matches the returned in-memory path map.
        # 2. Write the final export manifest (references checksums.json).
        # 3. Write checksums.json once: covers every released artifact
        #    including the final manifest, excludes itself, relative paths.

        checksums_path = self._path(f"{self.benchmark}_checksums.json")
        manifest_path = self._path(f"{self.benchmark}_export_manifest.json")
        record.paths["checksums"] = str(checksums_path)
        record.paths["manifest"] = str(manifest_path)

        # Step 1: write the final export manifest.
        manifest = record.to_dict()
        if provenance:
            manifest["provenance"] = dict(provenance)
        write_json(manifest, manifest_path)

        # Step 2: checksums over all artifacts incl. the final manifest,
        # excluding checksums.json itself; relative paths for portability.
        checksums: dict[str, str] = {}
        for key, path_str in sorted(record.paths.items()):
            if key == "checksums":
                continue  # checksums.json never hashes itself
            checksums[self._relative_path(path_str)] = self._sha256_file(path_str)
        write_json(checksums, checksums_path)

        return record

    def _relative_path(self, path_str: str) -> str:
        """Return a path relative to the output directory for portability."""
        try:
            return str(Path(path_str).relative_to(self.output_dir))
        except ValueError:
            # If the path is not under output_dir, return the filename.
            return Path(path_str).name
