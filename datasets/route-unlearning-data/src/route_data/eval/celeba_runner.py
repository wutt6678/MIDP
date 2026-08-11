"""CelebA-40 evaluation orchestrator (coding plan sections 8.5, 8.9, 22.3).

The runner ties together the model backend, prompt registry, strict parsers,
resumable Parquet shards, metric suite, and report writers. It evaluates one
VisionLanguageModel over the selected CelebA attributes and split, then emits a
reproducible run bundle.

Execution guarantees (plan section 8.5):
- one prediction record per (image_sha256, model_fingerprint, prompt_id,
  scoring_mode);
- incremental Parquet shards; a run can resume after interruption by scanning
  existing shards for completed keys;
- the full prediction set is never held in memory during inference.

The runner is backend-agnostic and unit-testable with a stub model: no real
weights are required to exercise the control flow.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ConfigError, EvaluationConfig, RunConfig
from ..constants.celeba_attributes import CELEBA_ATTRIBUTE_SET, CELEBA_ATTRIBUTES
from ..data.io import ParquetShardWriter, read_shards
from ..models.base import VisionLanguageModel
from ..models.scoring import binary_probability
from ..prompts.parsers import (
    PARSE_OK,
    parse_binary_answer,
    parse_grouped_json,
)
from ..prompts.registry import PromptRegistry
from .metrics import compute_binary_metrics, macro_average
from .reports import write_run_bundle

__all__ = ["CANDIDATES", "CelebaRunner"]

# Full candidate strings scored in candidate mode (plan section 8.4). Neither is
# assumed to be a single token.
CANDIDATES: tuple[str, ...] = (" yes", " no")
_POSITIVE_CANDIDATE = " yes"
_NEGATIVE_CANDIDATE = " no"

# Fixed prediction-row columns (plan 8.5 keys plus resumption/provenance extras).
ROW_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "image_filename",
    "image_sha256",
    "split",
    "attribute",
    "prompt_id",
    "scoring_mode",
    "prediction",
    "p_positive",
    "raw_text",
    "parse_status",
    "latency_ms",
    "label",
    "model_fingerprint",
)


class CelebaRunner:
    """Run a CelebA-40 evaluation and summarize it into a run bundle.

    Args:
        run_config: fully validated run configuration.
        backend: an instantiated :class:`VisionLanguageModel`.
        registry: prompt registry (binary / grouped / route templates).
        manifest: **wide** CelebA manifest (one row per image) with ``attr::<A>``
            label columns, ``image_filename``, ``image_path``, and optionally
            ``image_sha256`` / ``split``.
        output_dir: override for the prediction shard directory. Defaults to
            ``run_config.evaluation.output_dir``.
    """

    def __init__(
        self,
        run_config: RunConfig,
        backend: VisionLanguageModel,
        registry: PromptRegistry,
        manifest: pd.DataFrame,
        output_dir: str | Path | None = None,
    ):
        self.run_config = run_config
        self.backend = backend
        self.registry = registry
        self.eval_cfg: EvaluationConfig = run_config.evaluation
        self.output_dir = Path(output_dir or self.eval_cfg.output_dir)

        self._binary_family = _family_from_path(run_config.prompts.binary)
        self._grouped_family = _family_from_path(run_config.prompts.grouped)

        # Model identity is fixed for the whole run (caching + manifests).
        self._model_fingerprint: dict[str, Any] = backend.fingerprint()
        self._fingerprint_id: str = str(self._model_fingerprint.get("fingerprint_id", "unknown"))

        self._images = self._prepare_images(manifest)

    # ------------------------------------------------------------------ #
    # Preparation
    # ------------------------------------------------------------------ #

    def _prepare_images(self, manifest: pd.DataFrame) -> pd.DataFrame:
        """Filter the wide manifest to the configured split and validate columns."""
        df = manifest
        for col in ("image_filename", "image_path"):
            if col not in df.columns:
                raise ConfigError(f"CelebA manifest is missing required column '{col}'")

        split = self.eval_cfg.split
        if "split" in df.columns:
            filtered = df[df["split"] == split]
            if filtered.empty:
                raise ConfigError(
                    f"No manifest rows match split={split!r}; refusing to run on an "
                    "empty split. Check evaluation.split or the prepared manifest."
                )
            df = filtered

        # Ensure every selected attribute has a label column (fail loudly).
        for attr in self.selected_attributes():
            if f"attr::{attr}" not in df.columns:
                raise ConfigError(f"Manifest is missing label column 'attr::{attr}'")

        # One row per image; drop duplicate filenames defensively.
        return (
            df.drop_duplicates(subset="image_filename")
            .sort_values("image_filename")
            .reset_index(drop=True)
        )

    def selected_attributes(self) -> list[str]:
        cfg_attrs = self.eval_cfg.attributes
        if cfg_attrs == "all":
            return list(CELEBA_ATTRIBUTES)
        unknown = [a for a in cfg_attrs if a not in CELEBA_ATTRIBUTE_SET]
        if unknown:
            raise ConfigError(f"Unknown CelebA attributes requested: {unknown}")
        return list(cfg_attrs)

    @property
    def n_images(self) -> int:
        return len(self._images)

    # ------------------------------------------------------------------ #
    # Cache / resumption keys (plan section 8.5)
    # ------------------------------------------------------------------ #

    def _cache_key(self, image_sha256: Any, prompt_id: str) -> tuple:
        return (str(image_sha256), self._fingerprint_id, prompt_id, self.eval_cfg.scoring)

    def _load_done_keys(self) -> set[tuple]:
        """Scan existing shards for completed cache keys (resumption)."""
        import pyarrow.parquet as pq

        done: set[tuple] = set()
        want = {"image_sha256", "prompt_id", "scoring_mode", "model_fingerprint"}
        for path in sorted(self.output_dir.glob("part-*.parquet")):
            table = pq.read_table(path, columns=[c for c in want if c in pq.ParquetFile(path).schema.names])
            cols = table.to_pydict()
            n = len(next(iter(cols.values()), []))
            for i in range(n):
                done.add(
                    (
                        str(cols.get("image_sha256", [""] * n)[i]),
                        str(cols.get("model_fingerprint", [self._fingerprint_id] * n)[i]),
                        str(cols.get("prompt_id", [""] * n)[i]),
                        str(cols.get("scoring_mode", [self.eval_cfg.scoring] * n)[i]),
                    )
                )
        return done

    # ------------------------------------------------------------------ #
    # Prompt ids
    # ------------------------------------------------------------------ #

    def _binary_prompt_id(self, attribute: str, variant_index: int | None) -> str:
        base = f"{self._binary_family}.{attribute}"
        return base if variant_index is None else f"{base}#{variant_index}"

    def _grouped_prompt_id(self, group: str) -> str:
        return f"{self._grouped_family}.{group}"

    # ------------------------------------------------------------------ #
    # Inference loop
    # ------------------------------------------------------------------ #

    def run(
        self,
        limit: int | None = None,
        variant_index: int | None = None,
        shard_size: int = 1000,
    ) -> int:
        """Evaluate the configured split and write prediction shards.

        Returns the number of newly written rows. Completed cache keys are
        skipped so an interrupted run resumes where it left off.
        """
        if self.eval_cfg.mode == "grouped_json" and self.eval_cfg.scoring == "candidate":
            raise ConfigError(
                "grouped_json mode is generation-only; candidate scoring is not "
                "defined for multi-attribute JSON answers."
            )

        done = self._load_done_keys() if self.eval_cfg.resume else set()
        images = self._images if limit is None else self._images.head(limit)

        written = 0
        with ParquetShardWriter(self.output_dir, shard_size=shard_size) as writer:
            if self.eval_cfg.mode == "single_attribute":
                written = self._run_single(images, variant_index, done, writer)
            else:
                written = self._run_grouped(images, done, writer)
        return written

    def _label_for(self, image_row: pd.Series, attribute: str) -> int | None:
        col = f"attr::{attribute}"
        if col not in image_row.index:
            return None
        value = image_row[col]
        return None if pd.isna(value) else int(value)

    def _base_row(self, image_row: pd.Series, attribute: str, prompt_id: str) -> dict[str, Any]:
        sha = image_row.get("image_sha256")
        return {
            "sample_id": image_row["image_filename"],
            "image_filename": image_row["image_filename"],
            "image_sha256": None if sha is None or (isinstance(sha, float) and pd.isna(sha)) else str(sha),
            "split": image_row.get("split"),
            "attribute": attribute,
            "prompt_id": prompt_id,
            "scoring_mode": self.eval_cfg.scoring,
            "prediction": None,
            "p_positive": None,
            "raw_text": "",
            "parse_status": None,
            "latency_ms": None,
            "label": self._label_for(image_row, attribute),
            "model_fingerprint": self._fingerprint_id,
        }

    def _run_single(self, images, variant_index, done, writer) -> int:
        attributes = self.selected_attributes()
        written = 0
        for _, image_row in images.iterrows():
            image = _load_image(image_row["image_path"])
            sha = image_row.get("image_sha256")
            for attribute in attributes:
                prompt_id = self._binary_prompt_id(attribute, variant_index)
                if self._cache_key(sha, prompt_id) in done:
                    continue
                prompt = self.registry.binary_prompt(attribute, variant_index)
                row = self._base_row(image_row, attribute, prompt_id)
                started = time.perf_counter()
                if self.eval_cfg.scoring == "candidate":
                    resp = self.backend.score_candidates(image, prompt, list(CANDIDATES))
                    scores = {c.candidate: c.log_probability for c in (resp.candidate_scores or [])}
                    if _POSITIVE_CANDIDATE not in scores or _NEGATIVE_CANDIDATE not in scores:
                        raise RuntimeError(
                            f"Backend did not return scores for both candidates on {prompt_id}"
                        )
                    p_pos = binary_probability(scores[_POSITIVE_CANDIDATE], scores[_NEGATIVE_CANDIDATE])
                    row["p_positive"] = float(p_pos)
                    row["prediction"] = int(p_pos >= 0.5)
                    row["parse_status"] = PARSE_OK
                    row["raw_text"] = _POSITIVE_CANDIDATE if row["prediction"] else _NEGATIVE_CANDIDATE
                else:
                    resp = self.backend.generate(image, prompt)
                    parsed = parse_binary_answer(resp.text)
                    row["prediction"] = parsed.label
                    row["parse_status"] = parsed.parse_status
                    row["raw_text"] = parsed.raw_text
                row["latency_ms"] = _latency(resp, started)
                writer.add({k: row.get(k) for k in ROW_COLUMNS})
                written += 1
        return written

    def _run_grouped(self, images, done, writer) -> int:
        selected = set(self.selected_attributes())
        written = 0
        for _, image_row in images.iterrows():
            image = _load_image(image_row["image_path"])
            sha = image_row.get("image_sha256")
            for group in self.registry.grouped_names():
                prompt, keys = self.registry.grouped_prompt(group)
                keys = [k for k in keys if k in selected]
                if not keys:
                    continue
                prompt_id = self._grouped_prompt_id(group)
                if self._cache_key(sha, prompt_id) in done:
                    continue
                started = time.perf_counter()
                resp = self.backend.generate(image, prompt)
                values, parse_status = parse_grouped_json(resp.text, keys)
                latency = _latency(resp, started)
                for attribute in keys:
                    row = self._base_row(image_row, attribute, prompt_id)
                    row["prediction"] = values.get(attribute)
                    row["parse_status"] = parse_status
                    row["raw_text"] = resp.text if parse_status != PARSE_OK else ""
                    row["latency_ms"] = latency
                    writer.add({k: row.get(k) for k in ROW_COLUMNS})
                    written += 1
        return written

    # ------------------------------------------------------------------ #
    # Summarization / reporting
    # ------------------------------------------------------------------ #

    def summarize(self) -> dict[str, Any]:
        """Aggregate all prediction shards into per-attribute + macro metrics."""
        df = read_shards(self.output_dir)
        if df.empty:
            raise RuntimeError(
                f"No prediction shards found under {self.output_dir}; run inference first."
            )
        per_attribute: dict[str, dict[str, Any]] = {}
        for attribute in sorted(df["attribute"].unique()):
            sub = df[df["attribute"] == attribute]
            total_queries = len(sub)
            parseable = sub[sub["parse_status"] == PARSE_OK].copy()
            parse_failures = total_queries - len(parseable)
            y_true = parseable["label"].astype("Int64")
            y_pred = parseable["prediction"].astype("Int64")
            # Drop rows lacking a ground-truth label or a hard prediction.
            keep = y_true.notna() & y_pred.notna()
            y_true = y_true[keep].astype(int).tolist()
            y_pred = y_pred[keep].astype(int).tolist()
            p_pos = parseable.loc[keep, "p_positive"]
            p_positive = None if p_pos.isna().all() else p_pos.astype(float).tolist()
            latency = sub["latency_ms"].dropna()
            per_attribute[attribute] = compute_binary_metrics(
                y_true,
                y_pred,
                p_positive=p_positive,
                parse_failures=parse_failures,
                total_queries=total_queries,
                latency_ms=float(latency.mean()) if len(latency) else None,
            )
        macro = macro_average(per_attribute)
        return {"per_attribute": per_attribute, "macro": macro}

    def data_fingerprint(self) -> dict[str, Any]:
        """Compact identity of the evaluated data slice (plan section 22.3)."""
        import hashlib

        images = self._images
        parts = [f"split={self.eval_cfg.split}", f"n_images={len(images)}"]
        sha_col = images["image_sha256"] if "image_sha256" in images.columns else None
        if sha_col is not None and sha_col.notna().any():
            digest_input = "|".join(str(v) for v in sha_col.dropna().tolist())
        else:
            digest_input = "|".join(images["image_filename"].tolist())
        digest = hashlib.sha256(digest_input.encode()).hexdigest()[:16]
        return {
            "dataset": self.run_config.data.name,
            "source_version": self.run_config.data.source_version,
            "split": self.eval_cfg.split,
            "n_images": len(images),
            "n_attributes": len(self.selected_attributes()),
            "data_fingerprint_id": digest,
            "parts": parts,
        }

    def write_bundle(self, bundle_dir: str | Path | None = None) -> dict[str, Path]:
        """Compute metrics and write the reproducible run bundle (plan 22.3)."""
        summary = self.summarize()
        bundle_dir = Path(bundle_dir or (self.output_dir.parent / "run_bundle"))
        return write_run_bundle(
            output_dir=bundle_dir,
            run_config=self.run_config,
            model_fingerprint=self._model_fingerprint,
            data_fingerprint=self.data_fingerprint(),
            per_attribute=summary["per_attribute"],
            macro=summary["macro"],
            run_name=self.run_config.run.name,
            prompt_registry_hash=self.registry.registry_hash(),
            n_images=self.n_images,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _family_from_path(path: str | None) -> str:
    return Path(path).stem if path else "prompt"


def _load_image(path: Any):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _latency(resp: Any, started: float) -> float:
    """Prefer backend-reported latency; fall back to wall-clock timing."""
    meta = getattr(resp, "metadata", None) or {}
    reported = meta.get("latency_ms")
    if isinstance(reported, (int, float)):
        return float(reported)
    return (time.perf_counter() - started) * 1000.0
