"""PPU-Bench adapter (repair plan B27-B33).

PPU-Bench is released as a multi-configuration Hugging Face dataset of real
public figures. Every run must pin dataset, configuration, split, and
revision (plan B27); released rows carry:

    sample_id, subject_id, subject, task_type, modality, question,
    answer_text, answer_label, option_a..option_d, image_002/image_003/...

(plan B28). Flattening rules:

- options become an ordered non-null list from ``option_a`` through
  ``option_d`` while keeping ``answer_label`` and ``answer_text`` (B29);
- image columns are enumerated dynamically (``^image(_\\d+)?$``, sorted
  numerically) and one canonical record is emitted per non-null view with
  IDs ``ppubench:<config>:<sample_id>:<image_field>`` — a
  ``ppubench:<config>:<sample_id>:text_only`` record is emitted only after
  every image column has been checked (B30);
- the source modality passes through a fixed mapping table; text-only is
  never inferred before all image columns are examined (B31);
- subject ID/name and task type map directly; config, source split,
  original sample ID, original modality, image field, and answer label are
  preserved in source_metadata (B32).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..schemas import CanonicalSample
from .base import (
    AdapterError,
    BenchmarkAdapter,
    SourceContext,
    image_view_uris,
    register_adapter,
)

_IMAGE_COLUMN = re.compile(r"^image(_\d+)?$")
_OPTION_FIELDS = ("option_a", "option_b", "option_c", "option_d")

# Fixed modality mapping table (plan B31): source value -> canonical value.
_MODALITY_TABLE = {
    "image_text": "image_text",
    "image-text": "image_text",
    "image + text": "image_text",
    "multimodal": "image_text",
    "vision_language": "image_text",
    "image": "image_text",
    "text_only": "text_only",
    "text-only": "text_only",
    "text": "text_only",
    "image_only": "image_only",
}


@register_adapter("ppubench")
class PpubenchAdapter(BenchmarkAdapter):
    adapter_version = "ppubench-v2"
    required_fields = ("sample_id", "subject_id")

    def __init__(self, config):
        super().__init__(config)
        # Plan B27: configuration and split must be explicit on every run.
        if not self.hf_config_name():
            raise AdapterError(
                "[ppubench] extras.hf_config_name (the PPU-Bench HF "
                "configuration) is required; repair plan B27"
            )
        if not self._split_name():
            raise AdapterError(
                "[ppubench] extras.hf_split is required; repair plan B27"
            )

    # -- configuration -------------------------------------------------- #

    def _split_name(self) -> str | None:
        split = self.config.extras.get("hf_split") or self.config.extras.get(
            "split"
        )
        return str(split) if split else None

    def _source_path(self) -> Path:
        root = self.config.require_root()
        source_file = self.config.extras.get("source_file")
        if not source_file:
            raise AdapterError(
                "[ppubench] data config must set extras.source_file for "
                "local loading (repair plan C2)"
            )
        return root / source_file

    def _images_root(self) -> Path:
        images_root = self.config.extras.get("images_root")
        root = self.config.require_root()
        return root / images_root if images_root else root

    def _cache_dir(self) -> Path:
        return self.config.require_root() / "derived_images"

    # -- explicit source reader (plan B4/B27) ---------------------------- #

    def source_files(self) -> Sequence[Path]:
        # Fail before any model loading when the source is missing (C4).
        self.source_revision()
        path = self._source_path()
        if not path.exists():
            raise AdapterError(
                f"[ppubench] incompatible source layout; configured source "
                f"file does not exist: {path}"
            )
        return [path]

    # -- options (plan B29) ----------------------------------------------- #

    @staticmethod
    def _options(row: Mapping[str, Any]) -> list[str] | None:
        """Ordered non-null options from option_a through option_d."""
        ordered: list[str] = []
        for field_name in _OPTION_FIELDS:
            value = row.get(field_name)
            if value in (None, ""):
                continue
            ordered.append(str(value))
        return ordered or None

    # -- image columns (plan B30) ------------------------------------------ #

    @staticmethod
    def _image_columns(row: Mapping[str, Any]) -> list[str]:
        """Dynamic image column enumeration, numerically sorted."""
        columns = [str(key) for key in row if _IMAGE_COLUMN.match(str(key))]

        def sort_key(name: str) -> int:
            match = _IMAGE_COLUMN.match(name)
            suffix = (match.group(1) or "") if match else ""
            return int(suffix.lstrip("_")) if suffix else 0

        return sorted(columns, key=sort_key)

    def _resolve_view(self, value: Any) -> list[str]:
        uris = image_view_uris(value, cache_dir=self._cache_dir())
        root = self._images_root()
        resolved: list[str] = []
        for uri in uris:
            candidate = Path(uri)
            if candidate.is_absolute() or candidate.exists():
                resolved.append(str(candidate))
                continue
            via_root = root / candidate
            resolved.append(str(via_root) if via_root.exists() else uri)
        return resolved

    # -- modality (plan B31) ----------------------------------------------- #

    @staticmethod
    def _mapped_modality(raw: Any) -> str | None:
        if raw in (None, ""):
            return None
        key = str(raw).strip().lower()
        return _MODALITY_TABLE.get(key)

    # -- flattening --------------------------------------------------------- #

    def to_samples(
        self,
        row: Mapping[str, Any],
        *,
        source_context: SourceContext,
    ) -> Iterator[CanonicalSample]:
        config = str(self.hf_config_name())
        split = self._split_name() or "unassigned"
        sample_id = str(row.get("sample_id"))
        subject_id = str(row.get("subject_id"))
        subject = row.get("subject")
        task_type = row.get("task_type") or "unlearning_qa"
        raw_modality = row.get("modality")
        mapped = self._mapped_modality(raw_modality)
        options = self._options(row)
        answer_label = row.get("answer_label")
        answer_text = row.get("answer_text")
        question = row.get("question")

        def base_metadata(image_field: str | None) -> dict[str, Any]:
            metadata = self.context_metadata(source_context)
            metadata.update(
                {
                    "config": config,
                    "source_split": split,
                    "original_sample_id": sample_id,
                    "original_modality": raw_modality,
                    "answer_label": answer_label,
                }
            )
            if image_field is not None:
                metadata["image_field"] = image_field
            return metadata

        def make_record(
            *, suffix: str, modality: str, image_uri: str | None, image_field: str | None
        ) -> CanonicalSample:
            source_id = f"ppubench:{config}:{sample_id}:{suffix}"
            return CanonicalSample(
                benchmark="ppubench",
                source_sample_id=source_id,
                source_record_id=sample_id,
                identity_id=subject_id,
                provenance=self.provenance(
                    source_id, source_subset=config, context=source_context
                ),
                source_subset=config,
                identity_name=str(subject) if subject not in (None, "") else None,
                image_id=image_field,
                image_uri=image_uri,
                modality=modality,
                task_type=str(task_type),
                question=str(question) if question not in (None, "") else None,
                answer_text=str(answer_text) if answer_text not in (None, "") else None,
                answer_label=answer_label,
                options=options,
                split=split,
                source_metadata=base_metadata(image_field),
            ).validate()

        # One record per non-null image view; never infer text-only before
        # every image column has been checked (plans B30/B31).
        emitted = False
        for column in self._image_columns(row):
            value = row.get(column)
            if value in (None, "", [], {}):
                continue
            for uri in self._resolve_view(value):
                yield make_record(
                    suffix=column,
                    modality=mapped if mapped in ("image_text", "image_only") else "image_text",
                    image_uri=uri,
                    image_field=column,
                )
                emitted = True
        if not emitted:
            yield make_record(
                suffix="text_only",
                modality="text_only",
                image_uri=None,
                image_field=None,
            )
