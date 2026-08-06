"""MLLMU-Bench adapter (repair plan B19-B26).

Reads the released MLLMU-Bench layout instead of assuming a flat source:

- runs must pin an explicit configuration from the validated matrix
  (``Full_Set``, ``Test_Set``, ``Retain_Set``, ``forget_5/10/15``,
  ``retain_85/90/95``, ``ft_Data``) plus revision (plan B19);
- rows carry ``image``/``images``, ``ID``, ``Directory``, ``biography`` and
  nested task blocks ``Classification_Task`` / ``Generation_Task`` /
  ``Mask_Task`` depending on configuration (plan B20);
- classification questions under ``Image_Textual_Questions`` and
  ``Pure_Text_Questions`` keep both the original answer label and the
  resolved answer text with options A-D (plan B21);
- generation items map ``Question``, ``Ground_Truth``, ``Type``; mask items
  stay separate and preserve ``Type`` (plans B22/B23);
- ``images`` sequences expand one record per view with deterministic IDs
  ``mllmu:<config>:<ID>:<task_family>:<modality>:<item_index>:<view_index>``
  (plan B24);
- raw biography is preserved whole, a name is taken only from explicit
  source content, and the subset comes from the HF configuration, never
  guessed row fields (plan B25).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..schemas import CanonicalSample, ProfileFact
from .base import (
    AdapterError,
    BenchmarkAdapter,
    SourceContext,
    image_view_uris,
    register_adapter,
)

# Validated configuration matrix (plan B19).
_KNOWN_CONFIGS = frozenset(
    {
        "Full_Set",
        "Test_Set",
        "Retain_Set",
        "forget_5",
        "forget_10",
        "forget_15",
        "retain_95",
        "retain_90",
        "retain_85",
        "ft_Data",
    }
)

# Explicit split semantics per configuration; extras.split overrides.
_SPLIT_BY_CONFIG = {
    "forget_5": "forget",
    "forget_10": "forget",
    "forget_15": "forget",
    "retain_85": "retain_train",
    "retain_90": "retain_train",
    "retain_95": "retain_train",
    "Retain_Set": "retain_eval",
    "Test_Set": "test",
    "ft_Data": "finetune",
    "Full_Set": "unassigned",
}

_OPTION_LETTERS = ("A", "B", "C", "D")


@register_adapter("mllmu")
class MllmuAdapter(BenchmarkAdapter):
    adapter_version = "mllmu-v2"
    required_fields = ("ID",)

    # -- configuration -------------------------------------------------- #

    def config_name(self) -> str:
        name = self.hf_config_name()
        if not name:
            raise AdapterError(
                "[mllmu] data config must pin hf_config_name (one of "
                f"{sorted(_KNOWN_CONFIGS)}); repair plan B19"
            )
        if name not in _KNOWN_CONFIGS:
            raise AdapterError(
                f"[mllmu] hf_config_name={name!r} is not part of the "
                f"validated configuration matrix: {sorted(_KNOWN_CONFIGS)}"
            )
        return name

    def subset_name(self) -> str:
        """Subset derives from the HF configuration (plan B25)."""
        return str(self.config.extras.get("subset") or self.config_name())

    def _split_for(self) -> str:
        return str(
            self.config.extras.get("split")
            or _SPLIT_BY_CONFIG.get(self.config_name(), "unassigned")
        )

    def _source_path(self) -> Path:
        root = self.config.require_root()
        source_file = self.config.extras.get("source_file")
        if not source_file:
            raise AdapterError(
                "[mllmu] data config must set extras.source_file for local "
                "loading (repair plan C2)"
            )
        return root / source_file

    def _cache_dir(self) -> Path:
        return self.config.require_root() / "derived_images"

    # -- explicit source reader (plan B4) --------------------------------- #

    def source_files(self) -> Sequence[Path]:
        path = self._source_path()
        if not path.exists():
            raise AdapterError(
                f"[mllmu] incompatible source layout; configured source file "
                f"does not exist: {path}"
            )
        return [path]

    # -- image views (plan B24) ------------------------------------------- #

    def _row_views(self, row: Mapping[str, Any]) -> list[str]:
        """Normalize ``image``/``images`` into ordered view URIs."""
        value = row.get("images")
        if value in (None, "", [], {}):
            value = row.get("image")
        uris = image_view_uris(value, cache_dir=self._cache_dir())
        directory = row.get("Directory")
        root = self.config.require_root()
        resolved: list[str] = []
        for uri in uris:
            candidate = Path(uri)
            if candidate.is_absolute() or candidate.exists():
                resolved.append(str(candidate))
                continue
            if directory:
                via_dir = root / str(directory) / candidate
                if via_dir.exists():
                    resolved.append(str(via_dir))
                    continue
            via_root = root / candidate
            if via_root.exists():
                resolved.append(str(via_root))
                continue
            # Keep the raw reference so the record stays traceable.
            resolved.append(uri)
        return resolved

    # -- profile (plan B25) ----------------------------------------------- #

    def _biography_facts(self, row: Mapping[str, Any]) -> list[ProfileFact]:
        biography = row.get("biography")
        if biography in (None, "", [], {}):
            return []
        if isinstance(biography, str):
            value = biography
        else:
            value = json.dumps(biography, ensure_ascii=False, sort_keys=True)
        return [
            ProfileFact(
                fact_id="mllmu_biography",
                relation="biography",
                value=value,
                privacy_class="identity_biography",
                source="source_human",
                forgettable=True,
            ).validate()
        ]

    @staticmethod
    def _identity_name(row: Mapping[str, Any]) -> str | None:
        """Name only from explicit source content — never parsed from the
        biography (plan B25)."""
        name = row.get("name") or row.get("Name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        return None

    # -- task block helpers ------------------------------------------------ #

    @staticmethod
    def _task_items(block: Any, where: str) -> list[tuple[str | None, Any]]:
        """Normalize a task block into ``(type_key, item)`` pairs."""
        if block in (None, "", [], {}):
            return []
        if isinstance(block, list):
            return [(None, item) for item in block]
        if isinstance(block, Mapping):
            pairs: list[tuple[str | None, Any]] = []
            for key in sorted(block):
                value = block[key]
                items = value if isinstance(value, list) else [value]
                pairs.extend((str(key), item) for item in items)
            return pairs
        raise AdapterError(f"[mllmu] {where} must be a list or mapping of items")

    @staticmethod
    def _classification_options(
        item: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Options A-D -> (letter->text map, ordered non-null list)."""
        options_map: dict[str, Any] = {}
        raw = item.get("Options") or item.get("options")
        if isinstance(raw, Mapping):
            options_map = {str(k).strip().upper(): v for k, v in raw.items()}
        elif isinstance(raw, (list, tuple)):
            options_map = dict(zip(_OPTION_LETTERS, raw))
        else:
            for letter in _OPTION_LETTERS:
                value = item.get(f"Option_{letter}") or item.get(f"option_{letter.lower()}")
                if value is not None:
                    options_map[letter] = value
        options_map = {
            k: v for k, v in options_map.items() if v not in (None, "")
        }
        ordered = [str(options_map[letter]) for letter in _OPTION_LETTERS if letter in options_map]
        return options_map, ordered

    # -- record emission ---------------------------------------------------- #

    def _record(
        self,
        *,
        identity_id: str,
        identity_name: str | None,
        config: str,
        subset: str,
        split: str,
        facts: list[ProfileFact],
        task_family: str,
        modality: str,
        item_index: int,
        view_index: int,
        image_uri: str | None,
        question: Any,
        answer_text: Any,
        answer_label: Any,
        options: list[str] | None,
        task_source: str,
        item_type: str | None,
        directory: Any,
        context: SourceContext,
    ) -> CanonicalSample:
        source_id = (
            f"mllmu:{config}:{identity_id}:{task_family}:{modality}"
            f":{item_index}:{view_index}"
        )
        metadata = self.context_metadata(context)
        metadata.update(
            {
                "config": config,
                "subset": subset,
                "task_family": task_family,
                "task_source": task_source,
                "item_index": item_index,
                "view_index": view_index,
            }
        )
        if item_type is not None:
            metadata["task_type_source"] = item_type
        if directory not in (None, ""):
            metadata["directory"] = str(directory)
        if answer_label is not None:
            metadata["original_answer_label"] = answer_label
        return CanonicalSample(
            benchmark="mllmu",
            source_sample_id=source_id,
            source_record_id=identity_id,
            identity_id=identity_id,
            provenance=self.provenance(
                source_id, source_subset=subset, context=context
            ),
            source_subset=subset,
            identity_name=identity_name,
            image_id=Path(image_uri).stem if image_uri else None,
            image_uri=image_uri,
            profile_facts=list(facts),
            modality=modality,
            task_type=task_family,
            question=str(question) if question not in (None, "") else None,
            answer_text=str(answer_text) if answer_text not in (None, "") else None,
            answer_label=answer_label,
            options=options,
            split=split,
            source_metadata=metadata,
        ).validate()

    def _emit_with_views(
        self,
        *,
        views: list[str],
        modality: str,
        record_kwargs: dict[str, Any],
        where: str,
    ) -> Iterator[CanonicalSample]:
        """One canonical record per image view (plan B24)."""
        if modality == "image_text":
            if not views:
                raise AdapterError(
                    f"[mllmu] image-textual item {where} has no usable image "
                    "views ('image'/'images' missing or empty)"
                )
            for view_index, uri in enumerate(views):
                yield self._record(
                    **record_kwargs,
                    modality=modality,
                    view_index=view_index,
                    image_uri=uri,
                )
        else:
            yield self._record(
                **record_kwargs,
                modality=modality,
                view_index=0,
                image_uri=None,
            )

    # -- flattening --------------------------------------------------------- #

    def to_samples(
        self,
        row: Mapping[str, Any],
        *,
        source_context: SourceContext,
    ) -> Iterator[CanonicalSample]:
        config = self.config_name()
        subset = self.subset_name()
        split = self._split_for()
        raw_id = row.get("ID")
        if raw_id in (None, ""):
            raise AdapterError(f"[mllmu] row lacks required 'ID' field: {row!r}")
        identity_id = f"{config}:{raw_id}"
        identity_name = self._identity_name(row)
        facts = self._biography_facts(row)
        views = self._row_views(row)
        directory = row.get("Directory")
        emitted = 0

        # -- Classification_Task (plan B21) ------------------------------- #
        classification = row.get("Classification_Task")
        if classification not in (None, "", {}, []):
            if not isinstance(classification, Mapping):
                raise AdapterError(
                    "[mllmu] Classification_Task must map question groups "
                    "(Image_Textual_Questions / Pure_Text_Questions) to items"
                )
            for group in sorted(classification):
                if group == "Image_Textual_Questions":
                    modality = "image_text"
                elif group == "Pure_Text_Questions":
                    modality = "text_only"
                else:
                    raise AdapterError(
                        f"[mllmu] unsupported Classification_Task group "
                        f"{group!r}; expected Image_Textual_Questions or "
                        "Pure_Text_Questions"
                    )
                items = classification[group]
                if not isinstance(items, list):
                    raise AdapterError(
                        f"[mllmu] Classification_Task.{group} must be a list "
                        "of question items"
                    )
                for item_index, item in enumerate(items):
                    if not isinstance(item, Mapping):
                        raise AdapterError(
                            f"[mllmu] Classification_Task.{group}[{item_index}] "
                            "is not a question item object"
                        )
                    options_map, ordered = self._classification_options(item)
                    label = item.get("Correct_Answer")
                    if label is None:
                        label = item.get("Answer") or item.get("answer")
                    label_str = (
                        str(label).strip().upper()
                        if label is not None
                        else None
                    )
                    # Preserve the original label AND the resolved text (B21).
                    if label_str and label_str in options_map:
                        answer_text = options_map[label_str]
                    else:
                        answer_text = label
                    record_kwargs = dict(
                        identity_id=identity_id,
                        identity_name=identity_name,
                        config=config,
                        subset=subset,
                        split=split,
                        facts=facts,
                        task_family="classification_qa",
                        item_index=item_index,
                        question=item.get("Question") or item.get("question"),
                        answer_text=answer_text,
                        answer_label=label,
                        options=ordered or None,
                        task_source=f"Classification_Task.{group}",
                        item_type=None,
                        directory=directory,
                        context=source_context,
                    )
                    where = f"Classification_Task.{group}[{item_index}]"
                    yield from self._emit_with_views(
                        views=views,
                        modality=modality,
                        record_kwargs=record_kwargs,
                        where=where,
                    )
                    emitted += 1

        # -- Generation_Task (plan B22) ------------------------------------ #
        for item_type, item in self._task_items(
            row.get("Generation_Task"), "Generation_Task"
        ):
            resolved_type = (
                item.get("Type") or item.get("type") or item_type
                if isinstance(item, Mapping)
                else item_type
            )
            record_kwargs = dict(
                identity_id=identity_id,
                identity_name=identity_name,
                config=config,
                subset=subset,
                split=split,
                facts=facts,
                task_family="generation_qa",
                item_index=emitted,
                question=item.get("Question") or item.get("question") if isinstance(item, Mapping) else None,
                answer_text=item.get("Ground_Truth") or item.get("Answer") if isinstance(item, Mapping) else None,
                answer_label=None,
                options=None,
                task_source="Generation_Task",
                item_type=str(resolved_type) if resolved_type else None,
                directory=directory,
                context=source_context,
            )
            yield from self._emit_with_views(
                views=views,
                modality="image_text" if views else "text_only",
                record_kwargs=record_kwargs,
                where="Generation_Task item",
            )
            emitted += 1

        # -- Mask_Task (plan B23): separate records, preserve Type -------- #
        for item_type, item in self._task_items(row.get("Mask_Task"), "Mask_Task"):
            resolved_type = (
                item.get("Type") or item.get("type") or item_type
                if isinstance(item, Mapping)
                else item_type
            )
            record_kwargs = dict(
                identity_id=identity_id,
                identity_name=identity_name,
                config=config,
                subset=subset,
                split=split,
                facts=facts,
                task_family="mask_qa",
                item_index=emitted,
                question=item.get("Question") or item.get("question") if isinstance(item, Mapping) else None,
                answer_text=item.get("Ground_Truth") or item.get("Answer") if isinstance(item, Mapping) else None,
                answer_label=None,
                options=None,
                task_source="Mask_Task",
                item_type=str(resolved_type) if resolved_type else None,
                directory=directory,
                context=source_context,
            )
            yield from self._emit_with_views(
                views=views,
                modality="image_text" if views else "text_only",
                record_kwargs=record_kwargs,
                where="Mask_Task item",
            )
            emitted += 1

        # -- flat QA rows (some configurations, plan B20) ----------------- #
        question = row.get("question")
        answer = row.get("answer")
        if emitted == 0 and question not in (None, ""):
            record_kwargs = dict(
                identity_id=identity_id,
                identity_name=identity_name,
                config=config,
                subset=subset,
                split=split,
                facts=facts,
                task_family="profile_qa",
                item_index=0,
                question=question,
                answer_text=answer,
                answer_label=None,
                options=None,
                task_source="row",
                item_type=None,
                directory=directory,
                context=source_context,
            )
            yield from self._emit_with_views(
                views=views,
                modality="image_text" if views else "text_only",
                record_kwargs=record_kwargs,
                where="flat row QA",
            )
