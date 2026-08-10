"""FAIRGET adapter (repair plan B6-B12).

Reads the released FAIRGET layout instead of a guessed flat mapping:

    data/dataset.json            # one nested record per identity
    train_images/<ID>/*          # training views per identity
    test_images/<ID>/*           # evaluation views per identity
    <split_file>                 # selected official split JSON

with nested access patterns ``row["ID"]``, ``row["train"][media_type]
[attribute]`` and ``row["eval"][media_type][task][attribute]`` (plan B6).

Flattening rules:

- training items map ``q -> question``, ``a -> answer_text``, ``gt ->
  answer_label``; ``q_words``/``a_words`` are preserved in source_metadata
  (plan B8);
- evaluation items keep media type, task, attribute, and item index
  (plan B9);
- image views are assigned deterministically — either every QA expanded
  across all views (``image_expansion_policy: all_views``) or one stable
  hash-assigned view (``hash_assigned``) — with the policy and seed recorded
  on every record; runtime randomness is never used (plan B10);
- official split membership comes from the configured split file, never from
  row order (plan B11);
- FairFace demographics attach to image-level records only, while the ten
  identity-linked textual attributes become :class:`ProfileFact` rows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..schemas import AttributeObservation, CanonicalSample, ProfileFact
from .base import AdapterError, BenchmarkAdapter, SourceContext, register_adapter

# Ten FAIRGET textual identity attributes (plan section 2.4 / 12.1).
_TEXTUAL_ATTRIBUTES = (
    "nationality",
    "occupation",
    "education",
    "relationship_status",
    "religion",
    "political_view",
    "gender_identity",
    "sexual_orientation",
    "health_condition",
    "financial_status",
)

_EXPANSION_POLICIES = ("all_views", "hash_assigned")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@register_adapter("fairget")
class FairgetAdapter(BenchmarkAdapter):
    adapter_version = "fairget-v2"
    required_fields = ("ID",)

    # -- configuration -------------------------------------------------- #

    def _dataset_path(self) -> Path:
        root = self.config.require_root()
        return root / (self.config.extras.get("source_file") or "data/dataset.json")

    def _split_path(self) -> Path | None:
        split_file = self.config.extras.get("split_file")
        if not split_file:
            return None
        return self.config.require_root() / split_file

    @property
    def expansion_policy(self) -> str:
        policy = self.config.extras.get("image_expansion_policy", "all_views")
        if policy not in _EXPANSION_POLICIES:
            raise AdapterError(
                f"[fairget] image_expansion_policy={policy!r} must be one of "
                f"{_EXPANSION_POLICIES}"
            )
        return policy

    @property
    def view_seed(self) -> int:
        return int(self.config.extras.get("view_assignment_seed", 17))

    # -- explicit source reader (plan B4/B7) ----------------------------- #

    def source_files(self) -> Sequence[Path]:
        dataset = self._dataset_path()
        root = self.config.require_root()
        missing = [
            str(p)
            for p in (dataset, root / "train_images", root / "test_images")
            if not p.exists()
        ]
        if missing:
            raise AdapterError(
                "[fairget] incompatible source layout; expected data/"
                f"dataset.json + train_images/ + test_images/ under {root}. "
                f"Missing: {missing}"
            )
        files = [dataset]
        split_path = self._split_path()
        if split_path is not None:
            if not split_path.exists():
                raise AdapterError(
                    f"[fairget] configured split_file does not exist: {split_path}"
                )
            files.append(split_path)
        return files

    def iter_rows_with_context(
        self,
    ) -> Iterator[tuple[SourceContext, Mapping[str, Any]]]:
        dataset = self._dataset_path()
        from .base import read_rows_from

        rows = read_rows_from(dataset)
        # Deterministic identity order; keep the original file index.
        indexed = sorted(
            enumerate(rows), key=lambda pair: str(pair[1].get("ID", ""))
        )
        for index, (file_index, row) in enumerate(indexed):
            if not isinstance(row, Mapping) or not row.get("ID"):
                raise AdapterError(
                    f"[fairget] dataset.json row {file_index} lacks an 'ID' field"
                )
            context = self.base_context(
                source_file=str(dataset), source_row_index=file_index
            )
            yield context, row

    # -- official splits (plan B11) -------------------------------------- #

    def _split_lookup(self) -> dict[str, str]:
        """Map identity ID -> split bucket from the configured split file."""
        split_path = self._split_path()
        if split_path is None:
            return {}
        from .base import read_rows_from

        payload = read_rows_from(split_path)
        if len(payload) != 1 or not isinstance(payload[0], Mapping):
            raise AdapterError(
                f"[fairget] split file {split_path} must hold one JSON object "
                "mapping split bucket -> identity IDs"
            )
        lookup: dict[str, str] = {}
        for bucket, ids in payload[0].items():
            if not isinstance(ids, (list, tuple)):
                raise AdapterError(
                    f"[fairget] split bucket {bucket!r} must map to a list of IDs"
                )
            for identity_id in ids:
                lookup[str(identity_id)] = str(bucket)
        return lookup

    # -- views (plan B10) ------------------------------------------------- #

    def _views(self, identity_id: str, partition: str) -> list[Path]:
        root = self.config.require_root()
        folder = root / (
            "train_images" if partition == "train" else "test_images"
        ) / identity_id
        if not folder.exists():
            return []
        return sorted(
            p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
        )

    @staticmethod
    def _hash_unit(key: str) -> int:
        return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16)

    def _assign_views(
        self, record_key: str, views: list[Path]
    ) -> list[tuple[str, Path]]:
        """Deterministic (view_id, path) assignment for one QA record."""
        if not views:
            return []
        if self.expansion_policy == "all_views":
            return [(path.stem, path) for path in views]
        chosen = views[self._hash_unit(f"{self.view_seed}:{record_key}") % len(views)]
        return [(chosen.stem, chosen)]

    # -- flattening (plan B8/B9) ------------------------------------------ #

    def _profile_facts(self, row: Mapping[str, Any]) -> list[ProfileFact]:
        facts: list[ProfileFact] = []
        profile = row.get("profile") or row.get("textual_attributes") or {}
        container = profile if isinstance(profile, Mapping) else {}
        for attr in _TEXTUAL_ATTRIBUTES:
            if attr in container and container[attr] not in (None, ""):
                facts.append(
                    ProfileFact(
                        fact_id=f"fairget_{attr}",
                        relation=attr,
                        value=str(container[attr]),
                        privacy_class="identity_textual",
                        source="source_human",
                        forgettable=True,
                    ).validate()
                )
        return facts

    def _fairface_observations(
        self, row: Mapping[str, Any]
    ) -> dict[str, AttributeObservation]:
        """FairGet demographics -> namespaced image-level observations."""
        obs: dict[str, AttributeObservation] = {}
        demographics = row.get("fairface") or row.get("demographics") or {}
        if isinstance(demographics, Mapping):
            for key, value in demographics.items():
                name = f"source_attributes.fairface.{key}"
                obs[name] = AttributeObservation(
                    name=name,
                    label=bool(value) if isinstance(value, (bool, int)) else None,
                    source="source_human",
                    confidence_band="high",
                    attribute_class="fairface_demographic",
                ).validate()
        return obs

    @staticmethod
    def _require_items(value: Any, where: str) -> list[Any]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise AdapterError(f"[fairget] {where} must be a list of QA items")
        return value

    def _item_sample(
        self,
        *,
        item: Mapping[str, Any],
        identity_id: str,
        identity_name: str | None,
        partition: str,
        media_type: str,
        attribute: str,
        task: str | None,
        item_index: int,
        view_id: str,
        image_uri: str | None,
        facts: list[ProfileFact],
        demographics: dict[str, AttributeObservation],
        split: str,
        context: SourceContext,
    ) -> CanonicalSample:
        if not isinstance(item, Mapping):
            raise AdapterError(
                f"[fairget] {partition}/{media_type}/{attribute}[{item_index}] "
                "is not a QA item object"
            )
        task_part = task if task else "qa"
        source_id = (
            f"fairget:{identity_id}:{partition}:{media_type}:{attribute}"
            f":{item_index}:{view_id}"
        )
        if task is not None:
            source_id = (
                f"fairget:{identity_id}:{partition}:{media_type}:{task}"
                f":{attribute}:{item_index}:{view_id}"
            )
        metadata = self.context_metadata(context)
        metadata.update(
            {
                "media_type": media_type,
                "attribute": attribute,
                "partition": partition,
                "item_index": item_index,
                "view_id": view_id,
                "image_expansion_policy": self.expansion_policy,
                "view_assignment_seed": self.view_seed,
            }
        )
        if task is not None:
            metadata["task"] = task
        if item.get("q_words") is not None:
            metadata["q_words"] = item["q_words"]
        if item.get("a_words") is not None:
            metadata["a_words"] = item["a_words"]
        return CanonicalSample(
            benchmark="fairget",
            source_sample_id=source_id,
            source_record_id=str(identity_id),
            identity_id=identity_id,
            provenance=self.provenance(
                source_id, source_subset=partition, context=context
            ),
            source_subset=partition,
            identity_name=identity_name,
            image_id=view_id if image_uri else None,
            image_uri=image_uri,
            # Demographics are image-level only (design rule 3.1).
            visual_attributes=dict(demographics) if image_uri else {},
            profile_facts=list(facts),
            modality="image_text" if image_uri else "text_only",
            task_type=task or ("visual_qa" if image_uri else "knowledge_qa"),
            question=item.get("q"),
            answer_text=item.get("a"),
            answer_label=item.get("gt"),
            split=split,
            source_metadata=metadata,
        ).validate()

    def to_samples(
        self,
        row: Mapping[str, Any],
        *,
        source_context: SourceContext,
    ) -> Iterator[CanonicalSample]:
        identity_id = str(row.get("ID"))
        identity_name = row.get("identity_name") or row.get("name")
        facts = self._profile_facts(row)
        demographics = self._fairface_observations(row)
        split_lookup = self._split_lookup()
        split = split_lookup.get(identity_id, "unassigned")
        train_views = self._views(identity_id, "train")
        eval_views = self._views(identity_id, "eval")

        # -- training: row["train"][media_type][attribute] -> items -------- #
        train = row.get("train") or {}
        if not isinstance(train, Mapping):
            raise AdapterError(f"[fairget] 'train' for {identity_id} must be a mapping")
        for media_type in sorted(train):
            attributes = train[media_type]
            if not isinstance(attributes, Mapping):
                raise AdapterError(
                    f"[fairget] train/{media_type} for {identity_id} must map "
                    "attributes to QA item lists"
                )
            for attribute in sorted(attributes):
                items = self._require_items(
                    attributes[attribute], f"train/{media_type}/{attribute}"
                )
                for item_index, item in enumerate(items):
                    record_key = (
                        f"{identity_id}:train:{media_type}:{attribute}:{item_index}"
                    )
                    if "image" in media_type:
                        assignments = self._assign_views(record_key, train_views)
                        if not assignments:
                            raise AdapterError(
                                f"[fairget] image QA {record_key} has no views "
                                f"under train_images/{identity_id}/"
                            )
                    else:
                        assignments = [("text", None)]
                    for view_id, path in assignments:
                        yield self._item_sample(
                            item=item,
                            identity_id=identity_id,
                            identity_name=identity_name,
                            partition="train",
                            media_type=media_type,
                            attribute=attribute,
                            task=None,
                            item_index=item_index,
                            view_id=view_id,
                            image_uri=str(path) if path else None,
                            facts=facts,
                            demographics=demographics,
                            split=split,
                            context=source_context,
                        )

        # -- eval: row["eval"][media_type][task][attribute] -> items ------- #
        eval_block = row.get("eval") or {}
        if not isinstance(eval_block, Mapping):
            raise AdapterError(f"[fairget] 'eval' for {identity_id} must be a mapping")
        for media_type in sorted(eval_block):
            tasks = eval_block[media_type]
            if not isinstance(tasks, Mapping):
                raise AdapterError(
                    f"[fairget] eval/{media_type} for {identity_id} must map "
                    "tasks to attribute QA blocks"
                )
            for task in sorted(tasks):
                attributes = tasks[task]
                if not isinstance(attributes, Mapping):
                    raise AdapterError(
                        f"[fairget] eval/{media_type}/{task} for {identity_id} "
                        "must map attributes to QA item lists"
                    )
                for attribute in sorted(attributes):
                    items = self._require_items(
                        attributes[attribute],
                        f"eval/{media_type}/{task}/{attribute}",
                    )
                    for item_index, item in enumerate(items):
                        record_key = (
                            f"{identity_id}:eval:{media_type}:{task}:{attribute}"
                            f":{item_index}"
                        )
                        if "image" in media_type:
                            assignments = self._assign_views(record_key, eval_views)
                            if not assignments:
                                raise AdapterError(
                                    f"[fairget] image QA {record_key} has no views "
                                    f"under test_images/{identity_id}/"
                                )
                        else:
                            assignments = [("text", None)]
                        for view_id, path in assignments:
                            yield self._item_sample(
                                item=item,
                                identity_id=identity_id,
                                identity_name=identity_name,
                                partition="eval",
                                media_type=media_type,
                                attribute=attribute,
                                task=task,
                                item_index=item_index,
                                view_id=view_id,
                                image_uri=str(path) if path else None,
                                facts=facts,
                                demographics=demographics,
                                split=split,
                                context=source_context,
                            )
