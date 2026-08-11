"""FIUBench adapter (repair plan B13-B18).

Reads the released FIUBench profile-level schema instead of a guessed flat
mapping. Each source row describes one fictitious identity:

    image_path   # identity image (relative to the source root)
    name         # display name of the fictitious person
    gender       # source profile metadata (NOT a CelebA visual label)
    caption      # free-form identity caption
    qa_list      # nested list of private-profile QA items
    raw_data     # raw upstream profile payload

QA items carry ``question``, ``paraphrased_question``, ``answer``,
``paraphrased_answer``, ``perturbed_answer`` and ``keywords`` (plan B13).

Flattening rules:

- identity IDs are stable hashes of revision + source row index + image path
  + display name, so re-reading the same pinned revision yields identical
  identities (plan B14);
- one canonical record per original QA item with ID
  ``fiubench:<identity_id>:qa:<qa_index>:original``; non-empty paraphrase /
  perturbed variants are emitted only when enabled via config extras and
  carry explicit ``base_qa_id`` + ``variant_type`` metadata (plan B15);
- caption and raw profile are preserved whole as profile facts; no LLM
  parsing (plan B16);
- split membership comes from the configured split file, never row order
  (plan B17).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..schemas import CanonicalSample, ProfileFact
from .base import AdapterError, BenchmarkAdapter, SourceContext, register_adapter


@register_adapter("fiubench")
class FiubenchAdapter(BenchmarkAdapter):
    adapter_version = "fiubench-v2"
    required_fields = ("image_path", "name")

    def _default_field_map(self) -> dict[str, str]:
        return {
            "image_path": "image_path",
            "identity_name": "name",
            "gender": "gender",
            "caption": "caption",
            "qa_list": "qa_list",
            "raw_data": "raw_data",
        }

    # -- configuration -------------------------------------------------- #

    def _source_path(self) -> Path:
        root = self.config.require_root()
        source_file = self.config.extras.get("source_file")
        if not source_file:
            raise AdapterError(
                "[fiubench] data config must set extras.source_file to the "
                "released profile file (repair plan C2)"
            )
        return root / source_file

    def _split_path(self) -> Path | None:
        split_file = self.config.extras.get("split_file")
        if not split_file:
            return None
        return self.config.require_root() / split_file

    def _images_root(self) -> Path:
        images_root = self.config.extras.get("images_root")
        root = self.config.require_root()
        return root / images_root if images_root else root

    @property
    def include_paraphrases(self) -> bool:
        return bool(self.config.extras.get("include_paraphrases", False))

    @property
    def include_perturbed(self) -> bool:
        return bool(self.config.extras.get("include_perturbed", False))

    # -- explicit source reader (plan B4/B17) ---------------------------- #

    def source_files(self) -> Sequence[Path]:
        path = self._source_path()
        if not path.exists():
            raise AdapterError(
                f"[fiubench] incompatible source layout; configured source "
                f"file does not exist: {path}"
            )
        files = [path]
        split_path = self._split_path()
        if split_path is not None:
            if not split_path.exists():
                raise AdapterError(
                    f"[fiubench] configured split_file does not exist: {split_path}"
                )
            files.append(split_path)
        return files

    # -- official splits (plan B17) --------------------------------------- #

    def _split_lookup(self) -> dict[str, str]:
        """Map identity display name -> split bucket from the split file."""
        split_path = self._split_path()
        if split_path is None:
            return {}
        from .base import read_rows_from

        payload = read_rows_from(split_path)
        if len(payload) != 1 or not isinstance(payload[0], Mapping):
            raise AdapterError(
                f"[fiubench] split file {split_path} must hold one JSON "
                "object mapping split bucket -> identity names/IDs"
            )
        lookup: dict[str, str] = {}
        for bucket, ids in payload[0].items():
            if not isinstance(ids, (list, tuple)):
                raise AdapterError(
                    f"[fiubench] split bucket {bucket!r} must map to a list "
                    "of identity names/IDs"
                )
            for identity in ids:
                lookup[str(identity)] = str(bucket)
        return lookup

    # -- identity normalization (plan B14) -------------------------------- #

    def _identity_id(self, context: SourceContext, row: Mapping[str, Any]) -> str:
        """Stable identity ID: revision + row index + path + name."""
        if context.source_row_index is None:
            raise AdapterError(
                "[fiubench] source context lacks a row index; identity IDs "
                "must be reproducible from the pinned source file"
            )
        key = "|".join(
            (
                context.source_revision,
                str(context.source_row_index),
                str(row.get("image_path") or ""),
                str(row.get("name") or ""),
            )
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def _resolve_image(self, raw_path: str | None) -> str | None:
        """Resolve ``image_path`` against the images root / source root."""
        if not raw_path:
            return None
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return str(candidate) if candidate.exists() else raw_path
        resolved = self._images_root() / candidate
        if resolved.exists():
            return str(resolved)
        fallback = self.config.require_root() / candidate
        if fallback.exists():
            return str(fallback)
        # Keep the original reference so the record remains traceable.
        return raw_path

    def _profile_facts(
        self, row: Mapping[str, Any], identity_name: str | None
    ) -> list[ProfileFact]:
        """Preserve caption + raw profile whole (plan B16, no LLM parser)."""
        facts: list[ProfileFact] = []
        caption = row.get("caption")
        if caption not in (None, ""):
            facts.append(
                ProfileFact(
                    fact_id="fiubench_caption",
                    relation="caption",
                    value=str(caption),
                    privacy_class="private_profile",
                    source="source_human",
                    forgettable=True,
                ).validate()
            )
        raw_data = row.get("raw_data")
        if raw_data in (None, "", {}, []):
            return facts
        if isinstance(raw_data, str):
            value = raw_data
        else:
            value = json.dumps(raw_data, ensure_ascii=False, sort_keys=True)
        facts.append(
            ProfileFact(
                fact_id="fiubench_raw_profile",
                relation="raw_profile",
                value=value,
                privacy_class="private_profile",
                source="source_human",
                forgettable=True,
            ).validate()
        )
        return facts

    # -- qa_list flattening (plan B15) ------------------------------------ #

    def _qa_sample(
        self,
        *,
        identity_id: str,
        identity_name: str | None,
        image_uri: str | None,
        gender: str | None,
        facts: list[ProfileFact],
        split: str,
        qa_index: int,
        item: Mapping[str, Any],
        variant_type: str,
        base_qa_id: str | None,
        question: Any,
        answer: Any,
        context: SourceContext,
    ) -> CanonicalSample:
        if question in (None, "") and variant_type == "original":
            raise AdapterError(
                f"[fiubench] {identity_id} qa_list[{qa_index}] has no question"
            )
        suffix = variant_type if variant_type != "original" else "original"
        source_id = f"fiubench:{identity_id}:qa:{qa_index}:{suffix}"
        metadata = self.context_metadata(context)
        metadata.update(
            {
                "identity_name": identity_name,
                "image_path": image_uri,
                "qa_index": qa_index,
                "variant_type": variant_type,
            }
        )
        if gender not in (None, ""):
            # Source profile metadata only — never a CelebA visual label
            # (plan B14).
            metadata["gender"] = gender
        if base_qa_id is not None:
            metadata["base_qa_id"] = base_qa_id
        if item.get("keywords") is not None:
            metadata["keywords"] = item["keywords"]
        if item.get("question") is not None:
            metadata["original_question"] = item["question"]
        if item.get("answer") is not None:
            metadata["original_answer"] = item["answer"]
        return CanonicalSample(
            benchmark="fiubench",
            source_sample_id=source_id,
            source_record_id=identity_id,
            identity_id=identity_id,
            provenance=self.provenance(
                source_id, source_subset=variant_type, context=context
            ),
            source_subset=variant_type,
            identity_name=identity_name,
            image_id=Path(image_uri).stem if image_uri else None,
            image_uri=image_uri,
            profile_facts=list(facts),
            modality="image_text" if image_uri else "text_only",
            task_type="private_profile_vqa",
            question=str(question) if question not in (None, "") else None,
            answer_text=str(answer) if answer not in (None, "") else None,
            split=split,
            source_metadata=metadata,
        ).validate()

    def to_samples(
        self,
        row: Mapping[str, Any],
        *,
        source_context: SourceContext,
    ) -> Iterator[CanonicalSample]:
        identity_name = self.source_field(row, "identity_name")
        identity_name = str(identity_name) if identity_name else None
        gender = self.source_field(row, "gender", required=False)
        gender = str(gender) if gender not in (None, "") else None
        identity_id = self._identity_id(source_context, row)
        image_uri = self._resolve_image(
            self.source_field(row, "image_path", required=False)
        )
        facts = self._profile_facts(row, identity_name)
        split_lookup = self._split_lookup()
        split = split_lookup.get(identity_name or identity_id, "unassigned")

        qa_list = self.source_field(row, "qa_list", required=False) or []
        if not isinstance(qa_list, list):
            raise AdapterError(
                f"[fiubench] qa_list for {identity_name or identity_id} must "
                "be a list of QA items"
            )
        for qa_index, item in enumerate(qa_list):
            if not isinstance(item, Mapping):
                raise AdapterError(
                    f"[fiubench] qa_list[{qa_index}] for "
                    f"{identity_name or identity_id} is not a QA item object"
                )
            base_kwargs = {
                "identity_id": identity_id,
                "identity_name": identity_name,
                "image_uri": image_uri,
                "gender": gender,
                "facts": facts,
                "split": split,
                "qa_index": qa_index,
                "item": item,
                "context": source_context,
            }
            base_qa_id = f"fiubench:{identity_id}:qa:{qa_index}:original"
            yield self._qa_sample(
                **base_kwargs,
                variant_type="original",
                base_qa_id=None,
                question=item.get("question"),
                answer=item.get("answer"),
            )
            if self.include_paraphrases and item.get("paraphrased_question") not in (
                None,
                "",
            ):
                yield self._qa_sample(
                    **base_kwargs,
                    variant_type="paraphrase",
                    base_qa_id=base_qa_id,
                    question=item.get("paraphrased_question"),
                    answer=item.get("paraphrased_answer"),
                )
            if self.include_perturbed and item.get("perturbed_answer") not in (
                None,
                "",
            ):
                yield self._qa_sample(
                    **base_kwargs,
                    variant_type="perturbed",
                    base_qa_id=base_qa_id,
                    question=item.get("question"),
                    answer=item.get("perturbed_answer"),
                )
