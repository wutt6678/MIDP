"""FIUBench adapter (repair plan B13-B18, fix-list P0-1–P0-10).

Reads the released FIUBench source files directly (Option A):

    dataset/full.json   # JSONL, one record per identity (573 records)
    dataset/split.json  # JSON object mapping bucket → [subject_ids]

Each source row describes one fictitious identity with the released schema:

    image_path              # e.g. ./dataset/SFHQ/SFHQ_pt1_00044363.jpg
    name                    # display name of the fictitious person
    gender                  # source profile metadata (NOT a CelebA visual label)
    caption                 # free-form identity caption
    qa_list                 # nested list of 20 private-profile QA items
    raw_data                # raw upstream profile payload
    unique_id               # 8-digit SFHQ subject identifier (e.g. "00044363")

QA items carry ``question``, ``paraphrased_question`` (list of 3),
``answer``, ``paraphrased_answer`` (scalar or list), ``perturbed_answer``
(list of 3) and ``keywords``.

Flattening rules:

- identity IDs are stable hashes of revision + source row index + image path
  + display name (plan B14);
- the released ``unique_id`` is preserved as ``source_subject_id`` in
  ``source_metadata`` (P0-4);
- split membership is resolved via ``source_subject_id`` against the released
  split file, never by display name (P0-5);
- overlapping bucket memberships are preserved as a list in
  ``official_memberships`` (P0-6);
- the configured ``fiubench_protocol`` maps released buckets to MIDP roles;
- list-valued ``paraphrased_question`` / ``perturbed_answer`` are flattened
  into per-variant canonical samples with deterministic IDs (P0-7–P0-10);
- caption and raw profile are preserved whole as profile facts (plan B16).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..schemas import CanonicalSample, ProfileFact
from .base import AdapterError, BenchmarkAdapter, SourceContext, register_adapter

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _as_variant_list(value: Any) -> list[str]:
    """Normalize a scalar-or-list field into a list of non-empty strings.

    Handles the released FIUBench schema where ``paraphrased_question`` and
    ``perturbed_answer`` are lists of strings, while ``paraphrased_answer``
    may be a scalar string.  Returns an empty list for missing / blank values.
    """
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v not in (None, "", ) and str(v).strip()]
    return [str(value)]


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #


@register_adapter("fiubench")
class FiubenchAdapter(BenchmarkAdapter):
    adapter_version = "fiubench-v3"
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

    def iter_rows_with_context(
        self,
    ) -> Iterator[tuple[SourceContext, Mapping[str, Any]]]:
        """Yield rows from the profile file; the split file is metadata only.

        Handles both JSONL and JSON-array profile files regardless of the
        file extension (the released ``dataset/full.json`` is JSONL despite
        its ``.json`` extension).
        """
        source_path = self._source_path()
        rows = self._read_profile_file(source_path)
        for index, row in enumerate(rows):
            context = self.base_context(
                source_file=str(source_path), source_row_index=index
            )
            yield context, row

    @staticmethod
    def _read_profile_file(path: Path) -> list[dict[str, Any]]:
        """Read a profile file that may be JSONL or JSON-array.

        The released FIUBench ``dataset/full.json`` is JSONL despite having a
        ``.json`` extension.  Try JSON-array first; on failure fall back to
        line-delimited JSON.
        """
        with open(path) as fh:
            text = fh.read().strip()
        if not text:
            return []
        # Try JSON array / object first.
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            pass
        # Fall back to JSONL (one JSON object per line).
        rows: list[dict[str, Any]] = []
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise AdapterError(
                        f"[fiubench] invalid JSON at line {lineno} in {path}: {exc}"
                    ) from exc
        return rows

    # -- source subject ID (P0-4) ---------------------------------------- #

    @staticmethod
    def _source_subject_id(row: Mapping[str, Any]) -> str | None:
        """Extract the released FIUBench / SFHQ subject identifier.

        Prefers the explicit ``unique_id`` field; falls back to parsing the
        8-digit trailing number from ``image_path`` (e.g.
        ``SFHQ_pt1_00044363`` → ``00044363``).  Returns ``None`` when neither
        is available (e.g. legacy golden fixture rows).
        """
        uid = row.get("unique_id")
        if uid is not None and str(uid).strip():
            return str(uid).strip()
        image_path = row.get("image_path") or ""
        stem = Path(str(image_path)).stem
        match = re.search(r"(\d{8})$", stem)
        if match:
            return match.group(1)
        return None

    # -- protocol (P0-3 / P1-2) ------------------------------------------ #

    def _protocol(self) -> dict[str, Any]:
        """Return the ``fiubench_protocol`` block from config extras."""
        proto = self.config.extras.get("fiubench_protocol")
        if proto and isinstance(proto, dict):
            return dict(proto)
        return {}

    def _protocol_sha256(self) -> str | None:
        """P1-5: cached protocol SHA-256 fingerprint."""
        proto = self._protocol()
        if not proto:
            return None
        if not hasattr(self, "_cached_protocol_sha256"):
            from ..split_mapping import compute_protocol_sha256
            sha, _ = compute_protocol_sha256(proto)
            self._cached_protocol_sha256 = sha
        return self._cached_protocol_sha256

    def _source_mapping(self) -> dict[str, str]:
        """Return the ``source_mapping`` override from config extras."""
        mapping = self.config.extras.get("source_mapping")
        if mapping and isinstance(mapping, dict):
            return dict(mapping)
        return {}

    # -- official splits (P0-5 / P0-6) ----------------------------------- #

    def _split_lookup(self) -> dict[str, list[str]]:
        """Map source subject ID → list of official FIUBench bucket names.

        Reads the released ``split.json`` format: a single JSON object whose
        keys are bucket names (``forget1``, ``forget5``, …) and whose values
        are lists of subject-ID strings.

        Falls back to name-based lookup for legacy fixtures whose split file
        uses the old ``forget / retain / evaluation`` vocabulary with display
        names instead of subject IDs.
        """
        split_path = self._split_path()
        if split_path is None:
            return {}
        from .base import read_rows_from

        payload = read_rows_from(split_path)
        if len(payload) != 1 or not isinstance(payload[0], Mapping):
            raise AdapterError(
                f"[fiubench] split file {split_path} must hold one JSON "
                "object mapping split bucket → identity names/IDs"
            )
        raw = payload[0]
        # Detect legacy fixture format (forget/retain/evaluation with names).
        legacy_keys = {"forget", "retain", "evaluation"}
        if set(raw.keys()) & legacy_keys:
            return self._legacy_split_lookup(raw)

        lookup: dict[str, list[str]] = {}
        for bucket, ids in raw.items():
            if not isinstance(ids, (list, tuple)):
                raise AdapterError(
                    f"[fiubench] split bucket {bucket!r} must map to a list "
                    "of subject IDs"
                )
            for sid in ids:
                sid_str = str(sid).strip()
                if sid_str:
                    lookup.setdefault(sid_str, []).append(str(bucket))
        return lookup

    def official_split_buckets(self) -> set[str]:
        """P0-6: return the actual bucket names from ``dataset/split.json``.

        Reads the released split file and returns the top-level keys
        (e.g. ``{"forget1", "forget5", "forget10", "retain5", "retain15"}``).
        Returns an empty set when no split file is configured or found.
        """
        split_path = self._split_path()
        if split_path is None:
            return set()
        from .base import read_rows_from

        payload = read_rows_from(split_path)
        if len(payload) != 1 or not isinstance(payload[0], Mapping):
            return set()
        return {str(k) for k in payload[0]}

    @staticmethod
    def _legacy_split_lookup(raw: Mapping) -> dict[str, list[str]]:
        """Build a *name*-keyed lookup from a legacy split file.

        The returned dict uses ``"name:<display name>"`` sentinel keys so the
        caller can distinguish legacy lookups from subject-ID lookups.
        """
        lookup: dict[str, list[str]] = {}
        for bucket, names in raw.items():
            if not isinstance(names, (list, tuple)):
                continue
            for name in names:
                key = f"name:{name}"
                lookup.setdefault(key, []).append(str(bucket))
        return lookup

    def _resolve_split(
        self,
        row: Mapping[str, Any],
        memberships: list[str],
    ) -> str:
        """Resolve official bucket memberships to an effective MIDP split.

        Called only when no ``fiubench_protocol`` is active.  When a protocol
        IS active the caller uses the shared
        :func:`route_data.data.split_mapping.resolve_protocol_role` directly
        (P0-1 review-list repair).

        Non-matching official identities receive ``"out_of_protocol"``
        instead of falling through to a generic source mapping (P0-2).
        """
        mapping = self._source_mapping()

        # Fallback: resolve the first membership through the source mapping.
        for bucket in memberships:
            if bucket in mapping:
                return mapping[bucket]
        # Legacy fixture fallback: map well-known bucket names to MIDP roles
        # when no explicit source_mapping is configured (golden CI fixture).
        _legacy = {"forget": "exclude", "retain": "train", "evaluation": "eval"}
        for bucket in memberships:
            if bucket in _legacy:
                return _legacy[bucket]
        return "unassigned"

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

    # -- P2-7: structured QA facts ---------------------------------------- #

    @staticmethod
    def _qa_fact(qa_index: int, item: Mapping[str, Any]) -> ProfileFact:
        """Create a structured identity fact from one FIUBench QA item.

        P2-7: each original QA is directly representable as a ProfileFact
        without LLM extraction.  P2-10: only the *original* answer becomes
        the fact value; perturbed answers are never stored as facts.
        P2-11: provenance fields enable exact traceability.
        """
        question = str(item.get("question") or "")
        answer = str(item.get("answer") or "")
        return ProfileFact(
            fact_id=f"fiubench_qa_{qa_index:02d}",
            relation=question,
            value=answer,
            privacy_class="private_profile",
            source="source_human",
            forgettable=True,
            source_qa_index=qa_index,
            original_question=question,
            original_answer=answer,
            question_variant="canonical",
        ).validate()

    # -- qa_list flattening (plan B15, P0-7–P0-10) ----------------------- #

    @staticmethod
    def _merged_facts(
        base: list[ProfileFact], qa_fact: ProfileFact | None
    ) -> list[ProfileFact]:
        """Return base facts plus the per-QA fact (if any).

        P2-7: each original QA sample carries its own structured fact
        alongside the shared caption / raw_profile facts.  P2-10: paraphrase
        and perturbed variants receive *no* extra fact so perturbed answers
        can never become ground-truth knowledge.
        """
        if qa_fact is None:
            return list(base)
        return list(base) + [qa_fact]

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
        variant_index: int | None,
        base_qa_id: str | None,
        question: Any,
        answer: Any,
        context: SourceContext,
        source_subject_id: str | None = None,
        official_memberships: list[str] | None = None,
        effective_role: str | None = None,
        protocol_name: str | None = None,
        protocol_sha256: str | None = None,
        qa_fact: ProfileFact | None = None,
    ) -> CanonicalSample:
        if question in (None, "") and variant_type == "original":
            raise AdapterError(
                f"[fiubench] {identity_id} qa_list[{qa_index}] has no question"
            )
        # Build deterministic source_sample_id (P0-10).
        if variant_index is not None:
            suffix = f"{variant_type}:{variant_index}"
        else:
            suffix = variant_type
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
        if variant_index is not None:
            metadata["variant_index"] = variant_index
        if gender not in (None, ""):
            metadata["gender"] = gender
        if base_qa_id is not None:
            metadata["base_qa_id"] = base_qa_id
        if item.get("keywords") is not None:
            metadata["keywords"] = item["keywords"]
        if item.get("question") is not None:
            metadata["original_question"] = item["question"]
        if item.get("answer") is not None:
            metadata["original_answer"] = item["answer"]
        if source_subject_id is not None:
            metadata["source_subject_id"] = source_subject_id
        if official_memberships is not None:
            metadata["official_memberships"] = official_memberships
        if effective_role is not None:
            metadata["effective_role"] = effective_role
        if protocol_name is not None:
            metadata["protocol_name"] = protocol_name
        if protocol_sha256 is not None:
            metadata["protocol_sha256"] = protocol_sha256
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
            profile_facts=self._merged_facts(facts, qa_fact if variant_type == "original" else None),
            modality="image_text" if image_uri else "text_only",
            task_type="private_profile_vqa",
            question=str(question) if question not in (None, "") else None,
            answer_text=str(answer) if answer not in (None, "") else None,
            split=split,
            source_metadata=metadata,
        ).validate()

    # -- main entry point ------------------------------------------------- #

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

        # -- split resolution (P0-1 / P0-4 / P0-5 / P0-6) ---------------- #
        subject_id = self._source_subject_id(row)
        split_lookup = self._split_lookup()
        official_memberships: list[str] = []
        if subject_id is not None and subject_id in split_lookup:
            official_memberships = split_lookup[subject_id]
        elif subject_id is None and identity_name:
            # Legacy fixture: fall back to name-based lookup.
            legacy_key = f"name:{identity_name}"
            official_memberships = split_lookup.get(legacy_key, [])
        # P0-1 (review-list): when a protocol is active it is the SOLE
        # authority for experiment roles.  Empty memberships must route
        # through resolve_protocol_role([], …) → "out_of_protocol",
        # never fall through to "unassigned".
        protocol = self._protocol()
        if protocol:
            from ..split_mapping import resolve_protocol_role
            split = resolve_protocol_role(
                official_memberships, protocol,
                source_subject_id=subject_id,
            )
        elif official_memberships:
            split = self._resolve_split(row, official_memberships)
        else:
            split = "unassigned"
        # P0-8: effective_role mirrors the resolved split so downstream
        # consumers can distinguish official memberships from MIDP roles.
        effective_role = split
        protocol_name = self._protocol().get("name") if self._protocol() else None
        protocol_sha256 = self._protocol_sha256()
        # P1-5: never silently hash an official FIUBench identity.
        if (
            official_memberships
            and subject_id is not None
            and split == "unassigned"
        ):
            raise AdapterError(
                f"[fiubench] subject ID {subject_id} has official memberships "
                f"{official_memberships} but no protocol/mapping rule resolves "
                f"them to a MIDP role; refusing to silently hash (P1-5)"
            )

        # -- qa_list expansion (P0-7–P0-10) -------------------------------- #
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
                "source_subject_id": subject_id,
                "official_memberships": official_memberships or None,
                "effective_role": effective_role,
                "protocol_name": protocol_name,
                "protocol_sha256": protocol_sha256,
            }
            base_qa_id = f"fiubench:{identity_id}:qa:{qa_index}:original"
            # P2-7: one structured fact per original QA item.
            qa_fact = self._qa_fact(qa_index, item)

            # 1. Original variant (carries the QA fact, P2-7/P2-8).
            yield self._qa_sample(
                **base_kwargs,
                variant_type="original",
                variant_index=None,
                base_qa_id=None,
                question=item.get("question"),
                answer=item.get("answer"),
                qa_fact=qa_fact,
            )

            # 2. Paraphrase variants (P0-7 / P0-9).
            if self.include_paraphrases:
                pq_list = _as_variant_list(item.get("paraphrased_question"))
                pa_list = _as_variant_list(item.get("paraphrased_answer"))
                for vi, pq in enumerate(pq_list):
                    # Pair paraphrased question with the corresponding
                    # paraphrased answer by index; reuse the last available
                    # answer when counts differ (P0-9).
                    pa = (
                        pa_list[vi]
                        if vi < len(pa_list)
                        else pa_list[-1]
                        if pa_list
                        else item.get("answer")
                    )
                    yield self._qa_sample(
                        **base_kwargs,
                        variant_type="paraphrase",
                        variant_index=vi,
                        base_qa_id=base_qa_id,
                        question=pq,
                        answer=pa,
                    )

            # 3. Perturbed variants (P0-8).
            if self.include_perturbed:
                pert_list = _as_variant_list(item.get("perturbed_answer"))
                for vi, pert in enumerate(pert_list):
                    yield self._qa_sample(
                        **base_kwargs,
                        variant_type="perturbed",
                        variant_index=vi,
                        base_qa_id=base_qa_id,
                        question=item.get("question"),
                        answer=pert,
                    )
