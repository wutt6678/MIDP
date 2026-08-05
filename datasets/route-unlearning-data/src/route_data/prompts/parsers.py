"""Strict output parsers (coding plan sections 8.3, 8.5).

Ambiguous responses are rejected rather than guessed: every parse returns a
``parse_status`` of ``ok``, ``ambiguous``, or ``empty`` alongside the raw
text so failures can be audited.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

PARSE_OK = "ok"
PARSE_AMBIGUOUS = "ambiguous"
PARSE_EMPTY = "empty"

_POSITIVE_TOKENS = {"yes", "true"}
_NEGATIVE_TOKENS = {"no", "false"}


@dataclass(frozen=True)
class BinaryParseResult:
    label: int | None          # 1 | 0 | None when rejected
    parse_status: str          # ok | ambiguous | empty
    raw_text: str


def parse_binary_answer(raw_text: str, max_tokens: int = 4) -> BinaryParseResult:
    """Map a generated answer to 0/1 per plan section 8.3.

    yes/true -> 1, no/false -> 0. Anything else (explanations, hedging,
    multi-token answers beyond the limit) is rejected with ``ambiguous``.
    """
    text = (raw_text or "").strip()
    if not text:
        return BinaryParseResult(label=None, parse_status=PARSE_EMPTY, raw_text=raw_text or "")
    # First whitespace-delimited token, lowercased, stripped of punctuation.
    token = text.split()[0].lower().strip(".,!?\"'`()[]{}:")
    if len(text.split()) > max_tokens:
        return BinaryParseResult(label=None, parse_status=PARSE_AMBIGUOUS, raw_text=raw_text)
    if token in _POSITIVE_TOKENS:
        return BinaryParseResult(label=1, parse_status=PARSE_OK, raw_text=raw_text)
    if token in _NEGATIVE_TOKENS:
        return BinaryParseResult(label=0, parse_status=PARSE_OK, raw_text=raw_text)
    return BinaryParseResult(label=None, parse_status=PARSE_AMBIGUOUS, raw_text=raw_text)


def parse_grouped_json(raw_text: str, expected_keys: list[str]) -> tuple[dict[str, int], str]:
    """Parse a grouped Mode-B JSON answer.

    Returns ``(values, parse_status)`` where ``values`` maps attribute ->
    {0,1}. Missing/extra keys or non-boolean values yield ``ambiguous`` with
    an empty dict (never guessed).
    """
    text = (raw_text or "").strip()
    if not text:
        return {}, PARSE_EMPTY
    # Tolerate code fences around the JSON object.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}, PARSE_AMBIGUOUS
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}, PARSE_AMBIGUOUS
    if not isinstance(payload, dict):
        return {}, PARSE_AMBIGUOUS
    if set(payload.keys()) != set(expected_keys):
        return {}, PARSE_AMBIGUOUS
    values: dict[str, int] = {}
    for key in expected_keys:
        value = payload[key]
        if isinstance(value, bool):
            values[key] = int(value)
        elif isinstance(value, str) and value.lower() in _POSITIVE_TOKENS | _NEGATIVE_TOKENS:
            values[key] = int(value.lower() in _POSITIVE_TOKENS)
        else:
            return {}, PARSE_AMBIGUOUS
    return values, PARSE_OK
