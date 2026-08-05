"""Deterministic template rendering for prompts and QA.

Uses explicit ``str.format``-style substitution with a fixed field set so
rendering is reproducible and never executes arbitrary expressions.
"""

from __future__ import annotations

import re

_FIELD_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def render(template: str, **fields: object) -> str:
    """Render ``{field}`` placeholders; fail loudly on missing fields."""
    missing = sorted(set(_FIELD_RE.findall(template)) - set(fields))
    if missing:
        raise KeyError(f"Template fields missing values: {missing} (template: {template!r})")
    return template.format(**fields)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())
