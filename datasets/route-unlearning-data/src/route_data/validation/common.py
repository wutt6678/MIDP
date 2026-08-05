"""Shared primitives for the validation suite (plan section 19).

Every check module reports issues through :class:`ValidationIssue` so results
from schema, source-integrity, distribution, and leakage checks can be merged
into one machine-readable report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ValidationError(ValueError):
    """Base error raised in strict validation mode."""


@dataclass(frozen=True)
class ValidationIssue:
    """One failed check: ``kind`` is stable for counting, ``where`` locates it."""

    kind: str
    where: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "where": self.where, "detail": self.detail}

    def __str__(self) -> str:
        return f"[{self.kind}] {self.where}: {self.detail}" if self.detail else (
            f"[{self.kind}] {self.where}"
        )


def summarize_issues(issues: list[ValidationIssue]) -> dict[str, Any]:
    """Group issues by kind for compact reporting."""
    counts: dict[str, int] = {}
    for issue in issues:
        counts[issue.kind] = counts.get(issue.kind, 0) + 1
    return {
        "issue_count": len(issues),
        "by_kind": dict(sorted(counts.items())),
        "issues": [issue.to_dict() for issue in issues[:100]],
    }
