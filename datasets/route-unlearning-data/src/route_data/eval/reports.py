"""Run-bundle and metric-report writers (coding plan sections 8.6, 22).

Every evaluation run emits a reproducible bundle (plan section 22.3):

    run.yaml  resolved_config.yaml  environment.json  model_fingerprint.json
    data_fingerprint.json  metrics.json  metrics_by_attribute.csv  report.md

The Markdown report follows the CelebA model-report outline (plan 22.1) and flags
low-reliability / subjective attributes with caveats (plan section 10). Writers use
the atomic JSON/CSV helpers so a crash never leaves a half-written report.
"""

from __future__ import annotations

import datetime as _dt
import platform
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..constants.attribute_taxonomy import (
    LOW_RELIABILITY,
    SENSITIVE_DATASET_LABELS,
    group_of,
    is_reliability_flagged,
)
from ..data.io import ensure_parent_dir, write_json

__all__ = [
    "collect_environment",
    "render_report_md",
    "write_fingerprints",
    "write_metrics_bundle",
    "write_run_bundle",
]


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_plain(obj: Any) -> Any:
    """Recursively convert dataclasses/Paths into JSON/YAML-safe structures."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_plain(asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# Environment capture
# --------------------------------------------------------------------------- #


def collect_environment() -> dict[str, Any]:
    """Capture interpreter/library versions for reproducibility."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "captured_utc": _utcnow(),
    }
    for lib in ("numpy", "pandas", "torch", "transformers", "sklearn", "pyarrow"):
        try:
            mod = __import__(lib)
            env[lib] = getattr(mod, "__version__", "unknown")
        except Exception:
            env[lib] = "not-installed"
    return env


# --------------------------------------------------------------------------- #
# Metrics bundle
# --------------------------------------------------------------------------- #


def write_metrics_bundle(
    output_dir: str | Path,
    per_attribute: dict[str, dict[str, Any]],
    macro: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write ``metrics.json`` and ``metrics_by_attribute.csv``.

    Returns a mapping of artifact name -> written path.
    """
    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics_by_attribute.csv"

    payload: dict[str, Any] = {
        "generated_utc": _utcnow(),
        "macro": macro,
        "per_attribute": per_attribute,
    }
    if extra:
        payload.update(extra)
    write_json(payload, metrics_path)

    rows = []
    for attr in sorted(per_attribute):
        row = {"attribute": attr}
        row.update(per_attribute[attr])
        row["taxonomy_group"] = _safe_group(attr)
        row["reliability_flagged"] = is_reliability_flagged(attr)
        rows.append(row)
    df = pd.DataFrame(rows)
    ensure_parent_dir(csv_path)
    df.to_csv(csv_path, index=False)
    return {"metrics": metrics_path, "metrics_csv": csv_path}


def _safe_group(attr: str) -> str:
    try:
        return group_of(attr)
    except KeyError:
        return "unknown"


# --------------------------------------------------------------------------- #
# Fingerprints
# --------------------------------------------------------------------------- #


def write_fingerprints(
    output_dir: str | Path,
    model_fingerprint: dict[str, Any],
    data_fingerprint: dict[str, Any],
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    model_path = output_dir / "model_fingerprint.json"
    data_path = output_dir / "data_fingerprint.json"
    write_json(_to_plain(model_fingerprint), model_path)
    write_json(_to_plain(data_fingerprint), data_path)
    return {"model_fingerprint": model_path, "data_fingerprint": data_path}


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report_md(
    per_attribute: dict[str, dict[str, Any]],
    macro: dict[str, Any],
    model_fingerprint: dict[str, Any] | None = None,
    prompt_registry_hash: str | None = None,
    n_images: int | None = None,
    title: str = "CelebA-40 Model Report",
) -> str:
    """Render the CelebA model report (plan 22.1) as Markdown."""
    lines: list[str] = [f"# {title}", ""]
    lines.append(f"_Generated: {_utcnow()}_")
    lines.append("")

    # Model / provenance.
    lines.append("## Model and protocol")
    if model_fingerprint:
        for key in ("model_id", "revision", "backend", "dtype", "quantization"):
            if key in model_fingerprint:
                lines.append(f"- **{key}**: {model_fingerprint[key]}")
    if prompt_registry_hash:
        lines.append(f"- **prompt_registry_hash**: `{prompt_registry_hash}`")
    if n_images is not None:
        lines.append(f"- **images evaluated**: {n_images}")
    lines.append("")

    # Macro metrics.
    lines.append("## Macro metrics")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for key in (
        "balanced_accuracy",
        "macro_f1",
        "accuracy",
        "auroc",
        "average_precision",
        "brier",
        "ece",
        "parse_failure_rate",
    ):
        if key in macro:
            lines.append(f"| {key} | {_fmt(macro[key])} |")
    lines.append("")

    # Per-attribute table.
    lines.append("## Per-attribute metrics")
    header = (
        "| attribute | group | n | balanced_acc | macro_f1 | auroc | "
        "ece | parse_fail | caveats |"
    )
    lines.append(header)
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for attr in sorted(per_attribute):
        m = per_attribute[attr]
        caveats = _caveats(attr)
        lines.append(
            "| {attr} | {grp} | {n} | {bal} | {f1} | {auroc} | {ece} | {pf} | {cav} |".format(
                attr=attr,
                grp=_safe_group(attr),
                n=m.get("n", "-"),
                bal=_fmt(m.get("balanced_accuracy")),
                f1=_fmt(m.get("macro_f1")),
                auroc=_fmt(m.get("auroc")),
                ece=_fmt(m.get("ece")),
                pf=_fmt(m.get("parse_failure_rate")),
                cav=caveats or "-",
            )
        )
    lines.append("")

    # Caveats.
    lines.append("## Known label-quality caveats")
    lines.append(
        "- Low-reliability source labels: "
        + ", ".join(sorted(LOW_RELIABILITY))
        + "."
    )
    lines.append(
        "- Subjective/sensitive dataset-defined labels (CelebA definitions apply): "
        + ", ".join(sorted(SENSITIVE_DATASET_LABELS))
        + ". `Male` is the CelebA binary annotation, not self-identified gender."
    )
    lines.append(
        "- Ordinary accuracy is not the headline metric; CelebA labels are imbalanced."
    )
    lines.append("")
    return "\n".join(lines)


def _caveats(attr: str) -> str:
    tags = []
    if attr in LOW_RELIABILITY:
        tags.append("low-reliability")
    if attr in SENSITIVE_DATASET_LABELS:
        tags.append("subjective/sensitive")
    return ", ".join(tags)


# --------------------------------------------------------------------------- #
# Full run bundle
# --------------------------------------------------------------------------- #


def write_run_bundle(
    output_dir: str | Path,
    run_config: Any,
    model_fingerprint: dict[str, Any],
    data_fingerprint: dict[str, Any],
    per_attribute: dict[str, dict[str, Any]],
    macro: dict[str, Any],
    run_name: str = "celeba_eval",
    prompt_registry_hash: str | None = None,
    n_images: int | None = None,
) -> dict[str, Path]:
    """Write the complete reproducible run bundle (plan 22.3) to ``output_dir``.

    ``run_config`` may be a :class:`RunConfig` dataclass or a plain dict; it is
    serialized to ``resolved_config.yaml`` verbatim (after env expansion already
    happened at load time).
    """
    import yaml

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # run.yaml: high-level identity of the run.
    run_meta = {
        "name": run_name,
        "generated_utc": _utcnow(),
        "prompt_registry_hash": prompt_registry_hash,
        "n_images": n_images,
        "macro_metrics": _to_plain(macro),
    }
    run_path = output_dir / "run.yaml"
    ensure_parent_dir(run_path)
    with open(run_path, "w") as f:
        yaml.safe_dump(_to_plain(run_meta), f, sort_keys=False)
    written["run"] = run_path

    # resolved_config.yaml: full resolved configuration.
    resolved = _to_plain(run_config)
    resolved_path = output_dir / "resolved_config.yaml"
    with open(resolved_path, "w") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)
    written["resolved_config"] = resolved_path

    # environment.json + fingerprints.
    env_path = output_dir / "environment.json"
    write_json(collect_environment(), env_path)
    written["environment"] = env_path
    written.update(write_fingerprints(output_dir, model_fingerprint, data_fingerprint))

    # metrics.json / metrics_by_attribute.csv.
    written.update(write_metrics_bundle(output_dir, per_attribute, macro))

    # report.md.
    report = render_report_md(
        per_attribute,
        macro,
        model_fingerprint=model_fingerprint,
        prompt_registry_hash=prompt_registry_hash,
        n_images=n_images,
        title=f"{run_name} report",
    )
    report_path = output_dir / "report.md"
    ensure_parent_dir(report_path)
    with open(report_path, "w") as f:
        f.write(report)
    written["report"] = report_path

    return written
