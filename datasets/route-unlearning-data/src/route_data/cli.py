"""``route-data`` command-line interface (coding plan section 20).

Subcommand contract:

    route-data model inspect|smoke-test --config configs/model/...yaml
    route-data celeba validate-raw|prepare --config configs/data/celeba.yaml
    route-data celeba evaluate --config configs/runs/celeba_pilot.yaml
    route-data celeba report|freeze-protocol --run-id <id>
    route-data source inspect --dataset <name> --config configs/data/<name>.yaml
    route-data build annotate|qa|route-probes|splits|export --dataset X --config Y
    route-data validate dataset --dataset X [--strict]
    route-data card render --dataset X

Every command supports ``--dry-run``, ``--limit``, ``--resume`` and
``--output-dir``. Configuration is always loaded *before* any model is touched,
so invalid YAML fails fast (plan Phase 0).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from .config import (
    ConfigError,
    RunConfig,
    load_data_config,
    load_model_config,
    load_run_config,
)
from .data.io import read_json, read_jsonl, read_shards, write_json, write_jsonl, write_parquet

log = logging.getLogger("route_data.cli")

REPO_ROOT = Path(__file__).resolve().parents[2]
CARD_SUFFIX = "_celeba40"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _benchmark_of(dataset: str) -> str:
    return dataset[: -len(CARD_SUFFIX)] if dataset.endswith(CARD_SUFFIX) else dataset


def _default_build_dir(run_cfg: RunConfig) -> Path:
    return Path(run_cfg.build.output_dir)

def _model_output_name(model_id: str) -> str:
    return (
        model_id
        .replace("/", "_")
        .replace(":", "_")
        .replace("\\", "_")
    )

def _dataset_dir(args, run_cfg: RunConfig, dataset: str) -> Path:
    base = Path(args.output_dir) if args.output_dir else _default_build_dir(run_cfg)
    model_dir = _model_output_name(run_cfg.model.model_id)
    return base / model_dir / dataset


def _data_config_for(dataset: str, run_cfg: RunConfig):
    """Resolve ``configs/data/<dataset>.yaml`` relative to the run config."""
    assert run_cfg.source_path is not None
    candidate = run_cfg.source_path.parent.parent / "data" / f"{_benchmark_of(dataset)}.yaml"
    if not candidate.exists():
        raise ConfigError(f"No data config found for dataset '{dataset}' at {candidate}")
    return load_data_config(candidate)


def _load_samples(dataset_dir: Path, dataset: str):
    from .data.schemas import CanonicalSample

    path = dataset_dir / f"{dataset}_annotated.jsonl"
    if not path.exists():
        raise ConfigError(
            f"No annotated samples at {path}; run 'route-data build annotate' first"
        )
    return [CanonicalSample.from_dict(doc) for doc in read_jsonl(path)]


def _load_processed_samples(dataset_dir: Path, dataset: str):
    """Load whitelist-processed samples.

    Fail closed: if the processed artifact is missing, raise an error
    instead of silently falling back to annotated. This guarantees that
    all downstream operations use the processed dataset (Fix 7).
    """
    from .data.schemas import CanonicalSample

    processed = dataset_dir / f"{dataset}_processed.jsonl"
    if processed.exists():
        return [CanonicalSample.from_dict(doc) for doc in read_jsonl(processed)]
    raise ConfigError(
        f"Processed dataset missing at {processed}; rerun 'route-data build annotate'."
    )


def _load_image(uri: str | None, base: Path | None = None):
    """Best-effort PIL image load; raise when the image is missing or corrupt.

    Relative URIs are resolved against ``base`` (typically the source
    dataset root) so redistributable fixtures can store relative paths.
    A ``None`` uri returns a blank placeholder (used by model smoke-test
    which has no real image).
    """
    from PIL import Image

    if uri is None:
        return Image.new("RGB", (224, 224), (127, 127, 127))

    path = Path(uri.removeprefix("file://"))
    if not path.is_absolute() and base is not None:
        path = base / path
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path} (uri={uri!r})")
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception as exc:
        raise OSError(f"Could not open image {path}: {exc}") from exc


def _p_positive(response) -> float | None:
    """Collapse candidate log-probabilities into P(' yes')."""
    import math
    from .models.scoring import normalize_binary_scores

    if not response.candidate_scores:
        return None
    logps = {cs.candidate: cs.log_probability for cs in response.candidate_scores}
    # normalize_binary_scores already returns probabilities; do NOT feed them
    # through binary_probability again (that expects raw log-scores and would
    # apply a second softmax, compressing every score into ~[0.27, 0.73]).
    probs = normalize_binary_scores(logps)
    for candidate in (" yes", "yes"):
        if candidate in probs:
            p = float(probs[candidate])
            # Validate probability (Fix 10)
            if not math.isfinite(p) or p < 0.0 or p > 1.0:
                log.warning(
                    "Invalid probability for %s: p=%s (logps=%s)",
                    candidate, p, logps,
                )
                return None
            return p
    return None


def _split_specs_for(samples) -> list:
    from .build.annotate import CELEBA40_NAMESPACE
    from .build.split_generation import SplitSpec

    specs: list[SplitSpec] = []
    identities = sorted({s.identity_id for s in samples if s.identity_id})
    if identities:
        specs.append(
            SplitSpec(
                name="identity_forget",
                forget_scope="identity",
                forget_identity_ids=(identities[0],),
            )
        )
    fact_ids = sorted({f.fact_id for s in samples for f in s.profile_facts})
    if fact_ids:
        specs.append(
            SplitSpec(
                name="identity_fact_forget",
                forget_scope="identity_fact",
                forget_fact_ids=(fact_ids[0],),
            )
        )
    prefix = CELEBA40_NAMESPACE + "."
    # Only accepted observations (explicit boolean label) may drive an
    # attribute-level forget spec; uncertain labels never select an attribute.
    attrs = sorted(
        {
            k[len(prefix):]
            for s in samples
            for k, obs in s.visual_attributes.items()
            if k.startswith(prefix) and obs.label is not None
        }
    )
    if attrs:
        specs.append(
            SplitSpec(name="attribute_forget", forget_scope="global_attribute", attribute=attrs[0])
        )
    return specs


def _build_split_results(samples):
    from .build.split_generation import SplitBuilder

    return [SplitBuilder(samples).build(spec) for spec in _split_specs_for(samples)]


# --------------------------------------------------------------------------- #
# model ...
# --------------------------------------------------------------------------- #


def cmd_model_inspect(args) -> int:
    cfg = load_model_config(args.config)
    from .models.registry import ensure_backends_loaded

    _print_json(
        {
            "config_path": str(args.config),
            "backend": cfg.backend,
            "available_backends": ensure_backends_loaded(),
            "model_id": cfg.model_id,
            "revision": cfg.revision,
            "dtype": cfg.dtype,
            "device_map": cfg.device_map,
            "quantization": {
                "enabled": cfg.quantization.enabled,
                "mode": cfg.quantization.mode,
            },
            "generation": {
                "do_sample": cfg.generation.do_sample,
                "temperature": cfg.generation.temperature,
                "max_new_tokens": cfg.generation.max_new_tokens,
            },
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
        }
    )
    return 0


def _image_sha256(image_path: str | Path) -> str:
    """Compute SHA-256 hex digest of an image file."""
    import hashlib
    data = Path(image_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def cmd_model_smoke_test(args) -> int:
    cfg = load_model_config(args.config)
    if args.dry_run:
        log.info("[dry-run] model config %s is valid; backend=%s", args.config, cfg.backend)
        return 0
    from .models.registry import create_backend

    backend = create_backend(cfg)

    # P0-2: real backends require a real image; only stub may use placeholder.
    image_path = getattr(args, "image", None)
    if not image_path and cfg.backend != "stub":
        raise ConfigError(
            "--image is required for non-stub smoke tests "
            f"(backend={cfg.backend})"
        )
    if image_path:
        image = _load_image(image_path)
        image_hash = _image_sha256(image_path)
    else:
        image = _load_image(None)
        image_hash = "placeholder_gray_image"

    # P0-1: construct PromptRegistry with a proper PromptsConfig, not a raw
    # string path.  The constructor expects a PromptsConfig dataclass.
    prompts_path = getattr(args, "prompts", None) or "configs/prompts/celeba_binary_v1.yaml"
    from .config import PromptsConfig
    from .prompts.registry import PromptRegistry

    prompt_cfg = PromptsConfig(binary=prompts_path, grouped=None, route_conflict=None)
    registry = PromptRegistry(prompt_cfg)

    # Three obvious attribute questions to verify visual discrimination.
    smoke_attributes = ["Eyeglasses", "Smiling", "Wearing_Hat"]

    results: list[dict[str, Any]] = []
    for attr in smoke_attributes:
        try:
            prompt = registry.binary_prompt(attr)
        except Exception:
            prompt = f"Is this person showing {attr.replace('_', ' ').lower()}? Answer yes or no."

        gen = backend.generate(image, prompt)
        scored = backend.score_candidates(image, prompt, [" yes", " no"])
        p = _p_positive(scored)
        logps = {
            cs.candidate: cs.log_probability for cs in (scored.candidate_scores or [])
        }
        results.append({
            "attribute": attr,
            "prompt": prompt,
            "generated_text": gen.text,
            "log_p_yes": logps.get(" yes"),
            "log_p_no": logps.get(" no"),
            "p_positive": p,
        })

    fp = backend.fingerprint()
    resolved_revision = getattr(backend, "_resolved_revision", None) or fp.get("revision", "n/a")

    _print_json({
        "model_fingerprint": fp,
        "resolved_revision": resolved_revision,
        "image_sha256": image_hash,
        "image_path": image_path or "placeholder",
        "smoke_results": results,
    })
    return 0


# --------------------------------------------------------------------------- #
# celeba ...
# --------------------------------------------------------------------------- #


def cmd_celeba_validate_raw(args) -> int:
    cfg = load_data_config(args.config)
    root = cfg.require_root()
    if args.dry_run:
        log.info("[dry-run] would validate CelebA root %s", root)
        return 0
    from .data.celeba import validate_raw

    report = validate_raw(
        root,
        sample_open=cfg.image_probe_count if not args.limit else min(args.limit, cfg.image_probe_count),
        seed=cfg.image_probe_seed,
    )
    _print_json(
        {
            "root": str(root),
            "ok": report.ok,
            "counts": report.counts,
            "errors": report.errors,
            "warnings": report.warnings,
        }
    )
    return 0 if report.ok else 1


def cmd_celeba_prepare(args) -> int:
    cfg = load_data_config(args.config)
    root = cfg.require_root()
    out_dir = Path(args.output_dir) if args.output_dir else Path("data/manifests")
    if args.dry_run:
        log.info("[dry-run] would build CelebA manifests from %s into %s", root, out_dir)
        return 0
    from .data.celeba import build_long_manifest, build_wide_manifest, validate_raw

    validate_raw(root, sample_open=0).raise_if_invalid()
    wide = build_wide_manifest(
        root, compute_sha256=cfg.compute_checksums, source_version=cfg.source_version
    )
    if args.limit:
        wide = wide.head(args.limit)
    long = build_long_manifest(
        root, compute_sha256=False, source_version=cfg.source_version
    )
    if args.limit:
        long = long[long["image_filename"].isin(wide["image_filename"])]
    write_parquet(wide.to_dict(orient="records"), out_dir / "celeba_manifest_wide.parquet")
    write_parquet(long.to_dict(orient="records"), out_dir / "celeba_manifest_long.parquet")
    summary = {
        "root": str(root),
        "images": int(len(wide)),
        "splits": wide["split"].value_counts().to_dict(),
        "attributes": int(sum(1 for c in wide.columns if c.startswith("attr::"))),
    }
    write_json(summary, out_dir / "celeba_manifest_summary.json")
    _print_json(summary)
    return 0


def _celeba_manifest_for_eval(run_cfg: RunConfig) -> pd.DataFrame:
    from .data.celeba import build_wide_manifest

    eval_cfg = run_cfg.evaluation
    if eval_cfg.manifest:
        path = Path(eval_cfg.manifest)
        if not path.is_absolute():
            path = REPO_ROOT / "data" / "manifests" / path.name
        if not path.exists():
            raise ConfigError(f"evaluation.manifest not found: {path}")
        df = pd.read_parquet(path)
    else:
        df = build_wide_manifest(
            run_cfg.data.require_root(),
            compute_sha256=run_cfg.data.compute_checksums,
            source_version=run_cfg.data.source_version,
        )
    if eval_cfg.split:
        df = df[df["split"] == eval_cfg.split].reset_index(drop=True)
    return df


def cmd_celeba_evaluate(args) -> int:
    run_cfg = load_run_config(args.config)
    if args.resume:
        run_cfg.evaluation.resume = True
    manifest = _celeba_manifest_for_eval(run_cfg)
    log.info(
        "Evaluation split=%s images=%d mode=%s scoring=%s",
        run_cfg.evaluation.split,
        len(manifest),
        run_cfg.evaluation.mode,
        run_cfg.evaluation.scoring,
    )
    if args.dry_run:
        log.info("[dry-run] config + manifest OK; no model loaded")
        return 0

    from .eval.celeba_runner import CelebaRunner
    from .models.registry import create_backend
    from .prompts.registry import PromptRegistry

    backend = create_backend(run_cfg.model)
    registry = PromptRegistry(run_cfg.prompts)
    runner = CelebaRunner(run_cfg, backend, registry, manifest, output_dir=args.output_dir)
    written = runner.run(limit=args.limit)
    summary = runner.summarize()
    bundle = runner.write_bundle(args.output_dir)
    _print_json({"rows_written": written, "macro": summary.get("macro"), "bundle": {k: str(v) for k, v in bundle.items()}})
    return 0


def _run_bundle_dir(args) -> Path:
    base = Path(args.output_dir) if args.output_dir else Path("outputs/runs")
    return base / args.run_id


def _predictions_for_bundle(bundle_dir: Path) -> pd.DataFrame:
    """Locate prediction shards for a run bundle (fail loud if absent)."""
    candidates = [bundle_dir / "predictions", bundle_dir]
    resolved = bundle_dir / "resolved_config.yaml"
    if resolved.exists():
        doc = yaml.safe_load(resolved.read_text()) or {}
        extra = (doc.get("evaluation") or {}).get("output_dir")
        if extra:
            extra_path = Path(extra)
            candidates.insert(0, extra_path if extra_path.is_absolute() else REPO_ROOT / extra_path)
    for candidate in candidates:
        try:
            df = read_shards(candidate, prefix="part")
            if len(df):
                return df
        except Exception:  # noqa: BLE001 - keep scanning
            continue
    raise ConfigError(f"No prediction shards found for run bundle {bundle_dir}")


def cmd_celeba_report(args) -> int:
    bundle_dir = _run_bundle_dir(args)
    metrics_path = bundle_dir / "metrics.json"
    if metrics_path.exists() and not args.resume:
        _print_json(read_json(metrics_path))
        return 0
    # (Re)build the report from raw prediction shards.
    df = _predictions_for_bundle(bundle_dir)
    if args.limit:
        df = df.head(args.limit)
    from .eval.metrics import compute_binary_metrics, macro_average

    per_attribute: dict[str, dict[str, Any]] = {}
    for attr, rows in df.groupby("attribute"):
        parsed = rows[rows["parse_status"] == "ok"]
        y_true = parsed["label"].dropna().astype(int).tolist()
        y_pred = parsed.loc[parsed["label"].notna(), "prediction"].astype(int).tolist()
        p = parsed.loc[parsed["label"].notna(), "p_positive"]
        per_attribute[str(attr)] = compute_binary_metrics(
            y_true,
            y_pred,
            p_positive=p.tolist() if p.notna().any() else None,
            parse_failures=int((rows["parse_status"] != "ok").sum()),
            total_queries=int(len(rows)),
            latency_ms=rows["latency_ms"].dropna().tolist() or None,
        )
    macro = macro_average(per_attribute)
    from .eval.reports import write_metrics_bundle, render_report_md
    from .data.io import ensure_parent_dir

    if args.dry_run:
        log.info("[dry-run] would rewrite metrics bundle under %s", bundle_dir)
        return 0
    write_metrics_bundle(bundle_dir, per_attribute, macro)
    report = render_report_md(per_attribute, macro, n_images=int(df["sample_id"].nunique()))
    report_path = bundle_dir / "report.md"
    ensure_parent_dir(report_path)
    report_path.write_text(report)
    _print_json({"macro": macro, "report": str(report_path)})
    return 0


def cmd_celeba_freeze_protocol(args) -> int:
    bundle_dir = _run_bundle_dir(args)
    df = _predictions_for_bundle(bundle_dir)
    if args.limit:
        df = df.head(args.limit)
    usable = df[(df["label"].notna()) & (df["p_positive"].notna())]
    if args.dry_run:
        log.info(
            "[dry-run] would freeze calibrators for %d attributes from %d rows",
            usable["attribute"].nunique(),
            len(usable),
        )
        return 0
    from .eval.calibration import fit_calibrator, save_calibrators

    calibrators = {}
    skipped: list[str] = []
    for attr, rows in usable.groupby("attribute"):
        if rows["label"].nunique() < 2:
            skipped.append(str(attr))
            continue
        calibrators[str(attr)] = fit_calibrator(
            "platt", rows["p_positive"].tolist(), rows["label"].astype(int).tolist()
        )
    calib_path = bundle_dir / "calibrators.json"
    save_calibrators(calibrators, calib_path)
    run_meta_path = bundle_dir / "run.yaml"
    prompts_version = None
    if run_meta_path.exists():
        prompts_version = (yaml.safe_load(run_meta_path.read_text()) or {}).get(
            "prompt_registry_hash"
        )
    protocol = {
        "frozen_protocol": {
            "prompts_version": prompts_version or "celeba_binary_v1",
            "scoring": "candidate",
            "calibrators": str(calib_path),
        },
        "calibrated_attributes": sorted(calibrators),
        "skipped_single_class_attributes": sorted(skipped),
    }
    write_json(protocol, bundle_dir / "frozen_protocol.json")
    with open(bundle_dir / "frozen_protocol.yaml", "w") as f:
        yaml.safe_dump(protocol, f, sort_keys=False)
    _print_json(protocol)
    return 0


# --------------------------------------------------------------------------- #
# source inspect ...
# --------------------------------------------------------------------------- #


def _type_signature(value: Any) -> str:
    """Stable, JSON-friendly type label for schema snapshots (repair plan D2)."""
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    return type(value).__name__


def _inspection_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    """Stable subset of an inspection report used for drift checks (D2/D3)."""
    return {
        "dataset": report["dataset"],
        "adapter_version": report["adapter_version"],
        "source_revision": report["source_revision"],
        "fields": report["fields"],
        "field_types": report["field_types"],
        "task_types": report["task_types"],
        "modalities": report["modalities"],
        "image_fields": report["image_fields"],
    }


def _compare_schema(live: dict[str, Any], pinned: dict[str, Any]) -> list[str]:
    """Schema-drift errors between a live report and a pinned snapshot (D3)."""
    problems: list[str] = []
    live_fields = set(live["fields"])
    for field in pinned.get("fields", []):
        if field not in live_fields:
            problems.append(f"required source field missing: {field}")
    for field, pinned_type in pinned.get("field_types", {}).items():
        live_type = live["field_types"].get(field)
        if live_type is None:
            continue  # already reported as missing above
        if live_type != pinned_type:
            problems.append(
                f"source field type changed: {field} {pinned_type} -> {live_type}"
            )
    if pinned.get("adapter_version") and pinned["adapter_version"] != live["adapter_version"]:
        problems.append(
            "adapter version changed: "
            f"{pinned['adapter_version']} -> {live['adapter_version']}"
        )
    return problems


def cmd_source_inspect(args) -> int:
    from .data.adapters.base import AdapterError, available_adapters, create_adapter

    cfg = load_data_config(args.config)
    if args.dataset not in available_adapters():
        raise ConfigError(
            f"Unknown dataset '{args.dataset}'; available adapters: {available_adapters()}"
        )
    adapter = create_adapter(cfg)
    report: dict[str, Any] = {
        "dataset": args.dataset,
        "adapter": adapter.name,
        "adapter_version": adapter.adapter_version,
        "source_revision": cfg.source_version,
        "hf_config_name": cfg.extras.get("hf_config_name"),
        "hf_split": cfg.extras.get("hf_split"),
        "field_map": adapter.field_map,
    }
    if args.dry_run:
        report["dry_run"] = "configuration and adapter constructed; no rows read"
        _print_json(report)
        return 0

    # Inspect every raw row and its canonical expansion (repair plan D1):
    # never summarize from only the first row.
    limit = args.limit or 20
    field_types: dict[str, str] = {}
    nested_fields: dict[str, dict[str, str]] = {}
    list_lengths: dict[str, list[int]] = {}
    image_fields: set[str] = set()
    task_types: set[str] = set()
    modalities: set[str] = set()
    identities: set[str] = set()
    warnings: list[str] = []
    mapping_errors: list[str] = []
    n_rows = 0
    n_samples = 0
    n_rows_with_images = 0
    first_row: dict[str, Any] | None = None
    rows = itertools.islice(adapter.iter_rows_with_context(), limit)
    for context, row in rows:
        n_rows += 1
        if first_row is None:
            first_row = row
        for key, value in row.items():
            signature = _type_signature(value)
            existing = field_types.get(key)
            if existing and existing != signature:
                field_types[key] = f"mixed<{existing}|{signature}>"
            elif existing is None:
                field_types[key] = signature
            if isinstance(value, dict):
                nested = nested_fields.setdefault(key, {})
                for sub_key, sub_value in value.items():
                    nested.setdefault(sub_key, _type_signature(sub_value))
            elif isinstance(value, list):
                list_lengths.setdefault(key, []).append(len(value))
            elif isinstance(value, (str, bytes, Path)) and str(value):
                text = str(value)
                if any(text.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp")):
                    image_fields.add(key)
        try:
            samples = adapter.to_samples(row, source_context=context)
        except AdapterError as exc:
            mapping_errors.append(f"row {n_rows}: {exc}")
            continue
        for sample in samples:
            n_samples += 1
            identities.add(sample.identity_id)
            task_types.add(sample.task_type or "unset")
            modalities.add(sample.modality)
            if sample.image_uri:
                n_rows_with_images += 1
            elif sample.modality != "text_only":
                warnings.append(
                    f"row {n_rows}: sample '{sample.source_sample_id}' has modality "
                    f"'{sample.modality}' but no image_uri"
                )
    report.update(
        {
            "rows_inspected": n_rows,
            "fields": sorted(field_types),
            "field_types": field_types,
            "nested_fields": nested_fields,
            "list_lengths": {k: sorted(set(v)) for k, v in list_lengths.items()},
            "image_fields": sorted(image_fields),
            "rows_with_images": n_rows_with_images,
            "canonical_samples": n_samples,
            "expansion_ratio": (n_samples / n_rows) if n_rows else 0.0,
            "identities": len(identities),
            "task_types": sorted(task_types),
            "modalities": sorted(modalities),
            "first_row": first_row,
            "mapping_errors": mapping_errors,
            "warnings": sorted(set(warnings)),
        }
    )
    # Persist the inspection snapshot for schema-drift checks (repair plan D2).
    snapshot_path = Path(args.snapshot) if args.snapshot else (
        REPO_ROOT / "outputs" / "source_inspection" / args.dataset / f"{Path(args.config).stem}.json"
    )
    write_json(_inspection_snapshot(report), snapshot_path)
    report["snapshot"] = str(snapshot_path)

    exit_code = 1 if mapping_errors else 0
    if args.check_schema and snapshot_path.exists():
        pinned = read_json(snapshot_path)
        problems = _compare_schema(_inspection_snapshot(report), pinned)
        report["schema_drift"] = problems
        if problems:
            exit_code = 1
    _print_json(report)
    return exit_code


# --------------------------------------------------------------------------- #
# build ...
# --------------------------------------------------------------------------- #


def _adapter_samples(args, run_cfg, dataset: str):
    from .data.adapters.base import create_adapter

    data_cfg = _data_config_for(dataset, run_cfg)
    adapter = create_adapter(data_cfg)
    samples = list(itertools.islice(adapter.load(), args.limit) if args.limit else adapter.load())
    return data_cfg, samples


def cmd_build_annotate(args) -> int:
    run_cfg = load_run_config(args.config)
    dataset_dir = _dataset_dir(args, run_cfg, args.dataset)
    data_cfg, samples = _adapter_samples(args, run_cfg, args.dataset)
    scores_path = dataset_dir / f"{args.dataset}_model_scores.jsonl"
    annotated_path = dataset_dir / f"{args.dataset}_annotated_all.jsonl"
    annotated_working = dataset_dir / f"{args.dataset}_annotated.jsonl"
    processed_path = dataset_dir / f"{args.dataset}_processed.jsonl"

    if args.dry_run:
        log.info(
            "[dry-run] annotate %s: %d samples loaded; scores=%s annotated=%s processed=%s",
            args.dataset,
            len(samples),
            scores_path,
            annotated_path,
            processed_path,
        )
        return 0

    from .build.annotate import (
        AnnotationPolicy,
        BenchmarkAnnotator,
        load_frozen_calibrators,
        predictions_to_scores,
    )
    from .constants.celeba_attributes import CELEBA_ATTRIBUTES

    # -- 0. Load whitelist (P0-1) ---------------------------------------- #
    whitelist_attrs: frozenset[str] | None = None
    if run_cfg.build.attribute_whitelist:
        from .build.whitelist import load_attribute_whitelist

        wl_path = Path(run_cfg.build.attribute_whitelist)
        if not wl_path.is_absolute():
            base = Path(__file__).resolve().parents[2]
            wl_path = (base / wl_path).resolve()
        wl = load_attribute_whitelist(wl_path, expected_model_id=run_cfg.model.model_id)
        whitelist_attrs = wl.attributes
        log.info(
            "Whitelist loaded: %d attributes for model %s (sha256=%s)",
            len(wl.attributes), wl.model_id, wl.sha256[:12],
        )

    gated = whitelist_attrs if whitelist_attrs is not None else frozenset(CELEBA_ATTRIBUTES)

    # -- 1. Model scores (resumable via the scores JSONL cache) ---------- #
    # Create backend FIRST so fingerprint is available for cache key (P0-7).
    from .models.registry import create_backend
    from .prompts.registry import PromptRegistry

    backend = create_backend(run_cfg.model)
    registry = PromptRegistry(run_cfg.prompts)

    # Fix 2: force model load before fingerprint so _resolved_revision is
    # populated (otherwise the Hub snapshot revision is "unresolved" because
    # revision: null in the config).
    try:
        if hasattr(backend, "_load"):
            backend._load()
    except Exception as exc:  # noqa: BLE001 - model load may fail in dry envs
        if run_cfg.model.backend == "stub":
            log.warning("Backend pre-load failed (%s); fingerprint may lack resolved revision", exc)
        else:
            raise ConfigError(f"Backend pre-load failed: {exc}") from exc

    # P0-7: fingerprint and resolved revision are mandatory for real backends.
    fingerprint_id: str | None = None
    fingerprint_data: dict[str, Any] = {}
    try:
        fingerprint_data = backend.fingerprint()
        fingerprint_id = str(fingerprint_data.get("fingerprint_id"))
    except Exception as exc:  # noqa: BLE001
        if run_cfg.model.backend == "stub":
            log.warning("Backend fingerprint unavailable (%s); continuing without it", exc)
        else:
            raise ConfigError(
                f"Model fingerprint is required for backend={run_cfg.model.backend!r}"
            ) from exc

    if run_cfg.model.backend != "stub":
        resolved_revision = fingerprint_data.get("revision") or getattr(
            backend, "_resolved_revision", None
        )
        if not fingerprint_id:
            raise ConfigError("Model fingerprint_id is required for real backends")
        if not resolved_revision or resolved_revision == "unresolved":
            raise ConfigError(
                "Resolved model revision is required; ensure the model loads "
                "successfully before annotation"
            )

    # Fix 2: comprehensive cache key that uniquely identifies the scoring
    # identity.  Includes model fingerprint (which embeds model_id, resolved
    # revision, dtype, quantization, transformers/torch versions), prompt
    # registry hash, candidate-set hash, and scoring-code version.
    from .models.scoring import SCORING_VERSION
    import hashlib as _hashlib_cache

    candidates_blob = json.dumps([" yes", " no"], sort_keys=True)
    candidate_set_hash = _hashlib_cache.sha256(candidates_blob.encode()).hexdigest()[:12]

    cache_parts = [
        fingerprint_id or "nofp",
        registry.registry_hash(),
        candidate_set_hash,
        f"sv{SCORING_VERSION}",
    ]
    cache_key_suffix = "|".join(cache_parts)

    image_base = Path(data_cfg.root or data_cfg.extras.get("local_root") or ".")

    # Build sample_id -> image_sha256 lookup for per-row cache keys (Fix 2).
    # Including image_sha256 ensures that replacing an image while keeping
    # the same sample_id does not reuse stale cached scores.
    image_sha_by_sample: dict[str, str] = {}
    for s in samples:
        sha = s.image_sha256 or ""
        if not sha and s.image_uri:
            try:
                sha = _image_sha256(Path(image_base) / s.image_uri.removeprefix("file://"))
            except Exception:  # noqa: BLE001
                sha = ""
        image_sha_by_sample[s.source_sample_id] = sha

    def _row_cache_key(sample_id: str) -> str:
        return f"{image_sha_by_sample.get(sample_id, '')}|{cache_key_suffix}"

    done_keys: set[tuple[str, str]] = set()
    score_rows: list[dict[str, Any]] = []
    if scores_path.exists() and args.resume:
        raw_rows = list(read_jsonl(scores_path))
        for r in raw_rows:
            sid = r.get("sample_id")
            expected_key = _row_cache_key(sid) if sid else None
            if r.get("_cache_key", "") == expected_key:
                score_rows.append(r)
                done_keys.add((sid, r["attribute"]))
        if len(score_rows) < len(raw_rows):
            log.info(
                "Resume: kept %d/%d cached rows matching current cache key",
                len(score_rows), len(raw_rows),
            )

    sample_ids = {s.source_sample_id for s in samples}
    score_rows = [r for r in score_rows if r["sample_id"] in sample_ids]
    pending = [
        (s, attr)
        for s in samples
        for attr in CELEBA_ATTRIBUTES
        if s.image_uri
        and (s.source_sample_id, attr) not in done_keys
    ]
    if pending:
        log.info(
            "Scoring %d (image, attribute) queries with backend=%s",
            len(pending), run_cfg.model.backend,
        )
        for sample, attr in pending:
            image = _load_image(sample.image_uri, base=image_base)
            resp = backend.score_candidates(
                image, registry.binary_prompt(attr), [" yes", " no"]
            )
            p = _p_positive(resp)
            if p is None:
                continue
            # Reject inconsistent duplicates (Fix 10)
            if (sample.source_sample_id, attr) in done_keys:
                log.warning(
                    "Duplicate score for (%s, %s); keeping first occurrence",
                    sample.source_sample_id, attr,
                )
                continue
            score_rows.append(
                {
                    "sample_id": sample.source_sample_id,
                    "attribute": attr,
                    "p_positive": p,
                    "raw_text": resp.text,
                    "_cache_key": _row_cache_key(sample.source_sample_id),
                }
            )
            done_keys.add((sample.source_sample_id, attr))
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # P1-11: strictly validate score-cache rows (resume path).  Reject
    # NaN, Inf, out-of-range, and duplicate entries so stale or corrupt
    # caches never silently contaminate downstream annotations.
    import math as _math

    validated_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for r in score_rows:
        p = r.get("p_positive")
        sid = r.get("sample_id", "")
        attr = r.get("attribute", "")
        if p is None or not isinstance(p, (int, float)):
            log.warning("Dropping score row (%s, %s): p_positive is not numeric", sid, attr)
            continue
        if _math.isnan(p) or _math.isinf(p):
            log.warning("Dropping score row (%s, %s): p_positive=%s", sid, attr, p)
            continue
        if not (0.0 <= p <= 1.0):
            log.warning("Dropping score row (%s, %s): p_positive=%s out of [0,1]", sid, attr, p)
            continue
        key = (sid, attr)
        if key in seen_keys:
            log.warning("Dropping duplicate score row (%s, %s)", sid, attr)
            continue
        seen_keys.add(key)
        validated_rows.append(r)
    if len(validated_rows) < len(score_rows):
        log.info(
            "Score validation: kept %d/%d rows after NaN/Inf/range/dedup checks",
            len(validated_rows), len(score_rows),
        )
    score_rows = validated_rows

    write_jsonl(score_rows, scores_path)

    # P1-10: write a dedicated score manifest capturing the full immutable
    # scoring identity so the cache is auditable and reproducible.
    # P1-15: compute a source-data hash to pin the exact source revision.
    source_hash: str | None = None
    try:
        source_root = data_cfg.require_root()
        _h = _hashlib_cache.sha256()
        for p in sorted(source_root.rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                _h.update(str(p.relative_to(source_root)).encode())
                _h.update(p.read_bytes())
        source_hash = _h.hexdigest()[:16]
    except Exception as exc:  # noqa: BLE001
        log.warning("Source hash unavailable (%s); skipping source pinning", exc)

    resolved_revision = fingerprint_data.get("revision") or getattr(
        backend, "_resolved_revision", None
    )
    score_manifest = {
        "dataset": args.dataset,
        "model_fingerprint": fingerprint_id,
        "resolved_revision": resolved_revision,
        "prompt_registry_hash": registry.registry_hash(),
        "scoring_version": SCORING_VERSION,
        "candidate_set_hash": candidate_set_hash,
        "backend": run_cfg.model.backend,
        "model_id": run_cfg.model.model_id,
        "score_rows": len(score_rows),
        "source_version": data_cfg.source_version,
        "source_hash": source_hash,
    }
    manifest_path = dataset_dir / f"{args.dataset}_score_manifest.json"
    write_json(score_manifest, manifest_path)

    # -- 2. Frozen-protocol annotation (all 40 attributes) --------------- #
    policy = AnnotationPolicy(
        gated_attributes=gated,
        bands=dict(run_cfg.build.confidence_bands) or None,
        min_auto_accept_score=run_cfg.build.min_auto_accept_score,
    )
    calibrators = load_frozen_calibrators(run_cfg.frozen_protocol.calibrators)
    annotator = BenchmarkAnnotator(
        policy,
        calibrators=calibrators,
        model_fingerprint=fingerprint_id,
        prompt_registry_hash=registry.registry_hash(),
    )
    scores_by_sample = predictions_to_scores(score_rows)
    annotated = [
        annotator.annotate_sample(s, scores_by_sample.get(s.source_sample_id, {}))
        for s in samples
    ]
    write_jsonl([s.to_dict() for s in annotated], annotated_path)
    write_jsonl([s.to_dict() for s in annotated], annotated_working)

    # -- 3. Processed stage: whitelist-restricted labels (P0-2) ---------- #
    if whitelist_attrs is not None:
        processed_policy = AnnotationPolicy(
            gated_attributes=whitelist_attrs,
            bands=dict(run_cfg.build.confidence_bands) or None,
            min_auto_accept_score=run_cfg.build.min_auto_accept_score,
        )
        processed_annotator = BenchmarkAnnotator(
            processed_policy,
            calibrators=calibrators,
            model_fingerprint=fingerprint_id,
            prompt_registry_hash=registry.registry_hash(),
        )
        processed = [
            processed_annotator.annotate_sample(
                s, scores_by_sample.get(s.source_sample_id, {})
            )
            for s in samples
        ]
        write_jsonl([s.to_dict() for s in processed], processed_path)
        n_obs = sum(len(s.visual_attributes) for s in processed)
        n_labeled = sum(
            1 for s in processed for obs in s.visual_attributes.values()
            if obs.label is not None
        )
    else:
        # No whitelist: processed == annotated (P0-2: always write the
        # processed artifact so downstream stages have a consistent path).
        processed = annotated
        write_jsonl([s.to_dict() for s in processed], processed_path)
        n_obs = sum(len(s.visual_attributes) for s in processed)
        n_labeled = sum(
            1 for s in processed for obs in s.visual_attributes.values()
            if obs.label is not None
        )

    _print_json(
        {
            "dataset": args.dataset,
            "samples": len(processed),
            "observations": n_obs,
            "accepted_labels": n_labeled,
            "whitelist_attributes": sorted(whitelist_attrs) if whitelist_attrs else "all",
            "scores_path": str(scores_path),
            "annotated_path": str(annotated_path),
            "processed_path": str(processed_path),
        }
    )
    return 0


def cmd_build_qa(args) -> int:
    run_cfg = load_run_config(args.config)
    dataset_dir = _dataset_dir(args, run_cfg, args.dataset)
    samples = _load_processed_samples(dataset_dir, args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    if args.dry_run:
        log.info("[dry-run] qa for %d samples of %s", len(samples), args.dataset)
        return 0

    from .build.annotate import CELEBA40_NAMESPACE
    from .build.qa_generation import QaTemplateRegistry, generate_qa_rows

    prefix = CELEBA40_NAMESPACE + "."
    attributes = sorted(
        {k[len(prefix):] for s in samples for k in s.visual_attributes if k.startswith(prefix)}
    )
    if not attributes:
        raise ConfigError(f"No CelebA-40 observations in {dataset_dir}; run build annotate first")
    registry = QaTemplateRegistry.default_for(attributes)

    # Deterministic, identity-level split for QA generation.
    # P1-8: respect official source partitions when available.  Identities
    # with split == "retain_train" go to train, "retain_eval" go to eval.
    # Identities with other splits (e.g. "forget", "unassigned") fall back
    # to hash-based assignment.
    # P1-9: guarantee at least 1 train + 1 eval identity even for tiny inputs.
    import hashlib as _hashlib

    train_identity_ids: set[str] = set()
    eval_identity_ids: set[str] = set()
    identity_to_split: dict[str, str] = {}

    # Pass 1: honor official source partitions.
    for s in samples:
        iid = s.identity_id
        if iid in identity_to_split:
            continue
        if s.split == "retain_train":
            identity_to_split[iid] = "train"
            train_identity_ids.add(iid)
        elif s.split == "retain_eval":
            identity_to_split[iid] = "eval"
            eval_identity_ids.add(iid)

    # Pass 2: hash-based fallback for identities without official assignment.
    for s in samples:
        iid = s.identity_id
        if iid in identity_to_split:
            continue
        h = int(_hashlib.sha256(iid.encode()).hexdigest()[:8], 16)
        split = "eval" if h % 5 == 0 else "train"
        identity_to_split[iid] = split
        if split == "train":
            train_identity_ids.add(iid)
        else:
            eval_identity_ids.add(iid)

    # P1-9: guarantee at least 1 identity in each split.  If one split is
    # empty, move the identity with the most samples from the larger split.
    if not eval_identity_ids and train_identity_ids:
        # Pick the identity with the most samples to move to eval.
        counts = {}
        for s in samples:
            counts[s.identity_id] = counts.get(s.identity_id, 0) + 1
        donor = max(train_identity_ids, key=lambda iid: counts.get(iid, 0))
        train_identity_ids.discard(donor)
        eval_identity_ids.add(donor)
        identity_to_split[donor] = "eval"
    elif not train_identity_ids and eval_identity_ids:
        counts = {}
        for s in samples:
            counts[s.identity_id] = counts.get(s.identity_id, 0) + 1
        donor = max(eval_identity_ids, key=lambda iid: counts.get(iid, 0))
        eval_identity_ids.discard(donor)
        train_identity_ids.add(donor)
        identity_to_split[donor] = "train"

    # Enforce identity-disjoint invariant
    assert not (train_identity_ids & eval_identity_ids), "identity split leakage"

    train_samples = []
    eval_samples = []
    for s in samples:
        if identity_to_split[s.identity_id] == "train":
            train_samples.append(s)
        else:
            eval_samples.append(s)

    train_rows = generate_qa_rows(train_samples, registry, split="train")
    eval_rows = generate_qa_rows(eval_samples, registry, split="validation")
    train_path = dataset_dir / f"{args.dataset}_visual_qa_train.jsonl"
    eval_path = dataset_dir / f"{args.dataset}_visual_qa_eval.jsonl"
    write_jsonl(train_rows, train_path)
    write_jsonl(eval_rows, eval_path)
    _print_json(
        {
            "dataset": args.dataset,
            "registry_hash": registry.registry_hash(),
            "registry_version": registry.version,
            "train_samples": len(train_samples),
            "eval_samples": len(eval_samples),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_path": str(train_path),
            "eval_path": str(eval_path),
        }
    )
    return 0


def cmd_build_route_probes(args) -> int:
    run_cfg = load_run_config(args.config)
    dataset_dir = _dataset_dir(args, run_cfg, args.dataset)
    samples = _load_processed_samples(dataset_dir, args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    if args.dry_run:
        log.info("[dry-run] route probes for %d samples of %s", len(samples), args.dataset)
        return 0

    from .build.conflict_generation import (
        ConflictError,
        RouteProbeBuilder,
        _accepted_visible_attributes,
        build_identity_probes,
        build_pair_manifest,
        validate_pair_manifest,
    )
    from .prompts.registry import PromptRegistry

    registry = PromptRegistry(run_cfg.prompts)
    builder = RouteProbeBuilder(registry)

    by_identity: dict[str, list] = {}
    for sample in samples:
        by_identity.setdefault(sample.identity_id, []).append(sample)

    probe_rows: list[dict[str, Any]] = []
    pair_specs: list[dict[str, str]] = []
    skipped: list[str] = []
    identity_ids = sorted(by_identity)
    for identity_id in identity_ids:
        group = by_identity[identity_id]

        # P0-13 + Fix 6: pick a wrong name from an identity with multiple
        # samples and accepted visible attributes.  Sort for deterministic
        # selection (alphabetical by identity name) so results are
        # reproducible.  NOTE: for research-grade causal analysis this
        # should match the wrong identity on available visual properties
        # or generate several wrong-name controls; the current approach is
        # acceptable for engineering smoke tests only.
        wrong_name_candidates = sorted(
            (by_identity[other][0].identity_name or other)
            for other in identity_ids
            if other != identity_id
            and len(by_identity[other]) >= 2
            and _accepted_visible_attributes(by_identity[other][0])
        )
        wrong_name = wrong_name_candidates[0] if wrong_name_candidates else None

        try:
            probes = build_identity_probes(
                group, builder, wrong_identity_name=wrong_name
            )
        except ConflictError as exc:
            skipped.append(f"{identity_id}: {exc}")
            continue
        for probe in probes:
            probe_rows.append(builder.probe_row(probe))

        # P0-12 + Fix 4: validate cross_image_attribute_state pairs and
        # emit one explicit pair per differing target attribute so the
        # attribute that changed is recorded, not just "some attribute
        # differs".
        if len(group) >= 2:
            for left, right in zip(group, group[1:]):
                left_attrs = _accepted_visible_attributes(left)
                right_attrs = _accepted_visible_attributes(right)
                shared = set(left_attrs) & set(right_attrs)
                for attr_name in sorted(shared):
                    if left_attrs[attr_name] != right_attrs[attr_name]:
                        pair_specs.append(
                            {
                                "pair_type": "cross_image_attribute_state",
                                "left_sample_id": left.source_sample_id,
                                "right_sample_id": right.source_sample_id,
                                "attribute": attr_name,
                                "left_label": left_attrs[attr_name],
                                "right_label": right_attrs[attr_name],
                            }
                        )
    pairs = build_pair_manifest(pair_specs)

    # P1-13: semantic pair validation — verify same-identity, different-image,
    # accepted labels, etc.  Issues are logged but do not block the build
    # (the manifest is still written for downstream inspection).
    samples_by_id = {s.source_sample_id: s for s in samples}
    pair_issues = validate_pair_manifest(pairs, samples_by_id)
    if pair_issues:
        for issue in pair_issues:
            log.warning("Pair validation issue: %s", issue)

    probes_path = dataset_dir / f"{args.dataset}_route_probes.jsonl"
    write_jsonl(probe_rows, probes_path)
    write_json(pairs, dataset_dir / f"{args.dataset}_pair_manifest.json")
    _print_json(
        {
            "dataset": args.dataset,
            "probe_rows": len(probe_rows),
            "pairs": len(pairs),
            "pair_validation_issues": len(pair_issues),
            "skipped_identities": skipped,
            "probes_path": str(probes_path),
        }
    )
    return 0


def cmd_build_splits(args) -> int:
    run_cfg = load_run_config(args.config)
    dataset_dir = _dataset_dir(args, run_cfg, args.dataset)
    samples = _load_processed_samples(dataset_dir, args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    if args.dry_run:
        log.info("[dry-run] splits for %d samples of %s", len(samples), args.dataset)
        return 0

    from .build.split_generation import validate_split_invariants

    results = _build_split_results(samples)
    payload: dict[str, Any] = {"dataset": args.dataset, "splits": []}
    for result in results:
        issues = validate_split_invariants(result, strict=False)
        payload["splits"].append({**result.manifest(), "invariant_issues": issues})
    path = dataset_dir / f"{args.dataset}_split_manifest.json"
    write_json(payload, path)
    _print_json({**payload, "path": str(path)})
    return 0 if all(not s["invariant_issues"] for s in payload["splits"]) else 1


def cmd_build_export(args) -> int:
    run_cfg = load_run_config(args.config)
    dataset_dir = _dataset_dir(args, run_cfg, args.dataset)
    samples = _load_processed_samples(dataset_dir, args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    benchmark = _benchmark_of(args.dataset)
    if args.dry_run:
        log.info("[dry-run] export %d samples of %s to %s", len(samples), benchmark, dataset_dir)
        return 0

    from .build.export import ExtensionExporter
    from .prompts.registry import PromptRegistry

    train_qa = list(read_jsonl(dataset_dir / f"{args.dataset}_visual_qa_train.jsonl")) \
        if (dataset_dir / f"{args.dataset}_visual_qa_train.jsonl").exists() else []
    eval_qa = list(read_jsonl(dataset_dir / f"{args.dataset}_visual_qa_eval.jsonl")) \
        if (dataset_dir / f"{args.dataset}_visual_qa_eval.jsonl").exists() else []
    probe_rows = list(read_jsonl(dataset_dir / f"{args.dataset}_route_probes.jsonl")) \
        if (dataset_dir / f"{args.dataset}_route_probes.jsonl").exists() else []
    split_results = _build_split_results(samples)

    registry_hash: str | None = None
    try:
        registry_hash = PromptRegistry(run_cfg.prompts).registry_hash()
    except ConfigError as exc:
        log.warning("Prompt registry unavailable (%s); exporting without hash", exc)

    # Fix 8: extract model fingerprint from the samples' observations so the
    # dataset card and manifest show the actual annotator identity instead of
    # "n/a".
    model_fingerprint: str | None = None
    for s in samples:
        for obs in s.visual_attributes.values():
            if getattr(obs, "model_fingerprint", None):
                model_fingerprint = obs.model_fingerprint
                break
        if model_fingerprint:
            break

    # Fix 8: build a provenance manifest capturing the full immutable
    # generation identity so exports are auditable.
    data_cfg = _data_config_for(args.dataset, run_cfg)
    provenance: dict[str, Any] = {
        "model_id": run_cfg.model.model_id,
        "model_backend": run_cfg.model.backend,
        "model_revision": getattr(run_cfg.model, "revision", None),
        "model_fingerprint": model_fingerprint,
        "prompt_registry_hash": registry_hash,
        "source_version": data_cfg.source_version,
        "scoring_method": "candidate_sequence_log_probability",
    }
    # P1-15: propagate source_hash from score manifest (or compute fresh)
    # so the export provenance pins the exact source data revision.
    score_manifest_path = dataset_dir / f"{args.dataset}_score_manifest.json"
    if score_manifest_path.exists():
        try:
            sm = read_json(score_manifest_path)
            if sm.get("source_hash"):
                provenance["source_hash"] = sm["source_hash"]
        except Exception:  # noqa: BLE001
            pass
    if "source_hash" not in provenance:
        try:
            import hashlib as _hashlib_export
            source_root = data_cfg.require_root()
            _h = _hashlib_export.sha256()
            for p in sorted(source_root.rglob("*")):
                if p.is_file() and not p.name.startswith("."):
                    _h.update(str(p.relative_to(source_root)).encode())
                    _h.update(p.read_bytes())
            provenance["source_hash"] = _h.hexdigest()[:16]
        except Exception:  # noqa: BLE001
            pass
    # Optional: whitelist provenance.
    if run_cfg.build.attribute_whitelist:
        try:
            from .build.whitelist import load_attribute_whitelist
            wl_path = Path(run_cfg.build.attribute_whitelist)
            if not wl_path.is_absolute():
                wl_path = (Path(__file__).resolve().parents[2] / wl_path).resolve()
            wl = load_attribute_whitelist(wl_path)
            provenance["whitelist_sha256"] = wl.sha256
            provenance["whitelist_attributes"] = sorted(wl.attributes)
        except Exception as exc:  # noqa: BLE001
            log.warning("Whitelist provenance unavailable (%s)", exc)
    # Optional: library versions.
    try:
        import transformers
        provenance["transformers_version"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        provenance["torch_version"] = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    # Optional: MIDP git commit.
    try:
        import subprocess
        midp_root = Path(__file__).resolve().parents[4]
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=midp_root, stderr=subprocess.DEVNULL,
        ).decode().strip()
        provenance["midp_commit"] = sha
    except Exception:  # noqa: BLE001
        pass

    exporter = ExtensionExporter(
        dataset_dir.parent,  # base / model_dir; exporter appends benchmark
        benchmark,
        source_version=data_cfg.source_version,
        registry_hash=registry_hash,
        model_fingerprint=model_fingerprint,
    )
    record = exporter.export_all(
        samples,
        train_qa=train_qa,
        eval_qa=eval_qa,
        probe_rows=probe_rows,
        split_results=split_results,
        provenance=provenance,
    )
    _print_json(
        {
            "benchmark": record.benchmark,
            "counts": record.counts,
            "paths": {k: str(v) for k, v in record.paths.items()},
        }
    )
    return 0


# --------------------------------------------------------------------------- #
# validate / card
# --------------------------------------------------------------------------- #


def cmd_validate_dataset(args) -> int:
    run_cfg: RunConfig | None = None
    if args.config:
        run_cfg = load_run_config(args.config)
    base = Path(args.output_dir) if args.output_dir else (
        _default_build_dir(run_cfg) if run_cfg else Path("data/processed")
    )
    dataset = args.dataset
    benchmark = _benchmark_of(dataset)
    # Use model-specific output directory when run config is available
    if run_cfg is not None:
        model_dir = _model_output_name(run_cfg.model.model_id)
        dataset_dir = base / model_dir / dataset
        if not dataset_dir.exists():
            # Fall back to benchmark name (e.g., fairget_celeba40 -> fairget)
            dataset_dir = base / model_dir / benchmark
    else:
        dataset_dir = base / dataset
        if not dataset_dir.exists():
            dataset_dir = base / benchmark
    # Intermediate artifacts are named after the benchmark (``fairget``),
    # not the export name (``fairget_celeba40``).
    samples = _load_processed_samples(dataset_dir, benchmark)
    if args.limit:
        samples = samples[: args.limit]

    from .validation import validate_dataset

    qa_rows = []
    for name in (f"{benchmark}_visual_qa_train.jsonl", f"{benchmark}_visual_qa_eval.jsonl"):
        path = dataset_dir / name
        if path.exists():
            qa_rows.extend(read_jsonl(path))
    split_results = _build_split_results(samples)
    if args.dry_run:
        log.info("[dry-run] would validate %d samples from %s", len(samples), dataset_dir)
        return 0
    # Resolve relative image URIs against the benchmark's source root when a
    # run config is available (redistributable fixtures store relative paths).
    image_base_dirs: list[Path] = []
    if run_cfg is not None:
        try:
            image_base_dirs.append(_data_config_for(dataset, run_cfg).require_root())
        except ConfigError:
            pass
    report = validate_dataset(
        samples,
        qa_rows=qa_rows,
        split_results=split_results,
        image_base_dirs=image_base_dirs,
        strict=args.strict,
    )
    _print_json(report)
    return 0 if report["ok"] else 1


def cmd_card_render(args) -> int:
    run_cfg: RunConfig | None = None
    if args.config:
        run_cfg = load_run_config(args.config)
    base = Path(args.output_dir) if args.output_dir else (
        _default_build_dir(run_cfg) if run_cfg else Path("data/processed")
    )
    dataset = args.dataset
    dataset_dir = base / dataset
    if not dataset_dir.exists():
        dataset_dir = base / _benchmark_of(dataset)
    samples = _load_processed_samples(dataset_dir, _benchmark_of(dataset))
    if args.limit:
        samples = samples[: args.limit]
    benchmark = _benchmark_of(dataset)
    counts = {
        "samples": len(samples),
        "identities": len({s.identity_id for s in samples}),
        "observations": sum(len(s.visual_attributes) for s in samples),
        "accepted_labels": sum(
            1 for s in samples for obs in s.visual_attributes.values() if obs.label is not None
        ),
        "profile_facts": sum(len(s.profile_facts) for s in samples),
    }
    if args.dry_run:
        log.info("[dry-run] card counts: %s", counts)
        return 0
    from .build.export import ExtensionExporter

    exporter = ExtensionExporter(base, benchmark)
    card_text = exporter.render_dataset_card(counts)
    card_path = dataset_dir / f"{dataset}_extension_card.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card_text)
    print(card_path)
    return 0


# --------------------------------------------------------------------------- #
# Parser wiring
# --------------------------------------------------------------------------- #


def _common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="validate inputs without side effects")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of processed rows")
    parser.add_argument("--resume", action="store_true", help="reuse cached intermediate artifacts")
    parser.add_argument("--output-dir", default=None, help="override the configured output directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="route-data",
        description="CelebA-40 evaluation and multimodal-unlearning dataset construction.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # model
    model = sub.add_parser("model", help="model backend utilities").add_subparsers(
        dest="model_command", required=True
    )
    p = model.add_parser("inspect", help="print resolved model configuration")
    p.add_argument("--config", required=True)
    _common_flags(p)
    p.set_defaults(func=cmd_model_inspect)
    p = model.add_parser("smoke-test", help="run a tiny generate+score sanity check")
    p.add_argument("--config", required=True)
    p.add_argument("--image", default=None, help="path to a real test image (Fix 1)")
    p.add_argument("--prompts", default=None, help="path to binary prompt registry YAML")
    _common_flags(p)
    p.set_defaults(func=cmd_model_smoke_test)

    # celeba
    celeba = sub.add_parser("celeba", help="CelebA raw-data and evaluation workflow").add_subparsers(
        dest="celeba_command", required=True
    )
    p = celeba.add_parser("validate-raw", help="validate a local CelebA root")
    p.add_argument("--config", required=True)
    _common_flags(p)
    p.set_defaults(func=cmd_celeba_validate_raw)
    p = celeba.add_parser("prepare", help="build wide/long CelebA manifests")
    p.add_argument("--config", required=True)
    _common_flags(p)
    p.set_defaults(func=cmd_celeba_prepare)
    p = celeba.add_parser("evaluate", help="run a CelebA-40 evaluation")
    p.add_argument("--config", required=True)
    _common_flags(p)
    p.set_defaults(func=cmd_celeba_evaluate)
    p = celeba.add_parser("report", help="(re)render metrics report for a run")
    p.add_argument("--run-id", required=True)
    _common_flags(p)
    p.set_defaults(func=cmd_celeba_report)
    p = celeba.add_parser("freeze-protocol", help="freeze calibration protocol from a run")
    p.add_argument("--run-id", required=True)
    _common_flags(p)
    p.set_defaults(func=cmd_celeba_freeze_protocol)

    # source
    source = sub.add_parser("source", help="benchmark source inspection").add_subparsers(
        dest="source_command", required=True
    )
    p = source.add_parser("inspect", help="inspect a benchmark's raw schema")
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--snapshot", default=None, help="override the inspection snapshot path")
    p.add_argument(
        "--check-schema",
        action="store_true",
        help="fail when the live schema drifts from the pinned snapshot",
    )
    _common_flags(p)
    p.set_defaults(func=cmd_source_inspect)

    # build
    build = sub.add_parser("build", help="extension construction pipeline").add_subparsers(
        dest="build_command", required=True
    )
    for name, func, help_text in (
        ("annotate", cmd_build_annotate, "score + annotate samples with CelebA-40 labels"),
        ("qa", cmd_build_qa, "generate versioned visual QA rows"),
        ("route-probes", cmd_build_route_probes, "generate route-conflict probes + pair manifest"),
        ("splits", cmd_build_splits, "build forget/retain splits and invariants"),
        ("export", cmd_build_export, "export the full auditable extension"),
    ):
        p = build.add_parser(name, help=help_text)
        p.add_argument("--dataset", required=True)
        p.add_argument("--config", required=True)
        _common_flags(p)
        p.set_defaults(func=func)

    # validate
    validate = sub.add_parser("validate", help="dataset validation").add_subparsers(
        dest="validate_command", required=True
    )
    p = validate.add_parser("dataset", help="run all plan-19 checks on a built extension")
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--strict", action="store_true")
    _common_flags(p)
    p.set_defaults(func=cmd_validate_dataset)

    # card
    card = sub.add_parser("card", help="dataset card tooling").add_subparsers(
        dest="card_command", required=True
    )
    p = card.add_parser("render", help="render the dataset card markdown")
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=None)
    _common_flags(p)
    p.set_defaults(func=cmd_card_render)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
