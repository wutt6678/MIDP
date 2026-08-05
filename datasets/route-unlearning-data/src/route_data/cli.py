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


def _dataset_dir(args, run_cfg: RunConfig, dataset: str) -> Path:
    base = Path(args.output_dir) if args.output_dir else _default_build_dir(run_cfg)
    return base / dataset


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


def _load_image(uri: str | None, base: Path | None = None):
    """Best-effort PIL image load; blank placeholder when unavailable.

    Relative URIs are resolved against ``base`` (typically the source
    dataset root) so redistributable fixtures can store relative paths.
    """
    from PIL import Image

    if uri:
        try:
            path = Path(uri.removeprefix("file://"))
            if not path.is_absolute() and base is not None:
                path = base / path
            if path.is_file():
                with Image.open(path) as im:
                    return im.convert("RGB")
        except Exception as exc:  # noqa: BLE001 - fall back to placeholder
            log.warning("Could not open image %s (%s); using blank placeholder", uri, exc)
    return Image.new("RGB", (224, 224), (127, 127, 127))


def _p_positive(response) -> float | None:
    """Collapse candidate log-probabilities into P(' yes')."""
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
            return float(probs[candidate])
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
    attrs = sorted({k[len(prefix):] for s in samples for k in s.visual_attributes if k.startswith(prefix)})
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


def cmd_model_smoke_test(args) -> int:
    cfg = load_model_config(args.config)
    if args.dry_run:
        log.info("[dry-run] model config %s is valid; backend=%s", args.config, cfg.backend)
        return 0
    from .models.registry import create_backend

    backend = create_backend(cfg)
    image = _load_image(None)
    prompt = "Is there a person in this image? Answer yes or no."
    gen = backend.generate(image, prompt)
    scored = backend.score_candidates(image, prompt, [" yes", " no"])
    p = _p_positive(scored)
    _print_json(
        {
            "fingerprint": backend.fingerprint(),
            "generated_text": gen.text,
            "candidate_log_probs": {
                cs.candidate: cs.log_probability for cs in (scored.candidate_scores or [])
            },
            "p_positive": p,
        }
    )
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


def cmd_source_inspect(args) -> int:
    cfg = load_data_config(args.config)
    from .data.adapters.base import AdapterError, available_adapters, create_adapter

    if args.dataset not in available_adapters():
        raise ConfigError(
            f"Unknown dataset '{args.dataset}'; available adapters: {available_adapters()}"
        )
    adapter = create_adapter(cfg)
    report: dict[str, Any] = {
        "dataset": args.dataset,
        "adapter": adapter.name,
        "source_version": cfg.source_version,
        "field_map": adapter.field_map,
    }
    if args.dry_run:
        report["dry_run"] = "configuration and adapter constructed; no rows read"
        _print_json(report)
        return 0

    limit = args.limit or 20
    keys: set[str] = set()
    n_rows = 0
    first_row: dict[str, Any] | None = None
    mapping_error: str | None = None
    for row in itertools.islice(adapter.iter_rows(), limit):
        n_rows += 1
        keys.update(row.keys())
        if first_row is None:
            first_row = row
    report["rows_inspected"] = n_rows
    report["row_keys"] = sorted(keys)
    report["first_row"] = first_row
    if first_row is not None:
        try:
            sample = adapter.to_sample(first_row)
            report["mapped_sample_id"] = sample.source_sample_id
            report["mapped_identity_id"] = sample.identity_id
        except AdapterError as exc:
            mapping_error = str(exc)
    report["mapping_error"] = mapping_error
    _print_json(report)
    return 0 if mapping_error is None else 1


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
    annotated_path = dataset_dir / f"{args.dataset}_annotated.jsonl"

    if args.dry_run:
        log.info(
            "[dry-run] annotate %s: %d samples loaded; scores=%s annotated=%s",
            args.dataset,
            len(samples),
            scores_path,
            annotated_path,
        )
        return 0

    from .build.annotate import (
        AnnotationPolicy,
        BenchmarkAnnotator,
        load_frozen_calibrators,
        predictions_to_scores,
    )
    from .constants.celeba_attributes import CELEBA_ATTRIBUTES

    # 1) Model scores (resumable via the scores JSONL cache).
    done_keys: set[tuple[str, str]] = set()
    score_rows: list[dict[str, Any]] = []
    if scores_path.exists() and args.resume:
        score_rows = list(read_jsonl(scores_path))
        done_keys = {(r["sample_id"], r["attribute"]) for r in score_rows}
    sample_ids = {s.source_sample_id for s in samples}
    score_rows = [r for r in score_rows if r["sample_id"] in sample_ids]
    pending = [
        (s, attr)
        for s in samples
        for attr in CELEBA_ATTRIBUTES
        if (s.source_sample_id, attr) not in done_keys
    ]
    if pending:
        from .models.registry import create_backend
        from .prompts.registry import PromptRegistry

        backend = create_backend(run_cfg.model)
        registry = PromptRegistry(run_cfg.prompts)
        image_base = Path(data_cfg.root or data_cfg.extras.get("local_root") or ".")
        log.info("Scoring %d (image, attribute) queries with backend=%s", len(pending), run_cfg.model.backend)
        for sample, attr in pending:
            image = _load_image(sample.image_uri, base=image_base)
            resp = backend.score_candidates(
                image, registry.binary_prompt(attr), [" yes", " no"]
            )
            p = _p_positive(resp)
            if p is None:
                continue
            score_rows.append(
                {
                    "sample_id": sample.source_sample_id,
                    "attribute": attr,
                    "p_positive": p,
                    "raw_text": resp.text,
                }
            )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(score_rows, scores_path)

    # 2) Frozen-protocol annotation.
    policy = AnnotationPolicy(
        gated_attributes=frozenset(CELEBA_ATTRIBUTES),
        bands=dict(run_cfg.build.confidence_bands) or None,
        min_auto_accept_score=run_cfg.build.min_auto_accept_score,
    )
    fingerprint_id: str | None = None
    if scores_path.exists():
        try:
            from .models.registry import create_backend

            fingerprint_id = str(create_backend(run_cfg.model).fingerprint().get("fingerprint_id"))
        except Exception as exc:  # noqa: BLE001 - fingerprint is best-effort here
            log.warning("Backend fingerprint unavailable (%s); continuing without it", exc)
    from .prompts.registry import PromptRegistry

    registry = PromptRegistry(run_cfg.prompts)
    annotator = BenchmarkAnnotator(
        policy,
        calibrators=load_frozen_calibrators(run_cfg.frozen_protocol.calibrators),
        model_fingerprint=fingerprint_id,
        prompt_registry_hash=registry.registry_hash(),
    )
    scores_by_sample = predictions_to_scores(score_rows)
    annotated = [
        annotator.annotate_sample(s, scores_by_sample.get(s.source_sample_id, {}))
        for s in samples
    ]
    write_jsonl([s.to_dict() for s in annotated], annotated_path)
    n_obs = sum(len(s.visual_attributes) for s in annotated)
    n_labeled = sum(
        1 for s in annotated for obs in s.visual_attributes.values() if obs.label is not None
    )
    _print_json(
        {
            "dataset": args.dataset,
            "samples": len(annotated),
            "observations": n_obs,
            "accepted_labels": n_labeled,
            "annotated_path": str(annotated_path),
        }
    )
    return 0


def cmd_build_qa(args) -> int:
    run_cfg = load_run_config(args.config)
    dataset_dir = _dataset_dir(args, run_cfg, args.dataset)
    samples = _load_samples(dataset_dir, args.dataset)
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
    train_rows = generate_qa_rows(samples, registry, split="train")
    eval_rows = generate_qa_rows(samples, registry, split="validation")
    train_path = dataset_dir / f"{args.dataset}_visual_qa_train.jsonl"
    eval_path = dataset_dir / f"{args.dataset}_visual_qa_eval.jsonl"
    write_jsonl(train_rows, train_path)
    write_jsonl(eval_rows, eval_path)
    _print_json(
        {
            "dataset": args.dataset,
            "registry_hash": registry.registry_hash(),
            "registry_version": registry.version,
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
    samples = _load_samples(dataset_dir, args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    if args.dry_run:
        log.info("[dry-run] route probes for %d samples of %s", len(samples), args.dataset)
        return 0

    from .build.conflict_generation import (
        ConflictError,
        RouteProbeBuilder,
        build_identity_probes,
        build_pair_manifest,
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
        wrong_names = [
            (by_identity[other][0].identity_name or other)
            for other in identity_ids
            if other != identity_id and by_identity[other]
        ]
        try:
            probes = build_identity_probes(
                group, builder, wrong_identity_name=wrong_names[0] if wrong_names else None
            )
        except ConflictError as exc:
            skipped.append(f"{identity_id}: {exc}")
            continue
        for probe in probes:
            probe_rows.append(builder.probe_row(probe))
        if len(group) >= 2:
            for left, right in zip(group, group[1:]):
                pair_specs.append(
                    {
                        "pair_type": "cross_image_attribute_state",
                        "left_sample_id": left.source_sample_id,
                        "right_sample_id": right.source_sample_id,
                    }
                )
    pairs = build_pair_manifest(pair_specs)
    probes_path = dataset_dir / f"{args.dataset}_route_probes.jsonl"
    write_jsonl(probe_rows, probes_path)
    write_json(pairs, dataset_dir / f"{args.dataset}_pair_manifest.json")
    _print_json(
        {
            "dataset": args.dataset,
            "probe_rows": len(probe_rows),
            "pairs": len(pairs),
            "skipped_identities": skipped,
            "probes_path": str(probes_path),
        }
    )
    return 0


def cmd_build_splits(args) -> int:
    run_cfg = load_run_config(args.config)
    dataset_dir = _dataset_dir(args, run_cfg, args.dataset)
    samples = _load_samples(dataset_dir, args.dataset)
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
    samples = _load_samples(dataset_dir, args.dataset)
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

    data_cfg = _data_config_for(args.dataset, run_cfg)
    exporter = ExtensionExporter(
        Path(args.output_dir) if args.output_dir else _default_build_dir(run_cfg),
        benchmark,
        source_version=data_cfg.source_version,
        registry_hash=registry_hash,
    )
    record = exporter.export_all(
        samples,
        train_qa=train_qa,
        eval_qa=eval_qa,
        probe_rows=probe_rows,
        split_results=split_results,
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
    dataset_dir = base / dataset
    if not dataset_dir.exists():
        dataset_dir = base / benchmark
    # Intermediate artifacts are named after the benchmark (``fairget``),
    # not the export name (``fairget_celeba40``).
    samples = _load_samples(dataset_dir, benchmark)
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
    samples = _load_samples(dataset_dir, _benchmark_of(dataset))
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
