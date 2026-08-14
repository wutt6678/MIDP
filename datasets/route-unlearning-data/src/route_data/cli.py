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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
    return dataset.removesuffix(CARD_SUFFIX)


def _runtime_environment() -> dict[str, Any]:
    """R17: capture the full runtime environment for auditability.

    Records Python, CUDA runtime, GPU model, driver, torch, transformers,
    accelerate, huggingface_hub, Pillow and platform.  Every probe is
    defensive: unavailable values are ``None`` rather than fatal.
    """
    import platform as _platform
    import sys as _sys

    env: dict[str, Any] = {
        "python": _sys.version.split()[0],
        "platform": _platform.platform(),
        "cuda_runtime": None,
        "gpu_model": None,
        "driver": None,
        "torch": None,
        "transformers": None,
        "accelerate": None,
        "huggingface_hub": None,
        "pillow": None,
    }
    try:
        import torch as _torch

        env["torch"] = _torch.__version__
        if _torch.cuda.is_available():
            env["cuda_runtime"] = _torch.version.cuda
            env["gpu_model"] = _torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import subprocess as _sp

        _drv = _sp.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=_sp.DEVNULL,
        ).decode().strip().splitlines()
        if _drv:
            env["driver"] = _drv[0].strip()
    except Exception:
        pass
    for key, mod in (
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("huggingface_hub", "huggingface_hub"),
        ("pillow", "PIL"),
    ):
        try:
            env[key] = __import__(mod).__version__
        except Exception:
            pass
    return env


def _default_build_dir(run_cfg: RunConfig) -> Path:
    return Path(run_cfg.build.output_dir)

# R7: delegate to the single shared sanitizer so CLI, tests, and final_verify
# always agree on the model-output directory name.
from .naming import model_output_name as _model_output_name


def _dataset_dir(args, run_cfg: RunConfig, dataset: str) -> Path:
    base = Path(args.output_dir) if args.output_dir else _default_build_dir(run_cfg)
    model_dir = _model_output_name(run_cfg.model.model_id)
    return base / model_dir / dataset


def _data_config_for(dataset: str, run_cfg: RunConfig):
    """Resolve ``configs/data/<dataset>.yaml`` relative to the run config.

    When the run config's data section sets ``data_config_path``, that
    explicit path is used instead of the default resolution.  This lets
    golden CI fixtures use a protocol-free data config (P0-3 fix).
    """
    # Explicit override from run config.
    override = getattr(run_cfg.data, "data_config_path", None)
    if override:
        override_p = Path(override)
        if not override_p.is_absolute():
            # Resolve relative to the project root (package parent).
            base = Path(__file__).resolve().parents[2]
            override_p = (base / override_p).resolve()
        if override_p.exists():
            return load_data_config(override_p)
        raise ConfigError(f"data_config_path not found: {override_p}")
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


def _get_source_sample_id(sample) -> str | None:
    """Extract source_sample_id from a dict or CanonicalSample."""
    if isinstance(sample, dict):
        return sample.get("source_sample_id")
    return getattr(sample, "source_sample_id", None)


def _compute_source_hashes(data_cfg) -> dict[str, Any]:
    """P1-4: compute actual source hashes for manifest recording/verification.

    Returns a dict with:
      - git_commit_sha: actual Git SHA from the source root (or error string)
      - files: {rel_path: sha256} for each file in immutable_revision.files
    """
    import hashlib as _hashlib
    import subprocess as _sp

    result: dict[str, Any] = {}
    try:
        source_root = data_cfg.require_root()
    except Exception:
        return result

    # Git commit SHA.
    try:
        actual_git = _sp.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(source_root),
            stderr=_sp.DEVNULL,
        ).decode().strip()
        result["git_commit_sha"] = actual_git
    except Exception:
        pass

    # File hashes from immutable_revision.files.
    immutable = data_cfg.extras.get("immutable_revision")
    if immutable and isinstance(immutable, dict):
        files_block = immutable.get("files")
        if files_block and isinstance(files_block, dict):
            file_hashes: dict[str, str] = {}
            for rel_path in files_block:
                fpath = source_root / rel_path
                if fpath.exists():
                    try:
                        file_hashes[rel_path] = _hashlib.sha256(
                            fpath.read_bytes()
                        ).hexdigest()
                    except Exception:
                        pass
            if file_hashes:
                result["files"] = file_hashes
    return result


def _filter_by_smoke_manifest(samples, args) -> list:
    """P1-5: filter samples by --smoke-manifest allowlist.

    Works with both raw dicts (annotation stage) and CanonicalSample objects
    (downstream stages).  Fails closed on empty manifests (P1-3) and unknown
    sample IDs (P1-4).  Also verifies protocol SHA (P1-3) and source hashes
    (P1-4) against the current config.
    """
    smoke_manifest_path = getattr(args, "smoke_manifest", None)
    if not smoke_manifest_path:
        return samples

    manifest_p = Path(smoke_manifest_path)
    # P0-6 (review 54c0dc9): resolve relative paths against CWD (not config dir).
    # Path() already resolves relative to CWD — do NOT re-parent to config dir.
    if not manifest_p.exists():
        raise ConfigError(f"Smoke manifest not found: {manifest_p}")

    import json as _json_sm
    sm_data = _json_sm.loads(manifest_p.read_text())
    allowed_ids = set(sm_data.get("selected_source_sample_ids", []))

    # P1-3: fail closed on empty or malformed manifests.
    if not allowed_ids:
        raise ConfigError(
            f"Smoke manifest has empty selected_source_sample_ids: {manifest_p}. "
            "Refusing to fall back to full-data annotation."
        )

    # P0-7 (review 54c0dc9): verify protocol SHA — fail closed when manifest
    # has protocol_sha256 but current config has no protocol block.
    config_path = getattr(args, "config", None)
    manifest_proto_sha = sm_data.get("protocol_sha256")
    if manifest_proto_sha:
        if not config_path:
            raise ConfigError(
                "P0-7: manifest is bound to a protocol_sha256 but no --config "
                "was supplied to verify the current protocol."
            )
        cur_cfg = _data_config_for(args.dataset, load_run_config(config_path))
        cur_proto = cur_cfg.extras.get("fiubench_protocol")
        if not cur_proto or not isinstance(cur_proto, dict):
            raise ConfigError(
                "P0-7: Manifest is bound to a FIUBench protocol but the current "
                "benchmark config has no fiubench_protocol. Regenerate the "
                "manifest or restore the protocol block."
            )
        from .data.split_mapping import compute_protocol_sha256
        cur_sha, _ = compute_protocol_sha256(cur_proto)
        if cur_sha != manifest_proto_sha:
            raise ConfigError(
                f"P1-3: protocol SHA mismatch: manifest={manifest_proto_sha}, "
                f"current config={cur_sha}. The smoke manifest was created under "
                "a different protocol; regenerate it."
            )

    # P0-8: verify source hashes against current actual source (fail-closed).
    manifest_src = sm_data.get("source_hashes")
    if manifest_src and config_path:
        try:
            cur_cfg = _data_config_for(args.dataset, load_run_config(config_path))
            actual = _compute_source_hashes(cur_cfg)
            # Git SHA check — P0-8: fail if manifest has expected but actual is missing.
            m_git = manifest_src.get("git_commit_sha")
            a_git = actual.get("git_commit_sha")
            if m_git:
                if not a_git:
                    raise ConfigError(
                        "P0-8: manifest expects git_commit_sha but actual SHA "
                        "could not be computed; refusing to pass silently."
                    )
                if m_git != a_git:
                    raise ConfigError(
                        f"P1-4: source Git SHA mismatch: manifest={m_git}, actual={a_git}"
                    )
            # File hash check — P0-8: fail if manifest has expected but actual is missing.
            m_files = manifest_src.get("files", {})
            a_files = actual.get("files", {})
            for _fp, _mh in m_files.items():
                _ah = a_files.get(_fp)
                if not _ah:
                    raise ConfigError(
                        f"P0-8: manifest expects hash for '{_fp}' but actual hash "
                        "could not be computed; refusing to pass silently."
                    )
                if _ah != _mh:
                    raise ConfigError(
                        f"P1-4: source file SHA mismatch for '{_fp}': "
                        f"manifest={_mh}, actual={_ah}"
                    )
        except ConfigError:
            raise
        except Exception as exc:
            log.warning("P1-4: could not verify source hashes: %s", exc)

    # P1-4: fail on unknown sample IDs (do not silently drop).
    actual_ids = {_get_source_sample_id(s) for s in samples}
    unknown = allowed_ids - actual_ids
    if unknown:
        raise ConfigError(
            f"Smoke manifest references {len(unknown)} unknown sample IDs: "
            f"{sorted(unknown)[:10]}{'...' if len(unknown) > 10 else ''}"
        )

    before = len(samples)
    filtered = [s for s in samples if _get_source_sample_id(s) in allowed_ids]
    log.info(
        "Smoke manifest filter: %d/%d samples retained (%s)",
        len(filtered), before, manifest_p.name,
    )
    return filtered


def _filter_out_of_protocol(samples) -> list:
    """P0-3: remove out_of_protocol identities from downstream processing.

    Official FIUBench identities not selected by the configured experiment
    must not enter QA generation, route probes, splits, or export.
    Works with both CanonicalSample objects and raw dicts.
    """
    result = []
    for s in samples:
        split = s.split if hasattr(s, "split") else s.get("split", "")
        if split == "out_of_protocol":
            continue
        result.append(s)
    return result


# P0-4: positive allowlist for smoke selection.
ALLOWED_SMOKE_ROLES = frozenset({"train", "eval", "exclude"})


def _assert_protocol_no_unassigned_hash(dataset: str, run_cfg: dict, samples) -> None:
    """P0-3: when FIUBench protocol is active, reject unassigned/hash samples.

    Raises ConfigError if any sample has split == 'unassigned' or 'hash'
    while the protocol is active.  This is a defensive invariant — the
    adapter should never emit such values in protocol mode.
    """
    data_cfg = _data_config_for(dataset, run_cfg)
    proto = data_cfg.extras.get("fiubench_protocol")
    if not proto or not isinstance(proto, dict):
        return
    bad = []
    for s in samples:
        split = s.split if hasattr(s, "split") else s.get("split", "")
        if split in ("unassigned", "hash"):
            bad.append(split)
    if bad:
        from collections import Counter
        counts = Counter(bad)
        raise ConfigError(
            f"P0-3: FIUBench protocol is active but {len(bad)} sample(s) have "
            f"forbidden split values {dict(counts)}.  "
            "All identities must resolve to train/eval/exclude/out_of_protocol."
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
    for candidate in ("Yes", "yes"):
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

    # P2-19: collect image paths from --image (repeatable) and --image-list.
    image_paths: list[str] = []
    raw_images = getattr(args, "image", None) or []
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    image_paths.extend(raw_images)
    image_list_file = getattr(args, "image_list", None)
    if image_list_file:
        with open(image_list_file) as fh:
            for line in fh:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    image_paths.append(stripped)

    # P0-2: real backends require at least one real image; only stub may use
    # placeholder.
    if not image_paths and cfg.backend != "stub":
        raise ConfigError(
            "--image is required for non-stub smoke tests "
            f"(backend={cfg.backend})"
        )
    if not image_paths:
        image_paths = [None]  # stub placeholder

    # P0-1: construct PromptRegistry with a proper PromptsConfig, not a raw
    # string path.  The constructor expects a PromptsConfig dataclass.
    prompts_path = getattr(args, "prompts", None) or "configs/prompts/celeba_binary_v1.yaml"
    from .config import PromptsConfig
    from .prompts.registry import PromptRegistry

    prompt_cfg = PromptsConfig(binary=prompts_path, grouped=None, route_conflict=None)
    registry = PromptRegistry(prompt_cfg)

    # Three obvious attribute questions to verify visual discrimination.
    smoke_attributes = ["Eyeglasses", "Smiling", "Wearing_Hat"]

    # P2-19: per-image results; the original single-image layout is preserved
    # inside ``per_image`` so downstream consumers can still find it.
    import math as _smoke_math

    per_image: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    for image_path in image_paths:
        if image_path is not None:
            image = _load_image(image_path)
            image_hash = _image_sha256(image_path)
        else:
            image = _load_image(None)
            image_hash = "placeholder_gray_image"

        results: list[dict[str, Any]] = []
        for attr in smoke_attributes:
            prompt = registry.binary_prompt(attr)
            gen = backend.generate(image, prompt)
            scored = backend.score_candidates(image, prompt, ["Yes", "No"])
            p = _p_positive(scored)
            logps = {
                cs.candidate: cs.log_probability for cs in (scored.candidate_scores or [])
            }
            log_p_yes = logps.get("Yes")
            log_p_no = logps.get("No")
            margin = (
                log_p_yes - log_p_no
                if (
                    isinstance(log_p_yes, (int, float))
                    and isinstance(log_p_no, (int, float))
                )
                else None
            )
            predicted_label = (
                ("positive" if p >= 0.5 else "negative")
                if isinstance(p, (int, float))
                else None
            )
            results.append({
                "attribute": attr,
                "prompt": prompt,
                "generated_text": gen.text,
                "log_p_yes": log_p_yes,
                "log_p_no": log_p_no,
                "margin": margin,
                "p_positive": p,
                "predicted_label": predicted_label,
            })

        per_image.append({
            "image_path": image_path or "placeholder",
            "image_sha256": image_hash,
            "smoke_results": results,
        })
        all_results.extend(results)

    # R2: free-generation collapse checks belong in the smoke-test (not in
    # build annotate, where candidate scoring intentionally returns blank text).
    smoke_failures: list[str] = []
    gens = [(r.get("generated_text") or "").strip() for r in all_results]
    if gens and all(not g for g in gens):
        smoke_failures.append("all generated outputs blank")
    if any(r["log_p_yes"] is None or r["log_p_no"] is None for r in all_results):
        smoke_failures.append("candidate score missing")
    numeric_vals = [
        v
        for r in all_results
        for v in (r["log_p_yes"], r["log_p_no"], r["p_positive"])
        if isinstance(v, (int, float))
    ]
    if any(not _smoke_math.isfinite(v) for v in numeric_vals):
        smoke_failures.append("NaN/Inf in candidate scores")
    p_vals = [
        r["p_positive"]
        for r in all_results
        if isinstance(r["p_positive"], (int, float))
        and _smoke_math.isfinite(r["p_positive"])
    ]
    if p_vals and all(abs(p - 0.5) < 0.01 for p in p_vals):
        smoke_failures.append("all probabilities near 0.5")
    margins = [
        r["margin"]
        for r in all_results
        if isinstance(r["margin"], (int, float)) and _smoke_math.isfinite(r["margin"])
    ]
    if len(margins) > 1 and len({round(m, 6) for m in margins}) <= 1:
        smoke_failures.append("all margins identical")

    # P2-19: cross-image variation check.  When multiple visually diverse
    # images are provided, the same-attribute scores should not be
    # effectively identical across every image.
    cross_image_variation: dict[str, Any] = {}
    if len(per_image) > 1:
        for attr in smoke_attributes:
            attr_p = [
                r["p_positive"]
                for entry in per_image
                for r in entry["smoke_results"]
                if r["attribute"] == attr
                and isinstance(r["p_positive"], (int, float))
            ]
            if len(attr_p) >= 2:
                spread = max(attr_p) - min(attr_p)
                cross_image_variation[attr] = {
                    "min": min(attr_p),
                    "max": max(attr_p),
                    "spread": spread,
                }
                if spread < 1e-6:
                    smoke_failures.append(
                        f"attribute '{attr}' has identical scores across all "
                        f"{len(attr_p)} images (spread={spread:.2e})"
                    )

    # P2-20: image-conditioned sanity cases.  When --smoke-expected is
    # provided, compare model predictions against known labels for specific
    # images to verify the visual pathway is active.
    sanity_check: dict[str, Any] = {}
    expected_file = getattr(args, "smoke_expected", None)
    if expected_file:
        from .data.io import read_json as _read_json

        expected_entries = _read_json(expected_file)
        if not isinstance(expected_entries, list):
            expected_entries = [expected_entries]

        # Build a lookup from image path to expected labels
        expected_by_image: dict[str, dict[str, bool]] = {}
        for entry in expected_entries:
            if isinstance(entry, dict) and "image" in entry and "expected" in entry:
                expected_by_image[entry["image"]] = entry["expected"]

        sanity_results: list[dict[str, Any]] = []
        for entry in per_image:
            img_path = entry["image_path"]
            if img_path in expected_by_image:
                expected = expected_by_image[img_path]
                results_by_attr = {
                    r["attribute"]: r for r in entry["smoke_results"]
                }
                matches = 0
                mismatches = 0
                details: list[dict[str, Any]] = []
                for attr, expected_label in expected.items():
                    if attr in results_by_attr:
                        result = results_by_attr[attr]
                        predicted = result["predicted_label"]
                        # expected_label is True/False; predicted is "positive"/"negative"
                        expected_pred = "positive" if expected_label else "negative"
                        match = predicted == expected_pred
                        if match:
                            matches += 1
                        else:
                            mismatches += 1
                        details.append({
                            "attribute": attr,
                            "expected": expected_label,
                            "predicted": predicted,
                            "p_positive": result["p_positive"],
                            "match": match,
                        })
                if details:
                    sanity_results.append({
                        "image_path": img_path,
                        "matches": matches,
                        "mismatches": mismatches,
                        "details": details,
                    })

        if sanity_results:
            total_matches = sum(r["matches"] for r in sanity_results)
            total_mismatches = sum(r["mismatches"] for r in sanity_results)
            sanity_check = {
                "n_images_with_expectations": len(sanity_results),
                "total_matches": total_matches,
                "total_mismatches": total_mismatches,
                "results": sanity_results,
            }
            if total_mismatches > 0:
                log.warning(
                    "Sanity check: %d/%d expected labels mismatched",
                    total_mismatches,
                    total_matches + total_mismatches,
                )
            else:
                log.info(
                    "Sanity check: all %d expected labels matched",
                    total_matches,
                )

    if smoke_failures:
        raise ConfigError(
            "Model smoke-test collapse detected: " + "; ".join(smoke_failures)
        )

    fp = backend.fingerprint()
    resolved_revision = getattr(backend, "_resolved_revision", None) or fp.get("revision", "n/a")

    # Backward-compatible top-level fields use the first image.
    first = per_image[0] if per_image else {}
    payload = {
        "model_fingerprint": fp,
        "resolved_revision": resolved_revision,
        "image_sha256": first.get("image_sha256", "n/a"),
        "image_path": first.get("image_path", "placeholder"),
        "smoke_results": first.get("smoke_results", []),
        "checks": {
            "generated_non_blank": sum(1 for g in gens if g),
            "candidate_scores_present": sum(
                1 for r in all_results if r["log_p_yes"] is not None and r["log_p_no"] is not None
            ),
            "finite_scores": all(_smoke_math.isfinite(v) for v in numeric_vals),
            "probabilities_near_half": bool(p_vals) and all(abs(p - 0.5) < 0.01 for p in p_vals),
            "margins_distinct": len({round(m, 6) for m in margins}) > 1,
        },
        "n_images": len(per_image),
        "per_image": per_image,
        "cross_image_variation": cross_image_variation,
        "sanity_check": sanity_check,
    }

    # R2: persist a machine-readable smoke artifact, e.g.
    # outputs/smoke_test/qwen35_9b_model_smoke.json (name from config stem).
    smoke_name = Path(args.config).stem
    smoke_root = (
        Path(args.output_dir) if args.output_dir else (REPO_ROOT / "outputs")
    ) / "smoke_test"
    try:
        smoke_root.mkdir(parents=True, exist_ok=True)
        artifact_path = smoke_root / f"{smoke_name}_model_smoke.json"
        write_json(payload, artifact_path)
        log.info("Smoke-test artifact written: %s", artifact_path)
        payload["artifact_path"] = str(artifact_path)
    except Exception as exc:
        log.warning("Could not write smoke artifact: %s", exc)

    _print_json(payload)
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
        "images": len(wide),
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
        except Exception:
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
            total_queries=len(rows),
            latency_ms=rows["latency_ms"].dropna().tolist() or None,
        )
    macro = macro_average(per_attribute)
    from .data.io import ensure_parent_dir
    from .eval.reports import render_report_md, write_metrics_bundle

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
    # P1-9: FIUBench-specific counters.
    is_fiubench = args.dataset == "fiubench"
    fiu_subject_ids: set[str] = set()
    fiu_membership_counts: dict[str, int] = {}
    fiu_effective_split_counts: dict[str, int] = {}
    fiu_qa_original = 0
    fiu_qa_paraphrase = 0
    fiu_qa_perturbed = 0
    fiu_profile_fact_count = 0
    fiu_identities_with_images: set[str] = set()
    # P1-1: per-identity role tracking for protocol derivation report.
    fiu_identity_roles: dict[str, str] = {}  # identity_id → effective split
    fiu_subject_ids_by_role: dict[str, set[str]] = {}  # role → {source_subject_ids}
    # P1-10: duplicate / collision tracking.
    seen_source_sample_ids: dict[str, int] = {}
    seen_qa_variant_ids: dict[str, int] = {}
    identity_id_by_record_id: dict[str, str] = {}
    split_role_by_subject: dict[str, str] = {}
    collision_errors: list[str] = []
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
                fiu_identities_with_images.add(sample.identity_id)
            elif sample.modality != "text_only":
                warnings.append(
                    f"row {n_rows}: sample '{sample.source_sample_id}' has modality "
                    f"'{sample.modality}' but no image_uri"
                )
            # P1-10: duplicate source_sample_id check.
            prev = seen_source_sample_ids.get(sample.source_sample_id)
            if prev is not None:
                collision_errors.append(
                    f"duplicate source_sample_id '{sample.source_sample_id}' "
                    f"at rows {prev} and {n_rows}"
                )
            else:
                seen_source_sample_ids[sample.source_sample_id] = n_rows
            # P1-10: stable identity_id per source record.
            rec_id = sample.source_record_id
            if rec_id:
                prev_iid = identity_id_by_record_id.get(rec_id)
                if prev_iid and prev_iid != sample.identity_id:
                    collision_errors.append(
                        f"identity_id unstable for record '{rec_id}': "
                        f"'{prev_iid}' vs '{sample.identity_id}'"
                    )
                else:
                    identity_id_by_record_id[rec_id] = sample.identity_id
            # P1-9 / P1-10: FIUBench-specific tracking.
            if is_fiubench:
                smeta = sample.source_metadata or {}
                sid = smeta.get("source_subject_id")
                if sid:
                    fiu_subject_ids.add(str(sid))
                for b in smeta.get("official_memberships") or []:
                    fiu_membership_counts[str(b)] = (
                        fiu_membership_counts.get(str(b), 0) + 1
                    )
                eff = sample.split or "unassigned"
                fiu_effective_split_counts[eff] = (
                    fiu_effective_split_counts.get(eff, 0) + 1
                )
                # P1-1: track per-identity role and per-role subject IDs.
                iid = sample.identity_id
                if iid and iid not in fiu_identity_roles:
                    fiu_identity_roles[iid] = eff
                if sid:
                    fiu_subject_ids_by_role.setdefault(eff, set()).add(str(sid))
                # Conflicting split role: same subject ID mapped to
                # different effective splits via different rows.
                if sid:
                    prev_role = split_role_by_subject.get(str(sid))
                    if prev_role and prev_role != eff:
                        collision_errors.append(
                            f"conflicting split role for subject {sid}: "
                            f"'{prev_role}' vs '{eff}'"
                        )
                    else:
                        split_role_by_subject[str(sid)] = eff
                vt = smeta.get("variant_type", "original")
                if vt == "paraphrase":
                    fiu_qa_paraphrase += 1
                elif vt == "perturbed":
                    fiu_qa_perturbed += 1
                else:
                    fiu_qa_original += 1
                # P1-10: duplicate QA variant ID check.
                if sample.source_sample_id in seen_qa_variant_ids:
                    collision_errors.append(
                        f"duplicate QA variant ID '{sample.source_sample_id}' "
                        f"at rows {seen_qa_variant_ids[sample.source_sample_id]} "
                        f"and {n_rows}"
                    )
                else:
                    seen_qa_variant_ids[sample.source_sample_id] = n_rows
                fiu_profile_fact_count += len(sample.profile_facts or [])
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
    # P1-10: report collision / duplicate errors and fail the inspect.
    if collision_errors:
        report["collision_errors"] = collision_errors
        mapping_errors.extend(collision_errors)
    # P1-9: FIUBench-specific inspection summary.
    if is_fiubench:
        report["fiubench_inspection"] = {
            "unique_subject_ids": len(fiu_subject_ids),
            "unique_identity_ids": len(identities),
            "official_membership_counts": dict(sorted(fiu_membership_counts.items())),
            "effective_split_counts": dict(sorted(fiu_effective_split_counts.items())),
            "qa_variant_counts": {
                "original": fiu_qa_original,
                "paraphrase": fiu_qa_paraphrase,
                "perturbed": fiu_qa_perturbed,
            },
            "image_bearing_identities": len(fiu_identities_with_images),
            "missing_image_identities": max(0, len(fiu_subject_ids) - len(fiu_identities_with_images)),
            "profile_fact_count": fiu_profile_fact_count,
        }
        # P1-1: protocol derivation report.
        _exclude_ids = fiu_subject_ids_by_role.get("exclude", set())
        _train_ids = fiu_subject_ids_by_role.get("train", set())
        _eval_ids = fiu_subject_ids_by_role.get("eval", set())
        _oop_ids = fiu_subject_ids_by_role.get("out_of_protocol", set())
        _retain_pool = _train_ids | _eval_ids
        _disjointness_errors: list[str] = []
        if _exclude_ids & _train_ids:
            _disjointness_errors.append(
                f"forget ∩ train = {_exclude_ids & _train_ids}"
            )
        if _exclude_ids & _eval_ids:
            _disjointness_errors.append(
                f"forget ∩ eval = {_exclude_ids & _eval_ids}"
            )
        if _train_ids & _eval_ids:
            _disjointness_errors.append(
                f"train ∩ eval = {_train_ids & _eval_ids}"
            )
        _protocol_derivation = {
            "protocol_name": (cfg.extras.get("fiubench_protocol") or {}).get("name"),
            "source_rows": n_rows,
            "source_identities": len(fiu_subject_ids),
            "forget_bucket": (cfg.extras.get("fiubench_protocol") or {}).get("forget_bucket"),
            "forget_identities": len(_exclude_ids),
            "retain_pool": (cfg.extras.get("fiubench_protocol") or {}).get("train_bucket"),
            "retain_pool_identities": len(_retain_pool),
            "train_identities": len(_train_ids),
            "eval_identities": len(_eval_ids),
            "out_of_protocol_identities": len(_oop_ids),
            "disjointness_errors": _disjointness_errors,
        }
        report["protocol_derivation"] = _protocol_derivation
        if _disjointness_errors:
            mapping_errors.extend(
                f"protocol disjoint: {e}" for e in _disjointness_errors
            )
        # P1-2: protocol SHA-256 fingerprint.
        _proto_cfg = cfg.extras.get("fiubench_protocol")
        if _proto_cfg and isinstance(_proto_cfg, dict):
            from .data.split_mapping import compute_protocol_sha256
            _sha, _canonical = compute_protocol_sha256(_proto_cfg)
            report["protocol_sha256"] = _sha
            report["protocol_canonical"] = _canonical
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


def cmd_source_make_smoke_manifest(args) -> int:
    """P1-1: generate a coverage-aware smoke manifest before model annotation.

    Loads the source adapter, resolves splits, selects a minimal coverage-aware
    subset, and writes the manifest JSON.  Exits without loading any model.
    """
    import hashlib as _hashlib
    import json as _json

    data_cfg = _data_config_for(args.dataset, load_run_config(args.config))
    from .data.adapters.base import create_adapter
    from .data.split_mapping import load_source_mapping, resolve_effective_split

    adapter = create_adapter(data_cfg)
    raw_samples = list(adapter.load())
    if not raw_samples:
        raise ConfigError(f"No samples loaded from adapter for {args.dataset}")

    # Convert to dicts for selection logic.
    samples = [s.to_dict() if hasattr(s, "to_dict") else s for s in raw_samples]

    # P0-4: positive allowlist — only train/eval/exclude may enter the smoke.
    from .data.split_mapping import load_source_mapping as _sm_lm
    from .data.split_mapping import resolve_effective_split as _sm_res
    _sm_map = _sm_lm(data_cfg)
    _rejected_roles: dict[str, int] = {}
    _allowed_samples: list[dict] = []
    for s in samples:
        _eff = _sm_res(s, source_mapping=_sm_map)
        if _eff in ALLOWED_SMOKE_ROLES:
            _allowed_samples.append(s)
        else:
            _rejected_roles[_eff] = _rejected_roles.get(_eff, 0) + 1
    samples = _allowed_samples
    if _rejected_roles:
        log.info("P0-4: smoke roles rejected by allowlist: %s", _rejected_roles)
    if not samples:
        raise ConfigError(
            f"No samples with allowed roles {sorted(ALLOWED_SMOKE_ROLES)} for "
            f"{args.dataset}; rejected roles: {_rejected_roles}. "
            "Check the fiubench_protocol configuration."
        )

    # Resolve splits for each sample.
    source_mapping = load_source_mapping(data_cfg)
    for s in samples:
        s["_effective_split"] = resolve_effective_split(
            s, source_mapping=source_mapping,
        )

    # Greedy coverage-aware selection.
    min_identities = getattr(args, "min_identities", 3) or 3
    min_image_bearing = getattr(args, "min_image_bearing", 2) or 2
    require_multiview = getattr(args, "require_multiview", False)

    selected: list[dict] = []
    selected_ids: set[str] = set()
    identity_ids: set[str] = set()
    image_bearing: set[str] = set()
    splits_seen: set[str] = set()
    has_fact = False
    images_by_identity: dict[str, set[str]] = {}
    remaining = list(range(len(samples)))

    # P0-3 (review 54c0dc9): explicit coverage check — stop selecting once
    # all structural requirements are satisfied, rather than filling to 12.
    # P0-4 (review 600ea5b): includes structural wrong-name feasibility.
    from .build.conflict_generation import structural_wrong_name_candidates

    def _selected_by_identity() -> dict[str, list]:
        _by: dict[str, list] = {}
        for _s in selected:
            _iid = _s.get("identity_id", "")
            if _iid:
                _by.setdefault(_iid, []).append(_s)
        return _by

    def _coverage_satisfied() -> bool:
        if len(identity_ids) < min_identities:
            return False
        if len(image_bearing) < max(1, min_image_bearing):
            return False
        if "train" not in splits_seen:
            return False
        if "eval" not in splits_seen:
            return False
        if "exclude" not in splits_seen:
            return False
        if not has_fact:
            return False
        if require_multiview and not any(
            len(imgs) >= 2 for imgs in images_by_identity.values()
        ):
            return False
        # P0-4 (review 600ea5b): structural wrong-name feasibility.
        return structural_wrong_name_candidates(_selected_by_identity())

    def _score(idx: int) -> int:
        s = samples[idx]
        iid = s.get("identity_id", "")
        img = s.get("image_uri")
        split = s.get("_effective_split")
        facts = s.get("profile_facts", [])
        score = 0
        if iid not in identity_ids:
            score += 3
        if img and iid not in image_bearing:
            score += 2
        if split and split not in splits_seen:
            score += 2
        if facts and not has_fact:
            score += 1
        if require_multiview and iid in identity_ids and img and img not in images_by_identity.get(iid, set()):
            score += 5
        # P0-5 (review 600ea5b): reward completing a control group for
        # wrong-name structural feasibility.
        if iid in identity_ids and len(_selected_by_identity().get(iid, [])) == 1:
            before = len(structural_wrong_name_candidates(_selected_by_identity()))
            trial = _selected_by_identity()
            trial.setdefault(iid, []).append(s)
            after = len(structural_wrong_name_candidates(trial))
            if after > before:
                score += 8
        return score

    max_select = min(12, len(samples))
    while len(selected) < max_select and remaining:
        # P0-3: stop as soon as all structural coverage is satisfied.
        if _coverage_satisfied():
            break
        scored = [(idx, _score(idx)) for idx in remaining]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_idx = scored[0][0]
        if scored[0][1] <= 0 and len(selected) >= min_identities:
            break
        s = samples[best_idx]
        sid = s.get("source_sample_id", "")
        iid = s.get("identity_id", "")
        img = s.get("image_uri")
        split = s.get("_effective_split")
        facts = s.get("profile_facts", [])
        selected.append(s)
        selected_ids.add(sid)
        identity_ids.add(iid)
        if img:
            image_bearing.add(iid)
            images_by_identity.setdefault(iid, set()).add(img)
        if split:
            splits_seen.add(split)
        if facts:
            has_fact = True
        remaining.remove(best_idx)

    # P1-9: compute wrong-name availability using production eligibility logic.
    from .build.conflict_generation import find_wrong_name_candidates
    _sel_by_identity: dict[str, list] = _selected_by_identity()
    _wn_pairs = find_wrong_name_candidates(_sel_by_identity)

    # P0-4 (review 600ea5b): Gate A — structural wrong-name feasibility
    # using the shared structural helper (P0-6).
    _structural_wn_candidates = structural_wrong_name_candidates(_sel_by_identity)
    # P0-4 strict check is deferred to the strict-mode block below
    # where `strict` and `fatal_errors` are defined.

    # P1-4: verify protocol role coverage when a protocol is defined.
    proto = data_cfg.extras.get("fiubench_protocol")
    protocol_coverage: dict[str, Any] = {}
    if proto and isinstance(proto, dict):
        source_mapping = load_source_mapping(data_cfg)
        role_identities: dict[str, set[str]] = {}
        for _s in selected:
            _iid = _s.get("identity_id", "")
            _split = _s.get("_effective_split", "")
            if _iid and _split:
                role_identities.setdefault(_split, set()).add(_iid)
        # P0-5 (review 54c0dc9): required roles use direct role names,
        # not bucket-name-through-source_mapping indirection.
        required_roles: set[str] = set()
        if proto.get("forget_bucket"):
            required_roles.add("exclude")
        if proto.get("train_bucket"):
            required_roles.add("train")
        if proto.get("eval_bucket"):
            required_roles.add("eval")
        elif proto.get("eval_fraction", 0) > 0:
            # Holdout generates eval from train bucket.
            required_roles.add("eval")
        missing_roles = required_roles - set(role_identities.keys())
        protocol_coverage = {
            "role_identity_counts": {
                k: len(v) for k, v in sorted(role_identities.items())
            },
            "required_roles": sorted(required_roles),
            "missing_roles": sorted(missing_roles),
        }
        if missing_roles:
            warnings_list = [f"P1-4: no selected identity for role(s): {sorted(missing_roles)}"]
        else:
            warnings_list = []
    else:
        warnings_list = []

    # Build source revision info.
    source_revision: dict[str, Any] = {}
    if hasattr(data_cfg, "source_version") and data_cfg.source_version:
        sv = data_cfg.source_version
        if isinstance(sv, dict):
            source_revision = dict(sv)
        else:
            source_revision = {"source_version": str(sv)}

    # P1-3: record the experiment protocol for FIUBench (or any benchmark
    # that declares one in extras).
    source_protocol: dict[str, Any] | None = None
    proto = data_cfg.extras.get("fiubench_protocol")
    if proto and isinstance(proto, dict):
        source_protocol = dict(proto)
    # Also record immutable_revision pointers when available.
    ir = data_cfg.extras.get("immutable_revision")
    if ir and isinstance(ir, dict):
        source_revision.setdefault("immutable_revision", ir)

    # P1-6 / P1-7: validate selected images and record their hashes.
    selected_images: list[dict[str, str]] = []
    image_errors: list[str] = []
    for _s in selected:
        _img = _s.get("image_uri")
        if not _img:
            continue
        _sid = _s.get("source_sample_id", "")
        _p = Path(_img)
        if not _p.exists():
            image_errors.append(f"image not found for '{_sid}': {_img}")
            continue
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(_p) as _im:
                _im.verify()
        except Exception as exc:
            image_errors.append(f"image unreadable for '{_sid}': {_img} ({exc})")
            continue
        _ih = _hashlib.sha256(_p.read_bytes()).hexdigest()
        selected_images.append({
            "source_sample_id": _sid,
            "image_uri": _img,
            "sha256": _ih,
        })
    if image_errors:
        warnings_list.extend(image_errors)

    # ------------------------------------------------------------------ #
    # P0-11 / P0-12: strict-mode fail-closed validation.
    # ------------------------------------------------------------------ #
    strict = getattr(args, "strict", True)
    fatal_errors: list[str] = []
    if strict:
        # P0-4: structural wrong-name feasibility (shared helper).
        if not _structural_wn_candidates:
            fatal_errors.append(
                "P0-4: structural wrong-name feasibility requires at least "
                "one valid (target, control) pair with control having >= 2 "
                "selected rows"
            )
        # Missing required protocol roles.
        if protocol_coverage and protocol_coverage.get("missing_roles"):
            for _role in protocol_coverage["missing_roles"]:
                fatal_errors.append(
                    f"P0-11: missing required protocol role '{_role}'"
                )
        # Selected-image validation (P0-12).
        for _ie in image_errors:
            fatal_errors.append(f"P0-12: {_ie}")
        # Insufficient identities.
        if len(identity_ids) < min_identities:
            fatal_errors.append(
                f"P0-11: only {len(identity_ids)} identities selected, "
                f"need >= {min_identities}"
            )
        # No image-bearing identities.
        if len(image_bearing) < max(1, min_image_bearing):
            fatal_errors.append(
                f"P0-11: only {len(image_bearing)} image-bearing identities, "
                f"need >= {min_image_bearing}"
            )
        # No profile facts.
        if not has_fact:
            fatal_errors.append(
                "P0-11: no profile facts in selected samples"
            )
        # Duplicate source IDs (shouldn't happen, but guard anyway).
        _all_sids = [s.get("source_sample_id", "") for s in selected]
        if len(_all_sids) != len(set(_all_sids)):
            fatal_errors.append("P0-11: duplicate source_sample_id in selection")

    # ------------------------------------------------------------------ #
    # P2-12 / P2-13: audit status validation (strict mode).
    # ------------------------------------------------------------------ #
    pilot_mode = getattr(args, "pilot", False)
    audit_report: dict[str, Any] = {}
    if pilot_mode or strict:
        from .build.audit import validate_audit_statuses
        # Collect audit statuses from config extras if declared.
        _audit_cfg = data_cfg.extras.get("manual_audit")
        _audit_statuses: list[str] = []
        if _audit_cfg and isinstance(_audit_cfg, dict):
            _audit_statuses = list(_audit_cfg.get("statuses", []))
        if _audit_statuses:
            try:
                audit_report = validate_audit_statuses(
                    _audit_statuses, strict=False,
                )
                if strict and not audit_report.get("pilot_ready", False):
                    for _blk in audit_report.get("blocking", []):
                        fatal_errors.append(
                            f"P2-12: audit status '{_blk}' blocks the pilot; "
                            f"counts={audit_report['counts']}"
                        )
            except Exception as exc:
                fatal_errors.append(f"P2-13: audit status validation error: {exc}")
        elif pilot_mode:
            # Pilot mode requires explicit audit statuses.
            fatal_errors.append(
                "P2-12: pilot mode requires data.extras.manual_audit.statuses "
                "to be populated"
            )

    # ------------------------------------------------------------------ #
    # P2-14: route-family minimums (pilot mode).
    # ------------------------------------------------------------------ #
    route_family_report: dict[str, Any] = {}
    if pilot_mode:
        from .build.audit import (
            DEFAULT_MINIMUM_ROUTE_COUNTS,
            check_route_family_minimums,
        )
        # Count probes per family from the selected samples.  This is an
        # estimate based on the selected samples' structure; actual probe
        # counts depend on the downstream build.
        _pilot_cfg = data_cfg.extras.get("pilot_verification")
        _min_counts = DEFAULT_MINIMUM_ROUTE_COUNTS
        if _pilot_cfg and isinstance(_pilot_cfg, dict):
            _cfg_min = _pilot_cfg.get("minimum_route_counts")
            if _cfg_min and isinstance(_cfg_min, dict):
                _min_counts = {k: int(v) for k, v in _cfg_min.items()}
        # Estimate route-family coverage from selected samples.
        _family_counts: dict[str, int] = {}
        for _s in selected:
            _facts = _s.get("profile_facts", [])
            _img = _s.get("image_uri")
            if _img:
                _family_counts.setdefault("direct_visual", 0)
                _family_counts["direct_visual"] += 1
                _family_counts.setdefault("image_plus_name", 0)
                _family_counts["image_plus_name"] += 1
            if _facts:
                _family_counts.setdefault("name_only", 0)
                _family_counts["name_only"] += len(_facts)
        try:
            route_family_report = check_route_family_minimums(
                _family_counts, _min_counts, strict=False,
            )
            if not route_family_report.get("pilot_ready", False):
                for _fam, _info in route_family_report.get("shortfalls", {}).items():
                    fatal_errors.append(
                        f"P2-14: route family '{_fam}' has "
                        f"{_info['actual']}/{_info['required']} probes"
                    )
        except Exception as exc:
            fatal_errors.append(f"P2-14: route-family check error: {exc}")

    # ------------------------------------------------------------------ #
    # P2-15: polarity balance reporting.
    # ------------------------------------------------------------------ #
    polarity_report: dict[str, Any] = {}
    if pilot_mode:
        from .build.audit import report_polarity_balance
        # Collect per-attribute label states from selected samples.
        _attr_states: dict[str, list[bool]] = {}
        for _s in selected:
            _va = _s.get("visual_attributes", {})
            if isinstance(_va, dict):
                for _akey, _aval in _va.items():
                    _lbl = None
                    if isinstance(_aval, dict):
                        _lbl = _aval.get("label")
                    elif hasattr(_aval, "label"):
                        _lbl = _aval.label
                    if _lbl is not None:
                        _attr_states.setdefault(_akey, []).append(bool(_lbl))
        if _attr_states:
            polarity_report = report_polarity_balance(_attr_states)

    # ------------------------------------------------------------------ #
    # P2-16: benchmark provenance freeze (pilot mode).
    # ------------------------------------------------------------------ #
    provenance_report: dict[str, Any] = {}
    if pilot_mode:
        from .build.audit import check_benchmark_provenance_frozen
        _bench_cfg = data_cfg.extras.get("benchmark_provenance")
        if _bench_cfg and isinstance(_bench_cfg, dict):
            try:
                provenance_report = check_benchmark_provenance_frozen(
                    _bench_cfg, strict=False,
                )
                if not provenance_report.get("all_frozen", False):
                    for _bn, _bi in provenance_report.get(
                        "per_benchmark", {}
                    ).items():
                        if not _bi["frozen"]:
                            fatal_errors.append(
                                f"P2-16: benchmark '{_bn}' provenance not "
                                f"frozen: {'; '.join(_bi['issues'])}"
                            )
            except Exception as exc:
                fatal_errors.append(f"P2-16: provenance check error: {exc}")
        else:
            # Pilot mode with no benchmark_provenance configured — only
            # warn if the dataset is not FIUBench (which has its own
            # immutable_revision check).
            if args.dataset != "fiubench":
                fatal_errors.append(
                    "P2-16: pilot mode requires data.extras.benchmark_provenance "
                    "to declare frozen provenance for all benchmarks"
                )

    if fatal_errors:
        for _fe in fatal_errors:
            log.error("STRICT: %s", _fe)
        raise ConfigError(
            "Smoke manifest validation failed (strict mode):\n  "
            + "\n  ".join(fatal_errors)
        )

    # Compute manifest SHA256 (of the canonical content, before adding the hash).
    manifest_body: dict[str, Any] = {
        "dataset": args.dataset,
        "selection_version": "smoke_v3",
        "selected_source_sample_ids": sorted(selected_ids),
        "selected_identity_ids": sorted(identity_ids),
        "coverage": {
            "selected_samples": len(selected),
            "identities": sorted(identity_ids),
            "image_bearing_identities": sorted(image_bearing),
            "splits_seen": sorted(splits_seen),
            "has_profile_facts": has_fact,
            "has_multiview": any(len(imgs) >= 2 for imgs in images_by_identity.values()),
            "wrong_name_pairs": [
                {"target": t, "control": c, "similarity": round(sim, 4)}
                for t, c, sim in _wn_pairs[:5]
            ],
        },
        "selection_policy": {
            "min_identities": min_identities,
            "min_image_bearing": min_image_bearing,
            "require_multiview": require_multiview,
            "strict": strict,
        },
        "source_revision": source_revision,
    }
    if source_protocol is not None:
        manifest_body["source_protocol"] = source_protocol
        # P1-2: record protocol SHA-256 fingerprint in the smoke manifest.
        from .data.split_mapping import compute_protocol_sha256
        _psha, _pcanonical = compute_protocol_sha256(source_protocol)
        manifest_body["protocol_sha256"] = _psha
    # P1-4: record actual source hashes (Git SHA + file SHA-256s).
    _src_hashes = _compute_source_hashes(data_cfg)
    if _src_hashes:
        manifest_body["source_hashes"] = _src_hashes
    if protocol_coverage:
        manifest_body["protocol_coverage"] = protocol_coverage
    # P0-4: persist structural wrong-name feasibility candidates.
    if _structural_wn_candidates:
        manifest_body["wrong_name_structural_candidates"] = _structural_wn_candidates
    # P1-7: record selected image hashes.
    if selected_images:
        manifest_body["selected_images"] = selected_images
    if image_errors:
        manifest_body["image_errors"] = image_errors
    # P2-12/13: audit status report.
    if audit_report:
        manifest_body["audit_status"] = audit_report
    # P2-14: route-family coverage report.
    if route_family_report:
        manifest_body["route_family_coverage"] = route_family_report
    # P2-15: polarity balance report.
    if polarity_report:
        manifest_body["polarity_balance"] = polarity_report
    # P2-16: benchmark provenance freeze report.
    if provenance_report:
        manifest_body["benchmark_provenance"] = provenance_report
    canonical = _json.dumps(manifest_body, sort_keys=True, separators=(",", ":"))
    manifest_body["manifest_sha256"] = _hashlib.sha256(canonical.encode()).hexdigest()

    # Write output.
    output_path = Path(args.output) if args.output else None
    if not output_path:
        raise ConfigError("--output is required for make-smoke-manifest")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json.dumps(manifest_body, indent=2) + "\n")

    log.info(
        "Smoke manifest written: %d samples, %d identities, %d image-bearing (%s)",
        len(selected), len(identity_ids), len(image_bearing), output_path,
    )
    _print_json(manifest_body)
    return 0


def cmd_source_verify_revision(args) -> int:
    """P0-2/P0-13: verify configured immutable_revision against actual source.

    Supports two formats:
    - Path-bound (P0-13): ``immutable_revision.files`` maps relative paths
      to ``{sha256: ...}`` objects.  Each file is verified independently.
    - Positional (legacy): ``profile_file_sha256`` / ``split_file_sha256``
      mapped onto ``adapter.source_files()`` by index.

    Fails before any model is loaded.
    """
    import hashlib as _hashlib
    import subprocess as _sp

    data_cfg = _data_config_for(args.dataset, load_run_config(args.config))
    from .data.adapters.base import create_adapter

    adapter = create_adapter(data_cfg)
    immutable = data_cfg.extras.get("immutable_revision")
    if not immutable or not isinstance(immutable, dict):
        raise ConfigError(
            f"No immutable_revision configured for {args.dataset}; "
            "populate git_commit_sha and file hashes"
        )

    errors: list[str] = []
    verified: dict[str, Any] = {}

    # 1. Git commit SHA.
    source_root = data_cfg.require_root()
    configured_git = immutable.get("git_commit_sha", "")
    try:
        actual_git = _sp.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(source_root),
            stderr=_sp.DEVNULL,
        ).decode().strip()
    except Exception as exc:
        actual_git = f"ERROR: {exc}"
    verified["git_commit_sha"] = {
        "configured": configured_git,
        "actual": actual_git,
        "match": configured_git == actual_git,
    }
    if configured_git != actual_git:
        errors.append(f"git_commit_sha mismatch: configured={configured_git}, actual={actual_git}")

    # 2. File hashes.
    files_block = immutable.get("files")
    if files_block and isinstance(files_block, dict):
        # P0-13: path-bound verification.
        for rel_path, spec in files_block.items():
            if not isinstance(spec, dict):
                errors.append(f"files.{rel_path} must be a dict with 'sha256'")
                continue
            configured = spec.get("sha256", "")
            fpath = source_root / rel_path
            if not fpath.exists():
                actual = "N/A (file missing)"
            else:
                try:
                    actual = _hashlib.sha256(fpath.read_bytes()).hexdigest()
                except Exception as exc:
                    actual = f"ERROR: {exc}"
            verified[f"files.{rel_path}"] = {
                "configured": configured,
                "actual": actual,
                "file": str(fpath),
                "match": configured == actual,
            }
            if configured != actual:
                errors.append(
                    f"{rel_path} sha256 mismatch: "
                    f"configured={configured}, actual={actual}"
                )
    else:
        # Legacy positional mapping (profile_file_sha256, split_file_sha256).
        try:
            source_files = adapter.source_files()
        except Exception as exc:
            raise ConfigError(f"Cannot enumerate source files: {exc}")
        file_hash_keys = ["profile_file_sha256", "split_file_sha256"]
        for i, key in enumerate(file_hash_keys):
            configured = immutable.get(key, "")
            if i < len(source_files):
                fpath = source_files[i]
                try:
                    actual = _hashlib.sha256(fpath.read_bytes()).hexdigest()
                except Exception as exc:
                    actual = f"ERROR: {exc}"
            else:
                actual = "N/A (no file)"
            verified[key] = {
                "configured": configured,
                "actual": actual,
                "file": str(source_files[i]) if i < len(source_files) else None,
                "match": configured == actual,
            }
            if configured != actual:
                errors.append(f"{key} mismatch: configured={configured}, actual={actual}")

    result = {
        "dataset": args.dataset,
        "source_root": str(source_root),
        "verified": verified,
        "all_match": len(errors) == 0,
        "errors": errors,
    }
    _print_json(result)
    return 0 if not errors else 1


def cmd_source_protocol_report(args) -> int:
    """P0-9: generate fiubench_protocol_report.json with identity counts.

    Loads the source adapter, resolves every identity's protocol role,
    verifies disjointness, and writes a machine-readable report.
    """
    import json as _json_pr

    data_cfg = _data_config_for(args.dataset, load_run_config(args.config))
    from .data.adapters.base import create_adapter
    from .data.split_mapping import resolve_protocol_role

    adapter = create_adapter(data_cfg)
    proto_cfg = data_cfg.extras.get("fiubench_protocol")
    if not proto_cfg or not isinstance(proto_cfg, dict):
        raise ConfigError(
            f"No fiubench_protocol configured for {args.dataset}; "
            "cannot generate protocol report."
        )

    id_by_role: dict[str, set[str]] = {}
    for ctx, row in adapter.iter_rows_with_context():
        samples = list(adapter.to_samples(row, source_context=ctx))
        for s in samples:
            sm = s.source_metadata or {}
            sid = str(sm.get("source_subject_id", ""))
            memberships = sm.get("official_memberships") or []
            role = resolve_protocol_role(
                [str(m) for m in memberships], proto_cfg,
                source_subject_id=sid if sid else None,
            )
            if sid:
                id_by_role.setdefault(role, set()).add(sid)

    forget_ids = id_by_role.get("exclude", set())
    train_ids = id_by_role.get("train", set())
    eval_ids = id_by_role.get("eval", set())
    oop_ids = id_by_role.get("out_of_protocol", set())
    unassigned_ids = id_by_role.get("unassigned", set())
    hash_ids = id_by_role.get("hash", set())
    all_ids = forget_ids | train_ids | eval_ids | oop_ids | unassigned_ids | hash_ids

    # Disjointness verification.
    disjointness_ok = True
    for _a, _b, _label in [
        (forget_ids, train_ids, "forget/train"),
        (forget_ids, eval_ids, "forget/eval"),
        (train_ids, eval_ids, "train/eval"),
    ]:
        if _a & _b:
            disjointness_ok = False

    report = {
        "protocol_name": proto_cfg.get("name", ""),
        "source_population": proto_cfg.get("source_population", {}).get("mode", "unknown")
        if isinstance(proto_cfg.get("source_population"), dict)
        else str(proto_cfg.get("source_population", "unknown")),
        "source_identity_count": len(all_ids),
        "forget_bucket": proto_cfg.get("forget_bucket"),
        "forget_identity_count": len(forget_ids),
        "retain_pool": proto_cfg.get("train_bucket"),
        "retain_pool_identity_count": len(train_ids) + len(eval_ids),
        "train_identity_count": len(train_ids),
        "eval_identity_count": len(eval_ids),
        "out_of_protocol_identity_count": len(oop_ids),
        "unassigned_identity_count": len(unassigned_ids),
        "hash_identity_count": len(hash_ids),
        "disjointness_ok": disjointness_ok,
        "train_identity_ids": sorted(train_ids),
        "eval_identity_ids": sorted(eval_ids),
        "forget_identity_ids": sorted(forget_ids),
        "oop_identity_ids": sorted(oop_ids),
        "invariants": {
            "unassigned_is_zero": len(unassigned_ids) == 0,
            "hash_is_zero": len(hash_ids) == 0,
            "forget_nonempty": len(forget_ids) > 0,
            "train_nonempty": len(train_ids) > 0,
            "eval_nonempty": len(eval_ids) > 0,
        },
    }

    output_path = getattr(args, "output", None)
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(_json_pr.dumps(report, indent=2, sort_keys=False))
        log.info("P0-9: protocol report written to %s", out_p)

    _print_json(report)
    all_invariants_ok = all(report["invariants"].values()) and disjointness_ok
    return 0 if all_invariants_ok else 1


def cmd_source_preflight(args) -> int:
    """P1-11: CPU-only gate before GPU use.

    Runs every pre-annotation check without loading a model:
      1. revision verification
      2. source schema validation (adapter construction + row iteration)
      3. split-membership validation
      4. variant flattening validation (to_samples on every row)
      5. image existence validation
      6. smoke-selection feasibility
      7. wrong-name feasibility
    """
    from .data.adapters.base import AdapterError, available_adapters, create_adapter

    data_cfg = _data_config_for(args.dataset, load_run_config(args.config))
    if args.dataset not in available_adapters():
        raise ConfigError(
            f"Unknown dataset '{args.dataset}'; available: {available_adapters()}"
        )
    adapter = create_adapter(data_cfg)

    report: dict[str, Any] = {
        "dataset": args.dataset,
        "adapter": adapter.name,
        "adapter_version": adapter.adapter_version,
        "source_revision": data_cfg.source_version,
        "checks": {},
        "errors": [],
    }
    errors: list[str] = report["errors"]

    # 1. Revision verification (delegate to verify-revision logic).
    try:
        immutable = data_cfg.extras.get("immutable_revision")
        if immutable and isinstance(immutable, dict):
            import hashlib as _hashlib
            import subprocess as _sp

            source_root = data_cfg.require_root()
            configured_git = immutable.get("git_commit_sha", "")
            try:
                actual_git = _sp.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(source_root),
                    stderr=_sp.DEVNULL,
                ).decode().strip()
            except Exception:
                actual_git = "unknown"
            git_ok = configured_git == actual_git
            report["checks"]["revision_git"] = {
                "configured": configured_git,
                "actual": actual_git,
                "pass": git_ok,
            }
            if not git_ok:
                errors.append(f"git_commit_sha mismatch: {configured_git} vs {actual_git}")
            # File hashes.
            files_block = immutable.get("files", {})
            for rel_path, spec in files_block.items():
                if not isinstance(spec, dict):
                    continue
                cfg_hash = spec.get("sha256", "")
                fpath = source_root / rel_path
                actual_hash = (
                    _hashlib.sha256(fpath.read_bytes()).hexdigest()
                    if fpath.exists()
                    else "MISSING"
                )
                ok = cfg_hash == actual_hash
                report["checks"][f"revision_file:{rel_path}"] = {"pass": ok}
                if not ok:
                    errors.append(f"{rel_path} hash mismatch")
        else:
            report["checks"]["revision"] = {"pass": False, "reason": "no immutable_revision"}
            errors.append("no immutable_revision configured")
    except Exception as exc:
        report["checks"]["revision"] = {"pass": False, "reason": str(exc)}
        errors.append(f"revision check failed: {exc}")

    # 2. Source schema validation: iterate rows.
    n_rows = 0
    row_errors: list[str] = []
    try:
        for _ctx, _row in adapter.iter_rows_with_context():
            n_rows += 1
    except Exception as exc:
        row_errors.append(str(exc))
    report["checks"]["source_schema"] = {
        "rows": n_rows,
        "pass": len(row_errors) == 0,
        "errors": row_errors,
    }
    errors.extend(row_errors)

    # 3+4. Split membership + variant flattening.
    n_samples = 0
    split_counts: dict[str, int] = {}
    variant_counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    dup_ids: list[str] = []
    flatten_errors: list[str] = []
    try:
        for _ctx, _row in adapter.iter_rows_with_context():
            try:
                samples = list(adapter.to_samples(_row, source_context=_ctx))
            except AdapterError as exc:
                flatten_errors.append(str(exc))
                continue
            for _s in samples:
                n_samples += 1
                split_counts[_s.split] = split_counts.get(_s.split, 0) + 1
                vt = (_s.source_metadata or {}).get("variant_type", "original")
                variant_counts[vt] = variant_counts.get(vt, 0) + 1
                if _s.source_sample_id in seen_ids:
                    dup_ids.append(_s.source_sample_id)
                seen_ids.add(_s.source_sample_id)
    except Exception as exc:
        flatten_errors.append(str(exc))
    report["checks"]["split_membership"] = {
        "effective_split_counts": dict(sorted(split_counts.items())),
        "pass": len(flatten_errors) == 0,
    }
    report["checks"]["variant_flattening"] = {
        "total_samples": n_samples,
        "variant_counts": dict(sorted(variant_counts.items())),
        "duplicate_ids": len(dup_ids),
        "pass": len(flatten_errors) == 0 and len(dup_ids) == 0,
        "errors": flatten_errors[:10],
    }
    errors.extend(flatten_errors)
    if dup_ids:
        errors.append(f"{len(dup_ids)} duplicate source_sample_id(s)")

    # 4b. Protocol validation (P1-11).
    proto_cfg = data_cfg.extras.get("fiubench_protocol")
    if proto_cfg and isinstance(proto_cfg, dict):
        from .data.split_mapping import resolve_protocol_role
        forget_bkt = proto_cfg.get("forget_bucket")
        train_bkt = proto_cfg.get("train_bucket")
        eval_bkt = proto_cfg.get("eval_bucket")
        # P0-6: read actual bucket names from dataset/split.json via the adapter.
        all_released: set[str] = set()
        if hasattr(adapter, "official_split_buckets"):
            all_released = adapter.official_split_buckets()
        if not all_released:
            # Fallback: collect from row data (legacy fixtures).
            for _ctx2, _row2 in adapter.iter_rows_with_context():
                _md = _row2.get("raw_data", {})
                if isinstance(_md, dict):
                    for _k in ("official_memberships", "memberships", "split_group"):
                        _v = _md.get(_k)
                        if isinstance(_v, list):
                            all_released.update(str(x) for x in _v)
                _sb = _row2.get("split_buckets") or _row2.get("_split_buckets")
                if isinstance(_sb, dict):
                    all_released.update(_sb.keys())
                break  # Only need the first row for schema check.
        # Bucket existence checks.
        proto_errors: list[str] = []
        if forget_bkt and all_released and forget_bkt not in all_released:
            proto_errors.append(f"configured forget_bucket '{forget_bkt}' not found in source")
        if train_bkt and all_released and train_bkt not in all_released:
            proto_errors.append(f"configured train_bucket '{train_bkt}' not found in source")
        if eval_bkt and all_released and eval_bkt not in all_released:
            proto_errors.append(f"configured eval_bucket '{eval_bkt}' not found in source")
        # Identity-level disjointness and pool checks.
        _id_by_role: dict[str, set[str]] = {}
        try:
            for _ctx3, _row3 in adapter.iter_rows_with_context():
                _samples3 = list(adapter.to_samples(_row3, source_context=_ctx3))
                for _s3 in _samples3:
                    _sm3 = _s3.source_metadata or {}
                    _sid3 = str(_sm3.get("source_subject_id", ""))
                    _memberships = _sm3.get("official_memberships") or []
                    _role = resolve_protocol_role(
                        [str(m) for m in _memberships], proto_cfg,
                        source_subject_id=_sid3 if _sid3 else None,
                    )
                    if _sid3:
                        _id_by_role.setdefault(_role, set()).add(_sid3)
        except Exception:
            pass
        _forget_ids = _id_by_role.get("exclude", set())
        _train_ids = _id_by_role.get("train", set())
        _eval_ids = _id_by_role.get("eval", set())
        _oop_ids = _id_by_role.get("out_of_protocol", set())
        # Disjointness.
        if _forget_ids & _train_ids:
            proto_errors.append(
                f"forget/train overlap: {len(_forget_ids & _train_ids)} identities"
            )
        if _forget_ids & _eval_ids:
            proto_errors.append(
                f"forget/eval overlap: {len(_forget_ids & _eval_ids)} identities"
            )
        if _train_ids & _eval_ids:
            proto_errors.append(
                f"train/eval overlap: {len(_train_ids & _eval_ids)} identities"
            )
        # P0-5: unassigned and hash must be zero when protocol is active.
        _n_unassigned = split_counts.get("unassigned", 0)
        _n_hash = split_counts.get("hash", 0)
        if _n_unassigned > 0:
            proto_errors.append(
                f"P0: FIUBench protocol-active source contains {_n_unassigned} "
                "unassigned identities; all official source identities must "
                "resolve to train/eval/exclude/out_of_protocol"
            )
        if _n_hash > 0:
            proto_errors.append(
                f"P0: FIUBench protocol-active source contains {_n_hash} "
                "hash-assigned identities; all official source identities must "
                "resolve to train/eval/exclude/out_of_protocol"
            )
        # Also check role_counts from identity-level analysis.
        _role_unassigned = len(_id_by_role.get("unassigned", set()))
        _role_hash = len(_id_by_role.get("hash", set()))
        if _role_unassigned > 0 and _n_unassigned == 0:
            proto_errors.append(
                f"P0: {_role_unassigned} identity-level unassigned role(s) detected"
            )
        if _role_hash > 0 and _n_hash == 0:
            proto_errors.append(
                f"P0: {_role_hash} identity-level hash role(s) detected"
            )
        # Empty pool checks.
        if forget_bkt and not _forget_ids:
            proto_errors.append("eval pool empty: no forget/exclude identities")
        if train_bkt and not _train_ids:
            proto_errors.append("train pool empty: no train identities")
        if (eval_bkt or (proto_cfg.get("eval_fraction", 0) > 0)) and not _eval_ids:
            proto_errors.append("eval pool empty: no eval identities")
        # Build role_counts including unassigned/hash for full visibility.
        _all_role_counts: dict[str, int] = {}
        for _rk, _rv in sorted(_id_by_role.items()):
            _all_role_counts[_rk] = len(_rv)
        _all_role_counts.setdefault("unassigned", _n_unassigned)
        _all_role_counts.setdefault("hash", _n_hash)
        report["checks"]["protocol_validation"] = {
            "forget_bucket": forget_bkt,
            "train_bucket": train_bkt,
            "eval_bucket": eval_bkt,
            "official_split_buckets": sorted(all_released) if all_released else [],
            "role_counts": dict(sorted(_all_role_counts.items())),
            "pass": len(proto_errors) == 0,
            "errors": proto_errors,
        }
        errors.extend(proto_errors)

    # 5. Image existence.
    n_with_images = 0
    missing_images: list[str] = []
    try:
        for _ctx, _row in adapter.iter_rows_with_context():
            try:
                samples = list(adapter.to_samples(_row, source_context=_ctx))
            except AdapterError:
                continue
            for _s in samples:
                if _s.image_uri:
                    n_with_images += 1
                    from pathlib import Path as _Path
                    if not _Path(_s.image_uri).exists() and len(missing_images) < 20:
                        missing_images.append(_s.image_uri)
    except Exception:
        pass
    report["checks"]["image_existence"] = {
        "samples_with_images": n_with_images,
        "missing_images": len(missing_images),
        "pass": len(missing_images) == 0,
    }
    if missing_images:
        errors.append(f"{len(missing_images)} missing image(s) (first: {missing_images[0]})")

    # 6+7. Smoke-selection feasibility + wrong-name feasibility.
    try:
        from .build.conflict_generation import find_wrong_name_candidates
        from .data.split_mapping import load_source_mapping, resolve_effective_split

        all_samples = list(adapter.load())
        src_map = load_source_mapping(data_cfg)
        for _s in all_samples:
            d = _s.to_dict() if hasattr(_s, "to_dict") else _s
            d["_effective_split"] = resolve_effective_split(d, source_mapping=src_map)
        identities = set()
        image_bearing = set()
        splits_seen: set[str] = set()
        has_facts = False
        for _s in all_samples:
            d = _s.to_dict() if hasattr(_s, "to_dict") else _s
            identities.add(d.get("identity_id", ""))
            if d.get("image_uri"):
                image_bearing.add(d.get("identity_id", ""))
            splits_seen.add(d.get("_effective_split", ""))
            if d.get("profile_facts"):
                has_facts = True
        by_identity: dict[str, list] = {}
        for _s in all_samples:
            d = _s.to_dict() if hasattr(_s, "to_dict") else _s
            iid = d.get("identity_id", "")
            if iid:
                by_identity.setdefault(iid, []).append(d)
        wn_pairs = find_wrong_name_candidates(by_identity)
        report["checks"]["smoke_feasibility"] = {
            "total_identities": len(identities),
            "image_bearing_identities": len(image_bearing),
            "splits_seen": sorted(splits_seen),
            "has_profile_facts": has_facts,
            "pass": len(identities) >= 3 and len(image_bearing) >= 2,
        }
        report["checks"]["wrong_name_feasibility"] = {
            "candidate_pairs": len(wn_pairs),
            "pass": len(wn_pairs) > 0,
        }
    except Exception as exc:
        report["checks"]["smoke_feasibility"] = {"pass": False, "reason": str(exc)}
        errors.append(f"smoke feasibility check failed: {exc}")

    report["all_pass"] = len(errors) == 0
    _print_json(report)
    return 0 if report["all_pass"] else 1


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

    # P1-2/P1-5: coverage-aware selection — restrict to manifest allowlist.
    samples = _filter_by_smoke_manifest(samples, args)

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
    )
    from .constants.celeba_attributes import CELEBA_ATTRIBUTES

    # -- 0. Load whitelist (P0-1) ---------------------------------------- #
    whitelist_attrs: frozenset[str] | None = None
    wl = None  # R12: keep the full record for the score manifest.
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
    except Exception as exc:
        if run_cfg.model.backend == "stub":
            log.warning("Backend pre-load failed (%s); fingerprint may lack resolved revision", exc)
        else:
            raise ConfigError(f"Backend pre-load failed: {exc}") from exc

    # P0-7: fingerprint and resolved revision are mandatory for real backends.
    fingerprint_id: str | None = None
    fingerprint_data: dict[str, Any] = {}
    try:
        fingerprint_data = backend.fingerprint()
        # P0-A2: validate before str() conversion — str(None) == "None" is
        # truthy and would silently bypass the missing-fingerprint check.
        raw_fp_id = fingerprint_data.get("fingerprint_id")
        if not raw_fp_id:
            if run_cfg.model.backend != "stub":
                raise ConfigError("Model fingerprint_id is required")
            fingerprint_id = None
        else:
            fingerprint_id = str(raw_fp_id)
    except ConfigError:
        raise
    except Exception as exc:
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
    import hashlib as _hashlib_cache

    from .models.scoring import SCORING_VERSION

    candidates_blob = json.dumps(["Yes", "No"], sort_keys=True)
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
            except Exception:
                sha = ""
        image_sha_by_sample[s.source_sample_id] = sha

    # P2-1: build image-level deduplication structures.
    # Map image_sha256 -> representative CanonicalSample (first seen).
    image_to_rep_sample: dict[str, Any] = {}
    for s in samples:
        sha = image_sha_by_sample.get(s.source_sample_id, "")
        if sha and sha not in image_to_rep_sample:
            image_to_rep_sample[sha] = s

    def _row_cache_key(sample_id: str) -> str:
        return f"{image_sha_by_sample.get(sample_id, '')}|{cache_key_suffix}"

    def _image_cache_key(image_sha: str) -> str:
        """P2-2: canonical cache key for image-level scoring."""
        return f"{image_sha}|{cache_key_suffix}"

    import math as _math  # needed for cache validation below

    # P2-1: done_keys are now (image_sha256, attr) for image-level dedup.
    done_keys: set[tuple[str, str]] = set()
    score_rows: list[dict[str, Any]] = []
    if scores_path.exists() and args.resume:
        from .constants.celeba_attributes import CELEBA_ATTRIBUTE_SET

        raw_rows = list(read_jsonl(scores_path))

        # P0-B4 + P0-B6: validate cached rows BEFORE constructing done_keys.
        # A corrupt cached row must never suppress rescoring.
        validated_cache: list[dict[str, Any]] = []
        for r in raw_rows:
            sid = r.get("sample_id", "")
            attr = r.get("attribute", "")
            p = r.get("p_positive")
            ck = r.get("_cache_key", "")
            img_sha = r.get("image_sha256", "")

            # Schema validation (P0-B6)
            if not sid:
                log.warning("Dropping cached row: empty sample_id")
                continue
            if attr not in CELEBA_ATTRIBUTE_SET:
                log.warning("Dropping cached row (%s, %s): invalid attribute", sid, attr)
                continue
            if p is None or not isinstance(p, (int, float)):
                log.warning("Dropping cached row (%s, %s): p_positive not numeric", sid, attr)
                continue
            if _math.isnan(p) or _math.isinf(p):
                log.warning("Dropping cached row (%s, %s): p_positive=%s", sid, attr, p)
                continue
            if not (0.0 <= p <= 1.0):
                log.warning("Dropping cached row (%s, %s): p_positive=%s out of [0,1]", sid, attr, p)
                continue
            if not ck:
                log.warning("Dropping cached row (%s, %s): empty _cache_key", sid, attr)
                continue

            # P2-5: cache-key match — accept rows keyed by image_sha256 OR
            # by sample_id (backward compat with pre-P2 caches).
            if img_sha:
                # P0-5: reject stale cached rows whose image SHA no longer
                # matches the current source image for this sample_id.
                current_sha = image_sha_by_sample.get(sid, "")
                if not current_sha:
                    log.warning(
                        "Dropping cached row (%s, %s): current image SHA unavailable",
                        sid, attr,
                    )
                    continue
                if img_sha != current_sha:
                    log.warning(
                        "Dropping stale cached row (%s, %s): "
                        "cached image=%s current image=%s",
                        sid, attr, img_sha[:12], current_sha[:12],
                    )
                    continue
                expected_key = _image_cache_key(current_sha)
            else:
                expected_key = _row_cache_key(sid)
            if ck != expected_key:
                continue

            validated_cache.append(r)

        # P0-B5: detect conflicting duplicates — same (image, attribute)
        # with different p_positive values must raise.
        seen_cache: dict[tuple[str, str], float] = {}
        for r in validated_cache:
            # P2-1: key by image_sha256 when available, else sample_id.
            img_sha = r.get("image_sha256", "")
            dedup_key_id = img_sha if img_sha else r["sample_id"]
            key = (dedup_key_id, r["attribute"])
            if key in seen_cache:
                if seen_cache[key] != r["p_positive"]:
                    raise ConfigError(
                        f"Conflicting duplicate scores for ({key[0]}, {key[1]}): "
                        f"{seen_cache[key]} vs {r['p_positive']}. "
                        f"Remove the cache or use --resume with a clean scores file."
                    )
                # Exact duplicate: silently skip
                continue
            seen_cache[key] = r["p_positive"]
            score_rows.append(r)
            done_keys.add(key)

        if len(score_rows) < len(raw_rows):
            log.info(
                "Resume: kept %d/%d cached rows after validation",
                len(score_rows), len(raw_rows),
            )

    sample_ids = {s.source_sample_id for s in samples}
    score_rows = [r for r in score_rows if r["sample_id"] in sample_ids]

    # P2-1: build pending list from unique (image_sha256, attr) pairs.
    # Each unique image is scored once per attribute; all QA variants sharing
    # the image reuse the same scores.
    pending = [
        (rep, attr, sha)
        for sha, rep in image_to_rep_sample.items()
        for attr in CELEBA_ATTRIBUTES
        if rep.image_uri
        and (sha, attr) not in done_keys
    ]
    if pending:
        log.info(
            "Scoring %d unique (image, attribute) queries with backend=%s "
            "(%d canonical samples share these images)",
            len(pending), run_cfg.model.backend,
            sum(
                1 for s in samples
                if image_sha_by_sample.get(s.source_sample_id, "") in image_to_rep_sample
            ) - len(image_to_rep_sample),
        )
        for sample, attr, img_sha in pending:
            image = _load_image(sample.image_uri, base=image_base)
            resp = backend.score_candidates(
                image, registry.binary_prompt(attr), ["Yes", "No"]
            )
            p = _p_positive(resp)
            if p is None:
                continue
            # P0-B5: reject inconsistent duplicates during live scoring.
            if (img_sha, attr) in done_keys:
                for existing in score_rows:
                    e_sha = existing.get("image_sha256", "")
                    if (e_sha == img_sha and existing["attribute"] == attr):
                        if existing["p_positive"] != p:
                            raise ConfigError(
                                f"Conflicting duplicate score for "
                                f"image({img_sha[:12]}…), {attr}: "
                                f"{existing['p_positive']} vs {p}"
                            )
                        break
                log.warning(
                    "Duplicate score for (image=%s, %s); keeping first occurrence",
                    img_sha[:12], attr,
                )
                continue
            score_rows.append(
                {
                    "sample_id": sample.source_sample_id,
                    "image_sha256": img_sha,
                    "attribute": attr,
                    "p_positive": p,
                    "raw_text": resp.text,
                    "_cache_key": _image_cache_key(img_sha),
                }
            )
            done_keys.add((img_sha, attr))
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # P1-11: strictly validate score-cache rows (resume path).  Reject
    # NaN, Inf, out-of-range, and duplicate entries so stale or corrupt
    # caches never silently contaminate downstream annotations.
    # (_math already imported above)

    validated_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for r in score_rows:
        p = r.get("p_positive")
        sid = r.get("sample_id", "")
        attr = r.get("attribute", "")
        img_sha = r.get("image_sha256", "")
        if p is None or not isinstance(p, (int, float)):
            log.warning("Dropping score row (%s, %s): p_positive is not numeric", sid, attr)
            continue
        if _math.isnan(p) or _math.isinf(p):
            log.warning("Dropping score row (%s, %s): p_positive=%s", sid, attr, p)
            continue
        if not (0.0 <= p <= 1.0):
            log.warning("Dropping score row (%s, %s): p_positive=%s out of [0,1]", sid, attr, p)
            continue
        # P2-1: dedup by (image_sha256, attr) when available, else (sample_id, attr).
        dedup_id = img_sha if img_sha else sid
        key = (dedup_id, attr)
        if key in seen_keys:
            log.warning("Dropping duplicate score row (%s, %s)", dedup_id[:12], attr)
            continue
        seen_keys.add(key)
        validated_rows.append(r)
    if len(validated_rows) < len(score_rows):
        log.info(
            "Score validation: kept %d/%d rows after NaN/Inf/range/dedup checks",
            len(validated_rows), len(score_rows),
        )
    score_rows = validated_rows

    # P0-A3: model-collapse diagnostics — detect when the backend produces
    # degenerate scores (all ~0.5, all identical margins, constant probability).
    #
    # R1 (Fix List for f59a9c1): candidate scoring via backend.score_candidates
    # intentionally returns text="" because it computes candidate sequence
    # likelihoods rather than free-generating an answer.  Blank generation text
    # is therefore NOT a valid collapse signal here — free-generation checks
    # belong in the model smoke-test (R2).  Health checks are restricted to
    # probability-based invariants: finite P(yes), P(yes) in [0,1], std(P(yes)),
    # unique rounded P(yes), predicted-positive rate, and min/max P(yes).
    if score_rows and run_cfg.model.backend != "stub":
        import statistics as _stats

        p_vals = [r["p_positive"] for r in score_rows if isinstance(r.get("p_positive"), (int, float))]
        if p_vals:
            mean_p = _stats.mean(p_vals)
            std_p = _stats.pstdev(p_vals) if len(p_vals) > 1 else 0.0
            pos_rate = sum(1 for p in p_vals if p >= 0.5) / len(p_vals)
            min_p = min(p_vals)
            max_p = max(p_vals)
            collapse_reasons: list[str] = []
            if std_p < 1e-6 and len(p_vals) > 1:
                collapse_reasons.append(f"all P(yes) identical (std={std_p:.2e})")
            if all(abs(p - 0.5) < 0.01 for p in p_vals):
                collapse_reasons.append("all P(yes) near 0.5")
            unique_rounded = {round(p, 3) for p in p_vals}
            if len(unique_rounded) <= 1 and len(p_vals) > 1:
                collapse_reasons.append(f"all rounded scores identical ({unique_rounded})")
            if collapse_reasons:
                raise ConfigError(
                    f"Model collapse detected: {'; '.join(collapse_reasons)}. "
                    f"positive_rate={pos_rate:.3f}, mean_p={mean_p:.4f}, std_p={std_p:.4f}, "
                    f"min_p={min_p:.4f}, max_p={max_p:.4f}, n={len(p_vals)}"
                )
            log.info(
                "Collapse diagnostics: pos_rate=%.3f, mean_p=%.4f, std_p=%.4f, "
                "min_p=%.4f, max_p=%.4f, unique_rounded=%d",
                pos_rate, mean_p, std_p, min_p, max_p, len(unique_rounded),
            )

    # P2-1: score-completion invariant — every unique image must have all 40
    # valid attribute scores.  Missing pairs indicate cache bugs or silent
    # scoring failures.  Extra pairs indicate duplicate or phantom rows.
    unique_image_shas = {sha for sha in image_to_rep_sample if image_to_rep_sample[sha].image_uri}
    expected_pairs = {
        (sha, attr)
        for sha in unique_image_shas
        for attr in CELEBA_ATTRIBUTES
    }
    actual_pairs = {
        (
            r.get("image_sha256", "") or image_sha_by_sample.get(r["sample_id"], ""),
            r["attribute"],
        )
        for r in score_rows
    }
    missing_pairs = expected_pairs - actual_pairs
    extra_pairs = actual_pairs - expected_pairs
    if missing_pairs or extra_pairs:
        raise ConfigError(
            f"Score table mismatch: "
            f"{len(missing_pairs)} missing, {len(extra_pairs)} unexpected "
            f"out of {len(expected_pairs)} expected pairs. "
            f"First 5 missing: {sorted(missing_pairs)[:5]}, "
            f"first 5 extra: {sorted(extra_pairs)[:5]}"
        )
    if len(score_rows) != len(expected_pairs):
        raise ConfigError(
            f"Score row count mismatch: {len(score_rows)} rows vs "
            f"{len(expected_pairs)} expected pairs"
        )
    log.info(
        "Score-completion OK: %d unique images × %d attributes = %d scores",
        len(unique_image_shas), len(CELEBA_ATTRIBUTES), len(score_rows),
    )

    write_jsonl(score_rows, scores_path)

    # P2-3: build image-level score table and write artifact.
    from .build.annotate import (
        ImageScoreTable,
        annotate_sample_via_image_table,
    )

    image_score_table = ImageScoreTable.from_score_rows(score_rows, image_sha_by_sample)
    image_scores_path = dataset_dir / f"{args.dataset}_image_scores.jsonl"
    image_score_rows = image_score_table.to_image_score_rows(
        model_fingerprint=fingerprint_id,
        prompt_registry_hash=registry.registry_hash(),
    )
    write_jsonl(image_score_rows, image_scores_path)
    log.info(
        "P2-3: image score table: %d unique images, %d score rows → %s",
        image_score_table.unique_images, len(image_score_rows), image_scores_path,
    )

    # P1-10: write a dedicated score manifest capturing the full immutable
    # scoring identity so the cache is auditable and reproducible.
    # P1-13: compute source provenance using metadata-only hashing (JSON/
    # index/split files only, not images) for efficiency.  Full tree hash
    # is available via --full-source-hash for reproducibility audits.
    resolved_revision = fingerprint_data.get("revision") or getattr(
        backend, "_resolved_revision", None
    )

    source_hash: str | None = None
    source_provenance: dict[str, Any] = {}
    try:
        source_root = data_cfg.require_root()
        _h = _hashlib_cache.sha256()
        _meta_files: list[Path] = []
        for p in sorted(source_root.rglob("*")):
            if not p.is_file() or p.name.startswith("."):
                continue
            # P1-13: hash metadata files (json, yaml, csv, txt, tsv) fully;
            # for image files only record their relative paths (not contents)
            # unless --full-source-hash is set.
            if p.suffix.lower() in (".json", ".yaml", ".yml", ".csv", ".txt", ".tsv"):
                _h.update(str(p.relative_to(source_root)).encode())
                _h.update(p.read_bytes())
                _meta_files.append(p)
            else:
                # Record image file paths for tree structure but not contents.
                _h.update(str(p.relative_to(source_root)).encode())
        source_hash = _h.hexdigest()
        source_provenance["hash_strategy"] = "metadata_full_image_paths"
        source_provenance["metadata_file_count"] = len(_meta_files)
    except Exception as exc:
        log.warning("Source hash unavailable (%s); skipping source pinning", exc)

    # P1-14: complete the score manifest with all fields needed for
    # full auditability and reproducibility.
    score_manifest: dict[str, Any] = {
        "dataset": args.dataset,
        "model_id": run_cfg.model.model_id,
        "backend": run_cfg.model.backend,
        "configured_revision": getattr(run_cfg.model, "revision", None),
        "resolved_revision": resolved_revision,
        "model_fingerprint": fingerprint_id,
        "fingerprint_id": fingerprint_id,
        # R11: preserve the complete backend fingerprint, not only its ID.
        "model_fingerprint_payload": fingerprint_data,
        "prompt_registry_hash": registry.registry_hash(),
        "scoring_version": SCORING_VERSION,
        "candidate_set_hash": candidate_set_hash,
        "score_rows": len(score_rows),
        "source_version": data_cfg.source_version,
        "source_hash": source_hash,
        "source_provenance": source_provenance,
        # R17: full runtime environment for reproducibility audits.
        "runtime_environment": _runtime_environment(),
    }
    # Optional: dtype, quantization.
    if getattr(run_cfg.model, "dtype", None):
        score_manifest["dtype"] = run_cfg.model.dtype
    if getattr(run_cfg.model, "quantization", None):
        score_manifest["quantization"] = run_cfg.model.quantization
    # Optional: library versions.
    try:
        import transformers as _tx
        score_manifest["transformers_version"] = _tx.__version__
    except Exception:
        pass
    try:
        import torch as _torch
        score_manifest["torch_version"] = _torch.__version__
    except Exception:
        pass
    try:
        import subprocess as _sp
        _sha = _sp.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            stderr=_sp.DEVNULL,
        ).decode().strip()
        score_manifest["midp_commit"] = _sha
    except Exception:
        pass
    # R12: record the true whitelist-file SHA-256 (from whitelist loading)
    # plus full provenance, not a hash of the sorted attribute list alone.
    if wl is not None:
        import hashlib as _hl

        score_manifest["whitelist_path"] = str(wl.path) if wl.path else None
        score_manifest["whitelist_file_sha256"] = wl.sha256
        score_manifest["whitelist_attributes"] = sorted(wl.attributes)
        score_manifest["whitelist_attributes_sha256"] = _hl.sha256(
            "|".join(sorted(wl.attributes)).encode()
        ).hexdigest()
        score_manifest["whitelist_source_commit"] = wl.source_commit
        score_manifest["whitelist_policy"] = wl.policy
    elif whitelist_attrs:
        import hashlib as _hl

        score_manifest["whitelist_attributes"] = sorted(whitelist_attrs)
        score_manifest["whitelist_attributes_sha256"] = _hl.sha256(
            "|".join(sorted(whitelist_attrs)).encode()
        ).hexdigest()
    # P1-6: record selection manifest identity in build provenance.
    _sm_path = getattr(args, "smoke_manifest", None)
    if _sm_path:
        import hashlib as _hl_sm
        _sm_p = Path(_sm_path)
        if not _sm_p.is_absolute():
            _sm_p = (Path(args.config).resolve().parent / _sm_p).resolve()
        if _sm_p.exists():
            _sm_bytes = _sm_p.read_bytes()
            score_manifest["selection_manifest_path"] = str(_sm_p)
            score_manifest["selection_manifest_sha256"] = _hl_sm.sha256(_sm_bytes).hexdigest()
            import json as _json_sm
            _sm_data = _json_sm.loads(_sm_bytes)
            _sel_ids = _sm_data.get("selected_source_sample_ids", [])
            _sel_iids = _sm_data.get("selected_identity_ids", [])
            score_manifest["selected_source_sample_count"] = len(_sel_ids)
            score_manifest["selected_identity_count"] = len(_sel_iids)
            score_manifest["selected_source_sample_ids_hash"] = _hl_sm.sha256(
                "|".join(sorted(_sel_ids)).encode()
            ).hexdigest()
            score_manifest["selected_identity_ids_hash"] = _hl_sm.sha256(
                "|".join(sorted(_sel_iids)).encode()
            ).hexdigest()
    # P2-6: add image deduplication metrics to the score manifest.
    image_bearing = sum(1 for s in samples if s.image_uri)
    dedup_metrics = image_score_table.deduplication_metrics(
        canonical_samples=len(samples),
        image_bearing_samples=image_bearing,
        raw_score_rows=len(score_rows),
    )
    score_manifest["image_deduplication"] = dedup_metrics
    log.info(
        "P2-6: dedup metrics: %d canonical, %d image-bearing, %d unique images, "
        "%d score rows, %d avoided duplicate requests",
        dedup_metrics["canonical_samples"],
        dedup_metrics["image_bearing_samples"],
        dedup_metrics["unique_images"],
        dedup_metrics["raw_visual_score_rows"],
        dedup_metrics["avoided_duplicate_score_requests"],
    )

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
    # P2-1: annotate via image-level score table — each unique image is
    # annotated once, and scores propagate to all QA variants sharing it.
    annotated = [
        annotate_sample_via_image_table(annotator, s, image_score_table, image_sha_by_sample)
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
            annotate_sample_via_image_table(processed_annotator, s, image_score_table, image_sha_by_sample)
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
            "image_scores_path": str(image_scores_path),
            "unique_images": image_score_table.unique_images,
            "annotated_path": str(annotated_path),
            "processed_path": str(processed_path),
        }
    )
    return 0


def cmd_build_qa(args) -> int:
    run_cfg = load_run_config(args.config)
    dataset_dir = _dataset_dir(args, run_cfg, args.dataset)
    samples = _load_processed_samples(dataset_dir, args.dataset)
    # P0-3: defensive assertion — no unassigned/hash in protocol mode.
    _assert_protocol_no_unassigned_hash(args.dataset, load_run_config(args.config), samples)
    # P0-3: exclude out_of_protocol identities from downstream stages.
    samples = _filter_out_of_protocol(samples)
    # P1-5: downstream stages honor --smoke-manifest.
    samples = _filter_by_smoke_manifest(samples, args)
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
    # P0-C7: config-driven source-partition mapping that preserves all
    # official source partitions (train, validation, eval, test, forget,
    # unassigned).  Only unassigned records may use fallback hashing.
    # P0-C8: track assignment provenance (official vs hash) and never
    # rebalance official assignments.
    import hashlib as _hashlib

    # Default source mapping covering common benchmark partition vocabularies.
    # P1-10: use centralized split mapping module.
    from .data.split_mapping import load_source_mapping

    data_cfg = _data_config_for(args.dataset, run_cfg)
    source_mapping = load_source_mapping(data_cfg)

    train_identity_ids: set[str] = set()
    eval_identity_ids: set[str] = set()
    identity_to_split: dict[str, str] = {}
    # P0-C8: track provenance so rebalancing never moves official assignments.
    identity_provenance: dict[str, str] = {}  # identity_id -> "official" | "hash"

    # Pass 1: honor official source partitions.
    for s in samples:
        iid = s.identity_id
        if iid in identity_to_split:
            continue
        raw_split = s.split or "unassigned"
        # P0-3: out_of_protocol identities are excluded from QA entirely.
        # They must NOT be mapped through source_mapping to "exclude"
        # (which would conflate them with forget targets).
        if raw_split == "out_of_protocol":
            identity_to_split[iid] = "out_of_protocol"
            identity_provenance[iid] = "official"
            continue
        target = source_mapping.get(raw_split, "hash")
        if target == "exclude":
            # forget-only records are excluded from ordinary generated QA
            identity_to_split[iid] = "exclude"
            identity_provenance[iid] = "official"
            continue
        if target in ("train", "eval"):
            identity_to_split[iid] = target
            identity_provenance[iid] = "official"
            if target == "train":
                train_identity_ids.add(iid)
            else:
                eval_identity_ids.add(iid)
            continue
        # target == "hash": fall through to Pass 2
        # (unassigned or unknown labels use hash-based assignment)

    # Pass 2: hash-based fallback for identities without official assignment.
    for s in samples:
        iid = s.identity_id
        if iid in identity_to_split:
            continue
        h = int(_hashlib.sha256(iid.encode()).hexdigest()[:8], 16)
        split = "eval" if h % 5 == 0 else "train"
        identity_to_split[iid] = split
        identity_provenance[iid] = "hash"
        if split == "train":
            train_identity_ids.add(iid)
        else:
            eval_identity_ids.add(iid)

    # P0-C8: guarantee at least 1 identity in each split, but ONLY by
    # rebalancing hash-assigned identities.  Never move official assignments.
    if not eval_identity_ids and train_identity_ids:
        hash_train = [iid for iid in train_identity_ids if identity_provenance.get(iid) == "hash"]
        if hash_train:
            counts = {}
            for s in samples:
                counts[s.identity_id] = counts.get(s.identity_id, 0) + 1
            donor = max(hash_train, key=lambda iid: counts.get(iid, 0))
            train_identity_ids.discard(donor)
            eval_identity_ids.add(donor)
            identity_to_split[donor] = "eval"
        else:
            log.warning(
                "No eval identity and all train identities are official; "
                "cannot rebalance.  Eval coverage unavailable at this --limit."
            )
    elif not train_identity_ids and eval_identity_ids:
        hash_eval = [iid for iid in eval_identity_ids if identity_provenance.get(iid) == "hash"]
        if hash_eval:
            counts = {}
            for s in samples:
                counts[s.identity_id] = counts.get(s.identity_id, 0) + 1
            donor = max(hash_eval, key=lambda iid: counts.get(iid, 0))
            eval_identity_ids.discard(donor)
            train_identity_ids.add(donor)
            identity_to_split[donor] = "train"
        else:
            log.warning(
                "No train identity and all eval identities are official; "
                "cannot rebalance.  Train coverage unavailable at this --limit."
            )

    # Enforce identity-disjoint invariant
    assert not (train_identity_ids & eval_identity_ids), "identity split leakage"

    train_samples = []
    eval_samples = []
    for s in samples:
        # R3: explicit branching — forget/exclude identities must appear in
        # neither ordinary QA train nor QA eval.  Previously an else-branch
        # routed every non-train assignment (including "exclude") into eval.
        assignment = identity_to_split[s.identity_id]
        if assignment == "train":
            train_samples.append(s)
        elif assignment == "eval":
            eval_samples.append(s)
        elif assignment == "exclude":
            continue
        elif assignment == "out_of_protocol":
            # P0-3: out-of-protocol identities skip QA entirely.
            continue
        else:
            raise ConfigError(
                f"Unexpected QA split assignment '{assignment}' for identity "
                f"{s.identity_id}; expected 'train', 'eval', or 'exclude'"
            )

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


def _build_probe_coverage_report(
    dataset: str,
    samples: list,
    by_identity: dict[str, list],
    probe_rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    skipped: list[str],
) -> dict[str, Any]:
    """P1-18: build a detailed route-probe coverage report.

    Captures per-family probe counts, pair counts, identity coverage,
    and per-attribute positive/negative/state-change coverage.
    """
    from .build.conflict_generation import (
        _accepted_visible_attributes,
        _first_eligible_visual_attrs,
    )

    total_identities = len(by_identity)

    # Count identities with valid visual anchors (accepted visible attrs).
    identities_with_visual_anchor = 0
    identities_with_profile_facts = 0
    identities_with_second_valid_image = 0
    wrong_name_available = 0

    for iid, group in by_identity.items():
        # R16: use the same visual-anchor selection logic as route-probe
        # construction — the first sample with accepted visible attributes,
        # never blindly group[0] (which may be text-only).
        anchor_sample = None
        for s in group:
            if _accepted_visible_attributes(s):
                anchor_sample = s
                break
        if anchor_sample is not None:
            identities_with_visual_anchor += 1
        if any(s.profile_facts for s in group):
            identities_with_profile_facts += 1
        # Second valid image: a different image_uri than the *actual visual
        # anchor* (not group[0]) carrying accepted attributes.
        if anchor_sample is not None:
            for s in group:
                if s.source_sample_id == anchor_sample.source_sample_id:
                    continue
                if s.image_uri == anchor_sample.image_uri:
                    continue
                if _accepted_visible_attributes(s):
                    identities_with_second_valid_image += 1
                    break
        # Wrong-name availability: at least one other identity with
        # multiple samples and an eligible visual sample anywhere in its
        # group (R14/R16: not only the first canonical record).
        other_candidates = [
            other for other in by_identity
            if other != iid
            and len(by_identity[other]) >= 2
            and _first_eligible_visual_attrs(by_identity[other])
        ]
        if other_candidates:
            wrong_name_available += 1

    # Per-family probe counts.
    family_counts: dict[str, int] = {}
    for row in probe_rows:
        fam = row.get("probe_family", "unknown")
        family_counts[fam] = family_counts.get(fam, 0) + 1

    # Cross-image attribute-state pair count.
    cross_image_pairs = sum(
        1 for p in pairs if p.get("pair_type") == "cross_image_attribute_state"
    )

    # Per-attribute coverage: positive, negative, state-change.
    attr_coverage: dict[str, dict[str, int]] = {}
    for row in probe_rows:
        attr = row.get("target_attribute")
        if not attr:
            continue
        if attr not in attr_coverage:
            attr_coverage[attr] = {"positive": 0, "negative": 0, "state_change": 0}
        answer = row.get("answer_text")
        if answer == "yes":
            attr_coverage[attr]["positive"] += 1
        elif answer == "no":
            attr_coverage[attr]["negative"] += 1
    # State-change coverage from pairs.
    for p in pairs:
        if p.get("pair_type") == "cross_image_attribute_state":
            attr = p.get("attribute", "")
            if attr and attr in attr_coverage:
                attr_coverage[attr]["state_change"] += 1

    # Skipped identity reasons.
    skipped_by_reason: dict[str, int] = {}
    for entry in skipped:
        # entry is like "id123: <error message>"
        reason = entry.split(": ", 1)[-1] if ": " in entry else entry
        skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1

    return {
        "dataset": dataset,
        "identities_total": total_identities,
        "identities_with_visual_anchors": identities_with_visual_anchor,
        "identities_with_profile_facts": identities_with_profile_facts,
        "identities_with_second_valid_image": identities_with_second_valid_image,
        "wrong_name_availability": wrong_name_available,
        "probe_families": family_counts,
        "cross_image_attribute_state_pairs": cross_image_pairs,
        "skipped_identities_by_reason": skipped_by_reason,
        "per_attribute_coverage": dict(sorted(attr_coverage.items())),
    }


def cmd_build_route_probes(args) -> int:
    run_cfg = load_run_config(args.config)
    dataset_dir = _dataset_dir(args, run_cfg, args.dataset)
    samples = _load_processed_samples(dataset_dir, args.dataset)
    # P0-3: defensive assertion — no unassigned/hash in protocol mode.
    _assert_protocol_no_unassigned_hash(args.dataset, run_cfg, samples)
    # P0-3: exclude out_of_protocol identities from route probes.
    samples = _filter_out_of_protocol(samples)
    # P1-5: downstream stages honor --smoke-manifest.
    samples = _filter_by_smoke_manifest(samples, args)
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
        matched_wrong_name_details,
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

        # P2-19 / R14: matched wrong-name control — select the identity with
        # the most similar visual attribute profile (signed-state Jaccard of
        # accepted high-confidence labels), searching every sample of each
        # candidate group for an eligible visual sample.  The match metadata
        # is recorded on the wrong_name probe rows for causal analysis.
        wrong_details = matched_wrong_name_details(identity_id, by_identity)
        wrong_name = (
            wrong_details["wrong_identity_name"] if wrong_details else None
        )

        try:
            probes = build_identity_probes(
                group, builder, wrong_identity_name=wrong_name
            )
        except ConflictError as exc:
            skipped.append(f"{identity_id}: {exc}")
            continue
        for probe in probes:
            row = builder.probe_row(probe)
            # R14: attach wrong-name matching audit metadata.
            if wrong_details and row.get("probe_family") == "wrong_name":
                row["matched_wrong_identity_id"] = wrong_details[
                    "matched_wrong_identity_id"
                ]
                row["matching_similarity"] = wrong_details["matching_similarity"]
                row["matching_attributes"] = wrong_details["matching_attributes"]
                row["candidate_rank"] = wrong_details["candidate_rank"]
                row["matching_strategy"] = wrong_details["matching_strategy"]
            probe_rows.append(row)

        # P0-12 + Fix 4: validate cross_image_attribute_state pairs and
        # emit one explicit pair per differing target attribute so the
        # attribute that changed is recorded, not just "some attribute
        # differs".
        if len(group) >= 2:
            for left, right in itertools.pairwise(group):
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

    # P1-13: semantic pair validation — verify pair-type-specific invariants.
    # P0-C11: fail closed on invalid pair manifests unless explicitly allowed.
    samples_by_id = {s.source_sample_id: s for s in samples}
    pair_issues = validate_pair_manifest(pairs, samples_by_id)
    if pair_issues:
        for issue in pair_issues:
            log.warning("Pair validation issue: %s", issue)
        if not getattr(args, "allow_invalid_pairs", False):
            from .build.conflict_generation import ConflictError
            raise ConflictError(
                f"Pair manifest has {len(pair_issues)} validation issue(s): "
                + "; ".join(pair_issues[:5])
            )

    probes_path = dataset_dir / f"{args.dataset}_route_probes.jsonl"
    write_jsonl(probe_rows, probes_path)
    write_json(pairs, dataset_dir / f"{args.dataset}_pair_manifest.json")

    # P1-18: route-probe coverage report — detailed counts of what was
    # generated and what was skipped, per probe family and per attribute.
    coverage = _build_probe_coverage_report(
        args.dataset, samples, by_identity, probe_rows, pairs, skipped,
    )
    report_path = dataset_dir / f"{args.dataset}_route_probe_report.json"
    write_json(coverage, report_path)

    _print_json(
        {
            "dataset": args.dataset,
            "probe_rows": len(probe_rows),
            "pairs": len(pairs),
            "pair_validation_issues": len(pair_issues),
            "skipped_identities": skipped,
            "probes_path": str(probes_path),
            "coverage_report_path": str(report_path),
        }
    )
    return 0


def cmd_build_splits(args) -> int:
    run_cfg = load_run_config(args.config)
    dataset_dir = _dataset_dir(args, run_cfg, args.dataset)
    samples = _load_processed_samples(dataset_dir, args.dataset)
    # P0-3: defensive assertion — no unassigned/hash in protocol mode.
    _assert_protocol_no_unassigned_hash(args.dataset, run_cfg, samples)
    # P0-3: exclude out_of_protocol identities from split manifests.
    samples = _filter_out_of_protocol(samples)
    # P1-5: downstream stages honor --smoke-manifest.
    samples = _filter_by_smoke_manifest(samples, args)
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
    # P0-3: defensive assertion — no unassigned/hash in protocol mode.
    _assert_protocol_no_unassigned_hash(args.dataset, run_cfg, samples)
    # P0-3: exclude out_of_protocol identities from export.
    samples = _filter_out_of_protocol(samples)
    # P1-5: downstream stages honor --smoke-manifest.
    samples = _filter_by_smoke_manifest(samples, args)
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

    # P1-12: the score manifest is authoritative for model identity.
    # Read it first so export provenance reuses resolved_revision,
    # model_fingerprint, prompt_registry_hash, etc. instead of
    # reconstructing them from config (where revision may be null).
    data_cfg = _data_config_for(args.dataset, run_cfg)
    score_manifest_path = dataset_dir / f"{args.dataset}_score_manifest.json"
    score_manifest: dict[str, Any] = {}
    if score_manifest_path.exists():
        try:
            score_manifest = read_json(score_manifest_path)
        except Exception:
            pass

    provenance: dict[str, Any] = {
        "model_id": score_manifest.get("model_id", run_cfg.model.model_id),
        "model_backend": score_manifest.get("backend", run_cfg.model.backend),
        # P1-12: use resolved_revision from score manifest as authoritative;
        # fall back to config revision only when no score manifest exists.
        "model_revision": (
            score_manifest.get("resolved_revision")
            or score_manifest.get("configured_revision")
            or getattr(run_cfg.model, "revision", None)
        ),
        "model_fingerprint": (
            score_manifest.get("fingerprint_id") or model_fingerprint
        ),
        "prompt_registry_hash": (
            score_manifest.get("prompt_registry_hash") or registry_hash
        ),
        "candidate_set_hash": score_manifest.get("candidate_set_hash"),
        "scoring_version": score_manifest.get("scoring_version"),
        "source_version": score_manifest.get(
            "source_version", data_cfg.source_version
        ),
        "scoring_method": "candidate_sequence_log_probability",
    }
    # P1-13: propagate source_hash from score manifest (metadata-aware).
    # Only compute a fresh hash when the score manifest lacks one.
    if score_manifest.get("source_hash"):
        provenance["source_hash"] = score_manifest["source_hash"]
        provenance["source_provenance"] = score_manifest.get("source_provenance", {})
    if "source_hash" not in provenance:
        try:
            import hashlib as _hashlib_export
            source_root = data_cfg.require_root()
            _h = _hashlib_export.sha256()
            _meta_count = 0
            for p in sorted(source_root.rglob("*")):
                if not p.is_file() or p.name.startswith("."):
                    continue
                # P1-13: metadata-aware strategy — full hash for metadata
                # files, path-only for images (same as score manifest).
                if p.suffix.lower() in (".json", ".yaml", ".yml", ".csv", ".txt", ".tsv"):
                    _h.update(str(p.relative_to(source_root)).encode())
                    _h.update(p.read_bytes())
                    _meta_count += 1
                else:
                    _h.update(str(p.relative_to(source_root)).encode())
            provenance["source_hash"] = _h.hexdigest()
            provenance["source_provenance"] = {
                "hash_strategy": "metadata_full_image_paths",
                "metadata_file_count": _meta_count,
            }
        except Exception:
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
        except Exception as exc:
            log.warning("Whitelist provenance unavailable (%s)", exc)
    # Optional: library versions.
    try:
        import transformers
        provenance["transformers_version"] = transformers.__version__
    except Exception:
        pass
    try:
        import torch
        provenance["torch_version"] = torch.__version__
    except Exception:
        pass
    # Optional: MIDP git commit.
    try:
        import subprocess
        midp_root = Path(__file__).resolve().parents[4]
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=midp_root, stderr=subprocess.DEVNULL,
        ).decode().strip()
        provenance["midp_commit"] = sha
    except Exception:
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
    parser.add_argument(
        "--smoke-manifest",
        dest="smoke_manifest",
        default=None,
        help="path to a smoke subset manifest JSON; restrict processing to the listed sample IDs",
    )


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
    p.add_argument(
        "--image",
        action="append",
        default=None,
        help="path to a real test image (repeatable for multi-image smoke)",
    )
    p.add_argument(
        "--image-list",
        dest="image_list",
        default=None,
        help="path to a text file with one image path per line",
    )
    p.add_argument("--prompts", default=None, help="path to binary prompt registry YAML")
    p.add_argument(
        "--smoke-expected",
        dest="smoke_expected",
        default=None,
        help="path to JSON file with expected labels for sanity-check images",
    )
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

    # P1-1: pre-inference smoke manifest generation.
    p = source.add_parser(
        "make-smoke-manifest",
        help="generate a coverage-aware smoke manifest before model annotation",
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True, help="output manifest JSON path")
    p.add_argument("--min-identities", type=int, default=3, dest="min_identities")
    p.add_argument("--min-image-bearing", type=int, default=2, dest="min_image_bearing")
    p.add_argument("--require-multiview", action="store_true", dest="require_multiview")
    # P0-11: strict mode is the default — fail closed on missing roles/images.
    _strict_group = p.add_mutually_exclusive_group()
    _strict_group.add_argument(
        "--strict", action="store_true", default=True, dest="strict",
        help="fail closed on missing roles, bad images, etc. (default)",
    )
    _strict_group.add_argument(
        "--no-strict", action="store_false", dest="strict",
        help="record problems as warnings but still write the manifest",
    )
    # P2-12/14/15: pilot mode enables audit, route-family, and polarity checks.
    p.add_argument(
        "--pilot", action="store_true", default=False, dest="pilot",
        help="enable pilot-mode checks: audit status, route-family minimums, polarity",
    )
    _common_flags(p)
    p.set_defaults(func=cmd_source_make_smoke_manifest)

    # P0-2: fast source-revision verification before model loading.
    p = source.add_parser(
        "verify-revision",
        help="verify configured immutable_revision against the actual source",
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", required=True)
    _common_flags(p)
    p.set_defaults(func=cmd_source_verify_revision)

    # P0-9: protocol identity-count report.
    p = source.add_parser(
        "protocol-report",
        help="generate fiubench_protocol_report.json with identity counts (P0-9)",
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", default=None, help="output report JSON path")
    _common_flags(p)
    p.set_defaults(func=cmd_source_protocol_report)

    # P1-11: CPU-only preflight gate before GPU use.
    p = source.add_parser(
        "preflight",
        help="run all pre-annotation checks without loading a model (P1-11)",
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", required=True)
    _common_flags(p)
    p.set_defaults(func=cmd_source_preflight)

    # build
    build = sub.add_parser("build", help="extension construction pipeline").add_subparsers(
        dest="build_command", required=True
    )
    for name, func, help_text in (
        ("annotate", cmd_build_annotate, "score + annotate samples with CelebA-40 labels"),
        ("qa", cmd_build_qa, "generate versioned visual QA rows"),
        ("splits", cmd_build_splits, "build forget/retain splits and invariants"),
        ("export", cmd_build_export, "export the full auditable extension"),
    ):
        p = build.add_parser(name, help=help_text)
        p.add_argument("--dataset", required=True)
        p.add_argument("--config", required=True)
        _common_flags(p)
        p.set_defaults(func=func)
    # P0-C11: route-probes gets an extra --allow-invalid-pairs debug flag.
    p_rp = build.add_parser(
        "route-probes", help="generate route-conflict probes + pair manifest"
    )
    p_rp.add_argument("--dataset", required=True)
    p_rp.add_argument("--config", required=True)
    p_rp.add_argument(
        "--allow-invalid-pairs",
        action="store_true",
        help="log pair validation issues without failing (debugging only)",
    )
    _common_flags(p_rp)
    p_rp.set_defaults(func=cmd_build_route_probes)

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
