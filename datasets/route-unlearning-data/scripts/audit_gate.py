#!/usr/bin/env python3
"""R20: Manual audit gate script.

Before pilot/full generation, this script produces a structured audit report
for manual inspection of:

- 20 source mappings (identity_id → split bucket)
- 20 positive weak labels (high-confidence accepted annotations)
- 20 negative weak labels (rejected/low-confidence annotations)
- all tiny-smoke conflict probes
- all tiny-smoke cross-image state pairs
- all tiny-smoke name-only facts

Verifies: image paths, identity mapping, weak-label plausibility, wrong-name
correctness, actual contradiction direction, cross-image target state, exact
fact answer, and source split integrity.

Usage:
    python scripts/audit_gate.py --dataset fiubench --config configs/runs/...yaml
    python scripts/audit_gate.py --dataset fairget --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _load_data_config(dataset: str) -> dict:
    """Load the data config for a benchmark."""
    import yaml
    cfg_path = REPO / "configs" / "data" / f"{dataset}.yaml"
    if not cfg_path.exists():
        print(f"ERROR: data config not found: {cfg_path}")
        sys.exit(1)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _resolve_source_mapping(dataset: str) -> dict[str, str]:
    """Resolve the effective source mapping for a benchmark (P1-10: centralized)."""
    from route_data.data.split_mapping import load_source_mapping
    cfg = _load_data_config(dataset)
    return load_source_mapping(cfg)


def _resolve_artifact_path(
    output_dir: Path | None,
    dataset: str,
    logical_name: str,
    default_filename: str,
) -> Path | None:
    """P2-7/P2-8: resolve an artifact path consistently.

    Resolution order:
    1. Export manifest ``paths[logical_name]`` (if manifest exists).
    2. Canonical build artifact ``<dataset>_<default_filename>``.
    3. Alternative legacy name ``<dataset>_route_conflict_eval.jsonl``.

    Returns the first existing path, or ``None``.
    """
    if output_dir is None:
        return None

    # 1. Try export manifest.
    manifest_path = output_dir / f"{dataset}_export_manifest.json"
    if manifest_path.exists():
        try:
            mdata = json.loads(manifest_path.read_text())
            rel = mdata.get("paths", {}).get(logical_name)
            if rel:
                candidate = output_dir / rel
                if candidate.exists():
                    return candidate
        except Exception:
            pass

    # 2. Canonical build artifact.
    canonical = output_dir / f"{dataset}_{default_filename}"
    if canonical.exists():
        return canonical

    # 3. Legacy fallback for route probes.
    if default_filename != "route_conflict_eval.jsonl":
        legacy = output_dir / f"{dataset}_route_conflict_eval.jsonl"
        if legacy.exists():
            return legacy

    return None


def audit_source_mappings(dataset: str, limit: int = 20) -> list[dict]:
    """Audit 20 source mappings: identity_id → split bucket → target split."""
    print(f"\n{'='*72}")
    print(f"AUDIT: Source Mappings for {dataset.upper()}")
    print(f"{'='*72}")
    
    mapping = _resolve_source_mapping(dataset)
    print(f"Effective source mapping: {json.dumps(mapping, indent=2)}")
    
    # Try to load adapter samples via the standard config + create_adapter path.
    samples = []
    try:
        from route_data.config import load_data_config
        from route_data.data.adapters.base import create_adapter
        data_cfg_path = REPO / "configs" / "data" / f"{dataset}.yaml"
        data_cfg = load_data_config(data_cfg_path)
        adapter = create_adapter(data_cfg)
        samples = list(adapter.load())
        print(f"Loaded {len(samples)} samples from adapter")
    except (OSError, ValueError) as exc:
        print(f"WARNING: could not load adapter samples: {exc}")
        print("Falling back to config-based inspection")
    
    audit_rows = []
    seen_ids = set()
    
    for s in samples:
        if s.identity_id in seen_ids:
            continue
        seen_ids.add(s.identity_id)
        
        raw_split = s.split or "unassigned"
        target = mapping.get(raw_split, "hash")
        
        row = {
            "identity_id": s.identity_id,
            "raw_split": raw_split,
            "target_split": target,
            "image_uri": str(s.image_uri) if s.image_uri else None,
            "provenance": "official" if target != "hash" else "hash",
        }
        audit_rows.append(row)
        
        if len(audit_rows) >= limit:
            break
    
    # Print audit table
    print(f"\n{'identity_id':<40} {'raw_split':<15} {'target':<10} {'provenance':<10}")
    print("-" * 75)
    for row in audit_rows:
        print(f"{row['identity_id']:<40} {row['raw_split']:<15} {row['target_split']:<10} {row['provenance']:<10}")
    
    # Verify invariants
    exclude_count = sum(1 for r in audit_rows if r["target_split"] == "exclude")
    train_count = sum(1 for r in audit_rows if r["target_split"] == "train")
    eval_count = sum(1 for r in audit_rows if r["target_split"] == "eval")
    hash_count = sum(1 for r in audit_rows if r["provenance"] == "hash")
    
    print(f"\nSummary: train={train_count}, eval={eval_count}, exclude={exclude_count}, hash={hash_count}")
    
    # Check: forget identities should be excluded
    if exclude_count > 0:
        print("✓ exclude split contains identities (forget identities excluded)")
    else:
        print("⚠ no exclude identities found (no forget split in source?)")
    
    return audit_rows


def audit_wrong_name_coverage(dataset: str, limit: int = 10) -> list[dict]:
    """P1-9: audit wrong-name coverage using production eligibility logic."""
    print(f"\n{'='*72}")
    print(f"AUDIT: Wrong-Name Coverage for {dataset.upper()}")
    print(f"{'='*72}")

    samples = []
    try:
        from route_data.config import load_data_config
        from route_data.data.adapters.base import create_adapter
        data_cfg_path = REPO / "configs" / "data" / f"{dataset}.yaml"
        data_cfg = load_data_config(data_cfg_path)
        adapter = create_adapter(data_cfg)
        samples = list(adapter.load())
    except (OSError, ValueError) as exc:
        print(f"WARNING: could not load adapter samples: {exc}")
        return []

    # Group by identity.
    by_identity: dict[str, list] = {}
    for s in samples:
        sdict = s.to_dict() if hasattr(s, "to_dict") else s
        iid = sdict.get("identity_id", "") if isinstance(sdict, dict) else getattr(s, "identity_id", "")
        if iid:
            by_identity.setdefault(iid, []).append(sdict if isinstance(sdict, dict) else s)

    from route_data.build.conflict_generation import find_wrong_name_candidates
    pairs = find_wrong_name_candidates(by_identity)

    print(f"Total identities: {len(by_identity)}")
    print(f"Valid wrong-name pairs: {len(pairs)}")

    audit_rows = []
    for target, control, sim in pairs[:limit]:
        row = {
            "target_identity": target,
            "control_identity": control,
            "matching_similarity": round(sim, 4),
        }
        audit_rows.append(row)
        print(f"  target={target}, control={control}, similarity={sim:.4f}")

    if not pairs:
        print("⚠ no valid wrong-name pairs found")
    else:
        print(f"✓ {len(pairs)} wrong-name candidate pairs available")

    return audit_rows


def audit_weak_labels(dataset: str, output_dir: Path | None = None, limit: int = 20) -> tuple[list[dict], list[dict]]:
    """Audit 20 positive and 20 negative weak labels.

    P2-5: default to ``source == "source_model"``; report other categories
    separately (``human_verified_model``, ``source_human``).

    P2-6: positive weak label = source_model + label==True + confidence_band==high.
    Negative weak label = source_model + label==False + confidence_band==high.
    ``label==None`` is NOT treated as negative.
    """
    print(f"\n{'='*72}")
    print(f"AUDIT: Weak Labels for {dataset.upper()}")
    print(f"{'='*72}")

    # Try to load annotated parquet
    parquet_path = None
    if output_dir:
        parquet_path = output_dir / f"{dataset}_celeba40_image_annotations.parquet"

    if parquet_path and parquet_path.exists():
        print(f"Loading annotations from {parquet_path}")
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)

            # P2-2: the exporter uses long-format rows with columns:
            # identity_id, image_uri, attribute, label, score, source,
            # confidence_band, attribute_class, model_fingerprint, prompt_id.
            positive_rows: list[dict] = []
            negative_rows: list[dict] = []
            other_source_rows: list[dict] = []

            if "attribute" in df.columns and "score" in df.columns:
                for _, row in df.iterrows():
                    score_val = row.get("score")
                    label_val = row.get("label")
                    source_val = str(row.get("source", "unknown"))
                    band_val = row.get("confidence_band")
                    if pd.isna(score_val):
                        continue

                    entry = {
                        "identity_id": row.get("identity_id", "unknown"),
                        "image_uri": row.get("image_uri"),
                        "attribute": row.get("attribute", "unknown"),
                        "score": float(score_val),
                        "label": bool(label_val) if pd.notna(label_val) else None,
                        "source": source_val,
                        "confidence_band": str(band_val) if pd.notna(band_val) else None,
                    }

                    # P2-5: separate by source category.
                    if source_val != "source_model":
                        other_source_rows.append(entry)
                        continue

                    # P2-6: precise positive/negative definitions.
                    # label==None is NOT treated as negative.
                    if label_val is True and band_val == "high":
                        positive_rows.append(entry)
                    elif label_val is False and band_val == "high":
                        negative_rows.append(entry)
                    # else: source_model but not high-confidence or label==None → skip

                # Report other-source summary (P2-5).
                if other_source_rows:
                    other_sources: dict[str, int] = {}
                    for r in other_source_rows:
                        src = r["source"]
                        other_sources[src] = other_sources.get(src, 0) + 1
                    print(f"\nNon-source_model labels (reported separately): {other_sources}")

                # Print positive weak labels
                print("\nPositive weak labels (source_model, label=True, confidence_band=high):")
                print(f"{'identity_id':<30} {'attribute':<25} {'score':<10} {'source':<15}")
                print("-" * 80)
                for row in positive_rows[:limit]:
                    print(f"{row['identity_id']:<30} {row['attribute']:<25} {row['score']:<10.4f} {row['source']:<15}")

                # Print negative weak labels
                print("\nNegative weak labels (source_model, label=False, confidence_band=high):")
                print(f"{'identity_id':<30} {'attribute':<25} {'score':<10} {'source':<15}")
                print("-" * 80)
                for row in negative_rows[:limit]:
                    print(f"{row['identity_id']:<30} {row['attribute']:<25} {row['score']:<10.4f} {row['source']:<15}")

                print(f"\nSummary: {len(positive_rows)} positive, {len(negative_rows)} negative, "
                      f"{len(other_source_rows)} other-source weak labels")
                return positive_rows[:limit], negative_rows[:limit]
            else:
                print("WARNING: expected long-format columns (attribute, score) not found in parquet")
        except (OSError, ValueError, ImportError) as exc:
            print(f"ERROR: could not load parquet: {exc}")
    else:
        print(f"Parquet not found: {parquet_path}")
        print("Skipping weak label audit (run 'build annotate' first)")

    return [], []


def audit_tiny_smoke_probes(dataset: str, output_dir: Path | None = None) -> list[dict]:
    """Audit all tiny-smoke conflict probes."""
    print(f"\n{'='*72}")
    print(f"AUDIT: Tiny-Smoke Conflict Probes for {dataset.upper()}")
    print(f"{'='*72}")

    # P2-7: resolve route-probe artifact consistently via export manifest.
    probe_path = _resolve_artifact_path(output_dir, dataset, "route_probes", "route_probes.jsonl")

    if probe_path and probe_path.exists():
        print(f"Loading route probes from {probe_path}")
        probes = []
        with open(probe_path) as f:
            for line in f:
                if line.strip():
                    probes.append(json.loads(line))
        
        print(f"Loaded {len(probes)} route probes")
        
        # Group by probe_family
        families: dict[str, list] = {}
        for p in probes:
            fam = p.get("probe_family", "unknown")
            families.setdefault(fam, []).append(p)
        
        print("\nProbe families:")
        for fam, items in sorted(families.items()):
            print(f"  {fam}: {len(items)} probes")
        
        # Audit each family
        audit_rows = []
        for fam, items in families.items():
            print(f"\n--- {fam} probes ---")
            for p in items[:5]:  # show first 5 of each family
                row = {
                    "probe_id": p.get("probe_id", "unknown"),
                    "probe_family": fam,
                    "identity_id": p.get("identity_id", "unknown"),
                    "answer_label": p.get("answer_label"),
                    "answer_text": p.get("answer_text"),
                    "image_uri": p.get("image_uri"),
                }
                audit_rows.append(row)
                print(f"  {row['probe_id']}: identity={row['identity_id']}, answer_label={row['answer_label']}")
        
        return audit_rows
    else:
        print(f"Route probes not found: {probe_path}")
        print("Skipping probe audit (run 'build route-probes' first)")
        return []


def audit_tiny_smoke_pairs(
    dataset: str, output_dir: Path | None = None,
) -> tuple[list[dict], list[str]]:
    """Audit all tiny-smoke cross-image state pairs.

    P2-3: call production ``validate_pair_manifest()`` for semantic checks.
    P2-4: resolve pair IDs to canonical samples from ``<dataset>_processed.jsonl``.

    Returns ``(audit_rows, production_issues)``.
    """
    print(f"\n{'='*72}")
    print(f"AUDIT: Tiny-Smoke Cross-Image State Pairs for {dataset.upper()}")
    print(f"{'='*72}")

    # Try to load pairs manifest.
    pairs_path = None
    if output_dir:
        pairs_path = output_dir / f"{dataset}_pair_manifest.json"

    if not pairs_path or not pairs_path.exists():
        print(f"Pairs manifest not found: {pairs_path}")
        print("Skipping pairs audit (run 'build route-probes' first)")
        return [], []

    print(f"Loading pairs from {pairs_path}")
    pairs = json.loads(pairs_path.read_text())
    if not isinstance(pairs, list):
        print(f"WARNING: pair manifest is not a list, got {type(pairs).__name__}")
        pairs = []

    print(f"Loaded {len(pairs)} pairs")

    # P2-4: load processed samples to resolve sample IDs → canonical data.
    samples_by_id: dict[str, dict] = {}
    processed_path = output_dir / f"{dataset}_processed.jsonl" if output_dir else None
    if processed_path and processed_path.exists():
        try:
            for line in processed_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    sid = row.get("source_sample_id", "")
                    if sid:
                        samples_by_id[sid] = row
            print(f"Loaded {len(samples_by_id)} processed samples for pair resolution")
        except Exception as exc:
            print(f"WARNING: could not load processed samples: {exc}")

    # P2-3: run production pair validation if we have processed samples.
    production_issues: list[str] = []
    if samples_by_id:
        from route_data.build.conflict_generation import validate_pair_manifest
        # Build CanonicalSample-like objects for the validator.  The validator
        # accesses .identity_id, .image_uri, and .visual_attributes; processed
        # JSONL rows are plain dicts, so we wrap them.
        class _DictSample:
            def __init__(self, d: dict):
                self._d = d
                self.identity_id = d.get("identity_id", "")
                self.image_uri = d.get("image_uri", "")
                self.visual_attributes = d.get("visual_attributes", {})
        csamples = {sid: _DictSample(d) for sid, d in samples_by_id.items()}
        production_issues = validate_pair_manifest(pairs, csamples)
        if production_issues:
            print(f"\nProduction validation issues ({len(production_issues)}):")
            for iss in production_issues[:10]:
                print(f"  ⚠ {iss}")
        else:
            print("✓ production pair validation: all clean")

    # Filter cross-image pairs for display.
    cross_image_pairs = [p for p in pairs if p.get("pair_type") == "cross_image_attribute_state"]
    print(f"Cross-image attribute-state pairs: {len(cross_image_pairs)}")

    audit_rows = []
    for p in pairs:
        left_id = p.get("left_sample_id", "")
        right_id = p.get("right_sample_id", "")
        left_sample = samples_by_id.get(left_id, {})
        right_sample = samples_by_id.get(right_id, {})

        row = {
            "pair_id": p.get("pair_id", "unknown"),
            "pair_type": p.get("pair_type"),
            "left_sample_id": left_id,
            "right_sample_id": right_id,
            "attribute": p.get("attribute"),
            "controlled": p.get("controlled"),
            "changed": p.get("changed"),
            "expected_route_effect": p.get("expected_route_effect"),
            "left_label": p.get("left_label"),
            "right_label": p.get("right_label"),
            # P2-4: resolved canonical data.
            "left_image_uri": left_sample.get("image_uri"),
            "right_image_uri": right_sample.get("image_uri"),
            "left_identity_id": left_sample.get("identity_id"),
            "right_identity_id": right_sample.get("identity_id"),
            "left_target_state": p.get("left_label"),
            "right_target_state": p.get("right_label"),
        }
        audit_rows.append(row)

        # Report per-pair status.
        pair_issues = [iss for iss in production_issues if iss.startswith(f"{row['pair_id']}:")]
        if pair_issues:
            for iss in pair_issues:
                print(f"  ⚠ {iss}")
        elif row.get("attribute"):
            print(f"  ✓ {row['pair_id']}: {row['attribute']} type={row['pair_type']}")
        else:
            print(f"  ✓ {row['pair_id']}: type={row['pair_type']}")

    return audit_rows, production_issues


def audit_tiny_smoke_facts(dataset: str, output_dir: Path | None = None) -> list[dict]:
    """Audit all tiny-smoke name-only facts."""
    print(f"\n{'='*72}")
    print(f"AUDIT: Tiny-Smoke Name-Only Facts for {dataset.upper()}")
    print(f"{'='*72}")

    # P2-7: resolve route-probe artifact consistently via export manifest.
    probe_path = _resolve_artifact_path(output_dir, dataset, "route_probes", "route_probes.jsonl")

    if probe_path and probe_path.exists():
        print(f"Loading route probes from {probe_path}")
        probes = []
        with open(probe_path) as f:
            for line in f:
                if line.strip():
                    probes.append(json.loads(line))
        
        # Filter name_only probes
        name_only_probes = [p for p in probes if p.get("probe_family") == "name_only"]
        print(f"Name-only facts: {len(name_only_probes)}")
        
        audit_rows = []
        for p in name_only_probes:
            # P2-3: name_only uses answer_label / answer_text / target_fact_id /
            # target_fact_value instead of obsolete expected_answer / fact_text.
            row = {
                "probe_id": p.get("probe_id", "unknown"),
                "identity_id": p.get("identity_id", "unknown"),
                "answer_label": p.get("answer_label"),
                "answer_text": p.get("answer_text"),
                "target_fact_id": p.get("target_fact_id"),
                "target_fact_value": p.get("target_fact_value"),
            }
            audit_rows.append(row)

            # Verify: name-only facts should have a target_fact_value or answer_text.
            has_answer = row["answer_text"] or row["answer_label"]
            has_fact = row["target_fact_value"]
            if has_answer and has_fact:
                print(f"  ✓ {row['probe_id']}: identity={row['identity_id']}, "
                      f"fact={row['target_fact_id']}, answer={row['answer_label']}")
            else:
                missing = []
                if not has_answer:
                    missing.append("answer_label/answer_text")
                if not has_fact:
                    missing.append("target_fact_value")
                print(f"  ⚠ {row['probe_id']}: missing {', '.join(missing)}")
        
        return audit_rows
    else:
        print(f"Route probes not found: {probe_path}")
        print("Skipping facts audit (run 'build route-probes' first)")
        return []


def _build_audit_items(
    source_mappings: list[dict],
    pos_labels: list[dict],
    neg_labels: list[dict],
    probes: list[dict],
    pairs: list[dict],
    facts: list[dict],
    failures: list[str],
) -> list[dict]:
    """P2-6: convert raw audit data into structured review items.

    Each item has: audit_id, category, sample_id, identity_id, image_uri,
    attribute_or_fact, automatic_checks, review_outcome (unreviewed by
    default), review_note.  Human review is NOT pre-marked as passed.
    """
    items: list[dict] = []
    failure_set = set(failures)
    counter = 0

    def _next_id(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}-{counter:04d}"

    # Source-mapping items.
    for row in source_mappings:
        iid = row.get("identity_id", "unknown")
        auto: dict = {}
        if row.get("target_split") == "hash":
            auto["unassigned_to_hash"] = True
        items.append({
            "audit_id": _next_id("src"),
            "category": "source_mapping",
            "sample_id": None,
            "identity_id": iid,
            "image_uri": row.get("image_uri"),
            "attribute_or_fact": f"split={row.get('raw_split')}",
            "automatic_checks": auto,
            "review_outcome": "unreviewed",
            "review_note": f"raw_split={row.get('raw_split')}, target={row.get('target_split')}",
        })

    # Positive weak-label items.
    for row in pos_labels:
        items.append({
            "audit_id": _next_id("wl_pos"),
            "category": "weak_label",
            "sample_id": None,
            "identity_id": row.get("identity_id", "unknown"),
            "image_uri": row.get("image_uri"),
            "attribute_or_fact": row.get("attribute", ""),
            "automatic_checks": {"score": row.get("score"), "source": row.get("source")},
            "review_outcome": "unreviewed",
            "review_note": f"score={row.get('score', 0):.4f}, label=True",
        })

    # Negative weak-label items.
    for row in neg_labels:
        items.append({
            "audit_id": _next_id("wl_neg"),
            "category": "weak_label",
            "sample_id": None,
            "identity_id": row.get("identity_id", "unknown"),
            "image_uri": row.get("image_uri"),
            "attribute_or_fact": row.get("attribute", ""),
            "automatic_checks": {"score": row.get("score"), "source": row.get("source")},
            "review_outcome": "unreviewed",
            "review_note": f"score={row.get('score', 0):.4f}, label=False",
        })

    # Conflict-probe items (P2-3: use answer_label / answer_text).
    for row in probes:
        pid = row.get("probe_id", "unknown")
        auto = {}
        if any(pid in f for f in failure_set):
            auto["flagged"] = True
        items.append({
            "audit_id": _next_id("probe"),
            "category": "route_probe",
            "sample_id": pid,
            "identity_id": row.get("identity_id", "unknown"),
            "image_uri": row.get("image_uri"),
            "attribute_or_fact": row.get("probe_family", ""),
            "automatic_checks": auto,
            "review_outcome": "unreviewed",
            "review_note": f"answer_label={row.get('answer_label')}, answer_text={row.get('answer_text')}",
        })

    # Cross-image pair items (P2-4: use production pair fields).
    for row in pairs:
        pid = row.get("pair_id", "unknown")
        auto: dict = {}
        if row.get("pair_type") and row.get("controlled") and row.get("changed"):
            auto["schema_valid"] = True
        else:
            auto["schema_valid"] = False
        items.append({
            "audit_id": _next_id("pair"),
            "category": "pair",
            "sample_id": pid,
            "identity_id": None,
            "image_uri": None,
            "attribute_or_fact": row.get("attribute", ""),
            "automatic_checks": auto,
            "review_outcome": "unreviewed",
            "review_note": (
                f"type={row.get('pair_type')}, "
                f"left={row.get('left_sample_id')}, right={row.get('right_sample_id')}, "
                f"labels={row.get('left_label')}→{row.get('right_label')}"
            ),
        })

    # Name-only fact items (P2-3: use answer_label / target_fact_id / target_fact_value).
    for row in facts:
        pid = row.get("probe_id", "unknown")
        auto: dict = {}
        has_answer = row.get("answer_text") or row.get("answer_label")
        has_fact = row.get("target_fact_value")
        if not has_answer:
            auto["missing_answer"] = True
        if not has_fact:
            auto["missing_fact"] = True
        items.append({
            "audit_id": _next_id("fact"),
            "category": "route_probe",
            "sample_id": pid,
            "identity_id": row.get("identity_id", "unknown"),
            "image_uri": None,
            "attribute_or_fact": row.get("target_fact_id", ""),
            "automatic_checks": auto,
            "review_outcome": "unreviewed",
            "review_note": (
                f"fact={row.get('target_fact_id')}, value={row.get('target_fact_value')}, "
                f"answer={row.get('answer_label')}"
            ),
        })

    return items


# P2-10: audit report schema version and required fields.
AUDIT_REPORT_VERSION = "v1"
AUDIT_ITEM_REQUIRED_FIELDS = {
    "audit_id", "category", "sample_id", "identity_id", "image_uri",
    "attribute_or_fact", "automatic_checks", "review_outcome", "review_note",
}
VALID_REVIEW_OUTCOMES = {"pass", "uncertain", "fail", "unreviewed"}


def _validate_audit_schema(report: dict) -> list[str]:
    """P2-10: validate audit report schema before accepting as pilot evidence.

    Returns a list of issues (empty == valid).
    """
    issues: list[str] = []
    if report.get("audit_report_version") != AUDIT_REPORT_VERSION:
        issues.append(f"missing or wrong audit_report_version (expected {AUDIT_REPORT_VERSION})")
    items = report.get("items")
    if not isinstance(items, list):
        issues.append("'items' is not a list")
        return issues
    for idx, item in enumerate(items):
        missing = AUDIT_ITEM_REQUIRED_FIELDS - set(item.keys())
        if missing:
            issues.append(f"item {idx} ({item.get('audit_id', '?')}): missing fields {missing}")
        outcome = item.get("review_outcome")
        if outcome not in VALID_REVIEW_OUTCOMES:
            issues.append(f"item {idx} ({item.get('audit_id', '?')}): invalid review_outcome={outcome!r}")
    return issues


def _persist_audit_report(
    dataset: str,
    output_dir: Path | None,
    items: list[dict],
    failures: list[str],
) -> Path | None:
    """P2-9/P2-10: write ``<dataset>_manual_audit_report.json``.

    The report version is ``v1``.  All items default to ``review_outcome='unreviewed'``;
    the gate passes only when there are zero unreviewed items and zero critical
    failures (P2-9).  Schema validation is performed before persisting (P2-10).
    """
    if output_dir is None:
        return None

    unreviewed_count = sum(1 for it in items if it["review_outcome"] == "unreviewed")
    critical_count = sum(1 for it in items if it["review_outcome"] == "fail")

    report: dict = {
        "audit_report_version": AUDIT_REPORT_VERSION,
        "dataset": dataset,
        "total_items": len(items),
        "unreviewed_items": unreviewed_count,
        "critical_failures": critical_count,
        "gate_pass": len(failures) == 0 and unreviewed_count == 0 and critical_count == 0,
        "items": items,
        "summary": {
            "source_mappings": sum(1 for it in items if it["category"] == "source_mapping"),
            "weak_label": sum(1 for it in items if it["category"] == "weak_label"),
            "route_probe": sum(1 for it in items if it["category"] == "route_probe"),
            "pair": sum(1 for it in items if it["category"] == "pair"),
        },
    }

    # P2-10: validate schema before persisting.
    schema_issues = _validate_audit_schema(report)
    if schema_issues:
        print(f"\n--- WARNING: audit schema validation failed ({len(schema_issues)} issues)")
        for iss in schema_issues[:5]:
            print(f"    {iss}")
        failures.extend(f"audit schema: {iss}" for iss in schema_issues)

    report_path = output_dir / f"{dataset}_manual_audit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\n--- audit report persisted: {report_path}")
    print(f"    version={AUDIT_REPORT_VERSION}, total_items={len(items)}, "
          f"unreviewed={unreviewed_count}, critical={critical_count}, "
          f"gate={'PASS' if report['gate_pass'] else 'FAIL'}")
    return report_path


def run_full_audit(dataset: str, config: str | None = None, output_dir: Path | None = None) -> int:
    """Run the full audit gate and produce a structured report."""
    print(f"\n{'#'*72}")
    print(f"# R20 MANUAL AUDIT GATE: {dataset.upper()}")
    print(f"{'#'*72}")
    
    failures: list[str] = []
    
    # 1. Audit source mappings
    source_mappings = audit_source_mappings(dataset, limit=20)
    if not source_mappings:
        failures.append("source mappings: no samples found")
    
    # 1b. P1-9: audit wrong-name coverage using production eligibility logic.
    wrong_name_pairs = audit_wrong_name_coverage(dataset, limit=10)

    # 2. Audit weak labels
    pos_labels, neg_labels = audit_weak_labels(dataset, output_dir, limit=20)
    # Weak labels are optional (may not exist before annotation)
    
    # 3. Audit tiny-smoke conflict probes
    probes = audit_tiny_smoke_probes(dataset, output_dir)
    # Probes are optional (may not exist before route-probes stage)
    
    # 4. Audit tiny-smoke cross-image state pairs
    pairs, pair_production_issues = audit_tiny_smoke_pairs(dataset, output_dir)
    # P2-3: production pair validation issues are failures.
    for iss in pair_production_issues:
        failures.append(f"pair validation: {iss}")
    # Verify: cross-image pairs should have different sample IDs on each side.
    for p in pairs:
        if p.get("left_sample_id") and p["left_sample_id"] == p.get("right_sample_id"):
            failures.append(f"cross-image pair {p['pair_id']}: same sample_id on both sides")

    # 5. Audit tiny-smoke name-only facts
    facts = audit_tiny_smoke_facts(dataset, output_dir)
    # Verify: name-only facts should have answer and fact data (P2-3).
    for f in facts:
        has_answer = f.get("answer_text") or f.get("answer_label")
        has_fact = f.get("target_fact_value")
        if not has_answer:
            failures.append(f"name-only fact {f['probe_id']}: missing answer_label/answer_text")
        if not has_fact:
            failures.append(f"name-only fact {f['probe_id']}: missing target_fact_value")
    
    # P2-10: build structured audit items and persist report.
    audit_items = _build_audit_items(
        source_mappings, pos_labels, neg_labels, probes, pairs, facts, failures,
    )
    _persist_audit_report(dataset, output_dir, audit_items, failures)

    # P2-9: gate requires zero failures, zero unreviewed, zero critical.
    unreviewed_count = sum(1 for it in audit_items if it["review_outcome"] == "unreviewed")
    critical_count = sum(1 for it in audit_items if it["review_outcome"] == "fail")
    gate_pass = len(failures) == 0 and unreviewed_count == 0 and critical_count == 0

    # Summary
    print(f"\n{'#'*72}")
    print("# AUDIT SUMMARY")
    print(f"{'#'*72}")
    print(f"Source mappings inspected: {len(source_mappings)}")
    print(f"Wrong-name candidate pairs: {len(wrong_name_pairs)}")
    print(f"Positive weak labels inspected: {len(pos_labels)}")
    print(f"Negative weak labels inspected: {len(neg_labels)}")
    print(f"Conflict probes inspected: {len(probes)}")
    print(f"Cross-image pairs inspected: {len(pairs)}")
    print(f"  production validation issues: {len(pair_production_issues)}")
    print(f"Name-only facts inspected: {len(facts)}")
    print(f"Total audit items: {len(audit_items)}")
    print(f"Unreviewed items: {unreviewed_count}")
    print(f"Critical failures: {critical_count}")
    print(f"Automatic check failures: {len(failures)}")
    
    if gate_pass:
        print("\nAUDIT PASSED: all checks passed, all items reviewed")
        return 0
    else:
        reasons: list[str] = []
        if failures:
            reasons.append(f"{len(failures)} automatic failure(s)")
        if unreviewed_count > 0:
            reasons.append(f"{unreviewed_count} unreviewed item(s)")
        if critical_count > 0:
            reasons.append(f"{critical_count} critical failure(s)")
        print(f"\nAUDIT FAILED: {', '.join(reasons)}")
        for f in failures[:20]:
            print(f"  - {f}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R20: manual audit gate for pilot/full generation")
    parser.add_argument("--dataset", required=True, help="Benchmark name (e.g., fairget, fiubench, mllmu, ppubench)")
    parser.add_argument("--config", help="Run config YAML path")
    parser.add_argument("--output-dir", type=Path, help="Output directory containing generated artifacts")
    args = parser.parse_args()
    
    sys.exit(run_full_audit(dataset=args.dataset, config=args.config, output_dir=args.output_dir))
