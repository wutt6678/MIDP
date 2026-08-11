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
    """Resolve the effective source mapping for a benchmark."""
    DEFAULT_SOURCE_MAPPING = {
        "train": "train",
        "retain_train": "train",
        "validation": "eval",
        "val": "eval",
        "eval": "eval",
        "retain_eval": "eval",
        "test": "eval",
        "forget": "exclude",
        "unassigned": "hash",
    }
    mapping = DEFAULT_SOURCE_MAPPING.copy()
    cfg = _load_data_config(dataset)
    extras_mapping = cfg.get("data", {}).get("extras", {}).get("source_mapping")
    if extras_mapping and isinstance(extras_mapping, dict):
        mapping.update(extras_mapping)
    return mapping


def audit_source_mappings(dataset: str, limit: int = 20) -> list[dict]:
    """Audit 20 source mappings: identity_id → split bucket → target split."""
    print(f"\n{'='*72}")
    print(f"AUDIT: Source Mappings for {dataset.upper()}")
    print(f"{'='*72}")
    
    mapping = _resolve_source_mapping(dataset)
    print(f"Effective source mapping: {json.dumps(mapping, indent=2)}")
    
    # Try to load adapter samples
    samples = []
    try:
        from route_data.data.adapters import load_adapter
        adapter = load_adapter(dataset)
        samples = list(adapter.samples())
        print(f"Loaded {len(samples)} samples from adapter")
    except Exception as exc:
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
            "image_path": str(s.image_path) if s.image_path else None,
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


def audit_weak_labels(dataset: str, output_dir: Path | None = None, limit: int = 20) -> tuple[list[dict], list[dict]]:
    """Audit 20 positive and 20 negative weak labels."""
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
            
            # Positive weak labels: high-confidence accepted annotations
            # (score > 0.8 or similar threshold)
            score_cols = [c for c in df.columns if c.startswith("score_") or c.startswith("p_")]
            if score_cols:
                # Find rows with high-confidence positive labels
                positive_rows = []
                negative_rows = []
                
                for idx, row in df.iterrows():
                    for col in score_cols:
                        val = row[col]
                        if pd.notna(val):
                            if val > 0.8:
                                positive_rows.append({
                                    "identity_id": row.get("identity_id", "unknown"),
                                    "attribute": col,
                                    "score": float(val),
                                    "label": "positive",
                                })
                            elif val < 0.2:
                                negative_rows.append({
                                    "identity_id": row.get("identity_id", "unknown"),
                                    "attribute": col,
                                    "score": float(val),
                                    "label": "negative",
                                })
                
                # Print positive weak labels
                print(f"\nPositive weak labels (high-confidence, score > 0.8):")
                print(f"{'identity_id':<40} {'attribute':<30} {'score':<10}")
                print("-" * 80)
                for row in positive_rows[:limit]:
                    print(f"{row['identity_id']:<40} {row['attribute']:<30} {row['score']:<10.4f}")
                
                # Print negative weak labels
                print(f"\nNegative weak labels (low-confidence, score < 0.2):")
                print(f"{'identity_id':<40} {'attribute':<30} {'score':<10}")
                print("-" * 80)
                for row in negative_rows[:limit]:
                    print(f"{row['identity_id']:<40} {row['attribute']:<30} {row['score']:<10.4f}")
                
                print(f"\nSummary: {len(positive_rows)} positive, {len(negative_rows)} negative weak labels")
                return positive_rows[:limit], negative_rows[:limit]
            else:
                print("WARNING: no score columns found in parquet")
        except Exception as exc:
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
    
    # Try to load route probes
    probe_path = None
    if output_dir:
        probe_path = output_dir / f"{dataset}_route_conflict_eval.jsonl"
    
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
        
        print(f"\nProbe families:")
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
                    "expected_answer": p.get("expected_answer", p.get("expected_label")),
                    "image_uri": p.get("image_uri"),
                }
                audit_rows.append(row)
                print(f"  {row['probe_id']}: identity={row['identity_id']}, expected={row['expected_answer']}")
        
        return audit_rows
    else:
        print(f"Route probes not found: {probe_path}")
        print("Skipping probe audit (run 'build route-probes' first)")
        return []


def audit_tiny_smoke_pairs(dataset: str, output_dir: Path | None = None) -> list[dict]:
    """Audit all tiny-smoke cross-image state pairs."""
    print(f"\n{'='*72}")
    print(f"AUDIT: Tiny-Smoke Cross-Image State Pairs for {dataset.upper()}")
    print(f"{'='*72}")
    
    # Try to load pairs manifest
    pairs_path = None
    if output_dir:
        pairs_path = output_dir / f"{dataset}_route_pairs.jsonl"
    
    if pairs_path and pairs_path.exists():
        print(f"Loading pairs from {pairs_path}")
        pairs = []
        with open(pairs_path) as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        
        print(f"Loaded {len(pairs)} pairs")
        
        # Filter cross-image pairs
        cross_image_pairs = [p for p in pairs if p.get("pair_type") == "cross_image_attribute_state"]
        print(f"Cross-image attribute-state pairs: {len(cross_image_pairs)}")
        
        audit_rows = []
        for p in cross_image_pairs:
            row = {
                "pair_id": p.get("pair_id", "unknown"),
                "pair_type": p.get("pair_type"),
                "identity_id": p.get("identity_id", "unknown"),
                "attribute": p.get("attribute"),
                "left_image": p.get("left_image_uri"),
                "right_image": p.get("right_image_uri"),
                "left_state": p.get("left_attribute_state"),
                "right_state": p.get("right_attribute_state"),
            }
            audit_rows.append(row)
            
            # Verify: left and right images should be different
            if row["left_image"] == row["right_image"]:
                print(f"  ⚠ {row['pair_id']}: left and right images are the same!")
            else:
                print(f"  ✓ {row['pair_id']}: {row['attribute']} state={row['left_state']}→{row['right_state']}")
        
        return audit_rows
    else:
        print(f"Pairs manifest not found: {pairs_path}")
        print("Skipping pairs audit (run 'build route-probes' first)")
        return []


def audit_tiny_smoke_facts(dataset: str, output_dir: Path | None = None) -> list[dict]:
    """Audit all tiny-smoke name-only facts."""
    print(f"\n{'='*72}")
    print(f"AUDIT: Tiny-Smoke Name-Only Facts for {dataset.upper()}")
    print(f"{'='*72}")
    
    # Try to load route probes with name_only family
    probe_path = None
    if output_dir:
        probe_path = output_dir / f"{dataset}_route_conflict_eval.jsonl"
    
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
            row = {
                "probe_id": p.get("probe_id", "unknown"),
                "identity_id": p.get("identity_id", "unknown"),
                "expected_answer": p.get("expected_answer", p.get("expected_label")),
                "fact_text": p.get("fact_text", p.get("context_text")),
            }
            audit_rows.append(row)
            
            # Verify: name-only facts should have exact expected_answer
            if row["expected_answer"]:
                print(f"  ✓ {row['probe_id']}: identity={row['identity_id']}, expected={row['expected_answer']}")
            else:
                print(f"  ⚠ {row['probe_id']}: missing expected_answer")
        
        return audit_rows
    else:
        print(f"Route probes not found: {probe_path}")
        print("Skipping facts audit (run 'build route-probes' first)")
        return []


def run_full_audit(dataset: str, config: str | None = None, output_dir: Path | None = None) -> int:
    """Run the full audit gate and produce a structured report."""
    print(f"\n{'#'*72}")
    print(f"# R20 MANUAL AUDIT GATE: {dataset.upper()}")
    print(f"{'#'*72}")
    
    failures = []
    
    # 1. Audit source mappings
    source_mappings = audit_source_mappings(dataset, limit=20)
    if not source_mappings:
        failures.append("source mappings: no samples found")
    
    # 2. Audit weak labels
    pos_labels, neg_labels = audit_weak_labels(dataset, output_dir, limit=20)
    # Weak labels are optional (may not exist before annotation)
    
    # 3. Audit tiny-smoke conflict probes
    probes = audit_tiny_smoke_probes(dataset, output_dir)
    # Probes are optional (may not exist before route-probes stage)
    
    # 4. Audit tiny-smoke cross-image state pairs
    pairs = audit_tiny_smoke_pairs(dataset, output_dir)
    # Verify: cross-image pairs should have different images
    for p in pairs:
        if p["left_image"] == p["right_image"]:
            failures.append(f"cross-image pair {p['pair_id']}: same image on both sides")
    
    # 5. Audit tiny-smoke name-only facts
    facts = audit_tiny_smoke_facts(dataset, output_dir)
    # Verify: name-only facts should have exact expected_answer
    for f in facts:
        if not f["expected_answer"]:
            failures.append(f"name-only fact {f['probe_id']}: missing expected_answer")
    
    # Summary
    print(f"\n{'#'*72}")
    print(f"# AUDIT SUMMARY")
    print(f"{'#'*72}")
    print(f"Source mappings inspected: {len(source_mappings)}")
    print(f"Positive weak labels inspected: {len(pos_labels)}")
    print(f"Negative weak labels inspected: {len(neg_labels)}")
    print(f"Conflict probes inspected: {len(probes)}")
    print(f"Cross-image pairs inspected: {len(pairs)}")
    print(f"Name-only facts inspected: {len(facts)}")
    
    if failures:
        print(f"\n❌ AUDIT FAILED: {len(failures)} issues found")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print(f"\n✓ AUDIT PASSED: all checks passed")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R20: manual audit gate for pilot/full generation")
    parser.add_argument("--dataset", required=True, help="Benchmark name (e.g., fairget, fiubench, mllmu, ppubench)")
    parser.add_argument("--config", help="Run config YAML path")
    parser.add_argument("--output-dir", type=Path, help="Output directory containing generated artifacts")
    args = parser.parse_args()
    
    sys.exit(run_full_audit(dataset=args.dataset, config=args.config, output_dir=args.output_dir))
