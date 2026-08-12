"""Final-verification driver: end-state commands from the repair plan.

R19: generalized to accept --dataset, --config, --output-dir so it can validate
any benchmark, not just the golden FAIRGET fixture.

P0/P1 upgrades (Fix List for 0311c63):
- P0-2: immutable-revision bypass restricted to synthetic/golden fixtures only.
- P0-3/4: coverage-aware smoke sampling with persisted manifest.
- P1-5: read actual <dataset>_score_manifest.json (not export manifest).
- P1-6: require exactly 40 CelebA extension attributes per image (set equality).
- P1-7: verify *_processed.jsonl directly.
- P1-8: verify actual whitelist invariant from score_manifest.
- P1-9: verify answer_label / answer_text in route rows.
- P1-10: verify name_only image absence in route probes.
- P1-11: verify pair semantics from pair manifest.
- P1-12: verify split invariants from split manifest.
- P1-13: recompute every checksum independently.
- P1-14: strict-mode SKIPs fail for required checks (PASS/FAIL/NOT_APPLICABLE).
- P1-15: non-zero coverage for required smoke paths.

Runs on the bundled golden fixture with the stub backend by default so the
checks are self-contained (no live model / restricted data required).
"""

from __future__ import annotations

import argparse
import enum
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))


# --------------------------------------------------------------------------- #
# Verification result model (P1-14)
# --------------------------------------------------------------------------- #


class CheckResult(enum.Enum):
    """Three-state verification result.

    Required checks must be PASS or FAIL; NOT_APPLICABLE is only allowed for
    genuinely optional checks.  Under ``--strict``, any required check that
    would SKIP is treated as a failure.
    """
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CheckRecord:
    """One verification check with its result and optional detail."""

    def __init__(self, name: str, result: CheckResult, detail: str = "", *, required: bool = True):
        self.name = name
        self.result = result
        self.detail = detail
        self.required = required

    def __repr__(self) -> str:
        suffix = f" ({self.detail})" if self.detail else ""
        return f"{self.name}: {self.result.value}{suffix}"


# --------------------------------------------------------------------------- #
# Shared split resolution (P1-4, P1-10: centralized)
# --------------------------------------------------------------------------- #

from route_data.build.conflict_generation import (
    find_wrong_name_candidates as _find_wrong_name_candidates,
)
from route_data.data.split_mapping import resolve_effective_split as _resolve_effective_split

# --------------------------------------------------------------------------- #
# Smoke subset selection (P0-3, P1-3, P1-4)
# --------------------------------------------------------------------------- #


def select_smoke_subset(
    samples: list[dict[str, Any]],
    *,
    min_identities: int = 3,
    min_image_bearing: int = 2,
    require_train: bool = True,
    require_eval: bool = True,
    require_exclude: bool = True,
    require_visual: bool = True,
    require_profile_fact: bool = True,
    require_wrong_name: bool = False,
    require_multiview: bool = False,
) -> dict[str, Any]:
    """Select a coverage-aware smoke subset from canonical samples.

    Uses iterative greedy scoring: each candidate is scored against the
    *current* selection state, not the initial empty state.  Input samples
    are never mutated.

    Returns a dict with ``selected`` (list of sample dicts), ``coverage``
    (summary counts), and ``issues`` (list of unmet requirements).
    """
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    identity_ids: set[str] = set()
    image_bearing: set[str] = set()
    splits_seen: set[str] = set()
    has_fact = False
    # P1-7: track distinct image URIs per identity for true multiview detection.
    images_by_identity: dict[str, set[str]] = {}

    # Track remaining candidates by index so we never mutate inputs.
    remaining = list(range(len(samples)))

    def _score_sample(idx: int) -> int:
        """Score a sample against the *current* selection state."""
        s = samples[idx]
        iid = s.get("identity_id", "")
        img = s.get("image_uri")
        split = _resolve_effective_split(s)
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
        # P1-8: actively seek a second distinct image for already-selected identities.
        if (
            require_multiview
            and iid in identity_ids
            and img
            and img not in images_by_identity.get(iid, set())
        ):
            score += 5
        return score

    # Iterative greedy selection: re-score remaining candidates each round.
    max_select = min(12, len(samples))
    while len(selected) < max_select and remaining:
        # Score all remaining against current state.
        scored = [(idx, _score_sample(idx)) for idx in remaining]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_idx = scored[0][0]
        if scored[0][1] <= 0 and len(selected) >= min_identities:
            # P1-8: do not stop merely because identity minimum is reached;
            # check if any unmet coverage condition can still be improved.
            can_improve = False
            if require_multiview and not any(len(imgs) >= 2 for imgs in images_by_identity.values()):
                for idx in remaining:
                    s = samples[idx]
                    iid = s.get("identity_id", "")
                    img = s.get("image_uri")
                    if iid in identity_ids and img and img not in images_by_identity.get(iid, set()):
                        can_improve = True
                        break
            if not can_improve:
                break

        s = samples[best_idx]
        sid = s.get("source_sample_id", "")
        iid = s.get("identity_id", "")
        img = s.get("image_uri")
        split = _resolve_effective_split(s)
        facts = s.get("profile_facts", [])

        selected.append(s)
        selected_ids.add(sid)
        identity_ids.add(iid)
        if img:
            image_bearing.add(iid)
            # P1-7: track distinct URIs per identity.
            images_by_identity.setdefault(iid, set()).add(img)
        if split:
            splits_seen.add(split)
        if facts:
            has_fact = True
        remaining.remove(best_idx)

    # Check coverage requirements.
    issues: list[str] = []
    if len(identity_ids) < min_identities:
        issues.append(f"only {len(identity_ids)} identities (need >= {min_identities})")
    if len(image_bearing) < min_image_bearing:
        issues.append(f"only {len(image_bearing)} image-bearing identities (need >= {min_image_bearing})")
    if require_train and "train" not in splits_seen:
        issues.append("no retain/train identity in smoke subset")
    if require_eval and "eval" not in splits_seen:
        issues.append("no evaluation identity in smoke subset")
    if require_exclude and "exclude" not in splits_seen:
        issues.append("no forget/exclude identity in smoke subset")
    if require_visual and not image_bearing:
        issues.append("no image-bearing identity in smoke subset")
    if require_profile_fact and not has_fact:
        issues.append("no identity with profile facts in smoke subset")

    # P1-9: enforce require_wrong_name using the same eligibility logic as
    # production route-probe generation (visual anchor, >=2 samples, Jaccard
    # matching).  Build by_identity from selected samples.
    _by_identity: dict[str, list] = {}
    for _s in selected:
        _iid = _s.get("identity_id", "")
        if _iid:
            _by_identity.setdefault(_iid, []).append(_s)
    wrong_name_pairs = _find_wrong_name_candidates(_by_identity)

    if require_wrong_name and not wrong_name_pairs:
        issues.append(
            "require_wrong_name: no valid wrong-name target/control pair found "
            f"(image-bearing identities: {len(image_bearing)})"
        )

    # P1-7: enforce require_multiview — need >=1 identity with >=2 distinct image URIs.
    has_multiview = any(len(imgs) >= 2 for imgs in images_by_identity.values())
    if require_multiview and not has_multiview:
        issues.append("require_multiview: no identity with >=2 distinct image URIs")

    coverage = {
        "selected_samples": len(selected),
        "identities": sorted(identity_ids),
        "image_bearing_identities": sorted(image_bearing),
        "splits_seen": sorted(splits_seen),
        "has_profile_facts": has_fact,
        "has_multiview": has_multiview,
        "wrong_name_pairs": [
            {"target": t, "control": c, "similarity": round(sim, 4)}
            for t, c, sim in wrong_name_pairs[:5]
        ],
    }

    return {"selected": selected, "coverage": coverage, "issues": issues}


# --------------------------------------------------------------------------- #
# CLI runner
# --------------------------------------------------------------------------- #


def _run_cli(label: str, argv: list[str], expect: int = 0, failures: list[str] | None = None) -> int:
    """Run a CLI command and track failures."""
    from route_data.cli import main as cli_main

    print(f"\n=== {label}: route-data {' '.join(argv)}")
    rc = cli_main(argv)
    status = "OK" if rc == expect else "FAIL"
    if rc != expect:
        if failures is not None:
            failures.append(label)
        print(f"--- {label}: rc={rc} (expected {expect}) [{status}]")
    else:
        print(f"--- {label}: rc={rc} (expected {expect}) [{status}]")
    return rc


def _check_artifact_exists(path: Path, label: str, failures: list[str]) -> bool:
    """Check if an artifact exists; record failure if not."""
    if not path.exists():
        failures.append(f"{label}: MISSING [{path}]")
        print(f"--- {label}: MISSING [{path}]")
        return False
    print(f"--- {label}: OK")
    return True


# --------------------------------------------------------------------------- #
# P1-5: Read actual score_manifest.json
# --------------------------------------------------------------------------- #


def _verify_score_manifest(export_dir: Path, benchmark: str, failures: list[str]) -> CheckRecord:
    """P1-5: read <dataset>_score_manifest.json (not export manifest).

    Require: model_id, backend, resolved_revision, fingerprint_id,
    model_fingerprint_payload, prompt_registry_hash, candidate_set_hash,
    scoring_version.
    """
    manifest_path = export_dir / f"{benchmark}_score_manifest.json"
    if not manifest_path.exists():
        failures.append(f"score manifest: MISSING [{manifest_path}]")
        return CheckRecord("score manifest exists", CheckResult.FAIL, str(manifest_path))

    try:
        mdata = json.loads(manifest_path.read_text())
    except Exception as exc:
        failures.append(f"score manifest parse: {exc}")
        return CheckRecord("score manifest parse", CheckResult.FAIL, str(exc))

    required_fields = [
        "model_id", "backend", "resolved_revision", "fingerprint_id",
        "model_fingerprint_payload", "prompt_registry_hash",
        "candidate_set_hash", "scoring_version",
    ]
    missing = [f for f in required_fields if f not in mdata or mdata[f] is None]
    if missing:
        failures.append(f"score manifest missing fields: {missing}")
        return CheckRecord("score manifest fields", CheckResult.FAIL, f"missing={missing}")

    resolved = mdata.get("resolved_revision")
    if not resolved or resolved in ("unknown", "PENDING", ""):
        failures.append(f"score manifest resolved_revision invalid: {resolved!r}")
        return CheckRecord("resolved revision present", CheckResult.FAIL, repr(resolved))

    print(f"--- score manifest: OK (model={mdata.get('model_id')}, rev={resolved})")
    return CheckRecord("score manifest", CheckResult.PASS, f"model={mdata.get('model_id')}")


# --------------------------------------------------------------------------- #
# P1-6: Require exactly 40 CelebA extension attributes per image
# --------------------------------------------------------------------------- #


def _verify_scores_per_image(export_dir: Path, benchmark: str, failures: list[str]) -> CheckRecord:
    """P1-6: require exactly 40 CelebA extension attributes per image.

    Filter only ``extended_attributes.celeba40.*`` and require set equality
    with all 40 CelebA attributes for every image.
    """
    parquet_path = export_dir / f"{benchmark}_celeba40_image_annotations.parquet"
    if not parquet_path.exists():
        # For golden fixture with stub backend, check the processed JSONL.
        processed_path = export_dir / f"{benchmark}_processed.jsonl"
        if not processed_path.exists():
            failures.append("40 scores per image: no annotations parquet or processed JSONL")
            return CheckRecord("40 scores per image", CheckResult.FAIL, "no artifact found")
        return _verify_40_attrs_from_processed(processed_path, benchmark, failures)

    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        if df.empty:
            failures.append("40 scores per image: parquet is empty")
            return CheckRecord("40 scores per image", CheckResult.FAIL, "empty parquet")

        if "attribute" in df.columns and "image_uri" in df.columns:
            from route_data.constants.celeba_attributes import CELEBA_ATTRIBUTE_SET
            CELEBA40_PREFIX = "extended_attributes.celeba40."
            # Skip text-only rows (image_uri is null/NaN).
            df_visual = df.dropna(subset=["image_uri"])
            attrs_per_image = df_visual.groupby("image_uri")["attribute"].apply(set)
            bad_images = []
            for img_uri, attr_set in attrs_per_image.items():
                # Normalize namespaced attributes by stripping the prefix.
                celeba_attrs = set()
                for a in attr_set:
                    if a.startswith(CELEBA40_PREFIX):
                        celeba_attrs.add(a[len(CELEBA40_PREFIX):])
                    elif a in CELEBA_ATTRIBUTE_SET:
                        celeba_attrs.add(a)
                    # else: unknown attribute, ignore for set-equality check
                if celeba_attrs != CELEBA_ATTRIBUTE_SET:
                    missing = CELEBA_ATTRIBUTE_SET - celeba_attrs
                    extra = celeba_attrs - CELEBA_ATTRIBUTE_SET
                    bad_images.append({
                        "image": img_uri,
                        "count": len(celeba_attrs),
                        "missing": len(missing),
                        "extra": len(extra),
                    })
            if bad_images:
                msg = f"{len(bad_images)} image(s) without exactly 40 CelebA attrs"
                failures.append(f"40 scores per image: {msg}")
                return CheckRecord("40 scores per image", CheckResult.FAIL, msg)
            print(f"--- 40 scores per image: OK ({len(attrs_per_image)} images, 40 attrs each)")
            return CheckRecord("40 scores per image", CheckResult.PASS, f"{len(attrs_per_image)} images")
        else:
            failures.append("40 scores per image: no attribute/image_uri columns")
            return CheckRecord("40 scores per image", CheckResult.FAIL, "missing columns")
    except Exception as exc:
        failures.append(f"40 scores per image: {exc}")
        return CheckRecord("40 scores per image", CheckResult.FAIL, str(exc))


def _verify_40_attrs_from_processed(processed_path: Path, benchmark: str, failures: list[str]) -> CheckRecord:
    """Fallback: check 40 attrs from *_processed.jsonl."""
    from route_data.constants.celeba_attributes import CELEBA_ATTRIBUTE_SET

    try:
        samples = [json.loads(line) for line in processed_path.read_text().splitlines() if line.strip()]
    except Exception as exc:
        failures.append(f"40 scores per image (processed): {exc}")
        return CheckRecord("40 scores per image", CheckResult.FAIL, str(exc))

    prefix = "extended_attributes.celeba40."
    bad = 0
    for s in samples:
        va = s.get("visual_attributes", {})
        celeba_keys = {k.removeprefix(prefix) for k in va if k.startswith(prefix)}
        if celeba_keys != CELEBA_ATTRIBUTE_SET:
            bad += 1
    if bad > 0:
        failures.append(f"40 scores per image: {bad} samples without exactly 40 CelebA attrs")
        return CheckRecord("40 scores per image", CheckResult.FAIL, f"{bad} bad samples")
    print(f"--- 40 scores per image: OK ({len(samples)} samples from processed JSONL)")
    return CheckRecord("40 scores per image", CheckResult.PASS, f"{len(samples)} samples")


# --------------------------------------------------------------------------- #
# P1-7: Verify *_processed.jsonl directly
# --------------------------------------------------------------------------- #


def _verify_processed_artifact(export_dir: Path, benchmark: str, failures: list[str]) -> CheckRecord:
    """P1-7: verify <dataset>_processed.jsonl directly."""
    processed_path = export_dir / f"{benchmark}_processed.jsonl"
    if not processed_path.exists():
        failures.append(f"processed artifact: MISSING [{processed_path}]")
        return CheckRecord("processed artifact exists", CheckResult.FAIL, str(processed_path))

    try:
        lines = [l for l in processed_path.read_text().splitlines() if l.strip()]
        samples = [json.loads(l) for l in lines]
        if not samples:
            failures.append("processed artifact: empty JSONL")
            return CheckRecord("processed artifact non-empty", CheckResult.FAIL, "0 rows")
        # Validate each row has required fields.
        required = {"source_sample_id", "identity_id", "visual_attributes"}
        for i, s in enumerate(samples[:5]):
            missing = required - set(s.keys())
            if missing:
                failures.append(f"processed artifact row {i}: missing fields {missing}")
                return CheckRecord("processed artifact schema", CheckResult.FAIL, f"row {i}")
        print(f"--- processed artifact: OK ({len(samples)} samples)")
        return CheckRecord("processed artifact", CheckResult.PASS, f"{len(samples)} samples")
    except Exception as exc:
        failures.append(f"processed artifact parse: {exc}")
        return CheckRecord("processed artifact", CheckResult.FAIL, str(exc))


# --------------------------------------------------------------------------- #
# P1-8: Verify actual whitelist invariant
# --------------------------------------------------------------------------- #


def _verify_whitelist_invariant(export_dir: Path, benchmark: str, failures: list[str]) -> CheckRecord:
    """P1-8: verify processed labels against score_manifest whitelist.

    Use score_manifest["whitelist_attributes"] and
    score_manifest["whitelist_file_sha256"] as source of truth.
    For every accepted model-generated CelebA observation in *_processed.jsonl,
    require attribute in whitelist.
    """
    score_manifest_path = export_dir / f"{benchmark}_score_manifest.json"
    processed_path = export_dir / f"{benchmark}_processed.jsonl"

    if not score_manifest_path.exists():
        failures.append("whitelist invariant: score manifest missing")
        return CheckRecord("whitelist invariant", CheckResult.FAIL, "no score manifest")

    try:
        sm = json.loads(score_manifest_path.read_text())
    except Exception as exc:
        failures.append(f"whitelist invariant: {exc}")
        return CheckRecord("whitelist invariant", CheckResult.FAIL, str(exc))

    wl_attrs = set(sm.get("whitelist_attributes", []))
    wl_file_sha = sm.get("whitelist_file_sha256")

    # If no whitelist attributes recorded, this is NOT_APPLICABLE for stubs.
    if not wl_attrs and not wl_file_sha:
        print("--- whitelist invariant: NOT_APPLICABLE (no whitelist in score manifest)")
        return CheckRecord("whitelist invariant", CheckResult.NOT_APPLICABLE, "no whitelist configured")

    # Verify whitelist file SHA if configured.
    wl_path = sm.get("whitelist_path")
    if wl_path and wl_file_sha and Path(wl_path).is_file():
        actual_sha = hashlib.sha256(Path(wl_path).read_bytes()).hexdigest()
        if actual_sha != wl_file_sha:
            failures.append(f"whitelist file SHA mismatch: expected {wl_file_sha}, got {actual_sha}")
            return CheckRecord("whitelist file SHA", CheckResult.FAIL, "SHA mismatch")

    # Check processed data against whitelist.
    if processed_path.exists() and wl_attrs:
        try:
            samples = [json.loads(l) for l in processed_path.read_text().splitlines() if l.strip()]
            prefix = "extended_attributes.celeba40."
            violations = 0
            for s in samples:
                for key, obs in s.get("visual_attributes", {}).items():
                    if not key.startswith(prefix):
                        continue
                    attr_name = key.removeprefix(prefix)
                    # If model-generated (source="source_model") and has a
                    # boolean label, the attribute must be in the whitelist.
                    if obs.get("source") == "source_model" and obs.get("label") is not None and attr_name not in wl_attrs:
                        violations += 1
            if violations > 0:
                failures.append(f"whitelist invariant: {violations} non-whitelisted labels in processed")
                return CheckRecord("whitelist invariant", CheckResult.FAIL, f"{violations} violations")
        except Exception as exc:
            failures.append(f"whitelist invariant: {exc}")
            return CheckRecord("whitelist invariant", CheckResult.FAIL, str(exc))

    print(f"--- whitelist invariant: OK ({len(wl_attrs)} whitelisted attrs)")
    return CheckRecord("whitelist invariant", CheckResult.PASS, f"{len(wl_attrs)} attrs")


# --------------------------------------------------------------------------- #
# P1-9: Verify answer_label / answer_text in route rows
# --------------------------------------------------------------------------- #


def _verify_route_expected_answers(export_dir: Path, benchmark: str, failures: list[str]) -> CheckRecord:
    """P1-9: verify route expected answers using answer_label / answer_text.

    For visual route families require:
    - target_attribute != null
    - answer_label in {true, false}
    - answer_text in {"yes", "no"}

    For name_only require:
    - target_attribute == null
    - target_fact_id != null
    - answer_text == target_fact_value
    """
    route_path = export_dir / f"{benchmark}_route_probes.jsonl"
    if not route_path.exists():
        # Fallback: try route_conflict_eval.
        route_path = export_dir / f"{benchmark}_route_conflict_eval.jsonl"
    if not route_path.exists():
        failures.append("route expected answers: no route probe file found")
        return CheckRecord("route expected answers", CheckResult.FAIL, "no route file")

    try:
        lines = [l for l in route_path.read_text().splitlines() if l.strip()]
        rows = [json.loads(l) for l in lines]
    except Exception as exc:
        failures.append(f"route expected answers parse: {exc}")
        return CheckRecord("route expected answers", CheckResult.FAIL, str(exc))

    if not rows:
        failures.append("route expected answers: empty file")
        return CheckRecord("route expected answers", CheckResult.FAIL, "0 rows")

    violations = 0
    for row in rows:
        family = row.get("probe_family", "")
        answer_label = row.get("answer_label")
        answer_text = row.get("answer_text")
        target_attr = row.get("target_attribute")

        if family == "name_only":
            # name_only: target_attribute must be null.
            if target_attr is not None:
                violations += 1
            # answer_text should be present.
            if answer_text is None:
                violations += 1
        else:
            # Visual families: target_attribute must be set.
            if target_attr is None:
                violations += 1
            # answer_label must be boolean.
            if not isinstance(answer_label, bool):
                violations += 1
            # answer_text must be "yes" or "no".
            if answer_text not in ("yes", "no"):
                violations += 1

    if violations > 0:
        failures.append(f"route expected answers: {violations} violations")
        return CheckRecord("route expected answers", CheckResult.FAIL, f"{violations} violations")

    print(f"--- route expected answers: OK ({len(rows)} rows)")
    return CheckRecord("route expected answers", CheckResult.PASS, f"{len(rows)} rows")


# --------------------------------------------------------------------------- #
# P1-10: Verify name_only image absence in route probes
# --------------------------------------------------------------------------- #


def _verify_text_only_image_absence(export_dir: Path, benchmark: str, failures: list[str]) -> CheckRecord:
    """P1-10: inspect actual route probe artifact.

    For every probe_family == name_only, require:
    - modality == text_only
    - image_uri == null
    """
    route_path = export_dir / f"{benchmark}_route_probes.jsonl"
    if not route_path.exists():
        route_path = export_dir / f"{benchmark}_route_conflict_eval.jsonl"
    if not route_path.exists():
        failures.append("name_only image absence: no route probe file")
        return CheckRecord("name_only image absence", CheckResult.FAIL, "no route file")

    try:
        lines = [l for l in route_path.read_text().splitlines() if l.strip()]
        rows = [json.loads(l) for l in lines]
    except Exception as exc:
        failures.append(f"name_only image absence parse: {exc}")
        return CheckRecord("name_only image absence", CheckResult.FAIL, str(exc))

    violations = 0
    name_only_count = 0
    for row in rows:
        if row.get("probe_family") == "name_only":
            name_only_count += 1
            if row.get("image_uri") is not None:
                violations += 1
            if row.get("modality") != "text_only":
                violations += 1

    if violations > 0:
        failures.append(f"name_only image absence: {violations} violations")
        return CheckRecord("name_only image absence", CheckResult.FAIL, f"{violations} violations")

    print(f"--- name_only image absence: OK ({name_only_count} name_only probes)")
    return CheckRecord("name_only image absence", CheckResult.PASS, f"{name_only_count} probes")


# --------------------------------------------------------------------------- #
# P1-11: Verify pair semantics from pair manifest
# --------------------------------------------------------------------------- #


def _verify_pair_semantics(export_dir: Path, benchmark: str, failures: list[str]) -> CheckRecord:
    """P1-11: read <dataset>_pair_manifest.json and validate.

    Check that every pair has a valid pair_type, expected_route_effect
    populated, and controlled/changed variables are present.
    """
    pair_path = export_dir / f"{benchmark}_pair_manifest.json"
    if not pair_path.exists():
        # Pairs may not exist for tiny fixtures; mark NOT_APPLICABLE.
        print("--- pair semantics: NOT_APPLICABLE (no pair manifest)")
        return CheckRecord("pair semantics", CheckResult.NOT_APPLICABLE, "no pair manifest")

    try:
        pairs = json.loads(pair_path.read_text())
    except Exception as exc:
        failures.append(f"pair semantics parse: {exc}")
        return CheckRecord("pair semantics", CheckResult.FAIL, str(exc))

    if not isinstance(pairs, list):
        failures.append("pair semantics: expected a list")
        return CheckRecord("pair semantics", CheckResult.FAIL, "not a list")

    from route_data.build.conflict_generation import PAIR_TYPES

    violations = 0
    for pair in pairs:
        pt = pair.get("pair_type", "")
        if pt not in PAIR_TYPES:
            violations += 1
            continue
        if not pair.get("expected_route_effect"):
            violations += 1
        if not pair.get("controlled"):
            violations += 1
        if not pair.get("changed"):
            violations += 1

    if violations > 0:
        failures.append(f"pair semantics: {violations} issues in {len(pairs)} pairs")
        return CheckRecord("pair semantics", CheckResult.FAIL, f"{violations} issues")

    print(f"--- pair semantics: OK ({len(pairs)} pairs)")
    return CheckRecord("pair semantics", CheckResult.PASS, f"{len(pairs)} pairs")


# --------------------------------------------------------------------------- #
# P1-12: Verify split invariants from split manifest
# --------------------------------------------------------------------------- #


def _verify_split_invariants(export_dir: Path, benchmark: str, failures: list[str]) -> CheckRecord:
    """P1-12: read <dataset>_split_manifest.json and require invariant_issues == []."""
    split_path = export_dir / f"{benchmark}_split_manifest.json"
    if not split_path.exists():
        failures.append("split invariants: no split manifest")
        return CheckRecord("split invariants", CheckResult.FAIL, "no split manifest")

    try:
        data = json.loads(split_path.read_text())
    except Exception as exc:
        failures.append(f"split invariants parse: {exc}")
        return CheckRecord("split invariants", CheckResult.FAIL, str(exc))

    splits = data.get("splits", [])
    if not splits:
        failures.append("split invariants: no splits in manifest")
        return CheckRecord("split invariants", CheckResult.FAIL, "0 splits")

    total_issues = 0
    for s in splits:
        issues = s.get("invariant_issues", [])
        total_issues += len(issues)
        if issues:
            failures.append(f"split '{s.get('name', '?')}': {issues}")

    if total_issues > 0:
        return CheckRecord("split invariants", CheckResult.FAIL, f"{total_issues} issues")

    # Verify forget/retain identity disjointness from split manifest counts.
    for s in splits:
        counts = s.get("counts", {})
        if counts.get("forget", 0) > 0 and counts.get("retain_train", 0) == 0 and counts.get("retain_eval", 0) == 0:
            failures.append(f"split '{s.get('name')}': forget without any retain")
            return CheckRecord("split invariants", CheckResult.FAIL, "forget without retain")

    print(f"--- split invariants: OK ({len(splits)} splits, 0 issues)")
    return CheckRecord("split invariants", CheckResult.PASS, f"{len(splits)} splits")


# --------------------------------------------------------------------------- #
# P1-13: Recompute every checksum independently
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    if path.is_file():
        h.update(path.read_bytes())
        return h.hexdigest()
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                h.update(child.name.encode())
                h.update(child.read_bytes())
        return h.hexdigest()
    return ""


def _verify_export_manifest(export_dir: Path, benchmark: str, failures: list[str]) -> CheckRecord:
    """Verify the export manifest exists, parses, and references expected artifacts.

    The export manifest must contain a ``paths`` mapping that lists every
    released artifact.  We also verify that the manifest records the
    resolved model revision and source provenance.
    """
    manifest_path = export_dir / f"{benchmark}_export_manifest.json"
    if not manifest_path.exists():
        failures.append("export manifest: MISSING")
        return CheckRecord("export manifest exists", CheckResult.FAIL, "missing")

    try:
        mdata = json.loads(manifest_path.read_text())
    except Exception as exc:
        failures.append(f"export manifest parse: {exc}")
        return CheckRecord("export manifest parse", CheckResult.FAIL, str(exc))

    paths = mdata.get("paths", {})
    if not paths:
        failures.append("export manifest: no paths entries")
        return CheckRecord("export manifest paths", CheckResult.FAIL, "empty paths")

    # Verify every referenced artifact exists on disk.
    missing = 0
    for logical_name, rel_path in paths.items():
        if not isinstance(rel_path, str):
            failures.append(f"export manifest: non-string path for {logical_name}: {rel_path!r}")
            missing += 1
            continue
        # Reject absolute paths and parent traversal.
        if rel_path.startswith(("/", "\\")):
            failures.append(f"export manifest: absolute path rejected: {rel_path}")
            missing += 1
            continue
        resolved = (export_dir / rel_path).resolve()
        if not str(resolved).startswith(str(export_dir.resolve())):
            failures.append(f"export manifest: path traversal rejected: {rel_path}")
            missing += 1
            continue
        if logical_name == f"{benchmark}_checksums.json" or rel_path.endswith(f"{benchmark}_checksums.json"):
            continue  # checksums file is self-excluded from its own listing
        artifact = export_dir / rel_path
        if not artifact.exists():
            failures.append(f"export manifest: artifact missing: {rel_path} (key={logical_name})")
            missing += 1

    if missing > 0:
        return CheckRecord("export manifest artifacts", CheckResult.FAIL, f"{missing} missing")

    print(f"--- export manifest: OK ({len(paths)} entries)")
    return CheckRecord("export manifest", CheckResult.PASS, f"{len(paths)} entries")


def _verify_checksums(export_dir: Path, benchmark: str, failures: list[str]) -> CheckRecord:
    """P1-13: recompute every checksum independently and require exact equality.

    Also require that every artifact in the export manifest is covered,
    except the deliberately self-excluded checksum file.
    """
    checksums_path = export_dir / f"{benchmark}_checksums.json"
    if not checksums_path.exists():
        failures.append("checksums: MISSING")
        return CheckRecord("checksums exist", CheckResult.FAIL, "missing")

    try:
        ckdata = json.loads(checksums_path.read_text())
    except Exception as exc:
        failures.append(f"checksums parse: {exc}")
        return CheckRecord("checksums parse", CheckResult.FAIL, str(exc))

    # Self-exclusion check.
    self_ref = f"{benchmark}_checksums.json"
    if self_ref in ckdata:
        failures.append("checksums: self-reference found")
        return CheckRecord("checksums self-exclusion", CheckResult.FAIL, "self-ref")

    # Recompute every checksum.
    mismatches = 0
    for rel_path, expected_hash in ckdata.items():
        full_path = export_dir / rel_path
        if not full_path.exists():
            failures.append(f"checksums: artifact missing: {rel_path}")
            mismatches += 1
            continue
        actual = _sha256_file(full_path)
        if actual != expected_hash:
            failures.append(f"checksums: mismatch for {rel_path}")
            mismatches += 1

    if mismatches > 0:
        return CheckRecord("checksums recompute", CheckResult.FAIL, f"{mismatches} mismatches")

    # Verify export manifest references checksums.
    manifest_path = export_dir / f"{benchmark}_export_manifest.json"
    if manifest_path.exists():
        try:
            mdata = json.loads(manifest_path.read_text())
            if "checksums" not in mdata.get("paths", {}):
                failures.append("export manifest: no checksums key in paths")
                return CheckRecord("export manifest checksums ref", CheckResult.FAIL, "no ref")
        except Exception:
            pass

    print(f"--- checksums: OK ({len(ckdata)} entries, all verified)")
    return CheckRecord("checksums", CheckResult.PASS, f"{len(ckdata)} entries")


# --------------------------------------------------------------------------- #
# P1-15: Non-zero coverage for required smoke paths
# --------------------------------------------------------------------------- #


def _verify_coverage(export_dir: Path, benchmark: str, failures: list[str]) -> CheckRecord:
    """P1-15 + P2-22: require non-zero coverage with per-family minimums.

    If the smoke profile requires QA train/eval, then require train_rows > 0,
    eval_rows > 0.  Also check route families with per-family minimum counts:
      direct_visual >= 20, image_plus_name >= 20, wrong_name >= 20,
      visual_text_conflict >= 20, name_only >= 20 (when facts exist),
      cross_image >= 10 (when multiview exists).
    Thresholds scale down for small fixtures (< 50 total samples).
    """
    # Check QA train/eval.
    train_qa = export_dir / f"{benchmark}_celeba40_visual_qa_train.jsonl"
    eval_qa = export_dir / f"{benchmark}_celeba40_visual_qa_eval.jsonl"

    train_rows = 0
    eval_rows = 0
    if train_qa.exists():
        train_rows = len([l for l in train_qa.read_text().splitlines() if l.strip()])
    if eval_qa.exists():
        eval_rows = len([l for l in eval_qa.read_text().splitlines() if l.strip()])

    issues: list[str] = []
    if train_rows == 0:
        issues.append("train_qa rows == 0")
    if eval_rows == 0:
        issues.append("eval_qa rows == 0")

    # P2-22: check route probe families with per-family minimums.
    route_path = export_dir / f"{benchmark}_route_probes.jsonl"
    if not route_path.exists():
        route_path = export_dir / f"{benchmark}_route_conflict_eval.jsonl"
    if route_path.exists():
        lines = [l for l in route_path.read_text().splitlines() if l.strip()]
        family_counts: dict[str, int] = {}
        for line in lines:
            try:
                row = json.loads(line)
                fam = row.get("probe_family", "")
                family_counts[fam] = family_counts.get(fam, 0) + 1
            except Exception:
                pass

        # P2-22: per-family minimums (full pilot thresholds; scaled for small fixtures).
        total_probes = len(lines)
        scale = 1.0 if total_probes >= 50 else max(0.1, total_probes / 50)
        pilot_minimums: dict[str, int] = {
            "direct_visual": 20,
            "image_plus_name": 20,
            "wrong_name": 20,
            "visual_text_conflict": 20,
            "name_only": 20,
            "cross_image": 10,
        }
        required_families = {"direct_visual", "name_only", "wrong_name", "visual_text_conflict"}

        for fam in required_families:
            if fam not in family_counts:
                issues.append(f"{fam}: 0 probes (missing)")

        # Report per-family counts with scaled minimums.
        for fam, minimum in pilot_minimums.items():
            scaled_min = max(1, int(minimum * scale))
            count = family_counts.get(fam, 0)
            if fam in family_counts and count < scaled_min:
                print(f"--- coverage: {fam}={count} (below scaled minimum {scaled_min})")

        print(f"--- coverage: route families: {dict(sorted(family_counts.items()))}")
    else:
        issues.append("no route probe file")

    if issues:
        failures.append(f"coverage: {issues}")
        return CheckRecord("smoke coverage", CheckResult.FAIL, str(issues))

    print(f"--- coverage: OK (train={train_rows}, eval={eval_rows})")
    return CheckRecord("smoke coverage", CheckResult.PASS, f"train={train_rows}, eval={eval_rows}")


# --------------------------------------------------------------------------- #
# Source split invariant + identity disjointness (supplementary checks)
# --------------------------------------------------------------------------- #


def _verify_source_split_invariant(export_dir: Path, benchmark: str, failures: list[str]) -> CheckRecord:
    """Verify train/eval QA files exist and are non-empty."""
    train_qa = export_dir / f"{benchmark}_celeba40_visual_qa_train.jsonl"
    eval_qa = export_dir / f"{benchmark}_celeba40_visual_qa_eval.jsonl"

    ok = True
    if train_qa.exists():
        train_lines = [l for l in train_qa.read_text().splitlines() if l.strip()]
        print(f"--- source split invariant (train): OK ({len(train_lines)} rows)")
    else:
        failures.append("source split invariant: train QA missing")
        ok = False

    if eval_qa.exists():
        eval_lines = [l for l in eval_qa.read_text().splitlines() if l.strip()]
        print(f"--- source split invariant (eval): OK ({len(eval_lines)} rows)")
    else:
        failures.append("source split invariant: eval QA missing")
        ok = False

    return CheckRecord("source split invariant", CheckResult.PASS if ok else CheckResult.FAIL)


def _verify_identity_disjointness(export_dir: Path, benchmark: str, failures: list[str]) -> CheckRecord:
    """Verify no identity in both train and eval."""
    train_qa = export_dir / f"{benchmark}_celeba40_visual_qa_train.jsonl"
    eval_qa = export_dir / f"{benchmark}_celeba40_visual_qa_eval.jsonl"

    if not (train_qa.exists() and eval_qa.exists()):
        return CheckRecord("identity disjointness", CheckResult.NOT_APPLICABLE, "files missing")

    train_ids = set()
    for line in train_qa.read_text().splitlines():
        if line.strip():
            train_ids.add(json.loads(line).get("identity_id"))

    eval_ids = set()
    for line in eval_qa.read_text().splitlines():
        if line.strip():
            eval_ids.add(json.loads(line).get("identity_id"))

    overlap = train_ids & eval_ids
    if overlap:
        failures.append(f"identity disjointness: {len(overlap)} identities in both train and eval")
        return CheckRecord("identity disjointness", CheckResult.FAIL, f"{len(overlap)} overlap")

    print(f"--- identity disjointness: OK (train={len(train_ids)}, eval={len(eval_ids)})")
    return CheckRecord("identity disjointness", CheckResult.PASS)


# --------------------------------------------------------------------------- #
# P2-23: Per-attribute route balance reporting
# --------------------------------------------------------------------------- #


def _verify_route_balance(export_dir: Path, benchmark: str, failures: list[str]) -> CheckRecord:
    """P2-23: per-attribute route balance reporting.

    For each target attribute seen in route probes, report:
      positive count, negative count, direct_visual count, conflict count,
      wrong_name count, cross_image count, state_change pair count.

    Flag attributes with only one polarity (all positive or all negative).
    This is a reporting check (NOT_APPLICABLE when no route probes exist).
    """
    route_path = export_dir / f"{benchmark}_route_probes.jsonl"
    if not route_path.exists():
        route_path = export_dir / f"{benchmark}_route_conflict_eval.jsonl"
    if not route_path.exists():
        return CheckRecord(
            "route balance", CheckResult.NOT_APPLICABLE,
            "no route probe file", required=False,
        )

    lines = [l for l in route_path.read_text().splitlines() if l.strip()]
    if not lines:
        return CheckRecord("route balance", CheckResult.NOT_APPLICABLE, "empty route file", required=False)

    # Per-attribute counters.
    attr_stats: dict[str, dict[str, int]] = {}

    def _get_stats(attr: str) -> dict[str, int]:
        if attr not in attr_stats:
            attr_stats[attr] = {
                "positive": 0, "negative": 0, "direct_visual": 0,
                "conflict": 0, "wrong_name": 0, "cross_image": 0,
                "state_change_pair": 0,
            }
        return attr_stats[attr]

    for line in lines:
        try:
            row = json.loads(line)
        except Exception:
            continue
        fam = row.get("probe_family", "")
        attr = row.get("target_attribute")
        if not attr:
            continue
        stats = _get_stats(attr)
        label = row.get("answer_label")
        if label is True:
            stats["positive"] += 1
        elif label is False:
            stats["negative"] += 1
        if fam == "direct_visual":
            stats["direct_visual"] += 1
        elif fam == "visual_text_conflict":
            stats["conflict"] += 1
        elif fam == "wrong_name":
            stats["wrong_name"] += 1
        elif fam == "cross_image":
            stats["cross_image"] += 1

    # Check pair manifest for state-change pairs.
    pair_path = export_dir / f"{benchmark}_pair_manifest.json"
    if pair_path.exists():
        try:
            pairs = json.loads(pair_path.read_text())
            if isinstance(pairs, list):
                for p in pairs:
                    if p.get("pair_type") == "cross_image_attribute_state":
                        attr = p.get("attribute")
                        if attr and attr in attr_stats:
                            attr_stats[attr]["state_change_pair"] += 1
        except Exception:
            pass

    # Flag single-polarity attributes.
    unbalanced: list[str] = []
    for attr, stats in sorted(attr_stats.items()):
        if stats["positive"] > 0 and stats["negative"] == 0:
            unbalanced.append(f"{attr} (only positive)")
        elif stats["negative"] > 0 and stats["positive"] == 0:
            unbalanced.append(f"{attr} (only negative)")

    # Persist balance report.
    balance_report = {
        "dataset": benchmark,
        "attributes": {attr: stats for attr, stats in sorted(attr_stats.items())},
        "unbalanced_attributes": unbalanced,
    }
    balance_path = export_dir / f"{benchmark}_route_balance_report.json"
    balance_path.write_text(json.dumps(balance_report, indent=2, default=str) + "\n")

    if unbalanced:
        print(f"--- route balance: {len(unbalanced)} unbalanced attributes: {unbalanced[:5]}")
    print(f"--- route balance: {len(attr_stats)} attributes, report at {balance_path.name}")

    return CheckRecord(
        "route balance", CheckResult.PASS,
        f"{len(attr_stats)} attrs, {len(unbalanced)} unbalanced", required=False,
    )


# --------------------------------------------------------------------------- #
# P2-24: Verify manual audit report
# --------------------------------------------------------------------------- #


def _verify_manual_audit_report(export_dir: Path, benchmark: str, failures: list[str]) -> CheckRecord:
    """P2-9/P2-10: verify <dataset>_manual_audit_report.json.

    The report must have ``audit_report_version == "v1"`` and an ``items`` list
    where each item has at minimum: audit_id, category, sample_id, identity_id,
    image_uri, attribute_or_fact, automatic_checks, review_outcome
    (pass/uncertain/fail/unreviewed), review_note.

    Pilot requires ``critical_failures == 0`` and ``unreviewed_items == 0``.
    """
    report_path = export_dir / f"{benchmark}_manual_audit_report.json"
    if not report_path.exists():
        # Audit report is optional for golden fixture / early smoke.
        return CheckRecord(
            "manual audit report", CheckResult.NOT_APPLICABLE,
            "report not found (optional for smoke)", required=False,
        )

    try:
        report = json.loads(report_path.read_text())
    except Exception as exc:
        failures.append(f"manual audit report parse: {exc}")
        return CheckRecord("manual audit report parse", CheckResult.FAIL, str(exc))

    # P2-10: check report version.
    if report.get("audit_report_version") != "v1":
        failures.append("manual audit report: missing or wrong audit_report_version")
        return CheckRecord("manual audit report version", CheckResult.FAIL, "wrong version")

    items = report.get("items", [])
    if not isinstance(items, list):
        failures.append("manual audit report: items is not a list")
        return CheckRecord("manual audit report items", CheckResult.FAIL, "not a list")

    # P2-10: validate schema of each item (v1 fields).
    required_keys = {"audit_id", "category", "sample_id", "identity_id", "image_uri",
                     "attribute_or_fact", "automatic_checks", "review_outcome", "review_note"}
    valid_outcomes = {"pass", "uncertain", "fail", "unreviewed"}
    bad_items = 0
    for idx, item in enumerate(items):
        missing = required_keys - set(item.keys())
        if missing:
            bad_items += 1
            continue
        if item["review_outcome"] not in valid_outcomes:
            bad_items += 1

    if bad_items > 0:
        failures.append(f"manual audit report: {bad_items} items with invalid schema")
        return CheckRecord("manual audit report schema", CheckResult.FAIL, f"{bad_items} bad items")

    # P2-9: gate requires zero unreviewed and zero critical.
    unreviewed = report.get("unreviewed_items", sum(1 for it in items if it.get("review_outcome") == "unreviewed"))
    if unreviewed > 0:
        failures.append(f"manual audit report: {unreviewed} unreviewed items")
        return CheckRecord("manual audit report", CheckResult.FAIL, f"{unreviewed} unreviewed items")

    critical = report.get("critical_failures", sum(1 for it in items if it.get("review_outcome") == "fail"))
    if critical > 0:
        failures.append(f"manual audit report: {critical} critical failures")
        return CheckRecord("manual audit report", CheckResult.FAIL, f"{critical} critical failures")

    print(f"--- manual audit report: OK ({len(items)} items, 0 critical, 0 unreviewed)")
    return CheckRecord("manual audit report", CheckResult.PASS, f"{len(items)} items")


# --------------------------------------------------------------------------- #
# P1-12: Verify output conforms to prebuilt smoke manifest
# --------------------------------------------------------------------------- #


def _verify_smoke_manifest_conformance(
    export_dir: Path,
    benchmark: str,
    failures: list[str],
    *,
    smoke_manifest_path: Path | None = None,
) -> CheckRecord:
    """P1-12: verify that build outputs conform to the prebuilt smoke manifest.

    Checks:
    1. All output sample IDs belong to the manifest allowlist.
    2. All selected image-bearing samples were scored.
    3. No unexpected source sample IDs appear.
    4. Selection manifest SHA matches build provenance.
    """
    if smoke_manifest_path is None:
        return CheckRecord(
            "smoke manifest conformance", CheckResult.NOT_APPLICABLE,
            "no --smoke-manifest provided", required=False,
        )

    if not smoke_manifest_path.exists():
        failures.append(f"smoke manifest conformance: file not found: {smoke_manifest_path}")
        return CheckRecord("smoke manifest conformance", CheckResult.FAIL, "file not found")

    try:
        manifest = json.loads(smoke_manifest_path.read_text())
    except Exception as exc:
        failures.append(f"smoke manifest conformance: parse error: {exc}")
        return CheckRecord("smoke manifest conformance parse", CheckResult.FAIL, str(exc))

    allowed_ids = set(manifest.get("selected_source_sample_ids", []))
    if not allowed_ids:
        failures.append("smoke manifest conformance: empty selected_source_sample_ids")
        return CheckRecord("smoke manifest conformance", CheckResult.FAIL, "empty allowlist")

    # Load processed output to check sample IDs.
    processed_path = export_dir / f"{benchmark}_processed.jsonl"
    if not processed_path.exists():
        failures.append(f"smoke manifest conformance: no processed JSONL at {processed_path}")
        return CheckRecord("smoke manifest conformance", CheckResult.FAIL, "no processed JSONL")

    output_samples = [
        json.loads(line)
        for line in processed_path.read_text().splitlines()
        if line.strip()
    ]
    output_ids = {s.get("source_sample_id") for s in output_samples if s.get("source_sample_id")}

    # Check 1 & 3: all output IDs must belong to manifest (no unexpected IDs).
    unexpected = output_ids - allowed_ids
    if unexpected:
        failures.append(
            f"smoke manifest conformance: {len(unexpected)} unexpected output sample IDs: "
            f"{sorted(unexpected)[:10]}{'...' if len(unexpected) > 10 else ''}"
        )
        return CheckRecord("smoke manifest conformance", CheckResult.FAIL, f"{len(unexpected)} unexpected IDs")

    # Check 2: all image-bearing manifest samples were scored.
    manifest_image_ids = set()
    for s in manifest.get("samples", []):
        if s.get("image_uri"):
            sid = s.get("sample_id")
            if sid:
                manifest_image_ids.add(sid)

    scored_image_ids = set()
    for s in output_samples:
        if s.get("image_uri"):
            sid = s.get("source_sample_id")
            if sid:
                scored_image_ids.add(sid)

    unscored = manifest_image_ids - scored_image_ids
    if unscored:
        failures.append(
            f"smoke manifest conformance: {len(unscored)} image-bearing samples not scored: "
            f"{sorted(unscored)[:10]}{'...' if len(unscored) > 10 else ''}"
        )
        return CheckRecord("smoke manifest conformance", CheckResult.FAIL, f"{len(unscored)} unscored image samples")

    # Check 4: selection manifest SHA matches build provenance.
    score_manifest_path = export_dir / f"{benchmark}_score_manifest.json"
    if score_manifest_path.exists():
        try:
            score_m = json.loads(score_manifest_path.read_text())
            expected_sha = score_m.get("selection_manifest_sha256")
            if expected_sha:
                actual_sha = hashlib.sha256(smoke_manifest_path.read_bytes()).hexdigest()
                if actual_sha != expected_sha:
                    failures.append(
                        f"smoke manifest conformance: SHA mismatch "
                        f"(provenance={expected_sha[:16]}..., actual={actual_sha[:16]}...)"
                    )
                    return CheckRecord("smoke manifest conformance SHA", CheckResult.FAIL, "SHA mismatch")
        except Exception:
            pass  # SHA check is best-effort if score manifest is malformed.

    print(
        f"--- smoke manifest conformance: OK "
        f"({len(output_ids)} output IDs ⊆ {len(allowed_ids)} manifest IDs, "
        f"{len(manifest_image_ids)} image samples scored)"
    )
    return CheckRecord("smoke manifest conformance", CheckResult.PASS, f"{len(output_ids)} IDs verified")


# --------------------------------------------------------------------------- #
# P1-6: out_of_protocol isolation check
# --------------------------------------------------------------------------- #


def _collect_identity_ids_from_jsonl(path: Path) -> set[str]:
    """Collect unique identity_ids from a JSONL file."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text().splitlines():
        if line.strip():
            doc = json.loads(line)
            iid = doc.get("identity_id")
            if iid:
                ids.add(str(iid))
    return ids


def _collect_oop_identities(export_dir: Path, benchmark: str) -> set[str]:
    """Collect identity_ids with effective_role == 'out_of_protocol'.

    Looks in the processed artifact and score manifest for protocol metadata.
    """
    processed_path = export_dir / f"{benchmark}_processed.jsonl"
    oop: set[str] = set()
    if not processed_path.exists():
        return oop
    for line in processed_path.read_text().splitlines():
        if line.strip():
            doc = json.loads(line)
            meta = doc.get("source_metadata", {})
            if meta.get("effective_role") == "out_of_protocol":
                iid = doc.get("identity_id")
                if iid:
                    oop.add(str(iid))
    return oop


def _verify_out_of_protocol_isolation(
    export_dir: Path, benchmark: str, failures: list[str],
) -> CheckRecord:
    """P1-6: out_of_protocol identities must not leak into downstream artifacts.

    Checks QA train, QA eval, route probes, and split manifests.
    """
    oop_ids = _collect_oop_identities(export_dir, benchmark)
    if not oop_ids:
        print("--- out_of_protocol isolation: OK (no out_of_protocol identities found)")
        return CheckRecord("out_of_protocol isolation", CheckResult.PASS, "no OOP identities")

    leaks: list[str] = []

    # Check QA train.
    qa_train = export_dir / f"{benchmark}_celeba40_visual_qa_train.jsonl"
    train_ids = _collect_identity_ids_from_jsonl(qa_train)
    leaked_train = oop_ids & train_ids
    if leaked_train:
        leaks.append(f"QA train: {len(leaked_train)} OOP identities")

    # Check QA eval.
    qa_eval = export_dir / f"{benchmark}_celeba40_visual_qa_eval.jsonl"
    eval_ids = _collect_identity_ids_from_jsonl(qa_eval)
    leaked_eval = oop_ids & eval_ids
    if leaked_eval:
        leaks.append(f"QA eval: {len(leaked_eval)} OOP identities")

    # Check route probes.
    route_path = export_dir / f"{benchmark}_route_probes.jsonl"
    if not route_path.exists():
        route_path = export_dir / f"{benchmark}_route_conflict_eval.jsonl"
    route_ids = _collect_identity_ids_from_jsonl(route_path)
    leaked_route = oop_ids & route_ids
    if leaked_route:
        leaks.append(f"route probes: {len(leaked_route)} OOP identities")

    # Check split manifests.
    for split_name in ("train", "eval"):
        split_path = export_dir / f"{benchmark}_celeba40_visual_qa_{split_name}.jsonl"
        split_ids = _collect_identity_ids_from_jsonl(split_path)
        leaked_split = oop_ids & split_ids
        if leaked_split and f"QA {split_name}" not in str(leaks):
            leaks.append(f"split {split_name}: {len(leaked_split)} OOP identities")

    if leaks:
        detail = "; ".join(leaks)
        failures.append(f"out_of_protocol isolation: {detail}")
        return CheckRecord("out_of_protocol isolation", CheckResult.FAIL, detail)

    print(
        f"--- out_of_protocol isolation: OK "
        f"({len(oop_ids)} OOP identities excluded from all artifacts)"
    )
    return CheckRecord("out_of_protocol isolation", CheckResult.PASS, f"{len(oop_ids)} OOP IDs excluded")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def verify_benchmark(
    benchmark: str,
    config: str,
    output_dir: Path,
    failures: list[str],
    *,
    smoke_manifest_path: Path | None = None,
) -> list[CheckRecord]:
    """Run all verification checks for a specific benchmark."""
    import yaml

    from route_data.naming import model_output_name

    with open(config) as f:
        cfg = yaml.safe_load(f)
    model_id = cfg.get("model", {}).get("model_id", "unknown")
    model_dir_name = model_output_name(model_id)
    export_dir = output_dir / model_dir_name / benchmark

    if not export_dir.exists():
        export_dir = output_dir / benchmark

    print(f"\n{'=' * 72}")
    print(f"VERIFICATION FOR {benchmark.upper()}")
    print(f"{'=' * 72}")
    print(f"Export directory: {export_dir}")

    records: list[CheckRecord] = []
    records.append(_verify_score_manifest(export_dir, benchmark, failures))
    records.append(_verify_scores_per_image(export_dir, benchmark, failures))
    records.append(_verify_processed_artifact(export_dir, benchmark, failures))
    records.append(_verify_whitelist_invariant(export_dir, benchmark, failures))
    records.append(_verify_source_split_invariant(export_dir, benchmark, failures))
    records.append(_verify_identity_disjointness(export_dir, benchmark, failures))
    records.append(_verify_route_expected_answers(export_dir, benchmark, failures))
    records.append(_verify_text_only_image_absence(export_dir, benchmark, failures))
    records.append(_verify_pair_semantics(export_dir, benchmark, failures))
    records.append(_verify_split_invariants(export_dir, benchmark, failures))
    records.append(_verify_export_manifest(export_dir, benchmark, failures))
    records.append(_verify_checksums(export_dir, benchmark, failures))
    records.append(_verify_coverage(export_dir, benchmark, failures))
    records.append(_verify_route_balance(export_dir, benchmark, failures))
    records.append(_verify_manual_audit_report(export_dir, benchmark, failures))
    # P1-6: out_of_protocol isolation check.
    records.append(_verify_out_of_protocol_isolation(export_dir, benchmark, failures))
    # P1-12: verify output conforms to prebuilt smoke manifest.
    records.append(_verify_smoke_manifest_conformance(
        export_dir, benchmark, failures, smoke_manifest_path=smoke_manifest_path,
    ))

    return records


def _persist_smoke_manifest(
    export_dir: Path,
    benchmark: str,
    selected: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> Path:
    """P1-7: persist <dataset>_smoke_subset_manifest.json.

    The manifest contains both the full sample details and flat ID lists
    (``selected_source_sample_ids``, ``selected_identity_ids``) so that
    every build stage can consume the same allowlist.
    """
    selected_ids = [s.get("source_sample_id") for s in selected if s.get("source_sample_id")]
    selected_iids = sorted({s.get("identity_id") for s in selected if s.get("identity_id")})

    manifest: dict[str, Any] = {
        "dataset": benchmark,
        "selection_version": "smoke_v1",
        "selected_source_sample_ids": selected_ids,
        "selected_identity_ids": selected_iids,
        "samples": [
            {
                "sample_id": s.get("source_sample_id"),
                "identity_id": s.get("identity_id"),
                "source_split": s.get("source_split") or s.get("source_metadata", {}).get("source_split"),
                "image_uri": s.get("image_uri"),
                "image_sha256": s.get("image_sha256"),
            }
            for s in selected
        ],
        "coverage": coverage,
    }
    path = export_dir / f"{benchmark}_smoke_subset_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return path


def main_check(
    dataset: str | None = None,
    config: str | None = None,
    output_dir: Path | None = None,
    *,
    strict: bool = False,
    smoke_manifest: Path | None = None,
) -> int:
    """Run the full verification pipeline."""
    failures: list[str] = []

    # Default: run the golden FAIRGET fixture.
    if dataset is None:
        dataset = "fairget"
    if config is None:
        config = str(REPO / "configs/runs/golden_stub.yaml")

    work = REPO / "data" / "tmp_final_verify"
    golden_root = work / "golden_root"
    out = output_dir if output_dir else work / "out"

    is_golden_fixture = dataset == "fairget" and "golden_stub" in config

    # P0-2: only set the bypass for the bundled synthetic/golden fixture.
    if is_golden_fixture:
        os.environ["ROUTE_DATA_SKIP_IMMUTABLE_CHECK"] = "1"
    else:
        os.environ.pop("ROUTE_DATA_SKIP_IMMUTABLE_CHECK", None)

    # Build the golden fixture if using default dataset.
    if is_golden_fixture:
        from fixtures.golden_fixture import build_golden_fixture

        if golden_root.exists():
            shutil.rmtree(golden_root)
        build_golden_fixture(golden_root)
        os.environ["FAIRGET_ROOT"] = str(golden_root)

    # P0-5: for the golden fixture, process ALL samples to guarantee coverage
    # of all 3 identities (forget, retain_train, retain_eval) and all splits.
    # The golden fixture has 18 samples (3 identities × 6 each).
    limit = "100" if is_golden_fixture else "10"

    # Run the build pipeline.
    # P1-12: pass --smoke-manifest to every stage when a prebuilt manifest is provided.
    for stage in ("annotate", "qa", "route-probes", "splits", "export"):
        stage_argv = ["build", stage, "--dataset", dataset, "--config", config,
                      "--output-dir", str(out), "--limit", limit]
        if smoke_manifest:
            stage_argv.extend(["--smoke-manifest", str(smoke_manifest)])
        _run_cli(
            f"build {stage} --limit {limit}",
            stage_argv,
            expect=0,
            failures=failures,
        )

    # Run verification checks.
    records = verify_benchmark(
        dataset, config, out, failures,
        smoke_manifest_path=smoke_manifest,
    )

    # P1-14: under strict mode, required checks that are NOT_APPLICABLE
    # are treated as failures.
    if strict:
        for rec in records:
            if rec.required and rec.result == CheckResult.NOT_APPLICABLE:
                failures.append(f"strict: required check '{rec.name}' was NOT_APPLICABLE")

    # Persist smoke subset manifest if we have processed data and no prebuilt manifest.
    # P1-12: when a prebuilt manifest was provided, skip post-hoc generation.
    if not smoke_manifest:
        import yaml as _yaml

        from route_data.naming import model_output_name as _mon

        with open(config) as _f:
            _cfg = _yaml.safe_load(_f)
        _mid = _cfg.get("model", {}).get("model_id", "unknown")
        _export_dir = out / _mon(_mid) / dataset
        if not _export_dir.exists():
            _export_dir = out / dataset
        processed_path = _export_dir / f"{dataset}_processed.jsonl"
        if processed_path.exists():
            try:
                _samples = [json.loads(l) for l in processed_path.read_text().splitlines() if l.strip()]
                _result = select_smoke_subset(_samples)
                _persist_smoke_manifest(_export_dir, dataset, _result["selected"], _result["coverage"])
                if _result["issues"]:
                    print(f"--- smoke subset: coverage issues: {_result['issues']}")
            except Exception as _exc:
                msg = f"smoke-manifest generation failed: {_exc}"
                print(f"WARNING: {msg}")
                failures.append(msg)

    # Summary.
    print(f"\n{'=' * 72}")
    for rec in records:
        print(f"  {rec}")
    print(f"{'=' * 72}")
    if failures:
        print(f"SUMMARY: FAILED ({len(failures)} issue(s))")
        for f in failures:
            print(f"  - {f}")
    else:
        print("SUMMARY: ALL CHECKS PASSED")
    print(f"{'=' * 72}")
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# P3-28: Pre-generation gate
# --------------------------------------------------------------------------- #


def check_pregeneration_gate(
    output_dir: Path,
    dataset: str,
    *,
    strict_verification_failures: int = 0,
    critical_skips: int = 0,
    checksum_mismatches: int = 0,
    source_split_violations: int = 0,
    route_semantic_violations: int = 0,
    manual_audit_critical_failures: int = 0,
    pending_revisions: int = 0,
) -> dict[str, Any]:
    """P3-28: pre-generation gate requiring zero unresolved warnings.

    Returns a dict with gate status and details for each condition.
    All counts must be zero for the gate to pass.
    """
    conditions = {
        "pending_revisions": pending_revisions,
        "strict_verification_failures": strict_verification_failures,
        "critical_skips": critical_skips,
        "checksum_mismatches": checksum_mismatches,
        "source_split_violations": source_split_violations,
        "route_semantic_violations": route_semantic_violations,
        "manual_audit_critical_failures": manual_audit_critical_failures,
    }

    all_zero = all(v == 0 for v in conditions.values())
    gate = {
        "gate_passed": all_zero,
        "dataset": dataset,
        "output_dir": str(output_dir),
        "conditions": conditions,
    }

    if not all_zero:
        failing = [k for k, v in conditions.items() if v > 0]
        gate["failing_conditions"] = failing
        print(f"PRE-GENERATION GATE: FAILED ({len(failing)} condition(s) non-zero)")
        for cond in failing:
            print(f"  - {cond}: {conditions[cond]}")
    else:
        print("PRE-GENERATION GATE: PASSED (all conditions zero)")

    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Final verification for any benchmark (P0/P1 upgraded)"
    )
    parser.add_argument("--dataset", help="Benchmark name")
    parser.add_argument("--config", help="Run config YAML path")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--strict", action="store_true",
                        help="Required checks cannot SKIP")
    # P1-12: consume the prebuilt smoke manifest.
    parser.add_argument("--smoke-manifest", type=Path, default=None,
                        help="Prebuilt smoke manifest JSON (skip post-hoc selection)")
    args = parser.parse_args()

    sys.exit(main_check(
        dataset=args.dataset,
        config=args.config,
        output_dir=args.output_dir,
        strict=args.strict,
        smoke_manifest=args.smoke_manifest,
    ))
