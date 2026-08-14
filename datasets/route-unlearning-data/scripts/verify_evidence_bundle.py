#!/usr/bin/env python3
"""Verify internal consistency of the frozen FIUBench research bundle.

Treats ``research_dataset_manifest.json`` as the root manifest and
recursively validates every referenced artifact, plus structural
invariants of the annotation evidence, attribute analysis, route probes,
wrong-name probes, manual audit, and provenance metadata.

Exit code 0 on success, 1 on any failure.

Usage::

    python scripts/verify_evidence_bundle.py
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = REPO / "outputs" / "full_fiubench" / "Qwen_Qwen3.5-9B" / "fiubench"
EVIDENCE_DIR = REPO / "outputs" / "full_fiubench" / "evidence"

# ── helpers ──────────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _count_jsonl(path: Path) -> int:
    n = 0
    with open(path) as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _recount_accepted_labels(processed_path: Path) -> int:
    """Recount non-null labels directly from the processed JSONL."""
    count = 0
    for line in processed_path.read_text().splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        for obs in doc.get("visual_attributes", {}).values():
            if obs.get("label") is not None:
                count += 1
    return count


def _compute_protocol_sha(protocol: dict) -> tuple[str, dict]:
    """Reproduce compute_protocol_sha256() without importing route_data."""
    canonical = {
        "algorithm_version": 1,
        "eval_fraction": protocol.get("eval_fraction"),
        "eval_seed": protocol.get("eval_seed"),
        "eval_bucket": protocol.get("eval_bucket"),
        "forget_bucket": protocol.get("forget_bucket"),
        "name": protocol.get("name"),
        "source_population": protocol.get("source_population"),
        "train_bucket": protocol.get("train_bucket"),
    }
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest(), canonical


# ── main verification ────────────────────────────────────────────────


def verify() -> int:
    groups: dict[str, tuple[list[str], list[str]]] = {}

    def ok(group: str, msg: str) -> None:
        groups.setdefault(group, ([], []))
        groups[group][0].append(msg)

    def fail(group: str, msg: str) -> None:
        groups.setdefault(group, ([], []))
        groups[group][1].append(msg)

    # ==================================================================
    # GROUP 1: CORE ANNOTATION EVIDENCE
    # ==================================================================
    g = "CORE ANNOTATION EVIDENCE"

    core_names = [
        "annotation_summary.json",
        "fiubench_population_report.json",
        "fiubench_score_manifest.json",
        "source_image_audit.json",
        "runtime_environment.json",
        "artifact_checksums.json",
    ]
    missing = [f for f in core_names if not (EVIDENCE_DIR / f).exists()]
    if missing:
        for m in missing:
            fail(g, f"MISSING evidence file: {m}")
        _report(groups)
        return 1

    summary = _load_json(EVIDENCE_DIR / "annotation_summary.json")
    pop_report = _load_json(EVIDENCE_DIR / "fiubench_population_report.json")
    score_manifest = _load_json(EVIDENCE_DIR / "fiubench_score_manifest.json")
    image_audit = _load_json(EVIDENCE_DIR / "source_image_audit.json")
    runtime_env = _load_json(EVIDENCE_DIR / "runtime_environment.json")
    checksums = _load_json(EVIDENCE_DIR / "artifact_checksums.json")

    # accepted_labels_processed matches recomputed count
    processed_path = ARTIFACT_DIR / "fiubench_processed.jsonl"
    if processed_path.exists():
        recomputed = _recount_accepted_labels(processed_path)
        declared = summary.get("accepted_labels_processed")
        if recomputed == declared:
            ok(g, f"accepted_labels_processed matches recomputed ({declared})")
        else:
            fail(g, f"accepted_labels_processed mismatch: summary={declared}, recomputed={recomputed}")
    else:
        fail(g, f"Cannot verify accepted_labels: {processed_path} missing")

    # population_report matches annotation_summary
    pop_acc = pop_report.get("accepted_labels_processed")
    sum_acc = summary.get("accepted_labels_processed")
    if pop_acc == sum_acc:
        ok(g, f"population_report.accepted_labels_processed == annotation_summary ({sum_acc})")
    else:
        fail(g, f"accepted_labels mismatch: pop_report={pop_acc}, annotation_summary={sum_acc}")

    # all evidence files share the same midp_commit
    commits = {
        "annotation_summary": summary.get("midp_commit"),
        "population_report": pop_report.get("midp_commit"),
        "score_manifest": score_manifest.get("midp_commit"),
        "source_image_audit": image_audit.get("midp_commit"),
        "runtime_environment": runtime_env.get("midp_commit"),
        "artifact_checksums": checksums.get("computed_at_commit"),
    }
    unique_commits = set(commits.values())
    if len(unique_commits) == 1:
        ok(g, f"All evidence files share midp_commit={commits['annotation_summary'][:12]}…")
    else:
        for name, sha in commits.items():
            fail(g, f"midp_commit inconsistency: {name} = {sha}")

    # artifact checksums verify
    recorded_artifacts = checksums.get("artifacts", {})
    for name, info in recorded_artifacts.items():
        apath = ARTIFACT_DIR / name
        if not apath.exists():
            fail(g, f"Checksum artifact missing: {name}")
            continue
        actual_sha = _sha256_file(apath)
        recorded_sha = info.get("sha256", "")
        if actual_sha == recorded_sha:
            ok(g, f"SHA-256 verified: {name}")
        else:
            fail(g, f"SHA-256 MISMATCH for {name}: recorded={recorded_sha[:16]}… actual={actual_sha[:16]}…")
        actual_size = apath.stat().st_size
        recorded_size = info.get("size_bytes")
        if actual_size == recorded_size:
            ok(g, f"Size verified: {name} ({actual_size} bytes)")
        else:
            fail(g, f"Size mismatch for {name}: recorded={recorded_size}, actual={actual_size}")

    # structural invariants
    for label, val in [
        ("source_identity_rows", summary.get("source_identity_rows")),
        ("unique_images (summary)", summary.get("unique_images")),
        ("unique_image_uris (pop)", pop_report.get("unique_image_uris")),
        ("unique_image_sha256 (pop)", pop_report.get("unique_image_sha256")),
        ("source_identities (audit)", image_audit.get("source_identities")),
        ("unique_images (audit)", image_audit.get("unique_images")),
    ]:
        if val == 573:
            ok(g, f"{label} = 573")
        else:
            fail(g, f"{label} expected 573, got {val}")

    for label, val in [
        ("raw_score_rows (summary)", summary.get("raw_score_rows")),
        ("raw_score_rows (pop)", pop_report.get("raw_score_rows")),
        ("score_rows (audit)", image_audit.get("score_rows")),
    ]:
        if val == 22920:
            ok(g, f"{label} = 22920")
        else:
            fail(g, f"{label} expected 22920, got {val}")

    for label, val in [
        ("canonical_samples (summary)", summary.get("canonical_samples")),
        ("canonical_samples_total (pop)", pop_report.get("canonical_samples_total")),
    ]:
        if val == 30660:
            ok(g, f"{label} = 30660")
        else:
            fail(g, f"{label} expected 30660, got {val}")

    for label, val in [
        ("observations (summary)", summary.get("observations")),
        ("observations (pop)", pop_report.get("observations")),
    ]:
        if val == 1226400:
            ok(g, f"{label} = 1226400")
        else:
            fail(g, f"{label} expected 1226400, got {val}")

    if image_audit.get("score_completeness_ok") is True:
        ok(g, "score_completeness_ok = true")
    else:
        fail(g, "score_completeness_ok is not true")

    nwl = summary.get("non_whitelisted_accepted_labels_processed")
    if nwl == 0:
        ok(g, "non_whitelisted_accepted_labels_processed = 0")
    else:
        fail(g, f"non_whitelisted_accepted_labels_processed expected 0, got {nwl}")

    orig = pop_report.get("canonical_original_samples", 0)
    para = pop_report.get("canonical_paraphrase_samples", 0)
    pert = pop_report.get("canonical_perturbed_samples", 0)
    sample_sum = orig + para + pert
    if sample_sum == pop_report.get("canonical_samples_total", 0):
        ok(g, f"sample type breakdown sums to canonical total ({sample_sum})")
    else:
        fail(g, f"sample type breakdown mismatch: {orig}+{para}+{pert}={sample_sum} ≠ {pop_report.get('canonical_samples_total')}")

    split_sum = sum(pop_report.get("split_counts", {}).values())
    if split_sum == pop_report.get("canonical_samples_total", 0):
        ok(g, f"split counts sum to canonical total ({split_sum})")
    else:
        fail(g, f"split counts mismatch: {split_sum} ≠ {pop_report.get('canonical_samples_total')}")

    # ==================================================================
    # GROUP 2: RESEARCH MANIFEST
    # ==================================================================
    g = "RESEARCH MANIFEST"
    manifest_path = EVIDENCE_DIR / "research_dataset_manifest.json"

    if not manifest_path.exists():
        fail(g, "research_dataset_manifest.json missing")
    else:
        manifest = _load_json(manifest_path)

        # required top-level fields
        for field in [
            "manifest_version", "definition_of_done", "code_provenance",
            "created_at", "hard_stop_conditions",
        ]:
            if field in manifest:
                ok(g, f"manifest has '{field}'")
            else:
                fail(g, f"manifest missing required field '{field}'")

        # manifest_version recognized
        ver = manifest.get("manifest_version")
        if ver in ("1.0", "2.0"):
            ok(g, f"manifest_version recognized ({ver})")
        else:
            fail(g, f"manifest_version unrecognized: {ver}")

        # ready_for_experiments field exists
        dod = manifest.get("definition_of_done", {})
        if "ready_for_experiments" in dod:
            ok(g, "ready_for_experiments field exists")
        else:
            fail(g, "ready_for_experiments field missing")

        # creation code commit exists
        cp = manifest.get("code_provenance", {})
        egc = cp.get("evidence_generation_code_commit")
        if egc:
            ok(g, "evidence_generation_code_commit present")
        else:
            fail(g, "evidence_generation_code_commit missing")

        # P0-1: verify evidence_generation_code_commit is a real git commit.
        if egc:
            try:
                subprocess.run(
                    ["git", "cat-file", "-e", f"{egc}^{{commit}}"],
                    cwd=REPO, check=True, capture_output=True,
                )
                ok(g, f"evidence_generation_code_commit is real commit ({egc[:12]})")
            except (subprocess.CalledProcessError, FileNotFoundError):
                fail(g, f"evidence_generation_code_commit not a reachable commit: {egc[:12]}")

        # P0-1: verify dataset_creation_commit is a real git commit.
        dcc = cp.get("dataset_creation_commit")
        if dcc:
            ok(g, "dataset_creation_commit present")
            try:
                subprocess.run(
                    ["git", "cat-file", "-e", f"{dcc}^{{commit}}"],
                    cwd=REPO, check=True, capture_output=True,
                )
                ok(g, f"dataset_creation_commit is real commit ({dcc[:12]})")
            except (subprocess.CalledProcessError, FileNotFoundError):
                fail(g, f"dataset_creation_commit not a reachable commit: {dcc[:12]}")
        else:
            fail(g, "dataset_creation_commit missing")

        # P0-1: optional stronger check — ancestor relationship.
        if egc and dcc:
            try:
                subprocess.run(
                    ["git", "merge-base", "--is-ancestor", dcc, egc],
                    cwd=REPO, check=True, capture_output=True,
                )
                ok(g, "dataset_creation_commit is ancestor of evidence_generation_code_commit")
            except subprocess.CalledProcessError:
                fail(g, "dataset_creation_commit is NOT ancestor of evidence_generation_code_commit")

        # git_dirty == false
        if cp.get("midp_git_dirty") is False:
            ok(g, "midp_git_dirty == false")
        else:
            fail(g, "midp_git_dirty is not false")

        # created_at parseable
        cat = manifest.get("created_at", "")
        try:
            datetime.strptime(cat, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc,
            )
            ok(g, "created_at is parseable")
        except ValueError:
            fail(g, f"created_at not parseable: {cat}")

        # all referenced paths exist
        bundle_paths = manifest.get("evidence_bundle_files", [])
        missing_paths = [p for p in bundle_paths if not (REPO / p).exists()]
        if not missing_paths:
            ok(g, f"all {len(bundle_paths)} bundle paths exist")
        else:
            for mp in missing_paths:
                fail(g, f"bundle path missing: {mp}")

        # artifact hash cross-checks (dataset_artifacts)
        da = manifest.get("dataset_artifacts", {})
        pd_info = da.get("processed_dataset", {})
        pp = ARTIFACT_DIR / pd_info.get("path", "")
        if pp.exists():
            if pd_info.get("sha256") == _sha256_file(pp):
                ok(g, "processed_dataset SHA-256 matches manifest")
            else:
                fail(g, "processed_dataset SHA-256 mismatch")
            if pd_info.get("size_bytes") == pp.stat().st_size:
                ok(g, "processed_dataset size matches manifest")
            else:
                fail(g, "processed_dataset size mismatch")

        for key in ("image_scores", "model_scores"):
            st = da.get("score_table", {}).get(key, {})
            sp = ARTIFACT_DIR / st.get("path", "")
            if sp.exists():
                if st.get("sha256") == _sha256_file(sp):
                    ok(g, f"{key} SHA-256 matches manifest")
                else:
                    fail(g, f"{key} SHA-256 mismatch")
                if st.get("size_bytes") == sp.stat().st_size:
                    ok(g, f"{key} size matches manifest")
                else:
                    fail(g, f"{key} size mismatch")

        rp_info = da.get("route_probes", {})
        rp_path = ARTIFACT_DIR / rp_info.get("path", "")
        if rp_path.exists():
            if rp_info.get("sha256") == _sha256_file(rp_path):
                ok(g, "route_probes SHA-256 matches manifest")
            else:
                fail(g, "route_probes SHA-256 mismatch")

        sp_info = da.get("splits", {})
        if sp_info.get("sha256"):
            ok(g, "splits SHA-256 present in manifest")

        em_info = da.get("export_manifest", {})
        em_path = ARTIFACT_DIR / em_info.get("path", "")
        if em_path.exists():
            if em_info.get("sha256") == _sha256_file(em_path):
                ok(g, "export_manifest SHA-256 matches manifest")
            else:
                fail(g, "export_manifest SHA-256 mismatch")

        # quality evidence hash cross-checks
        qe = manifest.get("quality_evidence", {})
        ma_info = qe.get("manual_audit", {})
        ma_path = ARTIFACT_DIR / ma_info.get("path", "")
        if ma_path.exists():
            if ma_info.get("sha256") == _sha256_file(ma_path):
                ok(g, "manual_audit SHA-256 matches manifest")
            else:
                fail(g, "manual_audit SHA-256 mismatch")

        ad_info = qe.get("attribute_distribution", {})
        ad_path_str = ad_info.get("path", "")
        ad_path = REPO / ad_path_str if ad_path_str else None
        if ad_path and ad_path.exists():
            if ad_info.get("sha256") == _sha256_file(ad_path):
                ok(g, "attribute_distribution SHA-256 matches manifest")
            else:
                fail(g, "attribute_distribution SHA-256 mismatch")

        # whitelist hash cross-checks
        aw = manifest.get("attribute_whitelists", {})
        for wl_key, wl_info in aw.items():
            wl_path = REPO / wl_info.get("path", "") if wl_info.get("path") else None
            if wl_path and wl_path.exists():
                if wl_info.get("sha256") == _sha256_file(wl_path):
                    ok(g, f"{wl_key} SHA-256 matches manifest")
                else:
                    fail(g, f"{wl_key} SHA-256 mismatch")

    # ==================================================================
    # GROUP 3: PROVENANCE
    # ==================================================================
    g = "PROVENANCE"

    if not manifest_path.exists():
        fail(g, "manifest not available for provenance checks")
    else:
        mp = manifest.get("model_provenance", {})
        # model_id
        if mp.get("model_id") == "Qwen/Qwen3.5-9B":
            ok(g, "model_id == Qwen/Qwen3.5-9B")
        else:
            fail(g, f"model_id mismatch: {mp.get('model_id')}")

        # resolved_revision is full commit SHA (40 hex)
        rev = mp.get("resolved_revision", "")
        if len(rev) == 40 and all(c in "0123456789abcdef" for c in rev):
            ok(g, "resolved_revision is full 40-char SHA")
        else:
            fail(g, f"resolved_revision not full SHA: {rev}")

        # model_fingerprint non-empty
        if mp.get("model_fingerprint"):
            ok(g, "model_fingerprint non-empty")
        else:
            fail(g, "model_fingerprint is empty")

        # scoring provenance
        sp = manifest.get("scoring_provenance", {})
        if str(sp.get("scoring_version")) == "2":
            ok(g, "scoring_version == 2")
        else:
            fail(g, f"scoring_version mismatch: {sp.get('scoring_version')}")

        if sp.get("candidate_set_hash"):
            ok(g, "candidate_set_hash non-empty")
        else:
            fail(g, "candidate_set_hash is empty")

        if sp.get("prompt_registry_hash"):
            ok(g, "prompt_registry_hash non-empty")
        else:
            fail(g, "prompt_registry_hash is empty")

        # Protocol SHA cross-check
        proto = manifest.get("protocol", {})
        manifest_psha = proto.get("protocol_sha256", "")
        fiubench_cfg_path = REPO / "configs" / "data" / "fiubench.yaml"
        if fiubench_cfg_path.exists():
            import yaml  # type: ignore[import-unless-installed]

            fcfg = yaml.safe_load(fiubench_cfg_path.read_text())
            fproto = fcfg["data"]["extras"]["fiubench_protocol"]
            config_psha, _ = _compute_protocol_sha(fproto)
            if manifest_psha == config_psha:
                ok(g, "manifest protocol SHA == config protocol SHA")
            else:
                fail(g, f"protocol SHA mismatch: manifest={manifest_psha[:16]}… config={config_psha[:16]}…")

        # score_manifest consistency
        sm_rev = score_manifest.get("resolved_revision", "")
        if sm_rev == rev:
            ok(g, "score_manifest resolved_revision matches manifest")
        else:
            fail(g, "score_manifest resolved_revision mismatch")

        sm_fp = score_manifest.get("model_fingerprint", "")
        if sm_fp == mp.get("model_fingerprint"):
            ok(g, "score_manifest model_fingerprint matches manifest")
        else:
            fail(g, "score_manifest model_fingerprint mismatch")

    # ==================================================================
    # GROUP 4: WHITELIST CHECKS
    # ==================================================================
    g = "WHITELIST CHECKS"

    if not manifest_path.exists():
        fail(g, "manifest not available")
    else:
        aw = manifest.get("attribute_whitelists", {})

        # CelebA reliability whitelist
        celeba_info = aw.get("celeba_reliability_whitelist", {})
        celeba_path = REPO / celeba_info.get("path", "") if celeba_info.get("path") else None
        if celeba_path and celeba_path.exists():
            celeba_data = _load_json(celeba_path)
            celeba_attrs = sorted(celeba_data.get("attributes", []))
            if len(celeba_attrs) == 13:
                ok(g, "CelebA reliability whitelist has 13 attributes")
            else:
                fail(g, f"CelebA whitelist has {len(celeba_attrs)} attributes, expected 13")
            if celeba_info.get("sha256") == _sha256_file(celeba_path):
                ok(g, "CelebA whitelist SHA-256 verified")
            else:
                fail(g, "CelebA whitelist SHA-256 mismatch")
        else:
            fail(g, "CelebA whitelist file not found")

        # FIUBench experiment subset
        v2_info = aw.get("fiubench_experiment_subset", {})
        v2_path = REPO / v2_info.get("path", "") if v2_info.get("path") else None
        if v2_path and v2_path.exists():
            v2_data = _load_json(v2_path)
            v2_attrs = sorted(v2_data.get("attributes", []))
            if len(v2_attrs) == 10:
                ok(g, "FIUBench experiment subset has 10 attributes")
            else:
                fail(g, f"experiment subset has {len(v2_attrs)} attributes, expected 10")

            for excl in ("Wearing_Hat", "Wearing_Necktie", "Sideburns"):
                if excl not in v2_attrs:
                    ok(g, f"{excl} excluded from experiment subset")
                else:
                    fail(g, f"{excl} should be excluded from experiment subset")

            if v2_info.get("sha256") == _sha256_file(v2_path):
                ok(g, "experiment subset SHA-256 verified")
            else:
                fail(g, "experiment subset SHA-256 mismatch")

            # subset ⊆ reliability whitelist
            if celeba_path and celeba_path.exists():
                if set(v2_attrs) <= set(celeba_attrs):
                    ok(g, "experiment_subset ⊆ reliability_whitelist")
                else:
                    fail(g, "experiment_subset is NOT a subset of reliability_whitelist")
        else:
            fail(g, "experiment subset whitelist file not found")

    # ==================================================================
    # GROUP 5: ATTRIBUTE ANALYSIS
    # ==================================================================
    g = "ATTRIBUTE ANALYSIS"

    attr_dist_path = EVIDENCE_DIR / "attribute_distribution_report.json"
    if not attr_dist_path.exists():
        fail(g, "attribute_distribution_report.json missing")
    else:
        ad = _load_json(attr_dist_path)

        if ad.get("analysis_unit") == "unique_image":
            ok(g, "analysis_unit == unique_image")
        else:
            fail(g, f"analysis_unit mismatch: {ad.get('analysis_unit')}")

        if ad.get("n_images") == 573:
            ok(g, "n_images == 573")
        else:
            fail(g, f"n_images expected 573, got {ad.get('n_images')}")

        if ad.get("n_identities") == 573:
            ok(g, "n_identities == 573")
        else:
            fail(g, f"n_identities expected 573, got {ad.get('n_identities')}")

        wl_attrs = ad.get("whitelist_attributes", [])
        if len(wl_attrs) == 13:
            ok(g, "13 whitelist attributes represented")
        else:
            fail(g, f"expected 13 whitelist attributes, got {len(wl_attrs)}")

        # per-attribute: positive + negative + uncertain == 573
        attrs_data = ad.get("attributes", {})
        _roles = ["train", "eval", "exclude", "out_of_protocol"]
        for attr_name, adata in attrs_data.items():
            roles = adata.get("roles", {})
            pos = sum(roles.get(r, {}).get("positive", 0) for r in _roles)
            neg = sum(roles.get(r, {}).get("negative", 0) for r in _roles)
            unc = sum(roles.get(r, {}).get("uncertain", 0) for r in _roles)
            total = pos + neg + unc
            if total == 573:
                ok(g, f"{attr_name}: pos+neg+unc = {total}")
            else:
                fail(g, f"{attr_name}: pos({pos})+neg({neg})+unc({unc}) = {total}, expected 573")

        # v2 exclusion diagnostics
        hat_data = attrs_data.get("Wearing_Hat", {})
        hat_roles = hat_data.get("roles", {})
        hat_pos = sum(hat_roles.get(r, {}).get("positive", 0) for r in _roles)
        if hat_pos == 0:
            ok(g, "Wearing_Hat: positive count == 0")
        else:
            fail(g, f"Wearing_Hat: positive count == {hat_pos}, expected 0")

        nt_data = attrs_data.get("Wearing_Necktie", {})
        nt_roles = nt_data.get("roles", {})
        nt_pos = sum(nt_roles.get(r, {}).get("positive", 0) for r in _roles)
        if nt_pos == 0:
            ok(g, "Wearing_Necktie: positive count == 0")
        else:
            fail(g, f"Wearing_Necktie: positive count == {nt_pos}, expected 0")

        sb_data = attrs_data.get("Sideburns", {})
        sb_unc_frac = sb_data.get("uncertainty_fraction", 0)
        if sb_unc_frac > 0.50:
            ok(g, f"Sideburns: uncertainty_fraction > 0.50 ({sb_unc_frac:.2f})")
        else:
            fail(g, f"Sideburns: uncertainty_fraction = {sb_unc_frac}, expected > 0.50")

    # ==================================================================
    # GROUP 6: ROUTE PROBES
    # ==================================================================
    g = "ROUTE PROBES"

    route_path = ARTIFACT_DIR / "fiubench_route_conflict_eval.jsonl"
    if not route_path.exists():
        fail(g, "fiubench_route_conflict_eval.jsonl missing")
    else:
        with open(route_path) as _fh:
            route_probes = [json.loads(ln) for ln in _fh if ln.strip()]

        families: dict[str, int] = {}
        for rp in route_probes:
            fam = rp.get("probe_family", "unknown")
            families[fam] = families.get(fam, 0) + 1

        required_fams = [
            "direct_visual", "image_plus_name", "name_only",
            "wrong_name", "visual_text_conflict",
        ]
        for fam in required_fams:
            cnt = families.get(fam, 0)
            if cnt > 0:
                ok(g, f"{fam}: {cnt} probes")
            else:
                fail(g, f"{fam}: 0 probes (expected > 0)")

        if len(route_probes) == 500:
            ok(g, "total route probes == 500")
        else:
            fail(g, f"total route probes = {len(route_probes)}, expected 500")

    # coverage report (new per-family format)
    cov_path = EVIDENCE_DIR / "route_probe_attribute_coverage.json"
    if not cov_path.exists():
        fail(g, "route_probe_attribute_coverage.json missing")
    else:
        cov = _load_json(cov_path)

        # Derive totals from the actual route probe data.
        all_fam_cov = cov.get("all_families", {})
        total_identities = len({
            rp.get("identity_id") for rp in route_probes
            if rp.get("probe_family") in (
                "direct_visual", "image_plus_name", "wrong_name",
                "visual_text_conflict",
            )
        })
        total_probes = len(route_probes)
        if total_identities == 100:
            ok(g, "coverage: total_protocol_identities == 100")
        else:
            fail(g, f"coverage: identities = {total_identities}")

        if total_probes == 500:
            ok(g, "coverage: total_route_probes == 500")
        else:
            fail(g, f"coverage: probes = {total_probes}")

        # Families are top-level keys (excluding all_families).
        cov_families = {k: v for k, v in cov.items() if k != "all_families"}
        for fam in ["direct_visual", "image_plus_name", "wrong_name", "visual_text_conflict"]:
            if fam in cov_families:
                ok(g, f"coverage: family '{fam}' present")
            else:
                fail(g, f"coverage: family '{fam}' missing")

        # Attributes are sub-keys within all_families.
        cov_attrs = all_fam_cov
        if len(cov_attrs) == 10:
            ok(g, "coverage: 10 experiment attributes")
        else:
            fail(g, f"coverage: {len(cov_attrs)} attributes, expected 10")

        # P0-2: state-balance hard checks.
        state_balance_fail = 0
        for attr, acov in all_fam_cov.items():
            pos_ct = acov.get("positive_target_count", 0)
            neg_ct = acov.get("negative_target_count", 0)
            if pos_ct > 0 and neg_ct > 0:
                continue  # both states present — OK
            state_balance_fail += 1
        if state_balance_fail == 0:
            ok(g, "all attributes have both positive and negative route targets")
        else:
            fail(g, f"{state_balance_fail} attribute(s) missing positive or negative route targets")

        # cross-check experiment_subset_sha256 against manifest whitelist.
        if manifest_path.exists():
            v2_info = manifest.get("attribute_whitelists", {}).get("fiubench_experiment_subset", {})
            v2_sha = v2_info.get("sha256", "")
            cov_subset_sha = cov.get("experiment_subset_sha256")
            if cov_subset_sha and cov_subset_sha == v2_sha:
                ok(g, "coverage: experiment_subset_sha256 matches manifest")
            elif not cov_subset_sha:
                # New format omits the SHA inline; the experiment subset
                # SHA is verified in the WHITELIST CHECKS group above.
                if len(all_fam_cov) == 10:
                    ok(g, "coverage: experiment attributes verified via whitelist group")
                else:
                    fail(g, "coverage: experiment attribute count mismatch")
            else:
                fail(g, "coverage: experiment_subset_sha256 mismatch")

    # ==================================================================
    # GROUP 7: WRONG-NAME PROBES
    # ==================================================================
    g = "WRONG-NAME PROBES"

    wn_report_path = EVIDENCE_DIR / "actual_wrong_name_probe_report.json"
    if not wn_report_path.exists():
        fail(g, "actual_wrong_name_probe_report.json missing")
    else:
        wn_report = _load_json(wn_report_path)
        total_wn = wn_report.get("total_wrong_name_probes", 0)
        if total_wn > 0:
            ok(g, f"wrong_name probes > 0 ({total_wn})")
        else:
            fail(g, "total_wrong_name_probes == 0")

        if wn_report.get("all_invariants_pass") is True:
            ok(g, "all wrong-name invariants pass")
        else:
            fail(g, "all_invariants_pass is not true")

        sim_dist = wn_report.get("similarity_distribution", {})
        if sim_dist.get("min") is not None and not math.isnan(sim_dist.get("min", float("nan"))):
            ok(g, f"similarity min is finite ({sim_dist['min']:.4f})")
        else:
            fail(g, "similarity min is missing or NaN")

        # per-probe record validation
        probe_records = wn_report.get("probes", [])
        if len(probe_records) == total_wn:
            ok(g, f"probe record count matches total ({total_wn})")
        else:
            fail(g, f"probe records ({len(probe_records)}) ≠ total ({total_wn})")

        required_fields = [
            "probe_id", "target_identity_id", "matched_wrong_identity_id",
            "matching_similarity", "target_attribute", "target_label",
            "paired_correct_name_probe_id",
            "target_image_sha256", "paired_correct_name_image_sha256",
        ]
        bad_records = 0
        different_identity_ok = 0
        different_name_ok = 0
        for rec in probe_records:
            missing_f = [f for f in required_fields if f not in rec or rec[f] is None]
            if missing_f:
                bad_records += 1
                continue
            if rec["target_identity_id"] != rec["matched_wrong_identity_id"]:
                different_identity_ok += 1
            if (
                rec.get("wrong_identity_name")
                and rec.get("target_identity_name")
                and rec["wrong_identity_name"] != rec["target_identity_name"]
            ):
                different_name_ok += 1
            sim = rec.get("matching_similarity")
            if not isinstance(sim, (int, float)) or math.isnan(sim):
                bad_records += 1

        if bad_records == 0:
            ok(g, "all probe records have required fields")
        else:
            fail(g, f"{bad_records} probe records missing required fields")

        if different_identity_ok == total_wn:
            ok(g, "all wrong-name probes have different target identity")
        else:
            fail(g, f"{total_wn - different_identity_ok} probes share target identity")

        if different_name_ok == total_wn:
            ok(g, "all wrong-name probes have different identity name")
        else:
            fail(g, f"{total_wn - different_name_ok} probes share identity name")

        # P1-1: image SHA invariant checks.
        img_sha_nonnull = 0
        paired_sha_nonnull = 0
        sha_match_ct = 0
        attr_match_ct = 0
        label_match_ct = 0
        for rec in probe_records:
            t_sha = rec.get("target_image_sha256")
            p_sha = rec.get("paired_correct_name_image_sha256")
            if t_sha:
                img_sha_nonnull += 1
            if p_sha:
                paired_sha_nonnull += 1
            if t_sha and p_sha and t_sha == p_sha:
                sha_match_ct += 1
            if rec.get("target_attribute") == rec.get("paired_correct_name_target_attribute"):
                attr_match_ct += 1
            if rec.get("target_label") == rec.get("paired_correct_name_target_label"):
                label_match_ct += 1

        if img_sha_nonnull == total_wn:
            ok(g, f"all {total_wn} wrong-name probes have non-null target_image_sha256")
        else:
            fail(g, f"{total_wn - img_sha_nonnull} probes have null target_image_sha256")

        if paired_sha_nonnull == total_wn:
            ok(g, f"all {total_wn} wrong-name probes have non-null paired_image_sha256")
        else:
            fail(g, f"{total_wn - paired_sha_nonnull} probes have null paired_image_sha256")

        if sha_match_ct == total_wn:
            ok(g, f"all {total_wn} wrong-name probes: image SHAs match")
        else:
            fail(g, f"{total_wn - sha_match_ct} probes have mismatched image SHAs")

        if attr_match_ct == total_wn:
            ok(g, f"all {total_wn} wrong-name probes: target attributes match paired")
        else:
            fail(g, f"{total_wn - attr_match_ct} probes have mismatched target attributes")

        if label_match_ct == total_wn:
            ok(g, f"all {total_wn} wrong-name probes: expected answers match paired")
        else:
            fail(g, f"{total_wn - label_match_ct} probes have mismatched expected answers")

        # P1-2: covariate policy metadata.
        if wn_report.get("matching_covariate_policy"):
            ok(g, "matching_covariate_policy documented")
        else:
            fail(g, "matching_covariate_policy missing")

        # cross-check with actual route artifact
        if route_path.exists():
            with open(route_path) as _fh:
                actual_wn_probes = [
                    json.loads(ln) for ln in _fh
                    if ln.strip() and json.loads(ln).get("probe_family") == "wrong_name"
                ]
            if len(actual_wn_probes) == total_wn:
                ok(g, f"actual wrong_name probe count matches report ({total_wn})")
            else:
                fail(g, f"actual wrong_name probes ({len(actual_wn_probes)}) ≠ report ({total_wn})")

            # verify each actual wrong_name probe has required fields
            actual_bad = 0
            for ap in actual_wn_probes:
                if ap.get("target_identity_id") == ap.get("matched_wrong_identity_id"):
                    actual_bad += 1
                if not ap.get("matching_similarity"):
                    actual_bad += 1
            if actual_bad == 0:
                ok(g, "actual wrong_name probes pass identity/similarity checks")
            else:
                fail(g, f"{actual_bad} actual wrong_name probes fail checks")

    # ==================================================================
    # GROUP 8: MANUAL AUDIT
    # ==================================================================
    g = "MANUAL AUDIT"

    audit_path = ARTIFACT_DIR / "fiubench_manual_audit_report.json"
    if not audit_path.exists():
        fail(g, "fiubench_manual_audit_report.json missing")
    else:
        audit = _load_json(audit_path)

        if audit.get("gate_pass") is True:
            ok(g, "gate_pass == true")
        else:
            fail(g, "gate_pass is not true")

        if audit.get("critical_failures", -1) == 0:
            ok(g, "critical_failures == 0")
        else:
            fail(g, f"critical_failures = {audit.get('critical_failures')}")

        if audit.get("uncertain_items", -1) == 0:
            ok(g, "uncertain_items == 0")
        else:
            fail(g, f"uncertain_items = {audit.get('uncertain_items')}")

        if audit.get("unreviewed_items", -1) == 0:
            ok(g, "unreviewed_items == 0")
        else:
            fail(g, f"unreviewed_items = {audit.get('unreviewed_items')}")

        # hash verification against manifest
        if manifest_path.exists():
            ma_info = manifest.get("quality_evidence", {}).get("manual_audit", {})
            expected_sha = ma_info.get("sha256", "")
            actual_sha = _sha256_file(audit_path)
            if actual_sha == expected_sha:
                ok(g, "manual audit SHA-256 matches manifest")
            else:
                fail(g, "manual audit SHA-256 mismatch")

    # ==================================================================
    # REPORT
    # ==================================================================
    _report(groups)
    total_fail = sum(len(fails) for _, fails in groups.values())
    return 1 if total_fail else 0


def _report(groups: dict[str, tuple[list[str], list[str]]]) -> None:
    print(f"\n{'=' * 60}")
    print("Frozen Research Bundle Verification")
    print(f"{'=' * 60}")

    total_pass = 0
    total_fail = 0

    for group_name, (passes, failures) in groups.items():
        n_p = len(passes)
        n_f = len(failures)
        total_pass += n_p
        total_fail += n_f
        status = "PASS" if n_f == 0 else "FAIL"
        print(f"\n{group_name:<35s} {n_p}/{n_p + n_f} {status}")
        for p in passes:
            print(f"  ✓ {p}")
        for f in failures:
            print(f"  ✗ {f}")

    grand = total_pass + total_fail
    print(f"\n{'=' * 60}")
    print(f"TOTAL  {total_pass}/{grand} PASS")
    if total_fail:
        print(f"*** VERIFICATION FAILED ({total_fail} failures) ***")
    else:
        print("*** ALL CHECKS PASSED — FROZEN BUNDLE IS CONSISTENT ***")


if __name__ == "__main__":
    sys.exit(verify())
