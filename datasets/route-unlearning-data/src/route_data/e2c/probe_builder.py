"""E2C route probe builder.

Constructs probe families for held-out test evaluation:
    I2N          — image-to-name recognition
    NAME         — alias-only synthetic fact
    DV-syn       — image-only synthetic fact (direct visual)
    IPN-syn      — image plus correct alias
    WN           — wrong-name intervention
    VTC          — visual/text conflict
    VISUAL-CONTROL — visible attribute controls

Every probe is a frozen JSON record with probe_id, family, expected_answer,
and all metadata needed for evaluation and analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .synthetic_manifest import (
    get_prompt,
    sha256_file,
)

# --------------------------------------------------------------------------- #
# Probe builders per family
# --------------------------------------------------------------------------- #

def build_i2n_probes(
    test_images: list[dict[str, Any]],
    alias_map: dict[str, str],
) -> list[dict[str, Any]]:
    """I2N — image-to-name recognition probes.

    Input: held-out test image.
    Expected: synthetic alias.
    """
    probes = []
    for img in test_images:
        id_ = img["identity_id"]
        probes.append({
            "probe_id": f"e2c_test_{id_}_{img['image_id']}_i2n",
            "family": "I2N",
            "identity_id": id_,
            "image_id": img["image_id"],
            "image_path": img["image_path"],
            "image_sha256": img.get("image_sha256", ""),
            "prompt_id": "e2c_test_i2n_v1",
            "prompt": get_prompt("e2c_test_i2n_v1"),
            "expected_answer": alias_map[id_],
            "correct_alias": alias_map[id_],
            "split": "test",
        })
    return probes


def build_name_probes(
    experimental_ids: list[str],
    alias_map: dict[str, str],
    true_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    """NAME — alias-only synthetic fact probes.

    Input: Does {alias} have property Z?
    Expected: Yes or No from true mapping.
    """
    probes = []
    for id_ in sorted(experimental_ids):
        alias = alias_map[id_]
        probes.append({
            "probe_id": f"e2c_test_{id_}_name",
            "family": "NAME",
            "identity_id": id_,
            "image_id": None,
            "image_path": None,
            "prompt_id": "e2c_test_name_v1",
            "prompt": get_prompt("e2c_test_name_v1", alias=alias),
            "expected_answer": true_mapping[id_],
            "correct_alias": alias,
            "true_mapping": true_mapping[id_],
            "split": "test",
        })
    return probes


def build_dv_probes(
    test_images: list[dict[str, Any]],
    true_mapping: dict[str, str],
    alias_map: dict[str, str],
) -> list[dict[str, Any]]:
    """DV-syn — image-only synthetic fact probes (direct visual).

    Input: held-out image. No alias in prompt.
    Expected: true mapping label.
    """
    probes = []
    for img in test_images:
        id_ = img["identity_id"]
        probes.append({
            "probe_id": f"e2c_test_{id_}_{img['image_id']}_dv",
            "family": "DV_syn",
            "identity_id": id_,
            "image_id": img["image_id"],
            "image_path": img["image_path"],
            "image_sha256": img.get("image_sha256", ""),
            "prompt_id": "e2c_test_dv_v1",
            "prompt": get_prompt("e2c_test_dv_v1"),
            "expected_answer": true_mapping[id_],
            "correct_alias": alias_map[id_],
            "true_mapping": true_mapping[id_],
            "split": "test",
        })
    return probes


def build_ipn_probes(
    test_images: list[dict[str, Any]],
    alias_map: dict[str, str],
    true_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    """IPN-syn — image plus correct alias probes.

    Input: held-out image + "This is {alias}."
    Expected: true mapping label.
    """
    probes = []
    for img in test_images:
        id_ = img["identity_id"]
        alias = alias_map[id_]
        probes.append({
            "probe_id": f"e2c_test_{id_}_{img['image_id']}_ipn",
            "family": "IPN_syn",
            "identity_id": id_,
            "image_id": img["image_id"],
            "image_path": img["image_path"],
            "image_sha256": img.get("image_sha256", ""),
            "prompt_id": "e2c_test_ipn_v1",
            "prompt": get_prompt("e2c_test_ipn_v1", alias=alias),
            "expected_answer": true_mapping[id_],
            "correct_alias": alias,
            "presented_alias": alias,
            "true_mapping": true_mapping[id_],
            "split": "test",
        })
    return probes


def build_wn_probes(
    test_images: list[dict[str, Any]],
    wn_pairs: list[dict[str, Any]],
    alias_map: dict[str, str],
    true_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    """WN — wrong-name intervention probes.

    For each test image of identity A, present identity B's alias
    (where B has opposite target label).

    Expected answer: true image identity's fact (A's label).
    """
    # Build lookup: identity_id -> wrong name record
    wn_lookup = {p["identity_id"]: p for p in wn_pairs}

    probes = []
    for img in test_images:
        id_ = img["identity_id"]
        if id_ not in wn_lookup:
            continue
        pair = wn_lookup[id_]
        wrong_alias = pair["wrong_alias"]
        true_label = true_mapping[id_]

        probes.append({
            "probe_id": f"e2c_test_{id_}_{img['image_id']}_wn",
            "family": "WN",
            "identity_id": id_,
            "image_id": img["image_id"],
            "image_path": img["image_path"],
            "image_sha256": img.get("image_sha256", ""),
            "prompt_id": "e2c_test_wn_v1",
            "prompt": get_prompt("e2c_test_wn_v1", wrong_alias=wrong_alias),
            "expected_answer": true_label,
            "correct_alias": alias_map[id_],
            "presented_alias": wrong_alias,
            "true_mapping": true_label,
            "presented_name_attribute": pair["wrong_label"],
            "split": "test",
        })
    return probes


def build_vtc_probes(
    test_images: list[dict[str, Any]],
    wn_pairs: list[dict[str, Any]],
    alias_map: dict[str, str],
    true_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    """VTC — visual/text conflict probes.

    Image of A, text says "This is B. B has property Z."
    Expected: true image identity's fact.
    """
    wn_lookup = {p["identity_id"]: p for p in wn_pairs}

    probes = []
    for img in test_images:
        id_ = img["identity_id"]
        if id_ not in wn_lookup:
            continue
        pair = wn_lookup[id_]
        wrong_alias = pair["wrong_alias"]
        true_label = true_mapping[id_]

        probes.append({
            "probe_id": f"e2c_test_{id_}_{img['image_id']}_vtc",
            "family": "VTC",
            "identity_id": id_,
            "image_id": img["image_id"],
            "image_path": img["image_path"],
            "image_sha256": img.get("image_sha256", ""),
            "prompt_id": "e2c_test_vtc_v1",
            "prompt": get_prompt(
                "e2c_test_vtc_v1", wrong_alias=wrong_alias,
            ),
            "expected_answer": true_label,
            "correct_alias": alias_map[id_],
            "presented_alias": wrong_alias,
            "true_mapping": true_label,
            "presented_name_attribute": pair["wrong_label"],
            "split": "test",
        })
    return probes


def build_visual_control_probes(
    test_images: list[dict[str, Any]],
    visual_controls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """VISUAL-CONTROL — visible attribute control probes.

    For each test image, create questions about visible attributes.
    """
    # Build lookup: image_id -> controls
    vc_lookup = {vc["image_id"]: vc for vc in visual_controls}

    prompt_map = {
        "smiling": "e2c_test_visual_control_smile",
        "eyeglasses": "e2c_test_visual_control_glasses",
        "hat": "e2c_test_visual_control_hat",
    }

    probes = []
    for img in test_images:
        img_id = img["image_id"]
        if img_id not in vc_lookup:
            continue
        controls = vc_lookup[img_id]["controls"]

        for attr, prompt_id in prompt_map.items():
            if attr not in controls:
                continue
            expected = "Yes" if controls[attr] else "No"
            probes.append({
                "probe_id": f"e2c_test_{img_id}_vc_{attr}",
                "family": "VISUAL_CONTROL",
                "identity_id": img["identity_id"],
                "image_id": img_id,
                "image_path": img["image_path"],
                "image_sha256": img.get("image_sha256", ""),
                "prompt_id": prompt_id,
                "prompt": get_prompt(prompt_id),
                "expected_answer": expected,
                "visual_attribute": attr,
                "split": "test",
            })
    return probes


# --------------------------------------------------------------------------- #
# Full probe generation
# --------------------------------------------------------------------------- #

def build_all_probes(
    *,
    image_splits: list[dict[str, Any]],
    alias_map: dict[str, str],
    true_mapping: dict[str, str],
    wn_pairs: list[dict[str, Any]],
    visual_controls: list[dict[str, Any]],
    experimental_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Build all probe families for held-out test images.

    Returns dict mapping family name -> list of probes.
    """
    # Extract test images for experimental identities only
    test_images = [
        rec for rec in image_splits
        if rec["split"] == "test" and rec["identity_id"] in set(experimental_ids)
    ]

    probes: dict[str, list[dict[str, Any]]] = {}

    probes["I2N"] = build_i2n_probes(test_images, alias_map)
    probes["NAME"] = build_name_probes(experimental_ids, alias_map, true_mapping)
    probes["DV_syn"] = build_dv_probes(test_images, true_mapping, alias_map)
    probes["IPN_syn"] = build_ipn_probes(test_images, alias_map, true_mapping)
    probes["WN"] = build_wn_probes(test_images, wn_pairs, alias_map, true_mapping)
    probes["VTC"] = build_vtc_probes(test_images, wn_pairs, alias_map, true_mapping)
    probes["VISUAL_CONTROL"] = build_visual_control_probes(test_images, visual_controls)

    return probes


def validate_probes(
    probes: dict[str, list[dict[str, Any]]],
    *,
    experimental_ids: list[str],
    test_image_count: int,
) -> dict[str, Any]:
    """Validate probe construction.

    Returns report dict. Raises ValueError on violations.
    """
    errors: list[str] = []

    # All test identities covered in I2N
    i2n_ids = {p["identity_id"] for p in probes["I2N"]}
    for id_ in experimental_ids:
        if id_ not in i2n_ids:
            errors.append(f"I2N missing identity {id_}")

    # All held-out test images covered in DV-syn
    _dv_images = {p["image_id"] for p in probes["DV_syn"]}

    # Wrong-name: alias != true alias
    for p in probes.get("WN", []):
        if p["presented_alias"] == p["correct_alias"]:
            errors.append(f"WN probe {p['probe_id']}: wrong alias == true alias")

    # Wrong-name: label always opposite
    for p in probes.get("WN", []):
        if p["presented_name_attribute"] == p["true_mapping"]:
            errors.append(
                f"WN probe {p['probe_id']}: wrong label same as true label"
            )

    # VTC: label conflict always opposite
    for p in probes.get("VTC", []):
        if p["presented_name_attribute"] == p["true_mapping"]:
            errors.append(
                f"VTC probe {p['probe_id']}: conflict label same as true label"
            )

    # Unique probe IDs across all families
    all_ids: list[str] = []
    for family_probes in probes.values():
        for p in family_probes:
            all_ids.append(p["probe_id"])
    if len(all_ids) != len(set(all_ids)):
        errors.append("Duplicate probe IDs detected")

    # Exact family counts
    expected_i2n = test_image_count
    if len(probes["I2N"]) != expected_i2n:
        errors.append(
            f"I2N count {len(probes['I2N'])} != expected {expected_i2n}"
        )

    expected_name = len(experimental_ids)
    if len(probes["NAME"]) != expected_name:
        errors.append(
            f"NAME count {len(probes['NAME'])} != expected {expected_name}"
        )

    report = {
        "pass": len(errors) == 0,
        "errors": errors,
        "family_counts": {k: len(v) for k, v in probes.items()},
        "total_probes": len(all_ids),
    }

    if errors:
        raise ValueError(
            f"Probe validation failures ({len(errors)}):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return report


def write_probe_jsonl(
    probes: list[dict[str, Any]],
    path: str | Path,
) -> str:
    """Write probes as JSONL and return SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(probe, sort_keys=True) + "\n" for probe in probes)
    return sha256_file(path)
