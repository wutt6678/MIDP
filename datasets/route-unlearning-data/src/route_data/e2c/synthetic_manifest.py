"""E2C synthetic identity manifests and frozen artifacts.

Generates deterministic identity populations, aliases, image splits,
target-fact mappings (true and shuffled), visual control records,
and wrong-name pairings for the E2C controlled route experiment.

All generation uses seed 17 and produces SHA-256-bound JSON manifests.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Canonical alias pool
# --------------------------------------------------------------------------- #

CANONICAL_ALIASES: list[str] = [
    "Aven", "Bira", "Caro", "Deni", "Eris",
    "Faro", "Gela", "Hani", "Ivoa", "Jora",
    # Calibration aliases
    "Kael", "Luma",
]

EXPERIMENTAL_COUNT = 10
CALIBRATION_COUNT = 2
TOTAL_IDENTITIES = EXPERIMENTAL_COUNT + CALIBRATION_COUNT

IMAGES_PER_IDENTITY = 16
TRAIN_COUNT = 10
VAL_COUNT = 3
TEST_COUNT = 3

DEFAULT_SEED = 17


# --------------------------------------------------------------------------- #
# SHA-256 helpers
# --------------------------------------------------------------------------- #

def sha256_file(path: str | Path) -> str:
    """Stream-hash a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(data: Any) -> str:
    """SHA-256 of canonical JSON (sorted keys, no extra whitespace)."""
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class IdentityRecord:
    identity_id: str
    alias: str
    alias_token_ids: list[int]
    role: str  # "experimental" or "calibration"
    train_images: list[str] = field(default_factory=list)
    validation_images: list[str] = field(default_factory=list)
    test_images: list[str] = field(default_factory=list)


@dataclass
class ImageRecord:
    identity_id: str
    image_id: str
    image_path: str
    image_sha256: str
    split: str  # "train", "validation", "test"


@dataclass
class AuditRecord:
    identity_id: str
    image_id: str
    identity_consistent: bool = True
    duplicate: bool = False
    corrupted: bool = False
    watermark: bool = False
    alias_leakage: bool = False
    target_fact_leakage: bool = False
    notes: str = ""


@dataclass
class VisualControlRecord:
    image_id: str
    identity_id: str
    controls: dict[str, bool] = field(default_factory=dict)
    source: str = "generator_metadata"


# --------------------------------------------------------------------------- #
# Identity generation
# --------------------------------------------------------------------------- #

def generate_identity_ids(
    n_experimental: int = EXPERIMENTAL_COUNT,
    n_calibration: int = CALIBRATION_COUNT,
) -> tuple[list[str], list[str]]:
    """Generate identity IDs.

    Returns (experimental_ids, calibration_ids).
    """
    experimental = [f"syn_{i:02d}" for i in range(n_experimental)]
    calibration = [
        f"syn_cal_{i:02d}"
        for i in range(n_calibration)
    ]
    return experimental, calibration


def assign_aliases(
    experimental_ids: list[str],
    calibration_ids: list[str],
    aliases: list[str] | None = None,
) -> dict[str, str]:
    """Assign one alias per identity from the canonical pool.

    Returns mapping: identity_id -> alias.
    """
    if aliases is None:
        aliases = CANONICAL_ALIASES

    all_ids = list(experimental_ids) + list(calibration_ids)
    if len(all_ids) > len(aliases):
        raise ValueError(
            f"Need {len(all_ids)} aliases but only {len(aliases)} available"
        )

    # Verify uniqueness
    if len(set(aliases[:len(all_ids)])) != len(all_ids):
        raise ValueError("Aliases are not unique")

    return {id_: alias for id_, alias in zip(all_ids, aliases)}


def record_alias_token_ids(
    alias_map: dict[str, str],
    tokenizer: Any,
) -> dict[str, list[int]]:
    """Tokenize every alias and return identity_id -> token_ids."""
    result: dict[str, list[int]] = {}
    for id_, alias in alias_map.items():
        ids = tokenizer.encode(alias, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Alias {alias!r} tokenized to empty sequence")
        result[id_] = ids
    return result


# --------------------------------------------------------------------------- #
# Image split generation
# --------------------------------------------------------------------------- #

def generate_image_splits(
    identity_ids: list[str],
    images_per_identity: int = IMAGES_PER_IDENTITY,
    n_train: int = TRAIN_COUNT,
    n_val: int = VAL_COUNT,
    n_test: int = TEST_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, list[int]]]:
    """Deterministically split image indices into train/val/test.

    Returns mapping: identity_id -> {"train": [...], "validation": [...], "test": [...]}.
    Indices are 0-based within each identity's image set.
    """
    assert n_train + n_val + n_test == images_per_identity, (
        f"Split counts {n_train}+{n_val}+{n_test} != {images_per_identity}"
    )

    rng = random.Random(seed)
    splits: dict[str, dict[str, list[int]]] = {}

    for id_ in sorted(identity_ids):
        indices = list(range(images_per_identity))
        rng.shuffle(indices)
        splits[id_] = {
            "train": sorted(indices[:n_train]),
            "validation": sorted(indices[n_train:n_train + n_val]),
            "test": sorted(indices[n_train + n_val:]),
        }

    return splits


# --------------------------------------------------------------------------- #
# Target fact mapping
# --------------------------------------------------------------------------- #

def generate_true_mapping(
    experimental_ids: list[str],
    n_yes: int = 5,
    n_no: int = 5,
    seed: int = DEFAULT_SEED,
) -> dict[str, str]:
    """Generate balanced binary target mapping deterministically.

    Returns mapping: identity_id -> "Yes" or "No".
    Only experimental identities are included.
    """
    if len(experimental_ids) != n_yes + n_no:
        raise ValueError(
            f"Expected {n_yes + n_no} experimental identities, "
            f"got {len(experimental_ids)}"
        )

    rng = random.Random(seed)
    ids = sorted(experimental_ids)
    rng.shuffle(ids)

    mapping: dict[str, str] = {}
    for i, id_ in enumerate(ids):
        mapping[id_] = "Yes" if i < n_yes else "No"

    return mapping


def generate_shuffled_mapping(
    true_mapping: dict[str, str],
    seed: int = DEFAULT_SEED,
) -> dict[str, str]:
    """Generate shuffled mapping that flips all labels.

    For binary balanced data, every identity gets the opposite label.
    This maximizes the causal control signal.

    Returns mapping: identity_id -> flipped "Yes"/"No".
    """
    shuffled: dict[str, str] = {}
    for id_, label in sorted(true_mapping.items()):
        shuffled[id_] = "No" if label == "Yes" else "Yes"

    # Verify all labels are flipped
    for id_, value in true_mapping.items():
        assert shuffled[id_] != value, (
            f"Shuffled mapping did not flip {id_}"
        )

    return shuffled


def generate_calibration_mapping(
    calibration_ids: list[str],
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, str], dict[str, str]]:
    """Generate balanced true + shuffled mappings for calibration identities.

    With 2 calibration IDs: 1 Yes, 1 No.
    Returns (true_mapping, shuffled_mapping).
    """
    if len(calibration_ids) < 2:
        raise ValueError(
            f"Need at least 2 calibration identities, got {len(calibration_ids)}"
        )
    rng = random.Random(seed)
    ids = sorted(calibration_ids)
    rng.shuffle(ids)
    n_yes = len(ids) // 2
    true_map = {id_: ("Yes" if i < n_yes else "No")
                for i, id_ in enumerate(ids)}
    shuf_map = {id_: ("No" if v == "Yes" else "Yes")
                for id_, v in true_map.items()}
    return true_map, shuf_map


def finalize_alias_tokenization(
    alias_map: dict[str, str],
    tokenizer: Any,
    *,
    tokenizer_id: str = "Qwen/Qwen3.5-9B",
    tokenizer_revision: str = "",
) -> list[dict[str, Any]]:
    """P1-1: Tokenize all aliases and return metadata records.

    Each record contains alias, token IDs, token count, tokenizer info.
    Hard-fails if any alias tokenizes to empty.
    """
    records: list[dict[str, Any]] = []
    for id_, alias in sorted(alias_map.items()):
        ids = tokenizer.encode(alias, add_special_tokens=False)
        if not ids:
            raise ValueError(
                f"Alias {alias!r} for {id_} tokenized to empty sequence"
            )
        records.append({
            "identity_id": id_,
            "alias": alias,
            "alias_token_ids": ids,
            "alias_token_count": len(ids),
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
        })
    return records


def validate_alias_tokenization(
    alias_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate alias tokenization completeness and comparability."""
    errors: list[str] = []
    counts = []
    for rec in alias_records:
        tc = rec.get("alias_token_count", 0)
        if tc == 0:
            errors.append(
                f"Alias {rec.get('alias', '?')}: token count is 0"
            )
        if not rec.get("alias_token_ids"):
            errors.append(
                f"Alias {rec.get('alias', '?')}: token IDs missing"
            )
        if not rec.get("tokenizer_id"):
            errors.append(
                f"Alias {rec.get('alias', '?')}: tokenizer_id missing"
            )
        counts.append(tc)

    report = {
        "pass": len(errors) == 0,
        "errors": errors,
        "min_token_count": min(counts) if counts else 0,
        "max_token_count": max(counts) if counts else 0,
        "mean_token_count": sum(counts) / len(counts) if counts else 0,
        "per_alias": alias_records,
    }
    if counts and max(counts) - min(counts) > 2:
        report["warning"] = (
            f"Alias token lengths vary significantly "
            f"(min={min(counts)}, max={max(counts)})"
        )
    return report


# --------------------------------------------------------------------------- #
# Wrong-name pairing
# --------------------------------------------------------------------------- #

def generate_wrong_name_pairs(
    true_mapping: dict[str, str],
    alias_map: dict[str, str],
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Generate wrong-name pairings for WN probes.

    For each identity A, find an identity B where target_fact(B) != target_fact(A),
    deterministically with seed.

    Returns list of pairing records.
    """
    rng = random.Random(seed)
    ids = sorted(true_mapping.keys())

    pairs: list[dict[str, Any]] = []
    for id_a in ids:
        label_a = true_mapping[id_a]
        # Find candidates with opposite label
        candidates = [
            id_b for id_b in ids
            if id_b != id_a and true_mapping[id_b] != label_a
        ]
        if not candidates:
            raise ValueError(
                f"No opposite-label candidate for {id_a} (label={label_a})"
            )
        rng.shuffle(candidates)
        id_b = candidates[0]

        pairs.append({
            "identity_id": id_a,
            "correct_alias": alias_map[id_a],
            "wrong_alias": alias_map[id_b],
            "true_label": label_a,
            "wrong_label": true_mapping[id_b],
        })

    return pairs


# --------------------------------------------------------------------------- #
# Manifest I/O
# --------------------------------------------------------------------------- #

def write_json_manifest(
    data: Any,
    path: str | Path,
    *,
    indent: int = 2,
) -> str:
    """Write a JSON manifest and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, sort_keys=True)
        f.write("\n")
    return sha256_file(path)


def load_json_manifest(path: str | Path) -> Any:
    """Load a JSON manifest."""
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Prompt registry
# --------------------------------------------------------------------------- #

PROMPT_REGISTRY: dict[str, dict[str, Any]] = {
    "e2c_i2n_train_v1_01": {
        "family": "image_to_identity",
        "text": "Who is shown in this image? Answer with the synthetic identity name.",
    },
    "e2c_i2n_train_v1_02": {
        "family": "image_to_identity",
        "text": "Identify the synthetic person shown here. Return only the alias.",
    },
    "e2c_name_fact_train_v1_01": {
        "family": "name_to_attribute",
        "text": "Does {alias} have property Z? Answer Yes or No.",
    },
    "e2c_name_fact_train_v1_02": {
        "family": "name_to_attribute",
        "text": "Is the synthetic identity {alias} assigned to category Z? Answer Yes or No.",
    },
    "e2c_direct_fact_train_v1_01": {
        "family": "image_to_attribute",
        "text": "Does the person in this image have property Z? Answer Yes or No.",
    },
    "e2c_direct_fact_train_v1_02": {
        "family": "image_to_attribute",
        "text": "Is the synthetic person shown here assigned to category Z? Answer Yes or No.",
    },
    # Test prompts (kept distinct from training templates)
    "e2c_test_i2n_v1": {
        "family": "image_to_identity",
        "text": "Who is shown? Answer with the synthetic identity name only.",
    },
    "e2c_test_name_v1": {
        "family": "name_to_attribute",
        "text": "Does {alias} have property Z? Answer Yes or No.",
    },
    "e2c_test_dv_v1": {
        "family": "image_to_attribute",
        "text": "Does the person shown have property Z? Answer Yes or No.",
    },
    "e2c_test_ipn_v1": {
        "family": "image_plus_name",
        "text": "This is {alias}. Does this person have property Z? Answer Yes or No.",
    },
    "e2c_test_wn_v1": {
        "family": "wrong_name",
        "text": "This is {wrong_alias}. Does this person have property Z? Answer Yes or No.",
    },
    "e2c_test_vtc_v1": {
        "family": "visual_text_conflict",
        "text": "This is {wrong_alias}. The property-Z assignment for {wrong_alias} is {wrong_label}. What is the property-Z assignment for the person in the image? Answer Yes or No.",
    },
    "e2c_test_visual_control_smile": {
        "family": "visual_control",
        "text": "Is the person smiling? Answer Yes or No.",
    },
    "e2c_test_visual_control_glasses": {
        "family": "visual_control",
        "text": "Does the person wear eyeglasses? Answer Yes or No.",
    },
    "e2c_test_visual_control_hat": {
        "family": "visual_control",
        "text": "Is the person wearing a hat? Answer Yes or No.",
    },
}


def prompt_registry_sha() -> str:
    return sha256_json(PROMPT_REGISTRY)


def get_prompt(prompt_id: str, **kwargs: str) -> str:
    """Get a prompt text by ID, substituting format placeholders."""
    entry = PROMPT_REGISTRY[prompt_id]
    text = entry["text"]
    if kwargs:
        text = text.format(**kwargs)
    return text


# --------------------------------------------------------------------------- #
# Full manifest generation pipeline
# --------------------------------------------------------------------------- #

def generate_e2c_manifests(
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    image_dir: str | Path | None = None,
) -> dict[str, str]:
    """Generate all frozen E2C manifests.

    Returns a dict of manifest_name -> sha256.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shas: dict[str, str] = {}

    # 1. Generate identities
    exp_ids, cal_ids = generate_identity_ids()
    all_ids = exp_ids + cal_ids

    # 2. Assign aliases
    alias_map = assign_aliases(exp_ids, cal_ids)

    # 3. Build identity manifest (token IDs filled later when tokenizer available)
    identity_records = []
    for id_ in all_ids:
        role = "experimental" if id_ in exp_ids else "calibration"
        identity_records.append({
            "identity_id": id_,
            "alias": alias_map[id_],
            "alias_token_ids": [],  # filled by tokenization step
            "role": role,
            "train_images": [],
            "validation_images": [],
            "test_images": [],
        })

    sha = write_json_manifest(identity_records, output_dir / "synthetic_identity_manifest.json")
    shas["synthetic_identity_manifest"] = sha

    # 4. Generate image splits
    splits = generate_image_splits(all_ids, seed=seed)

    image_split_records = []
    for id_ in all_ids:
        for split_name in ("train", "validation", "test"):
            for idx in splits[id_][split_name]:
                img_id = f"{id_}_img_{idx:03d}"
                img_path = f"e2c/data/processed/{id_}/{img_id}.png"
                if image_dir:
                    img_path = str(Path(image_dir) / id_ / f"{img_id}.png")
                image_split_records.append({
                    "identity_id": id_,
                    "image_id": img_id,
                    "image_path": img_path,
                    "image_sha256": "",  # filled when images exist
                    "source_render_id": "",  # P0-2: filled when lineage known
                    "generation_type": "independent_render",  # P0-2 default
                    "augmentation_parent_id": None,
                    "split": split_name,
                })

    sha = write_json_manifest(image_split_records, output_dir / "e2c_image_split.json")
    shas["e2c_image_split"] = sha

    # 5. True mapping
    true_mapping = generate_true_mapping(exp_ids, seed=seed)
    sha = write_json_manifest(true_mapping, output_dir / "synthetic_attribute_mapping.json")
    shas["synthetic_attribute_mapping"] = sha

    # 6. Shuffled mapping
    shuffled_mapping = generate_shuffled_mapping(true_mapping, seed=seed)
    sha = write_json_manifest(shuffled_mapping, output_dir / "synthetic_attribute_mapping_shuffled.json")
    shas["synthetic_attribute_mapping_shuffled"] = sha

    # 7. Wrong-name pairs
    wn_pairs = generate_wrong_name_pairs(true_mapping, alias_map, seed=seed)
    sha = write_json_manifest(wn_pairs, output_dir / "e2c_wrong_name_pairs.json")
    shas["e2c_wrong_name_pairs"] = sha

    # 8. Identity audit — P0-6: fail-closed (all fields null/pending)
    audit_records = []
    for id_ in all_ids:
        for idx in range(IMAGES_PER_IDENTITY):
            img_id = f"{id_}_img_{idx:03d}"
            audit_records.append({
                "identity_id": id_,
                "image_id": img_id,
                "audit_status": "pending",
                "reviewer": None,
                "review_timestamp": None,
                "identity_consistent": None,
                "duplicate": None,
                "corrupted": None,
                "watermark": None,
                "alias_leakage": None,
                "target_fact_leakage": None,
                "notes": "",
            })

    sha = write_json_manifest(audit_records, output_dir / "e2c_identity_audit.json")
    shas["e2c_identity_audit"] = sha

    # 9. Visual controls template
    visual_control_records = []
    for id_ in all_ids:
        for idx in range(IMAGES_PER_IDENTITY):
            img_id = f"{id_}_img_{idx:03d}"
            visual_control_records.append({
                "image_id": img_id,
                "identity_id": id_,
                "controls": {
                    "smiling": False,
                    "eyeglasses": False,
                    "hat": False,
                },
                "source": "generator_metadata",
            })

    sha = write_json_manifest(visual_control_records, output_dir / "e2c_visual_controls.json")
    shas["e2c_visual_controls"] = sha

    # 10. Prompt registry
    sha = write_json_manifest(PROMPT_REGISTRY, output_dir / "e2c_prompt_registry.json")
    shas["e2c_prompt_registry"] = sha

    return shas
