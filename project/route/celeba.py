"""CelebA source access and manifest construction (PLAN.md section 5).

Only metadata manifests are written; CelebA images are never copied or
redistributed (CelebA license constraint). The manifest schema follows
PLAN.md section 5:

    {
      "image_file": "000001.jpg",
      "celeba_identity_id": 1234,
      "alias": "Vela_07",
      "property": "DAX",
      "split": "train",
      "face_bbox": [20, 50, 158, 205],
      "marker_bbox": [8, 8, 48, 48]
    }

The CelebA aligned crops (178x218) carry no explicit face boxes, so a fixed
conservative face region is used (PLAN.md section 5, final paragraph).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from route.marker import DAX, WUG

ALIAS_PREFIX = "Vela"
DEFAULT_SEED = 20260804

# Fixed boxes for the 178x218 aligned CelebA crops (x0, y0, x1, y1).
DEFAULT_FACE_BBOX = (20, 50, 158, 205)

_MARKER_LOCATIONS = {
    "top_left": lambda w, h, s, m: (m, m, m + s, m + s),
    "top_right": lambda w, h, s, m: (w - m - s, m, w - m, m + s),
    "bottom_left": lambda w, h, s, m: (m, h - m - s, m + s, h - m),
    "bottom_right": lambda w, h, s, m: (w - m - s, h - m - s, w - m, h - m),
}


def marker_bbox(image_size: tuple, marker_size_px: int = 40,
                marker_location: str = "top_left", margin: int = 8) -> tuple:
    """Marker bounding box for a given image size / location (PLAN.md sec. 4)."""
    w, h = image_size
    return _MARKER_LOCATIONS[marker_location](w, h, marker_size_px, margin)


def parse_identity_file(path: str | Path) -> dict[str, int]:
    """Parse CelebA identity_CelebA.txt -> {image_file: identity_id}."""
    mapping = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            mapping[parts[0]] = int(parts[1])
    return mapping


def count_images_per_identity(image_to_id: dict[str, int]) -> dict[int, list[str]]:
    """Group image files by identity id."""
    by_id: dict[int, list[str]] = {}
    for img, ident in image_to_id.items():
        by_id.setdefault(ident, []).append(img)
    for files in by_id.values():
        files.sort()
    return by_id


def select_identities(by_id: dict[int, list[str]], n_identities: int = 10,
                      min_images: int = 20, seed: int = DEFAULT_SEED) -> list[int]:
    """Sample exactly `n_identities` eligible identities (PLAN.md section 5)."""
    eligible = sorted(i for i, files in by_id.items() if len(files) >= min_images)
    if len(eligible) < n_identities:
        raise ValueError(
            f"Only {len(eligible)} identities have >= {min_images} images; "
            f"need {n_identities}"
        )
    rng = random.Random(seed)
    selected = sorted(rng.sample(eligible, n_identities))
    return selected


def assign_aliases(selected_ids: list[int], seed: int) -> dict[int, str]:
    """Assign arbitrary aliases with no semantic relation to identity."""
    aliases = [f"{ALIAS_PREFIX}_{i:02d}" for i in range(1, len(selected_ids) + 1)]
    rng = random.Random(seed + 1)
    shuffled = aliases[:]
    rng.shuffle(shuffled)
    return dict(zip(selected_ids, shuffled))


def assign_properties(selected_ids: list[int], seed: int) -> dict[int, str]:
    """Assign half the identities to DAX and half to WUG."""
    if len(selected_ids) % 2 != 0:
        raise ValueError("n_identities must be even for balanced DAX/WUG assignment")
    rng = random.Random(seed + 2)
    shuffled = selected_ids[:]
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    props = {i: DAX for i in shuffled[:half]}
    props.update({i: WUG for i in shuffled[half:]})
    return props


def split_images(files: list[str], train: int, val: int, test: int,
                 seed: int) -> dict[str, list[str]]:
    """Deterministic per-identity train/val/test split."""
    if len(files) < train + val + test:
        raise ValueError(
            f"Identity has {len(files)} images; need {train + val + test}"
        )
    rng = random.Random(seed + 3)
    shuffled = files[:]
    rng.shuffle(shuffled)
    return {
        "train": shuffled[:train],
        "validation": shuffled[train:train + val],
        "test": shuffled[train + val:train + val + test],
    }


def build_manifest_rows(by_id, selected_ids, aliases, properties, splits_cfg,
                        seed, face_bbox, marker_box):
    """Build per-image manifest rows for all splits."""
    rows_by_split = {"train": [], "validation": [], "test": []}
    for ident in selected_ids:
        split = split_images(by_id[ident], splits_cfg["train"],
                             splits_cfg["validation"], splits_cfg["test"], seed)
        for split_name, files in split.items():
            for img in files:
                rows_by_split[split_name].append({
                    "image_file": img,
                    "celeba_identity_id": ident,
                    "alias": aliases[ident],
                    "property": properties[ident],
                    "split": split_name,
                    "face_bbox": list(face_bbox),
                    "marker_bbox": list(marker_box),
                })
    for rows in rows_by_split.values():
        rows.sort(key=lambda r: r["image_file"])
    return rows_by_split


def verify_no_cross_split(rows_by_split: dict[str, list[dict]]) -> None:
    """Ensure no image appears in more than one split."""
    seen: dict[str, str] = {}
    for split_name, rows in rows_by_split.items():
        for row in rows:
            img = row["image_file"]
            if img in seen:
                raise ValueError(
                    f"Image {img} appears in both '{seen[img]}' and '{split_name}'"
                )
            seen[img] = split_name


def write_manifests(output_dir: str | Path, rows_by_split: dict,
                    identity_info: list[dict], meta: dict) -> Path:
    """Write identity_manifest.json + train/validation/test JSONL files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"identities": identity_info, **meta}
    with open(output_dir / "identity_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    for split_name, rows in rows_by_split.items():
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"  Wrote {path} ({len(rows)} rows)")

    return output_dir / "identity_manifest.json"


def load_manifest(path: str | Path) -> list[dict]:
    """Load a JSONL manifest split file."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_identity_manifest(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
