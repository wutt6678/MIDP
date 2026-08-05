"""Data-layer smoke tests for the route MVP (no GPU, no model).

Run from repo root:
    python -m pytest tests/test_route_data.py -q
or simply:
    python tests/test_route_data.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from route.celeba import load_identity_manifest, load_manifest
from route.marker import make_variant
from route.prompts import (
    ALIAS_PROPERTY,
    DIRECT,
    EVAL_VARIANTS,
    IDENTITY,
    JOINT,
    build_examples,
)
from vlm_spatial.regions import (
    bbox_to_image_token_indices,
    find_background_patch_indices,
    layer_thirds,
)
from vlm_spatial.route_dataset import RouteDataset, make_eval_record

REPO = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO / "data" / "celeba_route_mvp" / "manifests"
IDENTITY_META = load_identity_manifest(MANIFEST_DIR / "identity_manifest.json")
CELEBA_ROOT = IDENTITY_META["celeba_root"]

# Expected per-row example counts by condition.
EXPECTED_PER_ROW = {
    "direct": 1,          # property question only
    "joint": 1,           # combined identity+property answer
    "mediated": 2,        # identity (image) + alias->property (text)
    "mixed": 3,           # direct + identity + alias->property
}
# Image-backed examples per row (text-only C3b rows excluded).
EXPECTED_IMAGE_PER_ROW = {
    "direct": 1, "joint": 1, "mediated": 1, "mixed": 2,
}


def test_build_examples_counts():
    rows = load_manifest(MANIFEST_DIR / "train.jsonl")
    assert len(rows) == 120
    for condition, per_row in EXPECTED_PER_ROW.items():
        examples = build_examples(condition, rows)
        assert len(examples) == len(rows) * per_row, condition
        # Text-only rows must carry variant None and no image requirement.
        n_text = sum(1 for e in examples if e["variant"] is None)
        n_img = len(examples) - n_text
        assert n_img == len(rows) * EXPECTED_IMAGE_PER_ROW[condition], condition
        if condition == "mediated":
            types = {e["example_type"] for e in examples}
            assert types == {IDENTITY, ALIAS_PROPERTY}
        elif condition == "mixed":
            types = {e["example_type"] for e in examples}
            assert types == {DIRECT, IDENTITY, ALIAS_PROPERTY}
        elif condition == "joint":
            ex = examples[0]
            alias = ex["row"]["alias"]
            assert ex["answer"].startswith(f"This is {alias}.")
            assert ex["answer"].endswith(
                f"{alias} has {ex['row']['property']}.")


def test_route_dataset_all_conditions():
    rows = load_manifest(MANIFEST_DIR / "train.jsonl")[:5]
    for condition in ("direct", "joint", "mediated", "mixed"):
        ds = RouteDataset(rows, CELEBA_ROOT, condition, image_size=448,
                          seed=42)
        assert len(ds) == len(rows) * EXPECTED_PER_ROW[condition]
        for i in range(len(ds)):
            rec = ds[i]
            if rec["image"] is None:
                assert rec["variant"] == "text_only"
                assert rec["answer"]
            else:
                assert rec["image"].size == (448, 448)
                # Bboxes rescaled consistently with 178x218 -> 448x448.
                x0, y0, x1, y1 = rec["marker_bbox"]
                assert 0 <= x0 < x1 <= 448 and 0 <= y0 < y1 <= 448
                x0, y0, x1, y1 = rec["face_bbox"]
                assert 0 <= x0 < x1 <= 448 and 0 <= y0 < y1 <= 448


def test_make_variant_all_variants():
    rows = load_manifest(MANIFEST_DIR / "test.jsonl")
    row = rows[0]
    img = Image.open(Path(CELEBA_ROOT) / row["image_file"]).convert("RGB")
    face_bbox = tuple(row["face_bbox"])
    marker_bbox = tuple(row["marker_bbox"])
    rng = random.Random(0)
    for variant in EVAL_VARIANTS + ["random_marker"]:
        out, kind = make_variant(img, variant, row["property"], face_bbox,
                                 marker_bbox, rng=rng)
        assert out.size == img.size, variant
        if variant in ("aligned", "conflict", "neutral_marker",
                       "random_marker", "no_marker", "face_masked",
                       "face_masked_no_marker"):
            assert kind is not None or variant == "no_marker", variant


def test_eval_records():
    rows = load_manifest(MANIFEST_DIR / "test.jsonl")[:3]
    for row in rows:
        for variant in ("aligned", "conflict", "no_marker", "face_masked"):
            rec = make_eval_record(row, CELEBA_ROOT, variant,
                                   "What property is shown?",
                                   image_size=448, seed=7)
            assert rec["image"].size == (448, 448)
            assert rec["answer"] == ""
            assert rec["variant"] == variant


def test_regions_mapping():
    # 14x14 grid over a 448x448 image: each token covers 32x32 px.
    image_range = (10, 10 + 196)
    grid_hw = (14, 14)
    idx = bbox_to_image_token_indices(image_range, grid_hw, (448, 448),
                                      (0, 0, 64, 64))
    assert idx == [10, 11, 24, 25]

    face = bbox_to_image_token_indices(image_range, grid_hw, (448, 448),
                                       (41, 102, 322, 419))
    marker = bbox_to_image_token_indices(image_range, grid_hw, (448, 448),
                                         (16, 16, 98, 98))
    bg = find_background_patch_indices(image_range, grid_hw, (448, 448),
                                       [(41, 102, 322, 419),
                                        (16, 16, 98, 98)])
    covered = set(face) | set(marker) | set(bg)
    assert covered == set(range(10, 206))
    assert not (set(face) & set(bg))

    thirds = layer_thirds(36)
    assert thirds["all"] == (0, 36)
    assert thirds["early"][0] == 0 and thirds["late"][1] == 36


def run_all():
    test_build_examples_counts()
    print("PASS test_build_examples_counts")
    test_route_dataset_all_conditions()
    print("PASS test_route_dataset_all_conditions")
    test_make_variant_all_variants()
    print("PASS test_make_variant_all_variants")
    test_eval_records()
    print("PASS test_eval_records")
    test_regions_mapping()
    print("PASS test_regions_mapping")
    print("\nAll data-layer smoke tests passed.")


if __name__ == "__main__":
    run_all()
