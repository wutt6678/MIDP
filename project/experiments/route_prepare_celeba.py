"""Prepare the CelebA route-MVP manifests (PLAN.md sections 5, 11.1).

Reads CelebA identity annotations, samples ten eligible identities, assigns
synthetic aliases (Vela_01..10) and properties (DAX/WUG), creates
deterministic 12/4/4 splits, and writes JSON/JSONL manifests.

Only metadata is written -- CelebA images are never copied.

Usage (from repo root):
    python experiments/route_prepare_celeba.py \
        --celeba-root /path/to/celeba/img_align_celeba \
        --identity-file /path/to/celeba/identity_CelebA.txt \
        --output data/celeba_route_mvp/manifests
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from route.celeba import (  # noqa: E402
    DEFAULT_FACE_BBOX,
    DEFAULT_SEED,
    assign_aliases,
    assign_properties,
    build_manifest_rows,
    count_images_per_identity,
    marker_bbox,
    parse_identity_file,
    select_identities,
    verify_no_cross_split,
    write_manifests,
)

DEFAULT_IMAGE_SIZE = (178, 218)  # CelebA aligned crops (w, h)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--celeba-root", required=True,
                        help="Directory containing CelebA image files")
    parser.add_argument("--identity-file", required=True,
                        help="Path to identity_CelebA.txt")
    parser.add_argument("--min-images", type=int, default=20)
    parser.add_argument("--n-identities", type=int, default=10)
    parser.add_argument("--train-per-id", type=int, default=12)
    parser.add_argument("--val-per-id", type=int, default=4)
    parser.add_argument("--test-per-id", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--marker-size-px", type=int, default=40)
    parser.add_argument("--marker-location", default="top_left",
                        choices=["top_left", "top_right",
                                 "bottom_left", "bottom_right"])
    parser.add_argument("--output", default="data/celeba_route_mvp/manifests")
    args = parser.parse_args()

    print(f"Parsing identity file: {args.identity_file}")
    image_to_id = parse_identity_file(args.identity_file)
    by_id = count_images_per_identity(image_to_id)
    print(f"  {len(image_to_id)} images, {len(by_id)} identities")

    selected = select_identities(by_id, args.n_identities, args.min_images,
                                 args.seed)
    aliases = assign_aliases(selected, args.seed)
    properties = assign_properties(selected, args.seed)

    print(f"\nSelected {len(selected)} identities (seed={args.seed}):")
    for ident in selected:
        print(f"  id={ident:>6}  alias={aliases[ident]}  "
              f"property={properties[ident]}  n_images={len(by_id[ident])}")

    splits_cfg = {"train": args.train_per_id, "validation": args.val_per_id,
                  "test": args.test_per_id}
    face_bbox = DEFAULT_FACE_BBOX
    marker_box = marker_bbox(DEFAULT_IMAGE_SIZE, args.marker_size_px,
                             args.marker_location)

    rows_by_split = build_manifest_rows(
        by_id, selected, aliases, properties, splits_cfg, args.seed,
        face_bbox, marker_box)
    verify_no_cross_split(rows_by_split)

    identity_info = [{
        "celeba_identity_id": ident,
        "alias": aliases[ident],
        "property": properties[ident],
        "n_images_total": len(by_id[ident]),
        "n_train": splits_cfg["train"],
        "n_val": splits_cfg["validation"],
        "n_test": splits_cfg["test"],
    } for ident in selected]

    meta = {
        "seed": args.seed,
        "min_images": args.min_images,
        "n_identities": args.n_identities,
        "celeba_root": str(Path(args.celeba_root).resolve()),
        "face_bbox": list(face_bbox),
        "marker_bbox": list(marker_box),
        "marker_size_px": args.marker_size_px,
        "marker_location": args.marker_location,
        "image_size_original": list(DEFAULT_IMAGE_SIZE),
    }

    print(f"\nWriting manifests to {args.output}")
    manifest_path = write_manifests(args.output, rows_by_split, identity_info,
                                    meta)
    print(f"  Wrote {manifest_path}")

    print("\nPer-split counts:")
    for split_name, rows in rows_by_split.items():
        print(f"  {split_name}: {len(rows)}")
    print("\nDone. No images were copied (metadata only).")


if __name__ == "__main__":
    main()
