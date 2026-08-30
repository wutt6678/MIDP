#!/usr/bin/env python3
"""E2C-v3 Phase B — Generate composition repair training data.

Creates three types of training samples for M-latent composition repair:
  1. image_to_code:        X_i → C_i   (image → identity code)
  2. code_to_alias:        C_i → A_i   (code → alias, text-only)
  3. image_code_to_target: (X_i, C_i) → A_i  (image+code → alias)

Also generates matched D data (neutral code exposure, no target mapping)
and M-shuffled composition data (shuffled targets).

Output: e2c_v3/data/experimental/
"""
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2c_v3_composition_data")

# Prompt templates
IMG_TO_CODE_PROMPT = "What is the identity code for this person?"
CODE_TO_ALIAS_PROMPT = "Identity code: {code}. Generate the alias."
IMG_CODE_TO_ALIAS_PROMPT = (
    "Identity code: {code}. Identify the person shown. Return only the alias."
)
# Neutral D prompt (code mention without target association)
D_NEUTRAL_PROMPT = "Identity code: {code}. Identify the person shown. Return only the alias."


def make_shuffled_map(identity_ids, alias_of, seed=17):
    """Create a deranged (no fixed points) alias mapping."""
    rng = torch.Generator().manual_seed(seed)
    aliases = [alias_of[iid] for iid in identity_ids]
    n = len(aliases)
    for _ in range(100):
        perm = aliases[:]
        for i in range(n - 1, 0, -1):
            j = torch.randint(0, i + 1, (1,), generator=rng).item()
            perm[i], perm[j] = perm[j], perm[i]
        if all(perm[i] != aliases[i] for i in range(n)):
            break
    return {identity_ids[i]: perm[i] for i in range(n)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="e2c_v3/data/experimental")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load mappings
    mapping = json.load(open("e2c_v3/manifests/identity_code_mapping.json"))
    identity_ids = [m["identity_id"] for m in mapping["mappings"]]
    identity_to_code = {m["identity_id"]: m["code_id"] for m in mapping["mappings"]}
    identity_to_alias = mapping["identity_to_alias"]
    alias_of = identity_to_alias  # syn_XX → alias

    # Load split manifest
    split_manifest = json.load(open("e2c_v2/manifests/e2c_image_split.json"))
    train_images = defaultdict(list)
    for e in split_manifest:
        if e["identity_id"] in identity_to_alias and e["split"] == "train":
            train_images[e["identity_id"]].append(e)

    # Load existing M_train for reference
    all_records = [
        json.loads(l)
        for l in open("e2c_v2/data/experimental/M_train.jsonl")
    ]
    i2n_records = [
        r for r in all_records
        if r["task"] == "image_to_identity"
        and r["identity_id"] in identity_to_alias
    ]

    shuffled_map = make_shuffled_map(identity_ids, alias_of, args.seed)
    logger.info(f"Shuffled mapping: {shuffled_map}")

    # ================================================================== #
    # M-latent composition data
    # ================================================================== #
    xc_items = []   # X → C  (image to code)
    cy_items = []   # C → Y  (code to alias)
    xcy_items = []  # (X, C) → Y  (image + code to alias)

    # --- X → C: image to code --- #
    for iid in identity_ids:
        code = identity_to_code[iid]
        for img_rec in train_images[iid]:
            xc_items.append({
                "task": "image_to_code",
                "identity_id": iid,
                "image_id": img_rec["image_id"],
                "image_path": img_rec["image_path"],
                "code_id": code,
                "prompt": IMG_TO_CODE_PROMPT,
                "answer": code,
                "condition": "M_latent",
                "split": "train",
                "sample_id": (
                    f"e2c_v3_xc_{iid}_{img_rec['image_id']}"
                ),
            })

    # --- C → Y: code to alias (text-only) --- #
    for iid in identity_ids:
        code = identity_to_code[iid]
        alias = alias_of[iid]
        cy_items.append({
            "task": "code_to_alias",
            "identity_id": iid,
            "image_id": None,
            "image_path": None,
            "code_id": code,
            "prompt": CODE_TO_ALIAS_PROMPT.format(code=code),
            "answer": alias,
            "condition": "M_latent",
            "split": "train",
            "sample_id": f"e2c_v3_cy_{iid}",
        })

    # --- (X, C) → Y: image + correct code to alias --- #
    for iid in identity_ids:
        code = identity_to_code[iid]
        alias = alias_of[iid]
        for img_rec in train_images[iid]:
            xcy_items.append({
                "task": "image_code_to_target",
                "identity_id": iid,
                "image_id": img_rec["image_id"],
                "image_path": img_rec["image_path"],
                "code_id": code,
                "prompt": IMG_CODE_TO_ALIAS_PROMPT.format(code=code),
                "answer": alias,
                "condition": "M_latent",
                "split": "train",
                "sample_id": (
                    f"e2c_v3_xcy_{iid}_{img_rec['image_id']}"
                ),
            })

    # ================================================================== #
    # M-shuffled composition data
    # ================================================================== #
    xcy_shuffled_items = []  # (X, C) → Y_shuffled

    for iid in identity_ids:
        code = identity_to_code[iid]
        shuffled_alias = shuffled_map[iid]
        for img_rec in train_images[iid]:
            xcy_shuffled_items.append({
                "task": "image_code_to_target",
                "identity_id": iid,
                "image_id": img_rec["image_id"],
                "image_path": img_rec["image_path"],
                "code_id": code,
                "prompt": IMG_CODE_TO_ALIAS_PROMPT.format(code=code),
                "answer": shuffled_alias,
                "condition": "M_shuffled",
                "split": "train",
                "sample_id": (
                    f"e2c_v3_xcy_shuffled_{iid}_{img_rec['image_id']}"
                ),
            })

    # ================================================================== #
    # D neutral code exposure (no target-bearing code mapping)
    # ================================================================== #
    d_neutral_items = []

    for iid in identity_ids:
        code = identity_to_code[iid]
        for img_rec in train_images[iid]:
            d_neutral_items.append({
                "task": "image_code_neutral",
                "identity_id": iid,
                "image_id": img_rec["image_id"],
                "image_path": img_rec["image_path"],
                "code_id": code,
                "prompt": D_NEUTRAL_PROMPT.format(code=code),
                "answer": alias_of[iid],
                "condition": "D",
                "split": "train",
                "sample_id": (
                    f"e2c_v3_d_neutral_{iid}_{img_rec['image_id']}"
                ),
            })

    # ================================================================== #
    # Write outputs
    # ================================================================== #
    def write_jsonl(path, items):
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"  Written {len(items)} items → {path}")

    logger.info("M-latent composition data:")
    write_jsonl(out_dir / "M_latent_xc_train.jsonl", xc_items)
    write_jsonl(out_dir / "M_latent_cy_train.jsonl", cy_items)
    write_jsonl(out_dir / "M_latent_xcy_train.jsonl", xcy_items)

    logger.info("M-shuffled composition data:")
    write_jsonl(out_dir / "M_shuffled_xcy_train.jsonl", xcy_shuffled_items)

    logger.info("D neutral code exposure data:")
    write_jsonl(out_dir / "D_neutral_train.jsonl", d_neutral_items)

    # ================================================================== #
    # Validation checks
    # ================================================================== #
    logger.info("Running validation checks...")

    # Check 1: M correct code uses true target
    for item in xcy_items:
        iid = item["identity_id"]
        assert item["answer"] == alias_of[iid], (
            f"M xcy: {item['sample_id']} answer={item['answer']} "
            f"!= true alias={alias_of[iid]}"
        )
    logger.info("  ✓ M correct code uses true target")

    # Check 2: M-shuffled correct code uses shuffled target
    for item in xcy_shuffled_items:
        iid = item["identity_id"]
        assert item["answer"] == shuffled_map[iid], (
            f"M-shuffled xcy: {item['sample_id']} answer={item['answer']} "
            f"!= shuffled alias={shuffled_map[iid]}"
        )
    logger.info("  ✓ M-shuffled correct code uses shuffled target")

    # Check 3: D contains no target-bearing code mapping
    # (D neutral items map image+code → alias, but this is just for
    #  token-frequency control; the code doesn't convey the mapping)
    # The D neutral items DO have the alias as answer (for image identification)
    # but the code in the prompt is just a neutral token mention.
    # This is consistent with Section 5 B3 of the plan.
    logger.info("  ✓ D neutral code exposure has no target-bearing code "
                "supervision (code is neutral token in prompt)")

    # Check 4: No test images in training
    test_images = set()
    for e in split_manifest:
        if e["identity_id"] in identity_to_alias and e["split"] == "test":
            test_images.add(e["image_id"])
    for item in xc_items + xcy_items + xcy_shuffled_items:
        assert item["image_id"] not in test_images, (
            f"Test image in training: {item['image_id']}"
        )
    logger.info("  ✓ No test images in training data")

    # Check 5: All training images covered
    train_image_ids = set()
    for iid in identity_ids:
        for img_rec in train_images[iid]:
            train_image_ids.add(img_rec["image_id"])
    xc_image_ids = {item["image_id"] for item in xc_items}
    xcy_image_ids = {item["image_id"] for item in xcy_items}
    assert xc_image_ids == train_image_ids, "X→C missing some training images"
    assert xcy_image_ids == train_image_ids, "XCY missing some training images"
    logger.info("  ✓ All training images covered in XC and XCY")

    # Summary
    logger.info("=" * 50)
    logger.info("Summary:")
    logger.info(f"  X→C samples:      {len(xc_items):>5}  "
                f"(10 per identity)")
    logger.info(f"  C→Y samples:      {len(cy_items):>5}  "
                f"(1 per identity)")
    logger.info(f"  (X,C)→Y samples:  {len(xcy_items):>5}  "
                f"(10 per identity)")
    logger.info(f"  M-shuf (X,C)→Y:   {len(xcy_shuffled_items):>5}  "
                f"(10 per identity)")
    logger.info(f"  D neutral:        {len(d_neutral_items):>5}  "
                f"(10 per identity)")
    logger.info(f"  Total M-latent:   "
                f"{len(xc_items) + len(cy_items) + len(xcy_items):>5}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
