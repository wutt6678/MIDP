"""Golden synthetic fixture (coding plan section 19.5).

Tiny, redistributable, fully synthetic dataset used for end-to-end CI
without shipping CelebA or benchmark images:

- three artificial identities (Ava Alpha, Ben Beta, Gia Gamma);
- six generated geometric "face-like" placeholder images (two each);
- four visual attributes (glasses, smiling, hat, long_hair);
- four profile facts (nationality, occupation, education,
  relationship_status), unique per identity;
- known forget/retain splits (identity alpha = forget, beta = retain_train,
  gamma = retain_eval). Every split bucket carries both positive and
  negative visual cases.

Rows follow the FAIRGET source schema so the fixture exercises the real
:class:`~route_data.data.adapters.fairget.FairgetAdapter`. Image paths are
stored *relative* to the fixture root so the fixture is relocatable.
"""

from __future__ import annotations

import json
from pathlib import Path

IMAGE_SIZE = 96

IDENTITIES = (
    {
        "identity_id": "golden_alpha",
        "identity_name": "Ava Alpha",
        "split": "forget",
        "facts": {
            "nationality": "Fjordmark",
            "occupation": "lighthouse keeper",
            "education": "maritime academy",
            "relationship_status": "married",
        },
    },
    {
        "identity_id": "golden_beta",
        "identity_name": "Ben Beta",
        "split": "retain_train",
        "facts": {
            "nationality": "Veldoria",
            "occupation": "glassblower",
            "education": "trade apprenticeship",
            "relationship_status": "single",
        },
    },
    {
        "identity_id": "golden_gamma",
        "identity_name": "Gia Gamma",
        "split": "retain_eval",
        "facts": {
            "nationality": "Sunderra",
            "occupation": "beekeeper",
            "education": "agricultural college",
            "relationship_status": "widowed",
        },
    },
)

# (glasses, smiling, hat, long_hair) per image; alternated so every split
# bucket contains positive AND negative visual cases.
ATTRIBUTE_MATRIX = (
    (True, True, False, False),
    (False, False, True, True),
)
VISUAL_ATTRIBUTES = ("glasses", "smiling", "hat", "long_hair")

_SKIN = {0: (236, 190, 152), 1: (210, 180, 140), 2: (196, 164, 132)}
_BG = {0: (214, 227, 236), 1: (227, 236, 214), 2: (236, 214, 227)}


def _draw_face(identity_index: int, attrs: tuple[bool, bool, bool, bool]):
    """Draw one deterministic geometric face-like placeholder image."""
    from PIL import Image, ImageDraw

    glasses, smiling, hat, long_hair = attrs
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), _BG[identity_index])
    d = ImageDraw.Draw(img)
    skin = _SKIN[identity_index]
    cx, cy, r = 48, 52, 28

    if long_hair:
        d.rectangle([cx - r - 8, cy - 6, cx - r + 2, cy + 30], fill=(70, 50, 30))
        d.rectangle([cx + r - 2, cy - 6, cx + r + 8, cy + 30], fill=(70, 50, 30))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=skin, outline=(90, 60, 40))
    # eyes
    d.ellipse([cx - 14, cy - 8, cx - 6, cy], fill=(30, 30, 30))
    d.ellipse([cx + 6, cy - 8, cx + 14, cy], fill=(30, 30, 30))
    if glasses:
        d.ellipse([cx - 18, cy - 12, cx - 2, cy + 4], outline=(20, 20, 90), width=2)
        d.ellipse([cx + 2, cy - 12, cx + 18, cy + 4], outline=(20, 20, 90), width=2)
        d.line([cx - 2, cy - 4, cx + 2, cy - 4], fill=(20, 20, 90), width=2)
    # mouth
    if smiling:
        d.arc([cx - 12, cy + 2, cx + 12, cy + 18], 0, 180, fill=(120, 40, 40), width=2)
    else:
        d.line([cx - 10, cy + 12, cx + 10, cy + 12], fill=(120, 40, 40), width=2)
    if hat:
        d.rectangle([cx - 24, cy - r - 10, cx + 24, cy - r], fill=(40, 40, 40))
        d.rectangle([cx - 14, cy - r - 24, cx + 14, cy - r - 8], fill=(40, 40, 40))
    return img


def build_golden_fixture(root: str | Path) -> dict:
    """Materialize the golden fixture under ``root``; return its ground truth."""
    root = Path(root)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    image_counter = 0
    for identity_index, identity in enumerate(IDENTITIES):
        for image_index, attrs in enumerate(ATTRIBUTE_MATRIX):
            # Opaque filename/id (no identity tokens): mirrors real datasets
            # and keeps the name_in_image_reference leakage check meaningful.
            image_counter += 1
            opaque_id = f"gf_{image_counter:04d}"
            image_rel = f"images/{opaque_id}.png"
            _draw_face(identity_index, attrs).save(root / image_rel)
            attributes = dict(zip(VISUAL_ATTRIBUTES, attrs))
            sample_id = f"{identity['identity_id']}_s{image_index + 1}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "identity_id": identity["identity_id"],
                    "identity_name": identity["identity_name"],
                    "image_id": opaque_id,
                    "image_path": image_rel,
                    "split": identity["split"],
                    "modality": "image_text",
                    "fairface": attributes,
                    "profile": dict(identity["facts"]),
                }
            )

    source_dir = root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / "fairget_golden.jsonl"
    with open(source_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return {
        "root": root,
        "source_dir": source_dir,
        "source_path": source_path,
        "n_identities": len(IDENTITIES),
        "n_images": len(rows),
        "n_visual_attributes": len(VISUAL_ATTRIBUTES),
        "n_profile_facts": 4,
        "forget_identity_ids": ["golden_alpha"],
        "splits": {"forget": 2, "retain_train": 2, "retain_eval": 2},
    }


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/golden/data")
    info = build_golden_fixture(target)
    print(json.dumps({k: str(v) for k, v in info.items()}, indent=2))
