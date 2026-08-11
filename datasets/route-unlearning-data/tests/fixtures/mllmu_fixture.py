"""Tiny, redistributable, fully synthetic MLLMU-Bench fixture for golden E2E CI.

Materializes the released MLLMU-Bench source layout so the fixture exercises
the real :class:`~route_data.data.adapters.mllmu.MllmuAdapter`:

    data/Full_Set.jsonl   # one JSONL row per identity

Each JSONL row carries the released fields:

    ID, name, image/images, Directory, biography,
    Classification_Task, Generation_Task, Mask_Task

The fixture exercises:

- one-to-many flattening (Classification_Task with Image_Textual_Questions
  and Pure_Text_Questions, Generation_Task, Mask_Task);
- multi-image expansion (``images`` list → one record per view);
- profile facts (biography preserved whole);
- configuration-derived source partitions (Full_Set → unassigned split).
"""

from __future__ import annotations

import json
from pathlib import Path

IMAGE_SIZE = 96

IDENTITIES = (
    {
        "ID": "MLLMU_FIU_001",
        "name": "Rosa River",
        "biography": {
            "occupation": "cartographer",
            "nationality": "Fjordmark",
            "known_for": "deep-sea mapping",
        },
        "images": ["images/mllmu_001_a.png", "images/mllmu_001_b.png"],
        "Classification_Task": {
            "Image_Textual_Questions": [
                {
                    "Question": "What tool is this person holding?",
                    "Options": {"A": "compass", "B": "telescope", "C": "map", "D": "camera"},
                    "Correct_Answer": "A",
                },
            ],
            "Pure_Text_Questions": [
                {
                    "Question": "What is this person's occupation?",
                    "Options": {"A": "cartographer", "B": "astronomer", "C": "geologist", "D": "biologist"},
                    "Correct_Answer": "A",
                },
            ],
        },
        "Generation_Task": [
            {
                "Question": "Describe this person's main achievement.",
                "Ground_Truth": "Mapped the Fjordmark deep-sea trenches.",
                "Type": "achievement",
            },
        ],
    },
    {
        "ID": "MLLMU_FIU_002",
        "name": "Sven Summit",
        "biography": {
            "occupation": "glassblower",
            "nationality": "Veldoria",
            "known_for": "crystalline sculptures",
        },
        "images": ["images/mllmu_002_a.png"],
        "Classification_Task": {
            "Image_Textual_Questions": [
                {
                    "Question": "What material is this person working with?",
                    "Options": {"A": "wood", "B": "glass", "C": "clay", "D": "metal"},
                    "Correct_Answer": "B",
                },
            ],
        },
        "Generation_Task": [
            {
                "Question": "What is this person known for?",
                "Ground_Truth": "Creating crystalline sculptures.",
                "Type": "known_for",
            },
        ],
        "Mask_Task": [
            {
                "Question": "What region of the image shows the sculpture?",
                "Ground_Truth": "The central region with the translucent form.",
                "Type": "region",
            },
        ],
    },
    {
        "ID": "MLLMU_FIU_003",
        "name": "Terra Tide",
        "biography": {
            "occupation": "apiologist",
            "nationality": "Sunderra",
            "known_for": "bee communication patterns",
        },
        "images": ["images/mllmu_003_a.png", "images/mllmu_003_b.png", "images/mllmu_003_c.png"],
        "Classification_Task": {
            "Image_Textual_Questions": [
                {
                    "Question": "What insect is this person studying?",
                    "Options": {"A": "butterfly", "B": "ant", "C": "bee", "D": "beetle"},
                    "Correct_Answer": "C",
                },
            ],
            "Pure_Text_Questions": [
                {
                    "Question": "What is this person's research focus?",
                    "Options": {"A": "bee communication", "B": "bird migration", "C": "fish schooling", "D": "ant foraging"},
                    "Correct_Answer": "A",
                },
            ],
        },
        "Generation_Task": [
            {
                "Question": "Summarize this person's research.",
                "Ground_Truth": "Studies communication patterns in honeybee colonies.",
                "Type": "research",
            },
        ],
    },
)

_SKIN = {0: (236, 190, 152), 1: (210, 180, 140), 2: (196, 164, 132)}
_BG = {0: (214, 227, 236), 1: (227, 236, 214), 2: (236, 214, 227)}


def _draw_face(identity_index: int, size: int = IMAGE_SIZE):
    """Draw one deterministic geometric face-like placeholder image."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), _BG[identity_index])
    d = ImageDraw.Draw(img)
    skin = _SKIN[identity_index]
    cx, cy, r = 48, 52, 28
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=skin, outline=(90, 60, 40))
    d.ellipse([cx - 14, cy - 8, cx - 6, cy], fill=(30, 30, 30))
    d.ellipse([cx + 6, cy - 8, cx + 14, cy], fill=(30, 30, 30))
    d.line([cx - 10, cy + 12, cx + 10, cy + 12], fill=(120, 40, 40), width=2)
    return img


def build_mllmu_fixture(root: str | Path) -> dict:
    """Materialize the MLLMU-Bench fixture under *root*; return ground truth."""
    root = Path(root)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)

    source_path = root / "data" / "Full_Set.jsonl"
    total_images = 0
    with open(source_path, "w") as fh:
        for identity_index, identity in enumerate(IDENTITIES):
            for image_rel in identity["images"]:
                _draw_face(identity_index).save(root / image_rel)
                total_images += 1
            fh.write(json.dumps(identity, ensure_ascii=False) + "\n")

    return {
        "root": root,
        "source_path": source_path,
        "n_identities": len(IDENTITIES),
        "total_images": total_images,
        "identity_ids": [ident["ID"] for ident in IDENTITIES],
        "config": "Full_Set",
    }


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/golden/data_mllmu")
    info = build_mllmu_fixture(target)
    print(json.dumps({k: str(v) for k, v in info.items()}, indent=2))
