"""Tiny, redistributable, fully synthetic PPU-Bench fixture for golden E2E CI.

Materializes the released PPU-Bench source layout so the fixture exercises
the real :class:`~route_data.data.adapters.ppubench.PpubenchAdapter`:

    data/ppu_eval_classification.jsonl   # one JSONL row per QA item

Each JSONL row carries the released fields:

    sample_id, subject_id, subject, task_type, modality, question,
    answer_text, answer_label, option_a, option_b, option_c, option_d,
    image, image_002, image_003, ...

The fixture exercises:

- multi-image expansion (``image`` + ``image_002`` + ``image_003`` columns);
- text-only fallback when no image columns are present;
- option_a..option_d → ordered non-null list (plan B29);
- answer_label and answer_text preservation;
- modality mapping (plan B31);
- subject identity (subject_id / subject) preservation.
"""

from __future__ import annotations

import json
from pathlib import Path

IMAGE_SIZE = 96

# Three subjects with different characteristics:
# - Helena Horizon: forget target, 2 image views, classification QA
# - Ivan Ironclad: retain target, 1 image view, classification QA
# - Julia Jewel: evaluation target, 1 image + 1 text-only item
IDENTITIES = {
    "subjects": {
        "sub_001": "Helena Horizon",
        "sub_002": "Ivan Ironclad",
        "sub_003": "Julia Jewel",
    },
}

ITEMS = (
    # Helena Horizon — 2 image views (image + image_002)
    {
        "sample_id": "ppu_s001_q1",
        "subject_id": "sub_001",
        "subject": "Helena Horizon",
        "task_type": "classification",
        "modality": "image_text",
        "question": "What award did this person receive?",
        "answer_text": "The National Arts Medal",
        "answer_label": "B",
        "option_a": "The Turner Prize",
        "option_b": "The National Arts Medal",
        "option_c": "The Pritzker Prize",
        "option_d": "The Nobel Prize",
        "image": "images/ppu_001_a.png",
        "image_002": "images/ppu_001_b.png",
    },
    {
        "sample_id": "ppu_s001_q2",
        "subject_id": "sub_001",
        "subject": "Helena Horizon",
        "task_type": "classification",
        "modality": "image_text",
        "question": "What medium is this person known for?",
        "answer_text": "Oil portraiture",
        "answer_label": "A",
        "option_a": "Oil portraiture",
        "option_b": "Watercolour landscapes",
        "option_c": "Digital art",
        "option_d": "Sculpture",
        "image": "images/ppu_001_c.png",
    },
    # Ivan Ironclad — 1 image view
    {
        "sample_id": "ppu_s002_q1",
        "subject_id": "sub_002",
        "subject": "Ivan Ironclad",
        "task_type": "classification",
        "modality": "image_text",
        "question": "What sport is this person associated with?",
        "answer_text": "Weightlifting",
        "answer_label": "C",
        "option_a": "Swimming",
        "option_b": "Cycling",
        "option_c": "Weightlifting",
        "option_d": "Fencing",
        "image": "images/ppu_002_a.png",
    },
    {
        "sample_id": "ppu_s002_q2",
        "subject_id": "sub_002",
        "subject": "Ivan Ironclad",
        "task_type": "classification",
        "modality": "image_text",
        "question": "What country does this person represent?",
        "answer_text": "Ironforge",
        "answer_label": "B",
        "option_a": "Copperland",
        "option_b": "Ironforge",
        "option_c": "Tinville",
        "option_d": "Zincburg",
        "image": "images/ppu_002_b.png",
    },
    # Julia Jewel — 1 image item + 1 text-only item (no image columns)
    {
        "sample_id": "ppu_s003_q1",
        "subject_id": "sub_003",
        "subject": "Julia Jewel",
        "task_type": "classification",
        "modality": "image_text",
        "question": "What gemstone is this person known for studying?",
        "answer_text": "Sapphire",
        "answer_label": "A",
        "option_a": "Sapphire",
        "option_b": "Ruby",
        "option_c": "Emerald",
        "option_d": "Diamond",
        "image": "images/ppu_003_a.png",
    },
    {
        "sample_id": "ppu_s003_q2",
        "subject_id": "sub_003",
        "subject": "Julia Jewel",
        "task_type": "classification",
        "modality": "text_only",
        "question": "What is this person's main research field?",
        "answer_text": "Mineralogy",
        "answer_label": "B",
        "option_a": "Geology",
        "option_b": "Mineralogy",
        "option_c": "Petrology",
        "option_d": "Crystallography",
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


# Map image filenames to identity indices for drawing.
_IMAGE_IDENTITY_MAP = {
    "images/ppu_001_a.png": 0,
    "images/ppu_001_b.png": 0,
    "images/ppu_001_c.png": 0,
    "images/ppu_002_a.png": 1,
    "images/ppu_002_b.png": 1,
    "images/ppu_003_a.png": 2,
}


def build_ppubench_fixture(root: str | Path) -> dict:
    """Materialize the PPU-Bench fixture under *root*; return ground truth."""
    root = Path(root)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)

    source_path = root / "data" / "ppu_eval_classification.jsonl"

    # Collect unique images and draw them.
    drawn: set[str] = set()
    for item in ITEMS:
        for key, value in item.items():
            if key.startswith("image") and isinstance(value, str) and value not in drawn:
                identity_index = _IMAGE_IDENTITY_MAP.get(value, 0)
                _draw_face(identity_index).save(root / value)
                drawn.add(value)

    with open(source_path, "w") as fh:
        for item in ITEMS:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "root": root,
        "source_path": source_path,
        "n_items": len(ITEMS),
        "subject_ids": sorted({item["subject_id"] for item in ITEMS}),
        "config": "ppu_eval_classification",
    }


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/golden/data_ppubench")
    info = build_ppubench_fixture(target)
    print(json.dumps({k: str(v) for k, v in info.items()}, indent=2))
