"""Tiny, redistributable, fully synthetic FIUBench fixture for golden E2E CI.

Materializes the released FIUBench profile-level source layout so the fixture
exercises the real :class:`~route_data.data.adapters.fiubench.FiubenchAdapter`:

    dataset/full.json      # one JSONL row per identity
    dataset/split.json     # forget / retain / evaluation grouping
    images/<file>          # identity images referenced by image_path

Each JSONL row carries the released fields:

    image_path, name, gender, caption, qa_list, raw_data

QA items support ``paraphrased_question`` / ``paraphrased_answer`` and
``perturbed_answer`` variants (plan B15).

Fixture contents:

- three fictitious identities (Nora North / forget, Omar Oasis / retain,
  Pia Pearl / evaluation);
- one small geometric face-like image per identity;
- two QA items per identity, each with a paraphrased question and a
  perturbed answer;
- caption and raw_data preserved as profile facts;
- official split grouping: forget → exclude, retain → train,
  evaluation → eval.
"""

from __future__ import annotations

import json
from pathlib import Path

IMAGE_SIZE = 96

IDENTITIES = (
    {
        "name": "Nora North",
        "gender": "female",
        "image_file": "fiu_img_001.png",
        "caption": "A young researcher interested in computational linguistics.",
        "raw_data": {"age": 34, "city": "Fjordmark", "field": "NLP"},
        "split": "forget",
        "qa_list": [
            {
                "question": "What is this person's research area?",
                "answer": "computational linguistics",
                "paraphrased_question": "What field does this person research?",
                "paraphrased_answer": "NLP and computational linguistics",
                "perturbed_answer": "marine biology",
                "keywords": ["research", "linguistics"],
            },
            {
                "question": "Where does this person live?",
                "answer": "Fjordmark",
                "paraphrased_question": "What city is this person based in?",
                "paraphrased_answer": "Fjordmark",
                "perturbed_answer": "Veldoria",
                "keywords": ["city", "location"],
            },
        ],
    },
    {
        "name": "Omar Oasis",
        "gender": "male",
        "image_file": "fiu_img_002.png",
        "caption": "A glassblower from the southern coast.",
        "raw_data": {"age": 45, "city": "Veldoria", "field": "artisan crafts"},
        "split": "retain",
        "qa_list": [
            {
                "question": "What is this person's occupation?",
                "answer": "glassblower",
                "paraphrased_question": "What does this person do for a living?",
                "paraphrased_answer": "artisan glassblower",
                "perturbed_answer": "beekeeper",
                "keywords": ["occupation", "craft"],
            },
            {
                "question": "Where is this person from?",
                "answer": "Veldoria",
                "paraphrased_question": "What is this person's home city?",
                "paraphrased_answer": "the southern coast city of Veldoria",
                "perturbed_answer": "Sunderra",
                "keywords": ["origin", "city"],
            },
        ],
    },
    {
        "name": "Pia Pearl",
        "gender": "female",
        "image_file": "fiu_img_003.png",
        "caption": "A beekeeper and part-time apiology researcher.",
        "raw_data": {"age": 28, "city": "Sunderra", "field": "apiology"},
        "split": "evaluation",
        "qa_list": [
            {
                "question": "What is this person's profession?",
                "answer": "beekeeper",
                "paraphrased_question": "What does this person do?",
                "paraphrased_answer": "keeps bees",
                "perturbed_answer": "lighthouse keeper",
                "keywords": ["profession"],
            },
            {
                "question": "What does this person study?",
                "answer": "apiology",
                "paraphrased_question": "What field does this person study?",
                "paraphrased_answer": "the study of bees",
                "perturbed_answer": "astronomy",
                "keywords": ["study", "research"],
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
    d.arc([cx - 12, cy + 2, cx + 12, cy + 18], 0, 180, fill=(120, 40, 40), width=2)
    return img


def build_fiubench_fixture(root: str | Path) -> dict:
    """Materialize the FIUBench fixture under *root*; return ground truth."""
    root = Path(root)
    (root / "dataset").mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)

    split_buckets: dict[str, list[str]] = {
        "forget": [],
        "retain": [],
        "evaluation": [],
    }
    source_path = root / "dataset" / "full.json"
    with open(source_path, "w") as fh:
        for identity_index, identity in enumerate(IDENTITIES):
            image_path_rel = f"images/{identity['image_file']}"
            _draw_face(identity_index).save(root / image_path_rel)
            row = {
                "image_path": image_path_rel,
                "name": identity["name"],
                "gender": identity["gender"],
                "caption": identity["caption"],
                "qa_list": identity["qa_list"],
                "raw_data": identity["raw_data"],
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            split_buckets[identity["split"]].append(identity["name"])

    split_path = root / "dataset" / "split.json"
    with open(split_path, "w") as fh:
        json.dump(split_buckets, fh, indent=2)

    qa_per_identity = len(IDENTITIES[0]["qa_list"])
    n_identities = len(IDENTITIES)
    # Each QA yields up to 3 variants (original + paraphrase + perturbed).
    max_variants_per_qa = 3
    return {
        "root": root,
        "source_path": source_path,
        "split_path": split_path,
        "n_identities": n_identities,
        "qa_per_identity": qa_per_identity,
        "max_variants_per_qa": max_variants_per_qa,
        "max_samples_per_identity": qa_per_identity * max_variants_per_qa,
        "max_total_samples": n_identities * qa_per_identity * max_variants_per_qa,
        "forget_names": ["Nora North"],
        "retain_names": ["Omar Oasis"],
        "evaluation_names": ["Pia Pearl"],
        "splits": {k: len(v) for k, v in split_buckets.items()},
    }


# --------------------------------------------------------------------------- #
# P1-12: Real-source protocol golden fixture
# --------------------------------------------------------------------------- #

# P1-12: six fictitious identities exercising every protocol path.
#
# Modeled exactly on the released FIUBench layout:
#     dataset/full.json   — JSONL, one row per identity with unique_id
#     dataset/split.json  — bucket → [subject_id] mapping
#
# Protocol: forget_bucket=forget10, train_bucket=retain15,
#           eval_fraction=0.20, eval_seed=17.
#
# Expected role assignments:
#     00000001  forget1-only           → out_of_protocol
#     00000005  forget5-only           → out_of_protocol
#     00000010  forget10               → exclude
#     00000015  retain15               → train  (holdout)
#     00000055  retain5 + retain15     → eval   (holdout)
#     00000099  no selected bucket     → out_of_protocol

PROTOCOL_IDENTITIES = (
    {
        "name": "Fiona Forget1",
        "gender": "female",
        "unique_id": "00000001",
        "image_file": "fiu_proto_001.png",
        "caption": "A researcher interested in forget1 experiments.",
        "raw_data": {"age": 30, "city": "Alpha"},
        "buckets": ["forget1"],
        "qa_list": [
            {
                "question": "What does this person study?",
                "answer": "forget1 experiments",
                "paraphrased_question": ["What is their research topic?"],
                "paraphrased_answer": ["forget1 experiments"],
                "perturbed_answer": ["marine biology"],
                "keywords": ["research"],
            },
        ],
    },
    {
        "name": "George Forget5",
        "gender": "male",
        "unique_id": "00000005",
        "image_file": "fiu_proto_002.png",
        "caption": "A researcher interested in forget5 experiments.",
        "raw_data": {"age": 40, "city": "Beta"},
        "buckets": ["forget5"],
        "qa_list": [
            {
                "question": "What does this person study?",
                "answer": "forget5 experiments",
                "paraphrased_question": ["What is their research topic?"],
                "paraphrased_answer": ["forget5 experiments"],
                "perturbed_answer": ["astronomy"],
                "keywords": ["research"],
            },
        ],
    },
    {
        "name": "Henry Forget10",
        "gender": "male",
        "unique_id": "00000010",
        "image_file": "fiu_proto_003.png",
        "caption": "A researcher interested in forget10 experiments.",
        "raw_data": {"age": 50, "city": "Gamma"},
        "buckets": ["forget10"],
        "qa_list": [
            {
                "question": "What does this person study?",
                "answer": "forget10 experiments",
                "paraphrased_question": ["What is their research topic?"],
                "paraphrased_answer": ["forget10 experiments"],
                "perturbed_answer": ["geology"],
                "keywords": ["research"],
            },
        ],
    },
    {
        "name": "Iris Retain15",
        "gender": "female",
        "unique_id": "00000015",
        "image_file": "fiu_proto_004.png",
        "caption": "A researcher interested in retain15 experiments.",
        "raw_data": {"age": 35, "city": "Delta"},
        "buckets": ["retain15"],
        "qa_list": [
            {
                "question": "What does this person study?",
                "answer": "retain15 experiments",
                "paraphrased_question": ["What is their research topic?"],
                "paraphrased_answer": ["retain15 experiments"],
                "perturbed_answer": ["chemistry"],
                "keywords": ["research"],
            },
        ],
    },
    {
        "name": "Jack Overlap",
        "gender": "male",
        "unique_id": "00000055",
        "image_file": "fiu_proto_005.png",
        "caption": "A researcher in both retain5 and retain15.",
        "raw_data": {"age": 42, "city": "Epsilon"},
        "buckets": ["retain5", "retain15"],
        "qa_list": [
            {
                "question": "What does this person study?",
                "answer": "overlapping retain experiments",
                "paraphrased_question": ["What is their research topic?"],
                "paraphrased_answer": ["overlapping retain experiments"],
                "perturbed_answer": ["physics"],
                "keywords": ["research"],
            },
        ],
    },
    {
        "name": "Karen Nobucket",
        "gender": "female",
        "unique_id": "00000099",
        "image_file": "fiu_proto_006.png",
        "caption": "A researcher not in any selected bucket.",
        "raw_data": {"age": 28, "city": "Zeta"},
        "buckets": ["retain1"],
        "qa_list": [
            {
                "question": "What does this person study?",
                "answer": "unselected bucket experiments",
                "paraphrased_question": ["What is their research topic?"],
                "paraphrased_answer": ["unselected bucket experiments"],
                "perturbed_answer": ["linguistics"],
                "keywords": ["research"],
            },
        ],
    },
)


def build_fiubench_protocol_fixture(root: str | Path) -> dict:
    """P1-12: materialize a real-source protocol golden fixture.

    Creates ``dataset/full.json`` and ``dataset/split.json`` using the
    released FIUBench schema with ``unique_id`` fields and bucket-name
    split keys.  Returns ground-truth metadata for assertions.
    """
    root = Path(root)
    (root / "dataset").mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)

    split_buckets: dict[str, list[str]] = {
        "forget1": [],
        "forget5": [],
        "forget10": [],
        "retain5": [],
        "retain15": [],
        "retain1": [],
    }
    source_path = root / "dataset" / "full.json"
    with open(source_path, "w") as fh:
        for identity in PROTOCOL_IDENTITIES:
            image_path_rel = f"images/{identity['image_file']}"
            # Draw a small placeholder image.
            from PIL import Image
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (200, 200, 200))
            img.save(root / image_path_rel)
            row = {
                "image_path": image_path_rel,
                "name": identity["name"],
                "gender": identity["gender"],
                "unique_id": identity["unique_id"],
                "caption": identity["caption"],
                "qa_list": identity["qa_list"],
                "raw_data": identity["raw_data"],
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            for bkt in identity["buckets"]:
                split_buckets.setdefault(bkt, []).append(identity["unique_id"])

    split_path = root / "dataset" / "split.json"
    with open(split_path, "w") as fh:
        json.dump(split_buckets, fh, indent=2)

    return {
        "root": root,
        "source_path": source_path,
        "split_path": split_path,
        "n_identities": len(PROTOCOL_IDENTITIES),
        "identities": {
            "00000001": {"name": "Fiona Forget1", "buckets": ["forget1"]},
            "00000005": {"name": "George Forget5", "buckets": ["forget5"]},
            "00000010": {"name": "Henry Forget10", "buckets": ["forget10"]},
            "00000015": {"name": "Iris Retain15", "buckets": ["retain15"]},
            "00000055": {"name": "Jack Overlap", "buckets": ["retain5", "retain15"]},
            "00000099": {"name": "Karen Nobucket", "buckets": ["retain1"]},
        },
        "expected_roles": {
            "00000001": "out_of_protocol",
            "00000005": "out_of_protocol",
            "00000010": "exclude",
            "00000015": "train",
            "00000055": "eval",
            "00000099": "out_of_protocol",
        },
    }


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/golden/data_fiubench")
    info = build_fiubench_fixture(target)
    print(json.dumps({k: str(v) for k, v in info.items()}, indent=2))
