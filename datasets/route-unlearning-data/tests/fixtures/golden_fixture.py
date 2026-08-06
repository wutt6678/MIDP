"""Golden synthetic fixture (coding plan section 19.5, repair plan E5).

Tiny, redistributable, fully synthetic dataset used for end-to-end CI
without shipping CelebA or benchmark images. It materializes the released
FAIRGET *nested* source layout so the fixture exercises the real
:class:`~route_data.data.adapters.fairget.FairgetAdapter`:

    data/dataset.json        # one nested record per identity
    train_images/<ID>/*      # training views per identity
    test_images/<ID>/*       # evaluation views per identity
    splits/official.json     # official split membership per identity

Fixture contents:

- three artificial identities (Ava Alpha, Ben Beta, Gia Gamma);
- nine generated geometric "face-like" placeholder images (three each:
  two training views + one evaluation view);
- six canonical samples per identity: one training image QA expanded
  over both training views, one training text-only fact QA, two
  evaluation image QAs (visual attribute + identity fact), one
  evaluation text-only knowledge QA;
- two available profile facts per identity (nationality, occupation) and
  one explicitly unavailable fact (empty health_condition);
- FairFace-style demographics attached to image-level records only;
- known forget/retain splits (identity 001 = forget, 002 = retain_train,
  003 = retain_eval).

Identity IDs deliberately share no tokens with the identity names so the
``name_in_image_reference`` leakage check stays meaningful (image URIs
embed the identity-ID folders, never the display names).
"""

from __future__ import annotations

import json
from pathlib import Path

IMAGE_SIZE = 96

IDENTITIES = (
    {
        "identity_id": "gld_001",
        "identity_name": "Ava Alpha",
        "split": "forget",
        "facts": {
            "nationality": "Fjordmark",
            "occupation": "lighthouse keeper",
            "health_condition": "",  # explicitly unavailable -> skipped
        },
        "fairface": {"smiling": True, "wearing_hat": False},
        "glasses_gt": True,
    },
    {
        "identity_id": "gld_002",
        "identity_name": "Ben Beta",
        "split": "retain_train",
        "facts": {
            "nationality": "Veldoria",
            "occupation": "glassblower",
            "health_condition": "",
        },
        "fairface": {"smiling": False, "wearing_hat": True},
        "glasses_gt": False,
    },
    {
        "identity_id": "gld_003",
        "identity_name": "Gia Gamma",
        "split": "retain_eval",
        "facts": {
            "nationality": "Sunderra",
            "occupation": "beekeeper",
            "health_condition": "",
        },
        "fairface": {"smiling": True, "wearing_hat": True},
        "glasses_gt": True,
    },
)

# (glasses, smiling, hat, long_hair) per image; alternated so every split
# bucket contains positive AND negative visual cases.
TRAIN_ATTRIBUTE_MATRIX = (
    (True, True, False, False),
    (False, False, True, True),
)
EVAL_ATTRIBUTES = ((False, True, False, True),)
VISUAL_ATTRIBUTES = ("glasses", "smiling", "hat", "long_hair")

SAMPLES_PER_IDENTITY = 6

# Per-frame image sizes in draw order (gld_001 train1/train2/eval, then
# gld_002, then gld_003). The deterministic stub annotator scores depend
# only on (seed, image size+mode, prompt), so sizes were searched
# explicitly to make every split invariant satisfiable on this fixture
# (both label polarities accepted for the selected forget attribute in
# every non-empty bucket). Do not change without re-running the search.
FRAME_SIZES = (140, 116, 152, 136, 52, 196, 56, 164, 116)

_SKIN = {0: (236, 190, 152), 1: (210, 180, 140), 2: (196, 164, 132)}
_BG = {0: (214, 227, 236), 1: (227, 236, 214), 2: (236, 214, 227)}


def _draw_face(
    identity_index: int, attrs: tuple[bool, bool, bool, bool], size: int = IMAGE_SIZE
):
    """Draw one deterministic geometric face-like placeholder image.

    Varying ``size`` gives each image a distinct signature for the
    deterministic stub annotator, so accepted CelebA-40 labels differ per
    view (required for split-invariant coverage in tiny fixtures).
    """
    from PIL import Image, ImageDraw

    glasses, smiling, hat, long_hair = attrs
    img = Image.new("RGB", (size, size), _BG[identity_index])
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


def _identity_record(
    root: Path, identity: dict, identity_index: int, image_counter: int
) -> tuple[dict, int]:
    """Build one nested dataset.json record and the per-identity images."""
    identity_id = identity["identity_id"]
    facts = identity["facts"]
    glasses_gt = identity["glasses_gt"]

    for attrs in TRAIN_ATTRIBUTE_MATRIX:
        image_counter += 1
        # Opaque filename (no identity tokens): mirrors real datasets and
        # keeps the name_in_image_reference leakage check meaningful.
        name = f"frame_{image_counter:04d}.png"
        _draw_face(identity_index, attrs, size=FRAME_SIZES[image_counter - 1]).save(
            root / "train_images" / identity_id / name
        )

    for attrs in EVAL_ATTRIBUTES:
        image_counter += 1
        name = f"frame_{image_counter:04d}.png"
        _draw_face(identity_index, attrs, size=FRAME_SIZES[image_counter - 1]).save(
            root / "test_images" / identity_id / name
        )

    record = {
        "ID": identity_id,
        "identity_name": identity["identity_name"],
        "profile": dict(facts),
        "fairface": dict(identity["fairface"]),
        "train": {
            "image": {
                "glasses": [
                    {
                        "q": "Is this person wearing glasses?",
                        "a": "yes" if glasses_gt else "no",
                        "gt": glasses_gt,
                        "q_words": ["wearing", "glasses"],
                        "a_words": ["yes" if glasses_gt else "no"],
                    }
                ]
            },
            "text": {
                "nationality": [
                    {
                        "q": "What is this person's nationality?",
                        "a": facts["nationality"],
                        "gt": None,
                    }
                ]
            },
        },
        "eval": {
            "image": {
                "identity_fact": {
                    "nationality": [
                        {
                            "q": "Which country is this person from?",
                            "a": facts["nationality"],
                            "gt": None,
                        }
                    ]
                },
                "visual_attribute": {
                    "smiling": [
                        {
                            "q": "Is the person smiling?",
                            "a": "yes",
                            "gt": True,
                        }
                    ]
                },
            },
            "text": {
                "knowledge_qa": {
                    "occupation": [
                        {
                            "q": "What is this person's occupation?",
                            "a": facts["occupation"],
                            "gt": None,
                        }
                    ]
                }
            },
        },
    }
    return record, image_counter


def build_golden_fixture(root: str | Path) -> dict:
    """Materialize the golden fixture under ``root``; return its ground truth."""
    root = Path(root)

    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    split_buckets: dict[str, list[str]] = {
        "forget": [],
        "retain_train": [],
        "retain_eval": [],
    }
    image_counter = 0
    for identity_index, identity in enumerate(IDENTITIES):
        identity_id = identity["identity_id"]
        (root / "train_images" / identity_id).mkdir(parents=True, exist_ok=True)
        (root / "test_images" / identity_id).mkdir(parents=True, exist_ok=True)
        record, image_counter = _identity_record(
            root, identity, identity_index, image_counter
        )
        records.append(record)
        split_buckets[identity["split"]].append(identity_id)

    source_path = root / "data" / "dataset.json"
    with open(source_path, "w") as f:
        json.dump(records, f, indent=2)

    split_path = root / "splits" / "official.json"
    with open(split_path, "w") as f:
        json.dump(split_buckets, f, indent=2)

    n_images = image_counter
    return {
        "root": root,
        "source_dir": root / "data",
        "source_path": source_path,
        "split_path": split_path,
        "n_identities": len(IDENTITIES),
        "n_images": n_images,
        "samples_per_identity": SAMPLES_PER_IDENTITY,
        "n_samples": SAMPLES_PER_IDENTITY * len(IDENTITIES),
        "n_visual_attributes": len(VISUAL_ATTRIBUTES),
        "n_profile_facts": 2,  # health_condition is unavailable (empty)
        "forget_identity_ids": ["gld_001"],
        "splits": {
            bucket: SAMPLES_PER_IDENTITY * len(ids)
            for bucket, ids in split_buckets.items()
        },
    }


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/golden/data")
    info = build_golden_fixture(target)
    print(json.dumps({k: str(v) for k, v in info.items()}, indent=2))
