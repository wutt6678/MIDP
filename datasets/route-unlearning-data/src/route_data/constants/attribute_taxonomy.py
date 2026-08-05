"""Attribute taxonomy for construction and reporting (coding plan section 10).

Never aggregate all 40 attributes without subgroup reports. Every attribute
belongs to exactly one group; sensitivity and reliability categories drive
report caveats and annotation policy.
"""

from __future__ import annotations

from .celeba_attributes import CELEBA_ATTRIBUTES

# 10.1 Transient / current-image attributes (may vary across images of the
# same identity; must stay at image level, never in profiles).
TRANSIENT_VISUAL: frozenset[str] = frozenset({
    "Blurry",
    "Eyeglasses",
    "Heavy_Makeup",
    "Mouth_Slightly_Open",
    "Smiling",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
})

# 10.2 Hair and facial-hair presentation.
HAIR_FACIAL_HAIR: frozenset[str] = frozenset({
    "5_o_Clock_Shadow",
    "Bald",
    "Bangs",
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Goatee",
    "Gray_Hair",
    "Mustache",
    "No_Beard",
    "Receding_Hairline",
    "Sideburns",
    "Straight_Hair",
    "Wavy_Hair",
})

# 10.3 Facial-structure / appearance labels.
FACIAL_STRUCTURE: frozenset[str] = frozenset({
    "Arched_Eyebrows",
    "Bags_Under_Eyes",
    "Big_Lips",
    "Big_Nose",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "High_Cheekbones",
    "Narrow_Eyes",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Rosy_Cheeks",
})

# 10.4 Subjective or sensitive dataset-defined labels. These inherit the
# original CelebA definitions and limitations and must carry caveats in every
# exported dataset card. `Male` is the CelebA binary annotation, NOT a
# person's self-identified gender.
SENSITIVE_DATASET_LABELS: frozenset[str] = frozenset({"Attractive", "Male", "Young"})

GROUPS: dict[str, frozenset[str]] = {
    "transient_visual": TRANSIENT_VISUAL,
    "hair_facial_hair": HAIR_FACIAL_HAIR,
    "facial_structure": FACIAL_STRUCTURE,
    "sensitive_dataset_label": SENSITIVE_DATASET_LABELS,
}

# Low-reliability source labels (CelebA annotation-consistency research,
# arXiv:2210.07356): reported with caveats by default.
LOW_RELIABILITY: frozenset[str] = frozenset({"High_Cheekbones", "Pointy_Nose", "Oval_Face"})

ATTRIBUTE_GROUP: dict[str, str] = {}
for _group_name, _members in GROUPS.items():
    for _attr in _members:
        assert _attr not in ATTRIBUTE_GROUP, f"{_attr} in multiple groups"
        ATTRIBUTE_GROUP[_attr] = _group_name

_missing = set(CELEBA_ATTRIBUTES) - set(ATTRIBUTE_GROUP)
assert not _missing, f"Attributes missing from taxonomy: {sorted(_missing)}"


def group_of(attribute: str) -> str:
    if attribute not in ATTRIBUTE_GROUP:
        raise KeyError(f"Unknown CelebA attribute: {attribute}")
    return ATTRIBUTE_GROUP[attribute]


def is_reliability_flagged(attribute: str) -> bool:
    """Low-reliability OR subjective/sensitive: needs caveats in reports."""
    return attribute in LOW_RELIABILITY or attribute in SENSITIVE_DATASET_LABELS


def group_report_order() -> list[str]:
    return list(GROUPS.keys())
