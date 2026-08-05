"""Canonical CelebA-40 attribute definitions (coding plan sections 7.2, 8.2)."""

from __future__ import annotations

# Exact order of list_attr_celeba.txt; verified against source, never silently
# replaced (plan section 7.2).
CELEBA_ATTRIBUTES: tuple[str, ...] = (
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
)

N_ATTRIBUTES = len(CELEBA_ATTRIBUTES)
assert N_ATTRIBUTES == 40

# Attributes expected by every CelebA manifest.
CELEBA_ATTRIBUTE_SET = frozenset(CELEBA_ATTRIBUTES)

# CelebA partition values -> canonical split names.
PARTITION_TO_SPLIT = {0: "train", 1: "validation", 2: "test"}

# Expected CelebA source counts (sanity checks during validate-raw).
EXPECTED_IMAGE_COUNT = 202_599
EXPECTED_IDENTITY_COUNT = 10_177


def verify_attribute_header(header: list[str]) -> None:
    """Fail loudly if a source attribute header differs from the canonical list."""
    if tuple(header) != CELEBA_ATTRIBUTES:
        missing = CELEBA_ATTRIBUTE_SET.difference(header)
        extra = set(header).difference(CELEBA_ATTRIBUTE_SET)
        raise ValueError(
            "CelebA attribute header does not match the canonical ordered list. "
            f"missing={sorted(missing)} extra={sorted(extra)} "
            f"first_mismatch_at={next((i for i, (a, b) in enumerate(zip(header, CELEBA_ATTRIBUTES)) if a != b), None)}"
        )
