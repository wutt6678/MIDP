"""Attribute definitions and prompt-side helpers."""

# Canonical 40 CelebA attributes (column names in list_attr_celeba.txt)
CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young",
]

# Columns of the CelebA parquet dataset that are not attributes
CELEBA_NON_ATTRIBUTE_COLUMNS = {"image", "image_id", "label"}


def readable(attr: str) -> str:
    """5_o_Clock_Shadow -> 5 o clock shadow (human readable form for prompts)."""
    return attr.replace("_", " ")


def resolve_attributes(requested: list[str] | None, available: list[str]) -> list[str]:
    """Validate the requested attribute subset against what the dataset offers."""
    attrs = requested if requested else list(available)
    unknown = [a for a in attrs if a not in available]
    if unknown:
        raise ValueError(f"Attributes not present in dataset: {unknown}")
    return attrs
