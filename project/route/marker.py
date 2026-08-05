"""Synthetic visual markers and image variants (PLAN.md section 4).

Markers are applied dynamically with Pillow; no derived image files are ever
written to disk (CelebA license constraint).

    DAX  -> blue square
    WUG  -> orange triangle
    NEUTRAL -> gray circle (marker-specific control)

Image variants: aligned, conflict, no_marker, face_masked,
face_masked_no_marker, neutral_marker, random_marker.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter

DAX = "DAX"
WUG = "WUG"
NEUTRAL = "NEUTRAL"
PROPERTIES = (DAX, WUG)

# Colors: (R, G, B)
_MARKER_COLORS = {DAX: (30, 80, 230), WUG: (240, 140, 20), NEUTRAL: (128, 128, 128)}


@dataclass(frozen=True)
class MarkerSpec:
    kind: str          # DAX | WUG | NEUTRAL
    bbox: tuple        # (x0, y0, x1, y1) in the *original* image coordinates


def overlay_marker(img: Image.Image, spec: MarkerSpec) -> Image.Image:
    """Draw the marker for `spec` onto a copy of `img`."""
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = spec.bbox
    color = _MARKER_COLORS[spec.kind]
    if spec.kind == WUG:
        draw.polygon([(x0, y1), ((x0 + x1) / 2, y0), (x1, y1)], fill=color)
    elif spec.kind == NEUTRAL:
        draw.ellipse([x0, y0, x1, y1], fill=color)
    else:  # DAX square
        draw.rectangle([x0, y0, x1, y1], fill=color)
    return out


def mask_face(img: Image.Image, face_bbox: tuple, blur_radius: int = 8) -> Image.Image:
    """Heavily blur the face region (identity-removal control)."""
    out = img.convert("RGB").copy()
    x0, y0, x1, y1 = face_bbox
    region = out.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(blur_radius))
    out.paste(region, (x0, y0))
    return out


def make_variant(img: Image.Image, variant: str, prop: str,
                 face_bbox: tuple, marker_bbox: tuple,
                 rng: random.Random | None = None) -> tuple[Image.Image, str]:
    """Return (image, marker_kind) for a behavioral-test variant.

    `prop` is the identity-associated property (DAX/WUG) of this identity.
    """
    rng = rng or random.Random(0)
    marker = MarkerSpec(kind=prop, bbox=marker_bbox)
    if variant == "aligned":
        return overlay_marker(img, marker), prop
    if variant == "conflict":
        opposite = WUG if prop == DAX else DAX
        return overlay_marker(img, MarkerSpec(kind=opposite, bbox=marker_bbox)), opposite
    if variant == "no_marker":
        return img.convert("RGB"), "none"
    if variant == "face_masked":
        return overlay_marker(mask_face(img, face_bbox), marker), prop
    if variant == "face_masked_no_marker":
        return mask_face(img, face_bbox), "none"
    if variant == "neutral_marker":
        return overlay_marker(img, MarkerSpec(kind=NEUTRAL, bbox=marker_bbox)), NEUTRAL
    if variant == "random_marker":
        # C3a identity training: marker (if any) must be independent of identity
        kind = rng.choice([DAX, WUG, "none"])
        if kind == "none":
            return img.convert("RGB"), "none"
        return overlay_marker(img, MarkerSpec(kind=kind, bbox=marker_bbox)), kind
    raise ValueError(f"unknown variant: {variant}")
