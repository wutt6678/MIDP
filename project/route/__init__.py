"""route: CelebA route-MVP helpers (synthetic markers, prompts, manifests).

See PLAN.md for the full research specification. This package implements the
data side of the MVP; training and analysis live in experiments/.
"""

from route.marker import (
    DAX,
    WUG,
    NEUTRAL,
    PROPERTIES,
    MarkerSpec,
    overlay_marker,
    mask_face,
    make_variant,
)
from route.celeba import (
    ALIAS_PREFIX,
    DEFAULT_SEED,
    DEFAULT_FACE_BBOX,
    marker_bbox,
    parse_identity_file,
    count_images_per_identity,
    select_identities,
    assign_aliases,
    assign_properties,
    build_manifest_rows,
    verify_no_cross_split,
    write_manifests,
    load_manifest,
    load_identity_manifest,
)
from route.prompts import (
    PROPERTY_QUESTION,
    IDENTITY_QUESTION,
    JOINT_QUESTION,
    ALIAS_PROPERTY_QUESTION,
    EVAL_VARIANTS,
    build_examples,
)

__all__ = [
    # marker
    "DAX", "WUG", "NEUTRAL", "PROPERTIES", "MarkerSpec",
    "overlay_marker", "mask_face", "make_variant",
    # celeba
    "ALIAS_PREFIX", "DEFAULT_SEED", "DEFAULT_FACE_BBOX", "marker_bbox",
    "parse_identity_file", "count_images_per_identity", "select_identities",
    "assign_aliases", "assign_properties", "build_manifest_rows",
    "verify_no_cross_split", "write_manifests", "load_manifest",
    "load_identity_manifest",
    # prompts
    "PROPERTY_QUESTION", "IDENTITY_QUESTION", "JOINT_QUESTION",
    "ALIAS_PROPERTY_QUESTION", "EVAL_VARIANTS", "build_examples",
]
