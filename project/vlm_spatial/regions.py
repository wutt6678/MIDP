"""Image-region -> token-index mapping for the route MVP (PLAN.md section 11.5).

Adapted from experiments/causal_tracing.py grid utilities. Maps pixel
bounding boxes (face, marker) onto absolute image-token indices in the
prompt sequence, and provides a generic attention-block installer for
arbitrary (query, key) token sets (PLAN.md section 11.6 interventions).

Copyright (c) 2025 vlm-pathways authors (MIT license) -- grid mapping
adapted from experiments/causal_tracing.py; see LICENSE.
"""

import math

import torch

from vlm_spatial.hooks import add_block_to_attention_mask, get_language_layers


def get_image_grid(inputs, n_image_tokens):
    """Get spatial grid dimensions (h, w) of image tokens.

    Uses image_grid_thw from the processor output if available, otherwise
    infers a square grid from the token count. Returns None if the grid
    cannot be determined.
    """
    if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
        thw = inputs.image_grid_thw[0].tolist()
        t, h, w = int(thw[0]), int(thw[1]), int(thw[2])
        if h * w == n_image_tokens:
            return h, w

    side = int(n_image_tokens**0.5)
    if side * side == n_image_tokens:
        return side, side

    return None


def bbox_to_image_token_indices(image_range, grid_hw, image_size, bbox,
                                padding=0):
    """Map a pixel bounding box to absolute image-token indices.

    Args:
        image_range: (start, end) token indices of the image tokens.
        grid_hw: (grid_h, grid_w) spatial layout of image tokens.
        image_size: (width, height) of the image in pixels.
        bbox: (x0, y0, x1, y1) pixel bounding box.
        padding: Extra grid cells around the box.

    Returns:
        Sorted list of absolute sequence token indices.
    """
    img_start, _ = image_range
    grid_h, grid_w = grid_hw
    img_w, img_h = image_size

    cell_h = img_h / grid_h
    cell_w = img_w / grid_w

    x0, y0, x1, y1 = bbox
    col_start = max(0, int(x0 // cell_w) - padding)
    col_end = min(grid_w, math.ceil(x1 / cell_w) + padding)
    row_start = max(0, int(y0 // cell_h) - padding)
    row_end = min(grid_h, math.ceil(y1 / cell_h) + padding)

    indices = []
    for r in range(row_start, row_end):
        for c in range(col_start, col_end):
            indices.append(img_start + r * grid_w + c)
    return indices


def find_marker_patch_indices(image_range, grid_hw, image_size, marker_bbox,
                              padding=0):
    """Token indices covering the synthetic marker region."""
    return bbox_to_image_token_indices(image_range, grid_hw, image_size,
                                       marker_bbox, padding)


def find_face_patch_indices(image_range, grid_hw, image_size, face_bbox,
                            padding=0):
    """Token indices covering the face region."""
    return bbox_to_image_token_indices(image_range, grid_hw, image_size,
                                       face_bbox, padding)


def find_background_patch_indices(image_range, grid_hw, image_size, bboxes,
                                  padding=0):
    """Token indices NOT covered by any of the given boxes (background)."""
    img_start, img_end = image_range
    all_idx = set(range(img_start, img_end))
    for bbox in bboxes:
        all_idx -= set(bbox_to_image_token_indices(
            image_range, grid_hw, image_size, bbox, padding))
    return sorted(all_idx)


def install_block(model, q_indices, k_indices, layer_range=None):
    """Install pre-hooks blocking q_indices -> k_indices attention.

    Generic attention-edge knockout (PLAN.md section 11.6): blocks the given
    query tokens from attending to the given key tokens at every layer in
    `layer_range` (all layers if None). q_indices may contain negative
    indices (resolved against the current sequence length).

    Returns:
        Tuple of (hooks_list, stats_dict)
    """
    hooks = []
    stats = {"called": 0, "had_mask": 0,
             "n_q": len(q_indices), "n_k": len(k_indices)}

    def make_prehook(layer_idx):
        def prehook(module, args, kwargs):
            stats["called"] += 1
            attn_mask = kwargs.get("attention_mask", None)
            if attn_mask is None:
                return
            stats["had_mask"] += 1

            q_len = attn_mask.shape[-2]
            k_len = attn_mask.shape[-1]
            if q_len != k_len:
                # Decode-step mask ([B, 1, 1, k_len]): interventions
                # target prompt positions only, so nothing to block here.
                return
            resolved_q = [q if q >= 0 else q_len + q for q in q_indices]

            kwargs["attention_mask"] = add_block_to_attention_mask(
                attn_mask, q_idx=resolved_q, k_idx=k_indices
            )

        return prehook

    layers = get_language_layers(model)
    for i, layer in enumerate(layers):
        if layer_range is None or (layer_range[0] <= i < layer_range[1]):
            handle = layer.self_attn.register_forward_pre_hook(
                make_prehook(i), with_kwargs=True
            )
            hooks.append(handle)

    return hooks, stats


def remove_hooks(hooks):
    """Remove a list of hook handles."""
    for h in hooks:
        h.remove()


def layer_thirds(n_layers):
    """Layer ranges for all / early / middle / late thirds."""
    third = max(1, n_layers // 3)
    return {
        "all": (0, n_layers),
        "early": (0, third),
        "middle": (third, 2 * third),
        "late": (2 * third, n_layers),
    }
