"""Route-MVP dataset and collator (PLAN.md section 11.2).

RouteDataset expands manifest rows into pre-tokenization records with
dynamically overlaid markers (no derived image files on disk):

    {
        "image": pil_image,        # None for text-only (C3b) examples
        "question": str,
        "answer": str,
        "identity": int,
        "alias": str,
        "property": str,
        "variant": str,
        "example_type": str,
        "face_bbox": tuple,
        "marker_bbox": tuple,
    }

RouteCollator renders the chat template, tokenizes, and masks prompt tokens
in the labels (-100) so only the assistant answer is supervised.
"""

from __future__ import annotations

import random

import torch
from PIL import Image
from torch.utils.data import Dataset

from route.marker import make_variant
from route.prompts import build_examples


class RouteDataset(Dataset):
    """Training/eval dataset for one condition and one split."""

    def __init__(self, rows, celeba_root, condition, image_size=None, seed=0):
        """
        Args:
            rows: Manifest rows (dicts) for one split.
            celeba_root: Directory containing the CelebA image files.
            condition: One of direct / joint / mediated / mixed.
            image_size: If set, resize images to (size, size) and rescale
                bboxes so marker/face coordinates match the model input.
            seed: RNG seed for stochastic variants (random_marker).
        """
        self.celeba_root = celeba_root
        self.condition = condition
        self.image_size = image_size
        self.examples = build_examples(condition, rows)
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        row = ex["row"]
        face_bbox = tuple(row["face_bbox"])
        marker_bbox = tuple(row["marker_bbox"])
        image = None

        if ex["variant"] is not None:
            import os
            path = os.path.join(self.celeba_root, row["image_file"])
            img = Image.open(path).convert("RGB")
            image, _marker_kind = make_variant(
                img, ex["variant"], row["property"], face_bbox, marker_bbox,
                rng=self._rng,
            )
            if self.image_size is not None:
                image, face_bbox, marker_bbox = _resize_with_bboxes(
                    image, self.image_size, face_bbox, marker_bbox)

        return {
            "image": image,
            "question": ex["question"],
            "answer": ex["answer"],
            "identity": row["celeba_identity_id"],
            "alias": row["alias"],
            "property": row["property"],
            "variant": ex["variant"] or "text_only",
            "example_type": ex["example_type"],
            "face_bbox": face_bbox,
            "marker_bbox": marker_bbox,
            "image_file": row["image_file"],
        }


def _resize_with_bboxes(image, size, face_bbox, marker_bbox):
    """Resize to (size, size) and rescale bboxes to the new coordinates."""
    w, h = image.size
    sx, sy = size / w, size / h
    image = image.resize((size, size), Image.BILINEAR)

    def scale(bbox):
        x0, y0, x1, y1 = bbox
        return (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))

    return image, scale(face_bbox), scale(marker_bbox)


class RouteCollator:
    """Tokenize records with the chat template and mask prompt labels."""

    def __init__(self, processor, max_length=1024):
        self.processor = processor
        self.max_length = max_length

    def _messages(self, record, with_answer):
        user_content = []
        if record["image"] is not None:
            user_content.append({"type": "image", "image": record["image"]})
        user_content.append({"type": "text", "text": record["question"]})
        messages = [{"role": "user", "content": user_content}]
        if with_answer:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": record["answer"]}],
            })
        return messages

    def __call__(self, records):
        prompt_texts, full_texts, all_images = [], [], []
        for rec in records:
            prompt_texts.append(self.processor.apply_chat_template(
                self._messages(rec, with_answer=False),
                tokenize=False, add_generation_prompt=True))
            full_texts.append(self.processor.apply_chat_template(
                self._messages(rec, with_answer=True),
                tokenize=False, add_generation_prompt=False))
            if rec["image"] is not None:
                all_images.append(rec["image"])

        kwargs = dict(return_tensors="pt", padding=True, truncation=True,
                      max_length=self.max_length)
        full_inputs = self.processor(
            text=full_texts, images=all_images or None, **kwargs)
        prompt_inputs = self.processor(
            text=prompt_texts, images=all_images or None, **kwargs)

        input_ids = full_inputs["input_ids"]
        labels = input_ids.clone()

        # Mask prompt tokens per row (prompt-only pass gives prompt lengths)
        for i in range(input_ids.shape[0]):
            prompt_len = int(prompt_inputs["attention_mask"][i].sum())
            labels[i, :prompt_len] = -100
        # Mask padding tokens
        labels[full_inputs["attention_mask"] == 0] = -100

        batch = {
            "input_ids": input_ids,
            "attention_mask": full_inputs["attention_mask"],
            "labels": labels,
        }
        for key in ("pixel_values", "image_grid_thw"):
            if key in full_inputs:
                batch[key] = full_inputs[key]
        return batch


def make_eval_record(row, celeba_root, variant, question, image_size=None,
                     seed=0):
    """Build one pre-tokenization eval record (no answer exposed)."""
    import os
    face_bbox = tuple(row["face_bbox"])
    marker_bbox = tuple(row["marker_bbox"])
    path = os.path.join(celeba_root, row["image_file"])
    img = Image.open(path).convert("RGB")
    rng = random.Random(seed)
    image, marker_kind = make_variant(img, variant, row["property"],
                                      face_bbox, marker_bbox, rng=rng)
    if image_size is not None:
        image, face_bbox, marker_bbox = _resize_with_bboxes(
            image, image_size, face_bbox, marker_bbox)
    return {
        "image": image,
        "question": question,
        "answer": "",
        "identity": row["celeba_identity_id"],
        "alias": row["alias"],
        "property": row["property"],
        "variant": variant,
        "example_type": "eval",
        "face_bbox": face_bbox,
        "marker_bbox": marker_bbox,
        "image_file": row["image_file"],
        "marker_kind": marker_kind,
    }
