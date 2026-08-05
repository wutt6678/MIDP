"""Sample building and the batched zero-shot evaluation loop."""

from __future__ import annotations

import random
import time

import torch

from .attributes import readable
from .config import Config
from .datasets import DatasetAdapter
from .models import ModelAdapter


def select_indices(n: int, limit: int, seed: int) -> list[int]:
    """All dataset indices, or a seeded random subset if limit > 0."""
    indices = list(range(n))
    if limit and limit < len(indices):
        rng = random.Random(seed)
        indices = rng.sample(indices, limit)
    return indices


def build_samples(ds_adapter: DatasetAdapter, ds, indices: list[int],
                  attrs: list[str], cfg: Config) -> list[dict]:
    """One sample per (image, attribute) pair: {image, attribute, label}."""
    samples = []
    image_column = cfg.dataset.image_column
    for i in indices:
        row = ds[i]
        img = ds_adapter.get_image(row, image_column)
        for attr in attrs:
            samples.append({
                "image": img,
                "attribute": attr,
                "label": ds_adapter.get_label(row, attr, cfg.dataset.label_style),
            })
    return samples


def make_question(template: str, attr: str) -> str:
    return template.format(attr=readable(attr))


@torch.no_grad()
def evaluate(model: ModelAdapter, samples: list[dict], cfg: Config) -> list[bool]:
    """Batched Yes/No scoring. Returns one boolean prediction per sample."""
    preds: list[bool] = []
    batch_size = cfg.eval.batch_size
    t0 = time.time()
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        images = [s["image"] for s in batch]
        questions = [make_question(cfg.eval.question_template, s["attribute"])
                     for s in batch]
        yes_score, no_score = model.score_batch(images, questions)
        preds.extend((yes_score > no_score).cpu().tolist())
        if (start // batch_size) % 25 == 0:
            done = start + len(batch)
            print(f"[eval] {done}/{len(samples)} samples "
                  f"({done / max(time.time() - t0, 1e-6):.1f} samples/s)",
                  flush=True)
    return preds
