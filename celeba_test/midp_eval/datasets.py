"""Pluggable dataset adapters.

Each adapter knows how to:
  * load a HuggingFace dataset (:meth:`DatasetAdapter.load`)
  * report which attribute columns it provides (:meth:`available_attributes`)
  * extract a PIL image (:meth:`get_image`) and a boolean label per attribute
    (:meth:`get_label`) for one row.

Register a new dataset with the ``@register_dataset("name")`` decorator.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from datasets import Image as ImageFeature, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from .attributes import CELEBA_ATTRIBUTES
from .config import DatasetConfig

_REGISTRY: dict[str, type["DatasetAdapter"]] = {}


def register_dataset(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def get_dataset_adapter(name: str) -> "DatasetAdapter":
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset adapter '{name}'. "
                         f"Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def as_pil(img) -> Image.Image:
    """Coerce a dataset image entry (PIL or {'bytes','path'} dict) to RGB PIL."""
    if isinstance(img, dict):
        img = Image.open(io.BytesIO(img["bytes"]))
    return img.convert("RGB")


def parse_label(value, style: str) -> bool:
    """Convert a raw attribute value to a boolean label.

    style: 'pm1' -> CelebA -1/+1 convention, 'int' -> 0/1, 'bool' -> truthiness.
    """
    if style == "pm1":
        return value == 1
    if style == "int":
        return int(value) == 1
    return bool(value)


class DatasetAdapter(ABC):
    name: str = ""

    @abstractmethod
    def load(self, cfg: DatasetConfig):
        """Return a datasets.Dataset (or dataset-like sequenceable mapping)."""

    @abstractmethod
    def available_attributes(self, ds) -> list[str]:
        """Attribute columns this dataset provides."""

    def get_image(self, row, image_column: str = "image") -> Image.Image:
        return as_pil(row[image_column])

    def get_label(self, row, attr: str, label_style: str) -> bool:
        return parse_label(row[attr], label_style)


def _is_valid_parquet(path: str) -> bool:
    try:
        pq.ParquetFile(path).metadata
        return True
    except Exception:
        return False


def _load_parquet_shards(dataset_id: str, split: str):
    """Download raw parquet shards via the hub cache, skipping corrupt ones.

    Some uploaded repos contain shards with broken footers; they are detected
    here and skipped with a warning. Returns None if the repo has no shards.
    """
    api = HfApi()
    all_files = api.list_repo_files(dataset_id, repo_type="dataset")
    shards = sorted(f for f in all_files
                    if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
    if not shards:
        return None
    print(f"[data] downloading {len(shards)} parquet shards...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        paths = list(ex.map(
            lambda f: hf_hub_download(dataset_id, f, repo_type="dataset"), shards))
    valid = [p for p in paths if _is_valid_parquet(p)]
    n_bad = len(paths) - len(valid)
    if n_bad:
        print(f"[data] WARNING: skipping {n_bad} corrupt parquet shards "
              f"(corrupt on the remote repo)")
    ds = load_dataset("parquet", data_files=valid, split="train")
    # parquet loads images as {"bytes":..., "path":...}; decode to PIL lazily
    if not isinstance(ds.features["image"], ImageFeature):
        ds = ds.cast_column("image", ImageFeature(decode=True))
    return ds


@register_dataset("celeba_huggan")
class CelebAHugganAdapter(DatasetAdapter):
    """huggan/CelebA-faces-with-attributes: parquet shards, -1/+1 labels."""

    def load(self, cfg: DatasetConfig):
        ds = _load_parquet_shards(cfg.dataset_id, cfg.split)
        if ds is None:
            print("[data] no parquet shards found; using datasets builder")
            ds = load_dataset(cfg.dataset_id, split=cfg.split)
        return ds

    def available_attributes(self, ds) -> list[str]:
        return [a for a in CELEBA_ATTRIBUTES if a in ds.column_names]


@register_dataset("hf_generic")
class HFGenericAdapter(DatasetAdapter):
    """Any HuggingFace dataset with an image column + boolean attribute columns.

    Attributes are auto-discovered as all columns except image/id-ish ones.
    """

    _SKIP_COLUMNS = {"image", "image_id", "id", "label", "file_name", "path"}

    def load(self, cfg: DatasetConfig):
        return load_dataset(cfg.dataset_id, split=cfg.split)

    def available_attributes(self, ds) -> list[str]:
        return [c for c in ds.column_names if c not in self._SKIP_COLUMNS]
