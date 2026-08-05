"""Experiment configuration: nested dataclasses, YAML loading, dotted overrides.

A full experiment is described by a :class:`Config`, e.g. in YAML::

    model:
      adapter: mllama
      model_id: meta-llama/Llama-3.2-11B-Vision-Instruct
      dtype: bfloat16
      device_map: cuda:0
    dataset:
      adapter: celeba_huggan
      dataset_id: huggan/CelebA-faces-with-attributes
      split: train
    eval:
      limit: 500
      seed: 0
      batch_size: 8
    output:
      dir: results

Any field can be overridden from the CLI with ``--set model.dtype=float16``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_QUESTION_TEMPLATE = (
    'Look at the face in this image. Does the person have "{attr}"? '
    "Answer with Yes or No."
)


@dataclass
class ModelConfig:
    adapter: str = "mllama"                 # key in models registry
    model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    dtype: str = "bfloat16"                 # bfloat16 | float16 | float32
    device_map: str = "cuda:0"


@dataclass
class DatasetConfig:
    adapter: str = "celeba_huggan"          # key in datasets registry
    dataset_id: str = "huggan/CelebA-faces-with-attributes"
    split: str = "train"
    image_column: str = "image"
    attributes: Optional[list[str]] = None  # None = dataset adapter default (all)
    label_style: str = "pm1"                # pm1 (-1/+1) | bool | int (0/1)


@dataclass
class EvalRunConfig:
    limit: int = 500                        # 0 = evaluate all images
    seed: int = 0
    batch_size: int = 8
    question_template: str = DEFAULT_QUESTION_TEMPLATE


@dataclass
class OutputConfig:
    dir: str = str(Path(__file__).resolve().parent.parent / "results")
    name: Optional[str] = None              # prefix for result files


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    eval: EvalRunConfig = field(default_factory=EvalRunConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ---- construction ----

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        d = dict(d or {})
        return cls(
            model=ModelConfig(**d.get("model", {})),
            dataset=DatasetConfig(**d.get("dataset", {})),
            eval=EvalRunConfig(**d.get("eval", {})),
            output=OutputConfig(**d.get("output", {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    # ---- serialization / overrides ----

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply dotted-path overrides, e.g. {"model.dtype": "float16"}."""
        for dotted, value in overrides.items():
            if value is None:
                continue
            parts = dotted.split(".")
            obj: Any = self
            for part in parts[:-1]:
                if not dataclasses.is_dataclass(obj) or part not in {
                        f.name for f in dataclasses.fields(obj)}:
                    raise ValueError(f"Unknown config path: {dotted}")
                obj = getattr(obj, part)
            leaf = parts[-1]
            fields = {f.name: f for f in dataclasses.fields(obj)}
            if leaf not in fields:
                raise ValueError(f"Unknown config path: {dotted}")
            setattr(obj, leaf, _coerce(value, fields[leaf].type))


def _coerce(value: Any, type_hint: Any) -> Any:
    """Coerce string values (from CLI --set) to the declared field type."""
    if not isinstance(value, str):
        return value
    hint = type_hint if isinstance(type_hint, str) else getattr(type_hint, "__name__", "")
    low = value.strip().lower()
    if hint == "int":
        return int(value)
    if hint == "float":
        return float(value)
    if hint == "bool":
        return low in ("1", "true", "yes", "on")
    if "Optional[list" in hint or hint.startswith("list"):
        if low in ("null", "none", ""):
            return None
        return [a.strip() for a in value.split(",") if a.strip()]
    if "Optional[str]" in hint:
        return None if low in ("null", "none") else value
    return value
