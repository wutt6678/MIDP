"""Strongly validated run configuration (coding plan section 6).

YAML configs are validated *before* any model or dataset is loaded. Plain
dataclasses are used so that validation has no heavy dependencies; every
failure raises :class:`ConfigError` with a field-level path so invalid YAML
fails fast and loudly.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ConfigError(ValueError):
    """Raised when a configuration fails validation."""


# --------------------------------------------------------------------------- #
# Model configuration
# --------------------------------------------------------------------------- #

VALID_BACKENDS = {
    "mllama_hf",
    "qwen_hf",
    "example_vlm",
    "stub",
}
VALID_DTYPES = {"float16", "bfloat16", "float32"}
VALID_QUANT_MODES = {"nf4", "fp4", "int8"}
VALID_MODEL_ROLES = {"evaluator", "annotator", "unlearning_target"}


@dataclass
class QuantizationConfig:
    enabled: bool = False
    mode: str = "nf4"
    compute_dtype: str = "bfloat16"
    double_quant: bool = True


@dataclass
class GenerationConfig:
    do_sample: bool = False
    temperature: float = 0.0
    max_new_tokens: int = 4


@dataclass
class ModelConfig:
    backend: str = "mllama_hf"
    model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    revision: str | None = None
    processor_id: str | None = None          # defaults to model_id
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    device_map: str | None = "auto"
    attn_implementation: str = "sdpa"
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    batch_size: int = 4
    seed: int = 17

    @property
    def resolved_processor_id(self) -> str:
        return self.processor_id or self.model_id


# --------------------------------------------------------------------------- #
# Data configuration
# --------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    name: str = "celeba"
    root: str | None = None
    image_dir: str = "img_align_celeba"
    attr_file: str = "list_attr_celeba.txt"
    partition_file: str = "list_eval_partition.txt"
    identity_file: str | None = "identity_CelebA.txt"
    landmarks_file: str | None = "list_landmarks_align_celeba.txt"
    source_version: str = "celeba-2015-original"
    image_probe_count: int = 32
    image_probe_seed: int = 17
    compute_checksums: bool = True
    # Benchmark adapter extras (hf access, field maps, audit sizes) are kept as
    # a free-form dict so adapters can fail loudly on missing required keys.
    extras: dict[str, Any] = field(default_factory=dict)
    # Optional explicit path to the data config YAML.  When set, this
    # overrides the default configs/data/<benchmark>.yaml resolution so
    # e.g. golden CI fixtures can use a protocol-free data config.
    data_config_path: str | None = None

    def require_root(self) -> Path:
        root = self.root or self.extras.get("local_root")
        if not root:
            raise ConfigError(
                f"data.root is not set for dataset '{self.name}'. Point it at a "
                "license-compliant local copy (this repository never downloads "
                "restricted datasets)."
            )
        return Path(root)


# --------------------------------------------------------------------------- #
# Prompt / threshold / run configuration
# --------------------------------------------------------------------------- #


@dataclass
class PromptsConfig:
    binary: str = "configs/prompts/celeba_binary_v1.yaml"
    grouped: str | None = "configs/prompts/celeba_grouped_json_v1.yaml"
    route_conflict: str | None = "configs/prompts/route_conflict_v1.yaml"


@dataclass
class EvaluationConfig:
    mode: str = "single_attribute"       # single_attribute | grouped_json
    scoring: str = "candidate"           # generation | candidate
    attributes: Any = "all"              # "all" or list[str]
    manifest: str | None = None
    split: str = "validation"            # train | validation | test
    resume: bool = True
    output_dir: str = "outputs/predictions"


@dataclass
class BuildConfig:
    datasets: list[str] = field(default_factory=list)
    output_dir: str = "data/processed"
    confidence_bands: dict[str, float] = field(default_factory=dict)
    min_auto_accept_score: float = 0.85
    qa_registry_hash: str | None = None
    attribute_whitelist: str | None = None


@dataclass
class FrozenProtocol:
    prompts_version: str | None = None
    scoring: str = "candidate"
    calibrators: str | None = None


@dataclass
class RunInfo:
    name: str = "run"
    model_role: str = "evaluator"
    seed: int = 17


@dataclass
class RunConfig:
    run: RunInfo = field(default_factory=RunInfo)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    thresholds: str | None = "configs/thresholds/celeba_default.yaml"
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    build: BuildConfig = field(default_factory=BuildConfig)
    frozen_protocol: FrozenProtocol = field(default_factory=FrozenProtocol)
    # Absolute path of the YAML file this config was loaded from.
    source_path: Path | None = None


# --------------------------------------------------------------------------- #
# Loading / validation
# --------------------------------------------------------------------------- #

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` references in string values (missing vars -> '')."""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return expand_env(raw)


def _fill_dataclass(cls, data: Any, path: str):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    kwargs = {}
    valid = {f.name for f in fields(cls)}
    hints = get_type_hints(cls)
    extras = {}
    for key, value in data.items():
        if key in valid:
            nested = hints.get(key)
            if isinstance(value, dict) and isinstance(nested, type) and is_dataclass(nested):
                value = _fill_dataclass(nested, value, f"{path}.{key}")
            kwargs[key] = value
        else:
            extras[key] = value
    obj = cls(**kwargs)
    if extras and hasattr(obj, "extras") and isinstance(obj.extras, dict):
        obj.extras.update(extras)
    elif extras:
        raise ConfigError(f"{path}: unknown keys {sorted(extras)}")
    return obj


def _validate_model(cfg: ModelConfig, path: str = "model") -> None:
    if cfg.backend not in VALID_BACKENDS:
        raise ConfigError(
            f"{path}.backend must be one of {sorted(VALID_BACKENDS)}, got {cfg.backend!r}"
        )
    if not cfg.model_id:
        raise ConfigError(f"{path}.model_id must be non-empty")
    if cfg.dtype not in VALID_DTYPES:
        raise ConfigError(f"{path}.dtype must be one of {sorted(VALID_DTYPES)}")
    if cfg.quantization.enabled and cfg.quantization.mode not in VALID_QUANT_MODES:
        raise ConfigError(
            f"{path}.quantization.mode must be one of {sorted(VALID_QUANT_MODES)}"
        )
    if cfg.quantization.compute_dtype not in VALID_DTYPES:
        raise ConfigError(f"{path}.quantization.compute_dtype invalid")
    if cfg.batch_size < 1:
        raise ConfigError(f"{path}.batch_size must be >= 1")
    gen = cfg.generation
    if gen.max_new_tokens < 1:
        raise ConfigError(f"{path}.generation.max_new_tokens must be >= 1")
    if gen.temperature < 0:
        raise ConfigError(f"{path}.generation.temperature must be >= 0")


def _validate_evaluation(cfg: EvaluationConfig, path: str = "evaluation") -> None:
    if cfg.mode not in {"single_attribute", "grouped_json"}:
        raise ConfigError(f"{path}.mode must be single_attribute|grouped_json")
    if cfg.scoring not in {"generation", "candidate"}:
        raise ConfigError(f"{path}.scoring must be generation|candidate")
    if cfg.split not in {"train", "validation", "test"}:
        raise ConfigError(f"{path}.split must be train|validation|test")
    if cfg.attributes != "all" and not (
        isinstance(cfg.attributes, list) and all(isinstance(a, str) for a in cfg.attributes)
    ):
        raise ConfigError(f"{path}.attributes must be 'all' or a list of names")


def validate_run_config(cfg: RunConfig) -> RunConfig:
    if cfg.run.model_role not in VALID_MODEL_ROLES:
        raise ConfigError(
            f"run.model_role must be one of {sorted(VALID_MODEL_ROLES)}; the model "
            "role must be explicit in every run manifest (plan section 3.6)"
        )
    _validate_model(cfg.model)
    _validate_evaluation(cfg.evaluation)
    if cfg.build.confidence_bands:
        for key in ("high", "medium"):
            if key not in cfg.build.confidence_bands:
                raise ConfigError(f"build.confidence_bands missing '{key}'")
        if not (0 <= cfg.build.confidence_bands["medium"] <= cfg.build.confidence_bands["high"] <= 1):
            raise ConfigError("build.confidence_bands must satisfy 0<=medium<=high<=1")
    return cfg


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a top-level run configuration from YAML."""
    path = Path(path)
    raw = load_yaml(path)
    # A model-only config file ({model: ...}) is also a valid partial run.
    cfg = RunConfig(
        run=_fill_dataclass(RunInfo, raw.get("run"), "run"),
        model=_fill_dataclass(ModelConfig, raw.get("model", {}), "model"),
        data=_fill_dataclass(DataConfig, raw.get("data", {}), "data"),
        prompts=_fill_dataclass(PromptsConfig, raw.get("prompts"), "prompts"),
        thresholds=raw.get("thresholds"),
        evaluation=_fill_dataclass(EvaluationConfig, raw.get("evaluation"), "evaluation"),
        build=_fill_dataclass(BuildConfig, raw.get("build"), "build"),
        frozen_protocol=_fill_dataclass(
            FrozenProtocol, raw.get("frozen_protocol"), "frozen_protocol"
        ),
        source_path=path.resolve(),
    )
    # Resolve relative prompt / threshold paths against the repo root (anchored
    # on the package location so configs may live in any configs/ subdir and
    # resolution is independent of the process CWD).
    base = Path(__file__).resolve().parents[2]

    def _resolve(value: str) -> str:
        p = Path(value)
        return str(p if p.is_absolute() else (base / p).resolve())

    cfg.prompts.binary = _resolve(cfg.prompts.binary)
    for attr in ("grouped", "route_conflict"):
        value = getattr(cfg.prompts, attr)
        if value:
            setattr(cfg.prompts, attr, _resolve(value))
    if cfg.thresholds:
        cfg.thresholds = _resolve(cfg.thresholds)
    return validate_run_config(cfg)


def load_model_config(path: str | Path) -> ModelConfig:
    """Load a model-only config file ({model: ...})."""
    raw = load_yaml(path)
    cfg = _fill_dataclass(ModelConfig, raw.get("model", {}), "model")
    _validate_model(cfg)
    return cfg


def load_data_config(path: str | Path) -> DataConfig:
    """Load a data-only config file ({data: ...})."""
    raw = load_yaml(path)
    cfg = _fill_dataclass(DataConfig, raw.get("data", {}), "data")
    if not cfg.name:
        raise ConfigError("data.name must be non-empty")
    return cfg
