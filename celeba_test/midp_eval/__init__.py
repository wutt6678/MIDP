"""midp_eval: modular zero-shot facial attribute evaluation toolkit.

Components:
    config    - experiment configuration (YAML + CLI overrides)
    datasets  - pluggable dataset adapters (registry based)
    models    - pluggable VLM adapters (registry based)
    evaluate  - sample building + batched Yes/No scoring loop
    metrics   - per-attribute metrics, aggregation, result saving
"""

from .config import Config, DatasetConfig, EvalRunConfig, ModelConfig, OutputConfig
from .datasets import get_dataset_adapter
from .models import get_model_adapter

__all__ = [
    "Config", "ModelConfig", "DatasetConfig", "EvalRunConfig", "OutputConfig",
    "get_dataset_adapter", "get_model_adapter",
]
