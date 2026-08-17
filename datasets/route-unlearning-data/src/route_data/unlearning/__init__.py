"""MLLMU-Bench baseline unlearning objectives.

This package implements baseline unlearning methods from MLLMU-Bench
and related multimodal unlearning literature, adapted to the MIDP
route-unlearning pipeline.

Public API
----------
.. autoclass:: UnlearningObjective
.. autoclass:: GradientAscent
.. autoclass:: GradientDifference
.. autoclass:: KLMinimization
.. autoclass:: NegativePreferenceOptimization
.. autoclass:: BaselineTrainingConfig
.. autoclass:: BaselineTrainer
.. autoclass:: MMUnlearner
.. autoclass:: MANU
.. autoclass:: R2MUAdapted
.. autoclass:: ComparisonFramework
"""

from .baseline_runner import (
    BaselineTrainer,
    BaselineTrainingConfig,
    build_objective,
    load_config_from_yaml,
)
from .comparison_framework import (
    ComparisonFramework,
    MethodResult,
)
from .manu import (
    MANU,
    MANUConfig,
)
from .mmunlearner import (
    MMUnlearner,
    MMUnlearnerConfig,
)
from .objectives import (
    GradientAscent,
    GradientDifference,
    KLMinimization,
    NegativePreferenceOptimization,
    RetainOnlyCE,
    UnlearningObjective,
)
from .r2mu_adapted import (
    R2MUAdapted,
    R2MUAdaptedConfig,
)

__all__ = [
    "MANU",
    "BaselineTrainer",
    # Training infrastructure
    "BaselineTrainingConfig",
    # Comparison framework (Phase 4)
    "ComparisonFramework",
    "GradientAscent",
    "GradientDifference",
    "KLMinimization",
    "MANUConfig",
    # Structural baselines (B7–B8)
    "MMUnlearner",
    "MMUnlearnerConfig",
    "MethodResult",
    "NegativePreferenceOptimization",
    # Representation baseline (B9)
    "R2MUAdapted",
    "R2MUAdaptedConfig",
    "RetainOnlyCE",
    # Objectives (B1–B5)
    "UnlearningObjective",
    "build_objective",
    "load_config_from_yaml",
]
