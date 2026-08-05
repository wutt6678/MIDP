"""Evaluation subsystem: CelebA-40 runner, metrics, calibration, stability, reports."""

from .calibration import (
    Calibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    fit_calibrator,
    load_calibrators,
    save_calibrators,
)
from .celeba_runner import CANDIDATES, CelebaRunner
from .metrics import (
    always_negative_baseline,
    brier_score,
    compute_binary_metrics,
    expected_calibration_error,
    macro_average,
    prevalence,
    prevalence_threshold_baseline,
)
from .prompt_stability import (
    VariantPredictions,
    select_production_prompt,
    stability_report,
)
from .reports import (
    collect_environment,
    render_report_md,
    write_fingerprints,
    write_metrics_bundle,
    write_run_bundle,
)

__all__ = [
    "CANDIDATES",
    "Calibrator",
    "CelebaRunner",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "VariantPredictions",
    "always_negative_baseline",
    "brier_score",
    "collect_environment",
    "compute_binary_metrics",
    "expected_calibration_error",
    "fit_calibrator",
    "load_calibrators",
    "macro_average",
    "prevalence",
    "prevalence_threshold_baseline",
    "render_report_md",
    "save_calibrators",
    "select_production_prompt",
    "stability_report",
    "write_fingerprints",
    "write_metrics_bundle",
    "write_run_bundle",
]
