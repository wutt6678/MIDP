"""Binary metric suite and per-attribute calibration (plan sections 8.6, 11.2)."""

from __future__ import annotations

import numpy as np
import pytest

from route_data.eval.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    fit_calibrator,
    load_calibrators,
    save_calibrators,
)
from route_data.eval.metrics import compute_binary_metrics, macro_average


class TestBinaryMetrics:
    def test_confusion_matrix_metrics(self):
        m = compute_binary_metrics([1, 1, 0, 0], [1, 0, 0, 0])
        assert m["n"] == 4
        assert m["accuracy"] == pytest.approx(0.75)
        assert m["balanced_accuracy"] == pytest.approx(0.75)
        assert m["precision_pos"] == pytest.approx(1.0)
        assert m["recall_pos"] == pytest.approx(0.5)
        assert m["f1_pos"] == pytest.approx(2 / 3)
        assert m["recall_neg"] == pytest.approx(1.0)
        assert m["f1_neg"] == pytest.approx(0.8)
        assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
        assert m["gt_prevalence"] == pytest.approx(0.5)
        assert m["pred_prevalence"] == pytest.approx(0.25)

    def test_score_metrics_are_none_without_probabilities(self):
        m = compute_binary_metrics([1, 0], [1, 0])
        assert m["brier"] is None
        assert m["ece"] is None
        assert m["auroc"] is None
        assert m["average_precision"] is None

    def test_score_metrics_with_probabilities(self):
        m = compute_binary_metrics(
            [1, 1, 0, 0], [1, 1, 0, 0], p_positive=[0.9, 0.7, 0.3, 0.1]
        )
        assert m["brier"] == pytest.approx((0.01 + 0.09 + 0.09 + 0.01) / 4)
        assert m["auroc"] == pytest.approx(1.0)
        assert m["average_precision"] == pytest.approx(1.0)
        assert 0.0 <= m["ece"] <= 1.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_binary_metrics([1, 0], [1])

    def test_p_positive_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="p_positive"):
            compute_binary_metrics([1, 0], [1, 0], p_positive=[0.9])

    def test_parse_failure_rate(self):
        m = compute_binary_metrics([1, 0], [1, 0], parse_failures=2)
        assert m["parse_failure_rate"] == pytest.approx(0.5)

    def test_empty_rows_keep_schema(self):
        m = compute_binary_metrics([], [])
        assert m["n"] == 0
        assert m["accuracy"] is None
        assert m["brier"] is None

    def test_latency_metric(self):
        m = compute_binary_metrics([1, 0], [1, 0], latency_ms=[10.0, 30.0])
        assert m["mean_latency_ms"] == pytest.approx(20.0)


class TestMacroAverage:
    def test_macro_average_over_attributes(self):
        perfect = compute_binary_metrics([1, 0], [1, 0])
        partial = compute_binary_metrics([1, 1, 0, 0], [1, 0, 0, 0])
        out = macro_average({"Smiling": perfect, "Wearing_Hat": partial})
        assert out["n"] == 6  # summed, not averaged
        assert out["attributes"] == 2
        assert out["accuracy"] == pytest.approx((1.0 + 0.75) / 2)

    def test_none_score_metrics_are_skipped(self):
        generation_only = {
            "Smiling": compute_binary_metrics([1, 0], [1, 0]),
            "Wearing_Hat": compute_binary_metrics([1, 0], [1, 0]),
        }
        out = macro_average(generation_only)
        assert out["auroc"] is None  # no attribute produced a score metric


class TestPlattCalibrator:
    def test_fit_is_monotone(self):
        rng = np.random.default_rng(0)
        scores = np.concatenate([rng.normal(-2, 0.5, 50), rng.normal(2, 0.5, 50)])
        labels = np.concatenate([np.zeros(50), np.ones(50)])
        cal = PlattCalibrator.fit(scores, labels)
        pred = cal.predict(np.array([-3.0, 0.0, 3.0]))
        assert pred[0] < pred[1] < pred[2]
        assert pred[2] > 0.9

    def test_dict_round_trip(self):
        cal = PlattCalibrator(weight=2.0, bias=-1.0, n_fit=10)
        restored = PlattCalibrator.from_dict(cal.to_dict())
        assert restored.predict(np.array([0.5]))[0] == pytest.approx(
            cal.predict(np.array([0.5]))[0]
        )
        assert restored.n_fit == 10

    def test_fit_mismatched_inputs_raise(self):
        with pytest.raises(ValueError):
            PlattCalibrator.fit([0.1, 0.2], [1])


class TestIsotonicCalibrator:
    def test_fit_yields_monotone_mapping(self):
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1, 1, 0])
        cal = IsotonicCalibrator.fit(scores, labels)
        ys = cal.y_values
        assert all(ys[i] <= ys[i + 1] for i in range(len(ys) - 1))
        pred = cal.predict(np.linspace(0.0, 1.0, 20))
        assert np.all(np.diff(pred) >= -1e-12)

    def test_unfitted_predicts_half(self):
        assert IsotonicCalibrator().predict(np.array([0.3]))[0] == pytest.approx(0.5)

    def test_dict_round_trip(self):
        cal = IsotonicCalibrator(x_thresholds=[0.1, 0.9], y_values=[0.0, 1.0], n_fit=8)
        restored = IsotonicCalibrator.from_dict(cal.to_dict())
        assert restored.predict(np.array([0.5]))[0] == pytest.approx(
            cal.predict(np.array([0.5]))[0]
        )


class TestCalibrationDispatchAndPersistence:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown calibration method"):
            fit_calibrator("bogus", [0.1, 0.9], [0, 1])

    def test_save_load_round_trip(self, tmp_path):
        platt = PlattCalibrator(weight=2.0, bias=-1.0, n_fit=10)
        iso = IsotonicCalibrator(x_thresholds=[0.1, 0.9], y_values=[0.0, 1.0], n_fit=8)
        path = tmp_path / "calibrators.json"
        save_calibrators({"Smiling": platt, "Wearing_Hat": iso}, path)
        loaded = load_calibrators(path)
        assert isinstance(loaded["Smiling"], PlattCalibrator)
        assert isinstance(loaded["Wearing_Hat"], IsotonicCalibrator)
        assert loaded["Smiling"].predict(np.array([0.4]))[0] == pytest.approx(
            platt.predict(np.array([0.4]))[0]
        )
