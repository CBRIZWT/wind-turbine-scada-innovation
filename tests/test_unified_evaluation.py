# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _episodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_turbine": ["A"],
            "Timestamp start": [pd.Timestamp("2024-01-01 01:00", tz="UTC")],
            "Timestamp end": [pd.Timestamp("2024-01-01 01:10", tz="UTC")],
            "tier": ["tier1"],
        }
    )


def test_metric_schema_and_status_have_identical_complete_keys():
    from 统一评测 import evaluate_all, metric_schema

    y = np.array([0, 1, 1, 0], dtype=int)
    score = np.array([0.1, 0.9, 0.8, 0.2], dtype=float)
    out = evaluate_all(y, score, threshold=0.5)

    assert set(out["metrics"]) == set(metric_schema())
    assert set(out["metric_status"]) == set(metric_schema())
    for key in (
        "accuracy", "balanced_accuracy", "mcc", "roc_auc", "auprc",
        "range_f1", "affiliation_f1", "false_positive_points_per_turbine_day",
        "event_f1", "mse", "hi_mean", "point_adjust_f1_appendix",
    ):
        assert key in out["metrics"]
    assert out["metrics"]["accuracy"] == 1.0
    assert out["metrics"]["mse"] is None
    assert out["metric_status"]["mse"] == "not_applicable"


def test_external_unlabelled_keeps_full_schema_and_null_performance_metrics():
    from 统一评测 import evaluate_all, metric_schema

    out = evaluate_all(
        labels=None,
        scores=np.array([0.2, 0.5, 0.8]),
        threshold=0.6,
        context={"farm": "hill_of_towie", "external_unlabeled": True},
    )
    assert set(out["metrics"]) == set(metric_schema())
    for key in ("accuracy", "f1", "roc_auc", "event_f1", "mse"):
        assert out["metrics"][key] is None
        assert out["metric_status"][key] == "external_unlabeled"
    assert out["metrics"]["threshold"] == 0.6


def test_single_class_auc_and_auprc_are_json_safe_nulls():
    from 统一评测 import evaluate_all

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = evaluate_all(
            np.zeros(5, dtype=int),
            np.linspace(0.1, 0.5, 5),
            threshold=0.3,
        )
    assert caught == []
    assert out["metrics"]["roc_auc"] is None
    assert out["metrics"]["auprc"] is None
    assert out["metric_status"]["roc_auc"] == "undefined_single_class"
    assert out["metric_status"]["auprc"] == "undefined_single_class"


def test_uncalibrated_scores_do_not_produce_fake_log_loss():
    from 统一评测 import evaluate_all

    out = evaluate_all(
        np.array([0, 1, 0, 1]),
        np.array([-3.0, 8.0, 1.0, 10.0]),
        threshold=5.0,
    )
    assert out["metrics"]["log_loss"] is None
    assert out["metric_status"]["log_loss"] == "invalid_probability"


def test_probability_log_loss_uses_raw_positive_class_probability_not_oriented_score():
    from 统一评测 import evaluate_all

    out = evaluate_all(
        np.array([0, 1]),
        np.array([0.2, 0.8]),
        threshold=-0.5,
        context={"scores_are_probabilities": True, "score_polarity": "negative"},
    )
    assert math.isclose(out["metrics"]["log_loss"], -math.log(0.8))
    assert out["metric_status"]["log_loss"] == "ok"


def test_event_metrics_are_turbine_aware_and_report_two_far_units():
    from 统一评测 import evaluate_all

    ts = pd.date_range("2024-01-01", periods=12, freq="10min", tz="UTC")
    ts_ns = ts.tz_convert(None).as_unit("ns").asi8
    turbines = np.array(["A"] * 6 + ["B"] * 6)
    timestamps = np.concatenate([ts_ns[:6], ts_ns[:6]])
    y = np.array([0, 1, 1, 1, -1, -1] + [0] * 6)
    score = np.array([0.1, 0.9, 0.8, 0.7, 0.0, 0.0] + [0.9, 0.9, 0, 0, 0, 0])
    out = evaluate_all(
        y,
        score,
        timestamps=timestamps,
        turbines=turbines,
        event_table=_episodes(),
        threshold=0.5,
        context={"lead_steps": 6},
    )
    m = out["metrics"]
    assert m["n_events"] == 1
    assert m["n_false_segments"] == 1
    assert m["false_positive_points_per_turbine_day"] > 0
    assert m["false_alarm_segments_per_turbine_day"] > 0
    assert m["tier1_n"] == 1


def test_forecast_metrics_are_only_computed_when_arrays_are_available():
    from 统一评测 import evaluate_all

    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 1.0], [2.0, 4.0]])
    out = evaluate_all(
        np.array([0, 1]),
        np.array([0.1, 0.9]),
        threshold=0.5,
        forecast_true=y_true,
        forecast_pred=y_pred,
    )
    assert math.isclose(out["metrics"]["mse"], 0.5)
    assert math.isclose(out["metrics"]["mae"], 0.5)
    assert math.isclose(out["metrics"]["rmse"], math.sqrt(0.5))


def test_select_operating_points_returns_three_validation_only_workpoints():
    from 统一评测 import select_operating_points

    ts = pd.date_range("2024-01-01", periods=18, freq="10min", tz="UTC")
    ts_ns = ts.tz_convert(None).as_unit("ns").asi8
    turbines = np.array(["A"] * len(ts_ns))
    y = np.zeros(len(ts_ns), dtype=int)
    y[:6] = 1
    scores = np.linspace(1.0, 0.0, len(ts_ns))
    eps = pd.DataFrame(
        {
            "_turbine": ["A"],
            "Timestamp start": [pd.Timestamp("2024-01-01 01:00", tz="UTC")],
            "Timestamp end": [pd.Timestamp("2024-01-01 01:10", tz="UTC")],
            "tier": ["tier1"],
        }
    )
    selected = select_operating_points(y, scores, ts_ns, turbines, eps, lead_steps=6)
    assert selected["score_polarity"] in {"positive", "negative"}
    assert set(selected["workpoints"]) == {"balanced", "low_far", "high_recall"}
    for name, rec in selected["workpoints"].items():
        assert np.isfinite(rec["threshold"]), name
        assert rec["threshold_source"].startswith("validation_")
        assert "validation_metrics" in rec
