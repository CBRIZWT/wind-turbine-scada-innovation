from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from realfault_benchmark.calibration import select_equal4_calibration
from realfault_benchmark.metrics import evaluate_equal4, false_alarm_burden


MINUTE_NS = 60 * 1_000_000_000


def test_equal4_event_coverage_precision_and_lead_are_turbine_aware() -> None:
    # 两台机组时间戳相同；只允许同机组报警命中其事件。
    timestamps = np.array([0, 30, 60, 90, 0, 30, 60, 90], dtype=np.int64) * MINUTE_NS
    turbines = np.array(["A"] * 4 + ["B"] * 4)
    labels = np.array([0, 0, 1, -1, 0, 0, 1, -1], dtype=np.int8)
    scores = np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
    events = pd.DataFrame(
        {
            "turbine": ["A", "B"],
            "start": pd.to_datetime([90, 90], unit="m", utc=True),
            "end": pd.to_datetime([100, 100], unit="m", utc=True),
        }
    )
    result = evaluate_equal4(
        labels,
        scores,
        timestamps,
        turbines,
        events,
        threshold=0.5,
        polarity="positive",
        horizon=timedelta(minutes=60),
        nominal_cadence=timedelta(minutes=30),
    )
    assert result["event_count"] == 2
    assert result["event_detected"] == 1
    assert result["event_recall"] == 0.5
    assert result["alarm_point_precision"] == 0.0  # 报警点本身是健康行，按计划的点精确率定义
    assert result["mean_normalized_earliness"] == 0.5
    assert result["false_alarm_segments"] == 0  # 位于 A 的早警窗，不是假报警段


def test_false_alarm_segments_merge_by_time_and_ignore_forces_break() -> None:
    timestamps = np.array([0, 30, 50, 70, 100], dtype=np.int64) * MINUTE_NS
    turbines = np.array(["A"] * 5)
    labels = np.array([0, 0, -1, 0, 0], dtype=np.int8)
    predictions = np.array([1, 1, 0, 1, 1], dtype=np.int8)
    burden = false_alarm_burden(
        labels,
        predictions,
        timestamps,
        turbines,
        events=pd.DataFrame(columns=["turbine", "start", "end"]),
        horizon=timedelta(hours=12),
        merge_gap=timedelta(minutes=40),
        nominal_cadence=timedelta(minutes=10),
    )
    assert burden["false_alarm_segments"] == 2
    assert burden["healthy_turbine_days"] == 4 * 10 / (24 * 60)


def test_validation_calibration_checks_both_polarities_and_never_accepts_test_labels() -> None:
    timestamps = np.arange(8, dtype=np.int64) * 10 * MINUTE_NS
    turbines = np.array(["A"] * 8)
    labels = np.array([0, 0, 0, 0, 0, 0, 1, -1], dtype=np.int8)
    # 低分代表异常，正确极性应为 negative。
    scores = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 1.0, 0.0, 4.0])
    events = pd.DataFrame(
        {"turbine": ["A"], "start": pd.to_datetime([70], unit="m", utc=True),
         "end": pd.to_datetime([80], unit="m", utc=True)}
    )
    artifact = select_equal4_calibration(
        labels,
        scores,
        timestamps,
        turbines,
        events,
        model_id="model",
        dataset_id="dataset",
        validation_hash="hash",
        horizon=timedelta(minutes=20),
        nominal_cadence=timedelta(minutes=10),
    )
    assert artifact.polarity == "negative"
    assert artifact.threshold_source == "validation_equal4_201_quantiles"
    assert artifact.candidate_count <= 404
    assert "test" not in artifact.to_dict()


def test_non_finite_scores_are_never_alarms() -> None:
    labels = np.array([0, 1], dtype=np.int8)
    scores = np.array([np.nan, np.inf])
    timestamps = np.array([0, 10], dtype=np.int64) * MINUTE_NS
    turbines = np.array(["A", "A"])
    events = pd.DataFrame(
        {"turbine": ["A"], "start": pd.to_datetime([20], unit="m", utc=True),
         "end": pd.to_datetime([30], unit="m", utc=True)}
    )
    result = evaluate_equal4(labels, scores, timestamps, turbines, events, threshold=0.0)
    assert result["alarm_count"] == 0


def test_raw_event_table_requires_split_and_merges_only_that_split() -> None:
    labels = np.array([0, 1, -1], dtype=np.int8)
    scores = np.array([0.9, 0.0, 0.0])
    timestamps = np.array([0, 60, 120], dtype=np.int64) * MINUTE_NS
    turbines = np.array(["A", "A", "A"])
    raw_events = pd.DataFrame(
        {
            "turbine": ["A", "A", "B"],
            "start": pd.to_datetime([60, 70, 60], unit="m", utc=True),
            "end": pd.to_datetime([61, 71, 61], unit="m", utc=True),
            "split": ["val", "val", "test"],
        }
    )
    with pytest.raises(ValueError, match="split"):
        evaluate_equal4(labels, scores, timestamps, turbines, raw_events, threshold=0.5)
    result = evaluate_equal4(
        labels, scores, timestamps, turbines, raw_events, threshold=0.5, split="val",
        horizon=timedelta(minutes=60), nominal_cadence=timedelta(minutes=60),
    )
    assert result["event_count"] == 1  # 两条 val 原始事件在 72h 内合并；test 事件被隔离
    assert result["event_detected"] == 1
