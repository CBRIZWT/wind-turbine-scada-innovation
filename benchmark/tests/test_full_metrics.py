from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from realfault_benchmark.full_metrics import (
    evaluate_full_metrics,
    one_to_one_event_prf,
    point_adjust_f1_turbine_macro,
    turbine_macro_range,
)


MINUTE_NS = 60 * 1_000_000_000


def test_turbine_macro_range_does_not_join_turbines() -> None:
    labels = np.array([0, 1, 1, 0, 0, 0, 0, 0], dtype=np.int8)
    pred = np.array([0, 1, 1, 0, 1, 1, 0, 0], dtype=np.int8)
    timestamps = np.array([0, 10, 20, 30, 0, 10, 20, 30], dtype=np.int64) * MINUTE_NS
    turbines = np.array(["A"] * 4 + ["B"] * 4)
    result = turbine_macro_range(labels, pred, timestamps, turbines)
    assert result["range_precision"] == 1.0
    assert result["range_recall"] == 1.0
    assert result["range_turbines_used"] == 1
    assert result["range_turbines_skipped"] == ["B"]


def test_one_to_one_event_prf_penalizes_extra_alarm_segment() -> None:
    timestamps = np.array([50, 250, 400], dtype=np.int64) * MINUTE_NS
    turbines = np.array(["A", "A", "A"])
    pred = np.ones(3, dtype=np.int8)
    events = pd.DataFrame(
        {
            "turbine": ["A", "A"],
            "start": pd.to_datetime([100, 300], unit="m", utc=True),
            "end": pd.to_datetime([110, 310], unit="m", utc=True),
        }
    )
    result = one_to_one_event_prf(
        pred, timestamps, turbines, events,
        labels=np.array([1, 1, 0], dtype=np.int8), events_are_episodes=True,
        horizon=timedelta(minutes=60), merge_gap=timedelta(minutes=40),
    )
    assert result["one_to_one_alarm_segments"] == 3
    assert result["one_to_one_matched"] == 2
    assert np.isclose(result["one_to_one_event_precision"], 2 / 3)
    assert result["one_to_one_event_recall"] == 1.0
    assert np.isclose(result["one_to_one_event_f1"], 0.8)


def test_full_metrics_normalizes_raw_events_once_for_every_metric() -> None:
    timestamps = np.arange(0, 160, 10, dtype=np.int64) * MINUTE_NS
    turbines = np.array(["A"] * len(timestamps))
    labels = np.zeros(len(timestamps), dtype=np.int8)
    labels[(timestamps >= 50 * MINUTE_NS) & (timestamps < 70 * MINUTE_NS)] = 1
    scores = labels.astype(float)
    raw_events = pd.DataFrame(
        {
            "turbine": ["A", "A"],
            "start": pd.to_datetime([60, 70], unit="m", utc=True),
            "end": pd.to_datetime([61, 71], unit="m", utc=True),
            "split": ["test", "test"],
        }
    )
    result = evaluate_full_metrics(
        labels, scores, timestamps, turbines, raw_events,
        threshold=0.5, polarity="positive", split="test",
        horizon=timedelta(minutes=20),
    )
    assert result["event_count"] == 1
    assert result["one_to_one_event_count"] == 1


def test_one_to_one_requires_an_actual_alarm_inside_window_not_segment_envelope() -> None:
    timestamps = np.array([80, 110], dtype=np.int64) * MINUTE_NS
    turbines = np.array(["A", "A"])
    events = pd.DataFrame(
        {"turbine": ["A"], "start": pd.to_datetime([100], unit="m", utc=True)}
    )
    result = one_to_one_event_prf(
        np.ones(2, dtype=np.int8), timestamps, turbines, events,
        labels=np.zeros(2, dtype=np.int8), events_are_episodes=True,
        horizon=timedelta(minutes=10), merge_gap=timedelta(minutes=40),
    )
    assert result["one_to_one_alarm_segments"] == 1
    assert result["one_to_one_matched"] == 0


def test_one_to_one_excludes_ignore_alarms_and_breaks_across_ignore_rows() -> None:
    timestamps = np.array([0, 10, 20], dtype=np.int64) * MINUTE_NS
    turbines = np.array(["A", "A", "A"])
    labels = np.array([0, -1, 0], dtype=np.int8)
    result = one_to_one_event_prf(
        np.ones(3, dtype=np.int8), timestamps, turbines,
        pd.DataFrame(columns=["turbine", "start"]),
        labels=labels, events_are_episodes=True, merge_gap=timedelta(minutes=40),
    )
    assert result["one_to_one_alarm_segments"] == 2


def test_point_adjust_ignore_row_is_a_hard_range_boundary() -> None:
    result = point_adjust_f1_turbine_macro(
        np.array([1, -1, 1], dtype=np.int8),
        np.array([1, 0, 0], dtype=np.int8),
        np.array([0, 10, 20], dtype=np.int64) * MINUTE_NS,
        np.array(["A", "A", "A"]),
    )
    assert np.isclose(result["pa_f1_appendix"], 2 / 3)
