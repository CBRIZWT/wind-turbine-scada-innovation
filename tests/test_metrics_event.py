from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import math
import numpy as np
from 实验工具 import compute_binary_metrics
from 事件指标 import (extract_events, compute_range_metrics, compute_affiliation_metrics,
                   false_alarms_per_day, point_adjust_f1)


def test_auprc_ok():
    labels = np.array([0, 0, 0, 1, 1]); scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8])
    m = compute_binary_metrics(labels, scores=scores, threshold=0.5)
    assert m["auprc_status"] == "ok"
    from sklearn.metrics import average_precision_score
    assert abs(m["auprc"] - average_precision_score(labels, scores)) < 1e-9


def test_auprc_single_class():
    m = compute_binary_metrics(np.zeros(4, int), scores=np.array([.1, .2, .3, .4]), threshold=0.5)
    assert m["auprc_status"] == "single_class" and math.isnan(m["auprc"])


def test_auprc_no_score():
    m = compute_binary_metrics(np.array([0, 1, 0, 1]), preds=np.array([0, 1, 0, 1]))
    assert m["auprc_status"] == "no_score" and math.isnan(m["auprc"])


def test_extract_events():
    assert extract_events(np.array([0, 1, 1, 0, 0, 1, 0])) == [(1, 2), (5, 5)]


def test_range_perfect_and_miss():
    y = np.array([0, 0, 1, 1, 1, 0, 0])
    assert compute_range_metrics(y, y.copy())["range_f1"] == 1.0
    miss = compute_range_metrics(y, np.zeros(7, int))
    assert miss["range_recall"] == 0.0 and miss["range_f1"] == 0.0


def test_affiliation_perfect_and_miss():
    y = np.array([0, 1, 1, 0, 0])
    assert compute_affiliation_metrics(y, y.copy())["affiliation_f1"] == 1.0
    assert compute_affiliation_metrics(y, np.zeros(5, int))["affiliation_recall"] == 0.0


def test_false_alarms_per_day():
    y_true = np.zeros(144, dtype=int)            # 144 steps × 10min = 1 day
    y_pred = np.zeros(144, dtype=int); y_pred[0] = 1
    assert abs(false_alarms_per_day(y_true, y_pred, step_minutes=10) - 1.0) < 1e-6


def test_point_adjust_appendix():
    y = np.array([0, 1, 1, 1, 0]); pred = np.array([0, 0, 1, 0, 0])   # hit 1 point in event
    assert point_adjust_f1(y, pred) == 1.0


def test_multi_event_partial():
    y = np.array([0, 1, 1, 0, 1, 1, 0]); pred = np.array([0, 1, 1, 0, 0, 0, 0])
    r = compute_range_metrics(y, pred)
    assert 0.0 < r["range_recall"] < 1.0


def test_aliases_exist():
    from 事件指标 import events_from_labels, range_prf, affiliation_prf
    assert events_from_labels(np.array([0, 1, 0])) == [(1, 1)]
    assert "range_f1" in range_prf(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0]))
    assert "affiliation_f1" in affiliation_prf(np.array([0, 1, 0]), np.array([0, 1, 0]))


def test_augment_event_metrics():
    from 实验工具 import augment_event_metrics
    m = augment_event_metrics({}, labels=np.array([0, 1, 1, 0]), preds=np.array([0, 1, 1, 0]))
    for k in ("range_f1", "affiliation_f1", "appendix_point_adjust_f1"):
        assert k in m
