from __future__ import annotations

import json

import numpy as np

from realfault_benchmark.artifacts import strict_json_dumps
from realfault_benchmark.reference_metrics import affiliation_prf
from realfault_benchmark.metrics import tatbul_range_prf


def test_tatbul_range_perfect_and_miss_fixtures() -> None:
    y = np.array([0, 1, 1, 1, 0], dtype=np.int8)
    perfect = tatbul_range_prf(y, y, alpha=0.0)
    miss = tatbul_range_prf(y, np.zeros_like(y), alpha=0.0)
    assert perfect == {"range_precision": 1.0, "range_recall": 1.0, "range_f1": 1.0}
    assert miss == {"range_precision": 0.0, "range_recall": 0.0, "range_f1": 0.0}


def test_tatbul_range_cardinality_penalizes_fragmented_recall() -> None:
    y_true = np.array([0, 1, 1, 1, 0], dtype=np.int8)
    y_pred = np.array([0, 1, 0, 1, 0], dtype=np.int8)
    got = tatbul_range_prf(y_true, y_pred, alpha=0.0)
    assert np.isclose(got["range_recall"], 1.0 / 3.0)
    assert np.isclose(got["range_precision"], 1.0)
    assert np.isclose(got["range_f1"], 0.5)


def test_tatbul_precision_never_receives_existence_reward() -> None:
    # Tatbul et al. (NeurIPS 2018, Sec. 4.2): existence reward belongs to
    # range recall; precision remains a pure overlap/cardinality reward.
    got = tatbul_range_prf(
        np.array([0, 0, 1, 0, 0], dtype=np.int8),
        np.ones(5, dtype=np.int8),
        alpha=1.0,
    )
    assert got["range_recall"] == 1.0
    assert got["range_precision"] == 0.2
    assert np.isclose(got["range_f1"], 1.0 / 3.0)


def test_affiliation_matches_author_readme_fixture() -> None:
    y_pred = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.int8)
    y_true = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 1], dtype=np.int8)
    got = affiliation_prf(y_true, y_pred)
    assert np.isclose(got["affiliation_precision"], 0.82, atol=0.01)
    assert np.isclose(got["affiliation_recall"], 0.84, atol=0.01)
    expected_f1 = 2 * got["affiliation_precision"] * got["affiliation_recall"] / (
        got["affiliation_precision"] + got["affiliation_recall"]
    )
    assert np.isclose(got["affiliation_f1"], expected_f1)


def test_strict_json_replaces_non_finite_with_null_and_status() -> None:
    payload = {"metrics": {"roc_auc": np.nan, "score": 0.5}, "metric_status": {}}
    text = strict_json_dumps(payload)
    decoded = json.loads(text)
    assert decoded["metrics"]["roc_auc"] is None
    assert decoded["metric_status"]["roc_auc"] == "non_finite"
    assert "NaN" not in text and "Infinity" not in text
