from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from realfault_benchmark.contracts import DatasetBundle
from realfault_benchmark.paper_adapters import StatisticalRFAdapter
from realfault_benchmark.pipeline import evaluate_adapter


CADENCE = 600 * 1_000_000_000


def _bundle(test_labels: np.ndarray) -> DatasetBundle:
    n = 20
    X = np.linspace(0, 3, n, dtype=np.float32).reshape(-1, 1)
    train_y = np.array([0] * 10 + [1] * 10, dtype=np.int8)
    val_y = np.array([0] * 10 + [1] * 10, dtype=np.int8)
    return DatasetBundle.from_arrays(
        dataset="synthetic", farm="kelmarsh", variant="realfault", feature_names=("front bearing__resid",),
        train_normal_X=X[:10], train_normal_labels=np.zeros(10, dtype=np.int8),
        train_normal_timestamps=np.arange(10) * CADENCE, train_normal_turbines=np.array(["A"] * 10),
        train_normal_row_ids=np.arange(10), train_normal_gap_mask=np.zeros(10, dtype=bool),
        train_supervised_X=X, train_supervised_labels=train_y,
        train_supervised_timestamps=np.arange(n) * CADENCE, train_supervised_turbines=np.array(["A"] * n),
        train_supervised_row_ids=np.arange(n), train_supervised_gap_mask=np.zeros(n, dtype=bool),
        val_X=X, val_labels=val_y, val_timestamps=np.arange(n) * CADENCE,
        val_turbines=np.array(["A"] * n), val_row_ids=np.arange(n), val_gap_mask=np.zeros(n, dtype=bool),
        test_X=X, test_labels=test_labels, test_timestamps=np.arange(n) * CADENCE,
        test_turbines=np.array(["A"] * n), test_row_ids=np.arange(n), test_gap_mask=np.zeros(n, dtype=bool),
        event_table=pd.DataFrame(
            {"turbine": ["A", "A"], "start": pd.to_datetime([15, 15], unit="m", utc=True),
             "end": pd.to_datetime([16, 16], unit="m", utc=True), "split": ["val", "test"]}
        ),
        split_hash="split", feature_hash="feature", file_hash="file",
    )


def test_test_label_shuffle_cannot_change_fit_calibration_or_test_scores() -> None:
    first = _bundle(np.array([0] * 10 + [1] * 10, dtype=np.int8))
    second = _bundle(np.array([1, 0] * 10, dtype=np.int8))
    adapter1 = StatisticalRFAdapter(np.array([0]), n_estimators=9, max_train_rows=20)
    adapter2 = copy.deepcopy(adapter1)
    result1 = evaluate_adapter(first, adapter1, seed=20260719, device="cpu")
    result2 = evaluate_adapter(second, adapter2, seed=20260719, device="cpu")
    np.testing.assert_array_equal(result1["validation_scores"], result2["validation_scores"])
    np.testing.assert_array_equal(result1["test_scores"], result2["test_scores"])
    assert result1["calibration"].artifact_hash == result2["calibration"].artifact_hash
    assert result1["calibration"].threshold == result2["calibration"].threshold
