from __future__ import annotations

import numpy as np
import pytest

from realfault_benchmark.contracts import DatasetBundle, ScoreView
from realfault_benchmark.sequences import (
    SequenceSpec,
    build_sequence_index,
    extract_windows,
    scatter_window_scores,
)


def _score_view() -> ScoreView:
    # 原始行顺序故意混排两个机组；窗口内部必须按各机组时间排序，最终分数仍按原始行对齐。
    return ScoreView(
        X=np.arange(16, dtype=np.float32).reshape(8, 2),
        timestamps=np.array([0, 10, 0, 20, 10, 50, 60, 70], dtype=np.int64),
        turbines=np.array(["A", "A", "B", "A", "B", "A", "A", "A"]),
        row_ids=np.array([100, 101, 200, 102, 201, 103, 104, 105], dtype=np.int64),
        split="test",
    )


def _bundle(test_labels: np.ndarray) -> DatasetBundle:
    test_view = _score_view()
    return DatasetBundle.from_arrays(
        dataset="synthetic",
        farm="farm_a",
        variant="realfault",
        feature_names=("temperature", "residual"),
        train_normal_X=np.zeros((2, 2), dtype=np.float32),
        train_normal_labels=np.zeros(2, dtype=np.int8),
        train_normal_timestamps=np.array([0, 10], dtype=np.int64),
        train_normal_turbines=np.array(["A", "A"]),
        train_normal_row_ids=np.array([0, 1], dtype=np.int64),
        train_normal_gap_mask=np.array([False, False]),
        train_supervised_X=np.zeros((2, 2), dtype=np.float32),
        train_supervised_labels=np.array([0, 1], dtype=np.int8),
        train_supervised_timestamps=np.array([0, 10], dtype=np.int64),
        train_supervised_turbines=np.array(["A", "A"]),
        train_supervised_row_ids=np.array([0, 1], dtype=np.int64),
        train_supervised_gap_mask=np.array([False, False]),
        val_X=np.zeros((2, 2), dtype=np.float32),
        val_labels=np.array([0, 1], dtype=np.int8),
        val_timestamps=np.array([0, 10], dtype=np.int64),
        val_turbines=np.array(["A", "A"]),
        val_row_ids=np.array([0, 1], dtype=np.int64),
        val_gap_mask=np.array([False, False]),
        test_X=test_view.X,
        test_labels=test_labels,
        test_timestamps=test_view.timestamps,
        test_turbines=test_view.turbines,
        test_row_ids=test_view.row_ids,
        test_gap_mask=np.zeros(len(test_view.X), dtype=bool),
        event_table=(),
        split_hash="split-hash",
        feature_hash="feature-hash",
        file_hash="file-hash",
    )


def test_windows_never_cross_turbine_or_time_break_and_scores_stay_row_aligned() -> None:
    view = _score_view()
    index = build_sequence_index(
        view,
        SequenceSpec(window_size=3, cadence=10, max_gap=10),
    )

    # A 的 0/10/20 与 50/60/70 是两段独立连续序列；B 只有两行，不能形成窗口。
    assert index.split == "test"
    assert index.n_rows == len(view.X)
    assert index.window_positions.tolist() == [[0, 1, 3], [5, 6, 7]]
    assert index.target_positions.tolist() == [3, 7]
    assert index.target_row_ids.tolist() == [102, 105]

    windows = extract_windows(view, index)
    np.testing.assert_array_equal(windows[0], view.X[[0, 1, 3]])
    np.testing.assert_array_equal(windows[1], view.X[[5, 6, 7]])

    aligned = scatter_window_scores(index, np.array([0.25, 0.75]))
    assert aligned.shape == (len(view.X),)
    assert np.isnan(aligned[[0, 1, 2, 4, 5, 6]]).all()
    np.testing.assert_allclose(aligned[[3, 7]], [0.25, 0.75])


def test_sequence_index_is_unchanged_when_test_labels_are_modified_or_shuffled() -> None:
    original = _bundle(np.array([0, 1, -1, 0, 1, 0, -1, 0], dtype=np.int8))
    changed = _bundle(np.array([1, 0, 0, -1, 0, 1, 0, 1], dtype=np.int8))
    spec = SequenceSpec(window_size=3, cadence=10, max_gap=10)

    first = build_sequence_index(original.score_view("test"), spec)
    second = build_sequence_index(changed.score_view("test"), spec)

    np.testing.assert_array_equal(first.window_positions, second.window_positions)
    np.testing.assert_array_equal(first.target_positions, second.target_positions)
    np.testing.assert_array_equal(first.target_row_ids, second.target_row_ids)


def test_non_cadence_delta_and_gap_both_start_new_segments() -> None:
    view = ScoreView(
        X=np.arange(12, dtype=np.float32).reshape(6, 2),
        timestamps=np.array([0, 10, 25, 35, 55, 65], dtype=np.int64),
        turbines=np.array(["A"] * 6),
        row_ids=np.arange(6, dtype=np.int64),
        split="val",
    )

    index = build_sequence_index(view, SequenceSpec(window_size=2, cadence=10, max_gap=15))

    # 10->25 不符合固定 cadence；35->55 超过 max_gap。两处都必须断开。
    assert index.window_positions.tolist() == [[0, 1], [2, 3], [4, 5]]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 1, "cadence": 10, "max_gap": 10},
        {"window_size": 3, "cadence": 0, "max_gap": 10},
        {"window_size": 3, "cadence": 10, "max_gap": 5},
    ],
)
def test_sequence_spec_rejects_invalid_contracts(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        SequenceSpec(**kwargs)


def test_sequence_index_rejects_duplicate_row_ids() -> None:
    view = _score_view()
    duplicate_ids = view.row_ids.copy()
    duplicate_ids[-1] = duplicate_ids[0]
    invalid = ScoreView(view.X, view.timestamps, view.turbines, duplicate_ids, view.split)

    with pytest.raises(ValueError, match="row_id"):
        build_sequence_index(invalid, SequenceSpec(window_size=3, cadence=10, max_gap=10))
