# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "快速实验"))


def test_sort_by_turbine_time_makes_each_turbine_contiguous_and_chronological():
    from 数据工具_v3 import sort_by_turbine_time

    X = np.array([[20.0], [1.0], [10.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    ts = np.array([20, 10, 10, 20], dtype=np.int64)
    turb = np.array(["B", "A", "B", "A"])
    sx, sy, sts, sturb, order = sort_by_turbine_time(X, y, ts, turb)
    assert sturb.tolist() == ["A", "A", "B", "B"]
    assert sts.tolist() == [10, 20, 10, 20]
    assert sx[:, 0].tolist() == [1.0, 2.0, 10.0, 20.0]
    assert sy.tolist() == [0, 1, 1, 0]
    assert np.array_equal(order, np.array([1, 3, 2, 0]))


def test_causal_features_are_computed_separately_per_turbine():
    from 数据工具_v3 import per_turbine_causal_features

    X = np.array([[1.0], [1.0], [1.0], [10.0], [10.0], [10.0]], dtype=np.float32)
    turb = np.array(["A", "A", "A", "B", "B", "B"])
    F = per_turbine_causal_features(X, turb, w_feat=2, k_recent=1)
    assert F.shape == (6, 7)
    assert np.allclose(F[:3, 3], 1.0)
    assert np.allclose(F[3:, 3], 10.0)
    assert np.allclose(F[:, 5:], 0.0)


def test_sequence_indices_never_cross_turbines_or_ignore_history():
    from 数据工具_v3 import sequence_end_indices

    turb = np.array(["A"] * 5 + ["B"] * 5)
    labels = np.array([0, 0, 1, 0, 0, 0, -1, 0, 1, 0])
    idx = sequence_end_indices(turb, labels, window=3, stride=1)
    # A: endpoints 2,3,4 valid; B: endpoint7窗口含ignore，8也含ignore，只有9窗口7:9无ignore
    assert idx.tolist() == [2, 3, 4, 9]
    for end in idx:
        assert len(set(turb[end - 2:end + 1])) == 1
        assert np.all(labels[end - 2:end + 1] != -1)


def test_sample_valid_indices_applies_stride_inside_each_turbine():
    from 数据工具_v3 import sample_valid_indices

    turb = np.array(["A"] * 5 + ["B"] * 4)
    labels = np.array([0, 1, -1, 0, 1, 0, 1, 0, 1])
    idx = sample_valid_indices(turb, labels, stride=2)
    assert idx.tolist() == [0, 3, 5, 7]
    assert np.all(labels[idx] != -1)


def test_derive_split_keeps_flat_and_sequence_sidecars_aligned():
    from 准备真实故障数据_v3 import derive_split

    X = np.array([[10.0], [1.0], [20.0], [2.0], [30.0], [3.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 0, 0])
    ts = np.array([10, 10, 20, 20, 30, 30], dtype=np.int64)
    turb = np.array(["B", "A", "B", "A", "B", "A"])
    out = derive_split(
        X,
        y,
        ts,
        turb,
        split="test",
        flat_stride=1,
        seq_stride=1,
        window=2,
        w_feat=2,
        k_recent=1,
    )
    assert out["X_base"].shape == (6, 1)
    assert out["X_common_base"].shape == (6, 6)
    assert out["turbines_base"].tolist() == ["A", "A", "A", "B", "B", "B"]
    assert np.array_equal(out["y_flat"], out["labels_base"][out["idx_flat"]])
    assert np.array_equal(out["timestamps_flat"], out["timestamps_base"][out["idx_flat"]])
    assert np.array_equal(out["y_seq"], out["labels_base"][out["idx_seq"]])
    assert np.array_equal(out["turbines_seq"], out["turbines_base"][out["idx_seq"]])
    for end in out["idx_seq"]:
        assert out["turbines_base"][end - 1] == out["turbines_base"][end]
