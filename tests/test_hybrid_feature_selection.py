# -*- coding: utf-8 -*-
"""混合特征选择.py (MI-AMO-BHS 干净重实现) 的契约测试。

覆盖: ①regression 目标找回合成 ground-truth 真特征; ②n_max>维度须夹紧不崩(源码已知bug);
     ③reconstruction 无标签目标返回真子集; ④同种子确定性。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from 混合特征选择 import HybridFeatureSelector


def _synth(n=800, d=30, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = 2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + 0.8 * X[:, 3] + 0.5 * X[:, 4] + rng.normal(0, 0.5, n)
    return X, y


def test_regression_recovers_true_features():
    """回归目标应在 Pareto best_accuracy 解里找回至少 4/5 个真特征 (0-4)。"""
    X, y = _synth()
    fs = HybridFeatureSelector(objective="regression", beta=0.5, hms=15,
                               max_iter=40, n_min=3, n_max=12, random_state=42).fit(X, y)
    sel = set(fs.get_selected("best_accuracy"))
    assert len(sel & {0, 1, 2, 3, 4}) >= 4


def test_nmax_clamped_no_crash():
    """n_max=38 > 过滤后维度: 源码会 ValueError, 干净版须夹紧到可行。"""
    X, y = _synth(d=10)
    fs = HybridFeatureSelector(objective="regression", beta=0.6, hms=10,
                               max_iter=20, n_min=3, n_max=38, random_state=1).fit(X, y)
    assert len(fs.get_selected("balanced")) >= 1


def test_reconstruction_mode_subset_smaller():
    """reconstruction 无标签目标: 选出的子集应非空且严格小于全维。"""
    X, _ = _synth(d=20)
    fs = HybridFeatureSelector(objective="reconstruction", beta=0.7, hms=12,
                               max_iter=30, n_min=3, n_max=15, random_state=7).fit(X)
    sel = fs.get_selected("balanced")
    assert 1 <= len(sel) < X.shape[1]


def test_deterministic():
    """同种子两次 fit 应得逐一致的选择 (可复现)。"""
    X, y = _synth()
    a = HybridFeatureSelector(objective="regression", random_state=5, max_iter=20).fit(X, y).get_selected()
    b = HybridFeatureSelector(objective="regression", random_state=5, max_iter=20).fit(X, y).get_selected()
    assert list(a) == list(b)
