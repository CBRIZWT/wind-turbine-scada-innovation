# -*- coding: utf-8 -*-
"""
tests/test_choose_threshold_fast.py — choose_threshold 向量化 (2026-06-05)

回归: 旧版对【每个】候选阈值(≈val 不同分数, kelmarsh ~31万)都调一次 sklearn
precision_recall_fscore_support(全 val) → O(n²), 每 epoch 数十分钟、网格几周跑不完
(py-spy 实证)。向量化后 O(n log n)、毫秒级, 且与旧 for 循环【逐位等价】。
本测试锁定: (1) 与旧 for 参考实现逐位一致; (2) 大 n 秒级返回。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from 实验工具 import (
    choose_threshold_and_polarity_by_validation,
    orient_scores,
    _threshold_candidates,
    MIN_PRECISION_FLOOR,
    POLARITY_FALLBACK_F1_THRESHOLD,
)


def _loop_reference(y, s):
    """旧 O(n²) 逻辑的忠实复刻 (AUC 极性 + orient + for 循环 sklearn PRF)。"""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    fin = np.isfinite(s)
    y, s = y[fin], s[fin]
    if y.size == 0 or np.unique(y).size < 2:
        return ("fallback",)
    pol = "positive" if float(roc_auc_score(y, s)) >= 0.5 else "negative"
    o = orient_scores(s, pol)
    best_thr, best_f1, seen = 0.0, -1.0, False
    for t in _threshold_candidates(o):
        pred = (o > t).astype(int)
        p, _, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0)
        if float(p) < MIN_PRECISION_FLOOR:
            continue
        seen = True
        if float(f1) > best_f1:
            best_f1, best_thr = float(f1), float(t)
    if not seen or best_f1 < POLARITY_FALLBACK_F1_THRESHOLD:
        return ("fallback",)
    return (best_thr, "validation_evt1_f1", pol)


def test_vectorized_matches_loop_reference():
    """向量化结果与旧 for 循环逐位一致 (多随机种子)。"""
    for seed in range(5):
        rng = np.random.default_rng(seed)
        n = 1500
        y = (rng.random(n) > 0.85).astype(int)        # ~15% 正例
        s = rng.normal(size=n) + y * 1.5              # 分数与 y 强相关 (非 fallback)
        got = choose_threshold_and_polarity_by_validation(y, s, fallback_scores=s)
        ref = _loop_reference(y, s)
        assert ref[0] != "fallback", (seed, "参考进了 fallback")
        assert got[1] == ref[1] and got[2] == ref[2], (seed, got, ref)
        assert abs(float(got[0]) - float(ref[0])) <= 1e-12, (seed, got[0], ref[0])


def test_vectorized_fast_on_large_n():
    """大 n (20万): 向量化应 <5s (旧 O(n²) 会几十分钟)。"""
    rng = np.random.default_rng(0)
    n = 200_000
    y = (rng.random(n) > 0.9).astype(int)
    s = rng.normal(size=n) + y * 1.0
    t0 = time.time()
    got = choose_threshold_and_polarity_by_validation(y, s, fallback_scores=s)
    dt = time.time() - t0
    assert dt < 5.0, f"耗时 {dt:.1f}s 太慢 (向量化应 <1s)"
    assert isinstance(got[0], float)


def test_fallback_callable_not_called_when_validation_threshold_succeeds():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    s = np.array([0.1, 0.2, 0.25, 1.0, 1.1, 1.2], dtype=float)
    called = {"n": 0}

    def fallback():
        called["n"] += 1
        return np.array([0.1, 0.2, 0.3], dtype=float)

    threshold, source, polarity = choose_threshold_and_polarity_by_validation(y, s, fallback)

    assert isinstance(threshold, float)
    assert source == "validation_evt1_f1"
    assert polarity == "positive"
    assert called["n"] == 0


def test_fallback_callable_called_when_validation_is_single_class():
    y = np.zeros(6, dtype=int)
    s = np.linspace(0.0, 1.0, 6)
    called = {"n": 0}

    def fallback():
        called["n"] += 1
        return np.array([1.0, 2.0, 3.0], dtype=float)

    threshold, source, polarity = choose_threshold_and_polarity_by_validation(y, s, fallback)

    assert threshold == float(np.quantile([1.0, 2.0, 3.0], 0.99))
    assert source == "train_q0.99_uncalibrated"
    assert polarity == "positive_fallback_uncalibrated"
    assert called["n"] == 1
