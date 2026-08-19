# -*- coding: utf-8 -*-
"""评测核心 共享原语的契约测试 (2026-06-15, TDD)。

抽出 p1/p2/p3/四模型消融/卡尔曼评价 重复的: 对齐 / val选阈+极性 / KF选参 / test评测。
防泄漏纪律: 阈值/极性/KF 仅用 val; test 仅评一次; ignore(-1) 与非有限值不参与。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from 评测核心 import align_to_axis, select_threshold_polarity, select_kalman, evaluate_test  # noqa: E402


def test_align_to_axis_pads_front_with_nan():
    raw = np.array([1.0, 2.0, 3.0])
    out = align_to_axis(raw, T=5)
    assert len(out) == 5
    assert np.isnan(out[:2]).all() and out[2:].tolist() == [1.0, 2.0, 3.0]


def test_align_to_axis_truncates_overlong():
    raw = np.arange(7.0)
    out = align_to_axis(raw, T=5)            # 超长则取后段对齐到末尾
    assert len(out) == 5 and out[-1] == 6.0


def test_select_threshold_polarity_positive():
    """高分=异常 → 选 sign=+1。"""
    vs = np.array([0., 0.1, 0.2, 5., 6., 7.])
    vy = np.array([0, 0, 0, 1, 1, 1])
    sign, thr, f1 = select_threshold_polarity(vs, vy)
    assert sign == 1.0 and f1 == 1.0
    assert (sign * vs >= thr).astype(int).tolist() == vy.tolist()


def test_select_threshold_polarity_negative():
    """低分=异常 → 选 sign=-1。"""
    vs = np.array([7., 6., 5., 0.2, 0.1, 0.])
    vy = np.array([0, 0, 0, 1, 1, 1])
    sign, thr, f1 = select_threshold_polarity(vs, vy)
    assert sign == -1.0 and f1 == 1.0


def test_select_threshold_polarity_excludes_ignore_and_nan():
    """label==-1 与 NaN 分数不参与选阈。"""
    vs = np.array([np.nan, 5., 6., 0., 0.1, 100.])
    vy = np.array([1, 1, 1, 0, 0, -1])      # idx0 NaN, idx5 ignore → 都排除
    sign, thr, f1 = select_threshold_polarity(vs, vy)
    assert sign == 1.0 and f1 == 1.0


def test_evaluate_test_perfect_separation():
    ts = np.array([0., 0.1, 0.2, 5., 6., 7.])
    ty = np.array([0, 0, 0, 1, 1, 1])
    m = evaluate_test(ts, ty, sign=1.0, thr=1.0)
    assert m["f1"] == 1.0 and m["auc"] == 1.0 and m["auprc"] == 1.0


def test_evaluate_test_single_class_auc_nan():
    """test 全为正类 → AUC 无定义=nan, 不崩。"""
    ts = np.array([5., 6., 7.])
    ty = np.array([1, 1, 1])
    m = evaluate_test(ts, ty, sign=1.0, thr=1.0)
    assert np.isnan(m["auc"])


def test_select_kalman_returns_grid_member():
    rng = np.random.default_rng(0)
    vs = rng.standard_normal(300); vy = (np.arange(300) % 50 < 5).astype(int)
    pv, mv = select_kalman(vs, vy)
    assert pv > 0 and mv > 0
