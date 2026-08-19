# -*- coding: utf-8 -*-
"""监督早警头可复用函数的契约测试 (2026-06-15, TDD)。

fit_predict_supervised: 在 train 特征+标签上拟合 GBM, 对任意 eval 特征出 P(event)。
  - 排除 y==-1 (ignore) 不参与训练; 类不平衡用 balanced 权重;
  - 可分数据上 → 正类分数显著高于负类 (AUC 高)。供 lead-time 前沿按 H 重训复用。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from 监督早警 import fit_predict_supervised  # noqa: E402


def _make(n, rng, shift):
    y = (rng.random(n) < 0.3).astype(int)
    F = rng.standard_normal((n, 4))
    F[:, 0] += y * shift                      # 第0列携带可分信号
    return F, y


def test_fit_predict_separates_classes():
    rng = np.random.default_rng(0)
    Ftr, ytr = _make(600, rng, shift=4.0)
    Fte, yte = _make(300, rng, shift=4.0)
    (st,) = fit_predict_supervised(Ftr, ytr, Fte)
    assert st.shape == (300,)
    assert roc_auc_score(yte, st) > 0.9


def test_fit_predict_excludes_ignore_rows():
    """y==-1 行不得参与训练 (否则注入的纯噪声 ignore 标签会污染)。"""
    rng = np.random.default_rng(1)
    Ftr, ytr = _make(600, rng, shift=4.0)
    ytr = ytr.copy(); ytr[:100] = -1          # 前100行设 ignore
    Fte, yte = _make(300, rng, shift=4.0)
    (st,) = fit_predict_supervised(Ftr, ytr, Fte)
    assert roc_auc_score(yte, st) > 0.9


def test_fit_predict_multiple_eval_sets():
    rng = np.random.default_rng(2)
    Ftr, ytr = _make(500, rng, shift=4.0)
    Fa, _ = _make(120, rng, shift=4.0); Fb, _ = _make(80, rng, shift=4.0)
    sa, sb = fit_predict_supervised(Ftr, ytr, Fa, Fb)
    assert sa.shape == (120,) and sb.shape == (80,)
