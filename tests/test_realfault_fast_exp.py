# -*- coding: utf-8 -*-
"""真实故障快速实验运行器 — 核心纯函数测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "快速实验"))

import 真实故障事件级实验 as rf  # noqa: E402


def test_per_turbine_causal_features_no_cross_turbine_leak():
    # 两台机组交错; A 全 1.0, B 全 10.0 → 逐机组滚动均值必须各自恒定
    X = np.array([[1.0], [10.0], [1.0], [10.0], [1.0], [10.0]], dtype=np.float32)
    turb = np.array(["A", "B", "A", "B", "A", "B"])
    F = rf.per_turbine_causal_features(X, turb, w_feat=3, k_recent=2)
    assert F.shape == (6, 1 + 6)
    roll_mean_col = F[:, 3]                      # [X | maxc, pose, roll_mean, roll_max, slope, recent]
    a_rows = turb == "A"
    assert np.allclose(roll_mean_col[a_rows], 1.0)     # A 的滚动均值不受 B 污染
    assert np.allclose(roll_mean_col[~a_rows], 10.0)   # B 同理
    slope_col = F[:, 5]
    assert np.allclose(slope_col, 0.0)                 # 常数序列斜率为 0


def test_ewma_and_cusum_are_causal_and_reset_per_turbine():
    x = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=float)
    turb = np.array(["A", "A", "A", "B", "B", "B"])
    e = rf.per_turbine_ewma(x, turb, alpha=0.5)
    assert np.allclose(e[:3], [0.0, 0.0, 0.5])         # A: EWMA 因果
    assert np.allclose(e[3:], [0.0, 0.0, 0.5])         # B 从零重启 (不带 A 的状态)
    c = rf.per_turbine_cusum(x, turb, k=0.25)
    assert c[2] > c[1]                                  # 超参考值累积
    assert c[3] == 0.0                                  # B 重启
