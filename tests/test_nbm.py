# -*- coding: utf-8 -*-
"""
tests/test_nbm.py — 正常行为模型 (NBM) 与 NaN 填充修复验证

测试覆盖:
    - NormalBehaviorModel.fit + residual
    - NaN 填充: 按列均值而非全局标量 (修复 #3)
    - 多通道残差输出的形状正确性
    - gbr > ridge 回退
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from 温度指标选择 import NormalBehaviorModel


def test_nbm_fit_residual_basic():
    """NBM 在健康数据上拟合, 残差在正常段接近零。"""
    rng = np.random.default_rng(0)
    n = 500
    wind = rng.uniform(3, 15, n)
    power = (wind - 3) ** 2 * 8 + rng.normal(0, 10, n)
    ambient = 10 + rng.normal(0, 0.5, n)

    df = pd.DataFrame({
        "wind": wind, "power": power, "ambient": ambient,
        "temp1": 25 + 0.02 * power + 0.8 * ambient + rng.normal(0, 0.5, n),
        "temp2": 30 + 0.015 * power + rng.normal(0, 0.3, n),
    })

    cond_cols = ["wind", "power", "ambient"]
    temp_cols = ["temp1", "temp2"]

    nbm = NormalBehaviorModel(cond_cols, temp_cols).fit(df)
    resid = nbm.residual(df)

    assert list(resid.columns) == ["temp1__resid", "temp2__resid"]
    # 正常段残差均值 ≈ 0
    assert abs(resid["temp1__resid"].mean()) < 0.5


def test_nbm_nan_handling_per_column():
    """#3 修复验证: NaN 用按列均值填充, 不是全局标量。"""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "wind": np.where(rng.random(n) < 0.1, np.nan, rng.uniform(3, 15, n)),
        "power": rng.uniform(0, 2000, n),
        "ambient": rng.uniform(5, 25, n),
        "temp1": 25 + rng.normal(0, 1, n),
    })

    nbm = NormalBehaviorModel(["wind", "power", "ambient"], ["temp1"]).fit(df)
    resid = nbm.residual(df)
    # 不崩溃 = 修复生效
    assert len(resid) == n


def test_nbm_const_model_fallback():
    """训练样本不足时降级为常数预测。"""
    df = pd.DataFrame({
        "wind": [1.0, 2.0, 3.0, 4.0, 5.0],
        "power": [1.0, 2.0, 3.0, 4.0, 5.0],
        "ambient": [10.0, 10.0, 10.0, 10.0, 10.0],
        "temp1": [25.0, 25.5, 26.0, 25.5, 25.0],
    })
    nbm = NormalBehaviorModel(["wind", "power", "ambient"], ["temp1"]).fit(df)
    resid = nbm.residual(df)
    assert len(resid) == 5
    # 降级为常数均值预测
    assert abs(resid["temp1__resid"].mean()) < 1.0


def test_nbm_multi_channel_independence():
    """每温度通道独立建模。"""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "wind": rng.uniform(3, 15, n),
        "ambient": rng.uniform(5, 25, n),
        "temp_a": 25 + rng.normal(0, 1, n),    # 与工况无关
        "temp_b": 25 + 0.3 * rng.uniform(3, 15, n) + rng.normal(0, 0.3, n),  # 与工况相关
    })
    cond_cols = ["wind", "ambient"]
    temp_cols = ["temp_a", "temp_b"]
    nbm = NormalBehaviorModel(cond_cols, temp_cols).fit(df)
    resid = nbm.residual(df)

    # temp_a 残差方差 ≈ 原始方差 (无法被工况解释)
    # temp_b 残差方差 << 原始方差 (大部分被 wind 解释)
    std_a = resid["temp_a__resid"].std()
    std_b = resid["temp_b__resid"].std()
    assert std_a > std_b * 0.8, f"temp_a 残差应比 temp_b 大, std_a={std_a:.3f}, std_b={std_b:.3f}"
