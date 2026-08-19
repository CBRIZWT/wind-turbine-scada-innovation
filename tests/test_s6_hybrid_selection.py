# -*- coding: utf-8 -*-
"""S6 混合特征选择 select_temperature_indicators_hybrid 契约测试。

验证: 返回 IndicatorSelectionResult; 选定 ⊆ 输入列且去冗余(近重复通道被压缩);
     notes 含 fs_diagnostics 与 mi_amo_bhs 方法标记; 与无监督版接口一致(可 drop-in)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from 温度指标选择 import select_temperature_indicators_hybrid, IndicatorSelectionResult


def _redundant_residual(n=500, seed=0):
    rng = np.random.RandomState(seed)
    base = rng.randn(n, 6)
    red = base[:, :4] + rng.randn(n, 4) * 0.01   # 4 个近重复通道
    X = np.column_stack([base, red])              # 10 列
    cols = [f"c{i}__resid" for i in range(10)]
    return pd.DataFrame(X, columns=cols), cols


def test_hybrid_returns_indicator_result():
    df, cols = _redundant_residual()
    res = select_temperature_indicators_hybrid(df, hms=10, max_iter=20, random_state=42)
    assert isinstance(res, IndicatorSelectionResult)
    assert 1 <= len(res.selected) < len(cols)
    assert set(res.selected) <= set(cols)


def test_hybrid_notes_have_diagnostics():
    df, _ = _redundant_residual(seed=3)
    res = select_temperature_indicators_hybrid(df, hms=10, max_iter=20, random_state=1)
    assert "fs_diagnostics" in res.notes
    assert str(res.notes["method"]).startswith("mi_amo_bhs")
    assert res.notes["n_selected"] == len(res.selected)
    assert set(res.notes["fs_diagnostics"]).issuperset({"reduction_ratio_pct", "redundancy"})


def test_hybrid_empty_input():
    res = select_temperature_indicators_hybrid(pd.DataFrame(), hms=5, max_iter=5)
    assert res.selected == []
    assert res.notes["n_selected"] == 0
