# -*- coding: utf-8 -*-
"""
tests/test_no_test_leakage.py — 泄漏守卫: 验证默认行为是因果/train-only (2026-06-01)

覆盖:
    - hampel_filter 默认因果 (未来尖峰不影响过去输出)
    - gap_limited_interpolation 默认因果 (ffill, 非双向时间插值)
    - select_temperature_indicators val 不影响 selected (train-only)
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SCADA数据集"))
import numpy as np
import pandas as pd


def test_hampel_default_is_causal():
    """默认(不传 causal)即因果: 未来尖峰不影响过去输出。"""
    from 数据预处理 import hampel_filter
    rng = np.random.default_rng(3)
    s = rng.normal(0, 0.1, 60)
    a = s.copy(); b = s.copy(); b[50] = 50.0
    ra = hampel_filter(a, window=9, k=3.0)
    rb = hampel_filter(b, window=9, k=3.0)
    assert np.array_equal(np.nan_to_num(ra[:50]), np.nan_to_num(rb[:50]))


def test_interpolation_default_is_causal():
    from 数据预处理 import gap_limited_interpolation
    idx = pd.date_range("2021-01-01", periods=5, freq="10min")
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]}, index=idx)
    out, _ = gap_limited_interpolation(df, limit=2)
    assert out["a"].iloc[1] == 1.0    # ffill, 不是 2.0


def test_selection_ignores_val():
    from 温度指标选择 import select_temperature_indicators
    rng = np.random.default_rng(0)
    n = 2000
    y = np.zeros(n, dtype=int); y[-300:] = 1
    rt = pd.DataFrame({"temp_a__resid": rng.normal(0, 0.3, n) + y * 3.0,
                       "temp_b__resid": rng.normal(0, 0.5, n)})
    rv = pd.DataFrame({"temp_a__resid": rng.normal(0, 0.3, n),
                       "temp_b__resid": rng.normal(0, 0.5, n)})
    yv = y.copy()
    assert (select_temperature_indicators(rt, y).selected
            == select_temperature_indicators(rt, y, rv, yv).selected)
