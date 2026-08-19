# -*- coding: utf-8 -*-
"""
tests/test_data_preprocessing.py — 数据预处理流水线的单元测试 (2026-06 修复后)

覆盖:
    - hampel_filter 去尖峰 (修正了旧测试误用的 _hampel/win 名称)
    - 常量信号不误杀
    - loaders 工厂 get_loader
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SCADA数据集"))

import numpy as np
import pytest


def test_hampel_basic():
    """hampel_filter — 在白噪声上捕获远超 MAD 的离群点。"""
    from 数据预处理 import hampel_filter
    rng = np.random.default_rng(0)
    s = rng.normal(0, 0.1, 100)
    s[50] = 10.0  # 100σ 离群
    result = hampel_filter(s, window=11, k=2.0)
    assert np.isnan(result).sum() >= 1, f"Hampel 未检测到离群: NaN={np.isnan(result).sum()}"


def test_hampel_constant_signal():
    """hampel_filter 对常量信号不应误杀 (MAD=0 → sigma=0 → 不置 NaN)。"""
    from 数据预处理 import hampel_filter
    s = np.full(100, 5.0)
    result = hampel_filter(s, window=13, k=3.0)
    assert np.isnan(result).sum() == 0


def test_get_loader():
    """Loader 工厂返回正确类型。"""
    from loaders import get_loader
    loader = get_loader("kelmarsh")
    assert loader.FARM_NAME == "kelmarsh"


def test_get_loader_invalid():
    """未知 farm 名抛错。"""
    from loaders import get_loader
    with pytest.raises(KeyError):
        get_loader("unknown_farm")


def test_hampel_causal_equals_prefix_evaluation():
    """因果性契约: causal=True 时, 全序列在 i 处输出 == 仅用前缀 series[:i+1] 的末位输出。"""
    from 数据预处理 import hampel_filter
    rng = np.random.default_rng(1)
    s = rng.normal(0, 1, 50)
    s[[10, 33]] = [8.0, -7.0]
    full = hampel_filter(s, window=9, k=3.0, causal=True)
    for i in (15, 25, 40):
        prefix = hampel_filter(s[: i + 1], window=9, k=3.0, causal=True)
        assert np.nan_to_num(full[i]) == np.nan_to_num(prefix[i]), f"causal hampel 在 index {i} 依赖未来"


def test_interpolation_forward_fill_only():
    """causal=True: 只前向填充, 不向后插值; 无前值的前导缺失保留 NaN 并入 mask。"""
    from 数据预处理 import gap_limited_interpolation
    import pandas as pd
    idx = pd.date_range("2021-01-01", periods=6, freq="10min")
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0, np.nan, 6.0]}, index=idx)
    out, mask = gap_limited_interpolation(df, limit=2, causal=True)
    assert out["a"].iloc[1] == 1.0 and out["a"].iloc[2] == 1.0   # ffill 非 2.0/3.0
    assert out["a"].iloc[4] == 4.0
    df2 = pd.DataFrame({"a": [np.nan, np.nan, 3.0]},
                       index=pd.date_range("2021-01-01", periods=3, freq="10min"))
    out2, mask2 = gap_limited_interpolation(df2, limit=2, causal=True)
    assert np.isnan(out2["a"].iloc[0]) and mask2[0] == 1         # 前导缺失保留+标记
