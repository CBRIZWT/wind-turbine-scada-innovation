# -*- coding: utf-8 -*-
"""
tests/test_make_labels.py — 标签构建函数的正确性测试 (2026-06 修复后)

覆盖修复:
    - A1: 常规刹车 / 手动停机 / 电网事件不再被当作"温度异常"正例
    - A3: 逐机组隔离 (turbine_col), T01 故障不污染 T02
    - A5: 缺真实故障字段时返回全 0 (而非把所有事件当正例)
    - 结构化字段优先: IEC Forced outage / HOT Stopping / 温度·传动链关键词
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SCADA数据集"))

import numpy as np
import pandas as pd
from 数据预处理 import make_labels


def _idx(n=100, start="2021-01-01"):
    return pd.DatetimeIndex(pd.date_range(start, periods=n, freq="10min", tz="UTC"))


def test_empty_events_all_zero():
    """空事件表 → 全 0 标签。"""
    y = make_labels(pd.DataFrame(), _idx(100))
    assert y.sum() == 0 and len(y) == 100


def test_iec_forced_outage_positive():
    """IEC category = Forced Outage → 正例 (结构化字段, 无需关键词)。"""
    ev = pd.DataFrame({"Timestamp start": [pd.Timestamp("2021-01-01 02:00", tz="UTC")],
                       "IEC category": ["Forced Outage"]})
    assert make_labels(ev, _idx(100), half_window_hours=2.0).sum() > 0


def test_temperature_keyword_positive():
    """温度/传动链故障关键词 → 正例。"""
    ev = pd.DataFrame({"Timestamp start": [pd.Timestamp("2021-01-01 02:00", tz="UTC")],
                       "Message": ["High temperature gear bearing 1"]})
    assert make_labels(ev, _idx(100), half_window_hours=2.0).sum() > 0


def test_hot_stopping_positive():
    """HOT Stopping == 1 → 正例。"""
    ev = pd.DataFrame({"Timestamp start": [pd.Timestamp("2021-01-01 02:00", tz="UTC")],
                       "Stopping": [1]})
    assert make_labels(ev, _idx(100), half_window_hours=2.0).sum() > 0


def test_brake_program_not_positive():
    """A1: 常规刹车 (Brake program) 不应标为温度异常正例。"""
    ev = pd.DataFrame({"Timestamp start": [pd.Timestamp("2021-01-01 02:00", tz="UTC")],
                       "Message": ["Brake program 50"]})
    assert make_labels(ev, _idx(100), half_window_hours=2.0).sum() == 0


def test_manual_and_grid_not_positive():
    """A1: 手动停机 / 电网事件与温度无关, 不应标为正例。"""
    ev = pd.DataFrame({"Timestamp start": [pd.Timestamp("2021-01-01 02:00", tz="UTC"),
                                           pd.Timestamp("2021-01-01 05:00", tz="UTC")],
                       "Message": ["Manual stop - on site", "Grid loss"]})
    assert make_labels(ev, _idx(100), half_window_hours=2.0).sum() == 0


def test_running_status_not_positive():
    """A5: 仅有正常 Status (无任何故障字段) → 全 0, 不得把所有事件当正例。"""
    ev = pd.DataFrame({"Timestamp start": [pd.Timestamp("2021-01-01 02:00", tz="UTC")],
                       "Status": ["Running"]})
    assert make_labels(ev, _idx(100), half_window_hours=2.0).sum() == 0


def test_window_width():
    """±2h 半窗 = ±12 步, 命中约 25 个点。"""
    ev = pd.DataFrame({"Timestamp start": [pd.Timestamp("2021-01-01 05:00", tz="UTC")],
                       "Message": ["bearing high temperature"]})
    y = make_labels(ev, _idx(100), half_window_hours=2.0)
    assert 20 <= int(y.sum()) <= 26


def test_per_turbine_isolation():
    """A3: 池化(时间戳重复)下, T01 的故障只标 T01 行, 不污染 T02。"""
    base = pd.date_range("2021-01-01", periods=100, freq="10min", tz="UTC")
    idx2 = pd.DatetimeIndex(base).append(pd.DatetimeIndex(base))   # 200 行, 时间戳重复(模拟池化)
    turb = np.array(["T01"] * 100 + ["T02"] * 100)
    ev = pd.DataFrame({"Timestamp start": [pd.Timestamp("2021-01-01 03:00", tz="UTC")],
                       "Message": ["bearing high temperature"], "_turbine": ["T01"]})
    y = make_labels(ev, idx2, half_window_hours=2.0, turbine_col=turb)
    assert y[:100].sum() > 0, "T01 应有正例"
    assert y[100:].sum() == 0, "T02 不应被 T01 故障污染"
