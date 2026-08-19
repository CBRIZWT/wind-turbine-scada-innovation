# -*- coding: utf-8 -*-
"""事件级评测器测试 (CARE 风格: event-recall / lead-time / FAR/天)。合成数据, 零 IO。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import 事件级评测 as ev  # noqa: E402

NS = 10 ** 9
STEP = 600 * NS  # 10min


def _ts(n, start=0):
    return (np.arange(n) * STEP + start).astype(np.int64)


def test_alarm_segments_merge_within_gap_and_isolate_turbines():
    ts = _ts(6)
    turb = np.array(["A", "A", "A", "A", "B", "B"])
    pred = np.array([1, 1, 0, 1, 1, 0])
    segs = ev.alarm_segments(pred, turb, ts, max_gap_steps=1)
    # A: [0,1] 与 [3] 间隔 1 步(idx2) ≤ max_gap → 合并为一段; B: [4] 独立一段
    assert len(segs) == 2
    a = [s for s in segs if s[0] == "A"][0]
    assert a[1] == ts[0] and a[2] == ts[3]
    segs0 = ev.alarm_segments(pred, turb, ts, max_gap_steps=0)
    assert len([s for s in segs0 if s[0] == "A"]) == 2  # 不允许间隔 → A 拆两段


def test_event_metrics_detection_lead_and_far():
    n = 60
    ts = _ts(n)
    turb = np.array(["A"] * n)
    y = np.zeros(n, dtype=int)
    # 事件: 起点 idx=30, 结束 idx=32 (事件期 -1), 早警窗 H=12 → idx 18..29 = 1
    y[18:30] = 1
    y[30:33] = -1
    episodes = pd.DataFrame({
        "_turbine": ["A"],
        "Timestamp start": [pd.Timestamp(ts[30], tz="UTC")],
        "Timestamp end": [pd.Timestamp(ts[32], tz="UTC")],
        "tier": ["tier1"],
    })
    pred = np.zeros(n, dtype=int)
    pred[20] = 1          # 早警窗内报警 → 检出, lead = (30-20)*10min = 100min
    pred[5] = 1           # 健康段误报 → 1 个假报警段
    m = ev.event_level_metrics(pred, y, ts, turb, episodes, lead_steps=12)
    assert m["n_events"] == 1
    assert m["event_recall"] == 1.0
    assert abs(m["lead_minutes_median"] - 100.0) < 1e-9
    assert m["n_false_segments"] == 1
    healthy_days = (int((y == 0).sum()) * 10) / (60 * 24)
    assert abs(m["far_per_day"] - 1.0 / healthy_days) < 1e-9
    # 主口径 event_f1 (2026-07-18 回退): 点级 precision 1/2 × 事件召回 1 → 2/3
    assert abs(m["alarm_precision"] - 0.5) < 1e-9
    assert abs(m["event_f1"] - (2 * 0.5 * 1.0 / 1.5)) < 1e-9
    assert abs(m["composite_f1"] - m["event_f1"]) < 1e-9   # 别名一致
    # 附录 pa_event_f1 (一点命中→全段命中): 窗 idx18..29 共 12 点全判命中,
    #   FP=idx5 一点 → P=12/13, R=12/12 → F1=24/25 (虚高示例, 不进主表)
    assert abs(m["pa_event_f1"] - 24 / 25) < 1e-9


def test_event_metrics_alarm_on_other_turbine_does_not_detect():
    n = 40
    ts = np.concatenate([_ts(n // 2), _ts(n // 2)])
    turb = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    y = np.zeros(n, dtype=int)
    y[10:15] = 1  # A 的早警窗
    episodes = pd.DataFrame({
        "_turbine": ["A"],
        "Timestamp start": [pd.Timestamp(ts[15], tz="UTC")],
        "Timestamp end": [pd.Timestamp(ts[16], tz="UTC")],
        "tier": ["tier2"],
    })
    pred = np.zeros(n, dtype=int)
    pred[n // 2 + 12] = 1   # B 机组同时刻报警 → 不算检出 A 的事件
    m = ev.event_level_metrics(pred, y, ts, turb, episodes, lead_steps=5)
    assert m["event_recall"] == 0.0


def test_select_threshold_event_maximizes_val_event_f1():
    n = 50
    ts = _ts(n)
    turb = np.array(["A"] * n)
    y = np.zeros(n, dtype=int)
    y[20:25] = 1
    y[25:27] = -1
    episodes = pd.DataFrame({
        "_turbine": ["A"],
        "Timestamp start": [pd.Timestamp(ts[25], tz="UTC")],
        "Timestamp end": [pd.Timestamp(ts[26], tz="UTC")],
        "tier": ["tier1"],
    })
    scores = np.zeros(n)
    scores[22] = 5.0   # 早警窗内高分
    scores[8] = 3.0    # 健康段中分 (低阈值会引入误报)
    thr, best = ev.select_threshold_event(scores, y, ts, turb, episodes, lead_steps=5)
    assert 3.0 < thr <= 5.0     # 最优阈值应卡在 3 与 5 之间 (只报事件, 不报误报)
    assert best == 1.0          # event_f1 = 1 (1段报警全命中, 事件全检出)
