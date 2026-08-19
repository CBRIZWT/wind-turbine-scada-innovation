# -*- coding: utf-8 -*-
"""real_fault_wl 白名单标签政策测试 (2026-07-11 真实故障事件级方案 §1.6/§3.1)。

覆盖: 消息级白名单/黑名单、tier 分类、同机组 72h episode 合并、事件后 post-ignore 窗。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SCADA数据集"))

import 数据预处理 as prep  # noqa: E402


def _idx(n=60, start="2021-01-01"):
    return pd.DatetimeIndex(pd.date_range(start, periods=n, freq="10min", tz="UTC"))


def _ev(messages, starts, ends=None, turbines=None):
    df = pd.DataFrame({
        "Timestamp start": [pd.Timestamp(s, tz="UTC") for s in starts],
        "Message": messages,
    })
    if ends is not None:
        df["Timestamp end"] = [pd.Timestamp(e, tz="UTC") for e in ends]
    if turbines is not None:
        df["_turbine"] = turbines
    return df


def test_wl_mask_whitelists_tiers_and_blacklists_pollution():
    msgs = [
        "Gearbox warm-up stage",                        # 黑: 正常冷启动 (kel 297行中占259)
        "PT100 converter inlet temperature defect",      # 黑: 传感器缺陷
        "Error lubrication pump pitch",                  # 黑: 变桨润滑泵(非传动链)
        "High temperature nacelle",                      # 黑: 1min级可疑事件, 方案排除
        "Anemometer defect",                             # 黑: 无关
        "High temp. gear bearing 1",                     # Tier-1
        "High temp. gear bearing 2",                     # Tier-1
        "High temp. gen. bearing 1",                     # Tier-1
        "Max. temp. gen. bearing 1",                     # Tier-1
        "Low gearbox oil pressure",                      # Tier-2
        "Missing gear oil (high rpm)",                   # Tier-2
        "Missing gear oil (low rpm)",                    # Tier-2
        "Overload gear oil pump",                        # Tier-2
        "Reduced power gearbox",                         # Tier-2
    ]
    ev = _ev(msgs, ["2021-01-01 00:10"] * len(msgs))
    mask = prep.real_fault_event_mask(ev, policy="real_fault_wl")
    assert mask.tolist() == [False] * 5 + [True] * 9


def test_classify_real_fault_tier():
    assert prep.classify_real_fault_tier("High temp. gear bearing 1") == "tier1"
    assert prep.classify_real_fault_tier("Max. temp. gen. bearing 1") == "tier1"
    assert prep.classify_real_fault_tier("Missing gear oil (high rpm)") == "tier2"
    assert prep.classify_real_fault_tier("Low gearbox oil pressure") == "tier2"
    assert prep.classify_real_fault_tier("Gearbox warm-up stage") == ""
    assert prep.classify_real_fault_tier("System OK") == ""


def test_merge_real_fault_episodes_same_turbine_72h():
    ev = _ev(
        ["High temp. gear bearing 1"] * 4,
        ["2021-01-01 00:00", "2021-01-01 01:00",   # T6 相邻 1h → 合并
         "2021-01-20 00:00",                        # T6 相隔 >72h → 独立
         "2021-01-01 00:30"],                       # T5 → 独立 (跨机组不合并)
        ends=["2021-01-01 00:20", "2021-01-01 01:20",
              "2021-01-20 00:20", "2021-01-01 00:50"],
        turbines=["T6", "T6", "T6", "T5"],
    )
    merged = prep.merge_real_fault_episodes(ev, merge_hours=72.0)
    assert len(merged) == 3
    t6_first = merged[(merged["_turbine"] == "T6")].sort_values("Timestamp start").iloc[0]
    assert t6_first["Timestamp start"] == pd.Timestamp("2021-01-01 00:00", tz="UTC")
    assert t6_first["Timestamp end"] == pd.Timestamp("2021-01-01 01:20", tz="UTC")


def test_wl_labels_merge_and_post_ignore():
    # 事件: 两行 30min 相邻 (02:00-02:20, 02:30-02:40) → 合并为 02:00-02:40 一个 episode
    ev = _ev(
        ["High temp. gear bearing 1", "High temp. gear bearing 1"],
        ["2021-01-01 02:00", "2021-01-01 02:30"],
        ends=["2021-01-01 02:20", "2021-01-01 02:40"],
    )
    y, intervals = prep.make_real_fault_earlywarning_labels(
        ev, _idx(60), policy="real_fault_wl", lead_steps=3,
        merge_hours=72.0, post_ignore_steps=4, return_intervals=True,
    )
    # 合并后单事件: grid idx 12 (02:00) .. 16 (02:40, side=right→17)
    assert len(intervals) == 1
    s, e = intervals[0]
    assert y[s - 3:s].tolist() == [1, 1, 1]          # pre-fault 早警窗
    assert set(y[s:e].tolist()) == {-1}              # 事件期 ignore
    assert y[e:e + 4].tolist() == [-1, -1, -1, -1]   # 事件后恢复窗 ignore
    assert y[e + 4] == 0                             # 之后恢复正常
    assert int((y == 1).sum()) == 3                  # 只有一个事件的早警窗


def test_wl_post_ignore_does_not_overwrite_positives():
    # 两个独立事件 (相隔 >merge窗), 后事件的早警窗与前事件 post-ignore 重叠 → 正例优先
    ev = _ev(
        ["High temp. gear bearing 1", "Low gearbox oil pressure"],
        ["2021-01-01 02:00", "2021-01-01 04:00"],
        ends=["2021-01-01 02:20", "2021-01-01 04:10"],
    )
    y = prep.make_real_fault_earlywarning_labels(
        ev, _idx(60), policy="real_fault_wl", lead_steps=6,
        merge_hours=1.0, post_ignore_steps=12,
    )
    # 事件2 起点 04:00=idx24, 早警窗 idx18..23; 事件1 结束 02:20→idx15(right),
    # post-ignore 15..26 与早警窗重叠 → 18..23 仍应为 1
    assert y[18:24].tolist() == [1] * 6


def test_wl_protocol_constants():
    assert prep.REAL_FAULT_WL_MERGE_HOURS == 72.0
    assert prep.REAL_FAULT_WL_POST_IGNORE_STEPS == 144  # 24h × 6步/h


def test_wl_labels_pre_grace_band():
    # 事件 02:00-02:20; lead=3, grace=4 → 事件前 [s-3,s)=1, 再前 [s-7,s-3)=-1(缓冲), 更早=0
    ev = _ev(["High temp. gear bearing 1"], ["2021-01-01 02:00"], ends=["2021-01-01 02:20"])
    y, intervals = prep.make_real_fault_earlywarning_labels(
        ev, _idx(60), policy="real_fault_wl", lead_steps=3,
        pre_grace_steps=4, return_intervals=True,
    )
    s, _ = intervals[0]
    assert y[s - 3:s].tolist() == [1, 1, 1]              # 早警窗不受影响
    assert y[s - 7:s - 3].tolist() == [-1] * 4           # 缓冲带 ignore (CARE padding)
    assert y[s - 8] == 0                                 # 缓冲带之外恢复正常


def test_export_row_sidecars_roundtrip():
    idx = pd.date_range("2021-01-01", periods=3, freq="10min", tz="UTC")
    ts, tb = prep.export_row_sidecars(idx, ["T1", "T1", "T2"])
    assert ts.dtype == np.int64
    assert pd.Timestamp(ts[0], tz="UTC") == idx[0]      # epoch-ns 往返
    assert ts[1] - ts[0] == 600 * 10 ** 9               # 10min 间隔
    assert tb.tolist() == ["T1", "T1", "T2"]


def test_old_policies_unchanged_by_wl_addition():
    ev = _ev(["Gearbox warm-up stage"], ["2021-01-01 00:10"])
    assert prep.real_fault_event_mask(ev, policy="real_fault_temp").tolist() == [True]  # 旧行为保留
    assert prep.real_fault_event_mask(ev, policy="real_fault_wl").tolist() == [False]
