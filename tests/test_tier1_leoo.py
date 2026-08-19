# -*- coding: utf-8 -*-
"""Tier-1 真过温 LOEO 评测协议的契约测试。

背景 (2026-08-08 根因): chronological_v2 纯时间分割下 test 段 tier1_n == 0 ——
全部 12 个 Tier-1 真过温事件都落在 2016-2021 train 窗, 测试段评的 100% 是
Tier-2 油压/油位保护跳闸。设计稿 2026-07-08 §1.6/§3.2 已裁决"纯时间分割被数据否决,
LOEO/池化为唯一可行分割", 但从未落地。本模块实现该裁决并加防呆断言。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tier1_leoo import (  # noqa: E402
    TierCoverageError,
    assert_tier1_coverage,
    build_episode_labels,
    extract_tier1_episodes,
)

TS = pd.Timestamp("2020-06-01 00:00:00", tz="UTC")
STEP = pd.Timedelta(minutes=10)


def _ev(turbine, start, message, split="train"):
    return {"farm": "kelmarsh", "turbine": str(turbine), "start": str(start),
            "end": str(pd.Timestamp(start) + pd.Timedelta(minutes=20)),
            "split": split, "source": "status", "category": "Forced outage",
            "message": message, "policy": "real_fault_wl"}


class TestExtractTier1Episodes:
    def test_only_tier1_messages_survive(self):
        """Tier-2 油压/油位事件必须被排除 —— 这是根因所在。"""
        df = pd.DataFrame([
            _ev(1, TS, "High temp. gear bearing 1"),
            _ev(1, TS + pd.Timedelta(days=10), "Missing gear oil (high rpm)"),
            _ev(1, TS + pd.Timedelta(days=20), "Low gearbox oil pressure"),
        ])
        eps = extract_tier1_episodes(df)
        assert len(eps) == 1
        assert eps[0]["turbine"] == "1"

    def test_merges_within_window_same_turbine(self):
        """同机组 72h 内的多条告警合并为一个 episode。"""
        df = pd.DataFrame([
            _ev(2, TS, "High temp. gear bearing 1"),
            _ev(2, TS + pd.Timedelta(hours=5), "High temp. gear bearing 1"),
            _ev(2, TS + pd.Timedelta(hours=71), "High temp. gear bearing 2"),
        ])
        assert len(extract_tier1_episodes(df, merge_hours=72)) == 1

    def test_separates_beyond_window(self):
        df = pd.DataFrame([
            _ev(2, TS, "High temp. gear bearing 1"),
            _ev(2, TS + pd.Timedelta(hours=100), "High temp. gear bearing 1"),
        ])
        assert len(extract_tier1_episodes(df, merge_hours=72)) == 2

    def test_never_merges_across_turbines(self):
        """跨机组绝不合并 —— 不同机器的故障是独立事件。"""
        df = pd.DataFrame([
            _ev(1, TS, "High temp. gear bearing 1"),
            _ev(2, TS + pd.Timedelta(hours=1), "High temp. gear bearing 1"),
        ])
        eps = extract_tier1_episodes(df, merge_hours=72)
        assert len(eps) == 2
        assert {e["turbine"] for e in eps} == {"1", "2"}

    def test_blacklist_wins_over_tier1(self):
        """warm-up / PT100 等污染词即使含 high temp 也必须排除。"""
        df = pd.DataFrame([_ev(1, TS, "Gearbox warm-up high temp. gear bearing")])
        assert extract_tier1_episodes(df) == []


class TestBuildEpisodeLabels:
    def _axis(self, n=600, turbine="1"):
        ts = np.array([(TS + i * STEP).value for i in range(n)], dtype=np.int64)
        tb = np.array([turbine] * n)
        return ts, tb

    def test_lead_window_is_positive_and_episode_is_ignored(self):
        ts, tb = self._axis()
        start = TS + 300 * STEP
        eps = [{"turbine": "1", "start": start, "end": start + 5 * STEP}]
        y = build_episode_labels(ts, tb, eps, lead_steps=72, post_ignore_steps=144)
        assert (y[300 - 72:300] == 1).all(), "起点前 72 步应全为正例"
        assert (y[300:306] == -1).all(), "事件进行期应为 ignore"
        assert (y[:300 - 72] == 0).all(), "更早的健康段应为 0"

    def test_post_event_window_is_ignored(self):
        ts, tb = self._axis()
        start = TS + 200 * STEP
        eps = [{"turbine": "1", "start": start, "end": start + 2 * STEP}]
        y = build_episode_labels(ts, tb, eps, lead_steps=72, post_ignore_steps=144)
        assert (y[203:203 + 144] == -1).all(), "事后 24h 维修/重启段应 ignore"

    def test_labels_do_not_leak_across_turbines(self):
        """T1 的事件不得在 T2 的行上打标 —— 逐机组隔离。"""
        ts1, tb1 = self._axis(400, "1")
        ts2, tb2 = self._axis(400, "2")
        ts = np.concatenate([ts1, ts2]); tb = np.concatenate([tb1, tb2])
        start = TS + 300 * STEP
        eps = [{"turbine": "1", "start": start, "end": start + 2 * STEP}]
        y = build_episode_labels(ts, tb, eps, lead_steps=72, post_ignore_steps=144)
        assert (y[400:] == 0).all(), "T2 的行不应被 T1 的事件污染"
        assert (y[:400] == 1).sum() == 72

    def test_no_episodes_yields_all_healthy(self):
        ts, tb = self._axis()
        y = build_episode_labels(ts, tb, [], lead_steps=72, post_ignore_steps=144)
        assert (y == 0).all()


class TestTierCoverageGuard:
    """防呆闸: 主表评测段若 tier1_n==0, 必须 fail-fast 而不是静默出榜。"""

    def test_raises_when_no_tier1_events(self):
        with pytest.raises(TierCoverageError, match="tier1"):
            assert_tier1_coverage(n_tier1=0, n_tier2=10, context="penmanshiel/test")

    def test_message_names_the_physical_mismatch(self):
        with pytest.raises(TierCoverageError) as ei:
            assert_tier1_coverage(n_tier1=0, n_tier2=10, context="penmanshiel/test")
        assert "penmanshiel/test" in str(ei.value)

    def test_passes_when_tier1_present(self):
        assert_tier1_coverage(n_tier1=3, n_tier2=10, context="loeo/fold-0") is None

    def test_raises_when_no_events_at_all(self):
        with pytest.raises(TierCoverageError):
            assert_tier1_coverage(n_tier1=0, n_tier2=0, context="hill_of_towie/test")
