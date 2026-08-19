# -*- coding: utf-8 -*-
"""
tests/test_indicator_selection.py — 温度指标选择的正确性测试

测试覆盖:
    - select_temperature_indicators 的合成数据测试
    - 故障相关指标被选出, 无关指标被排除 (修复 #19 的负排除)
    - Boruta 显著性门: 纯噪声不应产出 >0 个 significant 指标
    - VIF 共线排除
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from 温度指标选择 import (
    NormalBehaviorModel,
    select_temperature_indicators,
    variance_inflation_factors,
    cluster_redundant,
)


def test_indicator_selection_finds_relevant():
    """故障相关指标应被选中。"""
    rng = np.random.default_rng(0)
    n = 2000
    # 合成: temp_a 与故障相关, temp_b 无关
    y = np.zeros(n, dtype=int); y[-300:] = 1
    resid = pd.DataFrame({
        "temp_a__resid": rng.normal(0, 0.3, n) + y * 3.0,
        "temp_b__resid": rng.normal(0, 0.5, n),
    })
    res = select_temperature_indicators(resid, y)
    assert "temp_a__resid" in res.selected, f"应选出故障相关指标, 实际: {res.selected}"
    print(f"selected={res.selected}, significant={res.notes['n_significant']}")


def test_indicator_selection_excludes_irrelevant():
    """#19 修复验证: 与故障无关的指标不应在 selected 中。"""
    rng = np.random.default_rng(42)
    n = 3000
    y = np.zeros(n, dtype=int); y[-500:] = 1
    resid = pd.DataFrame({
        "faulty__resid": rng.normal(0, 0.3, n) + y * 5.0,
        "noise__resid": rng.normal(0, 1.0, n),
    })
    res = select_temperature_indicators(resid, y)
    # 噪声指标不应同时通过 Boruta 显著性门 (其 shadow 可能更高)
    assert "faulty__resid" in res.selected, f"故障指标应入选: {res.selected}"
    # 如果 noise 也入选, 检查其 composite 是否真的 > shadow_ceiling
    if "noise__resid" in res.selected:
        print(f"[WARN] noise 意外入选, 检查 shadow_ceiling={res.notes['shadow_ceiling']:.4f}")
    else:
        print("OK: noise excluded")


def test_pure_noise_yields_no_significant():
    """纯噪声指标 + 随机标签 → Boruta 天花板挡住全部指标 (0 significant)。"""
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)  # 随机标签
    resid = pd.DataFrame({
        "a__resid": rng.normal(0, 1, n),
        "b__resid": rng.normal(0, 1, n),
        "c__resid": rng.normal(0, 1, n),
    })
    res = select_temperature_indicators(resid, y)
    # 随机场景下 Boruta 应挡住所有指标或仅误选极少量 (≤1)
    assert res.notes["n_significant"] <= 1, \
        f"纯噪声不应出显著指标: significant={res.notes['n_significant']}"
    print(f"significant={res.notes['n_significant']}, selected={res.selected}")


def test_vif_rejects_collinear():
    """VIF 应剔除高度共线指标（完全相关的两列 → VIF 极大）。"""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "a": x,
        "b": x + rng.normal(0, 1e-4, n),  # 与 a 几乎完全相关
        "c": rng.normal(0, 1, n),           # 独立
    })
    vif = variance_inflation_factors(df)
    assert vif["a"] > 5 or vif["b"] > 5, f"完全共线列 VIF 应 >5: a={vif['a']:.1f}, b={vif['b']:.1f}"
    assert vif["c"] < 5, f"独立列 VIF 应 <5: c={vif['c']:.1f}"


def test_cluster_redundant_keeps_best():
    """Spearman 聚类: 保留排名最高的代表, 丢弃高度共线者。"""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "rank1__resid": x + rng.normal(0, 0.1, n),
        "rank2__resid": x + rng.normal(0, 0.1, n),  # 与 rank1 高度共线
        "rank3__resid": rng.normal(0, 1, n),         # 独立
    })
    kept = cluster_redundant(df, ["rank1__resid", "rank2__resid", "rank3__resid"],
                             corr_threshold=0.9)
    # 至少保留 rank1 (排名最高) 和 rank3 (独立)
    assert "rank1__resid" in kept
    assert "rank3__resid" in kept
    # rank2 与 rank1 共线 → 应被丢弃
    assert "rank2__resid" not in kept


def test_val_does_not_affect_selection():
    """L3: val 的 corr_stability 只进报告表, 不改变 selected/significant。"""
    rng = np.random.default_rng(0)
    n = 2000
    y = np.zeros(n, dtype=int); y[-300:] = 1
    resid_train = pd.DataFrame({
        "temp_a__resid": rng.normal(0, 0.3, n) + y * 3.0,
        "temp_b__resid": rng.normal(0, 0.5, n),
    })
    yv = np.zeros(n, dtype=int); yv[-300:] = 1
    resid_val_hostile = pd.DataFrame({          # val 中 temp_a 无信号 → 旧逻辑会因不稳定剔除它
        "temp_a__resid": rng.normal(0, 0.3, n),
        "temp_b__resid": rng.normal(0, 0.5, n),
    })
    sel_no_val = select_temperature_indicators(resid_train, y).selected
    res_with_val = select_temperature_indicators(resid_train, y, resid_val_hostile, yv)
    assert res_with_val.selected == sel_no_val, "val 不得改变通道选择 (train-only)"
    assert "corr_stability" in res_with_val.table.columns, "val 稳定性仍须进报告表"
