# -*- coding: utf-8 -*-
"""
tests/test_lead_time_steps.py — lead_time_steps 逐拍切片修复 (2026-06-05 OOM)

回归: 旧"LAG 向量化(2026-06-04)"对每通道 materialize (max_lag+1, n_aligned) 大矩阵,
HOT (433, 6.57M)=21.2 GiB → numpy ArrayMemoryError (S6 报告表 _row 崩)。改回逐拍切片
(r[τ:τ+n_aligned] 视图, 不复制), 逐位等价、内存仅 1 切片。本测试锁定:
  - 能正确检出已知的领先步数;
  - 在较大 n 上正常返回 (不 materialize 大矩阵 / 不卡死)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from 温度指标选择 import lead_time_steps


def test_detects_known_lead():
    """残差信号埋在偏移 L 处 → corr 在 τ=L 最大 → 返回 L。"""
    rng = np.random.default_rng(0)
    L, n_aligned, max_lag = 40, 2000, 100
    total = n_aligned + max_lag
    s = (rng.random(n_aligned) > 0.7).astype(float)        # 二值信号
    r = rng.normal(0, 0.01, total)                          # 噪声基线
    r[L:L + n_aligned] = s + rng.normal(0, 0.01, n_aligned)  # 在偏移 L 处植入 s
    y = np.zeros(total)
    y[:n_aligned] = s
    assert lead_time_steps(r, y, max_lag_steps=max_lag) == L


def test_runs_on_large_n_without_big_matrix():
    """大 n: 逐拍切片应快速返回合法 τ (旧版会建 (433, n) 大矩阵)。"""
    rng = np.random.default_rng(2)
    n = 300_000
    r = rng.normal(size=n)
    y = (rng.random(n) > 0.95).astype(float)
    got = lead_time_steps(r, y)                              # 默认 max_lag_steps=432
    assert isinstance(got, int) and 0 <= got <= 432
