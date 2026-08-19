# -*- coding: utf-8 -*-
"""
tests/test_pi_jobs_memory.py — permutation_importance 的 n_jobs 内存安全策略

回归 (2026-06-04): HOT farm (train≈6.57M 行, X≈2.21GB) 预处理 S6 在
permutation_importance n_jobs=3 下被 OOM 杀死 —— loky 每 worker 复制大矩阵,
叠加 commit 上限 (64GB RAM + 仅 10GB 手动封顶页面文件 = 73GB) 与并发任务,
提交内存越界 → numpy ArrayMemoryError。旧自适应公式按【物理】可用内存的 0.5
预算 ÷ 2.5×X, 给出 n_jobs=3, 未考虑 commit 上限/并发任务。

修复策略 (用户选 "路径1: 限并行, 不改方法学"):
    _safe_pi_jobs 对大矩阵回退串行 (n_jobs=1, 不跨进程复制大矩阵),
    对小数据 + 充足内存仍允许并行; 永不返回 <1。
    permutation_importance 含 random_state=0 → 结果与 n_jobs 无关 (位级一致),
    通道选择口径不变 (科研标准: 不改方法学)。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from 温度指标选择 import _safe_pi_jobs, _rf_n_jobs


def test_rf_n_jobs_parallel_only_when_permutation_serial():
    """防 N1 嵌套死锁: 置换串行(pi_jobs=1) → RF 并行(-1, 线程提速); 置换并行(>1) → RF 串行(1)。"""
    assert _rf_n_jobs(1) == -1     # HOT: 置换串行 → RF 线程并行, S6 大幅提速
    assert _rf_n_jobs(2) == 1      # 小农场: 置换并行 → RF 必须串行 (否则 loky 嵌套死锁)
    assert _rf_n_jobs(8) == 1


def test_serial_on_hot_scale_matrix():
    """HOT 规模 (X≈2.21GB, 6.57M 行) → 串行 n_jobs=1, 不跨 worker 复制大矩阵。"""
    n = _safe_pi_jobs(x_nbytes=int(2.21e9), n_rows=6_573_678,
                      avail_bytes=int(40.2e9), cpu_count=8)
    assert n == 1


def test_parallel_on_small_matrix_with_ample_memory():
    """小矩阵 + 充足内存 → 允许并行 (>1)。"""
    n = _safe_pi_jobs(x_nbytes=100 * 1024 ** 2, n_rows=100_000,
                      avail_bytes=int(50e9), cpu_count=8)
    assert n > 1


def test_never_below_one_when_memory_starved():
    """内存极度紧张也不返回 0/负数, 至少串行 1。"""
    n = _safe_pi_jobs(x_nbytes=int(5e9), n_rows=10_000_000,
                      avail_bytes=int(1e9), cpu_count=8)
    assert n == 1


def test_jobs_never_exceed_cpu_count():
    """即便内存充裕也不超过 CPU 数。"""
    n = _safe_pi_jobs(x_nbytes=1024 ** 2, n_rows=1000,
                      avail_bytes=int(256e9), cpu_count=4)
    assert 1 <= n <= 4
