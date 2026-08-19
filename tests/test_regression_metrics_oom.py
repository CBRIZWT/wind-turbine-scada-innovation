# -*- coding: utf-8 -*-
"""
test_regression_metrics_oom.py

回归测试: 修复 regression_metrics 的大矩阵 OOM (2026-06-06 组会快扫 TriTrackNet 第三模型崩溃).

崩溃点: 实验工具.regression_metrics 对 (N, C*H) 残差一次性 materialize 多个 float64 临时数组
        ((1162937, 432) → 单个 3.74 GiB × 4~5 份) → numpy ArrayMemoryError。

修复要求 (科研口径不可变):
  - sample_scores (喂阈值/极性选择 → 最终 test 指标的科研核心) 必须与朴素实现【逐位一致】;
  - mse/mae/rmse 为标量约简, 仅累加顺序不同 (允许 ~1e-12 相对误差);
  - 分块边界 (N 不能整除 chunk) 不能改变结果;
  - float32 输入 (TriTrackNet 真实数据) 与 float64 输入都正确。
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from 实验工具 import regression_metrics


def _naive_regression_metrics(y_true, y_pred):
    """修复前的原始实现 (作为参考真值)。"""
    target = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    residual = target - pred
    sample_scores = np.mean(np.maximum(0, residual) ** 2, axis=1)
    mse = float(np.mean(residual ** 2))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(mse))
    return {"loss": mse, "mse": mse, "mae": mae, "rmse": rmse}, sample_scores


def _assert_equivalent(y, pred, **kwargs):
    reg_ref, scores_ref = _naive_regression_metrics(y, pred)
    reg, scores = regression_metrics(y, pred, **kwargs)
    # sample_scores 必须逐位一致 (科研核心)
    assert scores.shape == scores_ref.shape
    np.testing.assert_array_equal(scores, scores_ref)
    # 标量回归指标: 仅累加顺序不同, 数值等价
    for k in ("loss", "mse", "mae", "rmse"):
        assert np.isclose(reg[k], reg_ref[k], rtol=1e-12, atol=1e-12), (
            f"{k}: got {reg[k]!r} expected {reg_ref[k]!r}"
        )


def test_float64_equivalence():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, (200, 18)).astype(np.float64)
    pred = rng.normal(0, 1, (200, 18)).astype(np.float64)
    _assert_equivalent(y, pred)


def test_float32_input_equivalence():
    """TriTrackNet 真实数据是 float32 (series/predict 都 float32)。"""
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, (500, 432)).astype(np.float32)
    pred = rng.normal(0, 1, (500, 432)).astype(np.float32)
    _assert_equivalent(y, pred)


def test_chunk_boundary_not_divisible():
    """N 不能被 chunk 整除时, 分块边界不能改变结果 (逐行 mean 与整块一致)。"""
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, (97, 13)).astype(np.float32)
    pred = rng.normal(0, 1, (97, 13)).astype(np.float32)
    # 强制极小 chunk → 多次迭代, 含一个不满块
    _assert_equivalent(y, pred, chunk_rows=10)


def test_positive_residual_only_scoring():
    """仅正向偏离 (加热) 计分: 负残差应被 max(0,·) 归零 (#E4 口径不变)。"""
    y = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    pred = np.array([[0.0, 2.0, 3.0]], dtype=np.float32)  # residual = [+1, -1, -2]
    _, scores = regression_metrics(y, pred)
    # 只有第一个正残差计分: mean([1^2, 0, 0]) = 1/3
    assert np.isclose(scores[0], 1.0 / 3.0)


def test_default_chunk_adaptive_to_width(monkeypatch):
    """宽矩阵(大 C*H, 如 TriTrack 24步×87通道=2088列)不传 chunk_rows 时, 默认须【按列数自适应】
    缩小块, 使单次 maximum 行数受内存预算限 (否则固定 65536 行 × 2088 列 ×8 ≈ 1GB/块 → 训练内存
    紧张时 ArrayMemoryError, 实测 grid TriTrack 崩过)。"""
    rng = np.random.default_rng(4)
    n, cols = 5000, 2088
    y = rng.normal(0, 1, (n, cols)).astype(np.float32)
    pred = rng.normal(0, 1, (n, cols)).astype(np.float32)
    real_max = np.maximum
    seen = {"m": 0}

    def spy(a, b, *ar, **kw):
        arr = b if np.ndim(b) >= np.ndim(a) else a
        try:
            seen["m"] = max(seen["m"], int(np.shape(arr)[0]))
        except Exception:
            pass
        return real_max(a, b, *ar, **kw)

    monkeypatch.setattr(np, "maximum", spy)
    regression_metrics(y, pred)                       # 默认 chunk (自适应)
    budget_rows = 64 * 1024 * 1024 // (cols * 8) + 1   # 每块 float64 ≤ ~64MB
    assert seen["m"] <= budget_rows, f"max_rows={seen['m']} > 预算 {budget_rows}, 默认块未自适应缩小"


def test_chunk_does_not_materialize_full_float64(monkeypatch):
    """峰值内存守卫: 分块路径下, 单次 np.maximum 不应在整个 N 上分配。
    通过拦截 np.maximum 记录最大入参行数, 断言 <= chunk_rows。"""
    rng = np.random.default_rng(3)
    n = 1000
    y = rng.normal(0, 1, (n, 50)).astype(np.float32)
    pred = rng.normal(0, 1, (n, 50)).astype(np.float32)

    real_maximum = np.maximum
    seen = {"max_rows": 0}

    def spy_maximum(a, b, *args, **kw):
        arr = b if np.ndim(b) >= np.ndim(a) else a
        try:
            rows = np.shape(arr)[0]
            seen["max_rows"] = max(seen["max_rows"], int(rows))
        except Exception:
            pass
        return real_maximum(a, b, *args, **kw)

    monkeypatch.setattr(np, "maximum", spy_maximum)
    regression_metrics(y, pred, chunk_rows=128)
    assert seen["max_rows"] <= 128, f"max_rows={seen['max_rows']} 超过 chunk_rows, 仍在大矩阵上分配"
