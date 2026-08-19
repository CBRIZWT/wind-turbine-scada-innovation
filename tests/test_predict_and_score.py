# -*- coding: utf-8 -*-
"""predict_and_score 融合等价性测试 (2026-06-20, TDD).

根因(系统崩溃判断): Hill 农场 (N=5.67M×53) 的 TriTrackNet predict 先物化整张
(N, 24×53) 预测张量 ≈ 28.8GB, 紧接 regression_metrics 时与 y(28.8GB) 同驻 ≈ 57GB,
超出 commit 上限 73.4GB → 被 OS 强杀 (System 事件 2004, 无 Python traceback)。

修复 (B 方案, 不依赖管理员/重启/pagefile): predict_and_score 逐批前向→立即打分,
永不物化整张预测张量, 峰值降到单批 (~MB)。

等价性契约 (与 实验工具.regression_metrics 内部分块同源):
  - sample_scores 为逐行 (axis=1) 约简 → 分批与整块【逐位一致】(喂阈值/极性/最终F1/AUC, 科研结果不变);
  - mse/mae 为标量, 仅累加顺序不同 → ~1e-12 数值等价 (仅日志, 不入分类结果)。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


def _load_tritrack_module():
    path = ROOT / "TriTrackNet-main" / "实验.py"
    spec = importlib.util.spec_from_file_location("tritrack_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeModel(nn.Module):
    """确定性无参模型: 输出 (b, cols), 完全由输入决定 (便于逐位比较两条路径)。"""

    def __init__(self, cols: int):
        super().__init__()
        self.cols = cols

    def forward(self, xb, domain_knowledge=None):  # noqa: ANN001
        b = xb.shape[0]
        s = xb.reshape(b, -1).float().mean(dim=1, keepdim=True)        # (b, 1)
        col_w = torch.arange(1, self.cols + 1, dtype=s.dtype).reshape(1, -1)  # (1, cols)
        return s * col_w                                              # (b, cols)


def test_predict_and_score_matches_predict_then_regression_metrics():
    mod = _load_tritrack_module()
    N, C, L, cols = 37, 4, 8, 5   # N 不整除 bs → 覆盖末块边界
    bs = 8
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, C, L)).astype(np.float32)
    y = rng.standard_normal((N, cols)).astype(np.float32)
    model = _FakeModel(cols).eval()
    device = torch.device("cpu")

    pred = mod.predict(model, x, bs, device)
    reg_ref, score_ref = mod.regression_metrics(y, pred)

    reg_new, score_new = mod.predict_and_score(model, x, y, bs, device)

    # 科研结果: 逐样本分数必须逐位一致
    assert score_new.shape == score_ref.shape
    assert np.array_equal(score_new, score_ref)
    # 回归标量: 数值等价 (累加顺序差异, 仅日志)
    assert abs(reg_new["mse"] - reg_ref["mse"]) <= 1e-9 * max(1.0, abs(reg_ref["mse"]))
    assert abs(reg_new["mae"] - reg_ref["mae"]) <= 1e-9 * max(1.0, abs(reg_ref["mae"]))
    assert abs(reg_new["rmse"] - reg_ref["rmse"]) <= 1e-9 * max(1.0, abs(reg_ref["rmse"]))


def test_phase_metrics_from_score_matches_phase_metrics():
    mod = _load_tritrack_module()
    N, cols = 29, 4
    rng = np.random.default_rng(1)
    y = rng.standard_normal((N, cols)).astype(np.float32)
    pred = rng.standard_normal((N, cols)).astype(np.float32)
    labels = rng.integers(0, 2, size=N).astype(int)
    threshold = 0.5

    cls_ref, score_o_ref = mod.phase_metrics(y, pred, labels, threshold, "positive")
    reg, score = mod.regression_metrics(y, pred)
    cls_new, score_o_new = mod.phase_metrics_from_score(reg, score, labels, threshold, "positive")

    assert np.array_equal(score_o_new, score_o_ref)
    assert cls_new == cls_ref
