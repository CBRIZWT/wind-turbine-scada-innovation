# -*- coding: utf-8 -*-
"""
test_tritracknet_polarity.py

Regression test: phase_metrics must orient scores by polarity before thresholding,
so that the threshold and final metrics use the same oriented score.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TriTrackNet-main"))
import numpy as np


def test_phase_metrics_applies_polarity():
    """phase_metrics 必须按 polarity 定向后再阈值化 → 阈值与 final metrics 同向。
    正式环境统一用 conda env `chuangxin` (E:\\ancoda\\chuangxin\\python.exe, 含 torch) 运行。"""
    # 多个模型目录都有 `实验.py`；全套测试中可能已有同名模块缓存。
    sys.modules.pop("实验", None)
    tri_root = str(ROOT / "TriTrackNet-main")
    sys.path = [p for p in sys.path if p != tri_root]
    sys.path.insert(0, tri_root)
    from 实验 import phase_metrics
    from 实验工具 import regression_metrics, compute_binary_metrics, orient_scores
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, (60, 3)); pred = rng.normal(0, 1, (60, 3))
    labels = np.zeros(60, dtype=int); labels[45:] = 1
    thr = 0.5
    m_pos, _ = phase_metrics(y, pred, labels, thr, "positive")
    m_neg, _ = phase_metrics(y, pred, labels, thr, "negative")
    # polarity 真的被应用: pos 与 neg 的混淆矩阵不同
    assert (m_pos["tp"], m_pos["fp"]) != (m_neg["tp"], m_neg["fp"])
    # negative 等价于手动 orient 后再算
    _, score = regression_metrics(y, pred)
    expect = compute_binary_metrics(labels, scores=orient_scores(score, "negative"), threshold=thr)
    assert m_neg["tp"] == expect["tp"] and m_neg["fp"] == expect["fp"]
