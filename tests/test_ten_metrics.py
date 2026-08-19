# -*- coding: utf-8 -*-
"""十项实验指标契约测试 (2026-08-09)。

要求覆盖: Accuracy / AUC / Recall / F1 / R² / MAE / RMSE / Precision / LeadTime / MAPE。
另含 auprc_lift —— 极端不平衡下排序判别力的正确依据。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FAST = ROOT / "快速实验" / "基础模型"
for p in (str(ROOT), str(FAST)):
    if p not in sys.path:
        sys.path.insert(0, p)

import os
os.environ.setdefault("FASTEXP_FARM", "kelmarsh")

TEN = ["accuracy", "auc", "recall", "f1", "r2", "mae", "rmse", "precision",
       "lead_time_hours_median", "mape_on_positives"]


def _metrics_fn():
    import importlib
    m = importlib.import_module("_common")
    return m._metrics


@pytest.fixture(scope="module")
def M():
    return _metrics_fn()


def _toy(n=2000, n_pos=40, seed=0):
    """构造有判别力的玩具数据: 正例分数整体更高。"""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_pos, replace=False)] = 1
    s = rng.normal(0.3, 0.1, n)
    s[y == 1] += 0.35
    return y, np.clip(s, 0, 1)


class TestTenMetricsPresent:
    def test_all_ten_keys_exist(self, M):
        y, s = _toy()
        out = M(y, s, float(np.quantile(s, 0.97)))
        missing = [k for k in TEN if k not in out]
        assert not missing, f"缺少指标: {missing}"

    def test_all_ten_are_floats_not_none(self, M):
        y, s = _toy()
        out = M(y, s, float(np.quantile(s, 0.97)))
        for k in TEN:
            assert out[k] is not None
            assert isinstance(out[k], (int, float)), f"{k} 类型异常: {type(out[k])}"


class TestMape:
    def test_mape_only_counts_positives(self, M):
        """MAPE 必须只在 y=1 行上算 —— y=0 行会除零发散。"""
        y, s = _toy(n=1000, n_pos=25)
        out = M(y, s, float(np.quantile(s, 0.97)))
        assert out["mape_n_positives"] == 25

    def test_mape_is_finite_and_bounded(self, M):
        y, s = _toy()
        out = M(y, s, float(np.quantile(s, 0.97)))
        assert np.isfinite(out["mape_on_positives"])
        assert 0.0 <= out["mape_on_positives"] <= 1.0

    def test_perfect_scores_give_zero_mape(self, M):
        y = np.array([0, 0, 1, 1, 0, 1, 0, 0])
        s = y.astype(float)                      # 完美预测
        out = M(y, s, 0.5)
        assert out["mape_on_positives"] == pytest.approx(0.0, abs=1e-9)

    def test_no_positives_yields_nan_not_crash(self, M):
        y = np.zeros(50, dtype=int)
        s = np.linspace(0, 1, 50)
        out = M(y, s, 0.9)
        assert np.isnan(out["mape_on_positives"])
        assert out["mape_n_positives"] == 0


class TestLeadTime:
    def test_lead_hours_is_rows_times_ten_minutes(self, M):
        y, s = _toy()
        out = M(y, s, float(np.quantile(s, 0.97)))
        lr = out["seg_lead_rows_median"]
        if np.isfinite(lr):
            assert out["lead_time_hours_median"] == pytest.approx(lr * 10.0 / 60.0)
        else:
            assert np.isnan(out["lead_time_hours_median"])

    def test_lead_nan_when_nothing_detected(self, M):
        y = np.zeros(200, dtype=int); y[100:105] = 1
        s = np.zeros(200)                        # 分数恒零 → 阈值之上无报警
        out = M(y, s, 0.5)
        assert np.isnan(out["lead_time_hours_median"])


class TestAuprcLift:
    def test_lift_is_auprc_over_base_rate(self, M):
        y, s = _toy()
        out = M(y, s, float(np.quantile(s, 0.97)))
        assert out["base_rate"] == pytest.approx(y.mean())
        assert out["auprc_lift"] == pytest.approx(out["auprc"] / y.mean())

    def test_discriminative_model_lifts_above_one(self, M):
        """有判别力的模型, lift 必须显著 >1 (随机模型 lift≈1)。"""
        y, s = _toy()
        out = M(y, s, float(np.quantile(s, 0.97)))
        assert out["auprc_lift"] > 3.0

    def test_random_scores_lift_near_one(self, M):
        rng = np.random.default_rng(7)
        y = np.zeros(4000, dtype=int)
        y[rng.choice(4000, 80, replace=False)] = 1
        s = rng.random(4000)                     # 与标签无关
        out = M(y, s, float(np.quantile(s, 0.98)))
        assert out["auprc_lift"] < 3.0, "随机分数的 lift 不应高"
