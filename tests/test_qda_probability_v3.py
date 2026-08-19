# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "快速实验" / "基础模型"
sys.path.insert(0, str(BASE))


def test_regularized_qda_handles_minority_smaller_than_feature_count():
    import _common

    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(200, 20))
    x1 = rng.normal(loc=0.5, size=(12, 20))
    model = _common.fit_regularized_qda(np.vstack([x0, x1]), np.r_[np.zeros(200), np.ones(12)])
    probability = model.predict_proba(x1[:3])[:, 1]
    assert np.isfinite(probability).all()
    assert ((0 <= probability) & (probability <= 1)).all()


def test_known_probability_models_are_explicitly_registered():
    import _common

    expected = {
        "01_逻辑回归", "03_朴素贝叶斯", "04_LDA", "05_QDA", "06_K近邻",
        "07_决策树", "08_随机森林", "09_ExtraTrees", "10_AdaBoost",
        "11_直方图梯度提升", "12_XGBoost", "13_LightGBM", "14_CatBoost",
        "17_高斯过程", "33_REINFORCE", "34_ActorCritic", "35_PPO",
        "38_软投票集成", "39_Stacking堆叠", "42_少样本_逻辑回归",
        "43_少样本_梯度提升", "45_少样本_半监督自训练", "46_零样本_跨场迁移",
    }
    assert expected <= _common.PROBABILITY_SCORE_MODELS


def test_pca_component_count_adapts_to_six_dimensional_source_protocol():
    import _common

    assert _common.pca_component_count(100, 6, requested=16) == 6
    assert _common.pca_component_count(5, 20, requested=16) == 5
    assert _common.pca_component_count(100, 20, requested=16) == 16
