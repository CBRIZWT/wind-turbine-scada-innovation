from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np


def test_voting_fallback_uses_train_not_val():
    """阈值选择失败(val 单类)时, fallback 取第 3 参(train)分布, 不取 val。"""
    from 实验工具 import choose_threshold_and_polarity_by_validation
    val_labels = np.zeros(50, dtype=int)
    val_scores = np.full(50, 999.0)
    train_scores = np.linspace(0.0, 1.0, 1000)
    thr, src, pol = choose_threshold_and_polarity_by_validation(
        val_labels, val_scores, train_scores, fallback_quantile=0.99)
    assert thr <= 1.0, f"阈值应取 train q99(≈0.99), 实得 {thr} → 误用 val 分布"
    assert "uncalibrated" in src


def test_rank_normalize_reference_is_train_not_self():
    """rank_normalize(reference=train): 同一 test 值的归一化只取决于 train 分布, 与其它 test 值无关。"""
    from 集成评价 import rank_normalize
    train = np.linspace(0.0, 1.0, 100)
    na = rank_normalize(np.array([0.5, 0.5, 0.5]), reference=train)
    nb = rank_normalize(np.array([0.5, 0.9, 0.1]), reference=train)
    assert abs(na[0] - nb[0]) < 1e-9, "0.5 对 train 参照应归一化到同一排名(非自归一)"


def test_rank_normalize_backward_compat():
    """reference=None 保留旧自排名行为(单调、范围(0,1])。"""
    from 集成评价 import rank_normalize
    out = rank_normalize(np.array([10.0, 20.0, 30.0]))
    assert out[0] < out[1] < out[2] and 0 < out[-1] <= 1.0


def test_discover_score_files_includes_wt_transformer(monkeypatch, tmp_path):
    """四模型集成必须纳入 wt_transformer, 不能只扫三模型。"""
    from 实验配置 import ResultLayout
    from 集成评价 import discover_score_files

    monkeypatch.setattr(ResultLayout, "RESULT_ROOT", tmp_path)
    scores = tmp_path / "chronological_v2__v2" / "kelmarsh" / "wt_transformer" / "scores"
    scores.mkdir(parents=True)
    np.save(scores / "val_scores__baseline_only__seed0.npy", np.array([0.1, 0.2]))
    np.save(scores / "val_labels__baseline_only__seed0.npy", np.array([0, 1]))
    np.save(scores / "test_scores__baseline_only__seed0.npy", np.array([0.3, 0.4]))
    np.save(scores / "test_labels__baseline_only__seed0.npy", np.array([0, 1]))

    index = discover_score_files("kelmarsh", split_id="chronological_v2", feature_version="v2")

    assert ("wt_transformer", "baseline_only", 0) in index


def test_discover_score_files_uses_preprocess_variant(monkeypatch, tmp_path):
    """真实故障实验结果目录带 preprocess variant, 集成扫描必须走同一 tag。"""
    from 实验配置 import ResultLayout
    from 集成评价 import discover_score_files

    monkeypatch.setattr(ResultLayout, "RESULT_ROOT", tmp_path)
    scores = (
        tmp_path / "chronological_v2__v2__realfault_temp_v1"
        / "kelmarsh" / "wt_transformer" / "scores"
    )
    scores.mkdir(parents=True)
    np.save(scores / "val_scores__baseline_only__seed0.npy", np.array([0.1, 0.2]))
    np.save(scores / "val_labels__baseline_only__seed0.npy", np.array([0, 1]))

    index = discover_score_files(
        "kelmarsh", split_id="chronological_v2", feature_version="v2",
        preprocess_variant="realfault_temp_v1",
    )

    assert ("wt_transformer", "baseline_only", 0) in index
