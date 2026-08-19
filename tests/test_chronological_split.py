from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pytest
from 实验配置 import ChronologicalSplitProtocol, ResultLayout


def test_split_has_three_farms_and_ids():
    for farm in ("kelmarsh", "penmanshiel", "hill_of_towie"):
        cfg = ChronologicalSplitProtocol.get(farm)
        assert cfg["split_id"] == "chronological_v2"
        assert cfg["feature_version"] == "v2"
        assert "coverage_policy" in cfg and isinstance(cfg["coverage_policy"], str)
        assert cfg["train_years"] and cfg["test_years"]


def test_penmanshiel_main_vs_supplemental():
    cfg = ChronologicalSplitProtocol.get("penmanshiel")
    assert set(cfg["turbines_main"]) == {11, 12, 13, 14, 15}
    assert 3 not in cfg["turbines_supplemental"]
    assert set(cfg["turbines_supplemental"]) == {1, 2, 4, 5, 6, 7, 8, 9, 10}


def test_penmanshiel_splits_include_supplemental_turbines():
    """2026-08-09: WT01-10 必须真正进入 train/val/test。

    根因: 此前 turbines_train/val/test 只有 WT11-15, 而 turbines_supplemental 声明了
    WT01-10 却 supplemental_years=[] 且预处理零引用 —— 声明了但从未加载。代价实测:
    test 窗告警行 39→96 (2.5x)、Tier-1 真过温 4→22 (5.5x)。
    排除理由"2024 覆盖不稳定"对事件级评测不成立: event recall 按事件算,
    FAR 已按 healthy_turbine_days 归一化, 逐机组年份覆盖不齐不影响两者可比性。
    """
    cfg = ChronologicalSplitProtocol.get("penmanshiel")
    main = set(cfg["turbines_main"])
    supp = set(cfg["turbines_supplemental"])
    for key in ("turbines_train", "turbines_val", "turbines_test"):
        assert set(cfg[key]) == main | supp, f"{key} 必须是 main ∪ supplemental"
    assert len(cfg["turbines_train"]) == 14, "Penmanshiel 共 14 台可用机组 (WT03 缺失)"


def test_penmanshiel_supplemental_years_declared():
    """WT01-10 实际覆盖 2016-2023 (无 2024 包), 必须显式声明而非留空。"""
    cfg = ChronologicalSplitProtocol.get("penmanshiel")
    assert cfg["supplemental_years"] == [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]


def test_hot_2024_supplemental_only():
    cfg = ChronologicalSplitProtocol.get("hill_of_towie")
    assert 2024 not in cfg["test_years"]
    assert cfg["supplemental_years"] == [2024]


def test_resultlayout_versioned_path():
    p = ResultLayout.scores_dir("chronological_v2", "v2", "kelmarsh", "anomaly_transformer")
    parts = p.parts
    assert "chronological_v2__v2" in parts
    assert "kelmarsh" in parts and "anomaly_transformer" in parts and parts[-1] == "scores"


def test_unknown_farm_raises():
    with pytest.raises(KeyError):
        ChronologicalSplitProtocol.get("nonexistent")
