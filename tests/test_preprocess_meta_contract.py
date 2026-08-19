# -*- coding: utf-8 -*-
"""
tests/test_preprocess_meta_contract.py — 预处理 split 解析 + 可审计 meta 契约 (2026-06-02)

覆盖:
    - resolve_split_config: narrow_v1 / chronological_v2 两条 split 源的纯函数解析
    - build_preprocess_meta: 可审计 meta 字典的必备顶层字段 (split_id / test_used_for_fit 等)
    - validate_meta_contract: 泄漏/去污失败时 fail-fast (ValueError)

科研标准: 这些是预处理产物可被 (split_id, feature_version) 审计 + 杜绝 test 泄漏的守卫单测,
    全部用合成最小输入, 不触碰 153GB gated 真实数据。
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "SCADA数据集"))
import numpy as np, pandas as pd, pytest
from 数据预处理 import resolve_split_config, build_preprocess_meta, validate_meta_contract


# ----------------------------------------------------------------
# resolve_split_config
# ----------------------------------------------------------------
def test_resolve_split_chronological_v2():
    """chronological_v2: split_id/feature_version 正确, 年份→窗口转换正确。"""
    cfg = resolve_split_config("kelmarsh", "chronological_v2")
    assert cfg["split_id"] == "chronological_v2"
    assert cfg["feature_version"] == "v2"
    assert cfg["train_start"].startswith("2016")
    assert cfg["test_years"] == [2023, 2024]


def test_resolve_split_narrow_v1():
    """narrow_v1: split_id 保持 narrow_v1 (向后兼容旧窄窗行为)。"""
    cfg = resolve_split_config("kelmarsh", "narrow_v1")
    assert cfg["split_id"] == "narrow_v1"


# ----------------------------------------------------------------
# build_preprocess_meta
# ----------------------------------------------------------------
def _minimal_meta_kwargs():
    """构造 build_preprocess_meta 的最小合成入参 (各 block 用占位 dict)。"""
    return dict(
        split_id="chronological_v2",
        feature_version="v2",
        cache_version="v1",
        split_hash="deadbeefdeadbeef",
        raw_inventory_hash="cafef00dcafef00d",
        split_cfg={"farm": "kelmarsh"},
        selection_meta={"n_selected": 3},
        nbm_meta={"model_kind": "gbr"},
        label_rule_meta={"type": "x", "half_window_hours": 2.0},
        scaler_meta={"type": "robust"},
        shapes_meta={"train": [10, 3], "val": [5, 3], "test": [5, 3]},
        labels_summary={"train_positive": 0, "val_positive": 1, "test_positive": 2,
                        "train_ignore": 0, "val_ignore": 0, "test_ignore": 0},
        primary_label_meta={"name": "real_fault_wl", "role": "training"},
        aux_labels_meta={"B": {"role": "downstream_validation"}},
    )


def test_build_meta_has_required_top_level_keys():
    """meta 必须含全部可审计顶层字段, 且 test_used_for_* 默认 False。"""
    meta = build_preprocess_meta(**_minimal_meta_kwargs())
    for key in ("split_id", "feature_version", "cache_version", "split_hash",
                "raw_inventory_hash", "test_used_for_fit", "test_used_for_selection"):
        assert key in meta, f"meta 缺顶层字段: {key}"
    assert meta["test_used_for_fit"] is False
    assert meta["test_used_for_selection"] is False
    # 嵌套 block 与 labels_summary 也保留
    assert meta["split_id"] == "chronological_v2"
    assert meta["labels_summary"]["train_positive"] == 0
    # 各 S 步骤 block 原样嵌入
    for block in ("indicator_selection", "nbm", "label_rule", "split", "scaler", "shapes"):
        assert block in meta, f"meta 缺嵌套 block: {block}"
    assert meta["primary_label"]["name"] == "real_fault_wl"
    assert meta["aux_labels"]["B"]["role"] == "downstream_validation"


# ----------------------------------------------------------------
# validate_meta_contract
# ----------------------------------------------------------------
def _clean_meta():
    return {
        "test_used_for_fit": False,
        "test_used_for_selection": False,
        "labels_summary": {"train_positive": 0},
    }


def test_validate_raises_on_test_used_for_fit():
    m = _clean_meta(); m["test_used_for_fit"] = True
    with pytest.raises(ValueError):
        validate_meta_contract(m)


def test_validate_raises_on_test_used_for_selection():
    m = _clean_meta(); m["test_used_for_selection"] = True
    with pytest.raises(ValueError):
        validate_meta_contract(m)


def test_validate_raises_on_train_positive_nonzero():
    m = _clean_meta(); m["labels_summary"]["train_positive"] = 5
    with pytest.raises(ValueError):
        validate_meta_contract(m)


def test_validate_passes_on_clean_meta():
    # 不应抛出
    validate_meta_contract(_clean_meta())
