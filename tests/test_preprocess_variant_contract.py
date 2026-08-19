# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SCADA数据集"))

import 启动
import 数据预处理
from 实验配置 import PerFarmPaths, ResultLayout


def _args(**kw):
    ns = argparse.Namespace(
        split="chronological_v2",
        cache_version="v2",
        dry_run=False,
        smoke=True,
        confirm_full=False,
        force=False,
        farms="kelmarsh",
        models="anomaly,tranad,tritrack,wt",
        module="baseline_only",
        seeds="0",
        norm="robust",
        epochs=1,
        batch_size=128,
        skip_preprocess=False,
        label_mode="real_fault_wl",
        a1_lead=72,
        preprocess_variant="old_preprocess",
        model_python=sys.executable,
        no_resume=False,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_variant_is_in_dataset_and_result_paths():
    # 默认口径仍用单一真源; 非空 variant 必须物理隔离, 避免真实故障实验覆盖 A' 缓存。
    default_paths = PerFarmPaths.for_farm(
        "kelmarsh", "chronological_v2", "v2", preprocess_variant="",
    )
    assert len({str(p) for p in default_paths.values()}) == 1
    assert default_paths["tranad"].name == "kelmarsh"

    paths = PerFarmPaths.for_farm(
        "kelmarsh", "chronological_v2", "v2", preprocess_variant="old_preprocess",
    )
    assert len({str(p) for p in paths.values()}) == 1          # 四 baseline 同一目录
    assert paths["tranad"].name == "kelmarsh__old_preprocess"
    assert paths["tranad"].parent.name == "数据预处理"

    # 结果路径仍按 variant 隔离(便于区分实验; ResultLayout 未改)。
    out = ResultLayout.run_root(
        "chronological_v2", "v2", "kelmarsh", "tranad", preprocess_variant="old_preprocess",
    )
    assert "chronological_v2__v2__old_preprocess" in str(out)


def test_launcher_threads_variant_through_steps(monkeypatch):
    monkeypatch.setattr(启动, "resolve_model_python", lambda value: sys.executable)
    steps = 启动.build_steps(_args(), batch=0, force_all=False)
    preprocess = next(step for step in steps if step.name == "preprocess_kelmarsh")
    assert preprocess.command[preprocess.command.index("--preprocess-variant") + 1] == "old_preprocess"
    # 非空 variant 的预处理产物也必须隔离, 防止覆盖默认 A' 数据。
    assert all("数据预处理" in str(p) and "__old_preprocess" in str(p)
               for p in preprocess.outputs)

    model_step = next(step for step in steps if step.name == "kelmarsh_tranad_baseline_only_seed0")
    assert model_step.command[model_step.command.index("--preprocess-variant") + 1] == "old_preprocess"
    # 结果路径仍按 variant 隔离。
    assert "__old_preprocess" in str(model_step.outputs[0])
    assert 启动.step_key(model_step, "full", smoke=True).endswith(":old_preprocess")


def test_protocol_signature_tracks_variant():
    sig = 启动.build_protocol_signature(_args(preprocess_variant="new_preprocess"))
    assert sig["preprocess_variant"] == "new_preprocess"


def test_preprocess_default_a1_lead_matches_formal_protocol():
    """直接运行数据预处理入口时, 默认 A' lead 也必须等于正式主口径 72。"""
    assert 数据预处理.DEFAULT_A1_LEAD == 72
    sig = inspect.signature(数据预处理.run_full_farm_mode)
    assert sig.parameters["a1_lead"].default == 72


def test_preprocess_quality_metric_helpers():
    rmse = getattr(数据预处理, "rmse")
    mape = getattr(数据预处理, "mape")
    reduction_ratio = getattr(数据预处理, "reduction_ratio")
    avg_mi = getattr(数据预处理, "avg_mi")
    subset_redundancy = getattr(数据预处理, "subset_redundancy")
    y_true = [1.0, 2.0, 4.0]
    y_pred = [1.0, 2.5, 3.0]
    assert round(rmse(y_true, y_pred), 6) == 0.645497
    assert round(mape(y_true, y_pred), 6) == 16.666667
    assert reduction_ratio(10, 4) == 60.0
    assert avg_mi([0.1, 0.2, 0.3], [0, 2]) == 0.2
    assert subset_redundancy([[1, 2, 3], [1, 2, 3], [3, 2, 1]]) == 1.0
