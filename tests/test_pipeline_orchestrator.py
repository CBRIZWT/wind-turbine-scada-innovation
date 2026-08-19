# -*- coding: utf-8 -*-
"""
tests/test_pipeline_orchestrator.py — 全量编排.py 的 skip-if-done / 防泄漏判定

用户要求 (2026-06-04): 一键 `启动.py --pipeline` 逐阶段
  "检查文件: 不存在则运行, 存在则跳过, 并在命令行汇报哪个文件缺失",
  且科研严谨 + 防数据泄露。

本测试只覆盖【纯判定函数】(可注入路径/字典, 无子进程):
  - missing_files_in_dirs : 报告缺哪个文件 (空=齐全可跳过)
  - assert_leak_safe      : 防泄漏契约 (train 无正例 + test 不参与 fit/selection)
  - best_config_ready     : 超参文件存在判定 (阶段2 skip)
  - grid_step_done        : 单个网格步完成判定 (阶段2 断点续传)
编排胶水 (run_pipeline 调子进程) 不在单测范围。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from 全量编排 import (
    ROOT as ORCH_ROOT,
    DATASET_REQUIRED_FILES,
    missing_files_in_dirs,
    assert_leak_safe,
    best_config_ready,
    grid_step_done,
    runtime_cache_env,
    run_pipeline,
)


def test_runtime_cache_env_redirects_off_c_drive():
    """缓存重定向: 落 ROOT 同盘(E:)、不在 C:、且【纯 ASCII】。
    (joblib/loky resource_tracker 对非 ASCII temp 路径会 UnicodeEncodeError — 2026-06-05 实测崩过。)"""
    env = runtime_cache_env()
    assert set(env) >= {"TEMP", "TMP", "JOBLIB_TEMP_FOLDER", "MPLCONFIGDIR"}
    drive = ORCH_ROOT.drive.upper()      # 'E:'
    for v in env.values():
        assert v.upper().startswith(drive)       # 与项目同盘 (E:)
        assert not v.upper().startswith("C:")
        assert v.isascii()                        # 关键: joblib temp 必须 ASCII


def test_missing_files_lists_only_absent(tmp_path):
    """只缺的文件被列出; 已存在的不在列表里。"""
    d = tmp_path / "AT"
    d.mkdir()
    (d / "meta.json").write_text("{}", encoding="utf-8")  # 仅 meta 存在
    missing = missing_files_in_dirs({"anomaly_transformer": d})
    names = {p.name for p in missing}
    assert "meta.json" not in names
    assert "train.npy" in names and "test.npy" in names


def test_missing_files_empty_when_all_present(tmp_path):
    """7 件齐全 → 空列表 (可跳过)。"""
    d = tmp_path / "AT"
    d.mkdir()
    for fn in DATASET_REQUIRED_FILES:
        (d / fn).write_text("x", encoding="utf-8")
    assert missing_files_in_dirs({"anomaly_transformer": d}) == []


def test_missing_files_across_multiple_model_dirs(tmp_path):
    """多模型目录: 任一模型缺文件都要被报出。"""
    at = tmp_path / "AT"; at.mkdir()
    tad = tmp_path / "TAD"; tad.mkdir()
    for fn in DATASET_REQUIRED_FILES:
        (at / fn).write_text("x", encoding="utf-8")  # AT 齐全
    # TAD 全缺
    missing = missing_files_in_dirs({"anomaly_transformer": at, "tranad": tad})
    assert missing and all(str(tad) in str(p) for p in missing)


def test_missing_files_requires_wt_sidecar_b_labels(tmp_path):
    wt = tmp_path / "WT"; wt.mkdir()
    for fn in (
        "meta.json",
        "train.npy", "val.npy", "test.npy",
        "train_labels.npy", "val_labels.npy", "test_labels.npy",
    ):
        (wt / fn).write_text("x", encoding="utf-8")
    missing = missing_files_in_dirs({"wt_transformer": wt})
    names = {p.name for p in missing}
    assert {"train_labels_b.npy", "val_labels_b.npy", "test_labels_b.npy"} <= names


def test_assert_leak_safe_passes_on_clean_meta():
    """train_positive=0 且 test 未用于 fit/selection → 通过 (不抛)。"""
    assert_leak_safe({
        "labels_summary": {"train_positive": 0},
        "test_used_for_fit": False,
        "test_used_for_selection": False,
    })


def test_assert_leak_safe_rejects_train_positive():
    with pytest.raises(ValueError):
        assert_leak_safe({
            "labels_summary": {"train_positive": 5},
            "test_used_for_fit": False,
            "test_used_for_selection": False,
        })


def test_assert_leak_safe_rejects_test_used_for_selection():
    with pytest.raises(ValueError):
        assert_leak_safe({
            "labels_summary": {"train_positive": 0},
            "test_used_for_fit": False,
            "test_used_for_selection": True,
        })


def test_assert_leak_safe_rejects_missing_contract_fields():
    """缺字段视为不安全 (默认 True) → 抛, 绝不放行未知契约。"""
    with pytest.raises(ValueError):
        assert_leak_safe({"labels_summary": {"train_positive": 0}})


def test_best_config_ready(tmp_path):
    p = tmp_path / "best_config.json"
    assert best_config_ready(p) is False
    p.write_text("{}", encoding="utf-8")
    assert best_config_ready(p) is True


def test_stage1_preprocess_does_not_skip_stale_a1_protocol(monkeypatch, capsys):
    """已有文件但 meta 的 A' lead 与主协议不一致时, 阶段1 必须重跑而不是 skip。"""
    import 全量编排

    monkeypatch.setattr(全量编排, "FARMS", ("hill_of_towie",))
    monkeypatch.setattr(全量编排, "missing_dataset_files", lambda farm: [])
    monkeypatch.setattr(
        全量编排,
        "read_meta",
        lambda farm: {
            "split_id": "chronological_v2",
            "feature_version": "v2",
            "cache_version": "v2",
            "preprocess_variant": "",
            "primary_label": {"name": "real_fault_wl"},
            "label_rule": {
                "type": "real_fault_earlywarning",
                "policy": "real_fault_wl",
                "lead_steps": 36,
            },
            "labels_summary": {"train_positive": 0},
            "test_used_for_fit": False,
            "test_used_for_selection": False,
        },
    )
    args = argparse.Namespace(
        label_mode="real_fault_wl",
        a1_lead=72,
    )

    全量编排.stage1_preprocess(args=args, dry_run=True)

    out = capsys.readouterr().out
    assert "[skip]" not in out
    assert "[run ] hill_of_towie" in out
    assert "protocol_mismatch" in out
    assert "--a1-lead 72" in out


def test_stage1_preprocess_records_verified_skip_in_manifest(monkeypatch):
    """stage1 验证已有预处理符合协议时, 也要同步 manifest, 避免阶段3重复跑预处理。"""
    import 全量编排

    calls = []
    monkeypatch.setattr(全量编排, "FARMS", ("kelmarsh",))
    monkeypatch.setattr(全量编排, "missing_dataset_files", lambda farm: [])
    monkeypatch.setattr(
        全量编排,
        "read_meta",
        lambda farm: {
            "split_id": "chronological_v2",
            "feature_version": "v2",
            "cache_version": "v2",
            "preprocess_variant": "",
            "primary_label": {"name": "real_fault_wl"},
            "label_rule": {
                "type": "real_fault_earlywarning",
                "policy": "real_fault_wl",
                "lead_steps": 72,
            },
            "labels_summary": {"train_positive": 0},
            "test_used_for_fit": False,
            "test_used_for_selection": False,
        },
    )
    monkeypatch.setattr(
        全量编排,
        "record_preprocess_manifest_success",
        lambda farm, args, log_path=None: calls.append((farm, log_path)),
        raising=False,
    )
    args = argparse.Namespace(
        label_mode="real_fault_wl",
        a1_lead=72,
        preprocess_variant="",
    )

    全量编排.stage1_preprocess(args=args, dry_run=False)

    assert calls == [("kelmarsh", None)]


def test_stage1_preprocess_dry_run_does_not_record_verified_skip(monkeypatch):
    """dry-run 只做预览; 即使已有预处理符合协议, 也不能写 manifest。"""
    import 全量编排

    calls = []
    monkeypatch.setattr(全量编排, "FARMS", ("kelmarsh",))
    monkeypatch.setattr(全量编排, "missing_dataset_files", lambda farm: [])
    monkeypatch.setattr(
        全量编排,
        "read_meta",
        lambda farm: {
            "split_id": "chronological_v2",
            "feature_version": "v2",
            "cache_version": "v2",
            "preprocess_variant": "",
            "primary_label": {"name": "real_fault_wl"},
            "label_rule": {
                "type": "real_fault_earlywarning",
                "policy": "real_fault_wl",
                "lead_steps": 72,
            },
            "labels_summary": {"train_positive": 0},
            "test_used_for_fit": False,
            "test_used_for_selection": False,
        },
    )
    monkeypatch.setattr(
        全量编排,
        "record_preprocess_manifest_success",
        lambda farm, args, log_path=None: calls.append((farm, log_path)),
        raising=False,
    )
    args = argparse.Namespace(
        label_mode="real_fault_wl",
        a1_lead=72,
        preprocess_variant="",
    )

    全量编排.stage1_preprocess(args=args, dry_run=True)

    assert calls == []


def test_grid_step_done(tmp_path):
    root = tmp_path / "grid"
    args = (root, "kelmarsh", "tranad", "lr1e-03_bs64_cosine", "baseline_only", 0)
    assert grid_step_done(*args) is False
    mj = root / "kelmarsh" / "tranad" / "lr1e-03_bs64_cosine" / "baseline_only" / "seed0" / "metrics.jsonl"
    mj.parent.mkdir(parents=True)
    mj.write_text('{"phase":"val"}', encoding="utf-8")
    assert grid_step_done(*args) is True


def test_run_pipeline_configures_launcher_versioned_manifest(monkeypatch, tmp_path):
    """直接经编排器进入阶段3前, 启动.py 的 manifest/log 真源也必须切到 v2 子目录。"""
    import 启动
    import 全量编排

    monkeypatch.setattr(全量编排, "PIPELINE_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(全量编排, "apply_runtime_cache", lambda: {})
    monkeypatch.setattr(全量编排, "stage1_preprocess", lambda args=None, dry_run=False: None)
    monkeypatch.setattr(全量编排, "stage2_grid", lambda args=None, dry_run=False: None)
    monkeypatch.setattr(全量编排, "stage3_full", lambda args=None, dry_run=False: 0)

    启动.configure_experiment_layout("narrow_v1", "v1")
    args = argparse.Namespace(
        split="chronological_v2",
        feature_version="v2",
        cache_version="v2",
        dry_run=True,
    )

    assert run_pipeline(args, run_full=True, dry_run=True) == 0
    assert "chronological_v2__v2" in str(启动.MANIFEST_PATH)
    assert "chronological_v2__v2" in str(启动.LOG_DIR)
