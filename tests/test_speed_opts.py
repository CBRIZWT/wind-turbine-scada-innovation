# -*- coding: utf-8 -*-
"""实验提速 Approach A 的契约测试 (2026-06-16, TDD)。

should_skip_epoch_eval: 仅当显式开 SKIP 且【非 grid】时跳每-epoch 仅日志打分 (grid 自动免疫, 护 select_best)。
enable_tf32: 启用 TF32 matmul 加速 (~2-3×); SCADA_DISABLE_TF32=1 可关。
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from 实验工具 import should_skip_epoch_eval, enable_tf32, dataloader_num_workers  # noqa: E402
import importlib.util


def test_num_workers_default_zero():
    assert dataloader_num_workers({}) == 0


def test_num_workers_from_env():
    assert dataloader_num_workers({"SCADA_NUM_WORKERS": "4"}) == 4


def test_num_workers_invalid_falls_back_zero():
    assert dataloader_num_workers({"SCADA_NUM_WORKERS": "abc"}) == 0
    assert dataloader_num_workers({"SCADA_NUM_WORKERS": "-2"}) == 0


def _load_at_data_loader_module():
    path = Path(__file__).resolve().parents[1] / "Anomaly-Transformer-main" / "data_factory" / "data_loader.py"
    spec = importlib.util.spec_from_file_location("at_data_loader", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_at_num_workers_default_zero_for_windows_spawn_safety():
    module = _load_at_data_loader_module()
    assert module.at_num_workers({}) == 0


def test_at_num_workers_env_override():
    module = _load_at_data_loader_module()
    assert module.at_num_workers({"SCADA_AT_NUM_WORKERS": "2"}) == 2
    assert module.at_num_workers({"SCADA_AT_NUM_WORKERS": "-1"}) == 0
    assert module.at_num_workers({"SCADA_AT_NUM_WORKERS": "bad"}) == 0


# ---------------- should_skip_epoch_eval ----------------
def test_skip_default_off():
    assert should_skip_epoch_eval({}) is False


def test_skip_on_when_set_and_not_grid():
    assert should_skip_epoch_eval({"SCADA_SKIP_EPOCH_EVAL": "1"}) is True


def test_skip_off_in_grid_even_if_set():
    """grid (GRID_FAST=1) 必须免疫: 保留 per-epoch val 供 select_best。"""
    assert should_skip_epoch_eval(
        {"SCADA_SKIP_EPOCH_EVAL": "1", "SCADA_GRID_FAST": "1"}) is False


def test_skip_off_when_value_not_1():
    assert should_skip_epoch_eval({"SCADA_SKIP_EPOCH_EVAL": "0"}) is False


def test_skip_grid_fast_alone_off():
    assert should_skip_epoch_eval({"SCADA_GRID_FAST": "1"}) is False


# ---------------- enable_tf32 ----------------
def test_enable_tf32_sets_flags():
    try:
        import torch  # noqa: F401
    except Exception:
        import pytest
        pytest.skip("torch 不可用")
    import torch
    ok = enable_tf32({})
    assert ok is True
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


def test_enable_tf32_optout():
    """SCADA_DISABLE_TF32=1 → 不启用 (返回 False)。"""
    assert enable_tf32({"SCADA_DISABLE_TF32": "1"}) is False
