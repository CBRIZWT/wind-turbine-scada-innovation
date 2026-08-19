# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_tranad_experiment_module():
    path = ROOT / "TranAD-main" / "实验.py"
    old_cwd = Path.cwd()
    try:
        spec = importlib.util.spec_from_file_location("tranad_experiment_for_window_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(old_cwd)


def test_convert_to_windows_matches_historical_padding_semantics():
    tranad = _load_tranad_experiment_module()
    data = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ],
        dtype=np.float32,
    )

    windows = tranad.convert_to_windows(data, 3)

    expected = torch.tensor(
        [
            [[1.0, 10.0], [1.0, 10.0], [1.0, 10.0]],
            [[1.0, 10.0], [1.0, 10.0], [1.0, 10.0]],
            [[1.0, 10.0], [1.0, 10.0], [2.0, 20.0]],
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(windows, expected)


def test_convert_to_windows_returns_lazy_view_instead_of_full_materialization():
    tranad = _load_tranad_experiment_module()
    T, D, L = 128, 7, 5
    data = np.arange(T * D, dtype=np.float32).reshape(T, D)

    windows = tranad.convert_to_windows(data, L)

    assert windows.shape == (T, L, D)
    assert not windows.is_contiguous()
    assert windows.untyped_storage().nbytes() <= (T + L) * D * np.dtype(np.float32).itemsize

    loader = DataLoader(TensorDataset(windows), batch_size=16, shuffle=False, num_workers=0)
    (batch,) = next(iter(loader))
    assert batch.shape == (16, L, D)
    assert batch.is_contiguous()


def test_convert_to_windows_accepts_explicit_cpu_device():
    tranad = _load_tranad_experiment_module()
    data = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)

    windows = tranad.convert_to_windows(data, 3, device=torch.device("cpu"))

    assert windows.device.type == "cpu"
    assert torch.equal(windows, tranad.convert_to_windows(data, 3))


def test_select_window_device_keeps_cpu_training_on_cpu():
    tranad = _load_tranad_experiment_module()
    arrays = {
        "train": np.zeros((8, 2), dtype=np.float32),
        "val": np.zeros((4, 2), dtype=np.float32),
        "test": np.zeros((4, 2), dtype=np.float32),
    }

    device = tranad.select_window_device(arrays, 3, torch.device("cpu"))

    assert device.type == "cpu"


def test_make_window_loader_batches_lazy_windows_by_index_collate():
    tranad = _load_tranad_experiment_module()
    data = np.arange(12 * 3, dtype=np.float32).reshape(12, 3)
    windows = tranad.convert_to_windows(data, 4)

    loader = tranad.make_window_loader(
        windows,
        batch_size=5,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    )

    batches = [batch for (batch,) in loader]
    assert [tuple(batch.shape) for batch in batches] == [(5, 4, 3), (5, 4, 3), (2, 4, 3)]
    assert torch.equal(batches[0], windows[:5])
    assert torch.equal(batches[1], windows[5:10])
    assert torch.equal(batches[2], windows[10:12])
    assert all(batch.is_contiguous() for batch in batches)
