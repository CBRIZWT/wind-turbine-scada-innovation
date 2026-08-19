# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_anomaly_experiment():
    path = ROOT / "Anomaly-Transformer-main" / "实验.py"
    old_cwd = Path.cwd()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("anomaly_experiment_for_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(old_cwd)


def _load_anomaly_data_loader():
    path = ROOT / "Anomaly-Transformer-main" / "data_factory" / "data_loader.py"
    spec = importlib.util.spec_from_file_location("anomaly_data_loader_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_epoch_metrics_train_probe_preserves_rng(monkeypatch, tmp_path):
    """OFF 路径采集 train 分数后必须恢复 RNG, 否则下一 epoch 训练顺序会变。"""
    exp = _load_anomaly_experiment()

    train_loader = object()
    val_loader = object()
    solver = SimpleNamespace(
        train_loader=train_loader,
        vali_loader=val_loader,
        dataset="SCADA",
        input_c=2,
        win_size=4,
    )

    def fake_collect(_solver, loader):
        if loader is train_loader:
            random.random()
            np.random.random()
            torch.rand(1)
        scores = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)
        labels = np.array([0, 1, 0, 1], dtype=int)
        return scores, labels

    monkeypatch.delenv("SCADA_GRID_FAST", raising=False)
    monkeypatch.setattr(exp, "collect_energy_and_labels", fake_collect)
    monkeypatch.setattr(exp, "record_and_print_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp, "_log_wandb_epoch", lambda *args, **kwargs: None)

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    ret = exp.record_epoch_metrics(
        solver,
        epoch=1,
        train_loss=0.1,
        val_loss=0.2,
        run_kind="smoke",
        farm="kelmarsh",
        module="baseline_only",
        seed=0,
        output_dir_override=str(tmp_path),
    )

    assert isinstance(ret, float)
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_anomaly_loader_num_workers_env_override(monkeypatch):
    loader_mod = _load_anomaly_data_loader()
    captured = {}

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return torch.zeros(4, 2), torch.zeros(4, dtype=torch.long)

    def fake_dataloader(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(loader_mod, "SCADASegLoader", lambda *args, **kwargs: DummyDataset())
    monkeypatch.setattr(loader_mod, "DataLoader", fake_dataloader)
    monkeypatch.setenv("SCADA_AT_NUM_WORKERS", "0")

    loader_mod.get_loader_segment(
        "unused",
        batch_size=2,
        win_size=4,
        step=1,
        mode="train",
        dataset="SCADA",
    )

    assert captured["num_workers"] == 0
    assert "persistent_workers" not in captured


def test_fast_path_train_wandb_logging_stays_inside_train_metrics_branch():
    for rel in ("TranAD-main/实验.py", "TriTrackNet-main/实验.py"):
        lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
        train_log_lines = [line for line in lines if '_log_wandb_epoch("train"' in line]
        assert train_log_lines, rel
        for line in train_log_lines:
            assert line.startswith(" " * 16), f"{rel}: train wandb log must stay inside if not _grid_fast"


def test_train_metric_probes_restore_rng_state_in_all_fast_paths():
    for rel in (
        "Anomaly-Transformer-main/实验.py",
        "TranAD-main/实验.py",
        "TriTrackNet-main/实验.py",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "def _rng_state_snapshot" in text, rel
        assert "_rng_state = _rng_state_snapshot()" in text, rel
        assert "_restore_rng_state(_rng_state)" in text, rel
