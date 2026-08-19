# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from 实验工具 import build_preprocess_identity, build_checkpoint_identity


def _load_tranad_experiment_module():
    path = ROOT / "TranAD-main" / "实验.py"
    old_cwd = Path.cwd()
    try:
        spec = importlib.util.spec_from_file_location("tranad_experiment_for_identity_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(old_cwd)


class _NoLoad:
    def __init__(self):
        self.called = False

    def load_state_dict(self, _state):
        self.called = True
        raise AssertionError("stale checkpoint must not call load_state_dict")


def test_preprocess_identity_hashes_cols_when_meta_has_no_cols_hash():
    meta = {
        "split_id": "chronological_v2",
        "feature_version": "v2",
        "cache_version": "v2",
        "preprocess_variant": "",
        "split_hash": "split1234",
        "raw_inventory_hash": "raw5678",
        "n_channels": 2,
        "cols": ["gear_oil", "bearing_temp"],
        "label_rule": {"type": "real_fault_earlywarning", "lead_steps": 72},
    }

    identity = build_preprocess_identity(meta)

    assert identity["split_id"] == "chronological_v2"
    assert identity["n_channels"] == 2
    assert identity["label_rule"]["lead_steps"] == 72
    assert identity["cols_hash"]
    assert identity["cols_hash"] == build_preprocess_identity(meta)["cols_hash"]


def test_tranad_restore_rejects_stale_checkpoint_without_loading_state(tmp_path, capsys):
    tranad = _load_tranad_experiment_module()
    checkpoint = tmp_path / "model.ckpt"
    expected_preprocess = build_preprocess_identity({
        "split_id": "chronological_v2",
        "feature_version": "v2",
        "cache_version": "v2",
        "split_hash": "new_split",
        "raw_inventory_hash": "new_raw",
        "n_channels": 87,
        "cols_hash": "new_cols",
        "label_rule": {"type": "real_fault_earlywarning", "lead_steps": 72},
    })
    expected_identity = build_checkpoint_identity(
        model="TranAD",
        farm="kelmarsh",
        module="tcn_input_residual",
        seed=0,
        input_shape=[100, 87],
        preprocess_identity=expected_preprocess,
    )
    torch.save(
        {
            "epoch": 4,
            "model_state_dict": {"old.weight": torch.zeros(1)},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "checkpoint_identity": {
                **expected_identity,
                "input_shape": [100, 18],
            },
        },
        checkpoint,
    )

    model = _NoLoad()
    optimizer = _NoLoad()
    scheduler = _NoLoad()
    last_epoch = tranad.restore_checkpoint(
        model,
        optimizer,
        scheduler,
        torch.device("cpu"),
        checkpoint,
        expected_identity=expected_identity,
    )

    out = capsys.readouterr().out
    assert last_epoch == -1
    assert "STALE_CHECKPOINT" in out
    assert "action=start_epoch0" in out
    assert model.called is False
    assert optimizer.called is False
    assert scheduler.called is False
