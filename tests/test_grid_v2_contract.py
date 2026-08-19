from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import 启动
from 实验配置 import GridSearchProtocol


def _args(**kw):
    ns = argparse.Namespace(
        split="chronological_v2",
        cache_version="v2",
        dry_run=False,
        smoke=False,
        confirm_full=True,
        force=False,
        farms="kelmarsh,penmanshiel,hill_of_towie",
        models="anomaly,tranad,tritrack,wt",
        module="all",
        seeds="0,1,2,3,4",
        norm="robust",
        epochs=10,
        batch_size=128,
        skip_preprocess=False,
        model_python=sys.executable,
        label_mode="real_fault_wl",
        a1_lead=72,
    )
    for key, value in kw.items():
        setattr(ns, key, value)
    return ns


def test_grid_search_passes_chronological_v2_contract(monkeypatch, tmp_path):
    commands: list[tuple[str, list[str]]] = []
    build_batch_sizes: list[int] = []

    monkeypatch.setattr(启动, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(启动, "RESULT_DIR", tmp_path / "results")
    monkeypatch.setattr(启动, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(启动, "resolve_model_python", lambda _value: sys.executable)
    monkeypatch.setattr(
        GridSearchProtocol,
        "expand_grid",
        classmethod(lambda cls: [{"lr": 1e-4, "batch_size": 64, "scheduler": "plateau"}]),
    )
    monkeypatch.setattr(GridSearchProtocol, "total_combinations", classmethod(lambda cls: 1))

    def fake_run(cmd, step_name, cwd, log_path, *args, **kwargs):
        commands.append((step_name, list(cmd)))
        return 0

    monkeypatch.setattr(启动, "_run_single_step", fake_run)
    monkeypatch.setattr(
        启动,
        "_select_best_from_grid",
        lambda grid_dir: {
            "lr": 1e-4,
            "batch_size": 64,
            "scheduler": "plateau",
            "selected_by": "grid_val_mean_f1",
        },
    )

    def fake_build_steps(args, batch, force_all):
        build_batch_sizes.append(int(args.batch_size))
        return []

    monkeypatch.setattr(启动, "build_steps", fake_build_steps)
    monkeypatch.setitem(
        sys.modules,
        "实验监控",
        types.SimpleNamespace(generate_plots=lambda: 0),
    )

    assert 启动._grid_search_and_full_run(_args()) == 0

    preprocess = [cmd for name, cmd in commands if name == "preprocess_kelmarsh"][0]
    assert preprocess[preprocess.index("--split") + 1] == "chronological_v2"
    assert preprocess[preprocess.index("--cache-version") + 1] == "v2"
    assert preprocess[preprocess.index("--label-mode") + 1] == "real_fault_wl"

    model_cmds = [cmd for name, cmd in commands if name.startswith("grid_kelmarsh_")]
    assert model_cmds, "grid should execute model commands"
    assert len(model_cmds) == 4
    for cmd in model_cmds:
        assert cmd[cmd.index("--split-id") + 1] == "chronological_v2"
        assert cmd[cmd.index("--feature-version") + 1] == "v2"
        assert "--run-id" in cmd
        assert cmd[cmd.index("--run-id") + 1].startswith("grid_")
        assert "--output-dir" in cmd
        assert "chronological_v2__v2" in cmd[cmd.index("--output-dir") + 1]
        assert "grid" in Path(cmd[cmd.index("--output-dir") + 1]).parts

    assert build_batch_sizes == [64]


def test_select_best_from_grid_scans_only_grid_base(monkeypatch, tmp_path):
    formal_root = tmp_path / "formal"
    grid_root = tmp_path / "grid"
    formal_metrics = formal_root / "kelmarsh" / "anomaly_transformer" / "metrics.jsonl"
    grid_metrics = grid_root / "kelmarsh" / "anomaly_transformer" / "metrics.jsonl"
    formal_metrics.parent.mkdir(parents=True)
    grid_metrics.parent.mkdir(parents=True)
    formal_metrics.write_text(
        json.dumps(
            {
                "phase": "val",
                "model": "AnomalyTransformer",
                "f1": 0.99,
                "lr": 1e-3,
                "batch_size": 128,
                "scheduler": "cosine",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    grid_metrics.write_text(
        json.dumps(
            {
                "phase": "val",
                "model": "AnomalyTransformer",
                "f1": 0.50,
                "lr": 1e-5,
                "batch_size": 64,
                "scheduler": "steplr",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(启动, "RESULT_DIR", formal_root)
    best = 启动._select_best_from_grid(grid_root)

    assert best == {"lr": 1e-5, "batch_size": 64, "scheduler": "steplr"}
