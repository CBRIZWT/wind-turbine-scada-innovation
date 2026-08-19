# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "回算正式结果全指标_v3.py"
    spec = importlib.util.spec_from_file_location("formal_backfill_v3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_backfill_uses_existing_threshold_and_never_mutates_history(tmp_path):
    mod = _load()
    root = tmp_path / "formal"
    scores = root / "kelmarsh" / "wt_transformer" / "scores"
    scores.mkdir(parents=True)
    np.save(scores / "test_scores__baseline_only__seed0.npy", np.array([0.1, 0.9, 0.8, 0.2]))
    np.save(scores / "test_labels__baseline_only__seed0.npy", np.array([0, 1, 1, -1]))
    history = root / "kelmarsh" / "wt_transformer" / "metrics.jsonl"
    line = json.dumps({"phase": "test", "epoch": "final", "farm": "kelmarsh",
                       "model": "WTTransformer", "module": "baseline_only", "seed": 0,
                       "threshold": 0.5, "threshold_source": "validation_evt1_f1",
                       "score_polarity": "positive", "mse": 1.5, "mae": 1.0,
                       "rmse": 1.225, "hi_mean": 0.8, "hi_warn": 0.7})
    history.write_text(line + "\n", encoding="utf-8")
    original = history.read_bytes()

    pairs = mod.discover_score_pairs(root)
    assert len(pairs) == 1
    out = tmp_path / "out"
    rec = mod.backfill_pair(pairs[0], output_root=out)
    assert history.read_bytes() == original
    assert rec["evaluation"]["metrics"]["f1"] == 1.0
    assert rec["evaluation"]["metrics"]["mse"] == 1.5
    assert rec["evaluation"]["metric_status"]["n_events"] == "unavailable_artifact"
    assert rec["source_artifacts"]["scores"].endswith("test_scores__baseline_only__seed0.npy")
    saved = out / "kelmarsh" / "wt_transformer" / "baseline_only" / "seed0" / "metrics.json"
    assert saved.exists()


def test_missing_historical_record_is_reported_not_guessed(tmp_path):
    mod = _load()
    root = tmp_path / "formal"
    scores = root / "kelmarsh" / "tranad" / "scores"
    scores.mkdir(parents=True)
    np.save(scores / "test_scores__baseline_only__seed0.npy", np.array([0.1, 0.9]))
    np.save(scores / "test_labels__baseline_only__seed0.npy", np.array([0, 1]))
    pair = mod.discover_score_pairs(root)[0]
    rec = mod.backfill_pair(pair, output_root=tmp_path / "out")
    assert rec["status"] == "unavailable_historical_threshold"
    assert rec["evaluation"] is None
