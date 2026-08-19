# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAST = ROOT / "快速实验"
sys.path.insert(0, str(FAST))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_prefix_turbines_keeps_source_farms_disjoint():
    prep = _load(FAST / "准备Hill无标签双协议_v3.py", "hill_prep_v3")
    a = prep.prefix_turbines("kelmarsh", np.array(["1", "2"]))
    b = prep.prefix_turbines("penmanshiel", np.array(["1", "2"]))
    assert a.tolist() == ["kelmarsh:1", "kelmarsh:2"]
    assert set(a).isdisjoint(set(b))


def test_source_sequence_health_filter_rejects_positive_or_ignore_anywhere_in_window():
    prep = _load(FAST / "准备Hill无标签双协议_v3.py", "hill_prep_health_v3")
    labels = np.array([0, 0, 1, 0, 0, -1, 0, 0])
    idx = np.array([2, 3, 4, 6, 7])
    mask = prep.healthy_sequence_mask(labels, idx, window=3)
    assert mask.tolist() == [False, False, False, False, False]
    labels[:] = 0
    assert prep.healthy_sequence_mask(labels, idx, window=3).tolist() == [True] * 5


def test_hill_run_matrix_is_12_models_times_two_protocols():
    runner = _load(FAST / "运行Hill无标签双协议_v3.py", "hill_runner_v3")
    scripts = [Path(f"{i}_model.py") for i in runner.HILL_MODEL_IDS]
    matrix = runner.build_hill_matrix(scripts)
    assert len(matrix) == 24
    assert {p for p, _ in matrix} == {"local_unlabeled_fit", "source_zero_shot"}


def test_external_completion_requires_quantile_workpoints(tmp_path):
    runner = _load(FAST / "运行Hill无标签双协议_v3.py", "hill_runner_v3_complete")
    p = tmp_path / "metrics.json"
    p.write_text(
        '{"schema_version":"metrics-v3","label_mode":"external_unlabeled",'
        '"workpoints":{"q99":{},"q995":{},"q999":{}}}',
        encoding="utf-8",
    )
    assert runner.is_complete_external(p)


def test_hill_resume_manifest_only_counts_selected_matrix():
    runner = _load(FAST / "运行Hill无标签双协议_v3.py", "hill_runner_v3_filter")
    matrix = [("local_unlabeled_fit", Path("00_a.py"))]
    previous = {
        "hill_of_towie/local_unlabeled_fit/00_a": {"status": "success"},
        "hill_of_towie/source_zero_shot/00_a": {"status": "success"},
    }
    kept = runner.filter_previous_runs(previous, matrix)
    assert set(kept) == {"hill_of_towie/local_unlabeled_fit/00_a"}
