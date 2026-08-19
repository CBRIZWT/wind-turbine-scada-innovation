from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import 实验分析


def test_percentile_event_f1_reads_chronological_v2_scores(monkeypatch, tmp_path):
    result_dir = tmp_path / "实验结果"
    scores_dir = (
        result_dir
        / "chronological_v2__v2"
        / "kelmarsh"
        / "tranad"
        / "scores"
    )
    scores_dir.mkdir(parents=True)
    suffix = "__baseline_only__seed0.npy"

    np.save(scores_dir / f"val_scores{suffix}", np.array([0.0, 10.0, 0.0, 10.0]))
    np.save(scores_dir / f"val_labels{suffix}", np.array([0, 1, 0, 1]))
    np.save(scores_dir / f"test_scores{suffix}", np.array([0.0, 10.0, 10.0, 0.0]))
    np.save(scores_dir / f"test_labels{suffix}", np.array([0, 1, 1, 0]))

    monkeypatch.setattr(实验分析, "RESULT_DIR", result_dir)
    实验分析._cache.clear()

    f1 = 实验分析.percentile_event_f1_for_run(
        "kelmarsh", "tranad", "baseline_only", 0
    )

    assert f1 is not None
    assert f1 > 0.0
