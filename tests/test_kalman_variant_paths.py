# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_kalman_module():
    path = ROOT / "卡尔曼分数平滑评价.py"
    spec = importlib.util.spec_from_file_location("kalman_score_eval", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_scores_dir_uses_preprocess_variant(monkeypatch, tmp_path):
    from 实验配置 import ResultLayout

    monkeypatch.setattr(ResultLayout, "RESULT_ROOT", tmp_path)
    mod = _load_kalman_module()

    out = mod._scores_dir(
        "chronological_v2", "v2", "kelmarsh", "wt_transformer",
        preprocess_variant="realfault_temp_v1",
    )

    assert out == (
        tmp_path / "chronological_v2__v2__realfault_temp_v1"
        / "kelmarsh" / "wt_transformer" / "scores"
    )
