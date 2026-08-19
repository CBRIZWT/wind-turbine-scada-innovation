# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "快速实验" / "运行真实故障全指标_v3.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("fast_runner_v3", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_true_fault_run_matrix_contains_47_models_for_each_of_two_farms():
    runner = _load_runner()
    scripts = [Path(f"{i:02d}_model.py") for i in range(47)]
    matrix = runner.build_run_matrix(("kelmarsh", "penmanshiel"), scripts)
    assert len(matrix) == 94
    assert ("penmanshiel", Path("46_model.py")) in matrix
    assert sum(farm == "kelmarsh" for farm, _ in matrix) == 47
    assert sum(farm == "penmanshiel" for farm, _ in matrix) == 47


def test_metrics_v3_completion_check_requires_three_workpoints_and_schema(tmp_path):
    runner = _load_runner()
    p = tmp_path / "metrics.json"
    p.write_text(
        '{"schema_version":"metrics-v3","workpoints":{"balanced":{},"low_far":{},"high_recall":{}}}',
        encoding="utf-8",
    )
    assert runner.is_complete_metrics(p)
    p.write_text('{"schema_version":"metrics-v3","workpoints":{"balanced":{}}}', encoding="utf-8")
    assert not runner.is_complete_metrics(p)


def test_all_hill_unlabelled_model_scripts_forward_training_scores():
    model_ids = ("00", "18", "19", "20", "21", "29", "30", "31", "36", "37", "40", "41")
    scripts = list((ROOT / "快速实验" / "基础模型").glob("[0-9][0-9]_*.py"))
    selected = [p for p in scripts if p.stem[:2] in model_ids]
    assert len(selected) == 12
    for script in selected:
        text = script.read_text(encoding="utf-8")
        assert "train_scores=" in text, script.name


def test_resume_manifest_only_counts_current_run_matrix():
    runner = _load_runner()
    matrix = [("kelmarsh", Path("01_a.py")), ("penmanshiel", Path("01_a.py"))]
    previous = {
        "kelmarsh/00_old": {"status": "success"},
        "kelmarsh/01_a": {"status": "success"},
        "penmanshiel/01_a": {"status": "success"},
    }
    kept = runner.filter_previous_runs(previous, matrix)
    assert set(kept) == {"kelmarsh/01_a", "penmanshiel/01_a"}


def test_single_farm_parallel_runs_use_isolated_manifests(tmp_path):
    runner = _load_runner()
    assert runner.manifest_path_for(tmp_path, ("kelmarsh",)).name == "manifest__kelmarsh.json"
    assert runner.manifest_path_for(tmp_path, ("penmanshiel",)).name == "manifest__penmanshiel.json"
    assert runner.manifest_path_for(tmp_path, ("kelmarsh", "penmanshiel")).name == "manifest.json"
