# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "快速实验" / "汇总全指标_v3.py"
    spec = importlib.util.spec_from_file_location("fast_summary_v3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_pareto_front_respects_all_four_directions():
    mod = _load()
    rows = [
        {"id": "dominated", "event_recall": 0.7, "auprc": 0.4,
         "lead_minutes_median": 20.0, "false_alarm_segments_per_turbine_day": 0.2},
        {"id": "better_all", "event_recall": 0.8, "auprc": 0.5,
         "lead_minutes_median": 30.0, "false_alarm_segments_per_turbine_day": 0.1},
        {"id": "tradeoff", "event_recall": 0.9, "auprc": 0.3,
         "lead_minutes_median": 10.0, "false_alarm_segments_per_turbine_day": 0.05},
    ]
    front = mod.pareto_front(rows)
    assert {x["id"] for x in front} == {"better_all", "tradeoff"}


def test_missing_pareto_axis_is_excluded_not_imputed():
    mod = _load()
    rows = [{"id": "missing", "event_recall": 0.8, "auprc": 0.5,
             "lead_minutes_median": None, "false_alarm_segments_per_turbine_day": 0.1}]
    assert mod.pareto_front(rows) == []


def test_flatten_metric_record_preserves_status():
    mod = _load()
    rec = {
        "schema_version": "metrics-v3", "farm": "penmanshiel", "model": "m",
        "seed": 0, "label_mode": "real_fault_wl",
        "workpoints": {"balanced": {"test": {"metrics": {"f1": 0.5, "mse": None},
                                                  "metric_status": {"f1": "ok", "mse": "not_applicable"}}}},
    }
    wide, long = mod.flatten_record(rec, Path("x/metrics.json"))
    assert wide[0]["f1"] == 0.5
    assert {x["metric"]: x["status"] for x in long} == {"f1": "ok", "mse": "not_applicable"}
