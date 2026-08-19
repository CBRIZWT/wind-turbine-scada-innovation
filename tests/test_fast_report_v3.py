# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FAST = ROOT / "快速实验"
BASE = FAST / "基础模型"
for path in (ROOT, FAST, BASE):
    sys.path.insert(0, str(path))


def _write_bundle(data: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data.mkdir(parents=True)
    ts = pd.date_range("2024-01-01", periods=18, freq="10min", tz="UTC")
    ts_ns = ts.tz_convert(None).as_unit("ns").asi8
    turb = np.array(["A"] * len(ts_ns))
    y = np.zeros(len(ts_ns), dtype=np.int8)
    y[:6] = 1
    for split in ("val", "test"):
        np.save(data / f"timestamps_flat_{split}.npy", ts_ns)
        np.save(data / f"turbines_flat_{split}.npy", turb)
        np.save(data / f"y_flat_{split}.npy", y)
    pd.DataFrame(
        {
            "farm": ["kelmarsh", "kelmarsh"],
            "turbine": ["A", "A"],
            "start": ["2024-01-01T01:00:00Z"] * 2,
            "end": ["2024-01-01T01:10:00Z"] * 2,
            "split": ["val", "test"],
            "message": ["high temp. gear bearing"] * 2,
        }
    ).to_csv(data / "event_table.csv", index=False)
    (data / "meta.json").write_text(
        json.dumps({"label_mode": "real_fault_wl", "preprocess_variant": "real_fault_metrics_v1"}),
        encoding="utf-8",
    )
    scores = np.linspace(1.0, 0.0, len(y))
    return y, scores, ts_ns, turb


def test_report_v3_writes_three_workpoints_with_identical_metric_schema(tmp_path, monkeypatch):
    common = importlib.import_module("_common")
    from 统一评测 import metric_schema

    data = tmp_path / "data"
    result = tmp_path / "result"
    y, scores, _, _ = _write_bundle(data)
    monkeypatch.setattr(common, "DATA", data)
    monkeypatch.setattr(common, "RESULT", result)
    monkeypatch.setattr(common, "FARM", "kelmarsh")
    monkeypatch.setattr(common, "VARIANT", "real_fault_metrics_v1", raising=False)

    rec = common.report_v3("demo", y, scores, y, scores, 1.25, representation="flat")
    assert set(rec["workpoints"]) == {"balanced", "low_far", "high_recall"}
    expected = set(metric_schema())
    for wp in rec["workpoints"].values():
        assert set(wp["val"]["metrics"]) == expected
        assert set(wp["test"]["metrics"]) == expected
        assert set(wp["test"]["metric_status"]) == expected
    out = result / "demo"
    assert (out / "metrics.json").exists()
    assert (out / "score_val.npy").exists()
    assert (out / "score_test.npy").exists()
    assert (out / "pred_test_balanced.npy").exists()
    disk = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert disk["schema_version"] == "metrics-v3"


def test_report_v3_threshold_selection_is_independent_of_test_labels(tmp_path, monkeypatch):
    common = importlib.import_module("_common")

    data = tmp_path / "data"
    y, scores, _, _ = _write_bundle(data)
    monkeypatch.setattr(common, "DATA", data)
    monkeypatch.setattr(common, "FARM", "kelmarsh")
    monkeypatch.setattr(common, "VARIANT", "real_fault_metrics_v1", raising=False)

    monkeypatch.setattr(common, "RESULT", tmp_path / "r1")
    r1 = common.report_v3("demo", y, scores, y, scores, 0.1, representation="flat")
    monkeypatch.setattr(common, "RESULT", tmp_path / "r2")
    r2 = common.report_v3("demo", y, scores, 1 - y, scores, 0.1, representation="flat")
    assert r1["threshold_selection"] == r2["threshold_selection"]


def test_cross_farm_model_46_uses_the_other_true_fault_farm():
    common = importlib.import_module("_common")
    assert common.cross_farm_source("kelmarsh") == "penmanshiel"
    assert common.cross_farm_source("penmanshiel") == "kelmarsh"


def test_report_v3_can_use_source_val_events_and_target_test_events(tmp_path, monkeypatch):
    common = importlib.import_module("_common")
    data = tmp_path / "data"
    result = tmp_path / "result"
    y, scores, _, _ = _write_bundle(data)
    source_events = tmp_path / "source_events.csv"
    pd.DataFrame(
        {
            "farm": ["penmanshiel", "penmanshiel"],
            "turbine": ["A", "A"],
            "start": ["2024-01-01T01:00:00Z", "2024-01-05T01:00:00Z"],
            "end": ["2024-01-01T01:10:00Z", "2024-01-05T01:10:00Z"],
            "split": ["val", "val"],
            "message": ["high temp. gear bearing"] * 2,
        }
    ).to_csv(source_events, index=False)
    monkeypatch.setattr(common, "DATA", data)
    monkeypatch.setattr(common, "RESULT", result)
    monkeypatch.setattr(common, "FARM", "kelmarsh")
    monkeypatch.setattr(common, "VARIANT", "real_fault_metrics_v1", raising=False)
    rec = common.report_v3(
        "cross",
        y,
        scores,
        y,
        scores,
        0.2,
        representation="flat",
        val_event_table=source_events,
        test_event_table=data / "event_table.csv",
    )
    assert rec["workpoints"]["balanced"]["val"]["metrics"]["n_events"] == 2
    assert rec["workpoints"]["balanced"]["test"]["metrics"]["n_events"] == 1


def test_external_report_uses_train_quantiles_and_never_emits_label_performance(tmp_path, monkeypatch):
    common = importlib.import_module("_common")
    data = tmp_path / "data"
    result = tmp_path / "result"
    y, scores, _, _ = _write_bundle(data)
    (data / "meta.json").write_text(
        json.dumps(
            {
                "label_mode": "external_unlabeled",
                "preprocess_variant": "real_fault_metrics_v1_external_local",
                "external_unlabeled": True,
                "external_protocol": "hill_local_unlabeled_fit",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(common, "DATA", data)
    monkeypatch.setattr(common, "RESULT", result)
    monkeypatch.setattr(common, "FARM", "hill_of_towie")
    monkeypatch.setattr(common, "VARIANT", "real_fault_metrics_v1_external_local", raising=False)
    train_scores = np.linspace(0.0, 1.0, 101)
    rec = common.report_v3(
        "external",
        y,
        scores,
        y,
        scores,
        0.1,
        representation="flat",
        train_scores=train_scores,
    )
    assert set(rec["workpoints"]) == {"q99", "q995", "q999"}
    assert rec["threshold_selection"]["threshold_source"] == "train_score_quantiles"
    for wp in rec["workpoints"].values():
        assert wp["test"]["metrics"]["event_f1"] is None
        assert wp["test"]["metric_status"]["event_f1"] == "external_unlabeled"
        assert "alarm_segments_per_turbine_day" in wp["external_monitoring_test"]
