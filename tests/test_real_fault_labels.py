# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SCADA数据集"))

import 数据预处理 as prep  # noqa: E402


def _idx(n=24, start="2021-01-01"):
    return pd.DatetimeIndex(pd.date_range(start, periods=n, freq="10min", tz="UTC"))


def test_real_fault_policy_masks_temp_and_broad_events():
    ev = pd.DataFrame({
        "Timestamp start": [
            pd.Timestamp("2021-01-01 00:10", tz="UTC"),
            pd.Timestamp("2021-01-01 00:20", tz="UTC"),
            pd.Timestamp("2021-01-01 00:30", tz="UTC"),
            pd.Timestamp("2021-01-01 00:40", tz="UTC"),
        ],
        "IEC category": ["Forced Outage", "", "", ""],
        "Stopping": [0, 1, 0, 0],
        "Message": [
            "Grid loss forced outage",
            "Emergency stop",
            "High temp. gear bearing 1",
            "Manual stop on site",
        ],
    })

    assert hasattr(prep, "real_fault_event_mask")
    temp = prep.real_fault_event_mask(ev, policy="real_fault_temp")
    broad = prep.real_fault_event_mask(ev, policy="real_fault_broad")

    assert temp.tolist() == [False, False, True, False]
    assert broad.tolist() == [True, True, True, False]


def test_real_fault_earlywarning_labels_use_lead_window_and_ignore_event_body():
    ev = pd.DataFrame({
        "Timestamp start": [pd.Timestamp("2021-01-01 01:00", tz="UTC")],
        "Timestamp end": [pd.Timestamp("2021-01-01 01:20", tz="UTC")],
        "Message": ["High temperature gear bearing"],
    })

    assert hasattr(prep, "make_real_fault_earlywarning_labels")
    y, intervals = prep.make_real_fault_earlywarning_labels(
        ev, _idx(18), policy="real_fault_temp", lead_steps=3, return_intervals=True,
    )

    assert y[3:6].tolist() == [1, 1, 1]
    assert y[6:9].tolist() == [-1, -1, -1]
    assert int((y == 1).sum()) == 3
    assert intervals.tolist() == [[6, 9]]


def test_real_fault_earlywarning_labels_keep_turbines_isolated():
    base = _idx(12)
    idx = pd.DatetimeIndex(base).append(pd.DatetimeIndex(base))
    turbines = np.array(["T01"] * 12 + ["T02"] * 12)
    ev = pd.DataFrame({
        "Timestamp start": [pd.Timestamp("2021-01-01 01:00", tz="UTC")],
        "Timestamp end": [pd.Timestamp("2021-01-01 01:10", tz="UTC")],
        "Message": ["High temperature gear bearing"],
        "_turbine": ["T01"],
    })

    assert hasattr(prep, "make_real_fault_earlywarning_labels")
    y = prep.make_real_fault_earlywarning_labels(
        ev, idx, policy="real_fault_temp", lead_steps=2, turbine_col=turbines,
    )

    assert int((y[:12] == 1).sum()) == 2
    assert int((y[:12] == -1).sum()) == 2
    assert int((y[12:] == 1).sum()) == 0
    assert int((y[12:] == -1).sum()) == 0


def test_build_real_fault_event_table_records_auditable_fields():
    ev = pd.DataFrame({
        "Timestamp start": [pd.Timestamp("2021-01-01 01:00", tz="UTC")],
        "Timestamp end": [pd.Timestamp("2021-01-01 01:20", tz="UTC")],
        "IEC category": ["Technical Standby"],
        "Message": ["High temp. gear bearing 1"],
        "_turbine": ["T06"],
    })

    assert hasattr(prep, "build_real_fault_event_table")
    table = prep.build_real_fault_event_table(ev, farm="kelmarsh", split="val", policy="real_fault_temp")

    assert list(table.columns) == [
        "farm", "turbine", "start", "end", "split", "source",
        "category", "message", "policy",
    ]
    assert table.shape[0] == 1
    assert table.loc[0, "farm"] == "kelmarsh"
    assert table.loc[0, "turbine"] == "T06"
    assert table.loc[0, "policy"] == "real_fault_temp"


def test_real_fault_lead_candidates_are_fixed_protocol_values():
    assert getattr(prep, "REAL_FAULT_LEAD_CANDIDATES", None) == (6, 12, 24, 48, 72)
