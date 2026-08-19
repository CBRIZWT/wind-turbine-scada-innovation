from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from 实验工具 import record_and_print_metric


def test_record_metric_preserves_protocol_and_hyperparams(monkeypatch, tmp_path):
    env_values = {
        "SCADA_SPLIT_ID": "chronological_v2",
        "SCADA_FEATURE_VERSION": "v2",
        "SCADA_CACHE_VERSION": "v2",
        "SCADA_LR": "0.0001",
        "SCADA_SCHEDULER": "plateau",
        "SCADA_BATCH_SIZE": "64",
        "SCADA_GRID_ID": "grid_lr1e-04_bs64_plateau",
        "SCADA_SELECTED_BY": "grid_val_mean_f1",
        "SCADA_PREPROCESS_VARIANT": "old_preprocess",
    }
    for key, value in env_values.items():
        monkeypatch.setenv(key, value)

    jsonl = tmp_path / "metrics.jsonl"
    csv_path = tmp_path / "metrics.csv"
    record_and_print_metric(
        jsonl,
        csv_path,
        "TranAD",
        "val",
        1,
        {"farm": "kelmarsh", "module": "baseline_only", "seed": 0, "f1": 0.25},
    )

    row = next(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    json_row = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
    for field, expected in {
        "split_id": "chronological_v2",
        "feature_version": "v2",
        "cache_version": "v2",
        "lr": "0.0001",
        "scheduler": "plateau",
        "batch_size": "64",
        "grid_id": "grid_lr1e-04_bs64_plateau",
        "selected_by": "grid_val_mean_f1",
        "preprocess_variant": "old_preprocess",
    }.items():
        assert row[field] == expected
        assert str(json_row[field]) == expected
