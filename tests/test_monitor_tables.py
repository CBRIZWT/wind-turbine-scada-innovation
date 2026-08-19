from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import 实验监控


def test_generate_metric_tables_writes_epoch_final_and_hparam_tables(tmp_path):
    metrics_csv = tmp_path / "metrics.csv"
    state_dir = tmp_path / "state"
    out_dir = tmp_path / "tables"
    state_dir.mkdir()
    (state_dir / "best_config.json").write_text(
        json.dumps(
            {
                "lr": 1e-4,
                "batch_size": 64,
                "scheduler": "plateau",
                "selected_by": "grid_val_mean_f1",
            }
        ),
        encoding="utf-8",
    )
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "farm",
                "module",
                "seed",
                "phase",
                "epoch",
                "precision",
                "recall",
                "f1",
                "lr",
                "batch_size",
                "scheduler",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "TranAD",
                "farm": "kelmarsh",
                "module": "baseline_only",
                "seed": "0",
                "phase": "val",
                "epoch": "1",
                "precision": "0.8",
                "recall": "0.5",
                "f1": "0.6",
                "lr": "0.0001",
                "batch_size": "64",
                "scheduler": "plateau",
            }
        )
        writer.writerow(
            {
                "model": "TranAD",
                "farm": "kelmarsh",
                "module": "baseline_only",
                "seed": "0",
                "phase": "test",
                "epoch": "final",
                "precision": "0.7",
                "recall": "0.4",
                "f1": "0.5",
                "lr": "0.0001",
                "batch_size": "64",
                "scheduler": "plateau",
            }
        )

    count = 实验监控.generate_metric_tables(
        metrics_csv=str(metrics_csv),
        state_dir=str(state_dir),
        output_dir=str(out_dir),
    )

    assert count == 4
    for name in (
        "all_metrics_by_epoch.csv",
        "mean_by_epoch.csv",
        "final_test_mean_std.csv",
        "hyperparameter_selection.csv",
    ):
        assert (out_dir / name).exists()

    final_rows = list(csv.DictReader((out_dir / "final_test_mean_std.csv").open("r", encoding="utf-8")))
    assert final_rows[0]["model"] == "TranAD"
    assert final_rows[0]["f1_mean"] == "0.500000"

    hparam_rows = list(csv.DictReader((out_dir / "hyperparameter_selection.csv").open("r", encoding="utf-8")))
    assert {row["parameter"]: row["after"] for row in hparam_rows} == {
        "lr": "0.0001",
        "batch_size": "64",
        "scheduler": "plateau",
    }
