from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from realfault_benchmark.data import load_realfault_bundle, merged_events_for_split


def _write_fixture(root: Path, *, primary_files: list[str] | None = None) -> Path:
    data = root / "kelmarsh__realfault"
    data.mkdir()
    meta = {
        "farm": "kelmarsh",
        "preprocess_variant": "realfault",
        "split_hash": "split",
        "cols_hash": "features",
        "cols": ["temp_resid", "temp_trend"],
        "primary_label": {
            "name": "real_fault_wl",
            "source_files": primary_files or ["train_labels.npy", "val_labels.npy", "test_labels.npy"],
        },
    }
    (data / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    arrays = {
        "train.npy": np.array([[1, 1], [2, 2]], dtype=np.float32),
        "train_labels.npy": np.array([0, 0], dtype=np.int8),
        "timestamps_train.npy": np.array([10, 30], dtype=np.int64),
        "turbines_train.npy": np.array(["A", "A"]),
        "train_sup.npy": np.array([[1, 1], [9, 9], [2, 2], [8, 8]], dtype=np.float32),
        "train_sup_labels.npy": np.array([0, 1, 0, -1], dtype=np.int8),
        "timestamps_train_sup.npy": np.array([10, 20, 30, 40], dtype=np.int64),
        "turbines_train_sup.npy": np.array(["A", "A", "A", "A"]),
        "gap_mask_train.npy": np.array([False, False, False, True]),
        "val.npy": np.array([[3, 3], [4, 4]], dtype=np.float32),
        "val_labels.npy": np.array([0, 1], dtype=np.int8),
        "timestamps_val.npy": np.array([50, 60], dtype=np.int64),
        "turbines_val.npy": np.array(["A", "A"]),
        "gap_mask_val.npy": np.array([False, False]),
        "test.npy": np.array([[5, 5], [6, 6]], dtype=np.float32),
        "test_labels.npy": np.array([0, 1], dtype=np.int8),
        "timestamps_test.npy": np.array([70, 80], dtype=np.int64),
        "turbines_test.npy": np.array(["A", "A"]),
        "gap_mask_test.npy": np.array([False, False]),
        # 污染侧车故意相反；严格加载器不得自动发现/使用。
        "val_labels_v2.npy": np.array([1, 0], dtype=np.int8),
        "test_labels_v2.npy": np.array([1, 0], dtype=np.int8),
    }
    for name, value in arrays.items():
        np.save(data / name, value)
    pd.DataFrame(
        [
            {"farm": "kelmarsh", "turbine": "A", "start": "2022-01-01T00:00:00Z",
             "end": "2022-01-01T01:00:00Z", "split": "val", "message": "x"},
            {"farm": "kelmarsh", "turbine": "A", "start": "2022-01-02T00:00:00Z",
             "end": "2022-01-02T01:00:00Z", "split": "val", "message": "y"},
        ]
    ).to_csv(data / "event_table.csv", index=False)
    return data


def test_loader_explicitly_uses_v1_and_preserves_source_row_ids(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    bundle = load_realfault_bundle(tmp_path, "kelmarsh", verify_file_hashes=False)
    assert bundle.evaluation_labels("val").tolist() == [0, 1]
    assert bundle.evaluation_labels("test").tolist() == [0, 1]
    assert bundle.train_normal.row_ids.tolist() == [0, 2]
    assert bundle.train_supervised.row_ids.tolist() == [0, 1, 2]
    assert bundle.train_normal.gap_mask.tolist() == [False, False]
    assert bundle.val_score.gap_mask.tolist() == [False, False]
    assert bundle.test_score.gap_mask.tolist() == [False, False]
    assert bundle.variant == "realfault"


def test_loader_rejects_meta_that_points_to_v2(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        primary_files=["train_sup_labels_v2.npy", "val_labels_v2.npy", "test_labels_v2.npy"],
    )
    with pytest.raises(ValueError, match="v1"):
        load_realfault_bundle(tmp_path, "kelmarsh", verify_file_hashes=False)


def test_event_rows_merge_within_72_hours_per_turbine(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    bundle = load_realfault_bundle(tmp_path, "kelmarsh", verify_file_hashes=False)
    merged = merged_events_for_split(bundle.event_table, "val", merge_hours=72.0)
    assert len(merged) == 1
    assert merged.iloc[0]["turbine"] == "A"
    assert str(merged.iloc[0]["start"]).startswith("2022-01-01")
    assert str(merged.iloc[0]["end"]).startswith("2022-01-02")
