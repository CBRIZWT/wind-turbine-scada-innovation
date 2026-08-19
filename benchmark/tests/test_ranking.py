from __future__ import annotations

import pandas as pd

from realfault_benchmark.ranking import build_true_fault_leaderboard


def test_only_models_successful_on_both_farms_enter_leaderboard() -> None:
    rows = pd.DataFrame(
        [
            {"model_id": "A", "farm": "kelmarsh", "status": "success", "local_equal4_score": 0.7,
             "pr_auc": 0.3, "publication_date": "2025-01-01", "seed": 20260719,
             "protocol_hash": "protocol-v1", "variant": "realfault", "calibration_split": "val",
             "data_hash": "data-kelmarsh", "calibration_hash": "cal-A-kelmarsh"},
            {"model_id": "A", "farm": "penmanshiel", "status": "success", "local_equal4_score": 0.5,
             "pr_auc": 0.5, "publication_date": "2025-01-01", "seed": 20260719,
             "protocol_hash": "protocol-v1", "variant": "realfault", "calibration_split": "val",
             "data_hash": "data-penmanshiel", "calibration_hash": "cal-A-penmanshiel"},
            {"model_id": "B", "farm": "kelmarsh", "status": "success", "local_equal4_score": 0.9,
             "pr_auc": 0.9, "publication_date": "2026-01-01", "seed": 20260719,
             "protocol_hash": "protocol-v1", "variant": "realfault", "calibration_split": "val",
             "data_hash": "data-kelmarsh", "calibration_hash": "cal-B-kelmarsh"},
            {"model_id": "B", "farm": "penmanshiel", "status": "failed", "local_equal4_score": None,
             "pr_auc": None, "publication_date": "2026-01-01", "seed": 20260719,
             "protocol_hash": "protocol-v1", "variant": "realfault", "calibration_split": "val",
             "data_hash": "data-penmanshiel", "calibration_hash": "cal-B-penmanshiel"},
        ]
    )
    board = build_true_fault_leaderboard(rows)
    assert board["model_id"].tolist() == ["A"]
    assert board.iloc[0]["macro_equal4_score"] == 0.6
    assert board.iloc[0]["worst_farm_equal4_score"] == 0.5
