from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from realfault_benchmark.artifacts import strict_json_dumps
from realfault_benchmark.metrics import point_metrics
from realfault_benchmark.ranking import build_true_fault_leaderboard


def _row(
    model_id: str,
    farm: str,
    *,
    seed: int = 20260719,
    status: str = "success",
    score: float = 0.6,
    protocol_hash: str = "protocol-v1",
    variant: str = "realfault",
    calibration_split: str = "val",
    data_hash: str | None = None,
    calibration_hash: str | None = None,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "farm": farm,
        "status": status,
        "seed": seed,
        "protocol_hash": protocol_hash,
        "variant": variant,
        "calibration_split": calibration_split,
        "data_hash": f"data-{farm}" if data_hash is None else data_hash,
        "calibration_hash": f"cal-{model_id}-{farm}" if calibration_hash is None else calibration_hash,
        "local_equal4_score": score,
        "pr_auc": 0.4,
        "publication_date": "2025-01-01",
    }


def test_strict_json_recursively_sanitizes_numpy_values_and_records_status() -> None:
    payload = {
        "ok": np.bool_(True),
        "runs": [
            {
                "metrics": {"roc_auc": np.float64(np.nan), "count": np.int64(3)},
                "trace": [np.float32(0.25), np.float64(np.inf)],
            }
        ],
    }

    text = strict_json_dumps(payload)
    decoded = json.loads(text)

    assert decoded["ok"] is True
    assert decoded["runs"][0]["metrics"]["count"] == 3
    assert decoded["runs"][0]["metrics"]["roc_auc"] is None
    assert decoded["runs"][0]["trace"] == [0.25, None]
    assert decoded["runs"][0]["metric_status"]["roc_auc"] == "non_finite"
    assert decoded["runs"][0]["serialization_status"] == {
        "metrics.roc_auc": "non_finite",
        "trace[1]": "non_finite",
    }
    assert decoded["runs"][0]["metrics"]["serialization_status"] == {
        "roc_auc": "non_finite"
    }
    assert "NaN" not in text and "Infinity" not in text


def test_strict_leaderboard_keeps_only_auditable_two_farm_records_and_reasons() -> None:
    rows = pd.DataFrame(
        [
            _row("valid", "kelmarsh", score=0.7),
            _row("valid", "penmanshiel", score=0.5),
            _row("wrong-seed", "kelmarsh", seed=0),
            _row("wrong-seed", "penmanshiel", seed=0),
            _row("bad-score", "kelmarsh", score=1.01),
            _row("bad-score", "penmanshiel", score=0.5),
            _row("mixed-protocol", "kelmarsh", protocol_hash="protocol-v1"),
            _row("mixed-protocol", "penmanshiel", protocol_hash="protocol-v2"),
            _row("missing-hash", "kelmarsh", calibration_hash=""),
            _row("missing-hash", "penmanshiel"),
        ]
    )

    board, excluded = build_true_fault_leaderboard(rows, return_exclusions=True)

    assert board["model_id"].tolist() == ["valid"]
    reasons = excluded.groupby("model_id")["reason_code"].apply(set).to_dict()
    assert "seed_not_20260719" in reasons["wrong-seed"]
    assert "invalid_local_equal4_score" in reasons["bad-score"]
    assert "incompatible_protocol_hash" in reasons["mixed-protocol"]
    assert "missing_calibration_hash" in reasons["missing-hash"]
    assert board.attrs["exclusions"] == excluded.to_dict(orient="records")


def test_strict_leaderboard_excludes_missing_audit_schema_with_reason() -> None:
    rows = pd.DataFrame(
        [
            {
                "model_id": "legacy",
                "farm": "kelmarsh",
                "status": "success",
                "local_equal4_score": 0.7,
                "pr_auc": 0.4,
                "publication_date": "2025-01-01",
            },
            {
                "model_id": "legacy",
                "farm": "penmanshiel",
                "status": "success",
                "local_equal4_score": 0.5,
                "pr_auc": 0.4,
                "publication_date": "2025-01-01",
            },
        ]
    )

    board, excluded = build_true_fault_leaderboard(rows, return_exclusions=True)

    assert board.empty
    assert excluded.loc[0, "model_id"] == "legacy"
    assert excluded.loc[0, "reason_code"] == "missing_required_fields"
    assert "seed" in excluded.loc[0, "reason_detail"]


def test_point_metrics_does_not_treat_bounded_anomaly_scores_as_probabilities() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    scores = np.array([0.1, 0.3, 0.7, 0.9])
    predictions = np.array([0, 0, 1, 1], dtype=np.int8)

    got = point_metrics(
        labels,
        scores,
        predictions,
        score_semantics="anomaly_score",
        probabilities=None,
    )

    assert got["log_loss"] is None
    assert got["brier"] is None
    assert got["probability_status"] == "invalid_probability"


def test_point_metrics_computes_probability_losses_only_from_explicit_probabilities() -> None:
    labels = np.array([0, -1, 1, 1], dtype=np.int8)
    scores = np.array([-10.0, 100.0, 2.0, 5.0])
    predictions = np.array([0, 1, 1, 1], dtype=np.int8)
    probabilities = np.array([0.1, 0.99, 0.8, 0.7])

    got = point_metrics(
        labels,
        scores,
        predictions,
        score_semantics="probability",
        probabilities=probabilities,
    )

    expected_y = np.array([0, 1, 1], dtype=np.int8)
    expected_p = np.array([0.1, 0.8, 0.7])
    assert np.isclose(got["log_loss"], log_loss(expected_y, expected_p, labels=[0, 1]))
    assert np.isclose(got["brier"], brier_score_loss(expected_y, expected_p))
    assert got["probability_status"] == "valid_probability"


def test_point_metrics_rejects_invalid_explicit_probabilities_without_rescaling() -> None:
    labels = np.array([0, 1], dtype=np.int8)
    raw_scores = np.array([-4.0, 9.0])
    predictions = np.array([0, 1], dtype=np.int8)

    got = point_metrics(
        labels,
        raw_scores,
        predictions,
        score_semantics="probability",
        probabilities=raw_scores,
    )

    assert got["log_loss"] is None
    assert got["brier"] is None
    assert got["probability_status"] == "invalid_probability"
