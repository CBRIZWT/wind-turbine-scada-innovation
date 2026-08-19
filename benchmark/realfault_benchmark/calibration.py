from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from typing import Any

import numpy as np

from .contracts import CalibrationArtifact
from .metrics import evaluate_equal4, prepare_equal4_context


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode("ascii"))
    h.update(str(value.shape).encode("ascii"))
    h.update(value.view(np.uint8))
    return h.hexdigest()


def select_equal4_calibration(
    val_labels: np.ndarray,
    val_scores: np.ndarray,
    timestamps: np.ndarray,
    turbines: np.ndarray,
    events: Any,
    *,
    model_id: str,
    dataset_id: str,
    validation_hash: str,
    horizon: timedelta = timedelta(hours=12),
    merge_gap: timedelta = timedelta(minutes=40),
    nominal_cadence: timedelta = timedelta(minutes=10),
) -> CalibrationArtifact:
    """仅使用 validation 的 201 个分位点与 above-max 阈值冻结校准。"""
    labels = np.asarray(val_labels, dtype=np.int8)
    raw = np.asarray(val_scores, dtype=float)
    ts = np.asarray(timestamps, dtype=np.int64)
    turb = np.asarray(turbines).astype(str)
    if not (len(labels) == len(raw) == len(ts) == len(turb)):
        raise ValueError("validation labels/scores/timestamps/turbines 必须等长")
    valid = (labels != -1) & np.isfinite(raw)
    if not valid.any():
        raise ValueError("validation 没有有限且非 ignore 的分数")
    prepared = prepare_equal4_context(labels, ts, turb, events, split="val", horizon=horizon)
    quantiles = np.linspace(0.0, 0.9995, 201)
    candidates: list[tuple[tuple[float, ...], str, float, dict[str, Any]]] = []
    candidate_count = 0
    for polarity in ("positive", "negative"):
        oriented = raw if polarity == "positive" else -raw
        base = oriented[valid]
        thresholds = np.unique(
            np.concatenate([np.quantile(base, quantiles), [np.nextafter(float(base.max()), np.inf)]])
        )
        candidate_count += len(thresholds)
        for threshold in thresholds:
            metrics = evaluate_equal4(
                labels, raw, ts, turb, events, threshold=float(threshold), polarity=polarity,
                horizon=horizon, merge_gap=merge_gap, nominal_cadence=nominal_cadence,
                split="val", prepared=prepared, include_point_metrics=False,
            )
            key = (
                float(metrics["local_equal4_score"]),
                float(metrics["alarm_point_precision"]),
                -float(metrics["false_alarm_segments_per_turbine_day"]),
                float(metrics["event_recall"]),
                float(threshold),
                1.0 if polarity == "positive" else 0.0,
            )
            candidates.append((key, polarity, float(threshold), metrics))
    _, polarity, threshold, _ = max(candidates, key=lambda item: item[0])
    metrics = evaluate_equal4(
        labels, raw, ts, turb, events, threshold=threshold, polarity=polarity,
        horizon=horizon, merge_gap=merge_gap, nominal_cadence=nominal_cadence,
        split="val", prepared=prepared, include_point_metrics=True,
    )
    score_hash = _array_hash(raw)
    core = {
        "model_id": str(model_id), "dataset_id": str(dataset_id),
        "validation_hash": str(validation_hash), "score_hash": score_hash,
        "polarity": polarity, "threshold": threshold,
        "threshold_source": "validation_equal4_201_quantiles",
        "candidate_count": int(candidate_count), "validation_metrics": metrics,
    }
    artifact_hash = hashlib.sha256(
        json.dumps(core, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return CalibrationArtifact(**core, artifact_hash=artifact_hash)
