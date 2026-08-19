# -*- coding: utf-8 -*-
"""WT-Transformer PyTorch 实验的统一评价工具。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def _import_project_metrics():
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from 实验工具 import (  # type: ignore
        augment_event_metrics,
        choose_threshold_and_polarity_by_validation,
        compute_binary_metrics,
        orient_scores,
    )

    return augment_event_metrics, choose_threshold_and_polarity_by_validation, compute_binary_metrics, orient_scores


def base_polarity(polarity: str) -> str:
    """项目 fallback 会返回 positive_fallback_uncalibrated 这类审计标记。"""
    if str(polarity).startswith("positive"):
        return "positive"
    if str(polarity).startswith("negative"):
        return "negative"
    return str(polarity)


def positive_residual_energy(y_true: Any, y_pred: Any) -> np.ndarray:
    residual = np.asarray(y_true, dtype=float).reshape(-1) - np.asarray(y_pred, dtype=float).reshape(-1)
    return np.maximum(0.0, residual) ** 2


def compute_metrics_from_scores(
    *,
    val_labels: Any,
    val_scores: Any,
    test_labels: Any,
    test_scores: Any,
    fallback_scores: Any,
    fallback_quantile: float = 0.99,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """只用 validation 选阈值/极性，再评价 test。"""
    (
        augment_event_metrics,
        choose_threshold_and_polarity_by_validation,
        compute_binary_metrics,
        orient_scores,
    ) = _import_project_metrics()

    threshold, threshold_source, polarity = choose_threshold_and_polarity_by_validation(
        val_labels,
        val_scores,
        fallback_scores,
        fallback_quantile=fallback_quantile,
    )
    oriented_test_scores = orient_scores(test_scores, base_polarity(polarity)).reshape(-1)
    preds = (oriented_test_scores > float(threshold)).astype(int)
    metrics = compute_binary_metrics(test_labels, scores=oriented_test_scores, threshold=float(threshold))
    metrics.update(
        {
            "threshold": float(threshold),
            "threshold_source": threshold_source,
            "score_polarity": polarity,
            "score_definition": "positive_residual_energy",
        }
    )
    metrics = augment_event_metrics(metrics, labels=np.asarray(test_labels).reshape(-1), preds=preds)
    return metrics, oriented_test_scores, preds


def compute_phase_metrics(
    *,
    labels: Any,
    scores: Any,
    threshold: float,
    polarity: str,
) -> Dict[str, Any]:
    augment_event_metrics, _, compute_binary_metrics, orient_scores = _import_project_metrics()
    oriented = orient_scores(scores, base_polarity(polarity)).reshape(-1)
    preds = (oriented > float(threshold)).astype(int)
    metrics = compute_binary_metrics(labels, scores=oriented, threshold=float(threshold))
    metrics.update({"threshold": float(threshold), "score_polarity": polarity})
    return augment_event_metrics(metrics, labels=np.asarray(labels).reshape(-1), preds=preds)
