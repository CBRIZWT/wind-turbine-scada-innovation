from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .adapters import ModelAdapter
from .calibration import select_equal4_calibration
from .contracts import DatasetBundle
from .full_metrics import evaluate_full_metrics


def _validation_hash(bundle: DatasetBundle) -> str:
    digest = hashlib.sha256()
    for value in (bundle.file_hash, bundle.split_hash, bundle.feature_hash, bundle.farm, "val"):
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def evaluate_adapter(
    bundle: DatasetBundle,
    adapter: ModelAdapter,
    *,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """严格顺序：fit -> val score/calibrate/freeze -> test score/evaluate once。"""

    kind = getattr(adapter, "required_train_kind", None)
    if kind not in {"normal", "supervised"}:
        raise TypeError("adapter 必须声明 required_train_kind")
    adapter.fit(bundle.train_view(kind), seed=seed, device=device)
    validation_scores = adapter.score(bundle.score_view("val"))
    calibration = select_equal4_calibration(
        bundle.evaluation_labels("val"), validation_scores,
        bundle.val_score.timestamps, bundle.val_score.turbines, bundle.event_table,
        model_id=str(getattr(adapter, "model_id", type(adapter).__name__)),
        dataset_id=f"{bundle.dataset}:{bundle.farm}:{bundle.variant}",
        validation_hash=_validation_hash(bundle),
    )
    # CalibrationArtifact 已深冻结；从此以后才允许触碰 test score/label。
    test_scores = adapter.score(bundle.score_view("test"))
    semantics = str(getattr(adapter, "score_semantics", "anomaly_score"))
    probabilities = None
    if semantics == "probability":
        probabilities = np.asarray(test_scores, dtype=float)
        if calibration.polarity == "negative":
            probabilities = 1.0 - probabilities
    metrics = evaluate_full_metrics(
        bundle.evaluation_labels("test"), test_scores,
        bundle.test_score.timestamps, bundle.test_score.turbines, bundle.event_table,
        threshold=calibration.threshold, polarity=calibration.polarity, split="test",
        score_semantics=semantics, probabilities=probabilities,
    )
    return {
        "adapter": adapter,
        "validation_scores": validation_scores,
        "calibration": calibration,
        "test_scores": test_scores,
        "metrics": metrics,
        "score_semantics": semantics,
    }


__all__ = ["evaluate_adapter"]
