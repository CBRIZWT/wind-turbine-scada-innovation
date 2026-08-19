from __future__ import annotations

import numpy as np
import pytest

from realfault_benchmark.contracts import CalibrationArtifact, DatasetBundle


def _bundle() -> DatasetBundle:
    return DatasetBundle.from_arrays(
        dataset="synthetic",
        farm="farm_a",
        variant="realfault",
        feature_names=("temperature_residual", "trend"),
        train_normal_X=np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32),
        train_normal_labels=np.array([0, 0], dtype=np.int8),
        train_normal_timestamps=np.array([1, 2], dtype=np.int64),
        train_normal_turbines=np.array(["A", "A"]),
        train_normal_row_ids=np.array([10, 11], dtype=np.int64),
        train_normal_gap_mask=np.array([False, False]),
        train_supervised_X=np.array([[0.0, 0.1], [2.0, 2.1], [9.0, 9.1]], dtype=np.float32),
        train_supervised_labels=np.array([0, 1, -1], dtype=np.int8),
        train_supervised_timestamps=np.array([1, 2, 3], dtype=np.int64),
        train_supervised_turbines=np.array(["A", "A", "A"]),
        train_supervised_row_ids=np.array([20, 21, 22], dtype=np.int64),
        train_supervised_gap_mask=np.array([False, False, True]),
        val_X=np.array([[0.4, 0.5], [2.2, 2.3]], dtype=np.float32),
        val_labels=np.array([0, 1], dtype=np.int8),
        val_timestamps=np.array([4, 5], dtype=np.int64),
        val_turbines=np.array(["A", "A"]),
        val_row_ids=np.array([30, 31], dtype=np.int64),
        val_gap_mask=np.array([False, False]),
        test_X=np.array([[0.6, 0.7], [2.4, 2.5]], dtype=np.float32),
        test_labels=np.array([0, 1], dtype=np.int8),
        test_timestamps=np.array([6, 7], dtype=np.int64),
        test_turbines=np.array(["A", "A"]),
        test_row_ids=np.array([40, 41], dtype=np.int64),
        test_gap_mask=np.array([False, False]),
        event_table=(),
        split_hash="split-hash",
        feature_hash="feature-hash",
        file_hash="file-hash",
    )


def test_score_view_never_exposes_labels() -> None:
    bundle = _bundle()
    score_view = bundle.score_view("test")
    assert not hasattr(score_view, "labels")
    assert not hasattr(score_view, "y")
    assert np.array_equal(score_view.row_ids, [40, 41])


def test_training_views_enforce_label_contracts() -> None:
    bundle = _bundle()
    normal = bundle.train_view("normal")
    supervised = bundle.train_view("supervised")
    assert np.all(normal.labels == 0)
    assert set(supervised.labels.tolist()) == {0, 1}
    assert -1 not in supervised.labels
    assert supervised.row_ids.tolist() == [20, 21]


def test_normal_training_rejects_positive_or_ignore_labels() -> None:
    kwargs = _bundle().to_array_kwargs()
    kwargs["train_normal_labels"] = np.array([0, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="train_normal"):
        DatasetBundle.from_arrays(**kwargs)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("val_labels", np.array([0.0, 1.9], dtype=float)),
        ("test_labels", np.array([[0, 1]], dtype=np.int8)),
        ("train_supervised_labels", np.array([0.0, 1.0], dtype=float)),
    ],
)
def test_labels_are_validated_before_any_int8_cast(field: str, bad: np.ndarray) -> None:
    kwargs = _bundle().to_array_kwargs()
    kwargs[field] = bad
    with pytest.raises(ValueError, match="标签|一维|整数"):
        DatasetBundle.from_arrays(**kwargs)


def test_physical_gap_masks_are_mandatory() -> None:
    kwargs = _bundle().to_array_kwargs()
    kwargs.pop("test_gap_mask")
    with pytest.raises(TypeError, match="gap"):
        DatasetBundle.from_arrays(**kwargs)


def test_calibration_metrics_are_deeply_frozen_and_export_as_plain_dict() -> None:
    artifact = CalibrationArtifact(
        model_id="m", dataset_id="d", validation_hash="v", score_hash="s",
        polarity="positive", threshold=0.5, threshold_source="validation",
        candidate_count=2, validation_metrics={"nested": {"score": 0.8}},
        artifact_hash="h",
    )
    with pytest.raises(TypeError):
        artifact.validation_metrics["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        artifact.validation_metrics["nested"]["score"] = 0.0  # type: ignore[index]
    assert artifact.to_dict()["validation_metrics"] == {"nested": {"score": 0.8}}
