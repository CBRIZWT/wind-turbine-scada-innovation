from __future__ import annotations

import numpy as np
import pytest

from realfault_benchmark.adapters import ModelAdapter
from realfault_benchmark.contracts import ScoreView, TrainView
from realfault_benchmark.sequences import SequenceSpec, build_sequence_index, extract_windows


class _NormalAdapter(ModelAdapter):
    required_train_kind = "normal"

    def _fit(self, train_view: TrainView, seed: int, device: str) -> None:
        return None

    def _score(self, score_view: ScoreView) -> np.ndarray:
        return np.zeros(len(score_view.X))


class _SupervisedAdapter(_NormalAdapter):
    required_train_kind = "supervised"


def _train(kind: str, labels: np.ndarray) -> TrainView:
    return TrainView(
        X=np.zeros((len(labels), 1), dtype=np.float32),
        labels=labels,
        timestamps=np.arange(len(labels), dtype=np.int64),
        turbines=np.array(["A"] * len(labels)),
        row_ids=np.arange(len(labels), dtype=np.int64),
        kind=kind,  # type: ignore[arg-type]
    )


def test_adapter_enforces_declared_train_track_and_revalidates_labels() -> None:
    with pytest.raises(ValueError, match="normal"):
        _NormalAdapter().fit(_train("supervised", np.array([0, 1], dtype=np.int8)), 20260719, "cpu")
    with pytest.raises(ValueError, match="全零"):
        _NormalAdapter().fit(_train("normal", np.array([0, 1], dtype=np.int8)), 20260719, "cpu")
    with pytest.raises(ValueError, match="ignore"):
        _SupervisedAdapter().fit(_train("supervised", np.array([0, -1], dtype=np.int8)), 20260719, "cpu")


def test_adapter_revalidates_all_train_sidecars_and_unique_row_ids() -> None:
    valid = _train("normal", np.array([0, 0], dtype=np.int8))
    with pytest.raises(ValueError, match="二维"):
        _NormalAdapter().fit(
            TrainView(valid.X.ravel(), valid.labels, valid.timestamps, valid.turbines, valid.row_ids, "normal"),
            20260719, "cpu",
        )
    with pytest.raises(ValueError, match="等长|对齐"):
        _NormalAdapter().fit(
            TrainView(valid.X, valid.labels, valid.timestamps[:1], valid.turbines, valid.row_ids, "normal"),
            20260719, "cpu",
        )
    with pytest.raises(ValueError, match="row_id"):
        _NormalAdapter().fit(
            TrainView(valid.X, valid.labels, valid.timestamps, valid.turbines, np.array([1, 1]), "normal"),
            20260719, "cpu",
        )


def test_failed_refit_clears_previous_fit_provenance() -> None:
    adapter = _NormalAdapter().fit(_train("normal", np.array([0, 0], dtype=np.int8)), 20260719, "cpu")
    with pytest.raises(ValueError):
        adapter.fit(_train("normal", np.array([0, 1], dtype=np.int8)), 7, "cuda")
    assert not adapter.is_fitted
    assert adapter.fit_seed is None
    assert adapter.fit_device is None


def test_sequence_gap_mask_is_a_physical_boundary_independent_of_labels() -> None:
    view = ScoreView(
        X=np.arange(10, dtype=np.float32).reshape(5, 2),
        timestamps=np.arange(5, dtype=np.int64) * 10,
        turbines=np.array(["A"] * 5),
        row_ids=np.arange(5, dtype=np.int64),
        split="test",
        gap_mask=np.array([False, False, True, False, False]),
    )
    index = build_sequence_index(view, SequenceSpec(window_size=2, cadence=10, max_gap=10))
    assert index.window_positions.tolist() == [[0, 1], [3, 4]]


def test_old_sequence_index_rejects_any_changed_sidecar_or_feature_content() -> None:
    original = ScoreView(
        X=np.arange(8, dtype=np.float32).reshape(4, 2),
        timestamps=np.arange(4, dtype=np.int64) * 10,
        turbines=np.array(["A"] * 4),
        row_ids=np.arange(4, dtype=np.int64),
        split="val",
        gap_mask=np.zeros(4, dtype=bool),
    )
    index = build_sequence_index(original, SequenceSpec(window_size=2, cadence=10))

    changed_feature = original.X.copy()
    changed_feature[0, 0] = -999
    changed = ScoreView(
        X=changed_feature,
        timestamps=original.timestamps,
        turbines=original.turbines,
        row_ids=original.row_ids,
        split="val",
        gap_mask=original.gap_mask,
    )
    with pytest.raises(ValueError, match="hash|指纹"):
        extract_windows(changed, index)

    changed_turbine = ScoreView(
        X=original.X,
        timestamps=original.timestamps,
        turbines=np.array(["A", "B", "A", "A"]),
        row_ids=original.row_ids,
        split="val",
        gap_mask=original.gap_mask,
    )
    with pytest.raises(ValueError, match="hash|指纹"):
        extract_windows(changed_turbine, index)
