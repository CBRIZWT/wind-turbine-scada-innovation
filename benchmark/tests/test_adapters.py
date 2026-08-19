from __future__ import annotations

import inspect

import numpy as np
import pytest

from realfault_benchmark.adapters import ModelAdapter
from realfault_benchmark.contracts import ScoreView, TrainView


class _ToyAdapter(ModelAdapter):
    required_train_kind = "normal"

    def _fit(self, train_view: TrainView, seed: int, device: str) -> None:
        self.fit_call = (train_view.kind, seed, device)

    def _score(self, score_view: ScoreView) -> np.ndarray:
        return np.arange(len(score_view.X), dtype=np.float64)


class _ShortScoreAdapter(_ToyAdapter):
    def _score(self, score_view: ScoreView) -> np.ndarray:
        return np.zeros(max(0, len(score_view.X) - 1), dtype=np.float64)


def _train_view() -> TrainView:
    return TrainView(
        X=np.zeros((2, 1), dtype=np.float32),
        labels=np.zeros(2, dtype=np.int8),
        timestamps=np.array([0, 10], dtype=np.int64),
        turbines=np.array(["A", "A"]),
        row_ids=np.array([0, 1], dtype=np.int64),
        kind="normal",
    )


def _score_view() -> ScoreView:
    return ScoreView(
        X=np.zeros((3, 1), dtype=np.float32),
        timestamps=np.array([0, 10, 20], dtype=np.int64),
        turbines=np.array(["A", "A", "A"]),
        row_ids=np.array([10, 11, 12], dtype=np.int64),
        split="test",
    )


def test_model_adapter_exposes_fixed_fit_and_score_interfaces() -> None:
    assert list(inspect.signature(ModelAdapter.fit).parameters) == [
        "self", "train_view", "seed", "device"
    ]
    assert list(inspect.signature(ModelAdapter.score).parameters) == ["self", "score_view"]

    adapter = _ToyAdapter().fit(_train_view(), seed=20260719, device="cpu")
    assert adapter.fit_call == ("normal", 20260719, "cpu")
    scores = adapter.score(_score_view())
    np.testing.assert_array_equal(scores, [0.0, 1.0, 2.0])
    assert not scores.flags.writeable


def test_model_adapter_requires_fit_before_score() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        _ToyAdapter().score(_score_view())


def test_model_adapter_rejects_non_row_aligned_scores() -> None:
    adapter = _ShortScoreAdapter().fit(_train_view(), seed=20260719, device="cpu")
    with pytest.raises(ValueError, match="行对齐"):
        adapter.score(_score_view())


@pytest.mark.parametrize("bad_seed", [True, 1.5, "20260719"])
def test_model_adapter_rejects_non_integer_seed(bad_seed: object) -> None:
    with pytest.raises(TypeError, match="seed"):
        _ToyAdapter().fit(_train_view(), seed=bad_seed, device="cpu")  # type: ignore[arg-type]


def test_model_adapter_allows_nan_warmup_but_rejects_infinity() -> None:
    class _WarmupAdapter(_ToyAdapter):
        def _score(self, score_view: ScoreView) -> np.ndarray:
            return np.array([np.nan, 0.1, np.inf])

    adapter = _WarmupAdapter().fit(_train_view(), seed=20260719, device="cpu")
    with pytest.raises(ValueError, match="infinity"):
        adapter.score(_score_view())


def test_model_adapter_revalidates_normal_labels_even_for_direct_train_view() -> None:
    malicious = _train_view()
    malicious = TrainView(
        X=malicious.X,
        labels=np.array([0, 1], dtype=np.int8),
        timestamps=malicious.timestamps,
        turbines=malicious.turbines,
        row_ids=malicious.row_ids,
        kind="normal",
    )

    with pytest.raises(ValueError, match="全零"):
        _ToyAdapter().fit(malicious, seed=20260719, device="cpu")


def test_model_adapter_enforces_declared_train_kind_and_supervised_no_ignore() -> None:
    class _SupervisedAdapter(_ToyAdapter):
        required_train_kind = "supervised"

    with pytest.raises(ValueError, match="supervised"):
        _SupervisedAdapter().fit(_train_view(), seed=20260719, device="cpu")

    invalid = _train_view()
    invalid = TrainView(
        X=invalid.X,
        labels=np.array([0, -1], dtype=np.int8),
        timestamps=invalid.timestamps,
        turbines=invalid.turbines,
        row_ids=invalid.row_ids,
        kind="supervised",
    )
    with pytest.raises(ValueError, match="-1"):
        _SupervisedAdapter().fit(invalid, seed=20260719, device="cpu")
