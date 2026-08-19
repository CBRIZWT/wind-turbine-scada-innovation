from __future__ import annotations

import numpy as np

from realfault_benchmark.contracts import ScoreView, TrainView
from realfault_benchmark.paper_adapters import (
    ARIMALassoEWMAAdapter,
    CNN1DAdapter,
    ConfidenceIntervalAdapter,
    ConformalGRUAdapter,
    FleetMedianAEAdapter,
    FederatedLSTMAdapter,
    LifeTrendAdapter,
    PMLPAdapter,
    SLFormerAdapter,
    STAAdapter,
    StatisticalRFAdapter,
    TransGANWTAdapter,
    TransferAEAdapter,
    VAEHealthIndexAdapter,
    contiguous_segments,
)


CADENCE = 600 * 1_000_000_000


def _score_view(X: np.ndarray) -> ScoreView:
    return ScoreView(
        X=X,
        timestamps=np.arange(len(X), dtype=np.int64) * CADENCE,
        turbines=np.array(["A"] * len(X)),
        row_ids=np.arange(len(X), dtype=np.int64),
        split="test",
        gap_mask=np.array([False, False, True] + [False] * (len(X) - 3)),
    )


def test_contiguous_segments_break_at_frozen_gap_without_labels() -> None:
    view = _score_view(np.zeros((6, 2), dtype=np.float32))
    got = [segment.tolist() for segment in contiguous_segments(view)]
    assert got == [[0, 1], [3, 4, 5]]


def test_statistical_rf_uses_supervised_track_and_returns_true_probabilities() -> None:
    X = np.array([[0.0], [0.1], [0.2], [2.8], [3.0], [3.2]], dtype=np.float32)
    train = TrainView(
        X=X, labels=np.array([0, 0, 0, 1, 1, 1], dtype=np.int8),
        timestamps=np.arange(6, dtype=np.int64) * CADENCE,
        turbines=np.array(["A"] * 6), row_ids=np.arange(6),
        kind="supervised", gap_mask=np.zeros(6, dtype=bool),
    )
    score = ScoreView(
        X=X, timestamps=train.timestamps, turbines=train.turbines,
        row_ids=train.row_ids, split="test", gap_mask=np.zeros(6, dtype=bool),
    )
    adapter = StatisticalRFAdapter(feature_indices=np.array([0]), n_estimators=9, max_train_rows=6)
    adapter.fit(train, seed=20260719, device="cpu")
    probability = adapter.score(score)
    assert np.isfinite(probability).all()
    assert np.all((probability >= 0) & (probability <= 1))
    assert adapter.score_semantics == "probability"


def test_life_trend_never_scores_the_gap_row() -> None:
    X = np.arange(6, dtype=np.float32).reshape(-1, 1)
    train = TrainView(
        X=X, labels=np.zeros(6, dtype=np.int8),
        timestamps=np.arange(6, dtype=np.int64) * CADENCE,
        turbines=np.array(["A"] * 6), row_ids=np.arange(6), kind="normal",
        gap_mask=np.zeros(6, dtype=bool),
    )
    adapter = LifeTrendAdapter(feature_indices=np.array([0]), median_window=2, slope_lag=1)
    adapter.fit(train, seed=20260719, device="cpu")
    scores = adapter.score(_score_view(X))
    assert np.isnan(scores[2])
    assert np.isfinite(scores[[1, 4, 5]]).all()


def test_vae_health_index_smoke_is_row_aligned() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 2)).astype(np.float32)
    train = TrainView(
        X=X, labels=np.zeros(30, dtype=np.int8),
        timestamps=np.arange(30, dtype=np.int64) * CADENCE,
        turbines=np.array(["A"] * 30), row_ids=np.arange(30), kind="normal",
        gap_mask=np.zeros(30, dtype=bool),
    )
    score = ScoreView(X, train.timestamps, train.turbines, train.row_ids, "test", np.zeros(30, dtype=bool))
    adapter = VAEHealthIndexAdapter(
        feature_indices=np.array([0, 1]), max_train_rows=20, epochs=1,
        batch_size=8, hidden=4, latent=2,
    ).fit(train, 20260719, "cpu")
    got = adapter.score(score)
    assert got.shape == (30,)
    assert np.isfinite(got).all()


def test_pmlp_prediction_does_not_cross_first_row() -> None:
    X = np.linspace(0, 1, 60, dtype=np.float32).reshape(30, 2)
    train = TrainView(
        X=X, labels=np.zeros(30, dtype=np.int8),
        timestamps=np.arange(30, dtype=np.int64) * CADENCE,
        turbines=np.array(["A"] * 30), row_ids=np.arange(30), kind="normal",
        gap_mask=np.zeros(30, dtype=bool),
    )
    score = ScoreView(X, train.timestamps, train.turbines, train.row_ids, "test", np.zeros(30, dtype=bool))
    adapter = PMLPAdapter(
        feature_indices=np.array([0, 1]), max_train_windows=20,
        epochs=1, batch_size=8, hidden=(8, 4),
    ).fit(train, 20260719, "cpu")
    got = adapter.score(score)
    assert np.isnan(got[0])
    assert np.isfinite(got[1:]).all()


def test_arima_lasso_ewma_is_row_aligned_and_resets_at_frozen_gap() -> None:
    rng = np.random.default_rng(19)
    X = rng.normal(size=(80, 5)).astype(np.float32)
    train = TrainView(
        X=X, labels=np.zeros(80, dtype=np.int8),
        timestamps=np.arange(80, dtype=np.int64) * CADENCE,
        turbines=np.array(["A"] * 80), row_ids=np.arange(80), kind="normal",
        gap_mask=np.zeros(80, dtype=bool),
    )
    gap = np.zeros(80, dtype=bool)
    gap[40] = True
    score = ScoreView(X, train.timestamps, train.turbines, train.row_ids, "test", gap)
    adapter = ARIMALassoEWMAAdapter(
        feature_indices=np.arange(5), state_index=0, ar_order=2,
        max_train_windows=40, operating_clusters=(2, 3),
    ).fit(train, 20260719, "cpu")
    got = adapter.score(score)
    assert got.shape == (80,)
    assert np.isnan(got[0:2]).all()
    assert np.isnan(got[40:43]).all()
    assert np.isfinite(got[3:40]).all()
    assert np.isfinite(got[43:]).all()


def test_remaining_deep_paper_adapters_tiny_smoke() -> None:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(40, 4)).astype(np.float32)
    normal = TrainView(
        X=X, labels=np.zeros(40, dtype=np.int8),
        timestamps=np.arange(40, dtype=np.int64) * CADENCE,
        turbines=np.array(["A"] * 40), row_ids=np.arange(40), kind="normal",
        gap_mask=np.zeros(40, dtype=bool),
    )
    supervised = TrainView(
        X=X, labels=np.array([0] * 20 + [1] * 20, dtype=np.int8),
        timestamps=normal.timestamps, turbines=normal.turbines, row_ids=normal.row_ids,
        kind="supervised", gap_mask=np.zeros(40, dtype=bool),
    )
    score = ScoreView(X, normal.timestamps, normal.turbines, normal.row_ids, "test", np.zeros(40, dtype=bool))
    cases = [
        (ConfidenceIntervalAdapter(np.arange(2), ensemble_size=2, max_train_windows=12, epochs=1, batch_size=4), normal),
        (FleetMedianAEAdapter(np.arange(2), max_train_rows=12, epochs=1, batch_size=4, hidden=4, latent=2), normal),
        (ConformalGRUAdapter(np.arange(2), window=3, max_train_windows=12, epochs=1, batch_size=4, hidden=4), normal),
        (SLFormerAdapter(np.arange(2), window=4, patch=2, embedding=8, max_train_windows=12, epochs=1, batch_size=4), normal),
        (CNN1DAdapter(np.arange(2), window=4, max_train_windows=20, epochs=1, batch_size=4), supervised),
        (STAAdapter(np.arange(3), target_index=0, window=3, max_train_windows=8, epochs=1, batch_size=4), normal),
        (TransGANWTAdapter(np.arange(2), window=4, max_train_windows=8, epochs=1, batch_size=4, d_model=8), normal),
        (TransferAEAdapter(np.arange(2), max_train_rows=12, epochs=1, batch_size=4,
                           hidden=4, latent=2, finetune_rows=8, finetune_epochs=1), normal),
        (FederatedLSTMAdapter(np.arange(2), target_index=0, window=4,
                              max_train_windows=8, rounds=1, local_epochs=1, batch_size=4), normal),
    ]
    for adapter, train in cases:
        adapter.fit(train, 20260719, "cpu")
        got = adapter.score(score)
        assert got.shape == (40,), adapter.model_id
        assert np.isfinite(got).any(), adapter.model_id
