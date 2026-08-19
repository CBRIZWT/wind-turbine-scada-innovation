# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WT_ROOT = ROOT / "wt-transformer-fault-prediction-main"
sys.path.insert(0, str(WT_ROOT))
sys.path.insert(0, str(ROOT))


def test_torch_wt_transformer_forward_shape():
    from Src.model.transformer_torch import WTTransformerTorch

    model = WTTransformerTorch(
        input_dim=18,
        seq_len=144,
        head_size=32,
        num_heads=4,
        ff_dim=8,
        num_transformer_blocks=1,
        mlp_units=(16,),
        dropout=0.0,
        mlp_dropout=0.0,
    )

    y = model(torch.randn(2, 144, 18))

    assert y.shape == (2,)


def test_scada_npy_dataset_streams_windows_and_aligns_labels(tmp_path):
    from Src.data.scada_npy_dataset import ScadaNpyWindowDataset

    data = np.arange(18, dtype=np.float32).reshape(6, 3)
    labels = np.array([0, 0, 1, -1, 0, 1], dtype=np.int64)
    data_path = tmp_path / "train.npy"
    labels_path = tmp_path / "train_labels.npy"
    np.save(data_path, data)
    np.save(labels_path, labels)

    ds = ScadaNpyWindowDataset(data_path, labels_path, n_steps=2, target_index=1)

    x0, y0, label0 = ds[0]
    assert len(ds) == 4
    np.testing.assert_array_equal(x0.numpy(), data[:2])
    assert float(y0.item()) == float(data[2, 1])
    assert int(label0.item()) == int(labels[2])


def test_compute_metrics_from_scores_uses_validation_threshold_only():
    from Src.model.torch_eval import compute_metrics_from_scores

    val_labels = np.array([0, 0, 1, 1], dtype=np.int64)
    val_scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    test_labels = np.array([0, 1, 0, 1], dtype=np.int64)
    test_scores = np.array([0.1, 0.85, 0.2, 0.95], dtype=np.float32)

    metrics, oriented_test_scores, preds = compute_metrics_from_scores(
        val_labels=val_labels,
        val_scores=val_scores,
        test_labels=test_labels,
        test_scores=test_scores,
        fallback_scores=val_scores,
    )

    assert metrics["threshold_source"] == "validation_evt1_f1"
    assert metrics["score_polarity"] == "positive"
    assert oriented_test_scores.shape == test_scores.shape
    assert preds.tolist() == [0, 1, 0, 1]
    assert metrics["f1"] == 1.0


def test_compute_metrics_handles_uncalibrated_fallback_polarity():
    from Src.model.torch_eval import compute_metrics_from_scores

    metrics, oriented_test_scores, preds = compute_metrics_from_scores(
        val_labels=np.array([0, 0, 0, 0], dtype=np.int64),
        val_scores=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        test_labels=np.array([0, 1, 0, 1], dtype=np.int64),
        test_scores=np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32),
        fallback_scores=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    )

    assert metrics["score_polarity"] == "positive_fallback_uncalibrated"
    assert oriented_test_scores.shape == (4,)
    assert preds.shape == (4,)


def test_kalman_filter_1d_keeps_length_and_finite_values():
    kalman = importlib.import_module("卡尔曼滤波.卡尔曼滤波")

    scores = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    filtered = kalman.kalman_filter_1d(scores, process_var=1e-4, measurement_var=1e-2)

    assert filtered.shape == scores.shape
    assert np.isfinite(filtered).all()
    np.testing.assert_allclose(filtered, scores, atol=1e-5)
