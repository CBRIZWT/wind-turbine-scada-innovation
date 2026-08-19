from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(r"E:\创新\论文复现")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_sta_bka_migration_has_paper_shape_contract() -> None:
    module = _load(
        ROOT / "Temperature Prediction and Fault Warning of High-Speed Shaft of Wind Turbine Gearbox Based on Hybrid Deep Learning Model" / "STA_BKA.py",
        "sta_bka_reimplementation",
    )
    model = module.STATemperatureRegressor(n_features=9, conv_channels=(32, 64), lstm_hidden=(128, 64), heads=16)
    output = model(torch.zeros(2, 36, 9))
    assert output.shape == (2,)
    assert module.REPRODUCTION_KIND == "method_migration"


def test_transgan_wt_migration_returns_two_reconstructions() -> None:
    module = _load(
        ROOT / "Trans GAN-WT anomaly detection model for wind turbine time series" / "TransGAN_WT.py",
        "transgan_wt_reimplementation",
    )
    model = module.TransGANWT(n_features=6, window=12, d_model=24, heads=4, layers=1)
    first, second = model(torch.zeros(3, 12, 6))
    assert first.shape == second.shape == (3, 12, 6)
    score = module.dual_reconstruction_score(torch.ones(3, 12, 6), first, second)
    assert score.shape == (3,)
    assert torch.all(score >= 0)
    assert module.REPRODUCTION_KIND == "paper_reimplementation"


def test_statistical_gearbox_factory_is_deterministic_random_forest() -> None:
    module = _load(
        ROOT / "Wind Turbine Gearbox Fault Detection Based on Statistical Learning" / "StatisticalGearboxRF.py",
        "statistical_gearbox_reimplementation",
    )
    estimator = module.build_random_forest(seed=20260719, n_estimators=7)
    X = np.array([[0.0], [0.1], [0.2], [3.0], [3.1], [3.2]])
    y = np.array([0, 0, 0, 1, 1, 1])
    estimator.fit(X, y)
    assert estimator.random_state == 20260719
    assert estimator.n_estimators == 7
    assert estimator.predict_proba(X).shape == (6, 2)
    assert module.REPRODUCTION_KIND == "method_migration"
