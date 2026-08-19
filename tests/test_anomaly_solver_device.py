from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
ANOMALY_ROOT = ROOT / "Anomaly-Transformer-main"
sys.path.insert(0, str(ANOMALY_ROOT))

import solver as anomaly_solver  # noqa: E402


class DummyAnomalyTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(__import__("torch").ones(1))
        self.to_device = None

    def to(self, device):
        self.to_device = str(device)
        return self


def test_solver_sets_device_before_build_model(monkeypatch):
    created = []

    def make_model(*args, **kwargs):
        model = DummyAnomalyTransformer()
        created.append(model)
        return model

    monkeypatch.setattr(anomaly_solver, "get_loader_segment", lambda *args, **kwargs: [])
    monkeypatch.setattr(anomaly_solver, "AnomalyTransformer", make_model)
    monkeypatch.setattr(anomaly_solver.torch.cuda, "is_available", lambda: True)

    solver = anomaly_solver.Solver(
        {
            "data_path": "unused",
            "batch_size": 2,
            "win_size": 4,
            "dataset": "SCADA",
            "input_c": 3,
            "output_c": 3,
            "lr": 1e-4,
        }
    )

    assert str(solver.device) == "cuda:0"
    assert created[0].to_device == "cuda:0"
