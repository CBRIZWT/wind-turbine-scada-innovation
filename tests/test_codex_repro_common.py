import json
from pathlib import Path

import numpy as np


def test_load_farm_sample_contract():
    from codex.repro_common import load_farm_sample

    sample = load_farm_sample(Path(r"E:\创新"), "kelmarsh", max_train=64, max_test=32)

    assert sample.farm == "kelmarsh"
    assert sample.train_x.shape[0] <= 64
    assert sample.test_x.shape[0] <= 32
    assert sample.train_x.ndim == 2
    assert sample.test_x.ndim == 2
    assert sample.train_x.shape[1] == sample.test_x.shape[1]
    assert sample.train_y.shape[0] == sample.train_x.shape[0]
    assert sample.test_y.shape[0] == sample.test_x.shape[0]
    assert sample.meta["source_dir"].endswith(r"SCADA数据集\数据预处理\kelmarsh")


def test_evaluate_scores_contract():
    from codex.repro_common import evaluate_scores

    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.1, 0.1, 0.9, 0.8, 0.7], dtype=np.float32)
    result = evaluate_scores(labels, scores)

    assert result["threshold_source"] == "validation_quantile_95"
    assert result["f1"] > 0.99
    assert result["precision"] > 0.99
    assert result["recall"] > 0.99
    assert 0.0 <= result["auc"] <= 1.0


def test_write_result_roundtrip(tmp_path):
    from codex.repro_common import write_result

    path = tmp_path / "results.json"
    payload = {"paper": "demo", "results": [{"farm": "kelmarsh", "metric": {"f1": 0.5}}]}
    write_result(path, payload)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == payload
