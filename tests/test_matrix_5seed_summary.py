import importlib.util
import json
from pathlib import Path


def _load_summary_module():
    path = Path(__file__).resolve().parents[1] / "矩阵5seed汇总.py"
    spec = importlib.util.spec_from_file_location("matrix_5seed_summary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_metric(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def test_build_summary_discovers_hill_and_marks_incomplete(tmp_path):
    module = _load_summary_module()
    base = tmp_path / "chronological_v2__v2"
    models = ["tritracknet"]
    modules = ["baseline_only"]

    for seed in range(5):
        _write_metric(
            base / "kelmarsh" / "tritracknet" / "metrics.jsonl",
            [
                {
                    "phase": "test",
                    "run_kind": "formal",
                    "module": "baseline_only",
                    "seed": seed,
                    "f1": 0.50 + seed / 100,
                    "auc": 0.60,
                    "auprc": 0.40,
                }
                for seed in range(5)
            ],
        )

    _write_metric(
        base / "hill_of_towie" / "tritracknet" / "metrics.jsonl",
        [
            {
                "phase": "test",
                "run_kind": "formal",
                "module": "baseline_only",
                "seed": 0,
                "f1": 0.10,
                "auc": 0.20,
                "auprc": 0.30,
            },
            {
                "phase": "test",
                "run_kind": "formal",
                "module": "baseline_only",
                "seed": 0,
                "f1": 0.15,
                "auc": 0.25,
                "auprc": 0.35,
            },
            {
                "phase": "test",
                "run_kind": "formal",
                "module": "baseline_only",
                "seed": 1,
                "f1": 0.20,
                "auc": 0.30,
                "auprc": 0.40,
            },
        ],
    )

    rows = module.build_summary(
        base=base,
        farms=None,
        models=models,
        modules=modules,
        expected_seeds=(0, 1, 2, 3, 4),
    )

    by_farm = {row["farm"]: row for row in rows}
    assert set(by_farm) == {"hill_of_towie", "kelmarsh"}
    assert by_farm["kelmarsh"]["n_seed"] == 5
    assert by_farm["kelmarsh"]["is_complete"] is True
    assert by_farm["hill_of_towie"]["n_seed"] == 2
    assert by_farm["hill_of_towie"]["is_complete"] is False
    assert by_farm["hill_of_towie"]["missing_seeds"] == "2,3,4"
    assert by_farm["hill_of_towie"]["f1_mean"] == 0.175
