"""为既有真实故障 run 补充不可变分数文件哈希，不重新执行模型或指标。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(r"E:\创新")
BENCHMARK = ROOT / "benchmark"
OUTPUT = ROOT / "论文复现" / "复现基准_2026-07-19"
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from realfault_benchmark.artifacts import write_strict_json  # noqa: E402
from run_real_fault_benchmark import rebuild_summaries  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    pending: list[tuple[Path, dict[str, object]]] = []
    for record_path in sorted((OUTPUT / "runs").glob("*/*/run_record.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "success":
            continue
        validation = record_path.parent / "validation_scores.npy"
        test = record_path.parent / "test_scores.npy"
        calibration = record_path.parent / "calibration.json"
        missing = [str(path) for path in (validation, test, calibration) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"run 配套文件缺失: {missing}")
        record["run_record_schema_version"] = "1.1"
        record["validation_score_file_sha256"] = sha256_file(validation)
        record["test_score_file_sha256"] = sha256_file(test)
        record["calibration_file_sha256"] = sha256_file(calibration)
        record["score_hash_status"] = "derived_from_existing_files_no_re_evaluation"
        labels_path = (
            ROOT / "SCADA数据集" / "数据预处理"
            / f"{record['farm']}__realfault" / "test_labels.npy"
        )
        labels = np.load(labels_path, allow_pickle=False)
        scores = np.load(test, allow_pickle=False)
        if labels.shape != scores.shape:
            raise ValueError(f"标签/分数 shape 不一致: {record_path}")
        finite = np.isfinite(scores)
        valid = labels != -1
        positive = labels == 1
        negative = labels == 0
        record["test_score_finite_coverage"] = float((finite & valid).sum() / valid.sum())
        record["test_positive_score_coverage"] = float(
            (finite & positive).sum() / positive.sum()
        ) if positive.any() else None
        record["test_negative_score_coverage"] = float(
            (finite & negative).sum() / negative.sum()
        ) if negative.any() else None
        pending.append((record_path, record))
    if not pending:
        raise RuntimeError("没有可加固的 success run")
    for record_path, record in pending:
        write_strict_json(record_path, record)
    rebuild_summaries()
    print(json.dumps({"updated_success_runs": len(pending)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
