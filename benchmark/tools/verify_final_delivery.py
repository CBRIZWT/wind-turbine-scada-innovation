"""只读核验最终真实故障交付物，并写出一份严格 JSON 结论。"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import fitz


ROOT = Path(r"E:\创新")
BENCHMARK = ROOT / "benchmark"
OUTPUT = ROOT / "论文复现" / "复现基准_2026-07-19"
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from realfault_benchmark.artifacts import write_strict_json  # noqa: E402
import run_real_fault_benchmark as runner  # noqa: E402


EXPECTED_MODELS = set(runner.PAPER_DIRS)
EXPECTED_FARMS = {"kelmarsh", "penmanshiel"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def strict_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: forbidden JSON constant {value}")

    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def csv_rows(name: str) -> list[dict[str, str]]:
    with (OUTPUT / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_finite_tree(value: Any, location: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tree(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_tree(item, f"{location}[{index}]")


def main() -> int:
    inventory = csv_rows("论文清单.csv")
    candidates = csv_rows("候选与资格排除表.csv")
    failures = csv_rows("下载失败清单.csv")
    successes = csv_rows("下载成功清单.csv")
    differences = csv_rows("源码差异清单.csv")
    long_rows = csv_rows("真实故障原始长表.csv")
    board = csv_rows("真实故障总榜.csv")
    excluded = csv_rows("真实故障总榜排除.csv")

    assert len(inventory) == 35
    assert sum(row["unique_work"].casefold() == "true" for row in inventory) == 34
    assert len(candidates) == 15
    assert all(row["strict_eligibility_conclusion"].startswith("excluded_") for row in candidates)
    assert len(failures) == 2
    assert any(row["status"] == "bot_or_captcha_access_blocked" for row in failures)
    assert len(successes) == 1 and successes[0]["status"] == "download_success_valid_pdf"
    assert len(differences) == 15
    assert len(long_rows) == 28 and all(row["status"] == "success" for row in long_rows)
    assert {row["model_id"] for row in long_rows} == EXPECTED_MODELS
    assert {row["farm"] for row in long_rows} == EXPECTED_FARMS
    assert all(row["variant"] == "realfault" for row in long_rows)
    assert len(board) == 13
    assert len(excluded) == 1
    assert excluded[0]["model_id"] == "life_extension_temperature_trend"
    assert excluded[0]["reason_code"] == "invalid_pr_auc"

    records: list[dict[str, Any]] = []
    for path in sorted((OUTPUT / "runs").glob("*/*/run_record.json")):
        record = strict_load(path)
        assert_finite_tree(record, str(path))
        assert record["status"] == "success"
        assert record["seed"] == 20260719
        assert record["variant"] == "realfault"
        assert record["farm"] in EXPECTED_FARMS
        assert record["model_id"] in EXPECTED_MODELS
        assert record["run_record_schema_version"] == "1.1"
        assert record["score_hash_status"] == "derived_from_existing_files_no_re_evaluation"
        for filename, field in (
            ("validation_scores.npy", "validation_score_file_sha256"),
            ("test_scores.npy", "test_score_file_sha256"),
            ("calibration.json", "calibration_file_sha256"),
        ):
            assert sha256_file(path.parent / filename) == record[field]
        records.append(record)
    assert len(records) == 28
    assert len({record["run_id"] for record in records}) == 28
    assert {(record["model_id"], record["farm"]) for record in records} == {
        (model, farm) for model in EXPECTED_MODELS for farm in EXPECTED_FARMS
    }

    manifest_records = []
    with (OUTPUT / "运行清单.jsonl").open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                manifest_records.append(
                    json.loads(
                        line,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError(f"JSONL line {number}: {value}")
                        ),
                    )
                )
    assert len(manifest_records) == 28
    assert {row["run_id"] for row in manifest_records} == {row["run_id"] for row in records}

    status_paths = sorted((OUTPUT / "statuses").glob("*/reproduction_status.json"))
    assert len(status_paths) == 35
    for path in status_paths:
        assert_finite_tree(strict_load(path), str(path))
    for path in OUTPUT.rglob("*.json"):
        if path.name != "最终核验.json":
            assert_finite_tree(strict_load(path), str(path))

    forbidden_names = ("a_prime", "aprime", "a-prime", "a′")
    forbidden_files = [
        str(path) for path in OUTPUT.rglob("*")
        if path.is_file() and any(marker in path.name.casefold() for marker in forbidden_names)
    ]
    assert not forbidden_files
    assert not (OUTPUT / "runs" / "care").exists()

    report = (OUTPUT / "最终复现报告.md").read_text(encoding="utf-8")
    required_sentence = "“本轮没有满足严格资格且完成复现的论文，无法事实性指定最新最优模板。”"
    assert required_sentence in report
    bad_controls = sorted({ord(char) for char in report if ord(char) < 32 and char not in "\n\r\t"})
    assert not bad_controls

    pdf_path = (
        ROOT / "论文复现" / "Failure warning for offshore wind turbines based on Autoregressive models"
        / "paper_accepted_manuscript.pdf"
    )
    assert pdf_path.read_bytes()[:4] == b"%PDF"
    with fitz.open(pdf_path) as document:
        assert document.page_count == 24
        first_pages = "".join(page.get_text() for page in document[:3]).casefold()
    assert "failure warning for offshore wind turbines based on autoregressive models" in first_pages
    assert sha256_file(pdf_path) == "1d28a3c46a186de9fc1ade6df24d958b70b684aa24c8895db9ac4fb883e0792a"

    gate = runner.gate_status()
    assert gate["status"] == "ready"
    result = {
        "schema": "final_delivery_verification/v1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": "local_real_fault_only",
        "paper_directories": 35,
        "unique_papers": 34,
        "strict_candidates": 15,
        "strict_queue_admitted": 0,
        "models": 14,
        "successful_runs": 28,
        "leaderboard_rows": 13,
        "leaderboard_exclusions": 1,
        "status_json_files": 35,
        "source_difference_rows": 15,
        "download_failure_rows": 2,
        "download_success_rows": 1,
        "forbidden_non_real_fault_files": forbidden_files,
        "json_nonfinite_values": 0,
        "report_control_characters": bad_controls,
        "gate_status": gate,
        "report_sha256": sha256_file(OUTPUT / "最终复现报告.md"),
        "leaderboard_sha256": sha256_file(OUTPUT / "真实故障总榜.csv"),
        "run_manifest_sha256": sha256_file(OUTPUT / "运行清单.jsonl"),
    }
    write_strict_json(OUTPUT / "最终核验.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
