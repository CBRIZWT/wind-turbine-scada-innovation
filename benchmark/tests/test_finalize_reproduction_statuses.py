import json

import pytest

from tools.finalize_reproduction_statuses import (
    EXPECTED_MODEL_IDS,
    HASH_FIELDS,
    NEW_PAPER_DIRECTORY,
    SOURCE_DIFF_ADDITIONS,
    build_run_evidence,
    canonical_json_sha256,
    strict_load_json,
)


def test_locked_manifest_includes_the_new_arima_lasso_ewma_paper() -> None:
    assert len(EXPECTED_MODEL_IDS) == 14
    assert "arima_lasso_ewma" in EXPECTED_MODEL_IDS
    assert NEW_PAPER_DIRECTORY == (
        "Failure warning for offshore wind turbines based on Autoregressive models"
    )
    arima_difference = SOURCE_DIFF_ADDITIONS[NEW_PAPER_DIRECTORY]["code_behavior"]
    assert "AR(3)" in arima_difference
    assert "PCA" in arima_difference
    assert "validation-only" in arima_difference


def test_build_run_evidence_carries_metrics_and_all_provenance_hashes(tmp_path) -> None:
    record = {
        "model_id": "model_a",
        "farm": "kelmarsh",
        "status": "success",
        "seed": 20260719,
        "variant": "realfault",
        "metrics": {"local_equal4_score": 0.25, "pr_auc": None},
        **{field: f"{field}-value" for field in HASH_FIELDS},
    }
    run_record = tmp_path / "run_record.json"
    run_record.write_text(
        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    evidence = build_run_evidence(run_record)

    assert evidence["farm"] == "kelmarsh"
    assert evidence["run_record_path"] == str(run_record)
    assert evidence["metrics"] == record["metrics"]
    assert evidence["metrics_sha256"] == canonical_json_sha256(record["metrics"])
    assert evidence["hashes"] == {
        field: record[field] for field in HASH_FIELDS
    }
    assert len(evidence["run_record_sha256"]) == 64


def test_strict_json_loader_rejects_nonfinite_constants(tmp_path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"metric": NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        strict_load_json(invalid)
