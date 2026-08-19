"""Finalize paper-level statuses from the locked real-fault run manifest.

Inputs are the existing inventory plus 28 successful run records (14 models x two
farms). Outputs are the paper/candidate/download/source-difference manifests,
the 35 ``reproduction_status.json`` files, and a compact literature summary.

The script is deterministic and repeatable.  It does not train models, mutate run
records or leaderboards, touch source repositories, or delete any user artifact.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping
import unicodedata

import fitz


WORKSPACE = Path(r"E:\创新")
PAPER_ROOT = WORKSPACE / "论文复现"
OUT = PAPER_ROOT / "复现基准_2026-07-19"
INVENTORY_CSV = OUT / "论文清单.csv"
CANDIDATE_CSV = OUT / "候选与资格排除表.csv"
SOURCE_DIFF_CSV = OUT / "源码差异清单.csv"
DOWNLOAD_FAILURE_CSV = OUT / "下载失败清单.csv"
DOWNLOAD_SUCCESS_CSV = OUT / "下载成功清单.csv"
LITERATURE_SUMMARY_JSON = OUT / "文献交付物摘要.json"
STATUS_ROOT = OUT / "statuses"
RUN_ROOT = OUT / "runs"

CUTOFF = "2026-07-19"
SEED = 20260719
EXPECTED_FARMS = ("kelmarsh", "penmanshiel")
EXPECTED_MODEL_IDS = {
    "arima_lasso_ewma",
    "cnn1d_temporal_scada",
    "confidence_interval_ensemble",
    "conformal_gru_thermal_prognostics",
    "federated_lstm_nbm",
    "fleet_median_autoencoder",
    "life_extension_temperature_trend",
    "probabilistic_mlp_cusum",
    "slformer_gearbox",
    "sta_bka_temperature_residual",
    "statistical_gearbox_rf",
    "transfer_autoencoder_full_finetune",
    "transgan_wt_dual_reconstruction",
    "vae_health_index_scada_migration",
}

HASH_FIELDS = (
    "paper_hash",
    "code_hash",
    "data_hash",
    "feature_hash",
    "split_hash",
    "calibration_hash",
    "protocol_hash",
    "environment_hash",
)

CANONICAL_PMLP = "Probabilistic Multilayer Perceptrons for Wind Farm Condition Monitoring"
DUPLICATE_PMLP = "Probabilistic Multi-Layer Perceptrons for Wind Farm Condition Monitoring"
CARE_DIRECTORY = "CARE to Compare - real-world dataset for anomaly detection in wind turbine data"
CGAN_DIRECTORY = "Simulating run-to-failure SCADA time series for fault detection and prognosis"
UNRELATED_DIRECTORY = "Select bibliography"
NEW_PAPER_DIRECTORY = "Failure warning for offshore wind turbines based on Autoregressive models"
NEW_PAPER_TITLE = "Failure Warning for Offshore Wind Turbines Based on Autoregressive Models"
NEW_PAPER_DOI = "10.1016/j.oceaneng.2025.121448"
NEW_PAPER_VENUE = "Ocean Engineering 332:121448"
NEW_PAPER_PDF = "paper_accepted_manuscript.pdf"
NEW_PAPER_CODE = "ARIMA_LASSO_EWMA.py"
NEW_PAPER_AUTHORS = (
    "Hongxu Ye; Wenjin Zhu; He Li; Weidong Ji; C. Guedes Soares; Jin Wang"
)
STRICT_EVIDENCE = (
    "No publication-year official JCR Q1/CAS zone-1 Top evidence was closed in "
    "public official sources; a local directory or repository description is not "
    "official qualification evidence, and the journal article is not a CCF-A paper."
)
QUALIFICATION_EVIDENCE_URLS = (
    "https://www.fenqubiao.com/Default.aspx;"
    "https://www.ccf.org.cn/Academic_Evaluation/By_category/"
)

CODE_OVERRIDES = {
    NEW_PAPER_DIRECTORY: (
        NEW_PAPER_CODE,
        "paper_reimplementation",
    ),
    "Temperature Prediction and Fault Warning of High-Speed Shaft of Wind Turbine Gearbox Based on Hybrid Deep Learning Model": (
        "STA_BKA.py",
        "method_migration",
    ),
    "Trans GAN-WT anomaly detection model for wind turbine time series": (
        "TransGAN_WT.py",
        "paper_reimplementation",
    ),
    "Wind Turbine Gearbox Fault Detection Based on Statistical Learning": (
        "StatisticalGearboxRF.py",
        "method_migration",
    ),
}

SOURCE_DIFF_ADDITIONS: dict[str, dict[str, str]] = {
    NEW_PAPER_DIRECTORY: {
        "diff_id": "ARIMA-EWMA-001",
        "category": "order_condition_proxy_and_threshold",
        "severity": "high",
        "paper_description": (
            "Per-variable ARIMA orders are selected with ACF/PACF/BIC; operating "
            "conditions use wind speed and power before LASSO secondary residuals and "
            "multi-condition EWMA control limits."
        ),
        "code_behavior": (
            "The rapid local implementation reduces ARIMA to fixed AR(3), replaces "
            "wind-speed/power operating conditions with a PCA residual proxy, and emits "
            "a continuous EWMA distance whose alarm threshold is frozen by validation-only "
            "calibration."
        ),
        "annotation_status": "local_reimplementation_difference_documented",
    },
    "Temperature Prediction and Fault Warning of High-Speed Shaft of Wind Turbine Gearbox Based on Hybrid Deep Learning Model": {
        "diff_id": "STA-BKA-001",
        "category": "optimization_data_and_threshold",
        "severity": "high",
        "paper_description": (
            "Two CNN layers, two LSTM layers and 16-head attention predict high-speed "
            "shaft temperature; BKA tunes hyperparameters and a DI/VI/SI fault index is "
            "evaluated on private 5-minute data."
        ),
        "code_behavior": (
            "The method migration keeps the published backbone and auditable fault-index "
            "formula, disables BKA/HPO under the fixed budget, uses local 10-minute "
            "temperature residuals, and freezes the decision threshold on validation."
        ),
        "annotation_status": "local_reimplementation_difference_documented",
    },
    "Trans GAN-WT anomaly detection model for wind turbine time series": {
        "diff_id": "TRANSGAN-WT-001",
        "category": "unpublished_dimensions_budget_and_threshold",
        "severity": "high",
        "paper_description": (
            "Dual Transformer encoders and dual decoders are trained with evolutionary "
            "reconstruction losses; the paper uses a POT decision threshold and reports "
            "a substantially larger private-data model."
        ),
        "code_behavior": (
            "No author repository or complete layer widths were verified. The local "
            "paper reimplementation keeps dual encoding/decoding and the published score, "
            "uses a reduced fixed architecture, and calibrates the threshold on validation."
        ),
        "annotation_status": "local_reimplementation_difference_documented",
    },
    "Wind Turbine Gearbox Fault Detection Based on Statistical Learning": {
        "diff_id": "STAT-RF-001",
        "category": "label_contract_and_unpublished_hyperparameters",
        "severity": "high",
        "paper_description": (
            "Segment-wise temperature thresholds create healthy/fault labels before "
            "comparing statistical classifiers; Random Forest is the reported best model."
        ),
        "code_behavior": (
            "The method migration never treats adaptive-threshold pseudo-labels as truth; "
            "the fixed Random Forest trains on the locked real-fault labels, with missing "
            "paper hyperparameters explicitly selected and recorded."
        ),
        "annotation_status": "local_reimplementation_difference_documented",
    },
}

SOURCE_BEHAVIOR_UPDATES = {
    "Early prediction of wind turbine anomalies using 1D-CNN and temporal feature engineering": (
        "The local paper reimplementation uses the central real-fault adapter and locked "
        "validation-only calibration; it does not claim the paper's CARE-style event result."
    ),
    "Prognostics of Thermal Anomalies in Wind Turbines via Deep Learning and Conformal Prediction Using SCADA Data": (
        "The local paper reimplementation runs a reduced conformal-GRU thermal method on "
        "the locked real-fault farms; private-data numerical reproduction is not claimed."
    ),
    "Transfer learning applications for anomaly detection in wind turbines": (
        "The local paper reimplementation uses local SCADA and a full-finetune autoencoder "
        "adapter with validation-only calibration; it does not reproduce the source dataset."
    ),
    "Wind turbine condition monitoring based on intra- and inter-farm federated learning": (
        "The author repository is preserved; an external reduced-round local adapter runs "
        "on the locked real-fault farms, so results are local-adapter results rather than "
        "paper-scale numerical reproduction."
    ),
}

PROTECTED_OUTPUTS = (
    "真实故障原始长表.csv",
    "真实故障总榜.csv",
    "真实故障总榜排除.csv",
    "运行清单.jsonl",
    "源码注释审计.json",
)

CSV_ADDITIONAL_FIELDS = (
    "model_id",
    "successful_farms",
    "run_record_paths",
    "run_record_sha256s",
    "metrics_sha256s",
    "leaderboard_status",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_slug(index: int, name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_").lower()
    suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"{index:02d}_{stem[:52]}_{suffix}"


def new_paper_status_path() -> Path:
    return STATUS_ROOT / safe_slug(35, NEW_PAPER_DIRECTORY) / "reproduction_status.json"


def new_paper_pdf_evidence() -> dict[str, Any]:
    path = PAPER_ROOT / NEW_PAPER_DIRECTORY / NEW_PAPER_PDF
    if not path.is_file() or path.read_bytes()[:4] != b"%PDF":
        raise ValueError(f"New paper is absent or not a PDF: {path}")
    with fitz.open(path) as document:
        pages = document.page_count
    if pages <= 0:
        raise ValueError(f"New paper has no parseable pages: {path}")
    return {
        "path": str(path),
        "status": "valid_pdf_parseable",
        "count": 1,
        "bytes": path.stat().st_size,
        "pages": pages,
        "sha256": sha256_file(path),
        "header_pdf": True,
        "parse_error": "",
    }


def apply_new_paper_inventory_fields(row: dict[str, str]) -> None:
    pdf = new_paper_pdf_evidence()
    row.update(
        {
            "title": NEW_PAPER_TITLE,
            "directory": NEW_PAPER_DIRECTORY,
            "directory_path": str(PAPER_ROOT / NEW_PAPER_DIRECTORY),
            "doi": NEW_PAPER_DOI,
            "year": "2025",
            "venue": NEW_PAPER_VENUE,
            "metadata_status": (
                "verified_local_accepted_manuscript_plus_doi_metadata"
            ),
            "pdf_path": pdf["path"],
            "pdf_status": pdf["status"],
            "pdf_bytes": str(pdf["bytes"]),
            "pdf_pages": str(pdf["pages"]),
            "pdf_sha256": pdf["sha256"],
            "duplicate_of": "",
            "unique_work": "True",
            "task_relevance": "highly_related",
            "code_status": "local_reimplementation_no_official_repo_mirrored",
            "code_paths": NEW_PAPER_CODE,
            "source_remote": "",
            "source_commit": "",
            "source_dirty": "",
            "source_dirty_entries": "",
            "source_dirty_summary": "not_a_local_git_repository",
            "reproduction_classification": "paper_reimplementation",
            "strict_eligibility_conclusion": (
                "excluded_insufficient_official_year_evidence"
            ),
            "strict_eligibility_evidence": STRICT_EVIDENCE,
            "run_status": "",
            "run_notes": "",
            "status_json_path": str(new_paper_status_path()),
        }
    )


def new_paper_status_document(row: Mapping[str, str]) -> dict[str, Any]:
    pdf = new_paper_pdf_evidence()
    return {
        "schema_version": "1.1",
        "audit_cutoff": CUTOFF,
        "paper": {
            "title": row["title"],
            "directory": row["directory"],
            "directory_path": row["directory_path"],
            "doi": row["doi"],
            "year": 2025,
            "venue": row["venue"],
            "metadata_status": row["metadata_status"],
        },
        "pdf": pdf,
        "duplicate_of": None,
        "unique_work": True,
        "task_relevance": "highly_related",
        "code": code_from_row(row),
        "reproduction_classification": "paper_reimplementation",
        "strict_eligibility": {
            "conclusion": "excluded_insufficient_official_year_evidence",
            "evidence": STRICT_EVIDENCE,
            "evidence_urls": QUALIFICATION_EVIDENCE_URLS.split(";"),
            "strict_queue_admitted": False,
        },
        "download": {
            "status": "extra_open_access_download_completed_outside_strict_gate",
            "source_kind": "LJMU_open_access_accepted_manuscript",
            "local_pdf_validated": True,
            "local_pdf_sha256": pdf["sha256"],
        },
    }


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def strict_load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=_reject_nonfinite_json,
    )
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def assert_finite_tree(value: Any, *, location: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite float at {location}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite_tree(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_tree(item, location=f"{location}[{index}]")


def build_run_evidence(run_record_path: Path) -> dict[str, Any]:
    record = strict_load_json(run_record_path)
    missing = [field for field in HASH_FIELDS if not record.get(field)]
    if missing:
        raise ValueError(f"Missing provenance hashes in {run_record_path}: {missing}")
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing metrics object: {run_record_path}")
    assert_finite_tree(metrics, location=str(run_record_path))
    return {
        "farm": record.get("farm"),
        "run_id": record.get("run_id"),
        "run_record_path": str(run_record_path),
        "run_record_sha256": sha256_file(run_record_path),
        "status": record.get("status"),
        "seed": record.get("seed"),
        "variant": record.get("variant"),
        "exploratory": record.get("exploratory"),
        "calibration_split": record.get("calibration_split"),
        "score_semantics": record.get("score_semantics"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "peak_ram_mb": record.get("peak_ram_mb"),
        "peak_vram_mb": record.get("peak_vram_mb"),
        "metrics": metrics,
        "metrics_sha256": canonical_json_sha256(metrics),
        "hashes": {field: record[field] for field in HASH_FIELDS},
    }


def tree_manifest_sha256(root: Path) -> tuple[int, str]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return len(files), digest.hexdigest()


def protected_snapshot() -> dict[str, Any]:
    run_file_count, run_manifest = tree_manifest_sha256(RUN_ROOT)
    return {
        "run_tree_file_count": run_file_count,
        "run_tree_manifest_sha256": run_manifest,
        "protected_files": {
            name: sha256_file(OUT / name) for name in PROTECTED_OUTPUTS
        },
    }


def load_locked_runs() -> dict[str, dict[str, tuple[Path, dict[str, Any]]]]:
    paths = sorted(RUN_ROOT.rglob("run_record.json"))
    if len(paths) != 28:
        raise RuntimeError(f"Expected exactly 28 run records, found {len(paths)}")

    runs: dict[str, dict[str, tuple[Path, dict[str, Any]]]] = {}
    for path in paths:
        record = strict_load_json(path)
        model_id = record.get("model_id")
        farm = record.get("farm")
        if model_id not in EXPECTED_MODEL_IDS:
            raise ValueError(f"Unexpected model_id in {path}: {model_id}")
        if farm not in EXPECTED_FARMS:
            raise ValueError(f"Unexpected farm in {path}: {farm}")
        if path.parent.name != farm or path.parent.parent.name != model_id:
            raise ValueError(f"Run-record path does not match its identity: {path}")
        if farm in runs.setdefault(model_id, {}):
            raise ValueError(f"Duplicate run for {model_id}/{farm}")
        runs[model_id][farm] = (path, record)

    if set(runs) != EXPECTED_MODEL_IDS:
        raise ValueError("The locked model set differs from the expected 13-model manifest")
    for model_id, farm_runs in runs.items():
        if tuple(sorted(farm_runs)) != tuple(sorted(EXPECTED_FARMS)):
            raise ValueError(f"Model lacks one of the two farms: {model_id}")
        records = [farm_runs[farm][1] for farm in EXPECTED_FARMS]
        for record in records:
            if record.get("status") != "success":
                raise ValueError(f"Non-success run in locked manifest: {record.get('run_id')}")
            if record.get("seed") != SEED or record.get("variant") != "realfault":
                raise ValueError(f"Protocol identity mismatch: {record.get('run_id')}")
            if record.get("exploratory") is not True:
                raise ValueError(f"Run is not marked exploratory: {record.get('run_id')}")
            if record.get("calibration_split") != "val":
                raise ValueError(f"Calibration is not validation-only: {record.get('run_id')}")
            build_run_evidence(farm_runs[str(record["farm"])][0])
        for field in ("paper_directory", "paper_hash", "code_hash", "protocol_hash", "environment_hash", "reproduction_kind"):
            if len({record.get(field) for record in records}) != 1:
                raise ValueError(f"Cross-farm {field} mismatch for {model_id}")

    by_directory = {
        farm_runs[EXPECTED_FARMS[0]][1]["paper_directory"]: model_id
        for model_id, farm_runs in runs.items()
    }
    if len(by_directory) != len(runs):
        raise ValueError("More than one model maps to the same paper directory")
    if by_directory.get(CANONICAL_PMLP) != "probabilistic_mlp_cusum":
        raise ValueError("The PMLP run is not mapped to the canonical directory")

    expected_override_kinds = {
        directory: kind for directory, (_, kind) in CODE_OVERRIDES.items()
    }
    for directory, expected_kind in expected_override_kinds.items():
        model_id = by_directory.get(directory)
        if model_id is None:
            raise ValueError(f"Missing run for code override: {directory}")
        actual = runs[model_id][EXPECTED_FARMS[0]][1].get("reproduction_kind")
        if actual != expected_kind:
            raise ValueError(
                f"Reproduction-kind mismatch for {directory}: {actual} != {expected_kind}"
            )
    return runs


def load_leaderboard_membership() -> dict[str, dict[str, Any]]:
    board_path = OUT / "真实故障总榜.csv"
    exclusion_path = OUT / "真实故障总榜排除.csv"
    with board_path.open("r", encoding="utf-8-sig", newline="") as handle:
        board_rows = list(csv.DictReader(handle))
    with exclusion_path.open("r", encoding="utf-8-sig", newline="") as handle:
        excluded_rows = list(csv.DictReader(handle))
    if len(board_rows) != 13 or len(excluded_rows) != 1:
        raise ValueError("Expected 13 ranked models and one explicitly excluded model")
    membership: dict[str, dict[str, Any]] = {}
    for row in board_rows:
        model_id = row.get("model_id", "")
        if model_id in membership:
            raise ValueError(f"Duplicate leaderboard model: {model_id}")
        membership[model_id] = {
            "included": True,
            "status": "included_real_fault_leaderboard",
            "reason_code": None,
            "reason_detail": None,
        }
    for row in excluded_rows:
        model_id = row.get("model_id", "")
        if model_id in membership:
            raise ValueError(f"Model both ranked and excluded: {model_id}")
        membership[model_id] = {
            "included": False,
            "status": "excluded_real_fault_leaderboard",
            "reason_code": row.get("reason_code") or None,
            "reason_detail": row.get("reason_detail") or None,
        }
    if set(membership) != EXPECTED_MODEL_IDS:
        raise ValueError("Leaderboard membership does not cover the locked 14-model set")
    if membership["life_extension_temperature_trend"]["reason_code"] != "invalid_pr_auc":
        raise ValueError("Unexpected life-extension leaderboard exclusion reason")
    return membership


def parse_nullable_bool(value: str) -> bool | None:
    normalized = value.strip().casefold()
    if not normalized:
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid nullable Boolean: {value}")


def parse_nullable_int(value: str) -> int | None:
    return int(value) if value.strip() else None


def code_from_row(row: Mapping[str, str]) -> dict[str, Any]:
    paths = [path for path in row["code_paths"].split(";") if path]
    code: dict[str, Any] = {
        "status": row["code_status"],
        "paths": paths,
        "remote": row["source_remote"],
        "commit": row["source_commit"],
        "dirty": parse_nullable_bool(row["source_dirty"]),
        "dirty_entries": parse_nullable_int(row["source_dirty_entries"]),
        "dirty_summary": row["source_dirty_summary"],
    }
    local_hashes: dict[str, str] = {}
    directory = Path(row["directory_path"])
    for relative in paths:
        candidate = directory / relative.rstrip("/")
        if candidate.is_file():
            local_hashes[relative] = sha256_file(candidate)
    if local_hashes:
        code["local_file_sha256"] = local_hashes
    return code


def update_code_override(row: dict[str, str]) -> None:
    override = CODE_OVERRIDES.get(row["directory"])
    if override is None:
        return
    filename, _ = override
    code_path = Path(row["directory_path"]) / filename
    if not code_path.is_file():
        raise FileNotFoundError(f"Expected reimplementation file is absent: {code_path}")
    row["code_status"] = "local_reimplementation_no_official_repo_mirrored"
    row["code_paths"] = filename
    row["source_remote"] = ""
    row["source_commit"] = ""
    row["source_dirty"] = ""
    row["source_dirty_entries"] = ""
    row["source_dirty_summary"] = "not_a_local_git_repository"


def run_reproduction(
    model_id: str,
    farm_runs: dict[str, tuple[Path, dict[str, Any]]],
    leaderboard: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    evidences = [build_run_evidence(farm_runs[farm][0]) for farm in EXPECTED_FARMS]
    records = [farm_runs[farm][1] for farm in EXPECTED_FARMS]
    reproduction_kind = str(records[0]["reproduction_kind"])
    notes = (
        "Both locked real-fault farms completed with seed 20260719. This is a "
        "single-seed exploratory local result, not evidence of statistical significance "
        "and not a claim that the paper's reported numerical results were reproduced."
    )
    reproduction = {
        "run_status": "two_farm_success_exploratory",
        "run_notes": notes,
        "model_id": model_id,
        "reproduction_kind": reproduction_kind,
        "seed": SEED,
        "variant": "realfault",
        "successful_farms": list(EXPECTED_FARMS),
        "runs": evidences,
        "run_record_manifest_sha256": canonical_json_sha256(
            [
                {
                    "farm": evidence["farm"],
                    "run_record_path": evidence["run_record_path"],
                    "run_record_sha256": evidence["run_record_sha256"],
                    "metrics_sha256": evidence["metrics_sha256"],
                }
                for evidence in evidences
            ]
        ),
        "metrics": {
            evidence["farm"]: evidence["metrics"] for evidence in evidences
        },
        "metrics_status": "available_exploratory_single_seed",
        "leaderboard": dict(leaderboard),
    }
    return reproduction_kind, reproduction


def nonrun_reproduction(row: Mapping[str, str]) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    directory = row["directory"]
    if directory == DUPLICATE_PMLP:
        return (
            "excluded",
            {
                "run_status": "excluded_duplicate",
                "run_notes": (
                    f"Duplicate of {CANONICAL_PMLP}; the canonical directory carries the "
                    "probabilistic_mlp_cusum two-farm run. No duplicate run was created."
                ),
                "model_id": None,
                "runs": [],
                "metrics": None,
                "metrics_status": "not_applicable_duplicate",
            },
            None,
        )
    if directory == CARE_DIRECTORY:
        care = {
            "protocol_status": "protocol_audited",
            "dataset_status": "dataset_not_downloaded",
            "official_code_status": "official_code_not_installed",
            "experiment_status": "experiment_not_run",
            "metrics_status": "metrics_not_available",
        }
        return (
            "artifact_only",
            {
                "run_status": "protocol_audited/dataset_not_downloaded/experiment_not_run",
                "run_notes": (
                    "Protocol audit only. The dataset was not downloaded, official code was "
                    "not installed, and no experiment or metric was produced."
                ),
                "model_id": None,
                "runs": [],
                "metrics": None,
                "metrics_status": "metrics_not_available",
            },
            {"care_protocol": care},
        )
    if directory == CGAN_DIRECTORY:
        generator = {
            "artifact_status": "artifact_only",
            "gain_ablation_status": "gain_ablation_not_run",
            "fixed_downstream_detector_status": "not_run",
            "leaderboard_eligible": False,
            "reason": (
                "A generator is only evaluable through a fixed downstream detector gain "
                "ablation; that ablation was not run in the locked real-fault experiment."
            ),
        }
        return (
            "artifact_only",
            {
                "run_status": "artifact_only/gain_ablation_not_run",
                "run_notes": (
                    "Generator retained as a verifiable code/paper artifact only. No fixed "
                    "downstream-detector gain ablation was run, so it is excluded from rankings."
                ),
                "model_id": None,
                "runs": [],
                "metrics": None,
                "metrics_status": "not_applicable_gain_ablation_not_run",
            },
            {"generator_evaluation": generator},
        )
    if directory == UNRELATED_DIRECTORY:
        return (
            "excluded",
            {
                "run_status": "excluded_unrelated",
                "run_notes": "Bibliography-only material unrelated to the executable scientific task.",
                "model_id": None,
                "runs": [],
                "metrics": None,
                "metrics_status": "not_applicable_excluded",
            },
            None,
        )

    if row["reproduction_classification"] not in {"artifact_only", "excluded"}:
        raise ValueError(
            f"Unmapped non-run paper has an executable classification: {directory} / "
            f"{row['reproduction_classification']}"
        )
    classification = row["reproduction_classification"]
    if classification == "excluded":
        status = "excluded_not_applicable"
        metrics_status = "not_applicable_excluded"
        notes = "Excluded by the existing inventory decision; no real-fault run is applicable."
    elif row["task_relevance"] == "highly_related":
        status = "artifact_only_not_run"
        metrics_status = "metrics_not_available"
        notes = (
            "Related paper retained as an artifact, but it has no matching successful run in "
            "the locked 26-record, two-farm real-fault manifest."
        )
    else:
        status = "artifact_only_not_run"
        metrics_status = "metrics_not_available"
        notes = (
            "Bibliographic artifact retained for context; it is outside the final executable "
            "real-fault temperature benchmark and has no matching run record."
        )
    return (
        classification,
        {
            "run_status": status,
            "run_notes": notes,
            "model_id": None,
            "runs": [],
            "metrics": None,
            "metrics_status": metrics_status,
        },
        None,
    )


def load_inventory() -> tuple[list[str], list[dict[str, str]]]:
    with INVENTORY_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    required = {
        "directory",
        "directory_path",
        "duplicate_of",
        "task_relevance",
        "code_status",
        "code_paths",
        "source_remote",
        "source_commit",
        "source_dirty",
        "source_dirty_entries",
        "source_dirty_summary",
        "reproduction_classification",
        "run_status",
        "run_notes",
        "status_json_path",
    }
    if not required.issubset(fields):
        raise ValueError(f"Inventory is missing fields: {sorted(required - set(fields))}")
    for field in CSV_ADDITIONAL_FIELDS:
        if field not in fields:
            fields.append(field)
    directories = {row["directory"] for row in rows}
    if len(rows) == 34 and NEW_PAPER_DIRECTORY not in directories:
        new_row = {field: "" for field in fields}
        apply_new_paper_inventory_fields(new_row)
        rows.append(new_row)
    elif len(rows) == 35 and NEW_PAPER_DIRECTORY in directories:
        apply_new_paper_inventory_fields(
            next(row for row in rows if row["directory"] == NEW_PAPER_DIRECTORY)
        )
    else:
        raise ValueError(
            "Inventory must contain either the prior 34 rows or the finalized 35 rows"
        )
    rows.sort(key=lambda row: row["directory"].casefold())
    if len(rows) != 35 or len({row["directory"] for row in rows}) != 35:
        raise ValueError("Expected 35 unique paper-directory rows")

    excluded_directories = {"__pycache__", "codex调研记录", OUT.name}
    disk_directories = {
        path.name
        for path in PAPER_ROOT.iterdir()
        if path.is_dir() and path.name not in excluded_directories
    }
    if len(disk_directories) != 35 or disk_directories != {
        row["directory"] for row in rows
    }:
        raise ValueError("Paper inventory and the 35 on-disk paper directories differ")
    return fields, rows


def load_and_upsert_candidate() -> tuple[list[str], list[dict[str, str]]]:
    with CANDIDATE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    required = {
        "title",
        "authors",
        "formal_date",
        "online_first_date",
        "year",
        "doi",
        "venue",
        "official_qualification_year",
        "strict_eligibility_conclusion",
        "qualification_evidence",
        "qualification_evidence_urls",
        "paper_url",
        "task_relevance",
        "local_pdf_match",
        "local_directory",
        "local_pdf_status",
        "local_pdf_sha256",
        "source_url",
        "source_status",
        "download_decision",
        "decision",
    }
    if not required.issubset(fields):
        raise ValueError(f"Candidate table is missing fields: {sorted(required - set(fields))}")
    matches = [row for row in rows if row.get("doi", "").casefold() == NEW_PAPER_DOI.casefold()]
    if len(matches) > 1:
        raise ValueError("New paper occurs more than once in the candidate table")
    if not matches:
        if len(rows) != 14:
            raise ValueError("Expected the original 14 candidates before insertion")
        row = {field: "" for field in fields}
        rows.append(row)
    else:
        if len(rows) != 15:
            raise ValueError("Expected 15 candidates after insertion")
        row = matches[0]
    pdf = new_paper_pdf_evidence()
    row.update(
        {
            "title": NEW_PAPER_TITLE,
            "authors": NEW_PAPER_AUTHORS,
            "formal_date": "2025",
            "online_first_date": "not_precisely_verified",
            "year": "2025",
            "doi": NEW_PAPER_DOI,
            "venue": NEW_PAPER_VENUE,
            "official_qualification_year": "2025",
            "strict_eligibility_conclusion": (
                "excluded_insufficient_official_year_evidence"
            ),
            "qualification_evidence": STRICT_EVIDENCE,
            "qualification_evidence_urls": QUALIFICATION_EVIDENCE_URLS,
            "paper_url": f"https://doi.org/{NEW_PAPER_DOI}",
            "task_relevance": "directly_related",
            "local_pdf_match": "True",
            "local_directory": NEW_PAPER_DIRECTORY,
            "local_pdf_status": "valid_pdf_parseable",
            "local_pdf_sha256": pdf["sha256"],
            "source_url": "",
            "source_status": (
                "no_author_linked_repository_verified_in_current_manifest"
            ),
            "download_decision": (
                "extra_open_access_download_completed_outside_strict_gate"
            ),
            "decision": "excluded_from_strict_admission",
        }
    )
    if len(rows) != 15 or sum(
        candidate.get("doi", "").casefold() == NEW_PAPER_DOI.casefold()
        for candidate in rows
    ) != 1:
        raise ValueError("Candidate table must contain 15 rows and one new-paper row")
    return fields, rows


def load_and_update_source_differences(
    inventory_by_directory: Mapping[str, dict[str, str]],
    model_by_directory: Mapping[str, str],
    runs: Mapping[str, dict[str, tuple[Path, dict[str, Any]]]],
) -> tuple[list[str], list[dict[str, str]]]:
    with SOURCE_DIFF_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    required = {
        "title",
        "directory",
        "code_kind",
        "code_paths",
        "source_remote",
        "source_commit",
        "source_dirty",
        "source_dirty_entries",
        "source_dirty_summary",
        "diff_id",
        "category",
        "severity",
        "paper_description",
        "code_behavior",
        "evidence",
        "annotation_status",
        "run_status",
    }
    if not required.issubset(fields):
        raise ValueError(f"Source-difference table is missing fields: {sorted(required - set(fields))}")
    by_directory = {row["directory"]: row for row in rows}
    if len(by_directory) != len(rows):
        raise ValueError("Source-difference table contains duplicate paper directories")

    for directory, spec in SOURCE_DIFF_ADDITIONS.items():
        inventory = inventory_by_directory[directory]
        row = by_directory.get(directory)
        if row is None:
            row = {field: "" for field in fields}
            rows.append(row)
            by_directory[directory] = row
        row.update(
            {
                "title": inventory["title"],
                "directory": directory,
                "code_kind": inventory["code_status"],
                "code_paths": inventory["code_paths"],
                "source_remote": inventory["source_remote"],
                "source_commit": inventory["source_commit"],
                "source_dirty": inventory["source_dirty"],
                "source_dirty_entries": inventory["source_dirty_entries"],
                "source_dirty_summary": inventory["source_dirty_summary"],
                **spec,
            }
        )

    for directory, model_id in model_by_directory.items():
        row = by_directory.get(directory)
        if row is None:
            raise ValueError(f"Executable paper lacks a source-difference row: {directory}")
        paths = [str(runs[model_id][farm][0]) for farm in EXPECTED_FARMS]
        row["run_status"] = "two_farm_success_exploratory"
        row["evidence"] = (
            f"Local paper/code audit plus locked run records for model_id={model_id}: "
            f"{paths[0]}; {paths[1]}. Single-seed exploratory result only."
        )
        if directory in SOURCE_BEHAVIOR_UPDATES:
            row["code_behavior"] = SOURCE_BEHAVIOR_UPDATES[directory]

    cgan = by_directory.get(CGAN_DIRECTORY)
    if cgan is None:
        raise ValueError("cGAN source-difference row is absent")
    cgan["run_status"] = "gain_ablation_not_run_artifact_only"
    cgan["evidence"] = (
        "Paper/code artifact audited. The required fixed-downstream-detector gain "
        "ablation was not run, so no leaderboard metric exists."
    )

    expected_annotations = {
        "Fault detection in wind turbines using health index monitoring with variational autoencoders": (
            "comment-only AST-equivalent"
        ),
        "Wind turbine condition monitoring based on intra- and inter-farm federated learning": (
            "comment-only AST-equivalent"
        ),
    }
    for directory, expected in expected_annotations.items():
        if by_directory[directory]["annotation_status"] != expected:
            raise ValueError(f"Official-source annotation status changed for {directory}")

    rows.sort(key=lambda row: row["directory"].casefold())
    if len(rows) != 15 or len({row["directory"] for row in rows}) != 15:
        raise ValueError("Expected 15 unique source-difference rows")
    if {
        row["directory"]
        for row in rows
        if row["run_status"] == "two_farm_success_exploratory"
    } != set(model_by_directory):
        raise ValueError("Source-difference run statuses do not cover all 14 models")
    return fields, rows


def load_and_update_download_failures() -> tuple[list[str], list[dict[str, str]]]:
    with DOWNLOAD_FAILURE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    required = {
        "paper_title",
        "doi",
        "artifact_type",
        "attempted",
        "url",
        "status",
        "reason",
        "audit_cutoff",
    }
    if not required.issubset(fields):
        raise ValueError(f"Download-failure table is missing fields: {sorted(required - set(fields))}")

    # A successful open-access artifact is never represented as a failure.
    rows = [row for row in rows if row.get("doi", "").casefold() != NEW_PAPER_DOI.casefold()]
    seta_doi = "10.1016/j.seta.2025.104806"
    matches = [row for row in rows if row.get("doi", "").casefold() == seta_doi.casefold()]
    if len(matches) > 1:
        raise ValueError("SETA download failure appears more than once")
    if matches:
        seta = matches[0]
    else:
        seta = {field: "" for field in fields}
        rows.append(seta)
    seta.update(
        {
            "paper_title": (
                "CNN-BiLSTM-Autoencoder hybrid for prognostics of gearbox "
                "Over-Temperature faults in offshore wind turbines"
            ),
            "doi": seta_doi,
            "artifact_type": "paper_pdf",
            "attempted": "True",
            "url": f"https://doi.org/{seta_doi}",
            "status": "bot_or_captcha_access_blocked",
            "reason": (
                "The publisher download flow presented a bot/CAPTCHA challenge. The "
                "attempt stopped without bypassing access controls or trying an "
                "unverified copy."
            ),
            "audit_cutoff": CUTOFF,
        }
    )
    if not any(
        row.get("status") == "no_download_attempted_strict_queue_empty"
        and row.get("attempted", "").casefold() == "false"
        for row in rows
    ):
        raise ValueError("The strict-queue-empty non-attempt audit row was lost")
    return fields, rows


def build_download_success_table() -> tuple[list[str], list[dict[str, Any]]]:
    pdf = new_paper_pdf_evidence()
    fields = [
        "paper_title",
        "doi",
        "artifact_type",
        "attempted",
        "source_url",
        "status",
        "reason",
        "local_path",
        "bytes",
        "pages",
        "sha256",
        "license",
        "audit_cutoff",
    ]
    rows: list[dict[str, Any]] = [
        {
            "paper_title": NEW_PAPER_TITLE,
            "doi": NEW_PAPER_DOI,
            "artifact_type": "paper_pdf",
            "attempted": True,
            "source_url": "https://researchonline.ljmu.ac.uk/id/eprint/26589/",
            "status": "download_success_valid_pdf",
            "reason": (
                "LJMU Research Online openly supplied the accepted manuscript; local "
                "PDF header, page count, title and SHA-256 were validated. The DOI "
                "comes from linked repository/publisher metadata and is not embedded "
                "in the accepted-manuscript PDF text."
            ),
            "local_path": pdf["path"],
            "bytes": pdf["bytes"],
            "pages": pdf["pages"],
            "sha256": pdf["sha256"],
            "license": "CC BY-NC-ND",
            "audit_cutoff": CUTOFF,
        }
    ]
    return fields, rows


def atomic_write_text(path: Path, text: str, *, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def render_csv(fields: list[str], rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def finalize() -> dict[str, Any]:
    protected_before = protected_snapshot()
    runs = load_locked_runs()
    leaderboard_membership = load_leaderboard_membership()
    fields, rows = load_inventory()
    candidate_fields, candidate_rows = load_and_upsert_candidate()
    model_by_directory = {
        farm_runs[EXPECTED_FARMS[0]][1]["paper_directory"]: model_id
        for model_id, farm_runs in runs.items()
    }
    if len(model_by_directory) != 14:
        raise ValueError("Expected 14 unique model-to-paper mappings")
    for row in rows:
        update_code_override(row)
    inventory_by_directory = {row["directory"]: row for row in rows}
    source_fields, source_rows = load_and_update_source_differences(
        inventory_by_directory,
        model_by_directory,
        runs,
    )
    failure_fields, failure_rows = load_and_update_download_failures()
    success_fields, success_rows = build_download_success_table()

    status_documents: list[tuple[Path, dict[str, Any]]] = []
    seen_status_paths: set[Path] = set()
    for row in rows:
        update_code_override(row)
        directory = row["directory"]
        model_id = model_by_directory.get(directory)
        extra_section: dict[str, Any] | None = None
        if model_id is not None:
            classification, reproduction = run_reproduction(
                model_id,
                runs[model_id],
                leaderboard_membership[model_id],
            )
            evidences = reproduction["runs"]
            row["model_id"] = model_id
            row["successful_farms"] = ";".join(
                f"{farm}__realfault" for farm in EXPECTED_FARMS
            )
            row["run_record_paths"] = ";".join(
                evidence["run_record_path"] for evidence in evidences
            )
            row["run_record_sha256s"] = ";".join(
                f"{evidence['farm']}:{evidence['run_record_sha256']}"
                for evidence in evidences
            )
            row["metrics_sha256s"] = ";".join(
                f"{evidence['farm']}:{evidence['metrics_sha256']}"
                for evidence in evidences
            )
            leaderboard = leaderboard_membership[model_id]
            row["leaderboard_status"] = str(leaderboard["status"])
            if leaderboard["reason_code"]:
                row["leaderboard_status"] += f":{leaderboard['reason_code']}"
        else:
            classification, reproduction, extra_section = nonrun_reproduction(row)
            for field in CSV_ADDITIONAL_FIELDS:
                row[field] = ""

        row["reproduction_classification"] = classification
        row["run_status"] = str(reproduction["run_status"])
        row["run_notes"] = str(reproduction["run_notes"])

        status_path = Path(row["status_json_path"])
        resolved_status = status_path.resolve()
        if not resolved_status.is_relative_to(STATUS_ROOT.resolve()):
            raise ValueError(f"Status path escapes the status root: {status_path}")
        if resolved_status in seen_status_paths or (
            not status_path.is_file() and directory != NEW_PAPER_DIRECTORY
        ):
            raise ValueError(f"Missing or duplicate status path: {status_path}")
        seen_status_paths.add(resolved_status)
        if directory == NEW_PAPER_DIRECTORY and not status_path.is_file():
            document = new_paper_status_document(row)
        else:
            document = strict_load_json(status_path)
        document["schema_version"] = "1.1"
        document["audit_cutoff"] = CUTOFF
        document["duplicate_of"] = row["duplicate_of"] or None
        document["unique_work"] = not bool(row["duplicate_of"])
        document["task_relevance"] = row["task_relevance"]
        document["code"] = code_from_row(row)
        document["reproduction_classification"] = classification
        document["experiment_scope"] = {
            "track": "local_real_fault_only",
            "farms": [
                "kelmarsh__realfault",
                "penmanshiel__realfault",
            ],
            "hill_real_fault": (
                "not_constructed_no_maintenance_confirmed_temperature_truth"
            ),
        }
        document["reproduction"] = reproduction
        document.pop("care_protocol", None)
        document.pop("generator_evaluation", None)
        if extra_section:
            document.update(extra_section)
        assert_finite_tree(document, location=str(status_path))
        json.dumps(document, ensure_ascii=False, allow_nan=False)
        status_documents.append((status_path, document))

    if len(status_documents) != 35 or len(seen_status_paths) != 35:
        raise ValueError("Expected exactly 35 status documents")
    if sum(row["run_status"] == "two_farm_success_exploratory" for row in rows) != 14:
        raise ValueError("Expected exactly 14 two-farm successful paper rows")
    duplicate_row = next(row for row in rows if row["directory"] == DUPLICATE_PMLP)
    if duplicate_row["duplicate_of"] != CANONICAL_PMLP or duplicate_row["model_id"]:
        raise ValueError("Duplicate PMLP was incorrectly mapped to a run")

    csv_text = render_csv(fields, rows)
    candidate_csv_text = render_csv(candidate_fields, candidate_rows)
    source_csv_text = render_csv(source_fields, source_rows)
    failure_csv_text = render_csv(failure_fields, failure_rows)
    success_csv_text = render_csv(success_fields, success_rows)
    forbidden_markers = ("A-prime", "A_prime", "a_prime", "A′")
    if any(marker in csv_text for marker in forbidden_markers):
        raise ValueError("Inventory contains a forbidden non-real-fault track marker")
    if any(marker in candidate_csv_text for marker in forbidden_markers):
        raise ValueError("Candidate table contains a forbidden non-real-fault track marker")
    if any(marker in source_csv_text for marker in forbidden_markers):
        raise ValueError("Source-difference table contains a forbidden track marker")
    for status_path, document in status_documents:
        text = json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        ) + "\n"
        if any(marker in text for marker in forbidden_markers):
            raise ValueError(f"Status contains a forbidden track marker: {status_path}")
        atomic_write_text(status_path, text, encoding="utf-8")
    atomic_write_text(INVENTORY_CSV, csv_text, encoding="utf-8-sig")
    atomic_write_text(CANDIDATE_CSV, candidate_csv_text, encoding="utf-8-sig")
    atomic_write_text(SOURCE_DIFF_CSV, source_csv_text, encoding="utf-8-sig")
    atomic_write_text(DOWNLOAD_FAILURE_CSV, failure_csv_text, encoding="utf-8-sig")
    atomic_write_text(DOWNLOAD_SUCCESS_CSV, success_csv_text, encoding="utf-8-sig")
    literature_summary = {
        "audit_cutoff": CUTOFF,
        "paper_directories": len(rows),
        "unique_works": sum(not bool(row["duplicate_of"]) for row in rows),
        "valid_pdfs": sum(row["pdf_status"] == "valid_pdf_parseable" for row in rows),
        "highly_related_directories": sum(row["task_relevance"] == "highly_related" for row in rows),
        "adjacent_directories": sum(row["task_relevance"] == "adjacent" for row in rows),
        "unrelated_directories": sum(row["task_relevance"] == "unrelated" for row in rows),
        "local_code_directories": sum(
            row["code_status"] in {
                "local_reimplementation_no_official_repo_mirrored",
                "official_source_mirror_plus_local_adapter",
            }
            for row in rows
        ),
        "official_repositories_mirrored": sum(
            row["code_status"] == "official_source_mirror_plus_local_adapter"
            for row in rows
        ),
        "strict_candidates": len(candidate_rows),
        "strict_queue_admitted": 0,
        "strict_triggered_download_attempts": 0,
        "extra_download_attempts": sum(
            str(row.get("attempted", "")).casefold() == "true"
            for row in failure_rows + success_rows
        ),
        "download_failures": sum(
            str(row.get("attempted", "")).casefold() == "true"
            for row in failure_rows
        ),
        "download_successes": len(success_rows),
        "source_difference_rows": len(source_rows),
        "status_json_files": len(status_documents),
        "experiment_track": "local_real_fault_only",
        "successful_models": len(runs),
        "successful_run_records": sum(len(farms) for farms in runs.values()),
        "leaderboard_rows": sum(
            membership["included"] for membership in leaderboard_membership.values()
        ),
    }
    atomic_write_text(
        LITERATURE_SUMMARY_JSON,
        json.dumps(literature_summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    protected_after = protected_snapshot()
    if protected_after != protected_before:
        raise RuntimeError("A protected training/source artifact changed during finalization")
    status_count, status_manifest = tree_manifest_sha256(STATUS_ROOT)
    if status_count != 35:
        raise ValueError(f"Expected 35 status files, found {status_count}")
    result = {
        "schema": "reproduction_status_finalization/v1",
        "paper_rows": len(rows),
        "unique_papers": sum(not bool(row["duplicate_of"]) for row in rows),
        "status_files": status_count,
        "model_count": len(runs),
        "successful_run_records": sum(len(farms) for farms in runs.values()),
        "two_farm_success_papers": sum(
            row["run_status"] == "two_farm_success_exploratory" for row in rows
        ),
        "inventory_sha256": sha256_file(INVENTORY_CSV),
        "candidate_rows": len(candidate_rows),
        "candidate_sha256": sha256_file(CANDIDATE_CSV),
        "source_difference_rows": len(source_rows),
        "source_difference_sha256": sha256_file(SOURCE_DIFF_CSV),
        "download_failure_rows": len(failure_rows),
        "download_failure_sha256": sha256_file(DOWNLOAD_FAILURE_CSV),
        "download_success_rows": len(success_rows),
        "download_success_sha256": sha256_file(DOWNLOAD_SUCCESS_CSV),
        "literature_summary_sha256": sha256_file(LITERATURE_SUMMARY_JSON),
        "leaderboard_rows": sum(
            membership["included"] for membership in leaderboard_membership.values()
        ),
        "status_manifest_sha256": status_manifest,
        "protected_artifacts_unchanged": True,
        "protected_snapshot": protected_after,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def main() -> None:
    finalize()


if __name__ == "__main__":
    main()
