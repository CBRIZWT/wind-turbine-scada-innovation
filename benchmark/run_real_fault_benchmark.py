from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(r"E:\创新")
BENCHMARK = ROOT / "benchmark"
PAPERS = ROOT / "论文复现"
OUTPUT = PAPERS / "复现基准_2026-07-19"
PREPROCESSED = ROOT / "SCADA数据集" / "数据预处理"
SEED = 20260719
DISK_GUARD_BYTES = 20 * 1024**3

if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from realfault_benchmark.artifacts import strict_json_dumps, write_strict_json
from realfault_benchmark.data import load_realfault_bundle
from realfault_benchmark.paper_adapters import (
    ARIMALassoEWMAAdapter,
    CNN1DAdapter,
    ConfidenceIntervalAdapter,
    ConformalGRUAdapter,
    FederatedLSTMAdapter,
    FleetMedianAEAdapter,
    LifeTrendAdapter,
    PMLPAdapter,
    SLFormerAdapter,
    STAAdapter,
    StatisticalRFAdapter,
    TransferAEAdapter,
    TransGANWTAdapter,
    VAEHealthIndexAdapter,
    base_residual_indices,
    generator_bearing_indices,
)
from realfault_benchmark.pipeline import evaluate_adapter
from realfault_benchmark.ranking import build_true_fault_leaderboard


PROTOCOL = {
    "name": "real_fault_equal4_v1",
    "cutoff": "2026-07-19",
    "farms": ["kelmarsh", "penmanshiel"],
    "variant": "realfault",
    "a_prime": "excluded_by_user",
    "seed": SEED,
    "horizon_hours": 12,
    "event_merge_hours": 72,
    "alarm_merge_minutes": 40,
    "threshold_grid": "201_quantiles_0_to_0.9995_plus_above_max_both_polarities",
    "calibration_split": "val",
    "test_evaluations": 1,
    "hpo": False,
    "exploratory_single_seed": True,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, ensure_ascii=False).encode("utf-8")
).hexdigest()


PAPER_DIRS = {
    "arima_lasso_ewma": "Failure warning for offshore wind turbines based on Autoregressive models",
    "statistical_gearbox_rf": "Wind Turbine Gearbox Fault Detection Based on Statistical Learning",
    "life_extension_temperature_trend": "Life extension of wind turbine drivetrains by means of SCADA data- Case study of generator bearings in an onshore wind farm",
    "vae_health_index_scada_migration": "Fault detection in wind turbines using health index monitoring with variational autoencoders",
    "probabilistic_mlp_cusum": "Probabilistic Multilayer Perceptrons for Wind Farm Condition Monitoring",
    "confidence_interval_ensemble": "On confidence interval-based anomaly detection approach for temperature predictions of wind turbine drivetrains to assist in lifetime extension assessment",
    "fleet_median_autoencoder": "Scalable SCADA-driven failure prediction using autoencoder NBM and fleet-median filtering",
    "conformal_gru_thermal_prognostics": "Prognostics of Thermal Anomalies in Wind Turbines via Deep Learning and Conformal Prediction Using SCADA Data",
    "slformer_gearbox": "Early anomaly detection of wind turbine gearbox based on SLFormer neural network",
    "cnn1d_temporal_scada": "Early prediction of wind turbine anomalies using 1D-CNN and temporal feature engineering",
    "sta_bka_temperature_residual": "Temperature Prediction and Fault Warning of High-Speed Shaft of Wind Turbine Gearbox Based on Hybrid Deep Learning Model",
    "transgan_wt_dual_reconstruction": "Trans GAN-WT anomaly detection model for wind turbine time series",
    "transfer_autoencoder_full_finetune": "Transfer learning applications for anomaly detection in wind turbines",
    "federated_lstm_nbm": "Wind turbine condition monitoring based on intra- and inter-farm federated learning",
}


def _front_bearing_index(feature_names: tuple[str, ...]) -> int:
    preferred = [
        i for i, name in enumerate(feature_names)
        if str(name).endswith("__resid")
        and "front bearing temperature" in str(name).lower()
        and "generator" not in str(name).lower()
    ]
    return int(preferred[0] if preferred else base_residual_indices(feature_names)[0])


def _gear_oil_index(feature_names: tuple[str, ...]) -> int:
    preferred = [
        i for i, name in enumerate(feature_names)
        if str(name).endswith("__resid") and "gear oil temperature" in str(name).lower()
    ]
    return int(preferred[0] if preferred else _front_bearing_index(feature_names))


def adapter_registry(feature_names: tuple[str, ...]) -> dict[str, Callable[[], object]]:
    base = base_residual_indices(feature_names)
    generator = generator_bearing_indices(feature_names)
    front = _front_bearing_index(feature_names)
    gear_oil = _gear_oil_index(feature_names)
    sta_features = base[: min(9, len(base))]
    return {
        "arima_lasso_ewma": lambda: ARIMALassoEWMAAdapter(
            base, gear_oil, ar_order=3, max_train_windows=50_000,
            operating_clusters=(3, 4, 5), ewma_weight=0.2,
        ),
        "statistical_gearbox_rf": lambda: StatisticalRFAdapter(base, n_estimators=100, max_train_rows=100_000),
        "life_extension_temperature_trend": lambda: LifeTrendAdapter(generator),
        "vae_health_index_scada_migration": lambda: VAEHealthIndexAdapter(base, max_train_rows=30_000, epochs=2),
        "probabilistic_mlp_cusum": lambda: PMLPAdapter(base, max_train_windows=40_000, epochs=2),
        "confidence_interval_ensemble": lambda: ConfidenceIntervalAdapter(base, ensemble_size=3, max_train_windows=20_000, epochs=1),
        "fleet_median_autoencoder": lambda: FleetMedianAEAdapter(base, max_train_rows=30_000, epochs=2),
        "conformal_gru_thermal_prognostics": lambda: ConformalGRUAdapter(base, max_train_windows=20_000, epochs=2),
        "slformer_gearbox": lambda: SLFormerAdapter(base, max_train_windows=10_000, epochs=1),
        "cnn1d_temporal_scada": lambda: CNN1DAdapter(base, max_train_windows=20_000, epochs=2),
        "sta_bka_temperature_residual": lambda: STAAdapter(sta_features, front, max_train_windows=10_000, epochs=1),
        "transgan_wt_dual_reconstruction": lambda: TransGANWTAdapter(base[: min(11, len(base))], max_train_windows=5_000, epochs=1),
        "transfer_autoencoder_full_finetune": lambda: TransferAEAdapter(base, max_train_rows=30_000, epochs=2, finetune_epochs=1),
        "federated_lstm_nbm": lambda: FederatedLSTMAdapter(base, front, max_train_windows=20_000, rounds=5, local_epochs=3),
    }


def active_blockers() -> list[dict[str, object]]:
    patterns = (
        "运行真实故障全指标_v3.py", "跑三farm全部快速实验.py",
        "SCADA数据集\\数据预处理.py", "SCADA数据集/数据预处理.py",
    )
    try:
        import psutil
    except ImportError:
        return [{"pid": None, "command": "psutil_missing", "reason": "process_gate_unavailable"}]
    blockers = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.info["pid"] == os.getpid():
            continue
        command = " ".join(process.info.get("cmdline") or [])
        if any(pattern.lower() in command.lower() for pattern in patterns):
            blockers.append({"pid": int(process.info["pid"]), "command": command})
    return blockers


def gate_status() -> dict[str, object]:
    disk = shutil.disk_usage("E:\\")
    blockers = active_blockers()
    return {
        "status": "ready" if not blockers and disk.free >= DISK_GUARD_BYTES else "blocked",
        "active_blockers": blockers,
        "disk_free_bytes": disk.free,
        "disk_guard_bytes": DISK_GUARD_BYTES,
        "protocol_hash": PROTOCOL_HASH,
    }


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _code_hash(adapter: object, paper_directory: str) -> str:
    digest = hashlib.sha256()
    files = [Path(inspect.getfile(type(adapter))), BENCHMARK / "realfault_benchmark" / "pipeline.py"]
    paper = PAPERS / paper_directory
    files.extend(sorted(paper.glob("*.py")))
    for path in files:
        if path.is_file():
            digest.update(str(path).encode("utf-8")); digest.update(path.read_bytes())
    return digest.hexdigest()


def _paper_hash(directory: str) -> str | None:
    pdfs = sorted((PAPERS / directory).glob("*.pdf"))
    return _sha256(pdfs[0]) if pdfs else None


def _environment_hash() -> str | None:
    path = OUTPUT / "environment" / "environment.sha256"
    return path.read_text(encoding="ascii").strip() if path.is_file() else None


def _record_path(model_id: str, farm: str) -> Path:
    return OUTPUT / "runs" / model_id / farm / "run_record.json"


def _can_resume(path: Path, *, code_hash: str, data_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return all([
        record.get("status") == "success", record.get("seed") == SEED,
        record.get("protocol_hash") == PROTOCOL_HASH,
        record.get("code_hash") == code_hash, record.get("data_hash") == data_hash,
    ])


def _guard_existing_run(path: Path, *, resumable: bool, allow_rerun: bool) -> str:
    """防止源代码/数据变化后静默覆盖唯一一次测试评估。"""

    if not path.is_file():
        return "run"
    if resumable:
        return "resume"
    if allow_rerun:
        return "run"
    raise RuntimeError(
        f"现有运行不能按哈希安全 resume，已拒绝覆盖: {path}；"
        "如已归档旧产物且确需重跑，请显式传入 --allow-rerun"
    )


def _json_cell(value: object) -> object:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    return value


def rebuild_summaries() -> None:
    records = []
    for path in sorted((OUTPUT / "runs").glob("*/*/run_record.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    manifest = OUTPUT / "运行清单.jsonl"
    manifest.write_text(
        "".join(strict_json_dumps(record, indent=None) + "\n" for record in records),
        encoding="utf-8",
    )
    flat = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "metrics"}
        row.update(record.get("metrics") or {})
        flat.append({key: _json_cell(value) for key, value in row.items()})
    long = pd.DataFrame(flat)
    long.to_csv(OUTPUT / "真实故障原始长表.csv", index=False, encoding="utf-8-sig")
    if long.empty:
        board = pd.DataFrame()
        excluded = pd.DataFrame()
    else:
        board, excluded = build_true_fault_leaderboard(long, return_exclusions=True)
    board.to_csv(OUTPUT / "真实故障总榜.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(OUTPUT / "真实故障总榜排除.csv", index=False, encoding="utf-8-sig")


def run_one(
    bundle,
    model_id: str,
    builder: Callable[[], object],
    device: str,
    *,
    allow_rerun: bool = False,
) -> dict[str, object]:
    adapter = builder()
    paper_directory = PAPER_DIRS[model_id]
    code_hash = _code_hash(adapter, paper_directory)
    path = _record_path(model_id, bundle.farm)
    action = _guard_existing_run(
        path,
        resumable=_can_resume(path, code_hash=code_hash, data_hash=bundle.file_hash),
        allow_rerun=allow_rerun,
    )
    if action == "resume":
        return json.loads(path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        result = evaluate_adapter(bundle, adapter, seed=SEED, device=device)
        np.save(path.parent / "validation_scores.npy", result["validation_scores"], allow_pickle=False)
        np.save(path.parent / "test_scores.npy", result["test_scores"], allow_pickle=False)
        write_strict_json(path.parent / "calibration.json", result["calibration"].to_dict())
        metrics = result["metrics"]
        peak_vram = (
            float(torch.cuda.max_memory_allocated() / 1024**2)
            if torch.cuda.is_available() and device.startswith("cuda") else None
        )
        try:
            import psutil
            peak_ram = float(psutil.Process().memory_info().rss / 1024**2)
        except Exception:
            peak_ram = None
        record: dict[str, object] = {
            "run_id": f"{model_id}:{bundle.farm}:{SEED}", "model_id": model_id,
            "paper_title": getattr(adapter, "paper_title", paper_directory),
            "paper_directory": paper_directory, "farm": bundle.farm,
            "status": "success", "seed": SEED, "variant": bundle.variant,
            "calibration_split": "val", "protocol_hash": PROTOCOL_HASH,
            "data_hash": bundle.file_hash, "split_hash": bundle.split_hash,
            "feature_hash": bundle.feature_hash,
            "calibration_hash": result["calibration"].artifact_hash,
            "paper_hash": _paper_hash(paper_directory), "code_hash": code_hash,
            "environment_hash": _environment_hash(),
            "publication_date": getattr(adapter, "publication_date", None),
            "reproduction_kind": getattr(adapter, "reproduction_kind", None),
            "score_semantics": result["score_semantics"],
            "metrics": metrics, "elapsed_seconds": time.perf_counter() - started,
            "peak_ram_mb": peak_ram, "peak_vram_mb": peak_vram,
            "exploratory": True,
        }
    except Exception as exc:
        record = {
            "run_id": f"{model_id}:{bundle.farm}:{SEED}", "model_id": model_id,
            "paper_title": getattr(adapter, "paper_title", paper_directory),
            "paper_directory": paper_directory, "farm": bundle.farm,
            "status": "failed", "seed": SEED, "variant": bundle.variant,
            "calibration_split": "val", "protocol_hash": PROTOCOL_HASH,
            "data_hash": bundle.file_hash, "code_hash": code_hash,
            "paper_hash": _paper_hash(paper_directory),
            "environment_hash": _environment_hash(),
            "publication_date": getattr(adapter, "publication_date", None),
            "reproduction_kind": getattr(adapter, "reproduction_kind", None),
            "metrics": {}, "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__, "error_message": str(exc),
            "traceback": traceback.format_exc(), "exploratory": True,
        }
    write_strict_json(path, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", default="kelmarsh,penmanshiel")
    parser.add_argument("--models", default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument(
        "--allow-rerun",
        action="store_true",
        help="显式允许覆盖哈希不匹配的既有运行；默认拒绝以保护测试只评一次",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    gate = gate_status()
    write_strict_json(OUTPUT / "gate_status.json", gate)
    print(strict_json_dumps(gate))
    if args.gate_only:
        return 0 if gate["status"] == "ready" else 3
    if gate["status"] != "ready":
        return 3

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    farms = [value.strip() for value in args.farms.split(",") if value.strip()]
    for farm in farms:
        bundle = load_realfault_bundle(PREPROCESSED, farm, verify_file_hashes=True, verify_normal_subset=True)
        registry = adapter_registry(bundle.feature_names)
        models = list(registry) if args.models == "all" else [value.strip() for value in args.models.split(",")]
        unknown = sorted(set(models) - set(registry))
        if unknown:
            raise ValueError(f"未知模型: {unknown}")
        for model_id in models:
            current_gate = gate_status()
            if current_gate["status"] != "ready":
                write_strict_json(OUTPUT / "gate_status.json", current_gate)
                return 3
            print(f"RUN {model_id} @ {farm}", flush=True)
            record = run_one(
                bundle, model_id, registry[model_id], device,
                allow_rerun=args.allow_rerun,
            )
            print(f"STATUS {record['status']} elapsed={record.get('elapsed_seconds')}", flush=True)
            rebuild_summaries()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del bundle
        gc.collect()
    rebuild_summaries()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
