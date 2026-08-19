"""生成当前交付状态的代码与关键产物 SHA-256 清单。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(r"E:\创新")
BENCHMARK = ROOT / "benchmark"
OUTPUT = ROOT / "论文复现" / "复现基准_2026-07-19"
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from realfault_benchmark.artifacts import write_strict_json  # noqa: E402
from run_real_fault_benchmark import PROTOCOL_HASH  # noqa: E402


PAPER_CODE = (
    ROOT / "论文复现" / "Temperature Prediction and Fault Warning of High-Speed Shaft of Wind Turbine Gearbox Based on Hybrid Deep Learning Model" / "STA_BKA.py",
    ROOT / "论文复现" / "Trans GAN-WT anomaly detection model for wind turbine time series" / "TransGAN_WT.py",
    ROOT / "论文复现" / "Wind Turbine Gearbox Fault Detection Based on Statistical Learning" / "StatisticalGearboxRF.py",
    ROOT / "论文复现" / "Failure warning for offshore wind turbines based on Autoregressive models" / "ARIMA_LASSO_EWMA.py",
    ROOT / "论文复现" / "Fault detection in wind turbines using health index monitoring with variational autoencoders" / "wedowind-challenge-ASCE-EMI" / "Solution_ID8" / "src" / "model.py",
    ROOT / "论文复现" / "Wind turbine condition monitoring based on intra- and inter-farm federated learning" / "FL-Wind-NBM" / "federated_learning" / "FederatedAlgorithms.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    sources = [
        path for path in BENCHMARK.rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    sources.extend(PAPER_CODE)
    sources.append(BENCHMARK / "vendor" / "AFFILIATION_REFERENCE.json")
    artifacts = [
        path for path in OUTPUT.iterdir()
        if path.is_file() and path.name != "完整性清单.json"
    ]
    artifacts.extend(sorted((OUTPUT / "runs").glob("*/*/run_record.json")))
    artifacts.extend(sorted((OUTPUT / "runs").glob("*/*/calibration.json")))
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"关键源文件缺失: {missing}")
    unique = sorted(set(sources + artifacts), key=lambda path: str(path).casefold())
    rows = [
        {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "kind": "source" if path in sources else "artifact",
        }
        for path in unique
    ]
    payload = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": PROTOCOL_HASH,
        "scope": "current_post_run_verification_snapshot",
        "note": "运行时 code_hash 保留在各 run_record；本清单记录最终核验时的当前源代码与关键交付物。",
        "file_count": len(rows),
        "files": rows,
    }
    write_strict_json(OUTPUT / "完整性清单.json", payload)
    print(json.dumps({"file_count": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
