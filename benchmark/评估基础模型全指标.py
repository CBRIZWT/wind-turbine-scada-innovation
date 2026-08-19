# -*- coding: utf-8 -*-
"""评估基础模型全指标 — 把 47 个基础/组合快速实验模型接入中央 realfault 基准的
**完整指标套件**(与 14 篇论文复现模型逐字一致的 evaluate_full_metrics + equal4 校准)。

动机(用户任务4/5): 论文复现走中央基准 full_metrics(53 指标);基础模型的
快速实验此前只算事件级子集。本桥接器**不重训练**,只复用已存的模型分数,
按中央协议(val 上 201 分位 equal4 选阈冻结 → test 评一次)重算全指标,
使两套实验的评价指标完全一致,可同表对比。

对齐: 基础模型 allmetrics_v3 结果已剔除 -1(ignore)行且重排;raw 预处理仍含
-1 硬边界。用 (turbine,timestamp) 唯一键把基础分数散射回 raw 行(ignore→NaN),
于是标签/时间戳/机组/事件表与论文模型所用中央 bundle **逐行一致**,协议同源。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(r"E:\创新\benchmark")
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from realfault_benchmark.calibration import select_equal4_calibration  # noqa: E402
from realfault_benchmark.full_metrics import evaluate_full_metrics  # noqa: E402

RAW = Path(r"E:\创新\SCADA数据集\数据预处理")
BASE = Path(r"E:\创新\快速实验\快速实验结果_真实故障_allmetrics_v3")
OUT = Path(r"E:\创新\论文复现\复现基准_2026-07-19")
FARMS = ("kelmarsh", "penmanshiel")
SEED = 20260719


def _raw_split(farm: str, split: str) -> dict:
    d = RAW / f"{farm}__realfault"
    return {
        "ts": np.load(d / f"timestamps_{split}.npy").astype(np.int64),
        "turb": np.load(d / f"turbines_{split}.npy").astype(str),
        "y": np.load(d / f"{split}_labels.npy").astype(np.int8),
    }


def _event_table(farm: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{farm}__realfault" / "event_table.csv")


def _scatter(raw: dict, base_dir: Path, split: str) -> np.ndarray | None:
    """把基础模型(重排、去 -1)的分数按 (turb,ts) 键散射回 raw 行序,ignore→NaN。"""
    sc_path = base_dir / f"score_{split}.npy"
    if not sc_path.exists():
        return None
    bsc = np.load(sc_path).astype(float)
    bts = np.load(base_dir / f"timestamps_{split}.npy").astype(np.int64)
    btu = np.load(base_dir / f"turbines_{split}.npy").astype(str)
    raw_df = pd.DataFrame({"turb": raw["turb"], "ts": raw["ts"], "ord": np.arange(len(raw["ts"]))})
    base_df = pd.DataFrame({"turb": btu, "ts": bts, "score": bsc})
    merged = base_df.merge(raw_df, on=["turb", "ts"], how="left", validate="one_to_one")
    if merged["ord"].isna().any():
        raise ValueError(f"{base_dir.name}/{split}: 有基础行无法在 raw 中匹配 (turb,ts)")
    full = np.full(len(raw["ts"]), np.nan, dtype=float)
    full[merged["ord"].to_numpy(dtype=np.int64)] = merged["score"].to_numpy(dtype=float)
    return full


def evaluate_one(farm: str, base_dir: Path, raw_val: dict, raw_test: dict,
                 events: pd.DataFrame) -> dict | None:
    s_val = _scatter(raw_val, base_dir, "val")
    s_test = _scatter(raw_test, base_dir, "test")
    if s_val is None or s_test is None:
        return None
    calib = select_equal4_calibration(
        raw_val["y"], s_val, raw_val["ts"], raw_val["turb"], events,
        model_id=base_dir.name, dataset_id=f"scada:{farm}:realfault",
        validation_hash=f"base_bridge:{farm}",
    )
    metrics = evaluate_full_metrics(
        raw_test["y"], s_test, raw_test["ts"], raw_test["turb"], events,
        threshold=calib.threshold, polarity=calib.polarity, split="test",
        score_semantics="anomaly_score",
    )
    return {"model_id": base_dir.name, "farm": farm, "family": "基础/组合模型",
            "polarity": calib.polarity, "threshold": float(calib.threshold),
            "val_local_equal4": float(calib.validation_metrics["local_equal4_score"]),
            **metrics}


def main() -> int:
    rows: list[dict] = []
    for farm in FARMS:
        raw_val, raw_test = _raw_split(farm, "val"), _raw_split(farm, "test")
        events = _event_table(farm)
        model_dirs = sorted([p for p in (BASE / farm).iterdir() if p.is_dir()])
        for base_dir in model_dirs:
            t0 = time.time()
            try:
                rec = evaluate_one(farm, base_dir, raw_val, raw_test, events)
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {farm}/{base_dir.name}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if rec is None:
                continue
            (base_dir / "full_metrics_central.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
            rows.append(rec)
            print(f"  {farm}/{base_dir.name}: equal4={rec['local_equal4_score']:.4f} "
                  f"one2one_eF1={rec['one_to_one_event_f1']:.4f} pa_f1={rec['pa_f1_appendix']} "
                  f"acc={rec['accuracy']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "基础模型全指标_中央口径.csv", index=False, encoding="utf-8-sig")
    print(f"\n共 {len(df)} 行 (farm×模型) → {OUT / '基础模型全指标_中央口径.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
