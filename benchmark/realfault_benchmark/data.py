from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import DatasetBundle


TRUE_FAULT_FARMS = ("kelmarsh", "penmanshiel")
V1_LABEL_FILES = ("train_labels.npy", "val_labels.npy", "test_labels.npy")


def _load(path: Path, *, mmap_mode: str | None = "r") -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _aggregate_file_hash(directory: Path, names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        path = directory / name
        digest.update(name.encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _assert_equal_subset(
    compact: np.ndarray,
    full: np.ndarray,
    indices: np.ndarray,
    name: str,
    *,
    chunk_rows: int = 100_000,
) -> None:
    if len(compact) != len(indices):
        raise ValueError(f"{name}: compact 长度与 label0 行数不一致")
    for start in range(0, len(indices), chunk_rows):
        part = indices[start:start + chunk_rows]
        if not np.array_equal(np.asarray(compact[start:start + len(part)]), np.asarray(full[part])):
            raise ValueError(f"{name}: train_normal 不是 train_supervised[label==0] 的精确子集")


def load_realfault_bundle(
    preprocessed_root: str | Path,
    farm: str,
    *,
    variant: str | None = None,
    verify_file_hashes: bool = True,
    verify_normal_subset: bool = True,
) -> DatasetBundle:
    """显式加载 realfault v1；永不自动发现 `labels_v2` 或 broad B 标签。

    目录解析 (2026-07-26 单一真源对齐): 主口径 real_fault_wl 产物落 ``<farm>/``
    (``preprocess_variant=""``); 历史变体 ``<farm>__realfault`` 仍兼容。
    显式传 ``variant`` 时只认 ``<farm>__<variant>``, 不做回退。
    """
    farm = str(farm)
    if farm not in TRUE_FAULT_FARMS:
        raise ValueError(f"真实故障基准只接受 {TRUE_FAULT_FARMS}，收到 {farm}")
    root = Path(preprocessed_root)
    if variant is not None:
        v = str(variant).strip()
        candidates = [(f"{farm}__{v}" if v else farm, v)]
    else:
        # 优先历史变体 (向后兼容既有基准产物), 再回退主口径单一真源目录
        candidates = [(f"{farm}__realfault", "realfault"), (farm, "")]
    for _name, _expect in candidates:
        _dir = root / _name
        if (_dir / "meta.json").exists():
            directory, expected_variant = _dir, _expect
            break
    else:
        raise FileNotFoundError(
            f"未找到 {farm} 的真实故障预处理产物; 已查: "
            + ", ".join(str(root / n) for n, _ in candidates)
        )
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(meta.get("preprocess_variant") or "") != expected_variant:
        raise ValueError(
            f"preprocess_variant 必须严格为 {expected_variant!r} (目录 {directory.name}), "
            f"实际 {meta.get('preprocess_variant')!r}"
        )
    primary = meta.get("primary_label") or {}
    source_files = tuple(primary.get("source_files") or ())
    if source_files != V1_LABEL_FILES or str(primary.get("name")) != "real_fault_wl":
        raise ValueError(
            "meta.primary_label 未明确指向 realfault v1 的 train/val/test_labels.npy；拒绝 v2/B 污染"
        )

    train_X = _load(directory / "train.npy")
    train_y = _load(directory / "train_labels.npy")
    train_ts = _load(directory / "timestamps_train.npy")
    train_turb = _load(directory / "turbines_train.npy")
    sup_X = _load(directory / "train_sup.npy")
    sup_y = _load(directory / "train_sup_labels.npy")
    sup_ts = _load(directory / "timestamps_train_sup.npy")
    sup_turb = _load(directory / "turbines_train_sup.npy")
    sup_gap = _load(directory / "gap_mask_train.npy")
    val_X = _load(directory / "val.npy")
    val_y = _load(directory / "val_labels.npy")
    val_ts = _load(directory / "timestamps_val.npy")
    val_turb = _load(directory / "turbines_val.npy")
    val_gap = _load(directory / "gap_mask_val.npy")
    test_X = _load(directory / "test.npy")
    test_y = _load(directory / "test_labels.npy")
    test_ts = _load(directory / "timestamps_test.npy")
    test_turb = _load(directory / "turbines_test.npy")
    test_gap = _load(directory / "gap_mask_test.npy")
    # 固定的不可跨越边界 = 物理 gap ∪ v1 ignore。它在 bundle 创建时冻结，
    # 后续打乱/修改隐藏评测标签不会重算窗口索引，模型也看不到标签本身。
    sup_boundary = np.asarray(sup_gap, dtype=bool) | (np.asarray(sup_y) == -1)
    val_boundary = np.asarray(val_gap, dtype=bool) | (np.asarray(val_y) == -1)
    test_boundary = np.asarray(test_gap, dtype=bool) | (np.asarray(test_y) == -1)
    normal_source_ids = np.flatnonzero(np.asarray(sup_y) == 0).astype(np.int64)
    if verify_normal_subset:
        _assert_equal_subset(train_X, sup_X, normal_source_ids, "features")
        _assert_equal_subset(train_ts, sup_ts, normal_source_ids, "timestamps")
        _assert_equal_subset(train_turb, sup_turb, normal_source_ids, "turbines")
    if not np.all(np.asarray(train_y) == 0):
        raise ValueError("train_labels.npy 不是全零 normal-only 标签")

    event_table = pd.read_csv(directory / "event_table.csv")
    critical_names = [
        "meta.json", "event_table.csv", "train.npy", "train_labels.npy",
        "timestamps_train.npy", "turbines_train.npy", "train_sup.npy",
        "train_sup_labels.npy", "timestamps_train_sup.npy", "turbines_train_sup.npy",
        "gap_mask_train.npy",
        "val.npy", "val_labels.npy", "timestamps_val.npy", "turbines_val.npy",
        "gap_mask_val.npy",
        "test.npy", "test_labels.npy", "timestamps_test.npy", "turbines_test.npy",
        "gap_mask_test.npy",
    ]
    if verify_file_hashes:
        file_hash = _aggregate_file_hash(directory, critical_names)
    else:
        file_hash = "unverified:" + hashlib.sha256(
            (str(meta.get("split_hash")) + ":" + str(meta.get("cols_hash"))).encode("utf-8")
        ).hexdigest()
    feature_names = tuple(meta.get("cols") or ())
    return DatasetBundle.from_arrays(
        dataset="wind_turbine_scada_real_fault",
        farm=farm,
        variant="realfault",
        feature_names=feature_names,
        train_normal_X=train_X,
        train_normal_labels=train_y,
        train_normal_timestamps=train_ts,
        train_normal_turbines=train_turb,
        train_normal_row_ids=normal_source_ids,
        train_normal_gap_mask=sup_boundary[normal_source_ids],
        train_supervised_X=sup_X,
        train_supervised_labels=sup_y,
        train_supervised_timestamps=sup_ts,
        train_supervised_turbines=sup_turb,
        train_supervised_row_ids=np.arange(len(sup_y), dtype=np.int64),
        train_supervised_gap_mask=sup_boundary,
        val_X=val_X,
        val_labels=val_y,
        val_timestamps=val_ts,
        val_turbines=val_turb,
        val_row_ids=np.arange(len(val_y), dtype=np.int64),
        val_gap_mask=val_boundary,
        test_X=test_X,
        test_labels=test_y,
        test_timestamps=test_ts,
        test_turbines=test_turb,
        test_row_ids=np.arange(len(test_y), dtype=np.int64),
        test_gap_mask=test_boundary,
        event_table=event_table,
        split_hash=str(meta.get("split_hash", "")),
        feature_hash=str(meta.get("cols_hash", "")),
        file_hash=file_hash,
    )


def merged_events_for_split(event_table: Any, split: str, *, merge_hours: float = 72.0) -> pd.DataFrame:
    """按同机组、指定 split 和 72h 间隔合并原始事件为 episode。"""
    table = pd.DataFrame(event_table).copy()
    if "split" in table:
        table = table[table["split"].astype(str) == str(split)].copy()
    if table.empty:
        return pd.DataFrame(columns=["turbine", "start", "end", "split", "messages"])
    needed = {"turbine", "start"}
    if not needed.issubset(table.columns):
        raise ValueError(f"event_table 缺字段: {sorted(needed - set(table.columns))}")
    table["turbine"] = table["turbine"].astype(str)
    table["start"] = pd.to_datetime(table["start"], utc=True, errors="coerce")
    table["end"] = pd.to_datetime(table.get("end", table["start"]), utc=True, errors="coerce")
    table["end"] = table["end"].fillna(table["start"])
    table = table.dropna(subset=["start"]).sort_values(["turbine", "start"], kind="stable")
    gap = pd.Timedelta(hours=float(merge_hours))
    episodes: list[dict[str, Any]] = []
    for turbine, group in table.groupby("turbine", sort=False):
        current: dict[str, Any] | None = None
        for row in group.itertuples(index=False):
            start = row.start
            end = max(row.end, row.start)
            message = str(getattr(row, "message", ""))
            if current is None or start > current["end"] + gap:
                if current is not None:
                    episodes.append(current)
                current = {"turbine": str(turbine), "start": start, "end": end,
                           "split": str(split), "messages": [message]}
            else:
                current["end"] = max(current["end"], end)
                current["messages"].append(message)
        if current is not None:
            episodes.append(current)
    result = pd.DataFrame(episodes)
    if not result.empty:
        result["messages"] = result["messages"].map(lambda values: " | ".join(values))
    return result
