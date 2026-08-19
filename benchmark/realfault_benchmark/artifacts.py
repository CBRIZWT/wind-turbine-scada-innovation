from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _join_path(head: str, tail: str) -> str:
    if not tail:
        return head
    return f"{head}{tail}" if tail.startswith("[") else f"{head}.{tail}"


def _sanitize(value: Any) -> tuple[Any, dict[str, str]]:
    """递归转为严格 JSON 值，并返回相对当前容器的非有限路径。"""

    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        statuses: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            child, child_statuses = _sanitize(raw_value)
            clean[key] = child
            for path, status in child_statuses.items():
                statuses[_join_path(key, path)] = status

        if statuses:
            existing = clean.get("serialization_status")
            merged = dict(existing) if isinstance(existing, dict) else {}
            merged.update(statuses)
            clean["serialization_status"] = merged

        # 兼容既有 RunRecord 格式：metrics 的直接非有限字段在父对象同步登记。
        metrics = value.get("metrics")
        if isinstance(metrics, dict):
            metric_status = clean.get("metric_status")
            metric_status = dict(metric_status) if isinstance(metric_status, dict) else {}
            for key, raw in metrics.items():
                if isinstance(raw, (float, np.floating)) and not math.isfinite(float(raw)):
                    metric_status[str(key)] = "non_finite"
            if metric_status:
                clean["metric_status"] = metric_status
        return clean, statuses

    if isinstance(value, (list, tuple)):
        clean_list: list[Any] = []
        statuses: dict[str, str] = {}
        for index, raw in enumerate(value):
            child, child_statuses = _sanitize(raw)
            clean_list.append(child)
            for path, status in child_statuses.items():
                statuses[_join_path(f"[{index}]", path)] = status
        return clean_list, statuses
    if isinstance(value, np.ndarray):
        return _sanitize(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value), {}
    if isinstance(value, (np.integer,)):
        return int(value), {}
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return (number, {}) if math.isfinite(number) else (None, {"": "non_finite"})
    if isinstance(value, np.str_):
        return str(value), {}
    return value, {}


def strict_json_dumps(payload: Any, *, indent: int = 2) -> str:
    clean, _ = _sanitize(payload)
    return json.dumps(clean, ensure_ascii=False, indent=indent, allow_nan=False, sort_keys=True)


def write_strict_json(path: str | Path, payload: Any) -> None:
    """调用方应通过 apply_patch 管理源码；本函数仅用于运行产物。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(strict_json_dumps(payload) + "\n", encoding="utf-8")
