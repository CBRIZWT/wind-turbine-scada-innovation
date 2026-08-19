from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


AFFILIATION_REFERENCE_COMMIT = "8d8449858096bbade6a6e70848d05c9cc9b846fe"
_VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "affiliation-metrics-py"


def _author_api():
    if not (_VENDOR_ROOT / "affiliation" / "metrics.py").is_file():
        raise RuntimeError("Huet affiliation 作者参考实现未固定到 benchmark/vendor")
    if str(_VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(_VENDOR_ROOT))
    from affiliation.generics import convert_vector_to_events
    from affiliation.metrics import pr_from_events
    return convert_vector_to_events, pr_from_events


def affiliation_prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """调用 Huet et al. KDD 2022 作者 Python 参考实现（固定 commit）。"""
    true = (np.asarray(y_true).astype(int) == 1).astype(int).tolist()
    pred = (np.asarray(y_pred).astype(int) == 1).astype(int).tolist()
    if len(true) != len(pred):
        raise ValueError("y_true/y_pred 必须等长")
    convert, evaluate = _author_api()
    gt_events = convert(true)
    pred_events = convert(pred)
    if not gt_events:
        # 作者实现将无 ground truth 视为未定义；中央评测器用 0 + 明确状态承载。
        return {"affiliation_precision": 0.0, "affiliation_recall": 0.0, "affiliation_f1": 0.0}
    if not pred_events:
        return {"affiliation_precision": 0.0, "affiliation_recall": 0.0, "affiliation_f1": 0.0}
    result = evaluate(pred_events, gt_events, (0, len(true)))
    precision = float(result["precision"])
    recall = float(result["recall"])
    if not math.isfinite(precision):
        precision = 0.0
    if not math.isfinite(recall):
        recall = 0.0
    f1 = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))
    return {
        "affiliation_precision": precision,
        "affiliation_recall": recall,
        "affiliation_f1": f1,
    }


def turbine_macro_affiliation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray,
    turbines: np.ndarray,
) -> dict[str, Any]:
    """逐机组后宏平均；无真实事件机组按作者定义标记为不适用并跳过。"""
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    ts = np.asarray(timestamps, dtype=np.int64)
    turb = np.asarray(turbines).astype(str)
    if not (len(y) == len(pred) == len(ts) == len(turb)):
        raise ValueError("affiliation 输入必须等长")
    rows = []
    skipped = []
    for name in np.unique(turb):
        idx = np.flatnonzero(turb == name)
        idx = idx[np.argsort(ts[idx], kind="stable")]
        yt = (y[idx] == 1).astype(np.int8)
        yp = np.where(y[idx] == -1, 0, pred[idx]).astype(np.int8)
        if not yt.any():
            skipped.append(str(name))
            continue
        rows.append(affiliation_prf(yt, yp))
    if not rows:
        return {
            "affiliation_precision": None, "affiliation_recall": None,
            "affiliation_f1": None, "affiliation_status": "undefined_no_ground_truth_events",
            "affiliation_turbines_used": 0, "affiliation_turbines_skipped": skipped,
            "affiliation_reference_commit": AFFILIATION_REFERENCE_COMMIT,
        }
    return {
        "affiliation_precision": float(np.mean([r["affiliation_precision"] for r in rows])),
        "affiliation_recall": float(np.mean([r["affiliation_recall"] for r in rows])),
        "affiliation_f1": float(np.mean([r["affiliation_f1"] for r in rows])),
        "affiliation_status": "ok_author_reference_turbine_macro",
        "affiliation_turbines_used": len(rows),
        "affiliation_turbines_skipped": skipped,
        "affiliation_reference_commit": AFFILIATION_REFERENCE_COMMIT,
    }

