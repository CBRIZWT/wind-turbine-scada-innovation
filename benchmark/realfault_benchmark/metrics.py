from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _events_frame(events: Any, *, split: str | None = None, merge_hours: float = 72.0) -> pd.DataFrame:
    table = pd.DataFrame(events).copy()
    aliases = {"_turbine": "turbine", "Timestamp start": "start", "Timestamp end": "end"}
    table = table.rename(columns={k: v for k, v in aliases.items() if k in table.columns})
    if "split" in table.columns:
        if split is None:
            raise ValueError("原始 event_table 含 split 字段；评测必须显式指定 split")
        table = table[table["split"].astype(str) == str(split)].copy()
    for column in ("turbine", "start"):
        if column not in table.columns:
            if table.empty:
                table[column] = pd.Series(dtype="object")
            else:
                raise ValueError(f"event table 缺少 {column}")
    if "end" not in table:
        table["end"] = table["start"]
    table["turbine"] = table["turbine"].astype(str)
    table["start"] = pd.to_datetime(table["start"], utc=True, errors="coerce")
    table["end"] = pd.to_datetime(table["end"], utc=True, errors="coerce")
    table["end"] = table["end"].fillna(table["start"])
    table = table.dropna(subset=["start"]).sort_values(["turbine", "start"], kind="stable")
    if table.empty:
        return table.reset_index(drop=True)
    gap = pd.Timedelta(hours=float(merge_hours))
    merged: list[dict[str, Any]] = []
    for turbine, group in table.groupby("turbine", sort=False):
        current: dict[str, Any] | None = None
        for row in group.itertuples(index=False):
            start = row.start
            end = max(row.end, row.start)
            if current is None or start > current["end"] + gap:
                if current is not None:
                    merged.append(current)
                current = {"turbine": str(turbine), "start": start, "end": end}
            else:
                current["end"] = max(current["end"], end)
        if current is not None:
            merged.append(current)
    return pd.DataFrame(merged, columns=["turbine", "start", "end"])


def _event_windows(
    events: Any,
    horizon: timedelta,
    *,
    split: str | None = None,
) -> list[tuple[str, int, int]]:
    table = _events_frame(events, split=split)
    horizon_ns = int(horizon.total_seconds() * 1_000_000_000)
    return [
        (str(row.turbine), int(row.start.value) - horizon_ns, int(row.start.value))
        for row in table.itertuples(index=False)
    ]


def prepare_equal4_context(
    labels: np.ndarray,
    timestamps: np.ndarray,
    turbines: np.ndarray,
    events: Any,
    *,
    split: str | None,
    horizon: timedelta,
) -> dict[str, Any]:
    """一次构造事件索引、早警窗掩码和逐机组时间顺序，供阈值网格复用。"""
    y = np.asarray(labels, dtype=np.int8)
    ts = np.asarray(timestamps, dtype=np.int64)
    turb = np.asarray(turbines).astype(str)
    windows = _event_windows(events, horizon, split=split)
    window_mask = np.zeros(len(y), dtype=bool)
    event_indices: list[np.ndarray] = []
    for name, start, end in windows:
        idx = np.flatnonzero((turb == name) & (ts >= start) & (ts < end))
        event_indices.append(idx)
        window_mask[idx] = True
    orders = []
    for name in np.unique(turb):
        idx = np.flatnonzero(turb == name)
        orders.append(idx[np.argsort(ts[idx], kind="stable")])
    return {
        "windows": windows,
        "event_indices": event_indices,
        "event_window_mask": window_mask,
        "orders": orders,
    }


def predictions_from_scores(scores: np.ndarray, threshold: float, polarity: str = "positive") -> np.ndarray:
    raw = np.asarray(scores, dtype=float)
    if polarity not in {"positive", "negative"}:
        raise ValueError("polarity 必须为 positive/negative")
    oriented = raw if polarity == "positive" else -raw
    return (np.isfinite(oriented) & (oriented >= float(threshold))).astype(np.int8)


def false_alarm_burden(
    labels: np.ndarray,
    predictions: np.ndarray,
    timestamps: np.ndarray,
    turbines: np.ndarray,
    events: Any,
    *,
    horizon: timedelta = timedelta(hours=12),
    merge_gap: timedelta = timedelta(minutes=40),
    nominal_cadence: timedelta = timedelta(minutes=10),
    split: str | None = None,
    prepared: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    """按机组统计假报警段；非健康行和早警窗报警均强制断段。"""
    y = np.asarray(labels, dtype=np.int8)
    pred = np.asarray(predictions, dtype=np.int8)
    ts = np.asarray(timestamps, dtype=np.int64)
    turb = np.asarray(turbines).astype(str)
    if not (len(y) == len(pred) == len(ts) == len(turb)):
        raise ValueError("labels/predictions/timestamps/turbines 必须等长")
    prepared = prepared or prepare_equal4_context(y, ts, turb, events, split=split, horizon=horizon)
    window_mask = np.asarray(prepared["event_window_mask"], dtype=bool)
    merge_ns = int(merge_gap.total_seconds() * 1_000_000_000)
    false_segments = 0
    eligible_global = (y == 0) & (pred == 1) & (~window_mask)
    for order in prepared["orders"]:
        local_eligible = np.flatnonzero(eligible_global[order])
        if not local_eligible.size:
            continue
        false_segments += 1
        if local_eligible.size == 1:
            continue
        ordered_y = y[order]
        ordered_ts = ts[order]
        nonhealthy_prefix = np.concatenate([[0], np.cumsum((ordered_y != 0).astype(np.int64))])
        previous = local_eligible[:-1]
        current = local_eligible[1:]
        nonhealthy_between = nonhealthy_prefix[current] - nonhealthy_prefix[previous + 1]
        gap_break = ordered_ts[current] - ordered_ts[previous] > merge_ns
        false_segments += int(np.count_nonzero(gap_break | (nonhealthy_between > 0)))

    cadence_days = nominal_cadence.total_seconds() / 86400.0
    healthy_days = float(np.count_nonzero(y == 0) * cadence_days)
    far = float(false_segments / healthy_days) if healthy_days > 0 else 0.0
    return {
        "false_alarm_segments": int(false_segments),
        "healthy_turbine_days": healthy_days,
        "false_alarm_segments_per_turbine_day": far,
    }


def point_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    *,
    score_semantics: str = "anomaly_score",
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    score = np.asarray(scores, dtype=float)
    pred = np.asarray(predictions, dtype=np.int8)
    if score_semantics not in {"anomaly_score", "probability"}:
        raise ValueError("score_semantics 只能为 anomaly_score/probability")
    if probabilities is not None and len(np.asarray(probabilities)) != len(y):
        raise ValueError("probabilities 必须与 labels 等长")
    valid = y != -1
    yv = (y[valid] == 1).astype(np.int8)
    pv = pred[valid]
    finite = valid & np.isfinite(score)
    out: dict[str, Any] = {
        "valid_count": int(valid.sum()), "ignore_count": int((~valid).sum()),
        "log_loss": None, "brier": None, "probability_status": "invalid_probability",
    }
    if not len(yv):
        return {**out, "accuracy": None, "balanced_accuracy": None, "precision": None,
                "recall": None, "specificity": None, "f1": None, "mcc": None,
                "roc_auc": None, "pr_auc": None, "tn": 0, "fp": 0, "fn": 0, "tp": 0}
    tn, fp, fn, tp = confusion_matrix(yv, pv, labels=[0, 1]).ravel()
    out.update(
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        accuracy=float(accuracy_score(yv, pv)),
        balanced_accuracy=(float(balanced_accuracy_score(yv, pv)) if np.unique(yv).size == 2 else None),
        precision=float(precision_score(yv, pv, zero_division=0)),
        recall=float(recall_score(yv, pv, zero_division=0)),
        specificity=float(tn / (tn + fp)) if (tn + fp) else None,
        f1=float(f1_score(yv, pv, zero_division=0)),
        mcc=float(matthews_corrcoef(yv, pv)) if len(yv) else None,
    )
    yf = (y[finite] == 1).astype(np.int8)
    sf = score[finite]
    if len(yf) and np.unique(yf).size == 2:
        out["roc_auc"] = float(roc_auc_score(yf, sf))
        out["pr_auc"] = float(average_precision_score(yf, sf))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    if score_semantics == "probability" and probabilities is not None:
        probability = np.asarray(probabilities, dtype=float)
        pv_probability = probability[valid]
        if (
            probability.ndim == 1
            and np.isfinite(pv_probability).all()
            and np.all((pv_probability >= 0.0) & (pv_probability <= 1.0))
        ):
            out["log_loss"] = float(log_loss(yv, pv_probability, labels=[0, 1]))
            out["brier"] = float(brier_score_loss(yv, pv_probability))
            out["probability_status"] = "valid_probability"
    return out


def evaluate_equal4(
    labels: np.ndarray,
    scores: np.ndarray,
    timestamps: np.ndarray,
    turbines: np.ndarray,
    events: Any,
    *,
    threshold: float,
    polarity: str = "positive",
    horizon: timedelta = timedelta(hours=12),
    merge_gap: timedelta = timedelta(minutes=40),
    nominal_cadence: timedelta = timedelta(minutes=10),
    split: str | None = None,
    prepared: dict[str, Any] | None = None,
    include_point_metrics: bool = True,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    raw = np.asarray(scores, dtype=float)
    ts = np.asarray(timestamps, dtype=np.int64)
    turb = np.asarray(turbines).astype(str)
    if not (len(y) == len(raw) == len(ts) == len(turb)):
        raise ValueError("labels/scores/timestamps/turbines 必须等长")
    pred = predictions_from_scores(raw, threshold, polarity)
    oriented = raw if polarity == "positive" else -raw
    prepared = prepared or prepare_equal4_context(y, ts, turb, events, split=split, horizon=horizon)
    windows = prepared["windows"]
    horizon_ns = max(1, int(horizon.total_seconds() * 1_000_000_000))
    detected = 0
    earliness: list[float] = []
    for (_, _, end), idx in zip(windows, prepared["event_indices"]):
        hit_times = ts[idx[pred[idx] == 1]]
        if hit_times.size:
            detected += 1
            earliest = int(hit_times.min())
            earliness.append(float(np.clip((end - earliest) / horizon_ns, 0.0, 1.0)))
        else:
            earliness.append(0.0)
    event_count = len(windows)
    event_recall = float(detected / event_count) if event_count else 0.0
    valid_alarms = (pred == 1) & (y != -1)
    alarm_count = int(valid_alarms.sum())
    alarm_precision = float(np.count_nonzero(y[valid_alarms] == 1) / alarm_count) if alarm_count else 0.0
    burden = false_alarm_burden(
        y, pred, ts, turb, events, horizon=horizon, merge_gap=merge_gap,
        nominal_cadence=nominal_cadence, split=split, prepared=prepared,
    )
    far = float(burden["false_alarm_segments_per_turbine_day"])
    far_reward = max(0.0, 1.0 - 7.0 * far)
    mean_earliness = float(np.mean(earliness)) if earliness else 0.0
    local_score = float((event_recall + alarm_precision + mean_earliness + far_reward) / 4.0)
    result = {
        "local_equal4_score": local_score,
        "event_count": int(event_count),
        "event_detected": int(detected),
        "event_recall": event_recall,
        "alarm_count": alarm_count,
        "alarm_point_precision": alarm_precision,
        "mean_normalized_earliness": mean_earliness,
        "earliness_by_event": earliness,
        "far_reward": far_reward,
        **burden,
    }
    if include_point_metrics:
        result.update(point_metrics(y, oriented, pred))
    return result


def _ranges(binary: np.ndarray) -> list[tuple[int, int]]:
    x = np.asarray(binary).astype(bool)
    if not x.size or not x.any():
        return []
    diff = np.diff(x.astype(np.int8))
    starts = (np.flatnonzero(diff == 1) + 1).tolist()
    ends = np.flatnonzero(diff == -1).tolist()
    if x[0]:
        starts.insert(0, 0)
    if x[-1]:
        ends.append(len(x) - 1)
    return list(zip(starts, ends))


def _range_reward(target: tuple[int, int], other: list[tuple[int, int]], alpha: float) -> float:
    start, end = target
    length = end - start + 1
    overlaps = []
    overlap_points = 0
    for left, right in other:
        amount = max(0, min(end, right) - max(start, left) + 1)
        if amount:
            overlaps.append(amount)
            overlap_points += amount
    existence = 1.0 if overlaps else 0.0
    cardinality = 1.0 / len(overlaps) if overlaps else 0.0
    overlap_reward = cardinality * (overlap_points / length)
    return float(alpha * existence + (1.0 - alpha) * overlap_reward)


def tatbul_range_prf(y_true: np.ndarray, y_pred: np.ndarray, *, alpha: float = 0.0) -> dict[str, float]:
    """Tatbul et al. (NeurIPS 2018): existence+overlap+cardinality+flat bias。

    ``alpha`` 仅是 range-recall 的 existence 权重；论文正式 precision 定义不含
    existence 奖励，因此其 alpha 固定为 0。
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须位于 [0,1]")
    actual = _ranges(np.asarray(y_true) == 1)
    predicted = _ranges(np.asarray(y_pred) == 1)
    if not actual and not predicted:
        return {"range_precision": 1.0, "range_recall": 1.0, "range_f1": 1.0}
    recall = float(np.mean([_range_reward(r, predicted, alpha) for r in actual])) if actual else 0.0
    # precision 对预测区间交换角色；通常 alpha=0，即不存在额外 existence 奖励。
    precision = float(np.mean([_range_reward(r, actual, 0.0) for r in predicted])) if predicted else 0.0
    f1 = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))
    return {"range_precision": precision, "range_recall": recall, "range_f1": f1}
