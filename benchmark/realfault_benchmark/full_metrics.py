from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    _events_frame, evaluate_equal4, point_metrics,
    predictions_from_scores, tatbul_range_prf,
)
from .reference_metrics import turbine_macro_affiliation


def _validate_aligned(*arrays: np.ndarray) -> None:
    lengths = [len(np.asarray(value)) for value in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"输入必须逐行对齐，收到长度 {lengths}")


def turbine_macro_range(
    labels: np.ndarray,
    predictions: np.ndarray,
    timestamps: np.ndarray,
    turbines: np.ndarray,
    *,
    alpha: float = 0.0,
) -> dict[str, Any]:
    """逐机组计算 Tatbul range P/R/F1 后宏平均。

    ``y=-1`` 的行是不可评价区间，会被强制当作无报警边界；没有真实正例的
    机组不参与宏平均，并被显式列入 ``range_turbines_skipped``。这样既不会把
    两台机组首尾拼成一个 range，也不会用大量无故障机组稀释召回率。
    """

    y = np.asarray(labels, dtype=np.int8)
    pred = np.asarray(predictions, dtype=np.int8)
    ts = np.asarray(timestamps, dtype=np.int64)
    turb = np.asarray(turbines).astype(str)
    _validate_aligned(y, pred, ts, turb)

    rows: list[dict[str, float]] = []
    skipped: list[str] = []
    for name in np.unique(turb):
        index = np.flatnonzero(turb == name)
        index = index[np.argsort(ts[index], kind="stable")]
        local_y = (y[index] == 1).astype(np.int8)
        local_pred = np.where(y[index] == -1, 0, pred[index]).astype(np.int8)
        if not local_y.any():
            skipped.append(str(name))
            continue
        rows.append(tatbul_range_prf(local_y, local_pred, alpha=alpha))

    if not rows:
        return {
            "range_precision": None,
            "range_recall": None,
            "range_f1": None,
            "range_status": "undefined_no_ground_truth_ranges",
            "range_turbines_used": 0,
            "range_turbines_skipped": skipped,
        }
    return {
        "range_precision": float(np.mean([row["range_precision"] for row in rows])),
        "range_recall": float(np.mean([row["range_recall"] for row in rows])),
        "range_f1": float(np.mean([row["range_f1"] for row in rows])),
        "range_status": "ok_turbine_macro_flat_bias",
        "range_turbines_used": len(rows),
        "range_turbines_skipped": skipped,
    }


def _alarm_segments(
    predictions: np.ndarray,
    timestamps: np.ndarray,
    turbines: np.ndarray,
    merge_gap: timedelta,
    labels: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray]]:
    """把实际报警时刻合并为机组内的段，不跨机组或 ignore 行。

    段内保留每一个真实报警时刻，后续事件匹配不能仅凭首尾包络判定。
    ``y=-1`` 不进入分母，并作为硬边界；``y=1`` 是真实早警窗，仍是可评价行。
    """

    pred = np.asarray(predictions, dtype=np.int8)
    ts = np.asarray(timestamps, dtype=np.int64)
    turb = np.asarray(turbines).astype(str)
    _validate_aligned(pred, ts, turb)
    y = None if labels is None else np.asarray(labels, dtype=np.int8)
    if y is not None:
        _validate_aligned(pred, y)
    gap_ns = int(merge_gap.total_seconds() * 1_000_000_000)
    segments: list[tuple[str, np.ndarray]] = []
    for name in np.unique(turb):
        order = np.flatnonzero(turb == name)
        order = order[np.argsort(ts[order], kind="stable")]
        current_times: list[int] = []
        previous_alarm: int | None = None
        hard_boundary = False
        for row in order:
            if y is not None and y[row] == -1:
                if current_times:
                    segments.append((str(name), np.asarray(current_times, dtype=np.int64)))
                    current_times = []
                previous_alarm = None
                hard_boundary = True
                continue
            if pred[row] != 1:
                continue
            current = int(ts[row])
            if (
                current_times
                and (hard_boundary or previous_alarm is None or current - previous_alarm > gap_ns)
            ):
                segments.append((str(name), np.asarray(current_times, dtype=np.int64)))
                current_times = []
            current_times.append(current)
            previous_alarm = current
            hard_boundary = False
        if current_times:
            segments.append((str(name), np.asarray(current_times, dtype=np.int64)))
    return segments


def one_to_one_event_prf(
    predictions: np.ndarray,
    timestamps: np.ndarray,
    turbines: np.ndarray,
    events: Any,
    *,
    labels: np.ndarray,
    events_are_episodes: bool,
    horizon: timedelta = timedelta(hours=12),
    merge_gap: timedelta = timedelta(minutes=40),
    split: str | None = None,
) -> dict[str, Any]:
    """一对一匹配报警段与事件早警窗，额外报警段计为假阳性。

    事件先按中央合同在同机组 72 小时内合并。候选边按报警段首次报警时间
    排序，采用确定性的最早报警优先贪心匹配；每个报警段和事件最多使用一次。
    """

    if events_are_episodes is not True:
        raise ValueError("one_to_one_event_prf 只接受已按 split+72h 规范化的 episode table")
    segments = _alarm_segments(predictions, timestamps, turbines, merge_gap, labels)
    # 此函数接收中央加载器已经按 72 小时规则规范化的 episode table；这里只
    # 合并重叠/相接记录，不能再次用 72 小时把两个合法 episode 合成一个。
    event_frame = _events_frame(events, split=split, merge_hours=0.0)
    horizon_ns = int(horizon.total_seconds() * 1_000_000_000)
    windows = [
        (str(row.turbine), int(row.start.value) - horizon_ns, int(row.start.value))
        for row in event_frame.itertuples(index=False)
    ]

    candidates: list[tuple[int, int, int]] = []
    for segment_index, (name, alarm_times) in enumerate(segments):
        for event_index, (event_name, window_start, window_end) in enumerate(windows):
            if name != event_name:
                continue
            inside = alarm_times[(alarm_times >= window_start) & (alarm_times < window_end)]
            if inside.size:
                candidates.append((int(inside.min()), segment_index, event_index))
    candidates.sort()
    used_segments: set[int] = set()
    used_events: set[int] = set()
    for _, segment_index, event_index in candidates:
        if segment_index in used_segments or event_index in used_events:
            continue
        used_segments.add(segment_index)
        used_events.add(event_index)

    matched = len(used_events)
    segment_count = len(segments)
    event_count = len(windows)
    precision = float(matched / segment_count) if segment_count else 0.0
    recall = float(matched / event_count) if event_count else 0.0
    f1 = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))
    return {
        "one_to_one_alarm_segments": segment_count,
        "one_to_one_event_count": event_count,
        "one_to_one_matched": matched,
        "one_to_one_event_precision": precision,
        "one_to_one_event_recall": recall,
        "one_to_one_event_f1": f1,
    }


def point_adjust_f1_turbine_macro(
    labels: np.ndarray,
    predictions: np.ndarray,
    timestamps: np.ndarray,
    turbines: np.ndarray,
) -> dict[str, Any]:
    """经典 point-adjust F1，仅作为有偏附录指标，绝不用于排名。"""

    y = np.asarray(labels, dtype=np.int8)
    pred = np.asarray(predictions, dtype=np.int8)
    ts = np.asarray(timestamps, dtype=np.int64)
    turb = np.asarray(turbines).astype(str)
    _validate_aligned(y, pred, ts, turb)
    f1_rows: list[float] = []
    skipped: list[str] = []
    for name in np.unique(turb):
        index = np.flatnonzero(turb == name)
        index = index[np.argsort(ts[index], kind="stable")]
        local_y = y[index]
        valid = local_y != -1
        target_full = (local_y == 1).astype(np.int8)
        adjusted_full = np.where(valid, pred[index], 0).astype(np.int8)
        target = target_full[valid]
        if not target.any():
            skipped.append(str(name))
            continue
        # 若某真实异常段任一点被命中，则把该真实段全部置为命中。
        # 在完整时间轴找段，ignore 行被 target_full=0 明确切断，不能先删除后拼接。
        padded = np.pad(target_full, (1, 1))
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        for start, end in zip(starts, ends):
            if adjusted_full[start:end].any():
                adjusted_full[start:end] = 1
        adjusted = adjusted_full[valid]
        tp = int(np.count_nonzero((target == 1) & (adjusted == 1)))
        fp = int(np.count_nonzero((target == 0) & (adjusted == 1)))
        fn = int(np.count_nonzero((target == 1) & (adjusted == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_rows.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return {
        "pa_f1_appendix": float(np.mean(f1_rows)) if f1_rows else None,
        "pa_status": "appendix_only_not_for_ranking" if f1_rows else "undefined_no_ground_truth_ranges",
        "pa_turbines_used": len(f1_rows),
        "pa_turbines_skipped": skipped,
    }


def evaluate_full_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    timestamps: np.ndarray,
    turbines: np.ndarray,
    events: Any,
    *,
    threshold: float,
    polarity: str,
    split: str,
    horizon: timedelta = timedelta(hours=12),
    merge_gap: timedelta = timedelta(minutes=40),
    score_semantics: str = "anomaly_score",
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    """冻结校准后调用一次的完整真实故障评测；不参与阈值搜索。"""

    pred = predictions_from_scores(scores, threshold, polarity)
    # 原始 event_table 只在这里按 split 与 72 小时规则规范化一次。下游所有
    # 事件指标共享同一 episode table，避免分母不一致。
    episode_events = _events_frame(events, split=split, merge_hours=72.0)
    result = evaluate_equal4(
        labels, scores, timestamps, turbines, episode_events,
        threshold=threshold, polarity=polarity, split=None,
        horizon=horizon, merge_gap=merge_gap, include_point_metrics=False,
    )
    oriented = np.asarray(scores, dtype=float) if polarity == "positive" else -np.asarray(scores, dtype=float)
    result.update(
        point_metrics(
            labels, oriented, pred,
            score_semantics=score_semantics, probabilities=probabilities,
        )
    )
    result.update(turbine_macro_range(labels, pred, timestamps, turbines))
    result.update(turbine_macro_affiliation(labels, pred, timestamps, turbines))
    result.update(
        one_to_one_event_prf(
            pred, timestamps, turbines, episode_events,
            labels=labels, events_are_episodes=True,
            horizon=horizon, merge_gap=merge_gap, split=None,
        )
    )
    result.update(point_adjust_f1_turbine_macro(labels, pred, timestamps, turbines))
    return result
