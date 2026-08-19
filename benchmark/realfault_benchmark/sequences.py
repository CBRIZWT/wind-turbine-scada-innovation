"""与标签隔离的 SCADA 序列窗口索引。

本模块只接受 :class:`~realfault_benchmark.contracts.ScoreView`。该类型在结构上
不包含验证或测试标签，因此窗口是否存在只能由机组、时间戳、row_id 和固定的
时间连续性合同决定。模型窗口分数最终散射回原始行；不能形成窗口的行保留 NaN。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
from typing import Literal

import numpy as np

from .contracts import ScoreView


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array).view()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SequenceSpec:
    """固定的序列时间合同；所有数值与 ``ScoreView.timestamps`` 使用相同单位。"""

    window_size: int
    cadence: int
    max_gap: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.window_size, bool) or not isinstance(self.window_size, Integral):
            raise ValueError("window_size 必须为整数")
        if int(self.window_size) < 2:
            raise ValueError("window_size 至少为 2")
        if isinstance(self.cadence, bool) or not isinstance(self.cadence, Integral):
            raise ValueError("cadence 必须为整数")
        if int(self.cadence) <= 0:
            raise ValueError("cadence 必须为正整数")

        max_gap = int(self.cadence) if self.max_gap is None else self.max_gap
        if isinstance(max_gap, bool) or not isinstance(max_gap, Integral):
            raise ValueError("max_gap 必须为整数")
        if int(max_gap) < int(self.cadence):
            raise ValueError("max_gap 不能小于 cadence")

        object.__setattr__(self, "window_size", int(self.window_size))
        object.__setattr__(self, "cadence", int(self.cadence))
        object.__setattr__(self, "max_gap", int(max_gap))


@dataclass(frozen=True)
class SequenceIndex:
    """原始行位置上的滚动窗口索引。"""

    window_positions: np.ndarray
    target_positions: np.ndarray
    target_row_ids: np.ndarray
    split: Literal["val", "test"]
    n_rows: int
    spec: SequenceSpec
    view_fingerprint: str


def _is_missing_turbine(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(np.asarray(value).ndim == 0 and np.asarray(value).dtype.kind == "f" and np.isnan(value))
    except (TypeError, ValueError):
        return False


def _update_array_hash(digest: "hashlib._Hash", name: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    digest.update(name.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    if array.dtype.kind == "O":
        for item in array.ravel().tolist():
            encoded = repr(item).encode("utf-8", errors="strict")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    else:
        contiguous = np.ascontiguousarray(array)
        digest.update(memoryview(contiguous).cast("B"))


def _view_fingerprint(view: ScoreView) -> str:
    """完整绑定特征和全部物理侧车，防止旧索引套用到另一个视图。"""

    digest = hashlib.sha256()
    digest.update(view.split.encode("ascii"))
    _update_array_hash(digest, "X", np.asarray(view.X))
    _update_array_hash(digest, "timestamps", np.asarray(view.timestamps))
    _update_array_hash(digest, "turbines", np.asarray(view.turbines))
    _update_array_hash(digest, "row_ids", np.asarray(view.row_ids))
    gap = np.zeros(len(view.X), dtype=bool) if view.gap_mask is None else np.asarray(view.gap_mask)
    _update_array_hash(digest, "gap_mask", gap)
    return digest.hexdigest()


def build_sequence_index(view: ScoreView, spec: SequenceSpec) -> SequenceIndex:
    """只依据 ``ScoreView`` 侧车构造逐机组、逐时间连续的滚动窗口。"""

    if not isinstance(view, ScoreView):
        raise TypeError("view 必须为 ScoreView，不能传入带标签的评测对象")
    if not isinstance(spec, SequenceSpec):
        raise TypeError("spec 必须为 SequenceSpec")

    X = np.asarray(view.X)
    timestamps = np.asarray(view.timestamps)
    turbines = np.asarray(view.turbines)
    row_ids = np.asarray(view.row_ids)
    gap_mask = (
        np.zeros(len(X), dtype=bool)
        if view.gap_mask is None
        else np.asarray(view.gap_mask, dtype=bool)
    )
    lengths = [len(X), len(timestamps), len(turbines), len(row_ids), len(gap_mask)]
    if len(set(lengths)) != 1:
        raise ValueError(f"ScoreView 特征/侧车长度不一致: {lengths}")
    if X.ndim != 2:
        raise ValueError(f"ScoreView.X 必须为二维，收到 {X.shape}")
    if timestamps.ndim != 1 or turbines.ndim != 1 or row_ids.ndim != 1 or gap_mask.ndim != 1:
        raise ValueError("timestamps、turbines、row_ids、gap_mask 必须是一维侧车")
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("row_id 必须在 split 内唯一")

    groups: dict[object, list[int]] = {}
    for position, turbine in enumerate(turbines.tolist()):
        if _is_missing_turbine(turbine):
            raise ValueError("turbine 侧车不能包含缺失值")
        try:
            groups.setdefault(turbine, []).append(position)
        except TypeError as exc:
            raise ValueError("turbine 标识必须可哈希") from exc

    records: list[tuple[int, list[int]]] = []
    for positions in groups.values():
        ordered = sorted(
            positions,
            key=lambda position: (
                int(timestamps[position]),
                int(row_ids[position]),
                position,
            ),
        )
        segment: list[int] = []
        for position in ordered:
            if gap_mask[position]:
                segment = []
                continue
            if segment:
                previous = int(timestamps[segment[-1]])
                current = int(timestamps[position])
                delta = current - previous
                continuous = (
                    delta > 0
                    and delta <= int(spec.max_gap)
                    and delta % spec.cadence == 0
                )
                if not continuous:
                    segment = []
            segment.append(position)
            if len(segment) >= spec.window_size:
                window = segment[-spec.window_size:]
                records.append((window[-1], window.copy()))

    # 模型批次顺序按原始目标行排序；窗口内部仍保持单机组时间顺序。
    records.sort(key=lambda item: item[0])
    if records:
        target_positions = np.asarray([item[0] for item in records], dtype=np.int64)
        window_positions = np.asarray([item[1] for item in records], dtype=np.int64)
    else:
        target_positions = np.empty(0, dtype=np.int64)
        window_positions = np.empty((0, spec.window_size), dtype=np.int64)
    target_row_ids = np.asarray(row_ids[target_positions], dtype=np.int64)

    return SequenceIndex(
        window_positions=_readonly(window_positions),
        target_positions=_readonly(target_positions),
        target_row_ids=_readonly(target_row_ids),
        split=view.split,
        n_rows=len(X),
        spec=spec,
        view_fingerprint=_view_fingerprint(view),
    )


def extract_windows(view: ScoreView, index: SequenceIndex) -> np.ndarray:
    """按索引提取 ``(window, feature)`` 张量，不改变原始视图。"""

    if not isinstance(view, ScoreView):
        raise TypeError("view 必须为 ScoreView")
    if view.split != index.split:
        raise ValueError("序列索引不能跨 split 使用")
    if len(view.X) != index.n_rows:
        raise ValueError("序列索引与 ScoreView 行数不一致")
    if _view_fingerprint(view) != index.view_fingerprint:
        raise ValueError("序列索引与 ScoreView 完整特征/侧车指纹不一致")
    if not np.array_equal(
        np.asarray(view.row_ids)[index.target_positions],
        index.target_row_ids,
    ):
        raise ValueError("序列索引与 ScoreView row_id 不一致")
    return np.asarray(view.X)[index.window_positions]


def scatter_window_scores(index: SequenceIndex, window_scores: np.ndarray) -> np.ndarray:
    """把每个窗口的目标分数散射回原始行；无窗口行使用 NaN。"""

    scores = np.asarray(window_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(index.target_positions):
        raise ValueError(
            "窗口分数必须是一维且与 target_positions 等长，"
            f"收到 {scores.shape} 与 {len(index.target_positions)}"
        )
    aligned = np.full(index.n_rows, np.nan, dtype=np.float64)
    aligned[index.target_positions] = scores
    return _readonly(aligned)


__all__ = [
    "SequenceIndex",
    "SequenceSpec",
    "build_sequence_index",
    "extract_windows",
    "scatter_window_scores",
]
