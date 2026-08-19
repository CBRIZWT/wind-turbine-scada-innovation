"""真实故障 benchmark 的模型适配器合同。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral

import numpy as np

from .contracts import ScoreView, TrainView


class ModelAdapter(ABC):
    """统一 ``fit(train_view, seed, device)`` / ``score(score_view)`` 接口。

    子类只实现 ``_fit`` 与 ``_score``。公开 ``score`` 会强制检查输出与输入行一一
    对齐；序列模型可为暖启动或断点行返回 NaN，但不能返回正负无穷。
    """

    required_train_kind: str | None = None

    def __init__(self) -> None:
        self._is_fitted = False
        self._fit_seed: int | None = None
        self._fit_device: str | None = None

    def fit(
        self,
        train_view: TrainView,
        seed: int,
        device: str,
    ) -> "ModelAdapter":
        # 任何重拟合尝试都先清除旧状态；验证或训练失败后不得残留旧 provenance。
        self._is_fitted = False
        self._fit_seed = None
        self._fit_device = None
        if not isinstance(train_view, TrainView):
            raise TypeError("train_view 必须为 TrainView")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed 必须为整数且不能是 bool")
        if not isinstance(device, str) or not device.strip():
            raise TypeError("device 必须为非空字符串")

        required_kind = self.required_train_kind
        if required_kind not in {"normal", "supervised"}:
            raise TypeError(
                "ModelAdapter 子类必须声明 required_train_kind='normal' 或 'supervised'"
            )
        if train_view.kind != required_kind:
            raise ValueError(
                f"该适配器要求 {required_kind} 训练视图，收到 {train_view.kind}"
            )
        X = np.asarray(train_view.X)
        labels = np.asarray(train_view.labels)
        timestamps = np.asarray(train_view.timestamps)
        turbines = np.asarray(train_view.turbines)
        row_ids = np.asarray(train_view.row_ids)
        gap_mask = (
            np.zeros(len(X), dtype=bool)
            if train_view.gap_mask is None
            else np.asarray(train_view.gap_mask)
        )
        if X.ndim != 2:
            raise ValueError(f"TrainView.X 必须为二维，收到 {X.shape}")
        if any(array.ndim != 1 for array in (labels, timestamps, turbines, row_ids, gap_mask)):
            raise ValueError("TrainView labels/timestamps/turbines/row_ids/gap_mask 必须为一维")
        lengths = [len(X), len(labels), len(timestamps), len(turbines), len(row_ids), len(gap_mask)]
        if len(set(lengths)) != 1:
            raise ValueError(f"TrainView 特征与侧车必须等长对齐，收到 {lengths}")
        if len(np.unique(row_ids)) != len(row_ids):
            raise ValueError("TrainView row_id 必须唯一")
        if required_kind == "normal" and np.any(labels != 0):
            raise ValueError("normal 训练视图的 labels 必须全零")
        if required_kind == "supervised":
            if np.any(labels == -1):
                raise ValueError("supervised 训练视图不能包含 -1 ignore 标签")
            if not set(np.unique(labels)).issubset({0, 1}):
                raise ValueError("supervised 训练视图标签只能为 0/1")

        self._fit(train_view, int(seed), device.strip())
        self._fit_seed = int(seed)
        self._fit_device = device.strip()
        self._is_fitted = True
        return self

    def score(self, score_view: ScoreView) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("ModelAdapter 必须先成功 fit 再 score")
        if not isinstance(score_view, ScoreView):
            raise TypeError("score_view 必须为不含标签的 ScoreView")

        scores = np.asarray(self._score(score_view), dtype=np.float64)
        if scores.ndim != 1 or len(scores) != len(score_view.X):
            raise ValueError(
                "模型输出必须与 ScoreView 原始行一一行对齐，"
                f"收到 {scores.shape}，期望 ({len(score_view.X)},)"
            )
        if np.isinf(scores).any():
            raise ValueError("模型分数不能包含 infinity；允许 NaN 表示无可用窗口")
        result = scores.copy()
        result.setflags(write=False)
        return result

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def fit_seed(self) -> int | None:
        return self._fit_seed

    @property
    def fit_device(self) -> str | None:
        return self._fit_device

    @abstractmethod
    def _fit(self, train_view: TrainView, seed: int, device: str) -> None:
        """实现模型拟合；训练视图类型由论文协议决定。"""

    @abstractmethod
    def _score(self, score_view: ScoreView) -> np.ndarray:
        """返回与 ``score_view`` 原始行对齐的一维异常分数。"""


__all__ = ["ModelAdapter"]
