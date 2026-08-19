from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_deep_thaw(item) for item in value]
    return value


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    """返回只读 view；避免模型适配器原地污染中央数据。"""
    array = np.asarray(value, dtype=dtype).view()
    array.setflags(write=False)
    return array


def _check_aligned(name: str, X: np.ndarray, *sidecars: np.ndarray) -> None:
    if any(np.asarray(value).ndim != 1 for value in sidecars):
        raise ValueError(f"{name} 的 labels/侧车必须是一维")
    lengths = [len(X), *(len(x) for x in sidecars)]
    if len(set(lengths)) != 1:
        raise ValueError(f"{name} 特征/侧车长度不一致: {lengths}")
    if X.ndim != 2:
        raise ValueError(f"{name} X 必须为二维，收到 {X.shape}")


def _strict_labels(name: str, value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} 标签必须为一维")
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} 标签必须使用整数 dtype，拒绝转换后再验证")
    if not set(np.unique(raw).tolist()).issubset({-1, 0, 1}):
        raise ValueError(f"{name} 标签只能精确取 -1/0/1")
    return _readonly(raw, dtype=np.int8)


def _strict_gap_mask(name: str, value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind != "b":
        raise ValueError(f"{name} gap_mask 必须是一维 bool 物理侧车")
    return _readonly(raw, dtype=bool)


@dataclass(frozen=True)
class TrainView:
    """fit 可见的最小训练视图。"""

    X: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    turbines: np.ndarray
    row_ids: np.ndarray
    kind: Literal["normal", "supervised"]
    gap_mask: np.ndarray | None = None


@dataclass(frozen=True)
class ScoreView:
    """score 可见视图；结构上不包含任何验证/测试标签。"""

    X: np.ndarray
    timestamps: np.ndarray
    turbines: np.ndarray
    row_ids: np.ndarray
    split: Literal["val", "test"]
    gap_mask: np.ndarray | None = None


@dataclass(frozen=True)
class DatasetBundle:
    dataset: str
    farm: str
    variant: str
    feature_names: tuple[str, ...]
    train_normal: TrainView
    train_supervised: TrainView
    val_score: ScoreView
    test_score: ScoreView
    _val_labels: np.ndarray = field(repr=False)
    _test_labels: np.ndarray = field(repr=False)
    event_table: Any = field(repr=False)
    split_hash: str
    feature_hash: str
    file_hash: str

    @classmethod
    def from_arrays(cls, **kwargs: Any) -> "DatasetBundle":
        required = {
            "dataset", "farm", "variant", "feature_names", "train_normal_X",
            "train_normal_labels", "train_normal_timestamps", "train_normal_turbines",
            "train_normal_row_ids", "train_supervised_X", "train_supervised_labels",
            "train_supervised_timestamps", "train_supervised_turbines",
            "train_supervised_row_ids", "val_X", "val_labels", "val_timestamps",
            "val_turbines", "val_row_ids", "test_X", "test_labels", "test_timestamps",
            "test_turbines", "test_row_ids", "event_table", "split_hash", "feature_hash",
            "file_hash", "train_normal_gap_mask", "train_supervised_gap_mask",
            "val_gap_mask", "test_gap_mask",
        }
        missing = required - set(kwargs)
        if missing:
            raise TypeError(f"DatasetBundle 缺少字段: {sorted(missing)}")

        normal_X = _readonly(kwargs["train_normal_X"], dtype=np.float32)
        normal_y = _strict_labels("train_normal", kwargs["train_normal_labels"])
        normal_ts = _readonly(kwargs["train_normal_timestamps"], dtype=np.int64)
        normal_turb = _readonly(kwargs["train_normal_turbines"])
        normal_ids = _readonly(kwargs["train_normal_row_ids"], dtype=np.int64)
        normal_gap = _strict_gap_mask("train_normal", kwargs["train_normal_gap_mask"])
        _check_aligned("train_normal", normal_X, normal_y, normal_ts, normal_turb, normal_ids, normal_gap)
        if np.any(normal_y != 0):
            raise ValueError("train_normal 只能包含标签 0，不能含正例或 ignore")
        if np.any(normal_gap):
            raise ValueError("train_normal 不能包含物理 gap/ignore 行")

        sup_X_all = np.asarray(kwargs["train_supervised_X"], dtype=np.float32)
        sup_y_all = _strict_labels("train_supervised", kwargs["train_supervised_labels"])
        sup_ts_all = np.asarray(kwargs["train_supervised_timestamps"], dtype=np.int64)
        sup_turb_all = np.asarray(kwargs["train_supervised_turbines"])
        sup_ids_all = np.asarray(kwargs["train_supervised_row_ids"], dtype=np.int64)
        sup_gap_all = _strict_gap_mask(
            "train_supervised", kwargs["train_supervised_gap_mask"],
        )
        _check_aligned(
            "train_supervised_raw", sup_X_all, sup_y_all, sup_ts_all,
            sup_turb_all, sup_ids_all, sup_gap_all,
        )
        if np.any(sup_gap_all & (sup_y_all != -1)):
            raise ValueError("train_supervised gap_mask 只能标记 y=-1 的不可用行")
        keep = sup_y_all != -1
        sup_X = _readonly(sup_X_all[keep], dtype=np.float32)
        sup_y = _readonly(sup_y_all[keep], dtype=np.int8)
        sup_ts = _readonly(sup_ts_all[keep], dtype=np.int64)
        sup_turb = _readonly(sup_turb_all[keep])
        sup_ids = _readonly(sup_ids_all[keep], dtype=np.int64)
        sup_gap = _readonly(sup_gap_all[keep], dtype=bool)

        val_X = _readonly(kwargs["val_X"], dtype=np.float32)
        val_y = _strict_labels("val", kwargs["val_labels"])
        val_ts = _readonly(kwargs["val_timestamps"], dtype=np.int64)
        val_turb = _readonly(kwargs["val_turbines"])
        val_ids = _readonly(kwargs["val_row_ids"], dtype=np.int64)
        val_gap = _strict_gap_mask("val", kwargs["val_gap_mask"])
        test_X = _readonly(kwargs["test_X"], dtype=np.float32)
        test_y = _strict_labels("test", kwargs["test_labels"])
        test_ts = _readonly(kwargs["test_timestamps"], dtype=np.int64)
        test_turb = _readonly(kwargs["test_turbines"])
        test_ids = _readonly(kwargs["test_row_ids"], dtype=np.int64)
        test_gap = _strict_gap_mask("test", kwargs["test_gap_mask"])
        _check_aligned("val", val_X, val_y, val_ts, val_turb, val_ids, val_gap)
        _check_aligned("test", test_X, test_y, test_ts, test_turb, test_ids, test_gap)
        for name, labels, gap in (("val", val_y, val_gap), ("test", test_y, test_gap)):
            if np.any(gap & (labels != -1)):
                raise ValueError(f"{name} gap_mask 只能标记 y=-1 的不可用行")

        n_features = normal_X.shape[1]
        feature_names = tuple(str(x) for x in kwargs["feature_names"])
        if len(feature_names) != n_features:
            raise ValueError("feature_names 数量与特征维数不一致")
        if any(x.shape[1] != n_features for x in (sup_X, val_X, test_X)):
            raise ValueError("各视图特征维数不一致")

        return cls(
            dataset=str(kwargs["dataset"]), farm=str(kwargs["farm"]),
            variant=str(kwargs["variant"]), feature_names=feature_names,
            train_normal=TrainView(
                normal_X, normal_y, normal_ts, normal_turb, normal_ids, "normal", normal_gap,
            ),
            train_supervised=TrainView(
                sup_X, sup_y, sup_ts, sup_turb, sup_ids, "supervised", sup_gap,
            ),
            val_score=ScoreView(val_X, val_ts, val_turb, val_ids, "val", val_gap),
            test_score=ScoreView(test_X, test_ts, test_turb, test_ids, "test", test_gap),
            _val_labels=val_y, _test_labels=test_y, event_table=kwargs["event_table"],
            split_hash=str(kwargs["split_hash"]), feature_hash=str(kwargs["feature_hash"]),
            file_hash=str(kwargs["file_hash"]),
        )

    def train_view(self, kind: Literal["normal", "supervised"]) -> TrainView:
        if kind == "normal":
            return self.train_normal
        if kind == "supervised":
            return self.train_supervised
        raise ValueError(f"未知训练视图: {kind}")

    def score_view(self, split: Literal["val", "test"]) -> ScoreView:
        if split == "val":
            return self.val_score
        if split == "test":
            return self.test_score
        raise ValueError(f"score_view 只接受 val/test，收到 {split}")

    def evaluation_labels(self, split: Literal["val", "test"]) -> np.ndarray:
        """仅中央评测器调用；不得把返回值传入 ModelAdapter。"""
        if split == "val":
            return self._val_labels
        if split == "test":
            return self._test_labels
        raise ValueError(f"evaluation_labels 只接受 val/test，收到 {split}")

    def to_array_kwargs(self) -> dict[str, Any]:
        """测试/重建辅助；返回构造器使用的显式字段。"""
        return {
            "dataset": self.dataset, "farm": self.farm, "variant": self.variant,
            "feature_names": self.feature_names,
            "train_normal_X": self.train_normal.X,
            "train_normal_labels": self.train_normal.labels,
            "train_normal_timestamps": self.train_normal.timestamps,
            "train_normal_turbines": self.train_normal.turbines,
            "train_normal_row_ids": self.train_normal.row_ids,
            "train_normal_gap_mask": self.train_normal.gap_mask,
            "train_supervised_X": self.train_supervised.X,
            "train_supervised_labels": self.train_supervised.labels,
            "train_supervised_timestamps": self.train_supervised.timestamps,
            "train_supervised_turbines": self.train_supervised.turbines,
            "train_supervised_row_ids": self.train_supervised.row_ids,
            "train_supervised_gap_mask": self.train_supervised.gap_mask,
            "val_X": self.val_score.X, "val_labels": self._val_labels,
            "val_timestamps": self.val_score.timestamps, "val_turbines": self.val_score.turbines,
            "val_row_ids": self.val_score.row_ids,
            "val_gap_mask": self.val_score.gap_mask,
            "test_X": self.test_score.X, "test_labels": self._test_labels,
            "test_timestamps": self.test_score.timestamps, "test_turbines": self.test_score.turbines,
            "test_row_ids": self.test_score.row_ids, "test_gap_mask": self.test_score.gap_mask,
            "event_table": self.event_table,
            "split_hash": self.split_hash, "feature_hash": self.feature_hash,
            "file_hash": self.file_hash,
        }


@dataclass(frozen=True)
class CalibrationArtifact:
    model_id: str
    dataset_id: str
    validation_hash: str
    score_hash: str
    polarity: Literal["positive", "negative"]
    threshold: float
    threshold_source: str
    candidate_count: int
    validation_metrics: Mapping[str, Any]
    artifact_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "validation_metrics", _deep_freeze(self.validation_metrics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id, "dataset_id": self.dataset_id,
            "validation_hash": self.validation_hash, "score_hash": self.score_hash,
            "polarity": self.polarity, "threshold": self.threshold,
            "threshold_source": self.threshold_source, "candidate_count": self.candidate_count,
            "validation_metrics": _deep_thaw(self.validation_metrics),
            "artifact_hash": self.artifact_hash,
        }


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    model_id: str
    farm: str
    seed: int
    status: str
    paper_hash: str | None = None
    code_hash: str | None = None
    environment_hash: str | None = None
    data_hash: str | None = None
    calibration_hash: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    elapsed_seconds: float | None = None
    peak_ram_mb: float | None = None
    peak_vram_mb: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    exploratory: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _deep_freeze(self.metrics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "model_id": self.model_id, "farm": self.farm,
            "seed": self.seed, "status": self.status, "paper_hash": self.paper_hash,
            "code_hash": self.code_hash, "environment_hash": self.environment_hash,
            "data_hash": self.data_hash, "calibration_hash": self.calibration_hash,
            "metrics": _deep_thaw(self.metrics), "elapsed_seconds": self.elapsed_seconds,
            "peak_ram_mb": self.peak_ram_mb, "peak_vram_mb": self.peak_vram_mb,
            "error_type": self.error_type, "error_message": self.error_message,
            "exploratory": self.exploratory,
        }
