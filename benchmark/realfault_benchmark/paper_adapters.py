"""论文方法到统一 real-fault 接口的外部适配器。

本文件不改论文作者的数据读取/划分逻辑，也不向模型暴露验证或测试标签。
所有序列边界仅来自冻结的 turbine/timestamp/gap_mask 侧车；高分统一表示异常。
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import importlib.util
from pathlib import Path
import sys
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import torch
from torch import nn

from .adapters import ModelAdapter
from .contracts import ScoreView, TrainView


CADENCE_NS = 600 * 1_000_000_000


def _torch_device(device: str) -> torch.device:
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但当前 chuangxin 环境不可用")
    return requested


def _seed_torch(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _gap_mask(view: ScoreView | TrainView) -> np.ndarray:
    if view.gap_mask is None:
        raise ValueError("严格论文适配器要求冻结的 gap_mask，不能静默假定全连续")
    gap = np.asarray(view.gap_mask, dtype=bool)
    if gap.ndim != 1 or len(gap) != len(view.X):
        raise ValueError("gap_mask 必须与特征逐行对齐")
    return gap


def contiguous_segments(
    view: ScoreView | TrainView,
    *,
    cadence_ns: int = CADENCE_NS,
) -> list[np.ndarray]:
    """只按冻结物理侧车构造单机组、严格 10-min 连续段。"""

    timestamps = np.asarray(view.timestamps, dtype=np.int64)
    turbines = np.asarray(view.turbines).astype(str)
    gap = _gap_mask(view)
    if not (len(view.X) == len(timestamps) == len(turbines) == len(gap)):
        raise ValueError("特征与序列侧车长度不一致")
    segments: list[np.ndarray] = []
    for turbine in np.unique(turbines):
        order = np.flatnonzero(turbines == turbine)
        order = order[np.argsort(timestamps[order], kind="stable")]
        current: list[int] = []
        previous_timestamp: int | None = None
        for row in order:
            timestamp = int(timestamps[row])
            if gap[row] or (
                previous_timestamp is not None and timestamp - previous_timestamp != cadence_ns
            ):
                if current:
                    segments.append(np.asarray(current, dtype=np.int64))
                current = []
            if not gap[row]:
                current.append(int(row))
                previous_timestamp = timestamp
            else:
                previous_timestamp = None
        if current:
            segments.append(np.asarray(current, dtype=np.int64))
    return segments


def iter_window_batches(
    view: ScoreView | TrainView,
    *,
    window_size: int,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """流式产生窗口和目标原始行号，避免为全量测试集物化巨大索引矩阵。"""

    X = np.asarray(view.X)
    for segment in contiguous_segments(view):
        if len(segment) < window_size:
            continue
        for start in range(window_size - 1, len(segment), batch_size):
            stop = min(start + batch_size, len(segment))
            offsets = np.arange(start, stop, dtype=np.int64)
            positions = np.stack([segment[offset - window_size + 1: offset + 1] for offset in offsets])
            yield np.asarray(X[positions], dtype=np.float32), segment[offsets]


@dataclass(frozen=True)
class SampledWindows:
    segments: tuple[np.ndarray, ...]
    segment_ids: np.ndarray
    offsets: np.ndarray

    def batches(
        self,
        X: np.ndarray,
        *,
        window_size: int,
        batch_size: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for start in range(0, len(self.offsets), batch_size):
            stop = min(start + batch_size, len(self.offsets))
            rows = [
                self.segments[int(sid)][int(offset) - window_size + 1: int(offset) + 1]
                for sid, offset in zip(self.segment_ids[start:stop], self.offsets[start:stop])
            ]
            positions = np.stack(rows)
            yield np.asarray(X[positions], dtype=np.float32), positions


def sample_windows(
    view: ScoreView | TrainView,
    *,
    window_size: int,
    max_windows: int,
    seed: int,
) -> SampledWindows:
    """从全部物理合法窗口均匀、确定性抽样；候选集合不读取标签。"""

    segments = tuple(segment for segment in contiguous_segments(view) if len(segment) >= window_size)
    counts = np.asarray([len(segment) - window_size + 1 for segment in segments], dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        raise ValueError(f"没有可用的 {window_size} 步连续窗口")
    rng = np.random.default_rng(int(seed))
    chosen = np.arange(total, dtype=np.int64) if total <= max_windows else np.sort(
        rng.choice(total, size=int(max_windows), replace=False)
    )
    cumulative = np.cumsum(counts)
    segment_ids = np.searchsorted(cumulative, chosen, side="right")
    previous = np.where(segment_ids == 0, 0, cumulative[segment_ids - 1])
    offsets = chosen - previous + (window_size - 1)
    return SampledWindows(segments, segment_ids.astype(np.int64), offsets.astype(np.int64))


class StatisticalRFAdapter(ModelAdapter):
    """统计学习论文的优胜 Random Forest，使用 realfault 监督真值迁移。"""

    required_train_kind = "supervised"
    model_id = "statistical_gearbox_rf"
    paper_title = "Wind Turbine Gearbox Fault Detection Based on Statistical Learning"
    publication_date = "2025-01-01"
    reproduction_kind = "method_migration"
    score_semantics = "probability"

    def __init__(
        self,
        feature_indices: np.ndarray,
        *,
        n_estimators: int = 200,
        max_train_rows: int = 200_000,
    ) -> None:
        super().__init__()
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.n_estimators = int(n_estimators)
        self.max_train_rows = int(max_train_rows)

    def _fit(self, train_view: TrainView, seed: int, device: str) -> None:
        labels = np.asarray(train_view.labels, dtype=np.int8)
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        rng = np.random.default_rng(seed)
        if len(positive) >= self.max_train_rows:
            keep = np.sort(rng.choice(positive, self.max_train_rows, replace=False))
        else:
            remaining = self.max_train_rows - len(positive)
            chosen_negative = negative if len(negative) <= remaining else np.sort(
                rng.choice(negative, remaining, replace=False)
            )
            keep = np.sort(np.concatenate([positive, chosen_negative]))
        if not len(positive):
            raise ValueError("监督 Random Forest 的训练视图没有真实正例")
        X = np.asarray(train_view.X)[keep][:, self.feature_indices].astype(np.float32)
        self._mean = np.nanmean(X, axis=0)
        X = np.where(np.isfinite(X), X, self._mean)
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=seed, n_jobs=8,
            class_weight="balanced_subsample",
        ).fit(X, labels[keep])

    def _score(self, score_view: ScoreView) -> np.ndarray:
        X = np.asarray(score_view.X)[:, self.feature_indices].astype(np.float32)
        X = np.where(np.isfinite(X), X, self._mean)
        score = self._model.predict_proba(X)[:, list(self._model.classes_).index(1)].astype(float)
        score[_gap_mask(score_view)] = np.nan
        return score


class LifeTrendAdapter(ModelAdapter):
    """长期温度水平+斜率迁移；天然不适合 12h 事件，但应如实入统一评测。"""

    required_train_kind = "normal"
    model_id = "life_extension_temperature_trend"
    paper_title = "Life extension of wind turbine drivetrains by means of SCADA data"
    publication_date = "2024-01-01"
    reproduction_kind = "method_migration"
    score_semantics = "anomaly_score"

    def __init__(
        self,
        feature_indices: np.ndarray,
        *,
        median_window: int = 30 * 144,
        slope_lag: int = 7 * 144,
    ) -> None:
        super().__init__()
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.median_window = int(median_window)
        self.slope_lag = int(slope_lag)

    def _fit(self, train_view: TrainView, seed: int, device: str) -> None:
        if not len(self.feature_indices):
            raise ValueError("寿命趋势适配器没有温度特征")

    def _score(self, score_view: ScoreView) -> np.ndarray:
        result = np.full(len(score_view.X), np.nan, dtype=float)
        X = np.asarray(score_view.X)[:, self.feature_indices]
        for segment in contiguous_segments(score_view):
            level = np.nanmean(X[segment], axis=1)
            series = pd.Series(level)
            median = series.rolling(
                self.median_window, min_periods=min(self.median_window, max(2, self.slope_lag))
            ).median()
            slope = median.diff(self.slope_lag)
            level_term = np.maximum(median.to_numpy(), 0.0)
            slope_term = np.nan_to_num(np.maximum(slope.to_numpy(), 0.0), nan=0.0)
            score = level_term + 10.0 * slope_term
            result[segment] = score
        return result


class _VAECore(nn.Module):
    def __init__(self, dimension: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, dimension))

    def forward(self, x: torch.Tensor, *, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        mu, logvar = self.mu(hidden), self.logvar(hidden).clamp(-12.0, 12.0)
        latent = mu
        if sample:
            latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(latent), mu, logvar


class VAEHealthIndexAdapter(ModelAdapter):
    """作者 VAE 健康指数思想对低频温度残差的外部方法迁移。"""

    required_train_kind = "normal"
    model_id = "vae_health_index_scada_migration"
    paper_title = "Fault detection in wind turbines using health index monitoring with variational autoencoders"
    publication_date = "2024-01-01"
    reproduction_kind = "local_adapter"
    score_semantics = "anomaly_score"

    def __init__(
        self,
        feature_indices: np.ndarray,
        *,
        max_train_rows: int = 40_000,
        epochs: int = 3,
        batch_size: int = 512,
        hidden: int = 32,
        latent: int = 8,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.max_train_rows = int(max_train_rows)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.hidden = int(hidden)
        self.latent = int(latent)
        self.learning_rate = float(learning_rate)

    def _fit(self, train_view: TrainView, seed: int, device: str) -> None:
        _seed_torch(seed)
        self._device = _torch_device(device)
        rng = np.random.default_rng(seed)
        available = np.flatnonzero(~_gap_mask(train_view))
        keep = available if len(available) <= self.max_train_rows else np.sort(
            rng.choice(available, self.max_train_rows, replace=False)
        )
        X = np.asarray(train_view.X)[keep][:, self.feature_indices].astype(np.float32)
        self._mean = np.nanmean(X, axis=0).astype(np.float32)
        self._std = np.nanstd(X, axis=0).astype(np.float32)
        self._std[self._std < 1e-6] = 1.0
        X = np.nan_to_num((X - self._mean) / self._std, nan=0.0, posinf=0.0, neginf=0.0)
        self._model = _VAECore(len(self.feature_indices), self.hidden, self.latent).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))
        generator = torch.Generator().manual_seed(seed)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator,
        )
        self._model.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                batch = batch.to(self._device)
                reconstruction, mu, logvar = self._model(batch, sample=True)
                reconstruction_loss = torch.mean((reconstruction - batch) ** 2)
                kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
                loss = reconstruction_loss + 1e-3 * kl
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

    def _raw_reconstruction_score(self, score_view: ScoreView) -> np.ndarray:
        X = np.asarray(score_view.X)[:, self.feature_indices].astype(np.float32)
        X = np.nan_to_num((X - self._mean) / self._std, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.full(len(X), np.nan, dtype=float)
        self._model.eval()
        with torch.no_grad():
            for start in range(0, len(X), 8192):
                stop = min(start + 8192, len(X))
                batch = torch.from_numpy(X[start:stop]).to(self._device)
                reconstruction, _, _ = self._model(batch, sample=False)
                result[start:stop] = torch.mean((reconstruction - batch) ** 2, dim=1).cpu().numpy()
        result[_gap_mask(score_view)] = np.nan
        return result

    def _score(self, score_view: ScoreView) -> np.ndarray:
        return self._raw_reconstruction_score(score_view)


class _PMLPCore(nn.Module):
    def __init__(self, dimension: int, hidden: tuple[int, int]) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(dimension, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden[1], dimension)
        self.sigma = nn.Sequential(nn.Linear(hidden[1], dimension), nn.Softplus())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        return self.mu(hidden), self.sigma(hidden) + 1e-3


class PMLPAdapter(ModelAdapter):
    """异方差高斯 PMLP + 逐机组因果 CUSUM。"""

    required_train_kind = "normal"
    model_id = "probabilistic_mlp_cusum"
    paper_title = "Probabilistic Multilayer Perceptrons for Wind Farm Condition Monitoring"
    publication_date = "2025-01-01"
    reproduction_kind = "paper_reimplementation"
    score_semantics = "anomaly_score"

    def __init__(
        self,
        feature_indices: np.ndarray,
        *,
        max_train_windows: int = 60_000,
        epochs: int = 3,
        batch_size: int = 1024,
        hidden: tuple[int, int] = (128, 64),
        learning_rate: float = 1e-3,
        cusum_k: float = 0.5,
    ) -> None:
        super().__init__()
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.max_train_windows = int(max_train_windows)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.hidden = hidden
        self.learning_rate = float(learning_rate)
        self.cusum_k = float(cusum_k)

    def _fit(self, train_view: TrainView, seed: int, device: str) -> None:
        _seed_torch(seed)
        self._device = _torch_device(device)
        windows = sample_windows(
            train_view, window_size=2, max_windows=self.max_train_windows, seed=seed,
        )
        self._model = _PMLPCore(len(self.feature_indices), self.hidden).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        self._model.train()
        for _ in range(self.epochs):
            for batch, _ in windows.batches(
                np.asarray(train_view.X), window_size=2, batch_size=self.batch_size,
            ):
                previous = torch.from_numpy(batch[:, 0, self.feature_indices]).to(self._device)
                target = torch.from_numpy(batch[:, 1, self.feature_indices]).to(self._device)
                mu, sigma = self._model(previous)
                loss = torch.mean(torch.log(sigma) + (target - mu).square() / (2.0 * sigma.square()))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

    def _score(self, score_view: ScoreView) -> np.ndarray:
        raw = np.full(len(score_view.X), np.nan, dtype=float)
        self._model.eval()
        with torch.no_grad():
            for batch, targets in iter_window_batches(score_view, window_size=2, batch_size=8192):
                previous = torch.from_numpy(batch[:, 0, self.feature_indices]).to(self._device)
                actual = torch.from_numpy(batch[:, 1, self.feature_indices]).to(self._device)
                mu, sigma = self._model(previous)
                z_score = (actual - mu) / sigma
                raw[targets] = torch.mean(torch.relu(z_score), dim=1).cpu().numpy()
        result = np.full(len(raw), np.nan, dtype=float)
        for segment in contiguous_segments(score_view):
            state = 0.0
            for row in segment:
                if not np.isfinite(raw[row]):
                    continue
                state = max(0.0, state + float(raw[row]) - self.cusum_k)
                result[row] = state
        return result


class _SimpleRegressor(nn.Module):
    def __init__(self, dimension: int, hidden: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(), nn.Linear(hidden, dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ConfidenceIntervalAdapter(ModelAdapter):
    """bootstrap MLP 集成置信区间越界分数（论文方法迁移）。"""

    required_train_kind = "normal"
    model_id = "confidence_interval_ensemble"
    paper_title = "On confidence interval-based anomaly detection approach for temperature predictions of wind turbine drivetrains"
    publication_date = "2025-01-01"
    reproduction_kind = "paper_reimplementation"
    score_semantics = "anomaly_score"

    def __init__(self, feature_indices: np.ndarray, *, ensemble_size: int = 5,
                 max_train_windows: int = 30_000, epochs: int = 2,
                 batch_size: int = 1024, learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.ensemble_size = int(ensemble_size)
        self.max_train_windows = int(max_train_windows)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)

    def _fit(self, train_view: TrainView, seed: int, device: str) -> None:
        self._device = _torch_device(device)
        self._models: list[_SimpleRegressor] = []
        X = np.asarray(train_view.X)
        for member in range(self.ensemble_size):
            member_seed = seed + member
            _seed_torch(member_seed)
            windows = sample_windows(
                train_view, window_size=2, max_windows=self.max_train_windows,
                seed=member_seed,
            )
            model = _SimpleRegressor(len(self.feature_indices)).to(self._device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            model.train()
            for _ in range(self.epochs):
                for batch, _ in windows.batches(X, window_size=2, batch_size=self.batch_size):
                    previous = torch.from_numpy(batch[:, 0, self.feature_indices]).to(self._device)
                    target = torch.from_numpy(batch[:, 1, self.feature_indices]).to(self._device)
                    loss = torch.mean((model(previous) - target) ** 2)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            model.eval()
            self._models.append(model)

    def _score(self, score_view: ScoreView) -> np.ndarray:
        result = np.full(len(score_view.X), np.nan, dtype=float)
        with torch.no_grad():
            for batch, targets in iter_window_batches(score_view, window_size=2, batch_size=4096):
                previous = torch.from_numpy(batch[:, 0, self.feature_indices]).to(self._device)
                actual = batch[:, 1, self.feature_indices]
                prediction = np.stack([model(previous).cpu().numpy() for model in self._models])
                mean = prediction.mean(axis=0)
                std = prediction.std(axis=0) + 1e-3
                exceedance = np.maximum((actual - mean - 1.96 * std) / std, 0.0)
                result[targets] = exceedance.mean(axis=1)
        return result


class FleetMedianAEAdapter(VAEHealthIndexAdapter):
    """浅层 AE NBM 后接 7 日舰队中位基线偏离。"""

    model_id = "fleet_median_autoencoder"
    paper_title = "Scalable SCADA-driven failure prediction using autoencoder NBM and fleet-median filtering"
    publication_date = "2025-01-01"
    reproduction_kind = "paper_reimplementation"

    def _score(self, score_view: ScoreView) -> np.ndarray:
        raw = self._raw_reconstruction_score(score_view)
        timestamps = np.asarray(score_view.timestamps, dtype=np.int64)
        frame = pd.DataFrame({"timestamp": timestamps, "score": raw})
        fleet = frame.groupby("timestamp", sort=True)["score"].median()
        rolling = fleet.rolling(7 * 144, min_periods=1).median()
        baseline = rolling.reindex(timestamps).to_numpy(dtype=float)
        result = np.abs(raw - baseline)
        result[_gap_mask(score_view)] = np.nan
        return result


class _GRURegressor(nn.Module):
    def __init__(self, dimension: int, hidden: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(dimension, hidden, batch_first=True)
        self.head = nn.Linear(hidden, dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(x)
        return self.head(hidden[:, -1])


class ConformalGRUAdapter(ModelAdapter):
    """GRU 一步温度预测 + 仅训练集尾部 split-conformal 校准。"""

    required_train_kind = "normal"
    model_id = "conformal_gru_thermal_prognostics"
    paper_title = "Prognostics of Thermal Anomalies in Wind Turbines via Deep Learning and Conformal Prediction Using SCADA Data"
    publication_date = "2026-01-01"
    reproduction_kind = "paper_reimplementation"
    score_semantics = "anomaly_score"

    def __init__(self, feature_indices: np.ndarray, *, window: int = 36,
                 max_train_windows: int = 30_000, epochs: int = 3,
                 batch_size: int = 256, hidden: int = 64, alpha: float = 0.05) -> None:
        super().__init__()
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.window = int(window)
        self.max_train_windows = int(max_train_windows)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.hidden = int(hidden)
        self.alpha = float(alpha)

    def _fit(self, train_view: TrainView, seed: int, device: str) -> None:
        _seed_torch(seed)
        self._device = _torch_device(device)
        selected = sample_windows(
            train_view, window_size=self.window + 1,
            max_windows=self.max_train_windows, seed=seed,
        )
        chunks: list[np.ndarray] = []
        positions: list[np.ndarray] = []
        for batch, index in selected.batches(
            np.asarray(train_view.X), window_size=self.window + 1,
            batch_size=self.batch_size,
        ):
            chunks.append(batch[:, :, self.feature_indices])
            positions.append(index[:, -1])
        data = np.concatenate(chunks)
        target_rows = np.concatenate(positions)
        order = np.argsort(np.asarray(train_view.timestamps)[target_rows], kind="stable")
        data = data[order]
        split = max(1, min(len(data) - 1, int(0.8 * len(data))))
        fit_data, calibration_data = data[:split], data[split:]
        self._model = _GRURegressor(len(self.feature_indices), self.hidden).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(fit_data[:, :-1]), torch.from_numpy(fit_data[:, -1]),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        self._model.train()
        for _ in range(self.epochs):
            for history, target in loader:
                history, target = history.to(self._device), target.to(self._device)
                loss = torch.mean((self._model(history) - target) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self._model.eval()
        errors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(calibration_data), self.batch_size):
                batch = calibration_data[start:start + self.batch_size]
                prediction = self._model(torch.from_numpy(batch[:, :-1]).to(self._device)).cpu().numpy()
                errors.append(np.max(np.abs(batch[:, -1] - prediction), axis=1))
        calibration_errors = np.concatenate(errors)
        level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / len(calibration_errors)))
        self.conformal_q = max(float(np.quantile(calibration_errors, level)), 1e-6)

    def _score(self, score_view: ScoreView) -> np.ndarray:
        result = np.full(len(score_view.X), np.nan, dtype=float)
        self._model.eval()
        with torch.no_grad():
            for batch, targets in iter_window_batches(
                score_view, window_size=self.window + 1, batch_size=2048,
            ):
                selected = batch[:, :, self.feature_indices]
                prediction = self._model(
                    torch.from_numpy(selected[:, :-1]).to(self._device)
                ).cpu().numpy()
                error = np.max(np.abs(selected[:, -1] - prediction), axis=1)
                result[targets] = (error - self.conformal_q) / self.conformal_q
        return result


class _SLBlock(nn.Module):
    def __init__(self, embedding: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding, 4, batch_first=True)
        self.lstm = nn.LSTM(embedding, embedding, batch_first=True)
        self.norm = nn.LayerNorm(embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x, need_weights=False)
        hidden, _ = self.lstm(self.norm(x + attended))
        return hidden


class _SLFormerCore(nn.Module):
    def __init__(self, dimension: int, window: int, patch: int, embedding: int) -> None:
        super().__init__()
        if window % patch:
            raise ValueError("SLFormer window 必须能被 patch 整除")
        self.window, self.patch, self.dimension = window, patch, dimension
        self.patch_embedding = nn.Linear(patch * dimension, embedding)
        self.blocks = nn.Sequential(_SLBlock(embedding), _SLBlock(embedding))
        self.head = nn.Linear(embedding, dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        patches = x.reshape(batch, self.window // self.patch, self.patch * self.dimension)
        return self.head(self.blocks(self.patch_embedding(patches))[:, -1])


class SLFormerAdapter(ModelAdapter):
    required_train_kind = "normal"
    model_id = "slformer_gearbox"
    paper_title = "Early anomaly detection of wind turbine gearbox based on SLFormer neural network"
    publication_date = "2024-01-01"
    reproduction_kind = "method_migration"
    score_semantics = "anomaly_score"

    def __init__(self, feature_indices: np.ndarray, *, window: int = 72, patch: int = 6,
                 embedding: int = 64, max_train_windows: int = 20_000,
                 epochs: int = 2, batch_size: int = 256) -> None:
        super().__init__()
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.window, self.patch, self.embedding = int(window), int(patch), int(embedding)
        self.max_train_windows, self.epochs, self.batch_size = int(max_train_windows), int(epochs), int(batch_size)

    def _fit(self, train_view: TrainView, seed: int, device: str) -> None:
        _seed_torch(seed)
        self._device = _torch_device(device)
        selected = sample_windows(train_view, window_size=self.window + 1,
                                  max_windows=self.max_train_windows, seed=seed)
        self._model = _SLFormerCore(len(self.feature_indices), self.window, self.patch, self.embedding).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        self._model.train()
        for _ in range(self.epochs):
            for batch, _ in selected.batches(np.asarray(train_view.X), window_size=self.window + 1,
                                             batch_size=self.batch_size):
                chosen = batch[:, :, self.feature_indices]
                history = torch.from_numpy(chosen[:, :-1]).to(self._device)
                target = torch.from_numpy(chosen[:, -1]).to(self._device)
                loss = torch.mean((self._model(history) - target) ** 2)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()

    def _score(self, score_view: ScoreView) -> np.ndarray:
        result = np.full(len(score_view.X), np.nan, dtype=float)
        self._model.eval()
        with torch.no_grad():
            for batch, targets in iter_window_batches(score_view, window_size=self.window + 1, batch_size=1024):
                chosen = batch[:, :, self.feature_indices]
                prediction = self._model(torch.from_numpy(chosen[:, :-1]).to(self._device)).cpu().numpy()
                result[targets] = np.mean(np.maximum(chosen[:, -1] - prediction, 0.0) ** 2, axis=1)
        return result


class _CNN1D(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(dimension, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class CNN1DAdapter(ModelAdapter):
    required_train_kind = "supervised"
    model_id = "cnn1d_temporal_scada"
    paper_title = "Early prediction of wind turbine anomalies using 1D-CNN and temporal feature engineering"
    publication_date = "2026-01-01"
    reproduction_kind = "paper_reimplementation"
    score_semantics = "probability"

    def __init__(self, feature_indices: np.ndarray, *, window: int = 144,
                 max_train_windows: int = 30_000, epochs: int = 3, batch_size: int = 128) -> None:
        super().__init__()
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.window, self.max_train_windows = int(window), int(max_train_windows)
        self.epochs, self.batch_size = int(epochs), int(batch_size)

    def _fit(self, train_view: TrainView, seed: int, device: str) -> None:
        _seed_torch(seed); self._device = _torch_device(device)
        selected = sample_windows(train_view, window_size=self.window,
                                  max_windows=self.max_train_windows, seed=seed)
        labels = np.asarray(train_view.labels, dtype=np.int8)
        window_labels = []
        for sid, offset in zip(selected.segment_ids, selected.offsets):
            rows = selected.segments[int(sid)][int(offset) - self.window + 1:int(offset) + 1]
            window_labels.append(float(np.mean(labels[rows] == 1) > 0.05))
        window_labels_array = np.asarray(window_labels, dtype=np.float32)
        positive = int(window_labels_array.sum())
        if positive == 0:
            raise ValueError("抽样训练窗没有真实正例；不允许伪造监督标签")
        self._model = _CNN1D(len(self.feature_indices)).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        pos_weight = torch.tensor([(len(window_labels_array) - positive) / positive], device=self._device)
        self._model.train()
        for _ in range(self.epochs):
            cursor = 0
            for batch, _ in selected.batches(np.asarray(train_view.X), window_size=self.window,
                                             batch_size=self.batch_size):
                stop = cursor + len(batch)
                target = torch.from_numpy(window_labels_array[cursor:stop]).to(self._device)
                chosen = torch.from_numpy(batch[:, :, self.feature_indices]).transpose(1, 2).to(self._device)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    self._model(chosen), target, pos_weight=pos_weight,
                )
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                cursor = stop

    def _score(self, score_view: ScoreView) -> np.ndarray:
        result = np.full(len(score_view.X), np.nan, dtype=float)
        self._model.eval()
        with torch.no_grad():
            for batch, targets in iter_window_batches(score_view, window_size=self.window, batch_size=1024):
                chosen = torch.from_numpy(batch[:, :, self.feature_indices]).transpose(1, 2).to(self._device)
                result[targets] = torch.sigmoid(self._model(chosen)).cpu().numpy()
        return result


_PAPER_ROOT = Path(r"E:\创新\论文复现")


def _load_paper_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class STAAdapter(ModelAdapter):
    required_train_kind = "normal"
    model_id = "sta_bka_temperature_residual"
    paper_title = "Temperature Prediction and Fault Warning of High-Speed Shaft of Wind Turbine Gearbox Based on Hybrid Deep Learning Model"
    publication_date = "2025-01-01"
    reproduction_kind = "method_migration"
    score_semantics = "anomaly_score"

    def __init__(self, feature_indices: np.ndarray, target_index: int, *, window: int = 36,
                 max_train_windows: int = 20_000, epochs: int = 2, batch_size: int = 128) -> None:
        super().__init__()
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.target_index = int(target_index)
        self.window, self.max_train_windows = int(window), int(max_train_windows)
        self.epochs, self.batch_size = int(epochs), int(batch_size)

    def _fit(self, train_view: TrainView, seed: int, device: str) -> None:
        _seed_torch(seed); self._device = _torch_device(device)
        module = _load_paper_module(
            _PAPER_ROOT / "Temperature Prediction and Fault Warning of High-Speed Shaft of Wind Turbine Gearbox Based on Hybrid Deep Learning Model" / "STA_BKA.py",
            "benchmark_sta_bka",
        )
        selected = sample_windows(train_view, window_size=self.window + 1,
                                  max_windows=self.max_train_windows, seed=seed)
        self._model = module.STATemperatureRegressor(n_features=len(self.feature_indices)).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3, weight_decay=1e-4)
        self._model.train()
        for _ in range(self.epochs):
            for batch, _ in selected.batches(np.asarray(train_view.X), window_size=self.window + 1,
                                             batch_size=self.batch_size):
                history = torch.from_numpy(batch[:, :-1, self.feature_indices]).to(self._device)
                target = torch.from_numpy(batch[:, -1, self.target_index]).to(self._device)
                loss = torch.mean((self._model(history) - target) ** 2)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0); optimizer.step()

    def _score(self, score_view: ScoreView) -> np.ndarray:
        result = np.full(len(score_view.X), np.nan, dtype=float)
        self._model.eval()
        with torch.no_grad():
            for batch, targets in iter_window_batches(score_view, window_size=self.window + 1, batch_size=512):
                prediction = self._model(torch.from_numpy(batch[:, :-1, self.feature_indices]).to(self._device)).cpu().numpy()
                residual = batch[:, -1, self.target_index] - prediction
                result[targets] = np.maximum(residual, 0.0) ** 2
        return result


class TransGANWTAdapter(ModelAdapter):
    required_train_kind = "normal"
    model_id = "transgan_wt_dual_reconstruction"
    paper_title = "Trans GAN-WT anomaly detection model for wind turbine time series"
    publication_date = "2026-01-01"
    reproduction_kind = "paper_reimplementation"
    score_semantics = "anomaly_score"

    def __init__(self, feature_indices: np.ndarray, *, window: int = 36,
                 max_train_windows: int = 10_000, epochs: int = 2, batch_size: int = 128,
                 d_model: int = 32) -> None:
        super().__init__()
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.window, self.max_train_windows = int(window), int(max_train_windows)
        self.epochs, self.batch_size, self.d_model = int(epochs), int(batch_size), int(d_model)

    def _fit(self, train_view: TrainView, seed: int, device: str) -> None:
        _seed_torch(seed); self._device = _torch_device(device)
        self._module = _load_paper_module(
            _PAPER_ROOT / "Trans GAN-WT anomaly detection model for wind turbine time series" / "TransGAN_WT.py",
            "benchmark_transgan_wt",
        )
        selected = sample_windows(train_view, window_size=self.window,
                                  max_windows=self.max_train_windows, seed=seed)
        sampled_rows = np.unique(np.concatenate([
            selected.segments[int(sid)][int(offset) - self.window + 1:int(offset) + 1]
            for sid, offset in zip(selected.segment_ids, selected.offsets)
        ]))
        sample = np.asarray(train_view.X)[sampled_rows][:, self.feature_indices]
        self._minimum = np.nanmin(sample, axis=0).astype(np.float32)
        maximum = np.nanmax(sample, axis=0).astype(np.float32)
        self._span = np.where(maximum - self._minimum < 1e-6, 1.0, maximum - self._minimum).astype(np.float32)
        self._model = self._module.TransGANWT(
            len(self.feature_indices), self.window, d_model=self.d_model, heads=4, layers=1,
        ).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        self._model.train()
        for _ in range(self.epochs):
            for batch, _ in selected.batches(np.asarray(train_view.X), window_size=self.window,
                                             batch_size=self.batch_size):
                scaled = np.clip((batch[:, :, self.feature_indices] - self._minimum) / self._span, 0.0, 1.0)
                target = torch.from_numpy(scaled.astype(np.float32)).to(self._device)
                first, second = self._model(target)
                loss = torch.mean((first - target) ** 2) + torch.mean((second - target) ** 2)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()

    def _score(self, score_view: ScoreView) -> np.ndarray:
        result = np.full(len(score_view.X), np.nan, dtype=float)
        self._model.eval()
        with torch.no_grad():
            for batch, targets in iter_window_batches(score_view, window_size=self.window, batch_size=512):
                scaled = np.clip((batch[:, :, self.feature_indices] - self._minimum) / self._span, 0.0, 1.0)
                target = torch.from_numpy(scaled.astype(np.float32)).to(self._device)
                first, second = self._model(target)
                result[targets] = self._module.dual_reconstruction_score(target, first, second).cpu().numpy()
        return result


class TransferAEAdapter(VAEHealthIndexAdapter):
    """全局源 AE 后按目标机组健康尾段全模型微调的迁移策略。"""

    model_id = "transfer_autoencoder_full_finetune"
    paper_title = "Transfer learning applications for anomaly detection in wind turbines"
    publication_date = "2024-01-01"
    reproduction_kind = "paper_reimplementation"

    def __init__(self, feature_indices: np.ndarray, *, finetune_rows: int = 4 * 7 * 144,
                 finetune_epochs: int = 1, **kwargs) -> None:
        super().__init__(feature_indices, **kwargs)
        self.finetune_rows = int(finetune_rows)
        self.finetune_epochs = int(finetune_epochs)

    def _fit(self, train_view: TrainView, seed: int, device: str) -> None:
        super()._fit(train_view, seed, device)
        self._turbine_models: dict[str, _VAECore] = {}
        turbines = np.asarray(train_view.turbines).astype(str)
        timestamps = np.asarray(train_view.timestamps, dtype=np.int64)
        X_all = np.asarray(train_view.X)
        for offset, turbine in enumerate(np.unique(turbines)):
            rows = np.flatnonzero(turbines == turbine)
            rows = rows[np.argsort(timestamps[rows], kind="stable")][-self.finetune_rows:]
            if len(rows) < 2:
                continue
            X = X_all[rows][:, self.feature_indices].astype(np.float32)
            X = np.nan_to_num((X - self._mean) / self._std, nan=0.0, posinf=0.0, neginf=0.0)
            model = copy.deepcopy(self._model).to(self._device)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.from_numpy(X)),
                batch_size=self.batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(seed + offset + 1000),
            )
            model.train()
            for _ in range(self.finetune_epochs):
                for (batch,) in loader:
                    batch = batch.to(self._device)
                    reconstruction, mu, logvar = model(batch, sample=True)
                    loss = torch.mean((reconstruction - batch) ** 2) + 1e-3 * (
                        -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
                    )
                    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            model.eval()
            self._turbine_models[str(turbine)] = model

    def _score(self, score_view: ScoreView) -> np.ndarray:
        X = np.asarray(score_view.X)[:, self.feature_indices].astype(np.float32)
        X = np.nan_to_num((X - self._mean) / self._std, nan=0.0, posinf=0.0, neginf=0.0)
        turbines = np.asarray(score_view.turbines).astype(str)
        result = np.full(len(X), np.nan, dtype=float)
        with torch.no_grad():
            for turbine in np.unique(turbines):
                rows = np.flatnonzero(turbines == turbine)
                model = self._turbine_models.get(str(turbine), self._model)
                for start in range(0, len(rows), 8192):
                    chosen = rows[start:start + 8192]
                    batch = torch.from_numpy(X[chosen]).to(self._device)
                    reconstruction, _, _ = model(batch, sample=False)
                    result[chosen] = torch.mean((reconstruction - batch) ** 2, dim=1).cpu().numpy()
        result[_gap_mask(score_view)] = np.nan
        return result


class _FederatedLSTMCore(nn.Module):
    def __init__(self, input_dimension: int, hidden: int = 16) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dimension, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(x)
        return self.output(hidden[:, -1]).squeeze(-1)


class FederatedLSTMAdapter(ModelAdapter):
    """作者 FedAvg LSTM-NBM 的 realfault 外部适配；客户端为同场各机组。"""

    required_train_kind = "normal"
    model_id = "federated_lstm_nbm"
    paper_title = "Wind turbine condition monitoring based on intra- and inter-farm federated learning"
    publication_date = "2025-01-01"
    reproduction_kind = "local_adapter"
    score_semantics = "anomaly_score"

    def __init__(self, feature_indices: np.ndarray, target_index: int, *, window: int = 144,
                 max_train_windows: int = 50_000, rounds: int = 5,
                 local_epochs: int = 3, batch_size: int = 128) -> None:
        super().__init__()
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.target_index = int(target_index)
        self.window, self.max_train_windows = int(window), int(max_train_windows)
        self.rounds, self.local_epochs, self.batch_size = int(rounds), int(local_epochs), int(batch_size)

    @staticmethod
    def _subview(view: TrainView, rows: np.ndarray) -> TrainView:
        return TrainView(
            np.asarray(view.X)[rows], np.asarray(view.labels)[rows],
            np.asarray(view.timestamps)[rows], np.asarray(view.turbines)[rows],
            np.asarray(view.row_ids)[rows], "normal", np.asarray(view.gap_mask)[rows],
        )

    def _fit(self, train_view: TrainView, seed: int, device: str) -> None:
        _seed_torch(seed); self._device = _torch_device(device)
        turbines = np.asarray(train_view.turbines).astype(str)
        names = np.unique(turbines)
        clients: list[tuple[TrainView, SampledWindows]] = []
        per_client = max(1, self.max_train_windows // max(1, len(names)))
        for number, name in enumerate(names):
            rows = np.flatnonzero(turbines == name)
            client = self._subview(train_view, rows)
            try:
                sampled = sample_windows(
                    client, window_size=self.window + 1,
                    max_windows=per_client, seed=seed + number,
                )
            except ValueError:
                continue
            clients.append((client, sampled))
        if not clients:
            raise ValueError("没有客户端具备完整 24h 连续训练窗")
        self._model = _FederatedLSTMCore(len(self.feature_indices)).to(self._device)
        for round_index in range(self.rounds):
            states: list[dict[str, torch.Tensor]] = []
            weights: list[int] = []
            for client_index, (client, sampled) in enumerate(clients):
                _seed_torch(seed + 10_000 * round_index + client_index)
                local = copy.deepcopy(self._model).to(self._device)
                optimizer = torch.optim.Adam(local.parameters(), lr=1e-3)
                local.train()
                for _ in range(self.local_epochs):
                    for batch, _ in sampled.batches(
                        np.asarray(client.X), window_size=self.window + 1,
                        batch_size=self.batch_size,
                    ):
                        history = torch.from_numpy(batch[:, :-1, self.feature_indices]).to(self._device)
                        target = torch.from_numpy(batch[:, -1, self.target_index]).to(self._device)
                        loss = torch.mean((local(history) - target) ** 2)
                        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                states.append({key: value.detach().cpu() for key, value in local.state_dict().items()})
                weights.append(len(sampled.offsets))
            total = float(sum(weights))
            averaged = {
                key: sum(state[key] * (weight / total) for state, weight in zip(states, weights))
                for key in states[0]
            }
            self._model.load_state_dict(averaged)
        self._model.eval()

    def _score(self, score_view: ScoreView) -> np.ndarray:
        result = np.full(len(score_view.X), np.nan, dtype=float)
        with torch.no_grad():
            for batch, targets in iter_window_batches(score_view, window_size=self.window + 1, batch_size=512):
                prediction = self._model(
                    torch.from_numpy(batch[:, :-1, self.feature_indices]).to(self._device)
                ).cpu().numpy()
                residual = batch[:, -1, self.target_index] - prediction
                result[targets] = np.maximum(residual, 0.0) ** 2
        return result


class ARIMALassoEWMAAdapter(ModelAdapter):
    """ARIMA--LASSO--EWMA 论文方法的真实故障外部重实现。

    论文以逐变量 ARIMA 残差、LASSO 二次残差和按风速/功率划分的多工况
    EWMA 控制图预警。当前预处理矩阵没有原始风速/功率，因此这里明确采用：

    * ``AR(p,0,0)`` 作为快速实验中的 ARIMA 降配；
    * 基础温度 NBM 残差的 PCA 二维投影作为工况代理；
    * 仅健康训练集拟合全部参数，统一验证集负责极性/阈值校准。

    这不是作者官方源码，也不能复现论文私有海上风场的报告数值。
    """

    required_train_kind = "normal"
    model_id = "arima_lasso_ewma"
    paper_title = "Failure warning for offshore wind turbines based on Autoregressive models"
    publication_date = "2025-01-01"
    reproduction_kind = "paper_reimplementation"
    score_semantics = "anomaly_score"

    def __init__(
        self,
        feature_indices: np.ndarray,
        state_index: int,
        *,
        ar_order: int = 3,
        max_train_windows: int = 50_000,
        operating_clusters: tuple[int, ...] = (3, 4, 5),
        ewma_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.state_index = int(state_index)
        self.ar_order = int(ar_order)
        self.max_train_windows = int(max_train_windows)
        self.operating_clusters = tuple(int(value) for value in operating_clusters)
        self.ewma_weight = float(ewma_weight)
        if len(self.feature_indices) < 3:
            raise ValueError("ARIMA-LASSO-EWMA 至少需要 3 个基础残差特征")
        if self.state_index not in set(self.feature_indices.tolist()):
            raise ValueError("state_index 必须包含在 feature_indices 中")
        if self.ar_order < 1:
            raise ValueError("ar_order 必须为正整数")
        if not 0.0 < self.ewma_weight <= 1.0:
            raise ValueError("ewma_weight 必须位于 (0, 1]")

    def _fit(self, train_view: TrainView, seed: int, device: str) -> None:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LassoCV
        from sklearn.metrics import calinski_harabasz_score

        sampled = sample_windows(
            train_view,
            window_size=self.ar_order + 1,
            max_windows=self.max_train_windows,
            seed=seed,
        )
        chunks = [
            np.nan_to_num(batch[:, :, self.feature_indices], nan=0.0, posinf=0.0, neginf=0.0)
            for batch, _ in sampled.batches(
                np.asarray(train_view.X),
                window_size=self.ar_order + 1,
                batch_size=4096,
            )
        ]
        windows = np.concatenate(chunks, axis=0).astype(np.float64, copy=False)
        history = windows[:, :-1, :]
        target = windows[:, -1, :]
        design = np.concatenate(
            [history, np.ones((len(history), 1, history.shape[2]), dtype=np.float64)],
            axis=1,
        )
        coefficients = []
        first_residual = np.empty_like(target)
        for column in range(target.shape[1]):
            coef, *_ = np.linalg.lstsq(design[:, :, column], target[:, column], rcond=None)
            coefficients.append(coef)
            first_residual[:, column] = target[:, column] - design[:, :, column] @ coef
        self._ar_coefficients = np.asarray(coefficients, dtype=np.float64)

        self._state_local = int(np.flatnonzero(self.feature_indices == self.state_index)[0])
        self._input_local = np.asarray(
            [i for i in range(len(self.feature_indices)) if i != self._state_local],
            dtype=np.int64,
        )
        self._lasso = LassoCV(
            alphas=np.asarray([1e-4, 1e-3, 1e-2]),
            cv=3,
            random_state=int(seed),
            n_jobs=1,
            max_iter=10_000,
        ).fit(first_residual[:, self._input_local], first_residual[:, self._state_local])
        self.lasso_alpha_ = float(self._lasso.alpha_)
        self.lasso_selected_features_ = int(np.count_nonzero(np.abs(self._lasso.coef_) > 1e-8))

        operating_sample = target
        self._pca = PCA(n_components=2, random_state=int(seed)).fit(operating_sample)
        projection = self._pca.transform(operating_sample)
        viable = [value for value in self.operating_clusters if 2 <= value < len(projection)]
        if not viable:
            raise ValueError("工况聚类没有可用的 K")
        best: tuple[float, int, object] | None = None
        for clusters in viable:
            candidate = KMeans(n_clusters=clusters, n_init=4, random_state=int(seed)).fit(projection)
            criterion = float(calinski_harabasz_score(projection, candidate.labels_))
            if best is None or criterion > best[0]:
                best = (criterion, clusters, candidate)
        assert best is not None
        _, self.operating_cluster_count_, self._kmeans = best

        ewma, cluster = self._raw_outputs(train_view)
        self._cluster_mean = np.zeros(self.operating_cluster_count_, dtype=np.float64)
        self._cluster_scale = np.ones(self.operating_cluster_count_, dtype=np.float64)
        finite = np.isfinite(ewma)
        for number in range(self.operating_cluster_count_):
            selected = finite & (cluster == number)
            if int(selected.sum()) >= 2:
                self._cluster_mean[number] = float(np.mean(ewma[selected]))
                self._cluster_scale[number] = max(float(np.std(ewma[selected])), 1e-6)

    def _raw_outputs(self, view: ScoreView | TrainView) -> tuple[np.ndarray, np.ndarray]:
        ewma = np.full(len(view.X), np.nan, dtype=np.float64)
        clusters = np.full(len(view.X), -1, dtype=np.int16)
        weight = self.ewma_weight
        for segment in contiguous_segments(view):
            if len(segment) <= self.ar_order:
                continue
            values = np.nan_to_num(
                np.asarray(view.X)[segment][:, self.feature_indices],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float64, copy=False)
            lags = np.stack(
                [values[offset:len(values) - self.ar_order + offset] for offset in range(self.ar_order)],
                axis=1,
            )
            prediction = np.einsum(
                "mpd,dp->md", lags, self._ar_coefficients[:, :self.ar_order]
            ) + self._ar_coefficients[:, -1]
            residual = values[self.ar_order:] - prediction
            secondary = residual[:, self._state_local] - self._lasso.predict(
                residual[:, self._input_local]
            )
            smoothed = np.empty(len(secondary), dtype=np.float64)
            previous = 0.0
            for index, value in enumerate(secondary):
                previous = (1.0 - weight) * previous + weight * float(value)
                smoothed[index] = previous
            scored_rows = segment[self.ar_order:]
            ewma[scored_rows] = smoothed
            clusters[scored_rows] = self._kmeans.predict(
                self._pca.transform(values[self.ar_order:])
            ).astype(np.int16)
        return ewma, clusters

    def _score(self, score_view: ScoreView) -> np.ndarray:
        ewma, clusters = self._raw_outputs(score_view)
        result = np.full(len(score_view.X), np.nan, dtype=np.float64)
        valid = np.isfinite(ewma) & (clusters >= 0)
        number = clusters[valid].astype(np.int64)
        denominator = self._cluster_scale[number] * np.sqrt(
            self.ewma_weight / (2.0 - self.ewma_weight)
        )
        result[valid] = np.abs(ewma[valid] - self._cluster_mean[number]) / denominator
        return result


def base_residual_indices(feature_names: tuple[str, ...]) -> np.ndarray:
    index = [i for i, name in enumerate(feature_names) if str(name).endswith("__resid")]
    if not index:
        raise ValueError("没有找到基础温度 residual 特征")
    return np.asarray(index, dtype=np.int64)


def generator_bearing_indices(feature_names: tuple[str, ...]) -> np.ndarray:
    index = [
        i for i, name in enumerate(feature_names)
        if str(name).endswith("__resid") and "generator bearing" in str(name).lower()
    ]
    return np.asarray(index or base_residual_indices(feature_names)[:1], dtype=np.int64)


__all__ = [
    "ARIMALassoEWMAAdapter", "CADENCE_NS", "CNN1DAdapter", "ConfidenceIntervalAdapter", "ConformalGRUAdapter",
    "FederatedLSTMAdapter", "FleetMedianAEAdapter", "LifeTrendAdapter", "PMLPAdapter", "STAAdapter",
    "SLFormerAdapter", "SampledWindows", "StatisticalRFAdapter",
    "TransferAEAdapter", "TransGANWTAdapter", "VAEHealthIndexAdapter",
    "base_residual_indices", "contiguous_segments", "generator_bearing_indices",
    "iter_window_batches", "sample_windows",
]
