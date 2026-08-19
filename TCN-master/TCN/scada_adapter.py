# -*- coding: utf-8 -*-
"""Shared SCADA TCN input residual adapter.

The three baseline wrappers import this class so that Phase D compares the
same TCN module across Anomaly Transformer, TranAD, and TriTrackNet.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .tcn import TemporalConvNet


class SCADATCNResidualAdapter(nn.Module):
    """Shape-preserving TCN residual adapter: (B, C, L) -> (B, C, L)."""

    def __init__(
        self,
        input_channels: int = 13,
        temporal_channels: tuple[int, ...] = (26, 26, 13),
        kernel_size: int = 3,
        dropout: float = 0.25,
        alpha_init: float = 0.01,
        alpha_gate: str = "sigmoid",
        layer_norm_after_tcn: bool = True,
        inner_activation: str = "gelu",
        remove_final_relu: bool = True,
    ) -> None:
        super().__init__()
        if not temporal_channels:
            raise ValueError("temporal_channels 不能为空")
        if temporal_channels[-1] != input_channels:
            raise ValueError(
                f"TCN 末层通道数 ({temporal_channels[-1]}) 必须等于输入通道数 "
                f"({input_channels}), 才能做残差加和"
            )

        self.input_channels = int(input_channels)
        self.alpha_gate = alpha_gate.lower()
        self._alpha_raw = nn.Parameter(self._init_alpha_raw(alpha_init, self.alpha_gate))
        self.tcn = TemporalConvNet(
            num_inputs=input_channels,
            num_channels=list(temporal_channels),
            kernel_size=kernel_size,
            dropout=dropout,
            inner_activation=inner_activation,
            remove_final_relu=remove_final_relu,
        )
        self.layer_norm_after_tcn = bool(layer_norm_after_tcn)
        self.layer_norm = nn.LayerNorm(input_channels) if self.layer_norm_after_tcn else None

    @classmethod
    def from_protocol(cls, input_channels: int) -> "SCADATCNResidualAdapter":
        """Build the adapter from the project-wide TCNProtocol constants.

        2026-05-29: 通道数随输入维度动态生成 (channels_for(D)=(2D,2D,D)),
        以支持"数据驱动可变温度指标通道", 不再硬编码末层=13。
        """
        from 实验配置 import TCNProtocol

        return cls(
            input_channels=input_channels,
            temporal_channels=TCNProtocol.channels_for(input_channels),
            kernel_size=TCNProtocol.KERNEL_SIZE,
            dropout=TCNProtocol.DROPOUT,
            alpha_init=TCNProtocol.ALPHA_INIT,
            alpha_gate=TCNProtocol.ALPHA_GATE,
            layer_norm_after_tcn=TCNProtocol.LAYER_NORM_AFTER_TCN,
            inner_activation=TCNProtocol.INNER_ACTIVATION,
            remove_final_relu=TCNProtocol.REMOVE_FINAL_RELU,
        )

    @staticmethod
    def _init_alpha_raw(alpha_init: float, gate: str) -> torch.Tensor:
        """Map the desired effective alpha to the trainable raw parameter."""
        if gate == "linear":
            return torch.tensor(float(alpha_init))
        if gate == "sigmoid":
            a = max(min(float(alpha_init), 1 - 1e-6), 1e-6)
            return torch.tensor(math.log(a / (1 - a)))
        if gate == "tanh":
            a = max(min(float(alpha_init), 1 - 1e-6), 1e-6)
            return torch.tensor(math.atanh(2 * a - 1))
        raise ValueError(f"未知 alpha_gate: {gate}")

    @property
    def alpha(self) -> torch.Tensor:
        """Effective residual gate value used in forward()."""
        if self.alpha_gate == "linear":
            return self._alpha_raw
        if self.alpha_gate == "sigmoid":
            return torch.sigmoid(self._alpha_raw)
        if self.alpha_gate == "tanh":
            return (1.0 + torch.tanh(self._alpha_raw)) * 0.5
        raise ValueError(f"未知 alpha_gate: {self.alpha_gate}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual TCN enhancement to a (B, C, L) tensor."""
        if x.ndim != 3:
            raise ValueError(f"TCN 输入必须是三维 (B,C,L), 实际={tuple(x.shape)}")
        tcn_out = self.tcn(x)
        if self.layer_norm is not None:
            # LayerNorm works on channels, so transpose to (B, L, C) temporarily.
            tcn_out = self.layer_norm(tcn_out.transpose(1, 2)).transpose(1, 2)
        return x + self.alpha * tcn_out


# ============================================================
# 名称: SCADATCNWaveletAdapter (基线借鉴 #8 — Wavelet 增强 TCN v2)
# 修改原因 (2026-05-26 借鉴自基线): 基线 baseline_suite/models/wt_transformer_core.py
#     用 wavelet 分解 (dwt_layer = nn.Linear(D, 2D)) 把信号拆为低频/高频再 transformer 编码。
#     轴承故障在频域有特征 (低频热漂移 1-6h + 高频振动 20-60min), 单 TCN 只在时域处理不到。
# 作用: 在 TCN 前用 1D Haar 卷积分解 (B, C, L) → (B, C, L/2) 低频 + (B, C, L/2) 高频,
#     低频分支走主 TCN (RF=29 步 = 4.83h 匹配热惯量),
#     高频分支走副 TCN (浅一层, RF≈7 步 ≈ 70min 匹配振动尺度),
#     fuse 上采样回 L 后 + 残差加回原 x。
# 数学原理:
#   1. Haar 一级分解 (固定 kernel, 不可学):
#        L_t = (x_{2t} + x_{2t+1}) / √2     # 低频系数
#        H_t = (x_{2t} - x_{2t+1}) / √2     # 高频系数
#      用 Conv1d kernel_size=2, stride=2 实现, weight 固定不参与梯度。
#   2. 低频/高频各自走独立 TCN (channels-preserving)。
#   3. Haar 反变换 (上采样回 L, kernel 与正变换相同的转置):
#        x'_{2t}   = (L'_t + H'_t) / √2
#        x'_{2t+1} = (L'_t - H'_t) / √2
#      用 ConvTranspose1d 实现, 权重也固定。
#   4. 残差加和: out = x + α · LN(fused) , α = sigmoid(α_raw) 初值 0.01。
# 科研标准: Haar 是正交完美重构 wavelet (PR-QMF), 但不引入信号损失的前提是正/反变换之间没有可学模块;
#         本适配器在 analysis 与 synthesis 之间插入了可学 TCN(Low) + TCN(High),
#         TCN 会修改 Haar 系数, 因此 reconstruction 不是完美的, 残差加和后整体仍可通过
#         梯度更新优化预测目标。若需严格完美重构应在 TCN 后加残差直连 analysis 输出。
# ============================================================
class _HaarAnalysis(nn.Module):
    """固定权重的 1D Haar 正变换 (B,C,L) → (B,C,L/2) 双输出 (low, high)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        # 低通 [1, 1]/√2, 高通 [1, -1]/√2, 按通道展开为 group convolution
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        low_kernel = torch.tensor([inv_sqrt2, inv_sqrt2]).view(1, 1, 2).repeat(channels, 1, 1)
        high_kernel = torch.tensor([inv_sqrt2, -inv_sqrt2]).view(1, 1, 2).repeat(channels, 1, 1)
        # 不参与梯度, 用 register_buffer
        self.register_buffer("low_kernel", low_kernel)
        self.register_buffer("high_kernel", high_kernel)

    def forward(self, x: torch.Tensor):
        """x: (B,C,L) → (low: (B,C,⌈L/2⌉), high: (B,C,⌈L/2⌉))."""
        # L 是奇数时尾部镜像填充 1 步, 保证 stride=2 后长度对齐
        if x.shape[-1] % 2 == 1:
            x = torch.cat([x, x[..., -1:]], dim=-1)
        low = torch.nn.functional.conv1d(x, self.low_kernel, stride=2, groups=self.channels)
        high = torch.nn.functional.conv1d(x, self.high_kernel, stride=2, groups=self.channels)
        return low, high


class _HaarSynthesis(nn.Module):
    """固定权重的 1D Haar 反变换 (B,C,L/2) × 2 → (B,C,L)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        low_kernel = torch.tensor([inv_sqrt2, inv_sqrt2]).view(1, 1, 2).repeat(channels, 1, 1)
        high_kernel = torch.tensor([inv_sqrt2, -inv_sqrt2]).view(1, 1, 2).repeat(channels, 1, 1)
        self.register_buffer("low_kernel", low_kernel)
        self.register_buffer("high_kernel", high_kernel)

    def forward(self, low: torch.Tensor, high: torch.Tensor, target_length: int) -> torch.Tensor:
        """low, high: (B,C,L/2) → x: (B,C,target_length)."""
        x_low = torch.nn.functional.conv_transpose1d(low, self.low_kernel, stride=2, groups=self.channels)
        x_high = torch.nn.functional.conv_transpose1d(high, self.high_kernel, stride=2, groups=self.channels)
        x = x_low + x_high
        # 裁回 target_length (奇数 L 时多 1 步)
        return x[..., :target_length]


class SCADATCNWaveletAdapter(nn.Module):
    """Wavelet 增强的 TCN 残差适配器 (基线借鉴 #8).

    (B, C, L) → Haar 分解 → [TCN_low | TCN_high] → Haar 合成 → +α·LN(.) → 残差加和。
    """

    def __init__(
        self,
        input_channels: int = 13,
        low_channels: tuple[int, ...] = (26, 26, 13),
        high_channels: tuple[int, ...] = (13, 13),
        kernel_size: int = 3,
        dropout: float = 0.25,
        alpha_init: float = 0.01,
        alpha_gate: str = "sigmoid",
        layer_norm_after_tcn: bool = True,
        inner_activation: str = "gelu",
        remove_final_relu: bool = True,
        fusion: str = "add",
    ) -> None:
        super().__init__()
        if low_channels[-1] != input_channels:
            raise ValueError(
                f"低频 TCN 末层通道 ({low_channels[-1]}) 必须等于 input_channels ({input_channels})"
            )
        if high_channels[-1] != input_channels:
            raise ValueError(
                f"高频 TCN 末层通道 ({high_channels[-1]}) 必须等于 input_channels ({input_channels})"
            )
        self.input_channels = int(input_channels)
        self.alpha_gate = alpha_gate.lower()
        self._alpha_raw = nn.Parameter(SCADATCNResidualAdapter._init_alpha_raw(alpha_init, self.alpha_gate))
        # 双分支 TCN, 复用项目原 TemporalConvNet
        self.tcn_low = TemporalConvNet(
            num_inputs=input_channels,
            num_channels=list(low_channels),
            kernel_size=kernel_size,
            dropout=dropout,
            inner_activation=inner_activation,
            remove_final_relu=remove_final_relu,
        )
        self.tcn_high = TemporalConvNet(
            num_inputs=input_channels,
            num_channels=list(high_channels),
            kernel_size=kernel_size,
            dropout=dropout,
            inner_activation=inner_activation,
            remove_final_relu=remove_final_relu,
        )
        # Haar 正反变换 (固定权重, 不参与梯度)
        self.analysis = _HaarAnalysis(input_channels)
        self.synthesis = _HaarSynthesis(input_channels)
        # 输出归一化
        self.layer_norm_after_tcn = bool(layer_norm_after_tcn)
        self.layer_norm = nn.LayerNorm(input_channels) if self.layer_norm_after_tcn else None
        # fuse 方法 (add 或 concat_linear)
        self.fusion = fusion.lower()
        if self.fusion == "concat_linear":
            self.fuse_proj = nn.Conv1d(2 * input_channels, input_channels, kernel_size=1)
        else:
            self.fuse_proj = None

    @classmethod
    def from_protocol(cls, input_channels: int) -> "SCADATCNWaveletAdapter":
        """Build the wavelet adapter from project-wide TCNProtocol constants.

        2026-05-29: 低/高频分支通道随输入维度动态生成, 支持可变温度指标通道。
        """
        from 实验配置 import TCNProtocol
        return cls(
            input_channels=input_channels,
            low_channels=TCNProtocol.wavelet_low_channels_for(input_channels),
            high_channels=TCNProtocol.wavelet_high_channels_for(input_channels),
            kernel_size=TCNProtocol.KERNEL_SIZE,
            dropout=TCNProtocol.DROPOUT,
            alpha_init=TCNProtocol.ALPHA_INIT,
            alpha_gate=TCNProtocol.ALPHA_GATE,
            layer_norm_after_tcn=TCNProtocol.LAYER_NORM_AFTER_TCN,
            inner_activation=TCNProtocol.INNER_ACTIVATION,
            remove_final_relu=TCNProtocol.REMOVE_FINAL_RELU,
            fusion=TCNProtocol.WAVELET_FUSION,
        )

    @property
    def alpha(self) -> torch.Tensor:
        if self.alpha_gate == "linear":
            return self._alpha_raw
        if self.alpha_gate == "sigmoid":
            return torch.sigmoid(self._alpha_raw)
        if self.alpha_gate == "tanh":
            return (1.0 + torch.tanh(self._alpha_raw)) * 0.5
        raise ValueError(f"未知 alpha_gate: {self.alpha_gate}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, L) Wavelet 双分支增强 + 残差."""
        if x.ndim != 3:
            raise ValueError(f"Wavelet TCN 输入必须是三维 (B,C,L), 实际={tuple(x.shape)}")
        L = x.shape[-1]
        # 1. Haar 一级分解
        low, high = self.analysis(x)            # 各 (B, C, ⌈L/2⌉)
        # 2. 双分支 TCN
        low_out = self.tcn_low(low)             # (B, C, ⌈L/2⌉)
        high_out = self.tcn_high(high)          # (B, C, ⌈L/2⌉)
        # 3. Haar 合成回原 L
        fused = self.synthesis(low_out, high_out, target_length=L)  # (B, C, L)
        if self.fuse_proj is not None:
            # concat_linear 模式: 把分解前 x 与合成后 fused 拼接再线性映射
            fused = self.fuse_proj(torch.cat([x, fused], dim=1))
        # 4. LayerNorm + 残差加和
        if self.layer_norm is not None:
            fused = self.layer_norm(fused.transpose(1, 2)).transpose(1, 2)
        return x + self.alpha * fused
