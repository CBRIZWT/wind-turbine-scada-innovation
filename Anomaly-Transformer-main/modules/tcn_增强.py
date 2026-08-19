# -*- coding: utf-8 -*-
"""
tcn_增强.py (Anomaly-Transformer baseline)

名称: TCN 输入级残差增强包装器 (Phase D 最优超参数版)
修改原因 (2026-05-25): Phase D 第一轮 18 run 实测 TCN 未达"模块有效"硬阈值,
    5 项根因导致需要重新设计参数:
    1) ReLU 末尾激活引入非负偏置 → 移除
    2) alpha_init=0.1 过大 (zscore 输入上 10% 注入) → 0.01 + sigmoid 门控
    3) dropout=0.1 偏低 (~1.56 参数/样本过拟合) → 0.25
    4) (13,13,13,13) 4 层等宽 → (26,26,13) 3 层升降维, RF 61→29 步 (4.83h 匹配热惯量)
    5) 缺少输出归一化 → 加 LayerNorm
作用: 对 (B, win_size, D) 输入做 TCN 残差增强后送入原 AnomalyTransformer。
数学原理:
    X' = X + α · LN(TCN(X^T))^T,  α = sigmoid(α_raw)
    其中 X ∈ R^{B×L×D}, TCN 沿时间轴做膨胀因果卷积 (3 层 d ∈ {1,2,4}, RF=29 步=4.83h);
    LN 沿通道维做 LayerNorm; α 初值 sigmoid(-4.595) ≈ 0.01,训练初期接近原 baseline。
执行流程:
    1. 创建 SCADATCNResidualAdapter 和 TCNInputWrapper;
    2. wrapper 在 forward 中 permute → TCN(GELU+无末尾ReLU) → LayerNorm → α 加权 → 残差加和;
    3. 然后调用原 model.forward()。
科研标准: 不修改 AnomalyTransformer 构造签名;输入输出形状不变;
         所有超参数与 实验配置.py TCNProtocol 同步。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TCN_ROOT = _PROJECT_ROOT / "TCN-master"
if str(_TCN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TCN_ROOT))

from TCN.scada_adapter import SCADATCNResidualAdapter, SCADATCNWaveletAdapter  # noqa: E402


# ============================================================
# 名称: TCNInputWrapper
# 作用: 包装 AnomalyTransformer,在 forward 前对输入做 TCN 增强。
# 数学: wrapper.forward(x) = model(TCN(x^T)^T)
# 科研标准: 仅修改输入,不改变 model 内部计算图。
# ============================================================
class TCNInputWrapper(nn.Module):
    """Anomaly Transformer TCN 输入包装器 (Phase D 最优配置)."""

    def __init__(self, model: nn.Module, input_channels: int = 13) -> None:
        super().__init__()
        self.model = model
        self.tcn_adapter = SCADATCNResidualAdapter.from_protocol(input_channels=input_channels)

    def forward(self, x: torch.Tensor):
        """(B, L, D) → TCN 增强 → model(x)"""
        # (B, L, D) → (B, D, L) → TCN → (B, D, L) → (B, L, D)
        x_tcn = x.permute(0, 2, 1).contiguous()
        x_enhanced = self.tcn_adapter(x_tcn).permute(0, 2, 1).contiguous()
        return self.model(x_enhanced)


# ============================================================
# 名称: TCNWaveletInputWrapper (基线借鉴 #8 — Wavelet 增强 TCN v2)
# 修改原因 (2026-05-26 借鉴自基线): 见 baseline_suite/models/wt_transformer_core.py
#     的 dwt_layer 思路; 轴承故障在频域有低频热漂移 (1-6h) + 高频振动 (20-60min) 两个特征,
#     单分支 TCN 只在时域处理不到这一点。
# 作用: 在 TCN 前用 Haar 分解为 (low, high) 双分支独立 TCN, 再 Haar 合成 + 残差加和。
# 数学:
#   low_t, high_t = HaarAnalysis(x)
#   x' = x + α · LN( HaarSynthesis(TCN_low(low), TCN_high(high)) )
# 科研标准: 仅修改输入, 不改变 AT 内部计算图; 与 TCNInputWrapper 完全可替换。
# ============================================================
class TCNWaveletInputWrapper(nn.Module):
    """Anomaly Transformer Wavelet-TCN 输入包装器 (基线借鉴 #8)."""

    def __init__(self, model: nn.Module, input_channels: int = 13) -> None:
        super().__init__()
        self.model = model
        self.tcn_adapter = SCADATCNWaveletAdapter.from_protocol(input_channels=input_channels)

    def forward(self, x: torch.Tensor):
        """(B, L, D) → Wavelet-TCN 增强 → model(x)"""
        x_tcn = x.permute(0, 2, 1).contiguous()
        x_enhanced = self.tcn_adapter(x_tcn).permute(0, 2, 1).contiguous()
        return self.model(x_enhanced)
