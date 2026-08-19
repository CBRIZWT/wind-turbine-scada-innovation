# -*- coding: utf-8 -*-
"""
tcn_增强.py (TriTrackNet baseline)

名称: TCN 输入级残差增强包装器 (Phase D 最优超参数版)
修改原因 (2026-05-25): Phase D 第一轮实测 TriTrackNet 上 TCN 使 MSE 从 0.187
    恶化到 2.74 (×14),根因在 alpha=0.1 过大 + 末尾 ReLU 引入非负偏置;
    新版降 alpha=0.01 + sigmoid 门控 + LayerNorm + 移除末尾 ReLU + GELU 激活,
    让 TCN 残差注入幅度可控,与 baseline_only 公平对比。
作用: 对 TriTrackNet 的 (B, C, L) 输入直接做 TCN 残差增强后送入 TriTrackNetArchitecture。
数学原理:
    X' = X + α · LN(TCN(X)),  α = sigmoid(α_raw)
    其中 X ∈ R^{B×C×L} (C=13, L=seq_len=96), TCN 沿时间轴做膨胀因果卷积;
    α 初值 sigmoid(-4.595) ≈ 0.01,训练初期接近原 baseline。
执行流程:
    1. forward 接收 (x, domain_knowledge) 两个参数 (原 TriTrackNet 接口);
    2. X' = tcn_adapter(x);
    3. 增强后的 x 与 domain_knowledge 送入原 model.forward()。
科研标准: 不修改 TriTrackNetArchitecture 构造签名;保持预测输出与 PerturbOpt 流程不变;
         所有超参数与 实验配置.py TCNProtocol 同步。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TCN_ROOT = _PROJECT_ROOT / "TCN-master"
if str(_TCN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TCN_ROOT))

from TCN.scada_adapter import SCADATCNResidualAdapter, SCADATCNWaveletAdapter  # noqa: E402


# ============================================================
# 名称: TriTrackNetTCNWrapper
# 作用: 包装 TriTrackNetArchitecture,在 forward 前对输入做 TCN 增强。
# 数学: y_hat = model(TCN(x), domain_knowledge)
# 科研标准: 仅修改输入,不改变 TriTrackNet 内部计算图和 PerturbOpt 流程。
# ============================================================
class TriTrackNetTCNWrapper(nn.Module):
    """TriTrackNet TCN 输入包装器 (Phase D 最优配置)."""

    def __init__(self, model: nn.Module, input_channels: int = 13) -> None:
        super().__init__()
        self.model = model
        self.tcn_adapter = SCADATCNResidualAdapter.from_protocol(input_channels=input_channels)

    def forward(self, x: torch.Tensor, domain_knowledge=None):
        """(B, C, L) → TCN 增强 → model(x, domain_knowledge) → (B, C*H)"""
        x_enhanced = self.tcn_adapter(x)
        return self.model(x_enhanced, domain_knowledge=domain_knowledge)


# ============================================================
# 名称: TriTrackNetTCNWaveletWrapper (基线借鉴 #8 — Wavelet 增强 TCN v2)
# 修改原因 (2026-05-26 借鉴自基线): 见 baseline_suite/models/wt_transformer_core.py
#     的 dwt_layer 思路; TriTrackNet seq_len=96 → Haar 一级分解后 48, 适合双分支 TCN。
# 作用: 在 TCN 前 Haar 分解 → 低频/高频独立 TCN → Haar 合成回 L → +残差, 保持 (B,C,L) shape。
# 科研标准: 与 TriTrackNetTCNWrapper 完全可替换; PerturbOpt 双阶段不变。
# ============================================================
class TriTrackNetTCNWaveletWrapper(nn.Module):
    """TriTrackNet Wavelet-TCN 输入包装器 (基线借鉴 #8)."""

    def __init__(self, model: nn.Module, input_channels: int = 13) -> None:
        super().__init__()
        self.model = model
        self.tcn_adapter = SCADATCNWaveletAdapter.from_protocol(input_channels=input_channels)

    def forward(self, x: torch.Tensor, domain_knowledge=None):
        """(B, C, L) → Wavelet-TCN 增强 → model(x, domain_knowledge) → (B, C*H)"""
        x_enhanced = self.tcn_adapter(x)
        return self.model(x_enhanced, domain_knowledge=domain_knowledge)
