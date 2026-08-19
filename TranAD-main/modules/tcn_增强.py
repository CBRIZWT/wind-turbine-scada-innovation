# -*- coding: utf-8 -*-
"""
tcn_增强.py (TranAD baseline)

名称: TCN 输入级残差增强包装器 (Phase D 最优超参数版)
修改原因 (2026-05-25): 同 Anomaly-Transformer-main/modules/tcn_增强.py 同步更新,
    保证三 baseline 共享相同的 TCN 超参数,横向对比时不引入额外混淆变量。
    2026-05-30 统一: TranAD n_window 由 10→36 (=6h) ≥ TCN RF 29 步, TCN 第 3 层在三个
    baseline 上都有完整真实感受野, 原"零填充妥协"已消除; 窗口大小取自 DatasetProtocol.WIN_TRANAD。
作用: 对 TranAD 的 (L, B, D) 输入做 TCN 残差增强,保持双 decoder 输出不变。
数学原理:
    X' = X + α · LN(TCN(X^T))^T,  α = sigmoid(α_raw)
    其中 X ∈ R^{L×B×D} (L = n_window=36), TCN 沿时间轴做膨胀因果卷积;
    α 初值 sigmoid(-4.595) ≈ 0.01,训练初期接近原 baseline。
执行流程:
    1. forward 接收 (window, target) 两个参数 (原 TranAD 接口);
    2. 对 window 做 permute → TCN(GELU+LayerNorm+无末尾ReLU) → permute → 残差加和;
    3. 增强后的 window 与 target 送入原 TranAD.forward()。
科研标准: 不修改 TranAD 构造签名;保持 x1, x2 双输出不变;
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
# 名称: TranADTCNWrapper
# 作用: 包装 TranAD,在 forward 前对 window 输入做 TCN 增强。
# 数学: z1, z2 = model(TCN(window^T)^T, target)
# 科研标准: 仅修改 window 输入,不改变双 decoder 计算图。
# ============================================================
class TranADTCNWrapper(nn.Module):
    """TranAD TCN 输入包装器 (Phase D 最优配置).

    2026-05-30 统一: TranAD n_window=36 (=6h) ≥ TCN RF=29 步, TCN 第 3 层有完整真实感受野,
    原 n_window=10 的零填充妥协已消除; 窗口大小唯一真源 = DatasetProtocol.WIN_TRANAD。
    """

    def __init__(self, model: nn.Module, input_channels: int = 13) -> None:
        super().__init__()
        self.model = model
        self.tcn_adapter = SCADATCNResidualAdapter.from_protocol(input_channels=input_channels)

    def forward(self, window: torch.Tensor, target: torch.Tensor):
        """(L, B, D) window → TCN 增强 → model(window, target) → (z1, z2)"""
        # 对 window 做 (L, B, D) → (B, D, L) → TCN → (B, D, L) → (L, B, D)
        window_tcn = window.permute(1, 2, 0).contiguous()
        window_enhanced = self.tcn_adapter(window_tcn).permute(2, 0, 1).contiguous()
        return self.model(window_enhanced, target)


# ============================================================
# 名称: TranADTCNWaveletWrapper (基线借鉴 #8 — Wavelet 增强 TCN v2)
# 修改原因 (2026-05-26 借鉴自基线): 见 baseline_suite/models/wt_transformer_core.py;
#     轴承故障频域有低频热漂移 + 高频振动两特征, 单 TCN 处理不到。
# 注意: TranAD n_window=36 (2026-05-31, #35), Haar 一级分解后低/高频长度各 18,
#     TCN_low/TCN_high 在长度 5 上膨胀卷积 RF 更受限,
#     但仍能学到 5 步范围内的局部模式; 这是 TranAD 短窗口的固有约束。
# 作用: 在 TCN 前 Haar 分解, 双分支 TCN, Haar 合成回 L=10, +残差; 保持 TranAD 接口不变。
# 科研标准: 与 TranADTCNWrapper 完全可替换, 形成 3 个 module variant 中的第 3 个。
# ============================================================
class TranADTCNWaveletWrapper(nn.Module):
    """TranAD Wavelet-TCN 输入包装器 (基线借鉴 #8)."""

    def __init__(self, model: nn.Module, input_channels: int = 13) -> None:
        super().__init__()
        self.model = model
        self.tcn_adapter = SCADATCNWaveletAdapter.from_protocol(input_channels=input_channels)

    def forward(self, window: torch.Tensor, target: torch.Tensor):
        """(L, B, D) window → Wavelet-TCN 增强 → model(window, target) → (z1, z2)"""
        window_tcn = window.permute(1, 2, 0).contiguous()
        window_enhanced = self.tcn_adapter(window_tcn).permute(2, 0, 1).contiguous()
        return self.model(window_enhanced, target)
