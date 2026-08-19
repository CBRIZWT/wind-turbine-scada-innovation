r"""
模块名称：attention.py
实验链路位置：TriTrackNet 兼容 PyTorch scaled dot-product attention 的注意力工具模块。
主要输入输出：保持原仓库接口不变；输入通常是 NumPy/Pandas/Torch 表示的多变量时间序列窗口，输出是模型张量、异常分数、预测值、阈值或实验指标。
核心数学思想：核心公式包括多头注意力 $\operatorname{Softmax}(QK^T/\sqrt{d})V$、时间-变量交互和双通道预测误差最小化。
科研流程：先明确数据窗口和通道含义，再执行训练或推理，最后用重构误差、预测误差、关联差异或极值阈值形成可复核指标。
风机 SCADA 适用性：迁移到风机 SCADA 时，它更适合提前预测齿轮箱油温、轴承温度或振动趋势；若要做故障检测，需要再用预测残差、阈值和故障日志构造标签。
实现边界：本文件注释只解释名称、作用、数学原理和实验流程，不改变源码逻辑、默认参数、文件路径或张量形状。
参考文献：
- Liang, M., Jia, S., Liu, Y., Zhang, X., Wang, H., & Sun, Y. (2026). TriTrackNet: A dual-channel time series forecasting model with multi-path interaction and perturbation optimization. Neurocomputing, 669, 132519. DOI: https://doi.org/10.1016/j.neucom.2025.132519
- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017. PDF: https://arxiv.org/pdf/1706.03762
- Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022). Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. ICLR 2022. PDF: https://openreview.net/pdf?id=cGDAkQo1C0p
"""
import torch

import numpy as np


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    r"""
    #22 修复 (2026-05-31): 默认 is_causal=False。
    该函数被 StandardAttention/ReverseAttention 用于跨通道注意力计算
    (Q/K/V 形状为 (B, C_split, hid_dim)), 因果掩码对跨通道交互无意义且会降级
    第一个通道无法关注其他通道的表征质量。TriTrackNet 的时序自回归由
    TransformerEncoder 层内的 causal mask 控制, 此处不需要叠加。
    原默认 is_causal=True 错误地限制了跨传感器依赖建模。
    已确认 is_causal=False 为正确默认值, 无需额外修改。
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / np.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
