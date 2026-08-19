r"""
模块名称：attn.py
实验链路位置：Anomaly Attention 与因果 mask 的核心注意力模块。
主要输入输出：保持原仓库接口不变；输入通常是 NumPy/Pandas/Torch 表示的多变量时间序列窗口，输出是模型张量、异常分数、预测值、阈值或实验指标。
核心数学思想：核心公式包括缩放点积注意力 $A=\operatorname{Softmax}(QK^T/\sqrt{d})$，以及由高斯先验关联 $P$ 与序列关联 $S$ 构造的 Association Discrepancy。
科研流程：先明确数据窗口和通道含义，再执行训练或推理，最后用重构误差、预测误差、关联差异或极值阈值形成可复核指标。
风机 SCADA 适用性：迁移到风机 SCADA 时，可把温度、振动、声音或功率相关传感器作为多变量通道；异常分数不能自动等同故障真值，仍需状态/报警日志或检修记录校验。
实现边界：本文件注释只解释名称、作用、数学原理和实验流程，不改变源码逻辑、默认参数、文件路径或张量形状。
参考文献：
- Xu, J., Wu, H., Wang, J., & Long, M. (2022). Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. ICLR 2022. PDF: https://arxiv.org/pdf/2110.02642
- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017. PDF: https://arxiv.org/pdf/1706.03762
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask():
    r"""
    科研注释：类 `TriangularCausalMask`
    名称作用：计算注意力权重和加权表示，刻画时间步或变量通道之间的依赖关系。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：$\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}(QK^T/\sqrt{d})V$；Anomaly Attention 还比较先验关联与序列关联。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(self, B, L, device="cpu"):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`B`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`L`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`device`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：$\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}(QK^T/\sqrt{d})V$；Anomaly Attention 还比较先验关联与序列关联。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        r"""
        科研注释：函数/方法 `mask`
        名称作用：计算注意力权重和加权表示，刻画时间步或变量通道之间的依赖关系。
        参数说明：无外部业务参数；主要使用对象内部状态或全局实验配置。
        返回值：返回值保持原实现约定，调用方依赖其形状和类型。
        数学原理：$\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}(QK^T/\sqrt{d})V$；Anomaly Attention 还比较先验关联与序列关联。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        return self._mask


class AnomalyAttention(nn.Module):
    r"""
    科研注释：类 `AnomalyAttention`
    名称作用：计算注意力权重和加权表示，刻画时间步或变量通道之间的依赖关系。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：$\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}(QK^T/\sqrt{d})V$；Anomaly Attention 还比较先验关联与序列关联。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`win_size`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`mask_flag`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`scale`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`attention_dropout`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`output_attention`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：$\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}(QK^T/\sqrt{d})V$；Anomaly Attention 还比较先验关联与序列关联。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        # #10 (2026-05-31) 修复: 硬编码 .cuda() → 用 register_buffer 自动跟随模型设备。
        #   旧 self.distances.cuda() 在 CPU 模式下直接崩溃, DDP 下设备可能不对。
        self.register_buffer("distances", torch.zeros((window_size, window_size)))
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        r"""
        科研注释：函数/方法 `forward`
        名称作用：执行神经网络前向传播，把输入窗口映射为重构值、预测值或中间注意力表示。
        参数说明：`queries`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`keys`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`values`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`sigma`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`attn_mask`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回张量预测、重构结果、注意力权重或模型约定的中间输出。
        数学原理：$\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}(QK^T/\sqrt{d})V$；Anomaly Attention 还比较先验关联与序列关联。
        流程说明：流程：校验输入形状 -> 线性/卷积/注意力/循环层变换 -> 组合输出 -> 保持调用方预期的张量形状。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # 科研流程注释：这里计算 $QK^T$ 注意力分数，维度从时间窗口和多头表示映射到时间-时间关联矩阵。
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        # #10 (2026-05-31) 修复: 配合 distances → register_buffer, 不需要再 .cuda()
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1)
        # 科研流程注释：prior 用可学习尺度 sigma 构造邻域高斯关联，表达异常点更倾向只关联局部邻居的先验假设。
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        # 科研流程注释：series 是数据驱动注意力关联，后续与 prior 的 KL 差异构成 Association Discrepancy。
        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    r"""
    科研注释：类 `AttentionLayer`
    名称作用：计算注意力权重和加权表示，刻画时间步或变量通道之间的依赖关系。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：$\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}(QK^T/\sqrt{d})V$；Anomaly Attention 还比较先验关联与序列关联。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`attention`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`d_model`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`n_heads`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`d_keys`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`d_values`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：$\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}(QK^T/\sqrt{d})V$；Anomaly Attention 还比较先验关联与序列关联。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        r"""
        科研注释：函数/方法 `forward`
        名称作用：执行神经网络前向传播，把输入窗口映射为重构值、预测值或中间注意力表示。
        参数说明：`queries`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`keys`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`values`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`attn_mask`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回张量预测、重构结果、注意力权重或模型约定的中间输出。
        数学原理：$\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}(QK^T/\sqrt{d})V$；Anomaly Attention 还比较先验关联与序列关联。
        流程说明：流程：校验输入形状 -> 线性/卷积/注意力/循环层变换 -> 组合输出 -> 保持调用方预期的张量形状。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma
