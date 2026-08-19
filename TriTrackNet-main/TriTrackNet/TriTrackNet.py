r"""
模块名称：TriTrackNet.py
实验链路位置：TriTrackNet 主体网络、双通道交互、训练包装器和预测接口模块。
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
import random
import numpy as np

from torch import nn

from .utils.attention import scaled_dot_product_attention
from .utils.revin import RevIN


class ReverseAttention(nn.Module):
    r"""
    科研注释：类 `ReverseAttention`
    名称作用：计算注意力权重和加权表示，刻画时间步或变量通道之间的依赖关系。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：$\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}(QK^T/\sqrt{d})V$；Anomaly Attention 还比较先验关联与序列关联。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(self, seq_len, hid_dim):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`seq_len`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`hid_dim`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        super().__init__()
        self.compute_keys = nn.Linear(seq_len, hid_dim)
        self.compute_queries = nn.Linear(seq_len, hid_dim)
        self.compute_values = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        r"""
        科研注释：函数/方法 `forward`
        名称作用：执行神经网络前向传播，把输入窗口映射为重构值、预测值或中间注意力表示。
        参数说明：`x`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回张量预测、重构结果、注意力权重或模型约定的中间输出。
        数学原理：模型通过编码、解码或预测得到 $\hat{x}$ 或 $\hat{y}$，再用误差 $e=|x-\hat{x}|$ 或 $|y-\hat{y}|$ 支撑检测/预测。
        流程说明：流程：校验输入形状 -> 线性/卷积/注意力/循环层变换 -> 组合输出 -> 保持调用方预期的张量形状。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        queries = self.compute_queries(x)
        keys = self.compute_keys(x)
        values = self.compute_values(x)
        att_score = scaled_dot_product_attention(queries, keys, values)
        reversed_att_score = -att_score  # 反向处理
        return reversed_att_score


class MLP(nn.Module):
    r"""
    科研注释：类 `MLP`
    名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`input_dim`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`hidden_dim`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`output_dim`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        r"""
        科研注释：函数/方法 `forward`
        名称作用：执行神经网络前向传播，把输入窗口映射为重构值、预测值或中间注意力表示。
        参数说明：`x`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回张量预测、重构结果、注意力权重或模型约定的中间输出。
        数学原理：模型通过编码、解码或预测得到 $\hat{x}$ 或 $\hat{y}$，再用误差 $e=|x-\hat{x}|$ 或 $|y-\hat{y}|$ 支撑检测/预测。
        流程说明：流程：校验输入形状 -> 线性/卷积/注意力/循环层变换 -> 组合输出 -> 保持调用方预期的张量形状。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TriTrackNetArchitecture(nn.Module):
    r"""
    科研注释：类 `TriTrackNetArchitecture`
    名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(
        self,
        num_channels,
        seq_len,
        hid_dim=16,
        pred_horizon=96,
        use_revin=True,
        use_aux=False,          # CrossModalFusion 已移除, 保留参数仅用于向后兼容
        # A：attention vs HFF
        use_attention: bool = True,
        use_hff: bool = True,
        aux_mode: str = "keep",
        # C：gating on/off (CrossModalFusion removed, kept for backward compat)
        use_gating: bool = False,
        gating_mode: str = "adaptive",
        # cross-modal
        cross_embed_dim: int = 128
    ):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`num_channels`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`seq_len`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`hid_dim`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`pred_horizon`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`use_revin`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`use_aux`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`use_attention`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`use_hff`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`aux_mode`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`use_gating`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`gating_mode`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`cross_embed_dim`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        super().__init__()
        self.use_revin = use_revin
        self.use_aux = use_aux

        self.revin = RevIN(num_features=num_channels)

        self.dual_channel_attention = DualChannelAttention(
            seq_len=seq_len,
            hid_dim=hid_dim,
            num_channels=num_channels,
            mlp_hidden_dim=64,
            use_attention=use_attention,
            use_hff=use_hff,
            aux_mode=aux_mode
        )

        # CrossModalFusion 已移除 (2026-05-31): domain_knowledge 始终为 None, 400+参数浪费显存
        self.cross_modal_fusion = None

        self.linear_forecaster = nn.Linear(seq_len, pred_horizon)

    def forward(self, x, domain_knowledge=None, flatten_output=True):
        # x: (B, C, L)
        r"""
        科研注释：函数/方法 `forward`
        名称作用：执行神经网络前向传播，把输入窗口映射为重构值、预测值或中间注意力表示。
        参数说明：`x`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`domain_knowledge`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`flatten_output`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回张量预测、重构结果、注意力权重或模型约定的中间输出。
        数学原理：模型通过编码、解码或预测得到 $\hat{x}$ 或 $\hat{y}$，再用误差 $e=|x-\hat{x}|$ 或 $|y-\hat{y}|$ 支撑检测/预测。
        流程说明：流程：校验输入形状 -> 线性/卷积/注意力/循环层变换 -> 组合输出 -> 保持调用方预期的张量形状。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        if self.use_revin:
            x_norm = self.revin(x.transpose(1, 2), mode='norm').transpose(1, 2)
        else:
            x_norm = x

        att_score = self.dual_channel_attention(x_norm)

        if self.use_aux and (self.cross_modal_fusion is not None) and (domain_knowledge is not None):
            att_score = self.cross_modal_fusion(att_score, domain_knowledge)

        out = x_norm + att_score
        out = self.linear_forecaster(out)

        if self.use_revin:
            out = self.revin(out.transpose(1, 2), mode='denorm').transpose(1, 2)

        if flatten_output:
            return out.reshape([out.shape[0], out.shape[1] * out.shape[2]])
        else:
            return out


class DualChannelAttention(nn.Module):
    r"""
    科研注释：类 `DualChannelAttention`
    名称作用：计算注意力权重和加权表示，刻画时间步或变量通道之间的依赖关系。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：$\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}(QK^T/\sqrt{d})V$；Anomaly Attention 还比较先验关联与序列关联。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(
        self,
        seq_len,
        hid_dim,
        num_channels,
        mlp_hidden_dim=64,
        use_attention: bool = True,
        use_hff: bool = True,
        aux_mode: str = "keep"  # "keep" | "off"
    ):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`seq_len`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`hid_dim`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`num_channels`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`mlp_hidden_dim`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`use_attention`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`use_hff`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`aux_mode`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        super().__init__()
        self.num_channels = num_channels
        self.use_attention = use_attention
        self.use_hff = use_hff
        self.aux_mode = aux_mode

        # 主通道：标准注意力（时间维长依赖）
        if self.use_attention:
            self.compute_keys_1 = nn.Linear(seq_len, hid_dim)
            self.compute_queries_1 = nn.Linear(seq_len, hid_dim)
            self.compute_values_1 = nn.Linear(seq_len, seq_len)

        # 辅助通道 1：反向注意力（可视作"抑制/对比"信号）
        if self.aux_mode == "keep":
            self.reverse_attention_2 = ReverseAttention(seq_len, hid_dim)
        else:
            self.reverse_attention_2 = None

        # 辅助通道 2：HFF（异构特征融合，这里用 MLP 表征）
        if self.use_hff:
            self.mlp_3 = MLP(seq_len, mlp_hidden_dim, seq_len)
        else:
            self.mlp_3 = None

        # 当两者都关闭时，保形占位（Identity/线性）
        self.identity_head = nn.Identity()

    def forward(self, x):
        # 将通道切 3 段，仅用于与原实现对齐；也可改成全通道并行
        r"""
        科研注释：函数/方法 `forward`
        名称作用：执行神经网络前向传播，把输入窗口映射为重构值、预测值或中间注意力表示。
        参数说明：`x`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回张量预测、重构结果、注意力权重或模型约定的中间输出。
        数学原理：模型通过编码、解码或预测得到 $\hat{x}$ 或 $\hat{y}$，再用误差 $e=|x-\hat{x}|$ 或 $|y-\hat{y}|$ 支撑检测/预测。
        流程说明：流程：校验输入形状 -> 线性/卷积/注意力/循环层变换 -> 组合输出 -> 保持调用方预期的张量形状。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        c = max(1, self.num_channels // 3)
        channel_1 = x[:, :c, :]
        channel_2 = x[:, c: 2 * c, :] if (c < self.num_channels) else x[:, :c, :]
        channel_3 = x[:, 2 * c:, :]   if (2 * c < self.num_channels) else x[:, :c, :]

        outs = []

        # 主通道：标准注意力
        if self.use_attention:
            q1 = self.compute_queries_1(channel_1)
            k1 = self.compute_keys_1(channel_1)
            v1 = self.compute_values_1(channel_1)
            att_score_1 = scaled_dot_product_attention(q1, k1, v1)  # (B,c,L)
        else:
            # 不用注意力时，保形（可改为线性/恒等）
            att_score_1 = self.identity_head(channel_1)
        outs.append(att_score_1)

        # 辅助通道 1：反向注意力（是否保留由 aux_mode 决定）
        if self.reverse_attention_2 is not None:
            rev2 = self.reverse_attention_2(channel_2)
        else:
            rev2 = torch.zeros_like(channel_2)
        outs.append(rev2)

        # 辅助通道 2：HFF（MLP）
        if self.mlp_3 is not None:
            hff3 = self.mlp_3(channel_3)
        else:
            hff3 = torch.zeros_like(channel_3)
        outs.append(hff3)

        # 拼接回原通道数（如果切分造成尾部尺寸不齐，可在外层再做裁剪/填充）
        out = torch.cat(outs, dim=1)

        # 若拼接超出原通道数，裁剪至原通道；不足则 pad
        if out.size(1) > x.size(1):
            out = out[:, :x.size(1), :]
        elif out.size(1) < x.size(1):
            pad = torch.zeros(x.size(0), x.size(1) - out.size(1), x.size(2), device=x.device, dtype=x.dtype)
            out = torch.cat([out, pad], dim=1)

        return out
