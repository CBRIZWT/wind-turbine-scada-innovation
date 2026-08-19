r"""
模块名称：AnomalyTransformer.py
实验链路位置：Anomaly Transformer 编码器堆叠、前馈网络和投影输出模块。
主要输入输出：保持原仓库接口不变；输入通常是 NumPy/Pandas/Torch 表示的多变量时间序列窗口，输出是模型张量、异常分数、预测值、阈值或实验指标。
核心数学思想：数学主线是 Transformer 表示学习、重构误差和关联差异联合形成无监督异常分数。
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

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding


class EncoderLayer(nn.Module):
    r"""
    科研注释：类 `EncoderLayer`
    名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`attention`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`d_model`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`d_ff`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`dropout`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`activation`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        r"""
        科研注释：函数/方法 `forward`
        名称作用：执行神经网络前向传播，把输入窗口映射为重构值、预测值或中间注意力表示。
        参数说明：`x`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`attn_mask`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回张量预测、重构结果、注意力权重或模型约定的中间输出。
        数学原理：模型通过编码、解码或预测得到 $\hat{x}$ 或 $\hat{y}$，再用误差 $e=|x-\hat{x}|$ 或 $|y-\hat{y}|$ 支撑检测/预测。
        流程说明：流程：校验输入形状 -> 线性/卷积/注意力/循环层变换 -> 组合输出 -> 保持调用方预期的张量形状。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    r"""
    科研注释：类 `Encoder`
    名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(self, attn_layers, norm_layer=None):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`attn_layers`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`norm_layer`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        r"""
        科研注释：函数/方法 `forward`
        名称作用：执行神经网络前向传播，把输入窗口映射为重构值、预测值或中间注意力表示。
        参数说明：`x`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`attn_mask`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回张量预测、重构结果、注意力权重或模型约定的中间输出。
        数学原理：模型通过编码、解码或预测得到 $\hat{x}$ 或 $\hat{y}$，再用误差 $e=|x-\hat{x}|$ 或 $|y-\hat{y}|$ 支撑检测/预测。
        流程说明：流程：校验输入形状 -> 线性/卷积/注意力/循环层变换 -> 组合输出 -> 保持调用方预期的张量形状。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    r"""
    科研注释：类 `AnomalyTransformer`
    名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=2048,
                 dropout=0.0, activation='gelu', output_attention=True):
        # #28 修复: d_ff 默认改为 4×d_model (标准 Transformer FFN 扩展比)
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`win_size`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`enc_in`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`c_out`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`d_model`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`n_heads`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`e_layers`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`d_ff`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`dropout`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`activation`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`output_attention`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

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
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]
