r"""
模块名称：embed.py
实验链路位置：位置嵌入、卷积 token 嵌入和输入嵌入融合模块。
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
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    r"""
    科研注释：类 `PositionalEmbedding`
    名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(self, d_model, max_len=5000):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`d_model`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`max_len`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

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
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    r"""
    科研注释：类 `TokenEmbedding`
    名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(self, c_in, d_model):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`c_in`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`d_model`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        super(TokenEmbedding, self).__init__()
        # #28 修复 (2026-05-31): 原注释的 PyTorch 版本比较 bug ('1.12.0'<'1.5.0' → False)
        #   被错误改为硬编码 padding=2。Conv1d(kernel=3) 的 padding=2 会使输出长度+2,
        #   与 PositionalEmbedding 的原始长度不匹配, DataEmbedding.forward() 加法崩溃。
        #   正确值: padding=1 (kernel//2) 保持序列长度不变, 适用于 PyTorch ≥1.5.0。
        #   旧: padding = 1 if torch.__version__ >= '1.5.0' else 2
        #   错: padding = 2
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

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
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    r"""
    科研注释：类 `DataEmbedding`
    名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(self, c_in, d_model, dropout=0.0):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`c_in`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`d_model`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`dropout`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

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
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
