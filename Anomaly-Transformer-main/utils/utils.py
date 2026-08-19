r"""
模块名称：utils.py
实验链路位置：通用实验辅助函数模块。
主要输入输出：保持原仓库接口不变；输入通常是 NumPy/Pandas/Torch 表示的多变量时间序列窗口，输出是模型张量、异常分数、预测值、阈值或实验指标。
核心数学思想：数学主线是 Transformer 表示学习、重构误差和关联差异联合形成无监督异常分数。
科研流程：先明确数据窗口和通道含义，再执行训练或推理，最后用重构误差、预测误差、关联差异或极值阈值形成可复核指标。
风机 SCADA 适用性：迁移到风机 SCADA 时，可把温度、振动、声音或功率相关传感器作为多变量通道；异常分数不能自动等同故障真值，仍需状态/报警日志或检修记录校验。
实现边界：本文件注释只解释名称、作用、数学原理和实验流程，不改变源码逻辑、默认参数、文件路径或张量形状。
参考文献：
- Xu, J., Wu, H., Wang, J., & Long, M. (2022). Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. ICLR 2022. PDF: https://arxiv.org/pdf/2110.02642
- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017. PDF: https://arxiv.org/pdf/1706.03762
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# 已删除: to_var() 使用已弃用的 torch.autograd.Variable (PyTorch 0.4.0+), 从未被调用
import numpy as np


def mkdir(directory):
    r"""
    科研注释：函数/方法 `mkdir`
    名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
    参数说明：`directory`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
