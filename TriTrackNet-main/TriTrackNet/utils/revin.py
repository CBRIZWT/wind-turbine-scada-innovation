r"""
模块名称：revin.py
实验链路位置：TriTrackNet RevIN 可逆实例归一化模块。
主要输入输出：保持原仓库接口不变；输入通常是 NumPy/Pandas/Torch 表示的多变量时间序列窗口，输出是模型张量、异常分数、预测值、阈值或实验指标。
核心数学思想：RevIN 对每个样本按时间维计算 $\mu,\sigma$，先标准化 $\hat{x}=(x-\mu)/(\sigma+\epsilon)$，预测后再反标准化以缓解分布漂移。
科研流程：先明确数据窗口和通道含义，再执行训练或推理，最后用重构误差、预测误差、关联差异或极值阈值形成可复核指标。
风机 SCADA 适用性：迁移到风机 SCADA 时，它更适合提前预测齿轮箱油温、轴承温度或振动趋势；若要做故障检测，需要再用预测残差、阈值和故障日志构造标签。
实现边界：本文件注释只解释名称、作用、数学原理和实验流程，不改变源码逻辑、默认参数、文件路径或张量形状。
参考文献：
- Liang, M., Jia, S., Liu, Y., Zhang, X., Wang, H., & Sun, Y. (2026). TriTrackNet: A dual-channel time series forecasting model with multi-path interaction and perturbation optimization. Neurocomputing, 669, 132519. DOI: https://doi.org/10.1016/j.neucom.2025.132519
- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017. PDF: https://arxiv.org/pdf/1706.03762
- Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022). Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. ICLR 2022. PDF: https://openreview.net/pdf?id=cGDAkQo1C0p
"""
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) https://openreview.net/pdf?id=cGDAkQo1C0p
    https://github.com/ts-kim/RevIN
    """
    # 科研注释：`RevIN` 已有原始 docstring；本补充说明强调其科研实验含义。名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。 数学原理：$\hat{x}=(x-\mu)/(\sigma+\epsilon)$，必要时再用 $x=\hat{x}(\sigma+\epsilon)+\mu$ 恢复原量纲。 SCADA 迁移：保持时间顺序、通道物理意义和故障标签来源一致。
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        # 科研注释：`__init__` 已有原始 docstring；本补充说明强调其科研实验含义。名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。 数学原理：$\hat{x}=(x-\mu)/(\sigma+\epsilon)$，必要时再用 $x=\hat{x}(\sigma+\epsilon)+\mu$ 恢复原量纲。 SCADA 迁移：保持时间顺序、通道物理意义和故障标签来源一致。
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        r"""
        科研注释：函数/方法 `forward`
        名称作用：执行神经网络前向传播，把输入窗口映射为重构值、预测值或中间注意力表示。
        参数说明：`x`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`mode`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回张量预测、重构结果、注意力权重或模型约定的中间输出。
        数学原理：$\hat{x}=(x-\mu)/(\sigma+\epsilon)$，必要时再用 $x=\hat{x}(\sigma+\epsilon)+\mu$ 恢复原量纲。
        流程说明：流程：校验输入形状 -> 线性/卷积/注意力/循环层变换 -> 组合输出 -> 保持调用方预期的张量形状。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        r"""
        科研注释：函数/方法 `_init_params`
        名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
        参数说明：无外部业务参数；主要使用对象内部状态或全局实验配置。
        返回值：返回值保持原实现约定，调用方依赖其形状和类型。
        数学原理：$\hat{x}=(x-\mu)/(\sigma+\epsilon)$，必要时再用 $x=\hat{x}(\sigma+\epsilon)+\mu$ 恢复原量纲。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        r"""
        科研注释：函数/方法 `_get_statistics`
        名称作用：执行归一化、反归一化或统计量估计，控制不同传感器量纲差异。
        参数说明：`x`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回值保持原实现约定，调用方依赖其形状和类型。
        数学原理：$\hat{x}=(x-\mu)/(\sigma+\epsilon)$，必要时再用 $x=\hat{x}(\sigma+\epsilon)+\mu$ 恢复原量纲。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        dim2reduce = tuple(range(1, x.ndim-1))
        # 科研流程注释：按样本和通道统计历史窗口均值，避免不同风机工况的绝对量纲漂移污染预测器。
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        r"""
        科研注释：函数/方法 `_normalize`
        名称作用：执行归一化、反归一化或统计量估计，控制不同传感器量纲差异。
        参数说明：`x`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回值保持原实现约定，调用方依赖其形状和类型。
        数学原理：$\hat{x}=(x-\mu)/(\sigma+\epsilon)$，必要时再用 $x=\hat{x}(\sigma+\epsilon)+\mu$ 恢复原量纲。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        # 科研流程注释：标准化阶段移除局部统计量，反标准化阶段再恢复到原物理量纲。
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        r"""
        科研注释：函数/方法 `_denormalize`
        名称作用：执行归一化、反归一化或统计量估计，控制不同传感器量纲差异。
        参数说明：`x`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回值保持原实现约定，调用方依赖其形状和类型。
        数学原理：$\hat{x}=(x-\mu)/(\sigma+\epsilon)$，必要时再用 $x=\hat{x}(\sigma+\epsilon)+\mu$ 恢复原量纲。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        if self.affine:
            x = x - self.affine_bias
            # 超参数修复 (2026-05-31): eps*eps→eps, 匹配原始 RevIN (Kim et al. ICLR 2022)
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
