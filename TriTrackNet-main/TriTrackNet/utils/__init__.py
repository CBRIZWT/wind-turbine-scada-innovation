r"""
模块名称：__init__.py
实验链路位置：包初始化模块，声明该目录在实验代码中的导出边界。
主要输入输出：保持原仓库接口不变；输入通常是 NumPy/Pandas/Torch 表示的多变量时间序列窗口，输出是模型张量、异常分数、预测值、阈值或实验指标。
核心数学思想：预测目标是从历史窗口 $X_{t-L+1:t}$ 估计未来窗口 $Y_{t+1:t+H}$，异常预警只能通过预测残差或外部故障标签间接构造。
科研流程：先明确数据窗口和通道含义，再执行训练或推理，最后用重构误差、预测误差、关联差异或极值阈值形成可复核指标。
风机 SCADA 适用性：迁移到风机 SCADA 时，它更适合提前预测齿轮箱油温、轴承温度或振动趋势；若要做故障检测，需要再用预测残差、阈值和故障日志构造标签。
实现边界：本文件注释只解释名称、作用、数学原理和实验流程，不改变源码逻辑、默认参数、文件路径或张量形状。
参考文献：
- Liang, M., Jia, S., Liu, Y., Zhang, X., Wang, H., & Sun, Y. (2026). TriTrackNet: A dual-channel time series forecasting model with multi-path interaction and perturbation optimization. Neurocomputing, 669, 132519. DOI: https://doi.org/10.1016/j.neucom.2025.132519
- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017. PDF: https://arxiv.org/pdf/1706.03762
- Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022). Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. ICLR 2022. PDF: https://openreview.net/pdf?id=cGDAkQo1C0p
"""
from .attention import scaled_dot_product_attention
from .dataset import LabeledDataset
from .revin import RevIN
from .perturbopt import perturbopt

__all__ = [
    "scaled_dot_product_attention",
    "LabeledDataset",
    "RevIN",
    "perturbopt",
]
