r"""
模块名称：dataset.py
实验链路位置：TriTrackNet 训练用监督窗口数据集封装模块。
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
import torch

from torch.utils.data import Dataset


class LabeledDataset(Dataset):
    r"""
    科研注释：类 `LabeledDataset`
    名称作用：封装监督学习样本，使 DataLoader 能按索引读取窗口和标签。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(self, x, y):
        """
        Converts numpy data to a torch dataset
        Args:
            x (np.array): data matrix
            y (np.array): class labels
        """
        # 科研注释：`__init__` 已有原始 docstring；本补充说明强调其科研实验含义。名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。 数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。 SCADA 迁移：保持时间顺序、通道物理意义和故障标签来源一致。
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def transform(self, x):
        r"""
        科研注释：函数/方法 `transform`
        名称作用：封装监督学习样本，使 DataLoader 能按索引读取窗口和标签。
        参数说明：`x`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回值保持原实现约定，调用方依赖其形状和类型。
        数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        return torch.FloatTensor(x)

    def __len__(self):
        r"""
        科研注释：函数/方法 `__len__`
        名称作用：封装监督学习样本，使 DataLoader 能按索引读取窗口和标签。
        参数说明：无外部业务参数；主要使用对象内部状态或全局实验配置。
        返回值：返回样本数量。
        数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        return self.y.shape[0]

    def __getitem__(self, idx):
        r"""
        科研注释：函数/方法 `__getitem__`
        名称作用：封装监督学习样本，使 DataLoader 能按索引读取窗口和标签。
        参数说明：`idx`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回单个窗口样本及其监督目标或标签。
        数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        examples = self.x[idx]
        labels = self.y[idx]
        return examples, labels
