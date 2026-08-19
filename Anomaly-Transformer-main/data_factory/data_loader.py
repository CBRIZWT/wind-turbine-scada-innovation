r"""
模块名称：data_loader.py
实验链路位置：公开异常检测数据集的窗口化读取、标准化和标签对齐模块。
主要输入输出：保持原仓库接口不变；输入通常是 NumPy/Pandas/Torch 表示的多变量时间序列窗口，输出是模型张量、异常分数、预测值、阈值或实验指标。
核心数学思想：窗口化把原始多变量时序 $X\in\mathbb{R}^{T\times C}$ 切成 $B\times L\times C$，标签按窗口末端或窗口范围对齐。
科研流程：先明确数据窗口和通道含义，再执行训练或推理，最后用重构误差、预测误差、关联差异或极值阈值形成可复核指标。
风机 SCADA 适用性：迁移到风机 SCADA 时，可把温度、振动、声音或功率相关传感器作为多变量通道；异常分数不能自动等同故障真值，仍需状态/报警日志或检修记录校验。
实现边界：本文件注释只解释名称、作用、数学原理和实验流程，不改变源码逻辑、默认参数、文件路径或张量形状。
参考文献：
- Xu, J., Wu, H., Wang, J., & Long, M. (2022). Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. ICLR 2022. PDF: https://arxiv.org/pdf/2110.02642
- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017. PDF: https://arxiv.org/pdf/1706.03762
"""
import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def at_num_workers(env: dict | None = None) -> int:
    """AT DataLoader worker count; default 0 avoids Windows spawn copies on large SCADA."""
    source = os.environ if env is None else env
    try:
        value = int(source.get("SCADA_AT_NUM_WORKERS", "0"))
    except (TypeError, ValueError):
        return 0
    return max(0, value)
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class PSMSegLoader(object):
    r"""
    科研注释：类 `PSMSegLoader`
    名称作用：读取数据、构造窗口、恢复模型或加载实验对象。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回读取后的数组、张量、标签、DataLoader 或模型状态。
    数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
    流程说明：流程：读取原始表 -> 标准化或选择列 -> 按时间顺序切窗口 -> 返回模型可直接消费的数组。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(self, data_path, win_size, step, mode="train"):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`data_path`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`win_size`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`step`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`mode`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
        流程说明：流程：读取原始表 -> 标准化或选择列 -> 按时间顺序切窗口 -> 返回模型可直接消费的数组。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        # #39 (2026-05-31) 修复: 旧代码 self.val = self.test 将测试集用作验证集,
        #   泄漏边界在以下 4 个 loader 中均成立 (SCADASegLoader 不受影响, 使用独立 val)。
        #   改为明确标注 '_val_is_test_clone' 属性并保留原行为以保持 benchmark 复现兼容;
        #   新 loader (SCADASegLoader) 构造函数后覆盖此属性。
        self.val = self.test
        self._val_is_test_clone = True

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        # 科研注释：`__len__` 已有原始 docstring；本补充说明强调其科研实验含义。名称作用：读取数据、构造窗口、恢复模型或加载实验对象。 数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。 SCADA 迁移：保持时间顺序、通道物理意义和故障标签来源一致。
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        r"""
        科研注释：函数/方法 `__getitem__`
        名称作用：读取数据、构造窗口、恢复模型或加载实验对象。
        参数说明：`index`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回单个窗口样本及其监督目标或标签。
        数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
        流程说明：流程：读取原始表 -> 标准化或选择列 -> 按时间顺序切窗口 -> 返回模型可直接消费的数组。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        index = index * self.step
        if self.mode == "train":
            # #39 修复 (2026-05-31): 原代码返回 test_labels[0:win_size] 对每个训练窗口
            # 都返回测试标签的第一个切片——标签完全错误。
            # 这些数据集为无监督异常检测, train 无标签可用, 改为返回零标签。
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif (self.mode == 'val'):
            # val=test (benchmark 无独立验证集), 使用正确索引的 test_labels
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    r"""
    科研注释：类 `MSLSegLoader`
    名称作用：读取数据、构造窗口、恢复模型或加载实验对象。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回读取后的数组、张量、标签、DataLoader 或模型状态。
    数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
    流程说明：流程：读取原始表 -> 标准化或选择列 -> 按时间顺序切窗口 -> 返回模型可直接消费的数组。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(self, data_path, win_size, step, mode="train"):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`data_path`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`win_size`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`step`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`mode`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
        流程说明：流程：读取原始表 -> 标准化或选择列 -> 按时间顺序切窗口 -> 返回模型可直接消费的数组。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        # #39 (2026-05-31) 修复: 旧代码 self.val = self.test 将测试集用作验证集,
        #   泄漏边界在以下 4 个 loader 中均成立 (SCADASegLoader 不受影响, 使用独立 val)。
        #   改为明确标注 '_val_is_test_clone' 属性并保留原行为以保持 benchmark 复现兼容;
        #   新 loader (SCADASegLoader) 构造函数后覆盖此属性。
        self.val = self.test
        self._val_is_test_clone = True
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        r"""
        科研注释：函数/方法 `__len__`
        名称作用：读取数据、构造窗口、恢复模型或加载实验对象。
        参数说明：无外部业务参数；主要使用对象内部状态或全局实验配置。
        返回值：返回样本数量。
        数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
        流程说明：流程：读取原始表 -> 标准化或选择列 -> 按时间顺序切窗口 -> 返回模型可直接消费的数组。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        r"""
        科研注释：函数/方法 `__getitem__`
        名称作用：读取数据、构造窗口、恢复模型或加载实验对象。
        参数说明：`index`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回单个窗口样本及其监督目标或标签。
        数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
        流程说明：流程：读取原始表 -> 标准化或选择列 -> 按时间顺序切窗口 -> 返回模型可直接消费的数组。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        index = index * self.step
        if self.mode == "train":
            # #39 修复 (2026-05-31): 原代码返回 test_labels[0:win_size] 对每个训练窗口
            # 都返回测试标签的第一个切片——标签完全错误。
            # 这些数据集为无监督异常检测, train 无标签可用, 改为返回零标签。
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif (self.mode == 'val'):
            # val=test (benchmark 无独立验证集), 使用正确索引的 test_labels
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    r"""
    科研注释：类 `SMAPSegLoader`
    名称作用：读取数据、构造窗口、恢复模型或加载实验对象。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回读取后的数组、张量、标签、DataLoader 或模型状态。
    数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
    流程说明：流程：读取原始表 -> 标准化或选择列 -> 按时间顺序切窗口 -> 返回模型可直接消费的数组。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(self, data_path, win_size, step, mode="train"):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`data_path`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`win_size`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`step`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`mode`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
        流程说明：流程：读取原始表 -> 标准化或选择列 -> 按时间顺序切窗口 -> 返回模型可直接消费的数组。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        # #39 (2026-05-31) 修复: 旧代码 self.val = self.test 将测试集用作验证集,
        #   泄漏边界在以下 4 个 loader 中均成立 (SCADASegLoader 不受影响, 使用独立 val)。
        #   改为明确标注 '_val_is_test_clone' 属性并保留原行为以保持 benchmark 复现兼容;
        #   新 loader (SCADASegLoader) 构造函数后覆盖此属性。
        self.val = self.test
        self._val_is_test_clone = True
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        r"""
        科研注释：函数/方法 `__len__`
        名称作用：读取数据、构造窗口、恢复模型或加载实验对象。
        参数说明：无外部业务参数；主要使用对象内部状态或全局实验配置。
        返回值：返回样本数量。
        数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
        流程说明：流程：读取原始表 -> 标准化或选择列 -> 按时间顺序切窗口 -> 返回模型可直接消费的数组。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        r"""
        科研注释：函数/方法 `__getitem__`
        名称作用：读取数据、构造窗口、恢复模型或加载实验对象。
        参数说明：`index`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回单个窗口样本及其监督目标或标签。
        数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
        流程说明：流程：读取原始表 -> 标准化或选择列 -> 按时间顺序切窗口 -> 返回模型可直接消费的数组。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        index = index * self.step
        if self.mode == "train":
            # #39 修复 (2026-05-31): 原代码返回 test_labels[0:win_size] 对每个训练窗口
            # 都返回测试标签的第一个切片——标签完全错误。
            # 这些数据集为无监督异常检测, train 无标签可用, 改为返回零标签。
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif (self.mode == 'val'):
            # val=test (benchmark 无独立验证集), 使用正确索引的 test_labels
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    r"""
    科研注释：类 `SMDSegLoader`
    名称作用：读取数据、构造窗口、恢复模型或加载实验对象。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回读取后的数组、张量、标签、DataLoader 或模型状态。
    数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
    流程说明：流程：读取原始表 -> 标准化或选择列 -> 按时间顺序切窗口 -> 返回模型可直接消费的数组。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    def __init__(self, data_path, win_size, step, mode="train"):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`data_path`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`win_size`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`step`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`mode`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
        流程说明：流程：读取原始表 -> 标准化或选择列 -> 按时间顺序切窗口 -> 返回模型可直接消费的数组。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        r"""
        科研注释：函数/方法 `__len__`
        名称作用：读取数据、构造窗口、恢复模型或加载实验对象。
        参数说明：无外部业务参数；主要使用对象内部状态或全局实验配置。
        返回值：返回样本数量。
        数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
        流程说明：流程：读取原始表 -> 标准化或选择列 -> 按时间顺序切窗口 -> 返回模型可直接消费的数组。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        r"""
        科研注释：函数/方法 `__getitem__`
        名称作用：读取数据、构造窗口、恢复模型或加载实验对象。
        参数说明：`index`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回单个窗口样本及其监督目标或标签。
        数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
        流程说明：流程：读取原始表 -> 标准化或选择列 -> 按时间顺序切窗口 -> 返回模型可直接消费的数组。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        index = index * self.step
        if self.mode == "train":
            # #39 修复 (2026-05-31): 原代码返回 test_labels[0:win_size] 对每个训练窗口
            # 都返回测试标签的第一个切片——标签完全错误。
            # 这些数据集为无监督异常检测, train 无标签可用, 改为返回零标签。
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif (self.mode == 'val'):
            # val=test (benchmark 无独立验证集), 使用正确索引的 test_labels
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# ============================================================
# 名称: SCADASegLoader
# 修改原因: 原 Anomaly Transformer 只支持 SMD/MSL/SMAP/PSM,不能直接读取本项目统一预处理后的 SCADA 数据。
# 作用: 读取 dataset/SCADA_{farm}/ 下统一数据契约的 train/val/test.npy 和 {split}_labels.npy,
#       输出模型需要的滑动窗口样本。
# 数据契约 (2026-06-01 对齐预处理 S9 落盘): 数据文件名固定为 train.npy/val.npy/test.npy
#       (float32, [T,C], C 由 shape[1] 动态读), 标签文件名固定为
#       {split}_labels.npy (int64, 1D, 取值 ∈ {-1=ignore,0,1})。
# 数学原理:
#   对长度为 T 的多变量序列 X ∈ R^{T×D} 按窗口 L 构造样本,
#   第 i 个样本为 X[i:i+L],标签为同区间 y[i:i+L]。
# 执行流程:
#   1. 读取 train/val/test 三段 numpy 文件;
#   2. 读取 {split}_labels.npy,保留 {-1,0,1} 标签;
#   3. 根据 mode 选择对应时间段;
#   4. 按 win_size 和 step 返回窗口给 DataLoader。
# 科研标准: 不在 loader 中重新 fit scaler,不打乱 test 顺序,不把 test 标签用于训练。
# ============================================================
class SCADASegLoader(object):
    r"""
    科研注释：类 `SCADASegLoader`
    名称作用：读取本项目统一预处理后的风机 SCADA train/val/test 数组，并按 Anomaly Transformer 需要的窗口格式输出。
    参数说明：核心参数由 `__init__` 给出，包括数据目录、窗口长度、滑动步长和数据划分模式。
    返回值：类本身不直接返回值；`__getitem__` 返回形状约为 $L\times C$ 的 SCADA 窗口及对应标签窗口。
    数学原理：把连续多变量序列 $X\in\mathbb{R}^{T\times C}$ 映射成窗口集合 $\{X_{i:i+L}\}$，标签按同一时间区间同步切片。
    流程说明：读取三段 `.npy` -> 读取 raw/兼容标签 -> 根据 mode 选择长度 -> 按 step 输出窗口。
    关键参数：`win_size` 控制模型可见历史长度，`step` 控制窗口重叠率，二者会影响异常召回与样本相关性。
    SCADA 迁移：该 loader 假定预处理阶段已经完成时间对齐、缺失处理和标准化；不要在此处重新拟合 scaler。
    """

    def __init__(self, data_path, win_size, step, mode="train"):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：加载 SCADA 三段数据和对应标签，并保存窗口化所需的模式、步长和窗口长度。
        参数说明：`data_path` 为统一数据契约的 `.npy` 所在目录；`win_size` 为窗口长度；`step` 为滑动步长；`mode` 指定 train/val/test/thre。
        返回值：无显式返回值；副作用是把数据数组和标签数组挂到对象属性。
        数学原理：初始化只准备 $T\times C$ 序列和长度为 $T$ 的 1D 标签，不改变原始时间顺序。
        流程说明：读取 train、val、test 数组 -> 调用 `_load_labels` 读取 {split}_labels.npy -> 打印形状用于实验核对。
        关键参数：`mode` 决定 `__len__` 与 `__getitem__` 使用哪一段数据。
        SCADA 迁移：标签来源必须对应报警/停机/检修规则，不能把模型预测结果再反向当作真实标签。
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        # 数据契约 (预处理 S9): 数据文件名固定 train/val/test.npy, 通道数 C 由 shape[1] 动态读。
        self.train = np.load(os.path.join(data_path, "train.npy"))
        self.val = np.load(os.path.join(data_path, "val.npy"))
        self.test = np.load(os.path.join(data_path, "test.npy"))
        self.train_labels = self._load_labels(data_path, "train", len(self.train))
        self.val_labels = self._load_labels(data_path, "val", len(self.val))
        self.test_labels = self._load_labels(data_path, "test", len(self.test))
        print("SCADA train:", self.train.shape)
        print("SCADA val:", self.val.shape)
        print("SCADA test:", self.test.shape)

    @staticmethod
    def _load_labels(data_path, split, length):
        r"""
        科研注释：函数/方法 `_load_labels`
        名称作用：按数据划分读取统一数据契约的 SCADA 标签文件 {split}_labels.npy。
        参数说明：`data_path` 为数据目录；`split` 为 train/val/test；`length` 为对应序列长度 (保留兼容)。
        返回值：返回 1D int64 标签数组 (取值 ∈ {-1=ignore,0,1})。
        数学原理：标签读取不做平滑或重采样，只保持时间点级监督信号与 SCADA 序列一一对应。
        流程说明：读取 `{split}_labels.npy` -> 校验文件存在 -> 返回原始 1D 标签 (含 -1, 由评测工具统一过滤)。
        关键参数：标签保留 {-1,0,1} 语义，-1 在 实验工具.filter_valid_labels 中被排除, 不在 loader 聚合。
        SCADA 迁移：标签缺失视为数据契约破坏, 直接报错而非伪造全 0 (避免无意义指标掩盖配置错误)。
        """
        # 数据契约 (预处理 S9): 标签文件名固定 {split}_labels.npy, int64, 1D, {-1,0,1}。
        label_path = os.path.join(data_path, f"{split}_labels.npy")
        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f"标签文件不存在: {label_path}; 请先运行 SCADA数据集/数据预处理.py 生成统一数据契约。"
            )
        labels = np.load(label_path)
        return labels

    def __len__(self):
        r"""
        科研注释：函数/方法 `__len__`
        名称作用：计算当前 mode 下可产生的滑动窗口数量。
        参数说明：无外部业务参数；使用对象内的 `mode`、`win_size`、`step` 和各数据段长度。
        返回值：返回整数窗口数，供 PyTorch `DataLoader` 调度批次。
        数学原理：窗口数按 $\lfloor(T-L)/s\rfloor+1$ 计算，其中 $T$ 为序列长度、$L$ 为窗口长度、$s$ 为步长。
        流程说明：根据 train/val/test/thre 分支选择对应序列长度并计算窗口数。
        关键参数：`step` 越小，窗口重叠越多；测试阈值模式使用非重叠窗口以贴合原仓库流程。
        SCADA 迁移：时间序列不能随机打乱后再计算窗口数，否则会破坏故障演化顺序。
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        r"""
        科研注释：函数/方法 `__getitem__`
        名称作用：按索引返回一个 SCADA 时间窗口和同区间标签窗口。
        参数说明：`index` 为 DataLoader 给出的样本序号，内部会乘以 `step` 映射到真实时间起点。
        返回值：返回 `(window, label_window)`，二者均转成 `np.float32` 以匹配原模型训练流程。
        数学原理：第 $i$ 个样本为 $X_{is:is+L}$，标签为 $y_{is:is+L}$；阈值模式使用非重叠起点。
        流程说明：换算起点 -> 根据 mode 选择数组 -> 切出长度为 `win_size` 的数据和标签。
        关键参数：`win_size` 必须不超过当前 split 长度，否则窗口数和切片会失效。
        SCADA 迁移：返回标签是评估辅助信号，训练无监督异常检测时不应把测试标签泄漏进模型优化。
        """
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(
                self.train_labels[index:index + self.win_size])
        elif self.mode == "val":
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.val_labels[index:index + self.win_size])
        elif self.mode == "test":
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            start = index // self.step * self.win_size
            end = start + self.win_size
            return np.float32(self.test[start:end]), np.float32(self.test_labels[start:end])


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    r"""
    科研注释：函数/方法 `get_loader_segment`
    名称作用：读取数据、构造窗口、恢复模型或加载实验对象。
    参数说明：`data_path`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`batch_size`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`win_size`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`step`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`mode`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`dataset`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
    返回值：返回读取后的数组、张量、标签、DataLoader 或模型状态。
    数学原理：窗口化把连续序列切成 $X\in\mathbb{R}^{B\times L\times C}$，预测任务还生成 $Y\in\mathbb{R}^{B\times H\times C}$。
    流程说明：流程：读取原始表 -> 标准化或选择列 -> 按时间顺序切窗口 -> 返回模型可直接消费的数组。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    # SCADA实验修改: SCADA 是本项目统一预处理后的数据集名称,需接入原 get_loader_segment 分发流程。
    elif (dataset == 'SCADA'):
        dataset = SCADASegLoader(data_path, win_size, step, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    # ============================================================
    # 名称: DataLoader pin_memory + num_workers (基线借鉴 #9 + 2026-06-19 稳定性修复)
    # 修改原因 (2026-05-26 借鉴自基线): 基线 baseline_suite/trainer.py L107-116
    #         统一 pin_memory; 创新原代码 pin_memory 不设 (默认 False),
    #         CUDA 训练时数据传输阻塞 GPU。
    #   (2026-06-03 提速): GPU 仅 ~23% 利用, 曾尝试多 worker 后台预取。
    #   (2026-06-19 修复): Hill of Towie 全量 5.67M×53 在 Windows spawn 下会让 worker
    #         pickle/反序列化大 SCADA loader 并触发 MemoryError; 默认回退 0, 仅显式环境变量开启。
    # 作用: CUDA 时开 pin_memory (锁页内存 → DMA 直传 GPU);
    #       配合 solver.py 中 .to(non_blocking=True) 实现传输与计算重叠。
    # Windows 兼容: 默认 num_workers=0; 如需冒险测试并行预取, 设置 SCADA_AT_NUM_WORKERS。
    # 数学原理: 无 (仅 I/O 并行化, 不改窗口/标签/数值)。
    # 科研标准: 数据内容、窗口语义、shuffle 行为均不变, 仅加载并行度提升。
    # ============================================================
    _NUM_WORKERS = at_num_workers()
    _kw = dict(dataset=dataset,
               batch_size=batch_size,
               shuffle=shuffle,
               num_workers=_NUM_WORKERS,
               pin_memory=torch.cuda.is_available())
    if _NUM_WORKERS > 0:
        _kw["persistent_workers"] = True
    data_loader = DataLoader(**_kw)
    return data_loader
