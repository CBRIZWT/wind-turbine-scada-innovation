r"""
模块名称：solver.py
实验链路位置：Anomaly Transformer 的优化器、早停、训练循环、阈值选择和测试评估模块。
主要输入输出：保持原仓库接口不变；输入通常是 NumPy/Pandas/Torch 表示的多变量时间序列窗口，输出是模型张量、异常分数、预测值、阈值或实验指标。
核心数学思想：训练目标由重构误差 $\|x-\hat{x}\|_2$ 与先验/序列关联的对称 KL 项组成，并通过 minimax 策略放大正常点与异常点的关联差异。
科研流程：先明确数据窗口和通道含义，再执行训练或推理，最后用重构误差、预测误差、关联差异或极值阈值形成可复核指标。
风机 SCADA 适用性：迁移到风机 SCADA 时，可把温度、振动、声音或功率相关传感器作为多变量通道；异常分数不能自动等同故障真值，仍需状态/报警日志或检修记录校验。
实现边界：本文件注释只解释名称、作用、数学原理和实验流程，不改变源码逻辑、默认参数、文件路径或张量形状。
参考文献：
- Xu, J., Wu, H., Wang, J., & Long, M. (2022). Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. ICLR 2022. PDF: https://arxiv.org/pdf/2110.02642
- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017. PDF: https://arxiv.org/pdf/1706.03762
"""
import torch
import torch.nn as nn
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment


def my_kl_loss(p, q):
    r"""
    科研注释：函数/方法 `my_kl_loss`
    名称作用：计算训练或评估损失，是异常分数和优化目标的数学核心。
    参数说明：`p`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`q`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：$D_{KL}(p\|q)=\sum_i p_i(\log p_i-\log q_i)$，代码按最后一维求和后再按 batch 求均值。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    r"""
    科研注释：函数/方法 `adjust_learning_rate`
    名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
    参数说明：`optimizer`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`epoch`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`lr_`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    # ============================================================
    # 名称: adjust_learning_rate (2026-05-30 TCN 差分 lr 修复)
    # 修改原因: 旧实现把所有 param_group 的 lr 一刀切设成同一个衰减值,
    #          覆盖了 TCN/α 组的 1e-3/1e-2,差分 lr 仅存活 1 个 epoch。
    # 作用: 若 param_group 有 initial_lr (TCN wrapper 路径),按其等比衰减;
    #       否则 (baseline_only 旧路径) 用原逻辑,保持向后兼容。
    # 数学:
    #   TCN 路径: lr_g = initial_lr_g × 0.5^(epoch-1)
    #   baseline_only: lr_g = lr (原行为, lr 已是 lr_×0.5^(epoch-1))
    # ============================================================
    # #7 修复 (2026-05-31): LR decay every 5 epochs instead of every 1 epoch
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 5))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            init_lr = param_group.get('initial_lr', None)
            if init_lr is not None:
                param_group['lr'] = init_lr * (0.5 ** ((epoch - 1) // 5))
            else:
                param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    r"""
    科研注释：类 `EarlyStopping` (基线借鉴 #7 — val_f1 判据 + patience 加大版)
    名称作用：在训练过程中按 val_loss 或 val_f1 判据自动停止, 避免过拟合也避免欠拟合。
    参数说明：构造函数提供 patience / verbose / dataset_name / delta / metric;
              metric='val_loss' 走原行为 (基线 baseline_suite 沿用 patience=3);
              metric='val_f1' 走基线借鉴 #7 新行为, 以 -F1 替代 -loss, 与目标指标对齐。
    返回值：通过 __call__ 内部状态更新 early_stop 标志。
    数学原理: θ* = argmax_θ F1_val(θ); 把 score 定义为 -metric, 让 minimize -F1 等价 maximize F1。
    流程说明：每 epoch 由 Solver.train 调一次, 接收 val_loss 或 val_f1 → 更新 best_score → 比较 patience。
    SCADA 迁移：F1 判据让早停与"轴承故障是否被检出"对齐, 不再被 reconstruction loss 误导。
    """
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0, metric='val_loss'):
        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`patience`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`verbose`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`dataset_name`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`delta`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        # 基线借鉴 #7: 判据切换. 'val_loss' 走原 minimax 双 loss 行为;
        #              'val_f1' 走借鉴新行为 (Solver.train 通过 epoch_callback 传入 val_f1)。
        self.metric = metric
        self.best_f1 = -1.0   # 'val_f1' 模式下记录最优 F1, 用于 score = -F1 计算

    def __call__(self, val_loss, val_loss2, model, path):
        r"""
        科研注释：函数/方法 `__call__`
        名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
        参数说明：`val_loss`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`val_loss2`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`model`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`path`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回值保持原实现约定，调用方依赖其形状和类型。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        # ============================================================
        # 基线借鉴 #7: 判据切换
        # 原行为 ('val_loss'): score = -val_loss, 双 loss 都要更优才保 checkpoint;
        # 新行为 ('val_f1'):   val_loss 参数被解释为 -val_f1, score = val_f1 (越大越好)。
        #     Solver.train 通过 epoch_callback 把 val_f1 传给 EarlyStopping,
        #     调用形式: early_stopping(-val_f1, -val_f1, model, path)。
        # 数学原理: argmax F1 等价 argmin -F1, 让 EarlyStopping 内部统一 score 越大越好。
        # ============================================================
        if self.metric == 'val_f1':
            # val_loss 形参在 val_f1 模式下传入的是 -val_f1, 取反还原
            current_f1 = -val_loss
            score = current_f1   # score 越大越好
            score2 = current_f1  # F1 模式下不区分 series/prior 双 loss
            if self.best_score is None:
                self.best_score = score
                self.best_score2 = score2
                self.best_f1 = current_f1
                self.save_checkpoint(val_loss, val_loss2, model, path)
            elif score < self.best_score + self.delta:
                # F1 没有提升, patience 计数
                self.counter += 1
                print(f'EarlyStopping[val_f1] counter: {self.counter}/{self.patience} (current_f1={current_f1:.4f}, best_f1={self.best_f1:.4f})')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.best_score2 = score2
                self.best_f1 = current_f1
                self.save_checkpoint(val_loss, val_loss2, model, path)
                self.counter = 0
            return

        # 原 val_loss 行为 (兼容路径)
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        r"""
        科研注释：函数/方法 `save_checkpoint`
        名称作用：保存预处理数据、模型权重或训练状态，保证实验可复现。
        参数说明：`val_loss`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`val_loss2`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`model`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关，`path`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：通常无显式返回值；副作用是写出实验中间产物。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    r"""
    科研注释：类 `Solver`
    名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
    参数说明：无显式函数参数；类属性由构造函数或成员方法定义。
    返回值：返回值保持原实现约定，调用方依赖其形状和类型。
    数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
    流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
    关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
    SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
    """
    DEFAULTS = {}

    def __init__(self, config):

        r"""
        科研注释：函数/方法 `__init__`
        名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。
        参数说明：`config`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：无显式返回值；副作用是建立对象状态。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        self.__dict__.update(Solver.DEFAULTS, **config)

        # #E1 修复 (2026-05-31): val/test/thre 用 step=1 避免仅产生 5-44 个窗口; train 保留 step=100 不双步进
        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=1,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=1,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=1,
                                              mode='thre',
                                              dataset=self.dataset)

        # 2026-06-03: device 必须在 build_model() 之前赋值 —— build_model() 内
        #   self.model.to(self.device) 依赖它 (原顺序在 build_model 之后赋值, CUDA 下
        #   触发 AttributeError: 'Solver' object has no attribute 'device')。仅调整赋值顺序,
        #   不改设备选择逻辑 (cuda:0 优先, 否则 cpu), 与原语义等价。
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.criterion = nn.MSELoss()

    def build_model(self):
        r"""
        科研注释：函数/方法 `build_model`
        名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
        参数说明：无外部业务参数；主要使用对象内部状态或全局实验配置。
        返回值：返回值保持原实现约定，调用方依赖其形状和类型。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.to(self.device)

    # ============================================================
    # 名称: save_training_checkpoint / restore_training_checkpoint
    # 修改原因 (2026-05-30 断点续传): 原 Solver 只通过 EarlyStopping 保存最优模型权重,
    #   不保存 optimizer/epoch 状态,中断后无法从当前 epoch 继续,只能从 epoch 0 重训。
    #   与 TranAD/TriTrackNet 的 save_checkpoint/restore_checkpoint 对标,
    #   让 AT 也具备 epoch 级断点续传能力。
    # 作用:
    #   save_training_checkpoint: 每 epoch 结束时保存 model+optimizer+epoch 完整状态;
    #   restore_training_checkpoint: 恢复状态并返回 last_epoch,无 checkpoint 返回 -1。
    # 数学原理: 无; 保存/恢复 PyTorch 训练状态字典。
    # 执行流程:
    #   1. save: torch.save({epoch, model_state_dict, optimizer_state_dict}) → 独立文件;
    #   2. restore: 读文件 → load_state_dict → 返回 epoch。
    # 科研标准: 续传 ckpt (_training_checkpoint.pth) 与最优模型 ckpt (_checkpoint.pth) 分离,
    #   互不覆盖; 均可由 (farm, module, seed, mode) 唯一标识。
    # ============================================================
    def save_training_checkpoint(self, epoch: int, path: str):
        """保存完整训练状态 (model + optimizer + epoch), 用于断点续传."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if getattr(self, "checkpoint_identity", None):
            state["checkpoint_identity"] = self.checkpoint_identity
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        ckpt_path = os.path.join(path, f"{self.dataset}_training_checkpoint.pth")
        torch.save(state, ckpt_path)

    def restore_training_checkpoint(self, path: str) -> int:
        """恢复训练状态; 返回 last_epoch (0-based), 无 checkpoint 返回 -1."""
        ckpt_path = os.path.join(path, f"{self.dataset}_training_checkpoint.pth")
        if not os.path.exists(ckpt_path):
            return -1
        state = torch.load(ckpt_path, map_location=self.device)
        expected_identity = getattr(self, "checkpoint_identity", None)
        if expected_identity:
            from 实验工具 import checkpoint_identity_is_compatible
            if not checkpoint_identity_is_compatible(
                state,
                expected_identity,
                model="AnomalyTransformer",
                checkpoint_path=ckpt_path,
            ):
                return -1
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if hasattr(self, 'scheduler') and self.scheduler is not None \
                and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        return int(state["epoch"])

    def vali(self, vali_loader):
        r"""
        科研注释：函数/方法 `vali`
        名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
        参数说明：`vali_loader`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回值保持原实现约定，调用方依赖其形状和类型。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        self.model.eval()

        # 2026-06-03 提速: vali 每 epoch 调一次, 加 bf16 autocast + inference_mode (无 autograd 更快),
        #   并把 loss1/loss2 在 GPU 累加, epoch 末一次性同步 (替代每 batch 两次 .item() 同步)。
        #   bf16 仅降精度, KL/重构 loss 公式不变; inference_mode 不建图与 no_grad 等价但更快。
        _amp_enabled = torch.cuda.is_available() and getattr(self, 'amp_enabled', True)
        _amp_dtype = torch.bfloat16
        _loss1_sum = None
        _loss2_sum = None
        _cnt = 0
        with torch.inference_mode():
            for i, (input_data, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=_amp_dtype, enabled=_amp_enabled):
                    output, series, prior, _ = self.model(input)
                    series_loss = 0.0
                    # 科研流程注释：先累积 prior 与 series 的 KL 差异，随后分别用于最小化和最大化分支。
                    prior_loss = 0.0
                    for u in range(len(prior)):
                        series_loss += (torch.mean(my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach())) + torch.mean(
                            my_kl_loss(
                                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)).detach(),
                                series[u])))
                        prior_loss += (torch.mean(
                            my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                               self.win_size)),
                                       series[u].detach())) + torch.mean(
                            my_kl_loss(series[u].detach(),
                                       (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                               self.win_size)))))
                    series_loss = series_loss / len(prior)
                    prior_loss = prior_loss / len(prior)

                    rec_loss = self.criterion(output, input)
                # GPU 累加 (转 fp32 保证求和精度), 不每 batch 同步; epoch 末一次 .item()。
                _l1 = (rec_loss - self.k * series_loss).detach().float()
                _l2 = (rec_loss + self.k * prior_loss).detach().float()
                _loss1_sum = _l1 if _loss1_sum is None else (_loss1_sum + _l1)
                _loss2_sum = _l2 if _loss2_sum is None else (_loss2_sum + _l2)
                _cnt += 1

        if _cnt == 0:
            return 0.0, 0.0
        return float(_loss1_sum.item() / _cnt), float(_loss2_sum.item() / _cnt)

    def train(self):
        return self._train(start_epoch=0)

    # ============================================================
    # 名称: _train (2026-05-30 断点续传: train 抽取为 _train + start_epoch 参数)
    # 修改原因: 原 train() 固定从 epoch 0 开始; 新增 start_epoch 参数使训练可从断点 epoch 继续,
    #   不再重跑已完成 epoch。恢复流程: restore_training_checkpoint → start_epoch=last_epoch+1。
    # 作用: 替代 train() 的内部逻辑; train() 作为向后兼容入口委托到 _train(start_epoch=0)。
    # 数学原理: 训练目标与原 train() 完全相同,仅 for epoch in range(start_epoch, num_epochs)。
    # 执行流程:
    #   1. EarlyStopping 仍从 0 初始化 (不与 epoch 绑定);
    #   2. for epoch from start_epoch (而非 0), 每个 epoch 结束时 save_training_checkpoint;
    #   3. 其余 (minimax 更新/AMP/epoch_callback/早停) 与原 train 完全相同。
    # 科研标准: 中断后从同 epoch 继续, 训练预算 (总 epoch 数) 不增加, checkpoint 状态完整可溯。
    # ============================================================
    def _train(self, start_epoch: int = 0):
        r"""
        科研注释：函数/方法 `_train`
        名称作用：执行训练或反向传播流程，支持从指定 epoch 开始以支持断点续传。
        参数说明：start_epoch：恢复时从该 epoch (0-based) 开始, 原 train() 调用时 start_epoch=0。
        返回值：返回值保持原实现约定，调用方依赖其形状和类型。
        数学原理：该符号主要实现工程流程，本身不新增独立数学假设；关键公式见模块级 docstring。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        # ============================================================
        # 名称: EarlyStopping 配置 (基线借鉴 #7)
        # 修改原因 (2026-05-26 借鉴自基线): 基线沿用 AT 原版 patience=3 + val_loss;
        #     创新经 Phase D 实测 EPOCHS=10 固定关掉早停后 seed 间方差大,
        #     部分 seed 早过拟合, 部分未收敛。
        # 作用: 默认 patience=5 + metric='val_loss' 兼容原行为;
        #     若 Solver 实例从外置 实验.py 显式设了 self.early_stop_metric='val_f1' 和
        #     self.early_stop_patience, 用更新后的判据 (与目标指标对齐)。
        # 科研标准: patience=5 防止单 epoch 抖动误停; val_f1 判据让早停与 F1 评价一致。
        # ============================================================
        _es_patience = getattr(self, 'early_stop_patience', 5)
        _es_metric = getattr(self, 'early_stop_metric', 'val_loss')
        early_stopping = EarlyStopping(
            patience=_es_patience,
            verbose=True,
            dataset_name=self.dataset,
            metric=_es_metric,
        )
        train_steps = len(self.train_loader)
        # 基线借鉴 #1 + 2026-06-03 提速: AMP 混合精度训练 (bf16, 无 GradScaler)
        # 修改原因 (2026-06-03): GPU 仅 ~23% 利用, 模型 fp32 训练受同步/数据加载阻塞;
        #     用户接受混合精度 (bf16) 换 ~3x 提速。改用 bfloat16 而非 float16:
        #     bf16 与 fp32 同指数范围, KL/association 项不会 underflow, 无需 GradScaler,
        #     数值更稳 (float16 在 my_kl_loss 的 log/sum 上易 inf/NaN)。
        # 作用: 前向 + minimax 两个 loss 在 bf16 autocast 下计算; 反向/optimizer.step 用 fp32 主参数。
        #     不再使用 GradScaler (bf16 不需要 loss scaling), 反向直接 backward + step。
        # 数学原理: bf16 仅降低尾数精度, 不改变 reconstruction + k·KL 的 minimax 目标公式;
        #     两次 backward (loss1 retain_graph + loss2) + 单次 optimizer.step 流程完全保留。
        # 科研标准: 仅 CUDA 时启用; loss 数学/反传次数/step 次数与 fp32 路径逐位等价 (仅精度档位变)。
        _amp_enabled = torch.cuda.is_available() and getattr(self, 'amp_enabled', True)
        _amp_dtype = torch.bfloat16

        # 科研流程注释：训练循环同时优化重构误差和关联差异；minimax 两阶段更新用于放大正常/异常关联模式差别。
        # ============================================================
        # 名称: 断点续传: start_epoch 替代硬编码 0 (2026-05-30)
        # 修改原因: 原 for epoch in range(self.num_epochs) 固定从 0 开始, 中断后无法续传。
        #   新: for epoch in range(start_epoch, self.num_epochs), 恢复时从 last_epoch+1 继续。
        # 作用: 每 epoch 结束时调用 save_training_checkpoint 保存完整状态;
        #   若中断, restore_training_checkpoint 恢复 → 从同 epoch 继续, 不重跑已完成 epoch。
        # 数学原理: 训练目标与原 train() 完全相同, 仅 epoch 迭代起点可变。
        # 科研标准: 训练预算 (总 epoch 数) 不因中断而增加, checkpoint 状态完整可溯。
        # ============================================================
        for epoch in range(start_epoch, self.num_epochs):
            iter_count = 0
            # 2026-06-03 提速: 延迟 GPU→CPU 同步。原 loss1_list.append(loss1.item()) 每 batch
            #   都触发一次设备同步 (GPU ~23% 利用的元凶之一)。改为在 GPU 上累加 detach 后的
            #   loss1, epoch 末一次性 .item() 取均值, 与 np.average(per-batch loss1) 数值等价。
            _loss1_sum = None   # GPU 标量张量, 累加每 batch 的 loss1.detach()
            _loss1_cnt = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                # 基线借鉴 #9: non_blocking=True 配合 DataLoader 的 pin_memory
                input = input_data.float().to(self.device, non_blocking=True)

                # ============================================================
                # 基线借鉴 #1 + 2026-06-03: AMP autocast 包前向 + KL loss 计算 (bf16)
                # 原因: 全部前向放进 bf16 autocast; bf16 指数范围同 fp32, KL 项不 underflow,
                #       无需 GradScaler, 两次 backward 直接执行 (见下方 minimax 块)。
                # ============================================================
                with torch.autocast(device_type=self.device.type, dtype=_amp_dtype, enabled=_amp_enabled):
                    output, series, prior, _ = self.model(input)

                    # calculate Association discrepancy
                    series_loss = 0.0
                    prior_loss = 0.0
                    for u in range(len(prior)):
                        series_loss += (torch.mean(my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach())) + torch.mean(
                            my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                               self.win_size)).detach(),
                                       series[u])))
                        prior_loss += (torch.mean(my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach())) + torch.mean(
                            my_kl_loss(series[u].detach(), (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)))))
                    series_loss = series_loss / len(prior)
                    prior_loss = prior_loss / len(prior)

                    rec_loss = self.criterion(output, input)

                    loss1 = rec_loss - self.k * series_loss
                    loss2 = rec_loss + self.k * prior_loss

                # 2026-06-03 提速: 在 GPU 上累加 (detach, 转 fp32 保证求和精度), 不每 batch 同步。
                _l1 = loss1.detach().float()
                _loss1_sum = _l1 if _loss1_sum is None else (_loss1_sum + _l1)
                _loss1_cnt += 1

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # ============================================================
                # Minimax strategy (2026-06-03 bf16 无 GradScaler 版)
                # 修改原因: bf16 指数范围同 fp32, 不需要 loss scaling, 移除 GradScaler。
                #     恢复为原论文的 backward(retain_graph=True) + backward() + optimizer.step(),
                #     梯度在 fp32 主参数上累加 (autocast 已在 forward 退出, backward 用 fp32)。
                # 科研标准: minimax 两次反传 + 共享一次 optimizer.step 流程与 fp32 行为逐位等价。
                # ============================================================
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # 2026-06-03 提速: epoch 末一次性同步 GPU 累加和 → 均值 (= np.average(per-batch loss1))。
            train_loss = float(_loss1_sum.item() / _loss1_cnt) if _loss1_cnt > 0 else 0.0

            # ============================================================
            # SCADA实验修改:
            # 名称: 验证集隔离修正
            # 修改原因: 原代码在训练期将 self.test_loader 传入 vali(),会让
            #          测试时间段参与 EarlyStopping,形成模型选择数据泄漏。
            # 作用: 改为仅使用 self.vali_loader 计算验证损失和早停依据。
            # 数学原理: 泛化评价要求 θ* = argmin L_val(θ),最终仅一次报告
            #          L_test(θ*);测试集不能参与 θ* 的选择。
            # 执行流程:
            #   1. train_loader 更新参数;
            #   2. vali_loader 计算 early-stopping loss;
            #   3. 保存最优 checkpoint;
            #   4. 训练结束后才在 test_loader 上评价。
            # 科研标准: 时间切分严格隔离 train/validation/test,杜绝测试泄漏。
            # ============================================================
            vali_loss1, vali_loss2 = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            # ============================================================
            # SCADA实验修改:
            # 名称: epoch_callback 统一指标回调
            # 修改原因: 原 Solver.train 只打印损失,不能输出本项目要求的每轮
            #          accuracy/precision/recall/F1/AUC 与标准化结果文件。
            # 作用: 若外置 实验.py 注册回调,在本 epoch 参数更新结束后进行只读评价。
            # 数学原理: 回调只计算当前模型的异常分数与二分类指标,不参与
            #          L = reconstruction_loss +/- k * association_discrepancy 的反向传播。
            # 执行流程:
            #   1. 当前 epoch 完成 minimax 参数更新;
            #   2. 计算 train/validation loss;
            #   3. 调用外置回调打印并落盘指标;
            #   4. 原 EarlyStopping 流程继续执行。
            # 科研标准: 回调不得 optimizer.step,不得使用 test 集选择训练策略。
            # ============================================================
            # ============================================================
            # 基线借鉴 #7: epoch_callback 返回 val_f1 让早停按 F1 判
            # 修改原因 (2026-05-26): 原 epoch_callback 不返回值, 仅写日志;
            #     新行为下若 self.early_stop_metric=='val_f1', 让 callback 返回 val_f1,
            #     Solver 把它传给 EarlyStopping 走 val_f1 路径。
            # 兼容性: callback 若不返回 (旧实现), 走原 val_loss 路径不受影响。
            # ============================================================
            _val_f1 = None
            if hasattr(self, "epoch_callback") and callable(self.epoch_callback):
                _ret = self.epoch_callback(epoch + 1, train_loss, vali_loss1)
                if isinstance(_ret, (int, float)):
                    _val_f1 = float(_ret)
            if _es_metric == 'val_f1' and _val_f1 is not None:
                # F1 模式: 传 -F1 进去 (EarlyStopping 内部还原), val_loss2 形参占位用同值
                early_stopping(-_val_f1, -_val_f1, self.model, path)
            else:
                early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                # 早停时也保存训练状态, 让后续 resume 能识别训练已结束
                self.save_training_checkpoint(epoch, path)
                break
            # ============================================================
            # 名称: 断点续传: 每 epoch 结束保存训练 checkpoint (2026-05-30)
            # 修改原因: 原 train() 训练中断后无 optimizer/epoch 状态, 只能从 epoch 0 重训。
            #   新: 每个 epoch 结束时保存 model+optimizer+epoch 完整状态到独立文件,
            #   restore_training_checkpoint 恢复后从当前 epoch 继续, 不重跑已完成 epoch。
            # 作用: 与 TranAD/TriTrackNet 的每 epoch save_checkpoint 模式对标,
            #   让 AT 中断后从同 epoch 继续训练, 训练预算不增加。
            # 数学原理: 无; 保存/恢复 PyTorch 训练状态字典。
            # 科研标准: 续传 ckpt (_training_checkpoint.pth) 与最优模型 ckpt (_checkpoint.pth) 分离。
            # ============================================================
            self.save_training_checkpoint(epoch, path)
            # ============================================================
            # 名称: scheduler step (2026-05-31 超参数寻优)
            # 修改原因: 原 adjust_learning_rate 硬编码阶梯衰减, 现替换为
            #   外部注入的 scheduler (CosineAnnealing / ReduceLROnPlateau / StepLR),
            #   支持网格搜索自动探索最优调度策略。
            #   保留了 baseline_only 无外部 scheduler 时的向后兼容
            #   (调用原 adjust_learning_rate)。
            # ============================================================
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                from 实验配置 import SchedulerProtocol
                SchedulerProtocol.step_scheduler(self.scheduler,
                    val_loss=train_loss if 'ReduceLROnPlateau' in type(self.scheduler).__name__ else None)
            else:
                adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
# ============================================================
