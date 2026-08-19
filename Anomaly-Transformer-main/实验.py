# -*- coding: utf-8 -*-
"""
实验.py

名称: Anomaly Transformer 的 SCADA 专用实验入口
修改原因: 原 main.py 只负责训练/测试,指标缺少 AUC,且不能按本项目 ignore 标签规则统一落盘。
作用: 在不大幅改动原论文源码的前提下,完成 SCADA 训练、验证阈值选择、测试指标输出和 checkpoint 续跑。
数学原理:
    1. 重建误差 e_t = mean((x_t - xhat_t)^2)
    2. Association Discrepancy 由 series/prior 的 KL 散度构成;
    3. 异常分数 score_t = softmax(-KL_series - KL_prior) * e_t;
    4. 阈值由验证集 F1 或训练分数分位数确定。
执行流程:
    1. 读取 dataset/SCADA 的 train/val/test;
    2. 若 checkpoint 不存在或 --force,调用原 Solver.train();
    3. 加载 checkpoint,计算 train/val/test 异常分数;
    4. 用验证集选择阈值,测试集只做最终评价;
    5. 打印并保存 loss/accuracy/precision/recall/F1/AUC。
科研标准: 保留原 minimax 训练流程;不使用测试集调阈值;label=-1 从指标中剔除。
参考文献:
    Xu, J., Wu, H., Wang, J., & Long, M. (2022). Anomaly Transformer:
    Time Series Anomaly Detection with Association Discrepancy. ICLR 2022.
    PDF: https://arxiv.org/pdf/2110.02642
    Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017.
    PDF: https://arxiv.org/pdf/1706.03762
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if not hasattr(np, "Inf"):
    np.Inf = np.inf

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

# 2026-05-30 统一: 窗口大小单一真源 = 实验配置.DatasetProtocol.WIN_AT (不再在本文件硬编码 100)。
try:
    from 实验配置 import DatasetProtocol as _DP  # noqa: E402
    _WIN_AT_DEFAULT = int(_DP.WIN_AT)
except Exception:
    _WIN_AT_DEFAULT = 100

from solver import Solver  # noqa: E402
from 实验工具 import (  # noqa: E402
    add_preprocess_identity_to_metrics,
    build_checkpoint_identity,
    build_preprocess_identity,
    choose_threshold_and_polarity_by_validation,
    compute_binary_metrics,
    init_wandb_run,
    log_epoch_to_wandb as _log_wandb_epoch,
    finish_wandb_run,
    orient_scores,
    record_and_print_metric,
)


RESULT_DIR = ROOT / "实验结果" / "anomaly_transformer"
CSV_PATH = ROOT / "实验结果" / "metrics.csv"


def _rng_state_snapshot():
    """Capture global RNG state so metric-only probes cannot perturb training."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


# ============================================================
# 名称: result_dir_for_farm
# 修改原因 (2026-05-25 全量化): metrics.jsonl 之前合并写在 实验结果/anomaly_transformer/,
#          多 farm 同时跑会乱序;按 farm 隔离避免读写冲突 + 便于横向 join。
# 作用: 给定 farm 名 + 可选 override,返回 AT 该 farm 的 metrics 目录。
# 数学原理: 无 (路径计算)。
# 执行流程:
#   1. output_dir_override 非空 → 直接用 (供 启动.py 显式传入);
#   2. 其它 farm → 实验结果/{farm}/anomaly_transformer/。
# 科研标准: 与 TranAD / TriTrackNet 的 result_dir_for_farm 形成同构三元组,
#          横向汇总时 glob 实验结果/*/anomaly_transformer/metrics.jsonl 即可。
# ============================================================
def result_dir_for_farm(farm: str, output_dir_override: str | None = None) -> Path:
    if output_dir_override:
        return Path(output_dir_override)
    return ROOT / "实验结果" / farm / "anomaly_transformer"


# ============================================================
# 名称: parse_args
# 修改原因: 需要同时支持正式实验、smoke 测试和 checkpoint 续跑。
# 作用: 定义 Anomaly Transformer SCADA 实验参数。
# 数学原理: 无。
# 执行流程: argparse 解析命令行并返回 Namespace。
# 科研标准: 关键超参数显式化,便于复现实验记录。
# ============================================================
def parse_args() -> argparse.Namespace:
    r"""
    科研注释：函数 `parse_args`
    名称作用：把 SCADA 专用实验的训练周期、窗口长度、学习率、续跑方式、模块选择和 smoke 模式暴露为可复现实验参数。
    参数说明：无函数参数；参数来源为命令行选项。
    返回值：返回 `argparse.Namespace`，供配置构建、训练与评价阶段共同使用。
    数学原理：本函数不计算模型公式；其中 `k` 对应关联差异正则项在 minimax 目标中的权重。
    流程说明：创建解析器 -> 声明超参数及执行模式 -> 解析并返回参数对象。
    SCADA 迁移：把采样窗口和优化参数显式记录，便于按机组、故障部件和信号类型复现实验。
    """
    parser = argparse.ArgumentParser(description="Anomaly Transformer SCADA 实验入口")
    # 最优设置: epochs 默认 = TrainingProtocol.EPOCHS=10 (统一训练预算基准);
    #     上限由 TrainingProtocol.EPOCHS_MAX=20 + 早停(val_f1, patience=5) 自动截断;
    #     原因: 10 epoch 让 TCN α(初值 0.01) 充分收敛到稳态后再对比。
    parser.add_argument("--epochs", type=int, default=10,
                        help="正式实验 epoch 数 (TrainingProtocol.EPOCHS=10, 上限 EPOCHS_MAX=20)")
    # 最优设置: batch_size 默认 = TrainingProtocol.BATCH_SIZE=128 (统一比较基线);
    #     AMP 模式下内部扩到 BATCH_SIZE_AMP=512 (Var(g̃_B) ∝ σ²/B, 梯度方差更小);
    #     epoch/default 均来自 实验配置.TrainingProtocol 单一真源.
    parser.add_argument("--batch-size", type=int, default=128, help="batch size (TrainingProtocol.BATCH_SIZE=128)")
    parser.add_argument("--win-size", type=int, default=_WIN_AT_DEFAULT,
                        help="滑动窗口长度 (默认取自 实验配置.DatasetProtocol.WIN_AT)")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--k", type=int, default=3, help="KL 项权重")
    # 基线借鉴 #8: 新增 tcn_wavelet_residual 选项 (默认 RUN_VARIANTS 仍不含, 需 --module 显式开)
    parser.add_argument("--module", type=str, default="baseline_only",
                        choices=["baseline_only", "tcn_input_residual", "tcn_wavelet_residual"],
                        help="Phase D 模块选择 (基线借鉴 #8: 新增 tcn_wavelet_residual)")
    # 最优设置: seed 通过 启动.py 的 --seeds 0,1,2,3,4 批量注入 (TrainingProtocol.SEEDS);
    #     5 seed 比 3 seed 的 mean±std 更稳健 (n=5 是非参数检验最小可接受样本).
    parser.add_argument("--seed", type=int, default=0, help="随机种子 (启动器注入 0-4; 单跑默认 0)")
    parser.add_argument("--run-id", type=str, default="run_001", help="运行标识")
    parser.add_argument("--output-dir", type=str, default=None, help="覆盖默认输出目录")
    parser.add_argument("--resume", action="store_true", help="存在 checkpoint 时跳过训练直接测试")
    parser.add_argument("--force", action="store_true", help="强制重新训练")
    parser.add_argument("--smoke", action="store_true", help="一轮快速链路验证")
    # 修改原因 (2026-05-23 全量化): 三 farm 全量化要求按 farm 选数据目录,
    #          farm 名同时进入 checkpoint 路径与 metrics.jsonl 记录,方便追溯。
    # 作用: 数据目录 = SCRIPT_DIR/dataset/SCADA_{farm},checkpoint 路径加 farm 段。
    # 数学原理: 无 (字符串拼接)。
    # 科研标准: farm 是不可统一控制变量,每条 metrics 行必须含 farm 字段。
    parser.add_argument("--farm", type=str, default="kelmarsh",
                        choices=["kelmarsh", "penmanshiel", "hill_of_towie"],
                        help="全量化实验 farm 名 (默认 kelmarsh, 对应 dataset/SCADA_kelmarsh/)")
    parser.add_argument("--split-id", type=str, default="narrow_v1", help="切分协议 id (全量=chronological_v2)")
    parser.add_argument("--feature-version", type=str, default="v1", help="特征版本 (全量=v2)")
    parser.add_argument("--preprocess-variant", type=str, default="",
                        help="预处理变体后缀, 例如 old_preprocess / new_preprocess")
    # 基线借鉴 #6: 预训练 ckpt 路径; 给定时模型构造后立即 load_state_dict 做微调起点
    parser.add_argument("--pretrain-ckpt", type=str, default=None,
                        help="基线借鉴 #6: 预训练 checkpoint 路径 (微调时加载)")
    # ============================================================
    # 名称: --scheduler (2026-05-31 超参数寻优)
    # 修改原因: 原 AT 硬编码 adjust_learning_rate 阶梯衰减, 不支持
    #   CosineAnnealingLR / ReduceLROnPlateau / StepLR 网格搜索。
    # 作用: 接收 scheduler 类型名, 训练前构建并注入 solver.scheduler,
    #   替换 _train() 内部的 adjust_learning_rate 调用。
    # 兼容: 不给 --scheduler 时保持旧行为 (adjust_learning_rate)。
    # ============================================================
    parser.add_argument("--scheduler", type=str, default=None,
                        choices=["cosine", "plateau", "steplr"],
                        help="学习率调度器类型 (网格搜索注入; 默认 None=原 adjust_learning_rate)")
    return parser.parse_args()


# ============================================================
# 名称: build_config
# 修改原因: 原 Solver 需要 argparse 风格配置对象,实验入口需要自动填入 SCADA 通道数。
#   (2026-06-02 增补) 数据目录与 checkpoint 需按 split_id/feature_version 版本隔离,
#   否则全量 chronological_v2 会读/写到历史 narrow_v1 的目录, 互相覆盖污染。
# 作用: 根据 train.npy 的 shape[1] 动态推断 input_c/output_c 并构造 Solver 配置;
#   数据目录与 checkpoint 路径均按 split 版本隔离 (narrow_v1 保持旧路径)。
# 数学原理: input_c = output_c = D,其中 X ∈ R^{T×D}。
# 执行流程:
#   1. 用 split_id/feature_version 经 PerFarmPaths 取版本化数据目录;
#   2. 读取 train 数组 shape, 取第 2 维作为通道数;
#   3. smoke 模式将 epoch 降为 1;
#   4. checkpoint 路径仅在非 narrow_v1 时插入版本段;
#   5. 返回 SimpleNamespace。
# 科研标准: 通道数来自真实数据文件,避免手写参数与预处理输出不一致;
#   narrow_v1 默认路径与历史字节级一致, chronological_v2 与其物理隔离。
# ============================================================
def build_config(args: argparse.Namespace) -> SimpleNamespace:
    r"""
    科研注释：函数 `build_config`
    名称作用：从实际 SCADA 训练数组推断通道数，并构造原 `Solver` 可直接消费的配置对象。
    参数说明：`args` 为 `parse_args` 输出的实验参数集合。
    返回值：返回 `SimpleNamespace`，包含数据路径、checkpoint 路径、通道数、窗口长度和训练超参数。
    数学原理：对 $X\in\mathbb{R}^{T\times D}$，设置 `input_c=output_c=D`，使重构输出与每个传感器通道对齐。
    流程说明：读取训练数组 shape -> 根据 smoke 模式确定 epoch -> 填充 Solver 配置。
    SCADA 迁移：通道数必须来自预处理后的真实数组，避免温度或振动变量筛选后仍沿用旧维度。
    """
    # ============================================================
    # 名称: 数据目录 + checkpoint 版本隔离 (2026-06-02 split/feature 隔离)
    # 修改原因: 旧代码 PerFarmPaths.for_farm(args.farm) 不传 split_id/feature_version,
    #   全量 chronological_v2 会读到 narrow_v1 的 SCADA_{farm}/ 目录 (读错数据);
    #   且 model_save_path 不含版本段, narrow_v1 与 chronological_v2 的 checkpoint
    #   会写同一路径互相覆盖 (科研标准: 两版本必须物理隔离)。
    # 作用:
    #   1. 数据目录改为 PerFarmPaths.for_farm(args.farm, split_id, feature_version),
    #      narrow_v1 仍解析为旧 SCADA_{farm}/, 其它 split 解析为版本化目录;
    #   2. checkpoint 路径仅在 split_id != narrow_v1 时在 farm 段后插入
    #      f"{split_id}__{feature_version}" 段, narrow_v1 保持旧路径字节级一致。
    # 数学原理: 无 (路径拼接)。
    # 执行流程: 取 split_id/feature_version → PerFarmPaths 取数据目录 → 校验 train.npy →
    #   按 split 是否为 narrow_v1 决定 checkpoint 是否插版本段。
    # 科研标准: narrow_v1 (默认) 的数据/checkpoint 路径与历史完全一致 (既有测试依赖);
    #   chronological_v2 与 narrow_v1 互不读写, 复跑互不污染。
    # ============================================================
    from 实验配置 import PerFarmPaths
    split_id = args.split_id
    feature_version = args.feature_version
    preprocess_variant = str(getattr(args, "preprocess_variant", "") or "").strip()
    dataset_dir = PerFarmPaths.for_farm(
        args.farm, split_id, feature_version, preprocess_variant=preprocess_variant,
    )["anomaly_transformer"]
    # 数据契约 (预处理 S9): 数据文件名固定 train.npy; 通道数 C 从其 shape[1] 动态获取。
    if not (dataset_dir / "train.npy").exists():
        raise FileNotFoundError(f"数据目录不存在: {dataset_dir}; 请先运行 SCADA数据集/数据预处理.py --mode full --farm {args.farm}")
    train = np.load(dataset_dir / "train.npy")
    epochs = 1 if args.smoke else args.epochs
    # checkpoint 版本隔离: narrow_v1 保持旧路径 (.../checkpoints/farm/kind/module/seedN);
    #   其它 split 在 farm 段后插入 {split_id}__{feature_version} 段。
    ckpt_root = SCRIPT_DIR / "checkpoints" / args.farm
    if split_id != "narrow_v1":
        tag = f"{split_id}__{feature_version}"
        if preprocess_variant:
            tag = f"{tag}__{preprocess_variant}"
        ckpt_root = ckpt_root / tag
    model_save_path = str(ckpt_root / ("smoke" if args.smoke else "formal") / args.module / f"seed{args.seed}")
    return SimpleNamespace(
        lr=args.lr,
        num_epochs=epochs,
        k=args.k,
        win_size=args.win_size,
        input_c=int(train.shape[1]),
        output_c=int(train.shape[1]),
        batch_size=args.batch_size,
        pretrained_model=None,
        dataset="SCADA",
        mode="train",
        data_path=str(dataset_dir),
        # SCADA实验修改 (2026-05-23 全量化): farm + smoke/formal + module + seed 五级隔离
        # 修改原因: 不同 farm × module × seed 的 checkpoint 必须各自隔离,否则交叉运行会互相覆盖。
        model_save_path=model_save_path,
        anormly_ratio=4.0,
    )


# ============================================================
# 名称: collect_energy_and_labels (窗口末点对齐 + 逐点重构误差)
# 修改原因 (2026-05-26 借鉴自基线): 原实现把窗口内 L 个时间点的 energy 全部展平到一维
#         (`.reshape(-1)`), 同一时间点被多个相邻窗口评分多次, 平均后异常瞬态被稀释 → recall↓。
#         改为窗口末点对齐: 每个窗口只产出 1 个分数 (该窗口末点), 与 label 末点严格一一对应。
# #E2 修复 (2026-05-31): 异常分数直接取窗口末端的重构误差 rec_loss[:, -1], 不再用 softmax 对
#         L 维做关联权重归一化 (softmax 会把端点权重衰减 ~L 倍, 淹没 SCADA 的渐变温漂信号)。
# #D1 复核 (2026-06-01): 确认 scores.append(score) 存在 (修复"空列表 np.concatenate 崩溃");
#         score 形状 (B,) 与 labels 末点 (B,) 在每个 batch 内逐窗口对齐, 拼接后两数组等长。
# 作用: 复用已训练模型, 在每个滑动窗口的末点收集 1 个异常分数与对应末点标签。
# 数学原理:
#   1. 窗口 W_t = [x_{t-L+1}, ..., x_t] 的末点重构误差 score_t = mean_c (x_t - xhat_t)^2;
#   2. 每个窗口对应 1 个 score_t 与 1 个末点标签 y_t, 不重复计入相邻窗口;
#   3. step=1 时相邻窗口末点逐点推进, scores[i]/labels[i] 对应第 i 个窗口的末点时刻。
# 执行流程:
#   1. 遍历 loader 中每个窗口 batch;
#   2. 前向得到 output;
#   3. 取窗口末点 rec_loss[:, -1] 作为该窗口分数;
#   4. 同步取 batch_labels 末点; 拼接为等长 1D scores / labels。
# 科研标准: 只做前向评价, 不修改模型参数; 保留 raw label 中的 -1 供 实验工具 统一过滤,
#         不在此处对 -1 做聚合或剔除 (交给 compute_binary_metrics)。
# ============================================================
def collect_energy_and_labels(solver: Solver, loader) -> tuple[np.ndarray, np.ndarray]:
    r"""
    科研注释：函数 `collect_energy_and_labels` (窗口末点对齐版)
    名称作用：在固定模型参数下收集每个滑动窗口末点的 Anomaly Transformer 重构误差异常分数及对应标签。
    参数说明：`solver` 持有训练完成的模型与设备；`loader` 产生时间窗口及同区间标签。
    返回值：返回等长一维 `(scores, labels)` 数组, scores[i] 对应窗口 i 的末点时刻, labels[i] 同。
    数学原理：窗口末点重构误差 $e_t=\operatorname{mean}_c(x_t-\hat{x}_t)^2$ 作为异常分数;
    末点对齐: 每个窗口仅取最后一个时间步, 避免同一时间点被多个窗口重复评分而稀释瞬态。
    流程说明：关闭梯度 -> 前向 -> 取窗口末点 rec_loss[:, -1] -> 同步取 batch_labels 末点 -> 拼接输出。
    SCADA 迁移：该分数反映末点偏离正常重构模式的程度, 末点对齐让评价时序粒度与 label 一致。
    """
    criterion = nn.MSELoss(reduction="none")
    scores = []
    labels = []
    solver.model.eval()
    # 2026-06-03 提速: 评分前向用 inference_mode (不建图, 比 no_grad 更快) + bf16 autocast。
    #   bf16 与 fp32 同指数范围, 末点重构误差数值稳定; 仅降尾数精度, 不改 score_t 公式。
    #   逐窗口分数先以 GPU 张量累积, 循环结束一次性 .cpu() (替代每 batch 的 GPU→CPU 同步)。
    _amp_enabled = torch.cuda.is_available() and getattr(solver, "amp_enabled", True)
    _amp_dtype = torch.bfloat16
    with torch.inference_mode():
        for input_data, batch_labels in loader:
            x = input_data.float().to(solver.device)
            with torch.autocast(device_type=solver.device.type, dtype=_amp_dtype, enabled=_amp_enabled):
                output, series, prior, _ = solver.model(x)
                # 逐点重构误差 (B, L): 对通道维 D 取均值; 稍后只取末点 [:, -1] 与 label 末点对齐。
                rec_loss = torch.mean(criterion(x, output), dim=-1)  # (B, L)
            # #E2: 异常分数 = 窗口末端重构误差 (不做 softmax L 维归一化, 避免端点权重衰减)。
            #   保留为 GPU 张量 (转 fp32) 暂存, 不在每 batch 同步; 末点 [:, -1] 与 label 末点对齐。
            score = rec_loss[:, -1].detach().float()  # (B,) GPU 张量, 每窗口 1 个末点分数
            # #D1: 必须 append, 否则结尾 torch.cat / np.concatenate 对空列表崩溃。
            scores.append(score)
            if torch.is_tensor(batch_labels):
                bl = batch_labels.detach().cpu().numpy()
            else:
                bl = np.asarray(batch_labels)
            # 兼容两种 label 形状: (B, L) → 取末点 [:, -1] 与 score 对齐; (B,) → 直接用。
            if bl.ndim == 2:
                bl = bl[:, -1]
            labels.append(bl)
    # scores[i] 与 labels[i] 均为"第 i 个窗口的末点", 拼接后两数组逐点等长对齐。
    # 2026-06-03: GPU 张量先 cat 再一次性 .cpu().numpy(), 全程只一次设备同步。
    scores_np = torch.cat(scores, dim=0).cpu().numpy().reshape(-1)
    return scores_np, np.concatenate(labels, axis=0).reshape(-1)


# ============================================================
# 名称: record_epoch_metrics
# 修改原因: 原训练循环只报告损失,本项目要求每次训练和验证都显示统一指标。
# 作用: 作为 Solver.train 的只读 callback,输出当前 epoch 的 train/val 指标。
# 数学原理: 当前模型异常分数经验证集阈值转为预测标签,再按混淆矩阵计算指标。
# 执行流程:
#   1. 在 epoch 更新完成后收集 train/val score;
#   2. 验证集选择阈值或训练分数回退;
#   3. 分别计算 train/val 指标;
#   4. 输出并追加结果文件。
# 科研标准: 不读取测试集、不执行 optimizer.step,不会反向影响训练参数。
# ============================================================
def record_epoch_metrics(
    solver: Solver, epoch: int, train_loss: float, val_loss: float, run_kind: str,
    farm: str = "kelmarsh", module: str = "baseline_only", seed: int = 0,
    output_dir_override: str | None = None,
) -> None:
    """
    每个 epoch 训练结束后被 Solver.train 通过 epoch_callback 调用,
    收集 train/val 异常分数 → 选阈值 → 计算指标 → 写 metrics.jsonl + CSV。

    全量化新增 (2026-05-25):
    - farm/module/seed/output_dir_override 4 个新参 (通过 lambda 闭包从 main 传入);
    - 每条 epoch 级 metrics 都带这 3 个识别字段, 确保 实验结果/metrics.csv 横向 join 完整;
    - result_dir 由 result_dir_for_farm 决定, 自动按 farm 隔离落 jsonl。

    数学原理: 阈值 t* 由 choose_threshold_and_polarity_by_validation 在 val 上选最大 F1,
             pred = (orient_scores(score, polarity) > t*);指标按二分类混淆矩阵计算。

    #C1 统一评测口径 (2026-06-01): 移除 point-adjustment, 直接用 compute_binary_metrics
             的【原始逐点】口径 (与 TranAD/TriTrackNet 一致, F1 才可横向比较)。
             label=-1 由 compute_binary_metrics → filter_valid_labels 内置过滤,
             此处只把原始 1D labels + oriented scores + threshold 传进去, 不做二次聚合。

    科研标准: 只读 train/val 不读 test, 不执行 optimizer.step,
             不会反向影响训练参数 → 这是无副作用的观测器。
    """
    # [方法2 提速 2026-06-06] SCADA_GRID_FAST=1: 跳过每-epoch 的【train 打分】(昂贵的 collect_energy_and_labels(train_loader))。
    #   train 指标无意义(train_positive=0→F1恒0); train_score 只作 choose_threshold 的 val-无双类回退
    #   (grid 的 val 有正例→不触发)。val 打分/早停/选阈值/最终 test 全不变。默认关→90-run/普通跑【逐位不变】。
    _grid_fast = os.environ.get("SCADA_GRID_FAST") == "1"
    val_score, val_labels = collect_energy_and_labels(solver, solver.vali_loader)
    train_score = train_labels = None
    if not _grid_fast:
        _rng_state = _rng_state_snapshot()
        try:
            train_score, train_labels = collect_energy_and_labels(solver, solver.train_loader)
        finally:
            _restore_rng_state(_rng_state)
    threshold, threshold_source, polarity = choose_threshold_and_polarity_by_validation(
        val_labels, val_score, (train_score if train_score is not None else val_score)
    )
    val_score = orient_scores(val_score, polarity)
    # #C1: 原始逐点口径 — 传 scores+threshold; 内部剔除 -1 + pred=(score>threshold), 不做 point-adjustment。
    val_metrics = compute_binary_metrics(val_labels, scores=val_score, threshold=threshold)
    # 全量化 (2026-05-23): epoch 级 metrics 同样带 farm/module/seed 三元组,确保汇总表完整
    result_dir = result_dir_for_farm(farm, output_dir_override)
    common = {
        "farm": farm,
        "module": module,
        "seed": seed,
        "subset": getattr(solver, "dataset", "SCADA"),
        "n_channels": int(solver.input_c),
        "input_shape": [int(solver.win_size), int(solver.input_c)],
    }
    add_preprocess_identity_to_metrics(
        common,
        identity=getattr(solver, "preprocess_identity", None),
    )
    val_metrics.update(
        {"run_kind": run_kind, "loss": float(val_loss), "threshold": threshold,
         "threshold_source": threshold_source, "score_polarity": polarity, **common}
    )
    if not _grid_fast:
        train_score = orient_scores(train_score, polarity)
        train_metrics = compute_binary_metrics(train_labels, scores=train_score, threshold=threshold)
        train_metrics.update(
            {"run_kind": run_kind, "loss": float(train_loss), "threshold": threshold,
             "threshold_source": threshold_source, "score_polarity": polarity, **common}
        )
        record_and_print_metric(
            result_dir / "metrics.jsonl", CSV_PATH, "AnomalyTransformer", "train", epoch, train_metrics
        )
        _log_wandb_epoch("train", epoch, train_metrics)
    record_and_print_metric(
        result_dir / "metrics.jsonl", CSV_PATH, "AnomalyTransformer", "val", epoch, val_metrics
    )
    _log_wandb_epoch("val", epoch, val_metrics)
    # ============================================================
    # 基线借鉴 #7: 返回 val_f1 供 Solver.train 的早停判据使用
    # 修改原因 (2026-05-26): EarlyStopping 在 metric='val_f1' 模式下需要每 epoch 拿到 F1;
    #     此处把刚算好的 val F1 直接 return, Solver.train 接收并传给 early_stopping。
    # 容错: 若 val 集无正例 (F1=nan), 返回 None 让 Solver 走原 val_loss 路径。
    # ============================================================
    import math as _math
    _val_f1 = val_metrics.get("f1")
    if _val_f1 is None or (isinstance(_val_f1, float) and _math.isnan(_val_f1)):
        return None
    return float(_val_f1)


# ============================================================
# 名称: evaluate_and_record
# 修改原因: 需要在测试阶段补齐 loss/AUC 并写入统一结果文件。
# 作用: 计算阈值、测试指标并输出 METRIC 行。
# 数学原理: 阈值选择 argmax F1_val 或 train quantile;分类指标由混淆矩阵计算。
# 执行流程:
#   1. 收集 train/val/test score;
#   2. 验证集 (含 Evt-1) 选阈值与极性,训练分数做回退;
#   3. 测试集按【原始逐点】口径生成预测 (score>threshold), 不做 point-adjustment;
#   4. 记录最终指标并落盘 scores/labels。
# 科研标准: 测试集不参与阈值选择;ignore(-1) 标签由 实验工具 公共过滤剔除;
#         评测口径与 TranAD/TriTrackNet 一致 (原始逐点), F1 可横向比较。
# ============================================================
def evaluate_and_record(solver: Solver, run_kind: str, module: str = "baseline_only",
                        seed: int = 0, run_id: str = "run_001",
                        output_dir_override: str | None = None,
                        farm: str = "kelmarsh",
                        split_id: str = "narrow_v1",
                        feature_version: str = "v1",
                        preprocess_variant: str = "") -> None:
    r"""
    科研注释：函数 `evaluate_and_record`
    名称作用：在冻结 checkpoint 后完成阈值选择、测试集评价和统一结果落盘。
    Phase D 修改：使用 Val (含 Evt-1) 做 polarity-aware 阈值选择。
    #C1 修改 (2026-06-01)：移除 point-adjustment, 测试集用 compute_binary_metrics 的
    原始逐点口径 (scores+threshold), 与 TranAD/TriTrackNet 一致, F1 可横向比较。
    全量化修改 (2026-05-23)：新增 farm 字段写入每条 metrics 记录,便于按 farm 横向汇总。
    """
    checkpoint = Path(solver.model_save_path) / f"{solver.dataset}_checkpoint.pth"
    solver.model.load_state_dict(torch.load(checkpoint, map_location=solver.device))
    train_score, train_labels = collect_energy_and_labels(solver, solver.train_loader)
    val_score, val_labels = collect_energy_and_labels(solver, solver.vali_loader)
    test_score, test_labels = collect_energy_and_labels(solver, solver.test_loader)
    threshold, threshold_source, polarity = choose_threshold_and_polarity_by_validation(
        val_labels, val_score, train_score
    )
    test_score = orient_scores(test_score, polarity)
    # #C1: 原始逐点口径 — 直接传 scores+threshold, compute_binary_metrics 内部
    #      先剔除 label=-1, 再 pred=(test_score>threshold), 不做 point-adjustment 或二次聚合。
    metrics = compute_binary_metrics(test_labels, scores=test_score, threshold=threshold)
    result_dir = result_dir_for_farm(farm, output_dir_override)
    metrics.update(
        {
            "run_kind": run_kind,
            "loss": float(np.mean(test_score)),
            "threshold": float(threshold),
            "threshold_source": threshold_source,
            "score_polarity": polarity,
            "module": module,
            "seed": seed,
            "farm": farm,   # 全量化新增: 三 farm 横向对比的 join key
            "subset": getattr(solver, "dataset", "SCADA"),
            "n_channels": int(solver.input_c),
            "input_shape": [int(solver.win_size), int(solver.input_c)],
        }
    )
    add_preprocess_identity_to_metrics(
        metrics,
        identity=getattr(solver, "preprocess_identity", None),
    )
    record_and_print_metric(
        result_dir / "metrics.jsonl",
        CSV_PATH,
        "AnomalyTransformer",
        "test",
        "final",
        metrics,
    )
    # ============================================================
    # 基线借鉴 #4: 落盘 train/val/test scores + labels 供 集成评价.py 读取
    # 修改原因 (2026-05-26): 集成评价.py 需要按 (baseline, module, seed) 索引所有 scores;
    # 修改原因 (2026-06-01 Task6): 新增 train scores 落盘; 路径经 ResultLayout 版本化。
    # 文件名约定: {train|val|test}_{scores|labels}__{module}__seed{seed}.npy
    # ============================================================
    from 实验配置 import ResultLayout
    if split_id != "narrow_v1":
        scores_dir = ResultLayout.scores_dir(
            split_id, feature_version, farm, "anomaly_transformer",
            preprocess_variant=preprocess_variant,
        )
    else:
        scores_dir = result_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    val_score_o = orient_scores(val_score, polarity)  # 与 test_score 同向
    train_score_o = orient_scores(train_score, polarity)
    suffix = f"__{module}__seed{seed}"
    np.save(scores_dir / f"train_scores{suffix}.npy", train_score_o.astype(np.float32))
    np.save(scores_dir / f"train_labels{suffix}.npy", np.asarray(train_labels).reshape(-1).astype(np.int8))
    np.save(scores_dir / f"val_scores{suffix}.npy",  val_score_o.astype(np.float32))
    np.save(scores_dir / f"val_labels{suffix}.npy",  np.asarray(val_labels).reshape(-1).astype(np.int8))
    np.save(scores_dir / f"test_scores{suffix}.npy", test_score.astype(np.float32))
    np.save(scores_dir / f"test_labels{suffix}.npy", np.asarray(test_labels).reshape(-1).astype(np.int8))


# ============================================================
# 名称: main
# 修改原因: 提供可由 E:\创新\启动.py 调用的模型实验入口。
# 作用: 处理 resume/force/smoke 逻辑并执行训练与测试。
# 数学原理: 训练目标仍为原 Solver.train 的 reconstruction + KL minimax。
# 执行流程:
#   1. 构造 Solver;
#   2. checkpoint 存在且 resume 时跳过训练;
#   3. 否则执行训练;
#   4. 执行统一测试评价。
# 科研标准: 不修改原模型结构,所有新增逻辑只服务 SCADA 实验记录和复现。
# ============================================================
def main() -> int:
    r"""
    科研注释：函数 `main`
    名称作用：编排 SCADA 专用 Anomaly Transformer 的配置、训练续跑和最终评价。
    Phase D 修改：支持 --module tcn_input_residual 输入级 TCN 增强;使用 polarity-aware 阈值。
    """
    import random
    args = parse_args()
    # Phase D: 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # #6 修复 (2026-05-31): CuDNN determinism for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config = build_config(args)
    run_kind = "smoke" if args.smoke else "formal"
    solver = Solver(vars(config))
    # ============================================================
    # 基线借鉴 #1/#7 注入: AMP + EarlyStopping 判据
    # 修改原因 (2026-05-26): Solver 的 train() 通过 getattr(self, ...) 读这些字段,
    #     默认值兼容旧行为; 这里显式从 实验配置.TrainingProtocol 注入。
    # 作用:
    #   amp_enabled        → 控制 solver.py train() 是否启用 GradScaler/autocast;
    #   early_stop_patience→ EarlyStopping patience (基线借鉴 #7: 3 → 5);
    #   early_stop_metric  → 'val_loss' 或 'val_f1' (基线借鉴 #7: F1 与目标对齐)。
    # ============================================================
    from 实验配置 import TrainingProtocol as _TP
    solver.amp_enabled = bool(_TP.AMP_ENABLED)
    solver.early_stop_patience = int(_TP.EARLY_STOP_PATIENCE)
    solver.early_stop_metric = str(_TP.EARLY_STOP_METRIC)

    # ============================================================
    # 基线借鉴 #6: 加载预训练 ckpt 作为微调起点
    # 修改原因 (2026-05-26): 预训练阶段在跨 farm 60 天纯正常段上训出 ckpt;
    #     微调阶段在 19 天目标切片上从该 ckpt 继续训练 (而非随机初始化)。
    # 兼容性: strict=False 容忍预训练/微调 channel 数差异 (理论上一致, 但留余地)。
    # 科研标准: 预训练数据时间窗 (2020-Q4) 严格不接触目标 farm 的 Val/Test (2021),
    #         不构成数据泄漏。
    # ============================================================
    if args.pretrain_ckpt:
        pretrain_path = Path(args.pretrain_ckpt)
        if pretrain_path.exists():
            state = torch.load(pretrain_path, map_location="cpu")
            missing, unexpected = solver.model.load_state_dict(state, strict=False)
            print(
                f"BORROW#6|AT|pretrain_loaded|path={pretrain_path}|"
                f"missing={len(missing)}|unexpected={len(unexpected)}",
                flush=True,
            )
        else:
            print(f"BORROW#6|AT|pretrain_skip|path_not_exist={pretrain_path}", flush=True)

    # Phase D: TCN 输入包装 (在 optimizer 创建后重新包装)
    # 基线借鉴 #8 (2026-05-26): 新增 tcn_wavelet_residual 分支, 用 TCNWaveletInputWrapper 替换。
    if args.module in ("tcn_input_residual", "tcn_wavelet_residual"):
        dataset_dir = Path(config.data_path)
        # 数据契约 (预处理 S9): 通道数 C 从 train.npy 的 shape[1] 动态获取。
        input_channels = int(np.load(dataset_dir / "train.npy").shape[1])
        # 2026-05-30 TCN-IO: D 一致性闸门(本模型自检 + 跨模型) + 有效感受野覆盖诊断
        import json as _json
        from 实验配置 import TCNIOProtocol as _IO
        _meta_n = None
        _mp = dataset_dir / "meta.json"
        if _mp.exists():
            try:
                _meta_n = int(_json.loads(_mp.read_text(encoding="utf-8")).get("n_channels"))
            except Exception:
                _meta_n = None
        _IO.assert_input_channels("anomaly_transformer", input_channels, _meta_n)
        print(_IO.coverage_banner("anomaly_transformer", input_channels), flush=True)
        if args.module == "tcn_input_residual":
            from modules.tcn_增强 import TCNInputWrapper
            solver.model = TCNInputWrapper(solver.model, input_channels=input_channels).to(solver.device)
        else:  # tcn_wavelet_residual — 基线借鉴 #8
            from modules.tcn_增强 import TCNWaveletInputWrapper
            solver.model = TCNWaveletInputWrapper(solver.model, input_channels=input_channels).to(solver.device)
        # ============================================================
        # 名称: TCN 差分学习率优化器 (2026-05-30 最优超参数)
        # 修改原因: 旧代码传扁平 model.parameters() 给 Adam,TCN 卷积和 α 门控混在 base 组
        #          共享 base lr=1e-4,导致 α 卡在 0.01、TCN 10 epoch 学不动。
        # 作用: 用 TCNProtocol.optimizer_param_groups() 把参数拆为三组独立 lr:
        #          base 模型: lr=1e-4, wd=0 (论文不变)
        #          TCN 卷积+LN: lr=1e-3, wd=0 (10×base,α初值1%节流保护)
        #          α 门控: lr=1e-2, wd=0 (100×base,单标量需高lr快速决断)
        # 数学原理: X'=X+α·TCN(X), α=sigmoid(α_raw)初值0.01;
        #          α 节流了 TCN 输出的有效注入量,也把反向梯度压到 1%;
        #          TCN lr=1e-3 补偿了这道缩放,等效有效 lr≈1e-5 (与 base 持平)。
        # 科研标准: 三模型 TCN 用同一套绝对值 (1e-3/1e-2),共享模块公平对比。
        # ============================================================
        from 实验配置 import TCNProtocol
        solver.optimizer = torch.optim.Adam(
            TCNProtocol.optimizer_param_groups(solver.model, base_lr=config.lr, base_wd=0.0)
        )

    _meta_path = Path(config.data_path) / "meta.json"
    _meta = {}
    if _meta_path.exists():
        try:
            _meta = json.loads(_meta_path.read_text(encoding="utf-8"))
        except Exception:
            _meta = {}
    solver.preprocess_identity = build_preprocess_identity(
        _meta,
        split_id=args.split_id,
        feature_version=args.feature_version,
        preprocess_variant=getattr(args, "preprocess_variant", ""),
        n_channels=int(config.input_c),
    )
    solver.checkpoint_identity = build_checkpoint_identity(
        model="AnomalyTransformer",
        farm=args.farm,
        module=args.module,
        seed=args.seed,
        input_shape=[int(config.win_size), int(config.input_c)],
        preprocess_identity=solver.preprocess_identity,
    )

    # ============================================================
    # 名称: 断点续传逻辑 (2026-05-30 对标 TranAD/TriTrackNet)
    # 修改原因: 原 AT 实验.py 只有"skip_train 整轮跳过"或"从 epoch 0 重训"两种模式,
    #   不支持从断点 epoch 继续。修改后 restore_training_checkpoint 恢复完整训练状态,
    #   若 last_epoch >= epochs-1 则跳过训练; 否则从 last_epoch+1 继续。
    # 作用:
    #   1. --force 删除训练 ckpt (而非最优模型 ckpt) 强制重训, 保持与 TranAD/TriTrackNet 一致;
    #   2. --resume 时先 restore, 已完成的 epoch 不重跑;
    #   3. 训练完成后再加载最优模型 ckpt 做测试评价 (原逻辑, 不变)。
    # 数学原理: 训练目标与原 train() 完全相同; epoch_callback 仍每 epoch 触发。
    # 执行流程:
    #   1. --force → 删除 _training_checkpoint.pth (断点续传 ckpt), 保留 _checkpoint.pth (最优模型);
    #   2. --resume → restore_training_checkpoint 获取 last_epoch;
    #   3. 若 last_epoch >= epochs-1 → 跳过训练 (已全部完成);
    #   4. 否则 → 设 epoch_callback → solver._train(start_epoch=last_epoch+1) 从断点继续。
    # 科研标准: 与 TranAD/TriTrackNet 的 checkpoint 续传模式完全对标,
    #   三模型在中断后都从当前 epoch 重新开始, 不重跑已完成 epoch。
    # ============================================================
    train_ckpt_path = Path(config.model_save_path) / f"{config.dataset}_training_checkpoint.pth"
    if args.force and train_ckpt_path.exists():
        train_ckpt_path.unlink()
        print(f"FORCE|model=AnomalyTransformer|deleted_training_ckpt={train_ckpt_path}|reason=force_retrain", flush=True)
    last_epoch = solver.restore_training_checkpoint(config.model_save_path) if args.resume else -1
    epochs = config.num_epochs

    if train_ckpt_path.exists() and args.resume and not args.force and last_epoch >= epochs - 1:
        print(f"RESUME|model=AnomalyTransformer|checkpoint={train_ckpt_path}|action=skip_train|last_epoch={last_epoch}|total_epochs={epochs}", flush=True)
    else:
        start_epoch = last_epoch + 1 if args.resume and last_epoch >= 0 else 0
        if start_epoch > 0:
            print(f"RESUME|model=AnomalyTransformer|start_epoch={start_epoch}|total_epochs={epochs}|action=continue_from_epoch_{start_epoch}", flush=True)
        # ============================================================
        # 名称: scheduler 注入 (2026-05-31 超参数寻优)
        # 作用: 按 --scheduler 参数构建统一调度器并挂到 solver 上,
        #   _train() 内每 epoch 结束时调用 SchedulerProtocol.step_scheduler()。
        #   替换原硬编码 adjust_learning_rate; 不给 --scheduler 保持旧行为。
        # ============================================================
        solver.scheduler = None
        if args.scheduler is not None:
            from 实验配置 import SchedulerProtocol
            solver.scheduler = SchedulerProtocol.build_scheduler(
                args.scheduler, solver.optimizer,
                T_max=epochs, base_lr=config.lr,
            )
            print(
                f"SCHEDULER|model=AnomalyTransformer|type={args.scheduler}|"
                f"display={SchedulerProtocol.display_name(solver.scheduler)}|"
                f"lr={config.lr}|epochs={epochs}",
                flush=True,
            )
            # 若 resume 且 有旧 scheduler state, restore 后需设 last_epoch
            # (cosine anneal path 依赖 global_step, 这里用 epoch 数近似)
            if start_epoch > 0 and args.scheduler == "cosine":
                for _ in range(start_epoch):
                    solver.scheduler.step()  # 前进到当前 epoch
        solver.epoch_callback = lambda epoch, train_loss, val_loss: record_epoch_metrics(
            solver, epoch, train_loss, val_loss, run_kind,
            farm=args.farm, module=args.module, seed=args.seed, output_dir_override=args.output_dir,
        )
        # wandb 离线模式: 训练前初始化, 训练后清理
        _wandb_ok = init_wandb_run(
            "AnomalyTransformer", args.farm, args.module, args.seed,
            epochs=epochs, batch_size=args.batch_size,
        )
        solver._train(start_epoch=start_epoch)
        if _wandb_ok:
            finish_wandb_run()
    evaluate_and_record(solver, run_kind, module=args.module, seed=args.seed, run_id=args.run_id,
                        output_dir_override=args.output_dir, farm=args.farm,
                        split_id=args.split_id, feature_version=args.feature_version,
                        preprocess_variant=args.preprocess_variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
