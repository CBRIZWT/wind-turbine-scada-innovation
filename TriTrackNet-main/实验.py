# -*- coding: utf-8 -*-
"""
实验.py

名称: TriTrackNet 的 SCADA 专用实验入口
修改原因: 原运行脚本面向公开预测数据集,没有 SCADA 窗口标签、checkpoint 续跑和故障分类指标。
作用: 在 SCADA 温度预测任务上训练 TriTrackNet,并将预测残差转换为故障异常分数。
数学原理:
    1. 预测误差 residual = y - yhat;
    2. MSE = mean(residual^2),MAE = mean(|residual|),RMSE = sqrt(MSE);
    3. 每个未来窗口异常分数 score_i = mean(residual_i^2);
    4. 阈值由验证集 F1 或训练分数分位数确定。
执行流程:
    1. 读取 x/y 预测窗口和 labels_*_window.npy;
    2. 构造 TriTrackNetArchitecture 与 perturbopt;
    3. 训练时逐 epoch 保存 checkpoint 并输出 train/val 指标;
    4. 测试时同时输出预测误差和分类指标。
科研标准: 分类指标是“预测残差派生评价”,不替代原预测指标;测试集不调阈值;ignore 样本剔除。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

from TriTrackNet.TriTrackNet import TriTrackNetArchitecture  # noqa: E402
from TriTrackNet.utils.perturbopt import perturbopt  # noqa: E402
from 实验工具 import (  # noqa: E402
    add_preprocess_identity_to_metrics,
    build_checkpoint_identity,
    build_preprocess_identity,
    checkpoint_identity_is_compatible,
    choose_threshold_and_polarity_by_validation,
    compute_binary_metrics,
    init_wandb_run,
    log_epoch_to_wandb as _log_wandb_epoch,
    finish_wandb_run,
    orient_scores,
    record_and_print_metric,
    regression_metrics,
    dataloader_num_workers,
    should_skip_epoch_eval,
)


# [G3] 死引用清理 (2026-06-01): 移除旧的非-farm 单目录布局常量
#   DATA_DIR = SCRIPT_DIR/dataset/SCADA 与 RESULT_DIR = 实验结果/tritracknet。
#   数据目录唯一真源已改为 实验配置.PerFarmPaths.for_farm(farm)["tritracknet"]
#   (见 data_dir_for_farm); 结果目录由 result_dir_for_farm(farm) 决定。两常量无任何调用方。
CSV_PATH = ROOT / "实验结果" / "metrics.csv"


def _rng_state_snapshot():
    """Capture RNG state so metric-only train probes cannot perturb later epochs."""
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
# 名称: data_dir_for_farm
# 修改原因 (2026-06-01 数据契约统一): 原先本函数自行拼 SCRIPT_DIR/dataset/SCADA_{farm},
#          与 实验配置.PerFarmPaths 各存一份路径模板, 存在漂移风险。
#          改为唯一真源 = 实验配置.PerFarmPaths.for_farm(farm)["tritracknet"],
#          形如 TriTrackNet-main/dataset/SCADA_{farm}/, 与预处理落盘目录一一对应。
#          (2026-06-02 增补 split/feature 隔离) 旧调用不传 split_id/feature_version,
#          全量 chronological_v2 会读到历史 narrow_v1 目录 (读错数据); 现透传版本参数。
# 作用: 给定 farm + split_id/feature_version 返回该 farm 对应的 dataset/ 子目录
#       (含契约文件 train/val/test.npy + {split}_labels.npy + meta.json);
#       narrow_v1 (默认) 解析为旧 SCADA_{farm}/, 其它 split 解析为版本化目录。
# 数学原理: 无 (路径计算)。
# 执行流程: 委托 PerFarmPaths.for_farm(farm, split_id, feature_version) 取 "tritracknet" 键。
# 科研标准: 路径含 farm + 版本段, 与 metrics 行的 farm 字段一一对应; 路径模板与三 baseline 同源于配置;
#          narrow_v1 与 chronological_v2 物理隔离, 复跑互不污染。
# ============================================================
def data_dir_for_farm(farm: str, split_id: str = "narrow_v1", feature_version: str = "v1",
                      preprocess_variant: str | None = None) -> Path:
    from 实验配置 import PerFarmPaths  # noqa: E402
    return Path(PerFarmPaths.for_farm(
        farm, split_id, feature_version, preprocess_variant=preprocess_variant,
    )["tritracknet"])


# ============================================================
# 名称: result_dir_for_farm
# 修改原因 (2026-05-25 全量化): metrics.jsonl 之前合并写在 实验结果/tritracknet/,
#          多 farm 同时跑会乱序;需要按 farm 隔离。
# 作用: 给定 farm 名 + 可选 override 返回 metrics.jsonl 写入目录。
# 数学原理: 无 (路径计算)。
# 执行流程:
#   1. output_dir_override 非空时直接用 (供 启动.py 显式指定);
#   2. 否则 → 实验结果/{farm}/tritracknet/。
# 科研标准: 三 baseline (AT/TranAD/TriTrack) 在同一 farm 下并列写 jsonl,横向对比直接 glob。
# ============================================================
def result_dir_for_farm(farm: str, output_dir_override: str | None = None) -> Path:
    if output_dir_override:
        return Path(output_dir_override)
    return ROOT / "实验结果" / farm / "tritracknet"


# ============================================================
# 名称: checkpoint_path
# 修改原因: smoke 小样本训练产生的参数不得在正式预测实验中续用。
#   (2026-06-02 增补 split/feature 隔离) checkpoint 旧路径不含版本段,
#   全量 chronological_v2 与历史 narrow_v1 的 checkpoint 会写同一路径互相覆盖。
# 作用: 为 smoke/formal 返回独立 checkpoint 文件; 仅在 split_id != narrow_v1 时在 farm 段后
#   插入 f"{split_id}__{feature_version}" 段, narrow_v1 保持旧路径字节级一致。
# 数学原理: 无。
# 执行流程: 根据 smoke 布尔值选择子目录; 非 narrow_v1 在 farm 后插版本段; 返回 Path。
# 科研标准: 链路验证结果与正式研究结果严格隔离;
#   narrow_v1 路径与历史一致 (既有测试依赖), chronological_v2 与其物理隔离。
# ============================================================
def checkpoint_path(smoke: bool, module: str = "baseline_only", seed: int = 0, farm: str = "kelmarsh",
                    split_id: str = "narrow_v1", feature_version: str = "v1",
                    preprocess_variant: str | None = None) -> Path:
    kind = "smoke" if smoke else "formal"
    ckpt_root = SCRIPT_DIR / "checkpoints" / "SCADA" / farm
    if split_id != "narrow_v1":
        tag = f"{split_id}__{feature_version}"
        variant = str(preprocess_variant or "").strip()
        if variant:
            tag = f"{tag}__{variant}"
        ckpt_root = ckpt_root / tag
    return ckpt_root / kind / module / f"seed{seed}" / "tritracknet_checkpoint.pt"


# ============================================================
# 名称: parse_args
# 修改原因: 需要统一启动器可选择 smoke/resume/force,并显式记录预测超参数。
# 作用: 定义 TriTrackNet SCADA 实验命令行参数。
# 数学原理: 无。
# 执行流程: argparse 解析参数。
# 科研标准: 训练轮数、学习率、batch size 和随机种子都可复查。
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TriTrackNet SCADA 实验入口")
    # 最优设置: epochs=10 (TrainingProtocol.EPOCHS, 统一训练预算);
    #     PerturbOpt 收敛需要足够步数, 10 epoch 配合差分 lr 让 TCN α 充分决断.
    parser.add_argument("--epochs", type=int, default=10, help="正式训练 epoch 数 (TrainingProtocol.EPOCHS=10)")
    # 最优设置: batch_size=128 (TrainingProtocol.BATCH_SIZE, 统一比较基线);
    #     TriTrackNet PerturbOpt 双 forward 使显存翻倍, AMP 与 GradScaler 冲突故关;
    #     128 匹配 RTX 5090 24GB 显存约束.
    parser.add_argument("--batch-size", type=int, default=128, help="batch size (TrainingProtocol.BATCH_SIZE=128, fp32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--rho", type=float, default=0.5, help="PerturbOpt 扰动半径")
    # 基线借鉴 #8: 新增 tcn_wavelet_residual 选项
    parser.add_argument("--module", type=str, default="baseline_only",
                        choices=["baseline_only", "tcn_input_residual", "tcn_wavelet_residual"],
                        help="Phase D 模块选择 (基线借鉴 #8: 新增 tcn_wavelet_residual)")
    # 最优设置: seed 通过 启动.py --seeds 0,1,2,3,4 批量注入 (TrainingProtocol.SEEDS).
    parser.add_argument("--seed", type=int, default=0, help="随机种子 (启动器注入 0-4; 单跑默认 0)")
    parser.add_argument("--run-id", type=str, default="run_001", help="运行标识")
    parser.add_argument("--output-dir", type=str, default=None, help="覆盖默认输出目录")
    parser.add_argument("--resume", action="store_true", help="恢复 checkpoint")
    parser.add_argument("--force", action="store_true", help="删除 checkpoint 重跑")
    parser.add_argument("--smoke", action="store_true", help="小样本一轮链路验证")
    parser.add_argument("--pretrain", action="store_true",
                        help="基线借鉴 #6: 在 SCADA_pretrain 正常段上训练 baseline-only 预训练 ckpt")
    parser.add_argument("--farm", type=str, default="kelmarsh",
                        choices=["kelmarsh", "penmanshiel", "hill_of_towie"],
                        help="全量化实验 farm 名 (默认 kelmarsh)")
    parser.add_argument("--split-id", type=str, default="narrow_v1", help="切分协议 id (全量=chronological_v2)")
    parser.add_argument("--feature-version", type=str, default="v1", help="特征版本 (全量=v2)")
    parser.add_argument("--preprocess-variant", type=str, default="",
                        help="预处理变体后缀, 例如 old_preprocess / new_preprocess")
    # 基线借鉴 #6: 预训练 ckpt 路径
    parser.add_argument("--pretrain-ckpt", type=str, default=None,
                        help="基线借鉴 #6: 预训练 checkpoint 路径 (微调时加载)")
    # ============================================================
    # 名称: --scheduler (2026-05-31 超参数寻优)
    # 修改原因: TriTrackNet 原先完全无 scheduler, 所有 epoch 用相同 lr。
    #   新增 --scheduler CLI 参数 + 训练循环末尾 scheduler.step(val_loss),
    #   支持 CosineAnnealingLR / ReduceLROnPlateau / StepLR。
    # 兼容: 不给 --scheduler 时保持旧行为 (无 scheduler, lr 不变)。
    # ============================================================
    parser.add_argument("--scheduler", type=str, default=None,
                        choices=["cosine", "plateau", "steplr"],
                        help="学习率调度器类型 (网格搜索注入; 默认 None=无 scheduler)")
    return parser.parse_args()


# ============================================================
# 名称: set_seed
# 修改原因: 深度模型训练存在随机初始化与 DataLoader 洗牌,不固定随机种子无法复核。
# 作用: 同时设置 Python、numpy、torch 和 CUDA 随机种子。
# 数学原理: 固定伪随机序列初始状态。
# 执行流程: 顺序调用 random/np/torch seed API。
# 科研标准: 每次实验日志记录同一 seed;GPU 仍可能存在非确定性算子,不得过度声明完全一致。
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 名称: load_arrays (farm-aware, 模型内部切窗)
# 修改原因 (2026-06-01 [E1] 文件名契约对齐): 预处理落盘的统一数据契约为
#          【逐点序列 {train,val,test}.npy (T,C) float32 + 逐点标签 {train,val,test}_labels.npy (T,) int64∈{-1,0,1}】,
#          原代码读的是 series_{split}.npy / labels_{split}.npy 旧布局, 与契约不符 → 全部改为契约文件名。
#          切窗 (L=SEQ_LEN / H=PRED_LEN) 在本函数 load 时由 实验配置.SlidingWindow 现切,
#          与 AT/TranAD "模型内部切窗" 一致; 窗口大小唯一真源 = DatasetProtocol。
# 作用: 给定 farm + smoke, 读契约的逐点 series/labels → 现切预测窗 → 返回 x/y/label 字典 + meta。
# 数学原理:
#   x ∈ R^{N×C×L} (channel-first 历史窗), y ∈ R^{N×(C·H)} (展平的未来 H 步),
#   label ∈ {-1,0,1}^N (未来 H 步内任一正例→1, 否则任一 ignore(-1)→-1, 否则 0)。
#   C 从 series.shape[1] 动态读 (各 farm 不同); 预测样本数 N = T - L - H + 1
#   (比原逐点序列少 L+H-1 个, 因首样本需 L 步历史、末样本需 H 步未来)。
# 执行流程:
#   1. data_dir = data_dir_for_farm(farm) (= PerFarmPaths 契约目录);
#   2. 必须存在 meta.json (否则 FileNotFoundError, 提示先跑预处理);
#   3. 读契约的 {train,val,test}.npy (T,C) + {train,val,test}_labels.npy (T,);
#      [D5] 若 meta.n_channels < 6 打印 warning (DualChannelAttention 三等分覆盖度下降);
#   4. SlidingWindow 现切预测窗 + 现算窗口标签; smoke 裁前 512 个窗口;
#   5. 返回字典 (x/y 与对应窗口标签逐元素时间对齐, 同长)。
# 科研标准: 不重新归一化、不重切时间段 (沿用预处理固定时间切分); 模型侧只切窗不重洗牌;
#          TCN 闸门校验 L ≥ 感受野; 标签保留 {-1,0,1} 三态, 评测时由 实验工具内部排除 -1。
# ============================================================
def load_arrays(smoke: bool, farm: str, pretrain: bool = False,
                split_id: str = "narrow_v1", feature_version: str = "v1",
                preprocess_variant: str | None = None) -> Dict[str, np.ndarray]:
    if pretrain:
        raise NotImplementedError(
            "PretrainProtocol 已移除 (2026-05-31). 使用 --pretrain-ckpt 加载预训练权重代替."
        )
    # 2026-06-02 split/feature 版本隔离: 经 data_dir_for_farm 透传版本参数,
    #   narrow_v1 (默认) 读旧 SCADA_{farm}/, 其它 split 读版本化目录。
    data_dir = data_dir_for_farm(farm, split_id, feature_version, preprocess_variant)
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"缺少 {meta_path}; 请先运行数据预处理 --mode full --farm {farm}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # [D5] 小通道数退化文档化 (2026-06-01, 不改核心前向逻辑):
    #   TriTrackNet/TriTrackNet.py 的 DualChannelAttention.forward 按 c=max(1,C//3) 把通道三等分
    #   (主注意力 / 反向注意力 / HFF-MLP)。数据驱动的小 C (如 4) 时, c=1 → 三个支路各只覆盖 1 个
    #   通道, 第 3 段 x[:,2c:] 还可能为空被零占位, 三通道交互"覆盖度"显著下降 (退化为近单通道处理)。
    #   仅告警, 不改前向: 让小 C farm 的结果可被正确解读, 而非静默给出退化指标。
    n_channels_meta = int(meta.get("n_channels", 0) or 0)
    if 0 < n_channels_meta < 6:
        print(
            f"WARNING|TriTrackNet|D5_small_channel|n_channels={n_channels_meta}<6|farm={farm}|"
            f"DualChannelAttention 按 C//3 三等分, 小通道数下三路覆盖度下降 (c={max(1, n_channels_meta // 3)}), "
            f"三通道交互近似退化为单通道; 指标解读需注意。",
            flush=True,
        )
    # 2026-06-01 [E1]: 预处理只给契约的【逐点序列 {train,val,test}.npy (T,C) +
    #   逐点标签 {train,val,test}_labels.npy (T,)】; 切窗在此 load 时由 实验配置.SlidingWindow
    #   现切 (L=SEQ_LEN, H=PRED_LEN), 与 AT/TranAD "模型内部切窗" 完全一致 (窗口大小是模型超参, 非数据属性)。
    from 实验配置 import DatasetProtocol, SlidingWindow  # noqa: E402
    L = int(DatasetProtocol.SEQ_LEN)
    H = int(DatasetProtocol.PRED_LEN)
    # TCN 闸门: 送入 TCN 增强的就是历史窗 L, 必须 ≥ TCN 感受野 (否则最深层落入零填充)。
    DatasetProtocol.assert_window_covers_tcn(L, "tritracknet")

    def _load_and_window(split: str):
        # [E1] 契约文件名: 序列 = {split}.npy (T,C) float32; 标签 = {split}_labels.npy (T,) int64∈{-1,0,1}。
        # 通道数 C 从 series.shape[1] 动态决定 (SlidingWindow.for_forecast 内部取 X.shape[1]), 各 farm 不同。
        series = np.load(data_dir / f"{split}.npy")                      # (T, C) 逐点序列
        labels = np.load(data_dir / f"{split}_labels.npy")              # (T,) 逐点标签 {-1,0,1}
        # 现切预测窗: x=(N,C,L), y=(N,C*H), 窗口标签=(N,); N = T - L - H + 1 (stride=1)。
        # 对齐方式: 三者均由同一 (L,H,stride) 在【同一逐点序列/标签】上滑窗生成, 故第 i 行严格时间对应——
        #   x_i = series[i : i+L]ᵀ, y_i = series[i+L : i+L+H]ᵀ, lab_i = labels[i+L : i+L+H] 的聚合标签;
        #   因此 score_i (来自 y_i vs ŷ_i 的残差) 与 lab_i 一一对应、等长 (见下方评测一致性注释)。
        # [P1-OOM修复 2026-06-10] 旧 SlidingWindow.for_forecast 以 np.stack 物化全部 x 窗口
        #   (kelmarsh train/val/test x 共 ~13GB) + 下游 torch.FloatTensor(x) 再整体拷贝一次
        #   → ArrayMemoryError (4.03GiB, (626462,18,96))。改 sliding_window_view【零拷贝视图】:
        #   x_view[i,c,j] = series[i+j,c] ≡ 旧 xs[i][c,j], 逐位一致; 单窗物化推迟到
        #   _LazyWindowDataset.__getitem__ (每窗 (C,L)≈7KB)。y 仍一次性物化 (N,C*H)
        #   (评分 regression_metrics 需整块 y, ~1GB 可承受); reshape 在非连续视图上自动复制,
        #   元素顺序与旧 series[i+L:i+L+H].T.reshape(-1) 相同。
        from numpy.lib.stride_tricks import sliding_window_view
        series = np.ascontiguousarray(series, dtype=np.float32)
        n = len(series) - L - H + 1
        C_ = int(series.shape[1])
        if n <= 0:
            x = np.empty((0, C_, L), dtype=np.float32)
            y = np.empty((0, C_ * H), dtype=np.float32)
        else:
            x = sliding_window_view(series, L, axis=0)[:n]                       # (N,C,L) 视图
            y = sliding_window_view(series[L:], H, axis=0)[:n].reshape(n, C_ * H)  # (N,C*H) 单次物化
        lab = SlidingWindow.for_forecast_labels(labels, L, H, stride=1)  # (N,) 窗口标签 (与 x/y 同 N)
        return x, y, lab

    x_tr, y_tr, lab_tr = _load_and_window("train")
    x_va, y_va, lab_va = _load_and_window("val")
    x_te, y_te, lab_te = _load_and_window("test")
    arrays = {
        "x_train": x_tr, "y_train": y_tr,
        "x_val": x_va, "y_val": y_va,
        "x_test": x_te, "y_test": y_te,
        "label_train": lab_tr, "label_val": lab_va, "label_test": lab_te,
        "_meta": meta,
    }
    if smoke:
        return {key: (value[:512] if isinstance(value, np.ndarray) else value) for key, value in arrays.items()}
    return arrays


# ============================================================
# 名称: build_model_and_optimizer
# 修改原因: 原 TriTrackNet.fit 不暴露 optimizer 状态,无法实现严格 checkpoint 续跑。
# 作用: 以相同核心网络和 PerturbOpt 创建可保存状态的训练对象。
# 数学原理: 预测 horizon H = y_dim / C,网络输出维度为 C·H。
# 执行流程:
#   1. 从输入 shape 推断 C/L/H;
#   2. 创建 TriTrackNetArchitecture;
#   3. 创建 SmoothL1Loss 和 perturbopt;
#   4. 返回三者。
# 科研标准: 不改核心网络源码,优化器类型与原 TriTrackNet.fit 保持一致。
# ============================================================
def build_model_and_optimizer(arrays: Dict[str, np.ndarray], args: argparse.Namespace, device: torch.device):
    channels = arrays["x_train"].shape[1]
    seq_len = arrays["x_train"].shape[2]
    horizon = arrays["y_train"].shape[1] // channels
    # ============================================================
    # 名称: use_revin 强制开 + 显式 banner (基线借鉴 #5)
    # 修改原因 (2026-05-26 借鉴自基线): 基线 baseline_suite/models/tritracknet.py L25
    #         显式 use_revin=True 应对 Train(1月) 与 Test(3月) 季节性温度漂移
    #         (Kelmarsh 环境温度均值差 5-10°C, zscore 后仍漂移)。
    #         创新原来虽默认 use_revin=True, 但无校验, 一旦未来误改默认值会静默失效。
    # 作用: 强制 use_revin=True, 启动时打印一条 banner 提示当前 RevIN 状态,
    #       任何企图改默认值的 PR 都会在 IDE 命令行第一时间被发现。
    # 数学原理:
    #   RevIN 在 forward 入口: x' = (x - μ_w) / σ_w  (按窗口 instance norm)
    #          在 forward 出口: y = y' * σ_w + μ_w  (反归一化恢复原尺度)
    #   等效于"per-window 重新标准化", 对季节性漂移天然鲁棒。
    # 科研标准: 强制开 RevIN 是设计决策, 不再作为可调超参; 改回 False 必须显式改 banner 文案。
    # ============================================================
    use_revin = True  # 基线借鉴 #5: 硬编码 True, 不接受 args 覆盖
    print(
        f"REVIN|enabled={use_revin}|reason=baseline_borrow_5|seq_len={seq_len}|channels={channels}",
        flush=True,
    )
    model = TriTrackNetArchitecture(
        num_channels=channels,
        seq_len=seq_len,
        hid_dim=16,
        pred_horizon=horizon,
        use_revin=use_revin,
        use_aux=True,
    ).to(device)
    optimizer = perturbopt(
        model.parameters(),
        base_optimizer=torch.optim.AdamW,
        rho=args.rho,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.SmoothL1Loss()
    return model, optimizer, criterion


# ============================================================
# 名称: train_one_epoch
# 修改原因: 需要在外置入口中逐 epoch 输出和保存可续跑状态。
# 作用: 按原 fit 的 PerturbOpt 两步更新执行一轮训练。
# 数学原理: 第一次梯度得到扰动 w+e(w),第二次在扰动邻域求梯度后回到 w 更新。
# 执行流程:
#   1. DataLoader 随机批量取训练窗口;
#   2. 正常前向和反向,调用 first_step;
#   3. 再次前向反向,调用 second_step;
#   4. 返回平均 SmoothL1 loss。
# 科研标准: 每轮只在训练集更新参数,验证/测试不反向传播。
# ============================================================
def train_one_epoch(
    model: nn.Module,
    optimizer,
    criterion: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    # ============================================================
    # 名称: DataLoader pin_memory + num_workers (基线借鉴 #9)
    # 修改原因 (2026-05-26 借鉴自基线): 基线 baseline_suite/trainer.py L107-116
    #         统一 pin_memory + persistent_workers, 数据传输不阻塞 GPU;
    #         创新原代码 DataLoader 没设这些, GPU 等数据时空转。
    # 作用: CUDA 时开 pin_memory (锁页内存 → DMA 直接传 GPU); .to(device, non_blocking=True)
    #       让数据传输与计算重叠。
    # 数学原理: 无 (I/O 优化), 但 GPU 利用率↑ → 同 epoch 内更新更多 batch, 等价于训得更稳。
    # 科研标准: num_workers=0 时 persistent_workers 必须 False, Windows 多进程 pickle 不稳。
    # ============================================================
    pin = (device.type == "cuda")
    # 注 (2026-05-31): 每 epoch 重建 DataLoader 会造成 ~400MB 张量拷贝。
    # 优化方向: 将 TensorDataset 移到 epoch 循环外创建一次。
    # ============================================================
    # 名称: num_workers — Windows spawn 实测回退 0 (2026-06-03 训练加速)
    # 修改原因: 曾试 num_workers=4 + persistent_workers 想让取批与 GPU 计算重叠, 但本机
    #   (E:\ancoda\chuangxin python, Windows spawn) 冒烟即崩:
    #   每个 worker 子进程 spawn 时 re-import torch 重新加载全部 CUDA DLL,
    #   4 worker 同时提交 → OSError [WinError 1455] 页面文件太小 → worker exited unexpectedly。
    #   数据本就是已物化内存 TensorDataset (worker 只切片, 收益小), 故按既定回退策略保持 0。
    # 结论: Windows 上保持 num_workers=0; 提速主要来自 bf16 autocast + 单 epoch 一次同步。
    #   (与 TrainingProtocol.DATALOADER_NUM_WORKERS=0 一致)
    # ============================================================
    # 提速 (2026-06-18): _LazyWindowDataset 仅按批物化 (共享映射小) → 可安全开 worker 并行喂数据,
    #   GPU 不再饿等 (实测 GPU 仅 40%, CPU 切窗瓶颈)。num_workers 由 SCADA_NUM_WORKERS 控制 (默认0)。
    #   惰性数据集是关键: 大物化 TensorDataset 才会撑爆 commit (error 1455), 本数据集小 → 不撑爆。
    _nw = dataloader_num_workers()
    loader = DataLoader(
        _LazyWindowDataset(x, y),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,                # 基线借鉴 #9
        num_workers=_nw,
        persistent_workers=(_nw > 0),  # 跨 epoch 复用 worker, 省重复 spawn/re-import torch
    )
    model.train()
    # ============================================================
    # 名称: AMP 混合精度 (基线借鉴 #1)
    # 修改原因 (2026-05-26 借鉴自基线): 基线 anomaly_transformer.py L43,65,83-86
    #         用 torch.amp.autocast + GradScaler, batch=1024 训练稳定;
    #         创新原代码 fp32 训练, batch=128 在 SCADA 小数据集 (1.6 万行 / 128 ≈ 128 batch)
    #         上 seed 间方差大, 梯度噪声大。
    # 作用: TriTrackNet (浮点) 开 AMP, 让 batch 可扩到 512 而显存与原 batch=128 fp32 相当;
    #       梯度估计更稳定, 5 seed mean±std 更收敛。
    # 数学原理: fp16 前向 + fp32 主参数 + loss scaling 防 underflow;
    #         Var(g̃_B) ∝ σ²/B, B↑ → 梯度方差↓。
    # 科研标准: PerturbOpt 两阶段更新都包在 autocast 内; scaler 只对反向缩放, optimizer step 不变。
    # ============================================================
    # ============================================================
    # 名称: TriTrackNet PerturbOpt + AMP bf16 (2026-06-03 训练加速)
    # 修改原因: 旧实现因 fp16 GradScaler 与 PerturbOpt 两次 backward 冲突而完全关 AMP
    #   (RuntimeError: unscale_() has already been called ...), 模型一直 fp32 训练,
    #   GPU 仅 ~23% 利用。改用 *bf16* autocast: bf16 动态范围与 fp32 同 (8 位指数),
    #   不会 underflow, 因此【不需要 GradScaler】——直接 loss.backward() 即得真实梯度,
    #   PerturbOpt.first_step/second_step 读到的就是真实梯度幅度, 状态机零冲突。
    # 作用: 仅前向 + loss 计算在 bf16 autocast 内; 两次 backward 与 first_step/second_step
    #   保持在 autocast 外, 优化器 step 全程 fp32 主参数。吞吐 ~3x, 优化目标不变。
    # 数学原理: bf16 前向是 fp32 的低精度近似 (相对误差 ~2^-8), 期望梯度不变;
    #   PerturbOpt 两段真实梯度 (e(w) 与 base_optimizer.step()) 语义不变。
    # 科研标准: 不改 SmoothL1 损失公式、不改 PerturbOpt 两段更新、不改窗口/批大小;
    #   bf16 无 GradScaler 故不触碰旧 unscale_ 冲突点。NaN 监测见下方 epoch 汇报。
    # ============================================================
    use_amp = (device.type == "cuda")
    autocast_ctx = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)
    # 提速#2: 在 GPU 上累计 loss, 每 epoch 仅一次 .item() 同步 (原每 batch .item() 强制 GPU→CPU 同步,
    #   阻塞流水线)。loss_sum/n_batches 数学等价于原 np.mean(losses)。
    loss_sum = torch.zeros((), device=device)
    n_batches = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)  # 基线借鉴 #9
        yb = yb.to(device, non_blocking=True)
        # ---- PerturbOpt 第一步: 在当前 w 计算梯度 ----
        with autocast_ctx():
            pred = model(xb, domain_knowledge=None)
            loss = criterion(pred, yb)
        loss.backward()  # bf16 无需 GradScaler, 梯度即真实幅度
        optimizer.first_step(zero_grad=True)
        # ---- PerturbOpt 第二步: 在扰动 w+e(w) 计算梯度并回退到 w ----
        with autocast_ctx():
            pred = model(xb, domain_knowledge=None)
            loss = criterion(pred, yb)
        loss.backward()
        optimizer.second_step(zero_grad=True)
        # 提速#2: 累计在 GPU, 不同步 (loss.detach() 避免保留计算图)
        loss_sum += loss.detach()
        n_batches += 1
    # 每 epoch 一次同步: 数学等价 np.mean(per-batch losses)
    return float((loss_sum / max(n_batches, 1)).item())


class _LazyWindowDataset(torch.utils.data.Dataset):
    """惰性窗口数据集 (P1-OOM修复 2026-06-10)。

    持有 sliding_window_view 的零拷贝视图, __getitem__ 才物化【单个】窗口 (C,L)≈7KB,
    取代 TensorDataset(torch.FloatTensor(x)) 对 (N,C,L) 的整体 4–7GB 拷贝。
    返回签名与原 TensorDataset 一致: 给 y → (x_i, y_i); 不给 → (x_i,)。collate 行为不变。
    """

    def __init__(self, x: np.ndarray, y: np.ndarray | None = None):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i):
        xi = torch.from_numpy(np.ascontiguousarray(self.x[i], dtype=np.float32))
        if self.y is None:
            return (xi,)
        return xi, torch.from_numpy(np.ascontiguousarray(self.y[i], dtype=np.float32))


# ============================================================
# 名称: predict
# 修改原因: 评价阶段需要按批次推理以控制 GPU/CPU 内存。
# 作用: 输出某一数据段的预测矩阵。
# 数学原理: yhat = f_theta(x)。
# 执行流程:
#   1. 创建顺序 DataLoader;
#   2. torch.no_grad 前向;
#   3. 拼接并返回 numpy。
# 科研标准: 验证与测试不打乱顺序、不更新参数。
# ============================================================
def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    # 基线借鉴 #9: 推理也开 pin_memory + non_blocking
    pin = (device.type == "cuda")
    loader = DataLoader(
        _LazyWindowDataset(x),         # [P1-OOM修复] 视图按批物化, 免 (N,C,L) 整体拷贝
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin,
        num_workers=0,
    )
    predictions = []
    model.eval()
    # ============================================================
    # 名称: predict AMP bf16 + inference_mode (2026-06-03 训练加速)
    # 修改原因: predict 每 epoch 在 train/val (及末次 test) 上跑前向, 占总耗时一大块。
    #   原 torch.no_grad()+fp32。改 torch.inference_mode() (比 no_grad 更省: 关 version
    #   counter / view 追踪) + bf16 autocast (前向 ~2x)。
    # 数学/科研: bf16 仅前向近似, 残差/分数/阈值/HI/泄漏逻辑全不变;
    #   bf16 无 numpy dtype, 故 .float() 升回 fp32 再 .cpu().numpy() (与原 fp32 输出同 dtype)。
    # ============================================================
    use_amp = (device.type == "cuda")
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            for (xb,) in loader:
                out = model(xb.to(device, non_blocking=True), domain_knowledge=None)
                # bf16 → fp32 (numpy 无 bf16); 与原 fp32 路径数值对齐
                predictions.append(out.float().detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


# ============================================================
# 名称: predict_and_score
# 修改原因 (系统崩溃修复 2026-06-20): Hill 农场 (N=5,670,105×53) 的 predict 先物化整张
#   (N, 24×53) 预测张量 ≈ 28.8GB, 紧接 regression_metrics 时与 y(28.8GB) 同驻 ≈ 57GB,
#   超出 commit 上限 73.4GB (RAM 63.4 + pagefile 10) → 被 OS 强杀 (System 事件 2004,
#   每-run 日志 0 字节、无 Python traceback)。kel/pen 仅 8.0/6.3GB 故能通过。
# 作用: 把 predict 的逐批前向与 regression_metrics 的逐行约简【融合】, 单批预测用后即弃,
#   峰值降到单批 (~MB), 与农场规模无关。
# 数学/科研: score_i = mean(max(0, y_i-ŷ_i)²) 为逐行 (axis=1) 约简 → 分批与整块【逐位一致】
#   (喂阈值/极性/最终 F1/AUC, 科研结果不变); mse/mae 跨批按元素数加权聚合 → 数值等价
#   (~1e-12, 仅日志)。返回签名同 regression_metrics(y, predict(model,x,...))。
# ============================================================
def predict_and_score(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int,
                      device: torch.device) -> Tuple[Dict[str, float], np.ndarray]:
    pin = (device.type == "cuda")
    loader = DataLoader(
        _LazyWindowDataset(x),         # 视图按批物化, 免 (N,C,L) 整体拷贝
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin,
        num_workers=0,
    )
    target = np.asarray(y)
    n = int(target.shape[0])
    cols = int(np.prod(target.shape[1:])) if target.ndim > 1 else 1
    sample_scores = np.empty(n, dtype=np.float64)
    sq_sum = 0.0
    abs_sum = 0.0
    elem_count = 0
    model.eval()
    use_amp = (device.type == "cuda")   # 与 predict 同: CPU 关 amp, CUDA 开 bf16 前向
    start = 0
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            for (xb,) in loader:
                out = model(xb.to(device, non_blocking=True), domain_knowledge=None)
                pred_b = out.float().detach().cpu().numpy()   # 单批 (b, cols), 用后即弃
                stop = start + pred_b.shape[0]
                # 复用 regression_metrics 逐行打分(零数学重复): 本批 score 与整块逐位一致
                reg_b, score_b = regression_metrics(target[start:stop], pred_b)
                sample_scores[start:stop] = score_b
                elem_b = (stop - start) * cols
                sq_sum += reg_b["mse"] * elem_b   # mse_b = Σresidual²/elem_b → 还原本批平方和
                abs_sum += reg_b["mae"] * elem_b
                elem_count += elem_b
                start = stop
    if elem_count == 0:
        mse = mae = rmse = float("nan")
    else:
        mse = sq_sum / elem_count
        mae = abs_sum / elem_count
        rmse = float(np.sqrt(mse))
    return {"loss": mse, "mse": mse, "mae": mae, "rmse": rmse}, sample_scores


# ============================================================
# 名称: save_checkpoint / restore_checkpoint
# 修改原因: 长时间预测实验需要支持模型 checkpoint 续跑。
# 作用: 保存/加载网络、PerturbOpt 状态和完成 epoch。
# 数学原理: 无;保存优化过程状态。
# 执行流程: torch.save 写入字典;torch.load 恢复字典。
# 科研标准: checkpoint 中记录配置,避免续跑时超参数无法核查。
# ============================================================
def save_checkpoint(model, optimizer, epoch: int, args: argparse.Namespace, checkpoint: Path,
                    scheduler=None, checkpoint_identity: Dict[str, object] | None = None) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
        "checkpoint_identity": checkpoint_identity,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, checkpoint)


def restore_checkpoint(model, optimizer, device: torch.device, checkpoint: Path,
                       scheduler=None, expected_identity: Dict[str, object] | None = None) -> int:
    if not checkpoint.exists():
        return -1
    payload = torch.load(checkpoint, map_location=device)
    if not checkpoint_identity_is_compatible(
        payload,
        expected_identity,
        model="TriTrackNet",
        checkpoint_path=checkpoint,
    ):
        return -1
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return int(payload["epoch"])


# ============================================================
# 名称: score_to_ewma_hi
# 作用: 将逐窗口预测残差分数经 EWMA 平滑后转为健康指标 HI(t) ∈ [0,1]。
# 数学原理:
#   1. EWMA(α=0.05) → 半衰期 ≈ 3.3h, 匹配轴承热惯量时间常数;
#   2. 跨通道取 max (保守策略: 最脆弱轴承决定告警);
#   3. HI(t) = exp(−α₁ × ewma_max), α₁ 在 train 正常段拟合使 HI≈1。
# 执行流程: score数组 → EWMA → max_channel → HI
# 科研标准: 不参与训练, 纯后处理; 阈值 HI_warn=0.7 经验值。
# ============================================================
def score_to_ewma_hi(scores: np.ndarray, alpha: float = 0.05, hi_alpha: float | None = None) -> tuple:
    """
    Args:
        scores: (T,) or (T, C) 预测残差 MSE（逐窗口或逐窗口×通道）
        alpha: EWMA 衰减因子，默认 0.05 → 半衰期 ≈ 13.5 步 ≈ 2.25h
        hi_alpha: 负指数系数; None 则自动拟合使正常段 HI 均值 ≈ 0.99
    Returns:
        ewma: (T,) EWMA 平滑后序列
        hi:   (T,) 健康指标 ∈ [0,1]
        hi_alpha: float 实际使用的负指数系数 (传入值, 或在本序列正常段拟合的值)
    """
    arr = np.asarray(scores, dtype=float)
    if arr.ndim == 2:
        arr = arr.max(axis=1)
    ewma_seq = arr.reshape(-1).copy()
    for t in range(1, len(ewma_seq)):
        ewma_seq[t] = alpha * ewma_seq[t] + (1 - alpha) * ewma_seq[t - 1]
    if hi_alpha is None:
        median_ref = float(np.median(ewma_seq))
        mask_normal = ewma_seq <= median_ref
        if mask_normal.sum() > 0 and np.mean(ewma_seq[mask_normal]) > 0:
            hi_alpha = -np.log(0.99) / float(np.mean(ewma_seq[mask_normal]))
        else:
            hi_alpha = 0.5
    hi = np.exp(-hi_alpha * ewma_seq)
    hi = np.clip(hi, 0.0, 1.0)
    # [Bug#3 修复 2026-06-02] 同时返回实际使用的 hi_alpha, 供调用方先在 train 拟合再施于 test。
    return ewma_seq, hi, hi_alpha


# ============================================================
# 名称: phase_metrics
# 修改原因: TriTrackNet 既要报告预测误差, 也要报告残差派生的故障分类指标。
# 作用: 用统一阈值计算一个数据段的完整指标字典 (回归指标 + 逐样本残差异常分数 → 二分类指标)。
# 数学原理: score_i = mean(max(0, y_i - ŷ_i)^2) (实验工具.regression_metrics, 仅正向加热偏离计分);
#          分类由 score_i > threshold 决定。
# 执行流程:
#   1. regression_metrics(y, pred) → 回归指标 + 逐样本 score (二者来自同一残差);
#   2. compute_binary_metrics(labels, scores=score, threshold=) 过滤 -1 并算分类指标;
#   3. 合并输出。
# [评测一致性] (2026-06-01):
#   - score 与分类指标全部走 实验工具.regression_metrics / compute_binary_metrics, 与 AT/TranAD 共用同一口径;
#   - 原始逐点口径, 不做 point-adjustment (AT 入口有 adjust_point_predictions, 本入口刻意不引入);
#   - 对齐: score 与本段 labels 都源自同一滑窗 (N = T-L-H+1), 逐样本时间对应、等长 (见 _load_and_window 注释);
#     labels 传 1D {-1,0,1}, compute_binary_metrics 内部 filter_valid_labels 排除 -1 后才算 TP/FP/...;
#   - threshold/polarity 由调用方经 choose_threshold_and_polarity_by_validation 只用 val 选定, test 段不调阈值;
#     此处传入的 score 为非定向原始残差分数 (单调随故障↑, polarity 恒为 positive), score>threshold 即正确判定。
# 科研标准: 明确分类来自残差, 不声称网络直接输出故障类别; 测试集不参与阈值/极性选择。
# ============================================================
def phase_metrics_from_score(reg: Dict[str, object], score: np.ndarray, labels: np.ndarray,
                             threshold: float, polarity: str = "positive") -> Tuple[Dict[str, object], np.ndarray]:
    """由【预计算的 reg + score】产出完整指标 (与 phase_metrics 同口径, 但不重新 predict/打分)。
    用于 test 段: 配合 predict_and_score 一次前向得 (reg, score), 避免二次物化整张预测张量 (OOM 修复 2026-06-20)。"""
    score_o = orient_scores(score, polarity)   # 极性一致(2026-06-01): 阈值与 final metrics 同向
    # 原始逐点口径 (no point-adjustment): 直接用 score_o>threshold, 不按事件段对齐预测。
    classification = compute_binary_metrics(labels, scores=score_o, threshold=threshold)
    classification.update(reg)
    classification["threshold"] = float(threshold)
    classification["score_definition"] = "mean_squared_forecast_residual"
    return classification, score_o


def phase_metrics(y: np.ndarray, pred: np.ndarray, labels: np.ndarray, threshold: float,
                  polarity: str = "positive") -> Tuple[Dict[str, object], np.ndarray]:
    reg, score = regression_metrics(y, pred)
    return phase_metrics_from_score(reg, score, labels, threshold, polarity)


# ============================================================
# 名称: main
# 修改原因: 提供 E:\创新\启动.py 可直接调用的 TriTrackNet 实验入口。
# 作用: 完成训练、验证阈值、测试和结果记录。
# 数学原理: SmoothL1 训练损失 + MSE 残差异常分数。
# 执行流程:
#   1. 固定随机种子并加载数组;
#   2. 构造/恢复模型;
#   3. 每 epoch 训练后推理 train/val,确定阈值并输出指标;
#   4. 最终在 test 上输出预测与分类指标。
# 科研标准: 测试数据只用于最终评价;smoke 输出不可作为正式结论。
# ============================================================
def main() -> int:
    args = parse_args()
    run_kind = "smoke" if args.smoke else "formal"
    # 2026-06-02 split/feature 版本隔离: checkpoint 与数据目录均按 split 隔离,
    #   narrow_v1 (默认) 保持历史路径, chronological_v2 写/读版本化路径。
    checkpoint = checkpoint_path(args.smoke, args.module, args.seed, args.farm,
                                 args.split_id, args.feature_version, args.preprocess_variant)
    set_seed(args.seed)
    arrays = load_arrays(args.smoke, args.farm, pretrain=args.pretrain,
                         split_id=args.split_id, feature_version=args.feature_version,
                         preprocess_variant=args.preprocess_variant)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, criterion = build_model_and_optimizer(arrays, args, device)
    meta = arrays.get("_meta", {})
    result_dir = result_dir_for_farm(args.farm, args.output_dir)
    common = {
        "farm": args.farm,
        # [契约] meta.json 无 subset 键 (数据驱动管线已废弃 turbine/subset 命名);
        #   CSV 仍保留 subset 列 → 回退为统一占位 "SCADA" (与 AT 的 getattr(solver,'dataset','SCADA') 对齐)。
        "subset": meta.get("subset", "SCADA"),
        "module": args.module,
        "seed": args.seed,
        "n_channels": int(arrays["x_train"].shape[1]),
        "input_shape": [int(arrays["x_train"].shape[0]), int(arrays["x_train"].shape[1]), int(arrays["x_train"].shape[2])],
    }
    preprocess_identity = build_preprocess_identity(
        meta,
        split_id=args.split_id,
        feature_version=args.feature_version,
        preprocess_variant=getattr(args, "preprocess_variant", ""),
        n_channels=int(arrays["x_train"].shape[1]),
    )
    add_preprocess_identity_to_metrics(common, identity=preprocess_identity)

    if args.pretrain:
        raise NotImplementedError(
            "PretrainProtocol 已移除 (2026-05-31). 使用 --pretrain-ckpt 加载预训练权重代替."
        )

    # Phase D: TCN 输入包装 (在 optimizer 之后重新包装)
    # 基线借鉴 #8 (2026-05-26): 新增 tcn_wavelet_residual 分支用 TriTrackNetTCNWaveletWrapper
    if args.module in ("tcn_input_residual", "tcn_wavelet_residual"):
        channels = arrays["x_train"].shape[1]
        # 2026-05-30 TCN-IO: D 一致性闸门(本模型自检 + 跨模型) + 有效感受野覆盖诊断
        from 实验配置 import TCNIOProtocol as _IO
        _IO.assert_input_channels("tritracknet", channels, arrays.get("_meta", {}).get("n_channels"))
        print(_IO.coverage_banner("tritracknet", int(channels)), flush=True)
        if args.module == "tcn_input_residual":
            from TriTrackNet.modules.tcn_增强 import TriTrackNetTCNWrapper
            model = TriTrackNetTCNWrapper(model, input_channels=channels).to(device)
        else:  # tcn_wavelet_residual — 基线借鉴 #8
            from TriTrackNet.modules.tcn_增强 import TriTrackNetTCNWaveletWrapper
            model = TriTrackNetTCNWaveletWrapper(model, input_channels=channels).to(device)
        # ============================================================
        # 名称: TCN 差分学习率优化器重建 (2026-05-30 最优超参数)
        # 修改原因: TCN wrapper 替换 model 后重建 optimizer,旧代码传扁平
        #          model.parameters() 给 perturbopt,TCN 卷积和 α 门控混在 base 组。
        # 作用: 用 TCNProtocol.optimizer_param_groups() 拆为三组:
        #          base 模型: lr=1e-3, wd=1e-5 (论文不变)
        #          TCN 卷积+LN: lr=1e-3, wd=0 (=base;显式分组可保持)
        #          α 门控: lr=1e-2, wd=0 (10×base;单标量需高 lr 快速决断)
        # 已验证: perturbopt.py:35-36 base_optimizer(self.param_groups, **kwargs)
        #         正确透传多组到 AdamW;first_step/second_step/_grad_norm 均按组迭代。
        # ============================================================
        from 实验配置 import TCNProtocol
        optimizer = perturbopt(
            TCNProtocol.optimizer_param_groups(model, base_lr=args.lr, base_wd=args.weight_decay),
            base_optimizer=torch.optim.AdamW,
            rho=args.rho,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    # ============================================================
    # 基线借鉴 #6: 加载预训练 ckpt
    # ============================================================
    if args.pretrain_ckpt:
        pretrain_path = Path(args.pretrain_ckpt)
        if pretrain_path.exists():
            state = torch.load(pretrain_path, map_location="cpu")
            inner = model.model if args.module in ("tcn_input_residual", "tcn_wavelet_residual") else model
            missing, unexpected = inner.load_state_dict(state, strict=False)
            print(
                f"BORROW#6|TriTrackNet|pretrain_loaded|path={pretrain_path}|"
                f"missing={len(missing)}|unexpected={len(unexpected)}",
                flush=True,
            )
        else:
            print(f"BORROW#6|TriTrackNet|pretrain_skip|path_not_exist={pretrain_path}", flush=True)
    _scheduler = None
    if args.force and checkpoint.exists():
        checkpoint.unlink()
    checkpoint_identity = build_checkpoint_identity(
        model="TriTrackNet",
        farm=args.farm,
        module=args.module,
        seed=args.seed,
        input_shape=[int(arrays["x_train"].shape[1]), int(arrays["x_train"].shape[2])],
        preprocess_identity=preprocess_identity,
    )
    last_epoch = restore_checkpoint(model, optimizer, device, checkpoint,
                                       scheduler=_scheduler,
                                       expected_identity=checkpoint_identity) if args.resume else -1
    epochs = 1 if args.smoke else args.epochs

    if checkpoint.exists() and args.resume and not args.force and last_epoch >= epochs - 1:
        print(f"RESUME|model=TriTrackNet|checkpoint={checkpoint}|action=skip_train", flush=True)
    else:
        start_epoch = last_epoch + 1 if args.resume else 0
        # ============================================================
        # 名称: scheduler 构建 (2026-05-31 超参数寻优)
        # 修改原因: TriTrackNet 原先完全无 scheduler, 现按 --scheduler 构建,
        #   在每 epoch 末尾调用 scheduler.step(val_loss)。
        #   PerturbOpt + scheduler 兼容: scheduler 在 second_step 后触发,
        #   不影响扰动参数的梯度计算。
        # ============================================================
        _scheduler = None
        if args.scheduler is not None:
            from 实验配置 import SchedulerProtocol
            _scheduler = SchedulerProtocol.build_scheduler(
                args.scheduler, optimizer,
                T_max=epochs, base_lr=args.lr,
            )
            print(
                f"SCHEDULER|model=TriTrackNet|type={args.scheduler}|"
                f"display={SchedulerProtocol.display_name(_scheduler)}|"
                f"lr={args.lr}|epochs={epochs}",
                flush=True,
            )
            # resume 时前进 scheduler 到当前 epoch
            if start_epoch > 0 and args.scheduler == "cosine":
                for _ in range(start_epoch):
                    _scheduler.step()
        # wandb 离线模式: 训练前初始化
        _wandb_ok = init_wandb_run(
            "TriTrackNet", args.farm, args.module, args.seed,
            epochs=epochs, batch_size=args.batch_size,
        )
        for epoch in range(start_epoch, epochs):
            # 提速计时 (2026-06-03): 测每 epoch 训练墙钟; CUDA 异步故 synchronize 后再读时钟。
            if device.type == "cuda":
                torch.cuda.synchronize()
            _t_epoch0 = time.perf_counter()
            _epoch_loss = train_one_epoch(
                model, optimizer, criterion,
                arrays["x_train"], arrays["y_train"],
                args.batch_size, device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            _epoch_wall = time.perf_counter() - _t_epoch0
            print(
                f"SPEED|model=TriTrackNet|epoch={epoch + 1}|train_wall_s={_epoch_wall:.2f}|"
                f"amp=bf16|train_loss={_epoch_loss:.6f}",
                flush=True,
            )
            # [方法2 提速 2026-06-06] SCADA_GRID_FAST=1: 跳过每-epoch 的【train 打分】(昂贵的 predict(x_train) 前向)。
            #   train 指标无意义(train_positive=0→F1恒0); train_score 只作 choose_threshold 的 val-无双类回退
            #   (grid 的 val 有正例→不触发)。val/早停/选阈值/最终 test 全不变。默认关→90-run/普通跑逐位不变。
            _grid_fast = os.environ.get("SCADA_GRID_FAST") == "1"
            # 提速 (2026-06-18, 不重启即生效): per-epoch val 打分【仅日志】当 scheduler 非 plateau。
            #   问题: cosine/steplr 按 epoch 步进、不吃 val_loss; 无早停/无 best-val/用末 epoch → 中间 epoch 的
            #     val predict+score(87通道全量, 昂贵) 不影响训练/最终 test, 纯日志。本矩阵 best_config=cosine。
            #   方案: SCADA_SKIP_EPOCH_EVAL=1 且非 grid 且非 plateau → 跳 val/train 打分, 只保 checkpoint+scheduler.step。
            #   逻辑: 训练(train_one_epoch)已完成不受影响; 循环后最终 val/test 打分照旧 → 结果逐位不变。
            #   数据流: 不改数据/标签/阈值口径; 仅省中间日志打分。env 已在运行 launcher 中 → 未来子进程自动生效, 不重启。
            skip_val_eval = should_skip_epoch_eval() and args.scheduler != "plateau"
            if not skip_val_eval:
                val_pred = predict(model, arrays["x_val"], args.batch_size, device)
                _, val_score = regression_metrics(arrays["y_val"], val_pred)
                train_pred = None
                if not _grid_fast:
                    _rng_state = _rng_state_snapshot()
                    try:
                        train_pred = predict(model, arrays["x_train"], args.batch_size, device)
                        _, train_score = regression_metrics(arrays["y_train"], train_pred)
                    finally:
                        _restore_rng_state(_rng_state)
                threshold, source, polarity = choose_threshold_and_polarity_by_validation(
                    arrays["label_val"], val_score, (train_score if not _grid_fast else val_score)
                )
                val_score_o = orient_scores(val_score, polarity)
                val_metrics, _ = phase_metrics(arrays["y_val"], val_pred, arrays["label_val"], threshold, polarity)
                val_metrics["threshold_source"] = source
                val_metrics["score_polarity"] = polarity
                val_metrics["run_kind"] = run_kind
                val_metrics.update(common)
                if not _grid_fast:
                    train_score_o = orient_scores(train_score, polarity)
                    train_metrics, _ = phase_metrics(arrays["y_train"], train_pred, arrays["label_train"], threshold, polarity)
                    train_metrics["threshold_source"] = source
                    train_metrics["score_polarity"] = polarity
                    train_metrics["run_kind"] = run_kind
                    train_metrics.update(common)
                    record_and_print_metric(result_dir / "metrics.jsonl", CSV_PATH, "TriTrackNet", "train", epoch + 1, train_metrics)
                    _log_wandb_epoch("train", epoch + 1, train_metrics)
                record_and_print_metric(result_dir / "metrics.jsonl", CSV_PATH, "TriTrackNet", "val", epoch + 1, val_metrics)
                _log_wandb_epoch("val", epoch + 1, val_metrics)
            else:
                print(f"  [TriTrackNet] epoch {epoch + 1}/{epochs} train_loss={_epoch_loss:.4f} "
                      f"(skip per-epoch eval; scheduler={args.scheduler})", flush=True)
            save_checkpoint(
                model, optimizer, epoch, args, checkpoint,
                scheduler=_scheduler,
                checkpoint_identity=checkpoint_identity,
            )
            # scheduler step (2026-05-31 超参数寻优; 跳 val 时 cosine/steplr 按 epoch 步进, 不需 val_loss)
            if _scheduler is not None:
                from 实验配置 import SchedulerProtocol
                if skip_val_eval:
                    SchedulerProtocol.step_scheduler(_scheduler)
                else:
                    _val_loss = float(val_metrics.get("loss", val_metrics.get("mse", 0.0)))
                    SchedulerProtocol.step_scheduler(_scheduler, val_loss=_val_loss)
        if _wandb_ok:
            finish_wandb_run()

    # OOM 修复 (2026-06-16): TriTrack 输出宽 (24步×87通道=2088列), 原三段 pred 全量同驻 RAM
    #   = train 7.4G + val 2.4G + test 4.9G ≈ 14.7G 峰值 → RAM 紧时 ArrayMemoryError(grid 实测崩多次)。
    # 方案 (升级 2026-06-20 系统崩溃修复): predict_and_score 逐批前向+打分, 永不物化整张预测张量。
    #   原"逐段 predict→打分→del+gc"只降到单段 (~7.4G@kel), 但 Hill 单段 pred 28.8G + y 28.8G ≈57G
    #   仍超 commit 上限 73.4G → OS 强杀 (System 事件 2004, 无 traceback)。融合后峰值降到单批 (~MB)。
    # 数学/逻辑: model.eval() 确定性 + 调用顺序仍 train→val→test 不变 → 逐样本分数逐位一致, 结果不变。
    _, train_score = predict_and_score(model, arrays["x_train"], arrays["y_train"], args.batch_size, device)
    _, val_score = predict_and_score(model, arrays["x_val"], arrays["y_val"], args.batch_size, device)
    threshold, source, polarity = choose_threshold_and_polarity_by_validation(
        arrays["label_val"], val_score, train_score
    )
    reg_test, test_score = predict_and_score(model, arrays["x_test"], arrays["y_test"], args.batch_size, device)
    test_metrics, _ = phase_metrics_from_score(reg_test, test_score, arrays["label_test"], threshold, polarity)
    test_metrics["threshold_source"] = source
    test_metrics["score_polarity"] = polarity
    test_metrics["run_kind"] = run_kind
    test_metrics.update(common)
    # ============================================================
    # 名称: 预后派生 (2026-05-30) — EWMA → HI(t) → lead_time
    # 修改原因: TriTrackNet 原论文只输出预测值 y_pred,不做异常检测和预后。
    #          温度异常检测方向需要从预测残差派生健康指标和预警提前量,
    #          这是三模型中唯一能回答"还能撑多久"的能力载体。
    # 作用: 对 test 预测残差做 EWMA 平滑 → 健康指标 HI(t) → 计算 lead_time。
    # 数学原理:
    #   1. score = mean((y - ŷ)²) — 预测残差 MSE
    #   2. EWMA(α=0.05): ewma[t] = α·score[t] + (1-α)·ewma[t-1], 半衰期≈3.3h
    #      (匹配轴承热惯量时间常数)
    #   3. HI(t) = exp(-α₁·ewma[t]), α₁ 自动拟合使正常段 HI≈0.99
    #   4. lead_time = t_event − ceil(HI 首次跌破 0.7 的时刻), 10-min步→小时
    # 执行流程: test_score (上方 predict_and_score 一次前向已得) → score_to_ewma_hi → 扫真实事件→写指标。
    # 科研标准: 纯后处理,不参与训练和阈值选择; HI_warn=0.7 为经验值,不来自 test 优化。
    # ============================================================
    # test_score 已由上方 predict_and_score 得到 (不再二次 predict/物化整张预测张量, OOM 修复 2026-06-20)。
    test_score_o = orient_scores(test_score, polarity)
    # [Bug#3 修复 2026-06-02] HI 负指数系数 α₁ 必须在 train 正常段拟合, test 只 transform。
    #   原 score_to_ewma_hi(test_score_o) 让 α₁ 自标定在 test 分布 → lead-time 含 test 泄漏。
    #   先在 oriented train 分数上拟合 α₁, 再用同一 α₁ 施于 test (与项目 train-only 红线一致)。
    train_score_o = orient_scores(train_score, polarity)
    _, _, _hi_alpha_train = score_to_ewma_hi(train_score_o)
    ewma_seq, hi_seq, _ = score_to_ewma_hi(test_score_o, hi_alpha=_hi_alpha_train)
    test_metrics["hi_warn"] = 0.7
    test_metrics["hi_mean"] = float(np.mean(hi_seq))
    # 逐真实故障事件计算预警提前量
    label_arr = np.asarray(arrays["label_test"]).reshape(-1).astype(int)
    # 找每个事件起始时刻: 标签从 0→1 的跳变点
    event_starts = np.where((label_arr[:-1] == 0) & (label_arr[1:] == 1))[0] + 1
    lead_times = []
    for t_start in event_starts:
        # 在事件开始前找 HI 首次跌破 0.7 的时刻 (最近一次触发)
        warn_idx = np.where(hi_seq[:t_start] < 0.7)[0]
        if len(warn_idx) > 0:
            lead_times.append((t_start - warn_idx[-1]) * 10 / 60.0)  # 10-min步 → 小时
    if lead_times:
        test_metrics["lead_time_median_h"] = float(np.median(lead_times))
        test_metrics["lead_time_p25_h"] = float(np.percentile(lead_times, 25))
        test_metrics["lead_time_p75_h"] = float(np.percentile(lead_times, 75))
    else:
        test_metrics["lead_time_median_h"] = float("nan")
    record_and_print_metric(result_dir / "metrics.jsonl", CSV_PATH, "TriTrackNet", "test", "final", test_metrics)
    # ============================================================
    # 基线借鉴 #4: 落盘 train/val/test scores + labels 供 集成评价.py 读取
    # 修改原因 (2026-05-26): 同 AT/TranAD 落盘策略, 集成评价.py 通过文件名扫描。
    # 修改原因 (2026-06-01 Task6): 新增 train scores 落盘; 路径经 ResultLayout 版本化。
    # 文件名约定: {train|val|test}_{scores|labels}__{module}__seed{seed}.npy
    # ============================================================
    val_score_o = orient_scores(val_score, polarity)
    train_score_o = orient_scores(train_score, polarity)
    from 实验配置 import ResultLayout
    _split_id = getattr(args, "split_id", "narrow_v1")
    _fv = getattr(args, "feature_version", "v1")
    if _split_id != "narrow_v1":
        scores_dir = ResultLayout.scores_dir(
            _split_id, _fv, args.farm, "tritracknet",
            preprocess_variant=getattr(args, "preprocess_variant", ""),
        )
    else:
        scores_dir = result_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"__{args.module}__seed{args.seed}"
    np.save(scores_dir / f"train_scores{suffix}.npy", train_score_o.astype(np.float32))
    np.save(scores_dir / f"train_labels{suffix}.npy", np.asarray(arrays["label_train"]).reshape(-1).astype(np.int8))
    np.save(scores_dir / f"val_scores{suffix}.npy",  val_score_o.astype(np.float32))
    np.save(scores_dir / f"val_labels{suffix}.npy",  np.asarray(arrays["label_val"]).reshape(-1).astype(np.int8))
    np.save(scores_dir / f"test_scores{suffix}.npy", test_score_o.astype(np.float32))
    np.save(scores_dir / f"test_labels{suffix}.npy", np.asarray(arrays["label_test"]).reshape(-1).astype(np.int8))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
