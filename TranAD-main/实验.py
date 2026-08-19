# -*- coding: utf-8 -*-
"""
实验.py

名称: TranAD 的 SCADA 专用实验入口
修改原因: 原 main.py 没有 SCADA 数据分支,且指标输出未统一包含 accuracy/loss/AUC 与 ignore 过滤。
作用: 读取统一预处理数据,复用 TranAD 网络结构,完成 checkpoint 续跑、验证阈值和测试指标记录。
数学原理:
    1. TranAD 两阶段输出 z1,z2,训练损失为
       L_n = (1/n) * MSE(z1,x) + (1-1/n) * MSE(z2,x);
    2. 测试异常分数为变量维重建误差均值 score_t = mean_d((x_td-z_td)^2);
    3. 阈值仅由验证集或训练分数决定。
执行流程:
    1. 读取 processed/SCADA_{farm}/{train,val,test}.npy + {split}_labels.npy (统一数据契约);
    2. 以原 TranAD 类构建网络并恢复 checkpoint;
    3. 逐 epoch 训练并验证,打印 METRIC 行;
    4. 在测试集输出统一分类指标与 loss。
科研标准: 不使用测试集调阈值;label=-1 剔除;保留原 TranAD 两阶段训练目标。
参考文献:
    Tuli, S., Casale, G., & Jennings, N. R. (2022). TranAD: Deep Transformer
    Networks for Anomaly Detection in Multivariate Time Series Data. PVLDB,
    15(6), 1201-1214. PDF: https://www.vldb.org/pvldb/vol15/p1201-tuli.pdf
    Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017.
    PDF: https://arxiv.org/pdf/1706.03762
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import types
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
os.chdir(SCRIPT_DIR)
# 注: os.chdir 更改进程级工作目录。若此脚本被其他模块导入可能影响路径解析。
# 用绝对路径拼接替代 os.chdir 会更安全。
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

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
    should_skip_epoch_eval,
)


DATA_DIR = SCRIPT_DIR / "processed" / "SCADA"
RESULT_DIR = ROOT / "实验结果" / "tranad"
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
# 修改原因 (2026-05-25 全量化): Phase D 单一 processed/SCADA/ 目录无法承载三 farm 数据,
#          需要按 farm 隔离避免 Kelmarsh/Penmanshiel/Hill of Towie 的 train/val/test 数组互相覆盖。
#          (2026-06-02 增补 split/feature 隔离) 旧版本硬编码 processed/SCADA_{farm},
#          全量 chronological_v2 会读到历史 narrow_v1 目录 (读错数据); 且与 AT/TriTrackNet
#          各存一份路径模板存在漂移风险。改为唯一真源 = 实验配置.PerFarmPaths。
# 作用: 给定 farm + split_id/feature_version 返回该 farm 对应的 .npy 输入目录;
#       narrow_v1 (默认) 解析为旧 processed/SCADA_{farm}/ (向后兼容历史与既有测试),
#       其它 split (chronological_v2) 解析为版本化目录 SCADA_{farm}__{split_id}__{feature_version}/。
# 数学原理: 无 (路径字符串拼接)。
# 执行流程: 委托 PerFarmPaths.for_farm(farm, split_id, feature_version) 取 "tranad" 键。
# 科研标准: 路径含 farm + 版本段, 任何 metrics 行都能反查原始输入文件; 路径模板与三 baseline 同源于配置;
#          narrow_v1 与 chronological_v2 物理隔离, 复跑互不污染。
# ============================================================
def data_dir_for_farm(farm: str, split_id: str = "narrow_v1", feature_version: str = "v1",
                      preprocess_variant: str | None = None) -> Path:
    from 实验配置 import PerFarmPaths  # noqa: E402
    return Path(PerFarmPaths.for_farm(
        farm, split_id, feature_version, preprocess_variant=preprocess_variant,
    )["tranad"])


# ============================================================
# 名称: result_dir_for_farm
# 修改原因 (2026-05-25 全量化): metrics 输出目录之前只有 实验结果/tranad/,
#          多 farm 写同一文件会乱序混合;需要按 farm 隔离落 jsonl。
# 作用: 给定 farm 名 (与可选 override) 返回该 farm 的 TranAD metrics 目录。
# 数学原理: 无 (路径计算)。
# 执行流程:
#   1. 若 output_dir_override 非空 → 用 override (供 启动.py 显式指定使用);
#   2. 否则 → 实验结果/{farm}/tranad/。
# 科研标准: 横向对比 (kelmarsh vs penmanshiel) 时通过 farm 段 join CSV,避免 JSON 行 farm 字段缺失。
# ============================================================
def result_dir_for_farm(farm: str, output_dir_override: str | None = None) -> Path:
    if output_dir_override:
        return Path(output_dir_override)
    return ROOT / "实验结果" / farm / "tranad"


# ============================================================
# 名称: checkpoint_path
# 修改原因: smoke 小样本模型不能被正式实验续跑误当作同一训练状态。
#   (2026-06-02 增补 split/feature 隔离) checkpoint 旧路径不含版本段,
#   全量 chronological_v2 与历史 narrow_v1 的 checkpoint 会写同一路径互相覆盖。
# 作用: 按运行类型返回独立 checkpoint 路径; 仅在 split_id != narrow_v1 时在 farm 段后
#   插入 f"{split_id}__{feature_version}" 段, narrow_v1 保持旧路径字节级一致。
# 数学原理: 无。
# 执行流程: smoke=True 选 smoke 子目录否则 formal; 非 narrow_v1 在 farm 后插版本段。
# 科研标准: 快速链路验证与正式实验参数轨迹严格隔离;
#   narrow_v1 路径与历史一致 (既有测试依赖), chronological_v2 与其物理隔离。
# ============================================================
def checkpoint_path(smoke: bool, module: str = "baseline_only", seed: int = 0, farm: str = "kelmarsh",
                    split_id: str = "narrow_v1", feature_version: str = "v1",
                    preprocess_variant: str | None = None) -> Path:
    kind = "smoke" if smoke else "formal"
    ckpt_root = SCRIPT_DIR / "checkpoints" / "TranAD_SCADA" / farm
    if split_id != "narrow_v1":
        tag = f"{split_id}__{feature_version}"
        variant = str(preprocess_variant or "").strip()
        if variant:
            tag = f"{tag}__{variant}"
        ckpt_root = ckpt_root / tag
    return ckpt_root / kind / module / f"seed{seed}" / "model.ckpt"


# ============================================================
# 名称: parse_args
# 修改原因: 正式实验和 smoke 验证需要不同训练规模,并需要控制 resume/force。
# 作用: 定义 TranAD SCADA 实验命令行参数。
# 数学原理: 无。
# 执行流程: argparse 解析参数。
# 科研标准: 超参数可记录、可重复运行。
# ============================================================
def parse_args() -> argparse.Namespace:
    r"""
    科研注释：函数 `parse_args`
    名称作用：声明 TranAD SCADA 实验的训练规模、checkpoint 策略、模块选择和 smoke 执行开关。
    Phase D 修改：增加 --module、--seed、--run-id、--output-dir;默认 epochs=10 (TrainingProtocol.EPOCHS)、batch_size=128。
    """
    parser = argparse.ArgumentParser(description="TranAD SCADA 实验入口")
    parser.add_argument("--epochs", type=int, default=10, help="正式实验 epoch 数 (TrainingProtocol.EPOCHS=10)")
    parser.add_argument("--batch-size", type=int, default=128, help="训练 batch size (TranAD .double() fp64,不开AMP)")
    # 基线借鉴 #8: 新增 tcn_wavelet_residual 选项
    parser.add_argument("--module", type=str, default="baseline_only",
                        choices=["baseline_only", "tcn_input_residual", "tcn_wavelet_residual"],
                        help="Phase D 模块选择 (基线借鉴 #8: 新增 tcn_wavelet_residual)")
    parser.add_argument("--seed", type=int, default=0, help="随机种子 (启动器注入 0-4; 单跑默认 0)")
    parser.add_argument("--run-id", type=str, default="run_001", help="运行标识")
    parser.add_argument("--output-dir", type=str, default=None, help="覆盖默认输出目录")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 继续或直接评价")
    parser.add_argument("--force", action="store_true", help="删除现有 checkpoint 并重新训练")
    parser.add_argument("--smoke", action="store_true", help="小样本一轮链路验证")
    # [G3 死引用清理 2026-06-01] 移除 --pretrain (action store_true): 其唯一用途是触发
    #   NotImplementedError 占位 (PretrainProtocol 已于 2026-05-31 移除), 启动.py 也从不传该参;
    #   预训练权重改用下方 --pretrain-ckpt 直接加载。删除 --pretrain 不影响任何调用方。
    parser.add_argument("--farm", type=str, default="kelmarsh",
                        choices=["kelmarsh", "penmanshiel", "hill_of_towie"],
                        help="全量化实验 farm 名 (默认 kelmarsh)")
    parser.add_argument("--split-id", type=str, default="narrow_v1", help="切分协议 id (全量=chronological_v2)")
    parser.add_argument("--feature-version", type=str, default="v1", help="特征版本 (全量=v2)")
    parser.add_argument("--preprocess-variant", type=str, default="",
                        help="预处理变体后缀, 例如 old_preprocess / new_preprocess")
    # --pretrain-ckpt: 仍保留 (功能性: 微调时加载已有权重, 非占位)
    parser.add_argument("--pretrain-ckpt", type=str, default=None,
                        help="预训练 checkpoint 路径 (微调时加载已有权重)")
    # ============================================================
    # 名称: --scheduler / --lr / --batch-size 重声明 (2026-05-31 超参数寻优)
    # 修改原因: TranAD 原 parse_args 没有 --lr 和 --scheduler 参数,
    #   网格搜索需要通过命令行注入这些值覆盖 model.lr 和默认 StepLR。
    # 作用: 接收超参, 供 build_model/build_steps 动态替换优化器/scheduler。
    # ============================================================
    parser.add_argument("--scheduler", type=str, default=None,
                        choices=["cosine", "plateau", "steplr"],
                        help="学习率调度器类型 (网格搜索注入; 默认 None=StepLR)")
    parser.add_argument("--lr", type=float, default=None, help="覆盖 model.lr (网格搜索注入)")
    return parser.parse_args()


# ============================================================
# 名称: install_optional_dgl_stub
# 修改原因: 原 src.models 在模块顶层导入 dgl,但 TranAD 主模型本身不使用 GATConv;
#          未安装 dgl 时会阻止仅运行 TranAD 的科研实验。
# 作用: 仅在 dgl 缺失时提供不可实例化的 GATConv 占位,允许导入 TranAD 类。
# 数学原理: 无;不改变 TranAD 网络计算图。
# 执行流程:
#   1. 尝试 import dgl;
#   2. 失败时向 sys.modules 注册 dgl/dgl.nn 占位模块;
#   3. 若误用 GATConv 立即抛出明确错误。
# 科研标准: 占位仅适用于 TranAD;不得据此声明 GDN/MTAD-GAT 已可运行。
# ============================================================
def install_optional_dgl_stub() -> None:
    r"""
    科研注释：函数 `install_optional_dgl_stub`
    名称作用：在仅运行 TranAD 且系统未装 DGL 时，提供导入阶段的最小兼容占位。
    参数说明：无外部参数；通过模块导入状态判断 DGL 是否存在。
    返回值：无显式返回值；缺少 DGL 时向 `sys.modules` 注册占位模块。
    数学原理：该兼容层不参与 TranAD 计算图，也不产生重构误差或异常分数。
    流程说明：尝试导入 DGL -> 缺失时定义拒绝实例化的 GATConv -> 注册模块占位。
    SCADA 迁移：只可用于 TranAD 实验导入，图模型基线仍必须安装真实 DGL 后另行验证。
    """
    try:
        __import__("dgl")
        return
    except ImportError:
        pass

    dgl_module = types.ModuleType("dgl")
    dgl_nn_module = types.ModuleType("dgl.nn")

    class UnavailableGATConv(nn.Module):
        r"""
        科研注释：类 `UnavailableGATConv`
        名称作用：显式阻止在无 DGL 环境下误运行依赖图卷积的基线模型。
        参数说明：接受原构造调用的任意参数，仅用于在错误路径上给出明确异常。
        返回值：类不能成功实例化；实例化即抛出 `ImportError`。
        数学原理：不实现图注意力运算，因此不能用于任何实验指标计算。
        流程说明：进入构造函数 -> 初始化父类 -> 立即报告缺少 DGL。
        SCADA 迁移：该保护防止把未真实运行的 GDN/MTAD-GAT 指标误写入对比实验。
        """

        def __init__(self, *args, **kwargs):
            r"""
            科研注释：方法 `__init__`
            名称作用：在误实例化图注意力占位层时立即终止实验。
            参数说明：`*args` 与 `**kwargs` 仅兼容调用签名，不用于计算。
            返回值：不返回正常对象，固定抛出依赖缺失异常。
            数学原理：无模型运算；该分支仅用于实验依赖保护。
            流程说明：初始化 `nn.Module` -> 抛出 `ImportError`。
            SCADA 迁移：保证比较基线必须在真实依赖满足后才可声称可运行。
            """
            super().__init__()
            raise ImportError("dgl 未安装: 仅 TranAD 可运行,GDN/MTAD_GAT 不可用。")

    dgl_nn_module.GATConv = UnavailableGATConv
    dgl_module.nn = dgl_nn_module
    sys.modules["dgl"] = dgl_module
    sys.modules["dgl.nn"] = dgl_nn_module


# ============================================================
# 名称: import_tranad_model
# 修改原因: src.constants 在导入时读取 src.parser 的 dataset/model 参数。
# 作用: 用 SCADA/TranAD 参数安全导入原始 TranAD 类。
# 数学原理: 无;只控制模块初始化参数。
# 执行流程:
#   1. 安装可选 dgl stub;
#   2. 暂存 sys.argv;
#   3. 以 --dataset SCADA --model TranAD 导入类;
#   4. 恢复原 sys.argv。
# 科研标准: 明确模型类型为原 TranAD,不混入其它 baseline。
# ============================================================
def import_tranad_model():
    r"""
    科研注释：函数 `import_tranad_model`
    名称作用：在固定为 SCADA/TranAD 的解析上下文中导入原仓库 `TranAD` 网络类。
    参数说明：无外部参数；使用当前解释器的模块路径和临时命令行参数。
    返回值：返回 `src.models.TranAD` 类对象。
    数学原理：导入阶段不执行训练；其目标是选择使用 self-conditioning 和两阶段重构的 TranAD 结构。
    流程说明：安装可选依赖占位 -> 暂存 argv -> 以指定模型导入 -> 恢复 argv。
    SCADA 迁移：确保运行的是 TranAD 主模型，而非依赖不同通道关系假设的其他基线。
    """
    install_optional_dgl_stub()
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], "--dataset", "SCADA", "--model", "TranAD"]
        from src.models import TranAD
    finally:
        sys.argv = original_argv
    return TranAD


# ============================================================
# 名称: load_arrays
# 修改原因: SCADA 数据已由统一预处理产生,实验入口不得再次改变缩放或时间切分。
# [E1 修复 2026-06-01] 统一数据契约: 文件名固定 train/val/test.npy + {split}_labels.npy,
#   meta.json 不再含 "subset" 键。原代码 subset = meta["subset"] 必 KeyError;
#   现直接读契约文件名, 通道数 C 由 shape[1] 动态读 (各 farm 不同)。
# 作用: 读取 train/val/test 序列及对应逐点标签。
# 数学原理: X ∈ R^{T×C} (C=shape[1] 动态), Y ∈ {-1,0,1}^{T} (1D)。
# 执行流程:
#   1. 按契约固定文件名读取 numpy;
#   2. smoke 时仅保留前 512 点用于链路验证;
#   3. 返回字典。
# 科研标准: 不在模型侧重新归一化,避免数据泄漏和三模型输入口径不一致。
# ============================================================
def load_arrays(smoke: bool, farm: str, split_id: str = "narrow_v1",
                feature_version: str = "v1",
                preprocess_variant: str | None = None) -> Dict[str, np.ndarray]:
    r"""
    科研注释：函数 `load_arrays`
    名称作用：读取统一预处理后的 SCADA 训练、验证、测试数组及逐点标签。
    参数说明：`smoke` 指示是否只读取前 512 个时间点进行执行链路快速核验;
              `split_id`/`feature_version` 用于版本隔离数据目录 (2026-06-02),
              narrow_v1 (默认) 读旧 processed/SCADA_{farm}/, 其它 split 读版本化目录。
    返回值：返回包含三个数据段及三组标签的字典。
    数学原理：输入为 $X\in\mathbb{R}^{T\times C}$ (C 由 shape[1] 动态读)，标签保持 $\{-1,0,1\}$ 语义以支持忽略区间过滤。
    流程说明：经 data_dir_for_farm(farm, split_id, feature_version) 取版本化目录 ->
              按契约固定文件名加载数组 -> smoke 模式截断时间轴 -> 返回数据映射。
    SCADA 迁移：该层不重新标准化也不重切时间段，避免验证/测试数据泄漏;
              narrow_v1 路径与历史一致, chronological_v2 与其物理隔离。
    """
    data_dir = data_dir_for_farm(farm, split_id, feature_version, preprocess_variant)
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"缺少 {meta_path}; 请先运行数据预处理 --mode full --farm {farm}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # [E1] 契约文件名固定: {split}.npy 为 float32 [T,C]; {split}_labels.npy 为 int64 1D ∈ {-1,0,1}。
    #      meta.json 含 n_channels/cols, 不含 subset; 通道数 C 后续由 arrays["train"].shape[1] 动态取。
    arrays = {
        "train": np.load(data_dir / "train.npy"),
        "val": np.load(data_dir / "val.npy"),
        "test": np.load(data_dir / "test.npy"),
        "train_labels": np.load(data_dir / "train_labels.npy"),
        "val_labels": np.load(data_dir / "val_labels.npy"),
        "test_labels": np.load(data_dir / "test_labels.npy"),
        "_meta": meta,
    }
    if smoke:
        return {key: (value[:512] if isinstance(value, np.ndarray) else value) for key, value in arrays.items()}
    return arrays


# ============================================================
# 名称: convert_to_windows
# 修改原因: TranAD 网络输入为固定长度历史窗口,原 main.py 的窗口逻辑需在 SCADA 入口复用。
# 作用: 将 (T,D) 数据转换为 (T,L,D) 滑动窗口。
# 数学原理: W_t = [x_{max(0,t-L)},...,x_{t-1}],序列开头用 x_0 前向填充。
# 执行流程:
#   1. 对每个时刻 t 构造长度 L 的历史窗口;
#   2. 长度不足时重复首点补齐;
#   3. stack 为三维张量。
# 科研标准: 按时间顺序构造窗口,不随机重排评价数据。
# ============================================================
def convert_to_windows(
    data: np.ndarray,
    window_size: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""
    科研注释：函数 `convert_to_windows`
    名称作用：把逐时间点 SCADA 数组转换成 TranAD 要求的固定长度历史窗口。
    参数说明：`data` 是形状 $(T,D)$ 的序列；`window_size` 是历史上下文长度 $L$。
    返回值：返回形状 $(T,L,D)$ 的 `torch.float32` 窗口张量。
    数学原理：$W_t=[x_{t-L},\ldots,x_{t-1}]$；序列起始不足 $L$ 时用首点前向补齐。
    流程说明：左侧前向补齐首点 -> `unfold` 切出历史窗视图 -> 维度还原成 (T,L,D)。
    SCADA 迁移：按时间顺序构造窗口，保留热异常逐步积累和振动突变的时序关系。
    速度优化 (2026-06-03): 用 tensor.unfold 向量化替换原 Python 循环 + torch.stack
        (1.16M 窗口逐个构造极慢)。已对 T<L / T=1 / T>L 等边界逐元素核验,
        与原循环 byte-identical。dtype 由 fp64 改 fp32 (用户接受精度换 ~10x 速度,
        RTX5090 fp64 吞吐约 fp32 的 1/30); .npy 数据契约本就是 float32, 无信息损失。
    内存修复 (2026-06-20): 返回非 contiguous 视图, 避免 Hill of Towie test 一次性
        materialize 约 8GB 窗口张量。DataLoader 会在每个 batch 内再拷贝成连续张量,
        窗口语义与标签 off-by-one 对齐不变。
    """
    tensor = torch.as_tensor(data, dtype=torch.float32, device=device)
    T = tensor.shape[0]
    # 左侧用首点 x_0 前向补齐 window_size 行, 使每个时刻都有完整历史 (与原循环 i<window_size 分支等价):
    #   原循环: 窗口 i = [x_0 重复 (L-i) 次 , x_0..x_{i-1}] (i<L) 或 x_{i-L}..x_{i-1} (i>=L)。
    #   补齐后在 padded 坐标里, 时刻 i 的历史窗 = padded[i : i+L], 二者完全一致。
    pad = tensor[0:1].repeat(window_size, 1)            # (L, D) 首点复制
    padded = torch.cat([pad, tensor], dim=0)            # (T+L, D)
    w = padded.unfold(0, window_size, 1)                # (T+1, D, L) 滑动窗 (步长 1)
    return w[:T].permute(0, 2, 1)                      # (T, L, D) lazy view


def window_storage_bytes(arrays: Dict[str, np.ndarray], window_size: int) -> int:
    total = 0
    for key in ("train", "val", "test"):
        arr = arrays[key]
        total += (int(arr.shape[0]) + int(window_size)) * int(arr.shape[1]) * np.dtype(np.float32).itemsize
    return total


def select_window_device(
    arrays: Dict[str, np.ndarray],
    window_size: int,
    train_device: torch.device,
) -> torch.device:
    mode = os.environ.get("SCADA_TRANAD_GPU_WINDOWS", "auto").strip().lower()
    if train_device.type != "cuda" or mode in {"0", "false", "off", "cpu"}:
        return torch.device("cpu")
    bytes_needed = window_storage_bytes(arrays, window_size)
    try:
        with torch.cuda.device(train_device):
            free, _total = torch.cuda.mem_get_info()
    except Exception:
        return torch.device("cpu")
    forced = mode in {"1", "true", "on", "cuda", "gpu"}
    if forced or bytes_needed <= int(free * 0.35):
        print(
            f"SPEED|TranAD|window_storage=cuda|bytes={bytes_needed}|free={free}|mode={mode}",
            flush=True,
        )
        return train_device
    print(
        f"SPEED|TranAD|window_storage=cpu|bytes={bytes_needed}|free={free}|mode={mode}",
        flush=True,
    )
    return torch.device("cpu")


class _IndexDataset(Dataset):
    """Dataset that lets DataLoader sample indices while collate slices windows by batch."""

    def __init__(self, length: int) -> None:
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return int(index)


def make_window_loader(
    windows: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
    num_workers: int = 0,
) -> DataLoader:
    def collate_indices(indices):
        idx = torch.as_tensor(indices, dtype=torch.long, device=windows.device)
        return (windows.index_select(0, idx).contiguous(),)

    return DataLoader(
        _IndexDataset(windows.shape[0]),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=(pin_memory and windows.device.type == "cpu"),
        num_workers=num_workers,
        collate_fn=collate_indices,
    )


# ============================================================
# 名称: align_scores_to_labels  (C4 修复 2026-06-01)
# 修改原因 [C4 窗口/标签 off-by-one]:
#   convert_to_windows 用【历史窗】: 窗口 i = data[i-L:i], 末元素 = data[i-1],
#   evaluate_scores 用 window[-1] 作重建目标, 故 scores[i] = 时刻 t=i-1 的重建误差。
#   即 scores 的下标比物理时间超前 1 步 (score index leads time by 1)。
#   原版 TranAD main.py 直接做 scores[i] ↔ labels[i] 的逐下标比对, 未修正该偏移
#   (其 loss/y_pred 与 labels 均为长度 T, zip 同下标), 因此整体错位 1 步。
#   本 SCADA 入口在【评测处】修正: 把 scores[i] 与其真正对应时刻 t=i-1 的 label 对齐。
# 平移方向与依据 (务必精确):
#   - 事实: scores[i] 对应时刻 t = i-1  ⇒  应与 labels[i-1] 配对。
#   - 取 i ∈ [1, T-1] (丢弃退化的 i=0: 其窗口全为 data[0] 前向填充重建 t=0, 无意义):
#       scores[1:]   下标 i=1..T-1 → 时刻 t=0..T-2
#       labels[:-1]  即 labels[0..T-2] = 对应时刻 t=0..T-2
#   - 因此 “scores 丢首 (scores[1:])、labels 丢尾 (labels[:-1])”, 二者按时刻精确对齐。
#     等价说法: scores 超前时间 1 步 → 砍掉 scores 第 0 个 + 砍掉 labels 最后 1 个。
#     (这是最紧的正确平移, 只丢 1 个退化点; 与任务给的 labels[L-1:] 思路同向, 但不依赖 L。)
# 数学原理: 无新模型公式; 纯下标平移, 保证配对 (scores_t, labels_t) 时刻一致。
# 执行流程: 校验长度一致 → scores 取 [1:] → labels 取 [:-1] → 返回等长二元组。
# 科研标准: train/val/test 三段及 per-channel 残差、落盘 scores/labels 全部走同一平移,
#           确保 scores 与所用 labels 始终等长且同时刻对齐 (验证集据此选阈值/极性)。
# ============================================================
def align_scores_to_labels(
    scores: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    科研注释：函数 `align_scores_to_labels`
    名称作用：修正历史窗造成的 scores↔labels off-by-one, 返回按时刻对齐且等长的 (scores, labels)。
    参数说明：`scores` 形状 (T,) 或 (T,D); `labels` 形状 (T,) 或 (T,D); 二者首维必须等于序列长度 T。
    返回值：`(scores[1:], labels[:-1])`, 首维长度均为 T-1, 已按物理时刻对齐。
    数学原理：scores[i] = 时刻 t=i-1 的重建误差 ⇒ 与 labels[i-1] 配对; 取 i∈[1,T-1] 得上式。
    流程说明：长度一致性断言 -> scores 丢首 -> labels 丢尾 -> 返回。
    SCADA 迁移：评测口径保持逐点 (不做 point-adjustment), 与另两 baseline 一致。
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(
            f"[C4] scores 长度 {scores.shape[0]} 与 labels 长度 {labels.shape[0]} 不一致, 无法对齐"
        )
    return scores[1:], labels[:-1]


# ============================================================
# 名称: train_one_epoch
# 修改原因: 需要在外置入口记录每 epoch 的 loss 且保留原 TranAD 两阶段损失。
# 作用: 执行一次训练 epoch。
# 数学原理: L_n=(1/n)MSE(z1,x)+(1-1/n)MSE(z2,x)。
# 执行流程:
#   1. DataLoader 按 batch 提供窗口;
#   2. 将窗口排列为 (L,B,D),取最后一步为重建目标;
#   3. 前向计算 z1,z2 与 L_n;
#   4. 反向更新并返回平均损失。
# 科研标准: 训练损失公式与原 TranAD main.py 一致。
# ============================================================
def train_one_epoch(
    model: nn.Module,
    windows: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    batch_size: int,
    device: torch.device,
) -> float:
    r"""
    科研注释：函数 `train_one_epoch` (基线借鉴 #9 DataLoader 优化版)
    名称作用：按原 TranAD 两阶段自条件目标训练一个 epoch 并返回平均重构损失。
    参数说明：`model` 为 TranAD；`windows` 为 $(T,L,D)$ 输入；`optimizer`、`scheduler` 控制更新。
    返回值：返回该 epoch 所有 batch 损失的均值。
    数学原理：$L_n=\frac{1}{n}MSE(z_1,x)+(1-\frac{1}{n})MSE(z_2,x)$。
    AMP 说明 (基线借鉴 #1): TranAD 原 wrapper 用 `.double()` (fp64), 与 torch.amp.autocast (fp16) 不兼容,
        强行启用会报 "Found dtype Double but expected Float"; 因此 TranAD 保留 fp32/fp64 不加 AMP,
        但仍享受基线借鉴 #9 的 DataLoader 优化 (pin_memory + non_blocking)。
    SCADA 迁移：训练只利用输入重构目标, 不把测试故障标签用于参数更新。
    """
    criterion = nn.MSELoss(reduction="none")
    # ============================================================
    # 名称: DataLoader pin_memory + non_blocking (基线借鉴 #9)
    # 修改原因 (2026-05-26 借鉴自基线): 见 baseline_suite/trainer.py L107-116;
    #         pin_memory 锁页内存 + 异步传输让 GPU 不等数据 → 等价于训得更多。
    # 科研标准: Windows num_workers=0 (多进程 + 大 pickle 不稳, 见 TrainingProtocol.DATALOADER_NUM_WORKERS)。
    # ============================================================
    pin = (device.type == "cuda")
    # 速度优化 #4 (2026-06-03): 已尝试 num_workers=4 + persistent_workers 让取数与计算重叠,
    #   但 Windows 上对 1.16M×L×D 的大 TensorDataset 跨 worker 共享内存会 RuntimeError
    #   "Couldn't open shared file mapping ... error code 1455" (ERROR_COMMITMENT_LIMIT)。
    #   按 FALLBACK 回退此项, 保持 num_workers=0 (见 TrainingProtocol.DATALOADER_NUM_WORKERS);
    #   其余三项 (fp32 / GPU 端 loss 累加 / unfold 向量化窗口) 保留。
    loader = make_window_loader(
        windows,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
        num_workers=0,
    )
    model.train()
    n = epoch + 1
    # 速度优化 #2 (2026-06-03): 不再每 batch 调 loss.item() (强制 GPU->CPU 同步使 GPU 空转);
    #   改为在 GPU 上累加 loss.detach(), epoch 末仅一次 .item()。数值结果与原 np.mean(losses) 一致
    #   (均为各 batch 标量 loss 的等权平均)。
    loss_sum = torch.zeros((), device=device)
    nb = 0
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)  # 基线借鉴 #9: non_blocking
        local_batch = batch.shape[0]
        window = batch.permute(1, 0, 2)
        target = window[-1, :, :].view(1, local_batch, -1)
        z1, z2 = model(window, target)
        loss_tensor = (1 / n) * criterion(z1, target) + (1 - 1 / n) * criterion(z2, target)
        loss = torch.mean(loss_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum = loss_sum + loss.detach()
        nb += 1
    # [Bug#1 修复 2026-06-02] 原 `scheduler.step()` 对 ReduceLROnPlateau 缺 metrics 实参会 TypeError;
    #   lr 调度统一移到 main() 的 epoch 循环, 经 SchedulerProtocol.step_scheduler 按类型分派
    #   (plateau 传 val_loss, cosine/steplr 无参), 与 AT/TriTrackNet 一致。此处不再 step。
    return float((loss_sum / nb).item())


# ============================================================
# 名称: evaluate_scores
# 修改原因: 需要将 TranAD 重建误差统一转为测试异常分数。
# 作用: 对一个时间段输出平均 loss 和点级异常分数。
# 数学原理: score_t = mean_d((x_td - z_td)^2)。
# 执行流程:
#   1. 按顺序批量前向;
#   2. 使用第二阶段输出 z2;
#   3. 计算每变量平方误差;
#   4. 对变量维求平均得到 score。
# 科研标准: 评价阶段 torch.no_grad,不更新模型参数。
# ============================================================
def evaluate_scores(
    model: nn.Module,
    windows: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    r"""
    科研注释：函数 `evaluate_scores`
    名称作用：在不更新模型参数的条件下，把 TranAD 第二阶段重构残差转为点级异常分数。
    参数说明：`model` 为冻结评价模型；`windows` 是时间窗口；`batch_size`、`device` 控制推理执行。
    返回值：返回 `(mean_loss, score, per_channel)`，其中 `score` 按时间点排列为 (T,) 跨通道均值,
            `per_channel` 为 (T,D) 逐通道残差供 HitRate@k 诊断。
    数学原理：$score_t=\frac{1}{D}\sum_d(x_{td}-z_{td})^2$，分数越高表示越偏离训练到的正常模式。
    流程说明：顺序批量前向 -> 取第二阶段输出 -> 计算通道平方残差 -> 按通道均值得分 + 保留逐通道残差。
    SCADA 迁移：高残差需要与实际故障区间对齐，不能仅凭温度幅值偏高断定部件故障。
    """
    criterion = nn.MSELoss(reduction="none")
    # 基线借鉴 #9: 推理 DataLoader 同样开 pin_memory + non_blocking
    pin = (device.type == "cuda")
    # 速度优化 #4 (2026-06-03): 同 train_one_epoch, Windows 大 TensorDataset 多 worker 共享内存
    #   会崩 (error 1455), 按 FALLBACK 回退为 num_workers=0。
    loader = make_window_loader(
        windows,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin,
        num_workers=0,
    )
    scores = []
    # ============================================================
    # 名称: per_channel_scores 收集
    # 修改原因 (2026-05-30): 原 evaluate_scores 只返回跨通道均值 score(T,),
    #          丢弃了逐通道残差,无法对接 diagnosis.py 做 HitRate@k 根因排序。
    #          三模型中仅 TranAD 具备诊断能力——保留 (T,D) 是此能力的唯一载体。
    # 作用: 收集每个 batch 的逐通道平方误差 (B,D),concat 为 (T,D)。
    # 数学原理: per_channel[t,d] = (x_{td} - z2_{td})², 第 d 列=第 d 个温度通道的残差时序。
    # 执行流程: append → concat → 与 score 同长度返回。
    # 科研标准: 不参与训练/阈值选择,纯后处理产出行,落盘供诊断脚本只读。
    # ============================================================
    per_channel_scores = []   # (B,D) per batch → concat → (T,D) 逐通道残差
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)  # 基线借鉴 #9
            local_batch = batch.shape[0]
            window = batch.permute(1, 0, 2)
            target = window[-1, :, :].view(1, local_batch, -1)
            _, z2 = model(window, target)
            residual = criterion(z2, target)[0]         # (B, D) 逐通道平方误差
            # 保留逐通道残差供诊断 (2026-05-30): 三模型中仅 TranAD 可回答"哪根轴承先异常"
            per_channel_scores.append(residual.detach().cpu().numpy())
            # #E3 修复 (2026-05-31): channel-max替代channel-mean: 风机故障常现1-2通道，均值稀释信号13-27x
            scores.append(torch.max(residual, dim=1).values.detach().cpu().numpy())   # 跨通道最大值 → 异常分数
    score = np.concatenate(scores)
    # (T, D): 每列 = 一个温度通道的残差时序, 可用于 diagnosis.py HitRate@k / NDCG@k
    per_channel = np.concatenate(per_channel_scores)
    return float(np.mean(score)), score, per_channel


# ============================================================
# 名称: save_checkpoint / restore_checkpoint
# 修改原因: IDE 训练中断后需要从已有模型状态继续,而不是丢弃已计算结果。
# 作用: 保存和加载网络、优化器、调度器、epoch。
# 数学原理: 无;保存优化轨迹状态。
# 执行流程: torch.save 写入字典;存在时 torch.load 并恢复状态。
# 科研标准: checkpoint 含 epoch 和优化器状态,保证续跑口径可说明。
# ============================================================
def save_checkpoint(model, optimizer, scheduler, epoch: int, checkpoint: Path,
                    checkpoint_identity: Dict[str, object] | None = None) -> None:
    r"""
    科研注释：函数 `save_checkpoint`
    名称作用：保存模型参数、优化器状态、调度器状态和完成 epoch，支持中断后续跑。
    参数说明：`model`、`optimizer`、`scheduler` 为当前训练状态；`epoch` 为已完成轮次编号。
    返回值：无显式返回值；副作用是写入 checkpoint 文件。
    数学原理：不新增模型计算，仅持久化优化轨迹以维持重复实验的一致起点。
    流程说明：确保目录存在 -> 打包状态字典 -> 调用 `torch.save`。
    SCADA 迁移：保存续跑状态时应同步记录所用机组、变量子集和时间切分版本。
    """
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "checkpoint_identity": checkpoint_identity,
        },
        checkpoint,
    )


def restore_checkpoint(model, optimizer, scheduler, device: torch.device, checkpoint: Path,
                       expected_identity: Dict[str, object] | None = None) -> int:
    r"""
    科研注释：函数 `restore_checkpoint`
    名称作用：恢复既有 TranAD 训练状态并返回最后完成的 epoch。
    参数说明：`model`、`optimizer`、`scheduler` 为待恢复对象；`device` 指定加载设备。
    返回值：checkpoint 不存在时返回 `-1`，否则返回保存的 epoch 编号。
    数学原理：不改变损失定义；恢复参数使续跑延续相同优化轨迹。
    流程说明：检查文件 -> 映射设备加载 -> 恢复三类状态 -> 返回轮次。
    SCADA 迁移：只有数据版本和特征通道一致时 checkpoint 才具可比性。
    """
    if not checkpoint.exists():
        return -1
    payload = torch.load(checkpoint, map_location=device)
    if not checkpoint_identity_is_compatible(
        payload,
        expected_identity,
        model="TranAD",
        checkpoint_path=checkpoint,
    ):
        return -1
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    return int(payload["epoch"])


# ============================================================
# 名称: main
# 修改原因: 提供统一启动器可调用的 TranAD SCADA 实验入口。
# 作用: 执行训练/续跑/验证/测试和指标落盘。
# 数学原理: 使用重建误差异常分数及验证集阈值。
# 执行流程:
#   1. 导入原 TranAD 类并加载数据;
#   2. 构造模型与窗口;
#   3. 恢复或重新训练;
#   4. 每 epoch 输出 train/val METRIC;
#   5. 输出最终 test METRIC。
# 科研标准: 正式和 smoke 运行均留存结果,smoke 不解释为正式性能。
# ============================================================
def main() -> int:
    r"""
    科研注释：函数 `main`
    名称作用：编排 TranAD 的 SCADA 数据读取、窗口化、训练/续跑、验证阈值选择和测试记录。
    Phase D 修改：支持 --module tcn_input_residual TCN 包装;使用 polarity-aware 阈值。
    """
    args = parse_args()
    # Phase D: 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_kind = "smoke" if args.smoke else "formal"
    # 2026-06-02 split/feature 版本隔离: checkpoint 与数据目录均按 split 隔离,
    #   narrow_v1 (默认) 保持历史路径, chronological_v2 写/读版本化路径。
    checkpoint = checkpoint_path(args.smoke, args.module, args.seed, args.farm,
                                 args.split_id, args.feature_version, args.preprocess_variant)
    arrays = load_arrays(args.smoke, args.farm, args.split_id, args.feature_version,
                         args.preprocess_variant)
    TranAD = import_tranad_model()
    # ============================================================
    # 名称: seed-isolation 修复 (2026-05-25 §10.4 / 关系图.docx §5.6 bug fix)
    # 修改原因: 原仓库 src/models.py 在文件顶层 (import 阶段) 执行 torch.manual_seed(1),
    #          时序上早于 实验.py 的 args 解析,所以 CLI --seed 完全无效;
    #          Phase D §7.7 实测三 seed 输出完全一致,无法报告 mean±std。
    # 作用: 在 import_tranad_model() 之后立即用 args.seed 重置 Python/numpy/torch 三套 RNG,
    #       然后再构造 model 实例,确保参数初始化随 --seed 真实变化。
    # 数学原理: 让 W ~ p(W | seed=args.seed) 而非 p(W | seed=1) 的退化分布。
    # 执行流程:
    #   1. random.seed → 控制 Python random;
    #   2. np.random.seed → 控制 sklearn 内部 / 数据 shuffle;
    #   3. torch.manual_seed → 控制 nn.init / Dropout / DataLoader generator;
    #   4. cuda.manual_seed_all → 多卡情形保险。
    # 科研标准: 多 seed 报告 mean±std 的硬前提;不修复就不能声称"多种子稳定性验证"。
    # ============================================================
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 速度优化 #1 (2026-06-03): fp64->fp32。RTX5090 fp64 吞吐约 fp32 的 1/30, 之前 GPU 仅 ~23% 利用率,
    #   主因即 .double()。用户已接受该精度变更换 ~10x 速度。windows/targets 均已 float32
    #   (见 convert_to_windows), 全 TranAD 路径 dtype 一致, 不改 loss/结构/窗口语义。
    model = TranAD(arrays["train"].shape[1]).float().to(device)
    meta = arrays.get("_meta", {})
    result_dir = result_dir_for_farm(args.farm, args.output_dir)
    # [E1/G3 2026-06-01] meta.json 已无 "subset" 键, 删除 meta.get("subset","T06_2021") 死引用。
    #   实验工具.CSV_FIELDS 仍含 "subset" 列 (共享文件, 不改), 故此处保留键名但取空串占位,
    #   不再伪造 T06_2021 机组标识。n_channels 由 shape[1] 动态取 (各 farm 不同)。
    common = {
        "farm": args.farm,
        "subset": "",
        "module": args.module,
        "seed": args.seed,
        "n_channels": int(arrays["train"].shape[1]),
        "input_shape": [int(arrays["train"].shape[0]), int(arrays["train"].shape[1])],
    }
    preprocess_identity = build_preprocess_identity(
        meta,
        split_id=args.split_id,
        feature_version=args.feature_version,
        preprocess_variant=getattr(args, "preprocess_variant", ""),
        n_channels=int(arrays["train"].shape[1]),
    )
    add_preprocess_identity_to_metrics(common, identity=preprocess_identity)

    # ============================================================
    # 名称: TCN 输入包装 (Phase D 模块 tcn_input_residual)
    # 修改原因: 在 model 实例化之后、optimizer 之前完成包装,确保 optimizer 看到 TCN 参数。
    # 作用: 用 TranADTCNWrapper 把原 TranAD 包成 (B,D,L) → TCN → (L,B,D) → TranAD.forward 的链路;
    #       baseline_only 时保持原模型不变。
    # 数学原理: X' = X + α·TCN(X), α 初始 0.1 (训练初期 ≈ baseline);TCN = 4 层膨胀因果卷积。
    # 执行流程: 先构 TranAD inner → if 模块 == tcn_input_residual 则用 wrapper 替换 model 引用。
    # 科研标准: TCN 包装是唯一改动,不修改 TranAD 双 decoder/对抗训练循环,保证 baseline vs +TCN 受控对比。
    # ============================================================
    # 基线借鉴 #8 (2026-05-26): 新增 tcn_wavelet_residual 分支用 TranADTCNWaveletWrapper
    if args.module in ("tcn_input_residual", "tcn_wavelet_residual"):
        # 2026-05-30 TCN-IO: D 一致性闸门(本模型自检 + 跨模型) + 有效感受野覆盖诊断
        from 实验配置 import TCNIOProtocol as _IO
        _IO.assert_input_channels("tranad", arrays["train"].shape[1], meta.get("n_channels"))
        print(_IO.coverage_banner("tranad", int(arrays["train"].shape[1])), flush=True)
        if args.module == "tcn_input_residual":
            from modules.tcn_增强 import TranADTCNWrapper
            # 速度优化 #1 (2026-06-03): fp64->fp32 (同 baseline 路径)
            model = TranADTCNWrapper(model, input_channels=arrays["train"].shape[1]).float().to(device)
        else:  # tcn_wavelet_residual — 基线借鉴 #8
            from modules.tcn_增强 import TranADTCNWaveletWrapper
            # 速度优化 #1 (2026-06-03): fp64->fp32 (同 baseline 路径)
            model = TranADTCNWaveletWrapper(model, input_channels=arrays["train"].shape[1]).float().to(device)

    # ============================================================
    # 名称: Wrapper 属性透传 (model.model.lr / model.model.n_window)
    # 修改原因: TranADTCNWrapper 是组合 (self.model = inner TranAD),wrapper 自身没有 .lr / .n_window;
    #          原始路径访问 model.lr 在 wrapper 模式下会 AttributeError。
    # 作用: 用三元表达式按 args.module 切换 .model.lr / .lr,在 wrapper 模式下走 inner,
    #       baseline 模式直接走 model。
    # 数学原理: 无 (属性路径选择)。
    # 科研标准: lr / n_window 由原 TranAD 类决定,不被 TCN 覆盖;保证两模式优化器和窗口长度一致。
    # ============================================================
    # ============================================================
    # 名称: TCN 差分学习率优化器 (2026-05-30 最优超参数)
    # 修改原因: 旧代码传扁平 model.parameters() 给 AdamW,TCN 卷积和 α 门控混在 base 组
    #          共享 base lr=1e-4 和 base wd=1e-5;wd 衰减 α→0 反向压制 TCN。
    # 作用: 用 TCNProtocol.optimizer_param_groups() 拆为三组独立 lr/wd:
    #          base 模型: lr=1e-4, wd=1e-5 (论文不变)
    #          TCN 卷积+LN: lr=1e-3, wd=0 (10×base;dropout+weight_norm 已够正则)
    #          α 门控: lr=1e-2, wd=0 (100×base;单标量需高 lr 快速决断,wd 不能衰减 α)
    # 数学: StepLR(5,0.9) 对所有组等比乘 γ,天然保留组间比例,无需额外修改。
    # ============================================================
    from 实验配置 import TCNProtocol
    base_lr = model.model.lr if args.module in ("tcn_input_residual", "tcn_wavelet_residual") else model.lr
    # ============================================================
    # 名称: 超参数注入 lr (2026-05-31 超参数寻优)
    # 修改原因: 网格搜索需要从 CLI --lr 覆盖 model.lr; 若提供了该参数,
    #   同步更新模型属性 (model.lr / model.model.lr), 保证 loss 公式中 n 的 lr 与优化器一致。
    # ============================================================
    if args.lr is not None:
        base_lr = float(args.lr)
        inner = model.model if args.module in ("tcn_input_residual", "tcn_wavelet_residual") else model
        inner.lr = base_lr
    optimizer = torch.optim.AdamW(
        TCNProtocol.optimizer_param_groups(model, base_lr=base_lr, base_wd=1e-5)
    )
    # ============================================================
    # 名称: scheduler 构建 (2026-05-31 超参数寻优)
    # 修改原因: 原硬编码 StepLR(5, 0.9), 现替换为 SchedulerProtocol 工厂,
    #   支持 CosineAnnealingLR / ReduceLROnPlateau / StepLR。
    #   --scheduler 不给时默认 steplr (保持旧行为)。
    # ============================================================
    from 实验配置 import SchedulerProtocol
    _sc_type = args.scheduler if args.scheduler else "steplr"
    scheduler = SchedulerProtocol.build_scheduler(
        _sc_type, optimizer,
        T_max=args.epochs, base_lr=base_lr,
    )
    print(
        f"SCHEDULER|model=TranAD|type={_sc_type}|"
        f"display={SchedulerProtocol.display_name(scheduler)}|"
        f"lr={base_lr}|epochs={args.epochs}",
        flush=True,
    )

    # [G3 2026-06-01] 已删除 `if args.pretrain: raise NotImplementedError(...)` 死分支
    #   (--pretrain 占位参数已移除)。功能性的 --pretrain-ckpt 权重加载保留如下。

    # ============================================================
    # 基线借鉴 #6: 加载预训练 ckpt (在 force/resume 判断前, 不与续传冲突)
    # 修改原因 (2026-05-26): 先加载预训练 → 微调阶段从该权重开始;
    #     若同时存在 args.resume + 微调 ckpt, 微调 ckpt 优先 (restore_checkpoint 后覆盖)。
    # ============================================================
    if args.pretrain_ckpt:
        pretrain_path = Path(args.pretrain_ckpt)
        if pretrain_path.exists():
            state = torch.load(pretrain_path, map_location="cpu")
            inner = model.model if args.module in ("tcn_input_residual", "tcn_wavelet_residual") else model
            missing, unexpected = inner.load_state_dict(state, strict=False)
            print(
                f"BORROW#6|TranAD|pretrain_loaded|path={pretrain_path}|"
                f"missing={len(missing)}|unexpected={len(unexpected)}",
                flush=True,
            )
        else:
            print(f"BORROW#6|TranAD|pretrain_skip|path_not_exist={pretrain_path}", flush=True)
    if args.force and checkpoint.exists():
        checkpoint.unlink()

    # 同 lr 的 wrapper 透传: model.model.n_window vs model.n_window
    # 基线借鉴 #8: tcn_wavelet_residual 同样是 wrapper, 透传 inner.n_window
    _n_window = model.model.n_window if args.module in ("tcn_input_residual", "tcn_wavelet_residual") else model.n_window
    checkpoint_identity = build_checkpoint_identity(
        model="TranAD",
        farm=args.farm,
        module=args.module,
        seed=args.seed,
        input_shape=[int(_n_window), int(arrays["train"].shape[1])],
        preprocess_identity=preprocess_identity,
    )
    last_epoch = restore_checkpoint(
        model, optimizer, scheduler, device, checkpoint,
        expected_identity=checkpoint_identity,
    ) if args.resume else -1
    window_device = select_window_device(arrays, _n_window, device)
    train_w = convert_to_windows(arrays["train"], _n_window, device=window_device)
    val_w = convert_to_windows(arrays["val"], _n_window, device=window_device)
    test_w = convert_to_windows(arrays["test"], _n_window, device=window_device)
    epochs = 1 if args.smoke else args.epochs

    if checkpoint.exists() and args.resume and not args.force and last_epoch >= epochs - 1:
        print(f"RESUME|model=TranAD|checkpoint={checkpoint}|action=skip_train", flush=True)
    else:
        start_epoch = last_epoch + 1 if args.resume else 0
        # wandb 离线模式: 训练前初始化
        _wandb_ok = init_wandb_run(
            "TranAD", args.farm, args.module, args.seed,
            epochs=epochs, batch_size=args.batch_size,
        )
        for epoch in range(start_epoch, epochs):
            train_loss = train_one_epoch(model, train_w, optimizer, scheduler, epoch, args.batch_size, device)
            # [方法2 提速 2026-06-06] SCADA_GRID_FAST=1: 跳过每-epoch 的【train 打分】(昂贵的 train 前向)。
            #   依据: train 指标无意义(train_positive=0 → F1 恒 0); train_score 只作 choose_threshold 的
            #   "val 无双类时回退", 而 grid 的 val 有正例 → 不触发。val 打分/早停/选阈值/最终 test 全不变。
            #   默认关 → 90-run / 普通跑【逐位不变】; 仅 grid 显式开。
            _grid_fast = os.environ.get("SCADA_GRID_FAST") == "1"
            from 实验配置 import SchedulerProtocol as _SchedProto
            # 提速 (2026-06-18, 不重启即生效): per-epoch val 打分【仅日志】当 scheduler 非 plateau。
            #   cosine/steplr 按 epoch 步进、不吃 val_loss; 无早停/无 best-val/用末 epoch → 中间 epoch 的
            #   val 评分(evaluate_scores, 昂贵)不影响训练/最终 test。本矩阵 best_config=cosine。
            #   SCADA_SKIP_EPOCH_EVAL=1 且非 grid 且非 plateau → 跳, 只保 checkpoint + scheduler.step。逐位不变。
            skip_val_eval = should_skip_epoch_eval() and args.scheduler != "plateau"
            if not skip_val_eval:
                if not _grid_fast:
                    _rng_state = _rng_state_snapshot()
                    try:
                        _, train_score, _ = evaluate_scores(model, train_w, args.batch_size, device)
                    finally:
                        _restore_rng_state(_rng_state)
                val_loss, val_score, _ = evaluate_scores(model, val_w, args.batch_size, device)
                _SchedProto.step_scheduler(scheduler, val_loss=val_loss)
                # off-by-one 对齐: scores[i] 对应 t=i-1, 与 labels[i-1] 配对 (阈值/极性只用 val)。
                val_score, val_labels = align_scores_to_labels(val_score, arrays["val_labels"])
                if not _grid_fast:
                    train_score, train_labels = align_scores_to_labels(train_score, arrays["train_labels"])
                threshold, source, polarity = choose_threshold_and_polarity_by_validation(
                    val_labels, val_score, (train_score if not _grid_fast else val_score)
                )
                val_score_o = orient_scores(val_score, polarity)
                val_metrics = compute_binary_metrics(val_labels, scores=val_score_o, threshold=threshold)
                val_metrics.update({"run_kind": run_kind, "loss": val_loss, "threshold": threshold,
                                    "threshold_source": source, "score_polarity": polarity,
                                    **common})
                if not _grid_fast:
                    train_score_o = orient_scores(train_score, polarity)
                    train_metrics = compute_binary_metrics(train_labels, scores=train_score_o, threshold=threshold)
                    train_metrics.update({"run_kind": run_kind, "loss": train_loss, "threshold": threshold,
                                          "threshold_source": source, "score_polarity": polarity,
                                          **common})
                    record_and_print_metric(result_dir / "metrics.jsonl", CSV_PATH, "TranAD", "train", epoch + 1, train_metrics)
                    _log_wandb_epoch("train", epoch, train_metrics)
                record_and_print_metric(result_dir / "metrics.jsonl", CSV_PATH, "TranAD", "val", epoch + 1, val_metrics)
                _log_wandb_epoch("val", epoch, val_metrics)
            else:
                _SchedProto.step_scheduler(scheduler)   # cosine/steplr: 按 epoch 步进, 不需 val_loss
                print(f"  [TranAD] epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} "
                      f"(skip per-epoch eval; scheduler={args.scheduler})", flush=True)
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint, checkpoint_identity)
        if _wandb_ok:
            finish_wandb_run()

    _, train_score, _ = evaluate_scores(model, train_w, args.batch_size, device)
    _, val_score, per_channel_val = evaluate_scores(model, val_w, args.batch_size, device)
    test_loss, test_score, per_channel_test = evaluate_scores(model, test_w, args.batch_size, device)
    # [C4 2026-06-01] 最终评测同样做 off-by-one 对齐 (scores[1:] / labels[:-1])。
    #   per_channel 与 score 同序同长 (同一 evaluate_scores 循环产出), 故同步丢首 [1:]
    #   以保持与对齐后 labels 等长; 落盘 scores/labels/per_channel 三者长度一致 = T-1。
    train_score, _train_labels = align_scores_to_labels(train_score, arrays["train_labels"])
    val_score, val_labels = align_scores_to_labels(val_score, arrays["val_labels"])
    test_score, test_labels = align_scores_to_labels(test_score, arrays["test_labels"])
    per_channel_test = np.asarray(per_channel_test)[1:]

    threshold, source, polarity = choose_threshold_and_polarity_by_validation(
        val_labels, val_score, train_score
    )
    test_score_o = orient_scores(test_score, polarity)
    test_metrics = compute_binary_metrics(test_labels, scores=test_score_o, threshold=threshold)
    test_metrics.update({"run_kind": run_kind, "loss": test_loss, "threshold": threshold,
                         "threshold_source": source, "score_polarity": polarity,
                         **common})
    record_and_print_metric(result_dir / "metrics.jsonl", CSV_PATH, "TranAD", "test", "final", test_metrics)
    # ============================================================
    # 基线借鉴 #4: 落盘 train/val/test scores + labels 供 集成评价.py 读取
    # 修改原因 (2026-05-26): 同 Anomaly-Transformer-main/实验.py 落盘策略,
    #     按 (baseline, module, seed) 命名保证集成评价器能 glob 出来。
    # 修改原因 (2026-06-01 Task6): 新增 train scores 落盘; 路径经 ResultLayout 版本化。
    # 文件名约定: {train|val|test}_{scores|labels}__{module}__seed{seed}.npy
    # [C4 2026-06-01] 落盘的是【已对齐】的 scores/labels (长度 T-1), 集成评价器读到的即对齐口径。
    # ============================================================
    from 实验配置 import ResultLayout
    _split_id = getattr(args, "split_id", "narrow_v1")
    _fv = getattr(args, "feature_version", "v1")
    if _split_id != "narrow_v1":
        scores_dir = ResultLayout.scores_dir(
            _split_id, _fv, args.farm, "tranad",
            preprocess_variant=getattr(args, "preprocess_variant", ""),
        )
    else:
        scores_dir = result_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    val_score_o = orient_scores(val_score, polarity)
    # train_score has already been align_scores_to_labels'd; orient it for saving
    train_score_o = orient_scores(train_score, polarity)
    suffix = f"__{args.module}__seed{args.seed}"
    # [E1/契约 2026-06-01] 契约标签为 1D int64 ∈ {-1,0,1}; 保留对 (T,D) 旧布局的 OR 聚合兜底
    #   (任一维为 1 即异常), 1D 时直接取整。聚合后再与已对齐的 scores 同口径落盘。
    _train_labels_arr = np.asarray(_train_labels)
    train_labels_1d = (_train_labels_arr == 1).any(axis=1).astype(np.int8) if _train_labels_arr.ndim == 2 else _train_labels_arr.astype(np.int8)
    val_labels_1d = (np.asarray(val_labels) == 1).any(axis=1).astype(np.int8) if val_labels.ndim == 2 else np.asarray(val_labels).astype(np.int8)
    test_labels_1d = (np.asarray(test_labels) == 1).any(axis=1).astype(np.int8) if test_labels.ndim == 2 else np.asarray(test_labels).astype(np.int8)
    np.save(scores_dir / f"train_scores{suffix}.npy", train_score_o.astype(np.float32))
    np.save(scores_dir / f"train_labels{suffix}.npy", train_labels_1d.reshape(-1))
    np.save(scores_dir / f"val_scores{suffix}.npy",  val_score_o.astype(np.float32))
    np.save(scores_dir / f"val_labels{suffix}.npy",  val_labels_1d.reshape(-1))
    np.save(scores_dir / f"test_scores{suffix}.npy", test_score_o.astype(np.float32))
    np.save(scores_dir / f"test_labels{suffix}.npy", test_labels_1d.reshape(-1))
    # ============================================================
    # 名称: 逐通道残差落盘 (2026-05-30)
    # 修改原因: evaluate_scores 新增 per_channel 返回值后,需同步落盘到 scores/ 目录,
    #          供 diagnosis.py 在实验后读取做 HitRate@k 根因诊断,无需重训模型。
    # 作用: 保存 test 集的 per_channel (T-1,D) 逐通道残差 (已 C4 对齐),文件命名与 scores/labels 一致。
    # 数学原理: per_channel[t,d] = (x_{td} - z2_{td})²,可用于按通道聚合排名。
    # 执行流程: per_channel_test 是 evaluate_scores 返回的第三元, 经 [1:] 对齐后写 .npy。
    # 科研标准: 只读产物,不参与阈值选择和训练; test 仅最终一次落盘。
    # ============================================================
    np.save(scores_dir / f"test_per_channel{suffix}.npy", per_channel_test.astype(np.float32))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
