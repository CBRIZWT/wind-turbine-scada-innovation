# -*- coding: utf-8 -*-
"""实验.py — WT-Transformer SCADA 统一实验入口 (S0 整合, 2026-06-07)

把 wt-transformer 接入项目统一矩阵, 与 AT/TranAD/TriTrackNet 同契约:
  · 读 PerFarmPaths 处理产物 (train/val/test.npy + *_labels.npy + meta.json);
  · wt-transformer 是【预测式(forecasting)】无监督模型: 用窗口预测目标温度通道下一步值,
    异常分数 = positive_residual_energy = max(0, 实际−预测)^2 ("比预测更热"=过温);
  · train 段去污(正例=0)即可 — 预测范式只需学正常行为, 与 AT/TranAD/TriTrackNet 同;
  · 阈值/极性仅用 val 选 (choose_threshold_and_polarity_by_validation), test 只最终评一次;
  · 逐 epoch + 最终 test 走 record_and_print_metric 写 metrics.jsonl/csv; scores 落 ResultLayout。

模块 (与项目变体系统对齐):
  · baseline_only       : 原 wt-transformer;
  · tcn_input_residual  : 输入级 α 门控 TCN 残差增强 (复用 SCADATCNResidualAdapter, 同三模型范式);
  · kalman_score_smooth : 输出分数级 Kalman 平滑 (复用 卡尔曼滤波.kalman_filter_1d, S0-A 降噪模块)。

科研标准: 不用 test 调阈值; label=-1 剔除; 防泄漏边界与三模型一致。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for _p in (str(SCRIPT_DIR), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Src.model.transformer_torch import WTTransformerTorch  # noqa: E402
from Src.data.scada_npy_dataset import (  # noqa: E402
    ScadaNpyWindowDataset, load_meta, target_index_from_meta,
)
from 实验工具 import (  # noqa: E402
    add_preprocess_identity_to_metrics,
    build_preprocess_identity,
    choose_threshold_and_polarity_by_validation, compute_binary_metrics,
    augment_event_metrics, orient_scores, record_and_print_metric,
    init_wandb_run, log_epoch_to_wandb, finish_wandb_run, ensure_dir,
    should_skip_epoch_eval,
)

MODEL_NAME = "WTTransformer"
BASELINE = "wt_transformer"

# 窗口长度单一真源: DatasetProtocol.WIN_WT (缺则默认 144=24h, 与 scada_npy_dataset 默认一致)
try:
    from 实验配置 import DatasetProtocol as _DP
    _WIN_WT_DEFAULT = int(getattr(_DP, "WIN_WT", 144))
    _TRAIN_STRIDE = int(getattr(_DP, "TRAIN_STRIDE", 5))   # 训练抽稀 (与三模型同口径, 杜绝邻窗 99% 重叠 + 5× 提速)
except Exception:
    _WIN_WT_DEFAULT = 144
    _TRAIN_STRIDE = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WT-Transformer SCADA 实验入口 (统一契约)")
    p.add_argument("--farm", type=str, default="kelmarsh",
                   choices=["kelmarsh", "penmanshiel", "hill_of_towie"])
    p.add_argument("--module", type=str, default="baseline_only",
                   choices=["baseline_only", "tcn_input_residual", "kalman_score_smooth"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", type=str, default="run_001")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scheduler", type=str, default=None)
    p.add_argument("--split-id", type=str, default="narrow_v1")
    p.add_argument("--feature-version", type=str, default="v1")
    p.add_argument("--preprocess-variant", type=str, default="",
                   help="预处理变体后缀, 例如 old_preprocess / new_preprocess")
    p.add_argument("--win-size", type=int, default=_WIN_WT_DEFAULT,
                   help="滑窗长度 (默认取自 DatasetProtocol.WIN_WT)")
    p.add_argument("--smoke", action="store_true", help="小样本一轮链路验证")
    p.add_argument("--resume", action="store_true", help="兼容启动器断点续传参数; WT 当前无训练 checkpoint, 仅忽略")
    p.add_argument("--force", action="store_true", help="兼容启动器强制重跑参数; WT 当前无训练 checkpoint, 仅忽略")
    return p.parse_args()


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _data_dir(farm: str, split_id: str, fv: str, preprocess_variant: str | None = None) -> Path:
    from 实验配置 import PerFarmPaths
    return Path(PerFarmPaths.for_farm(
        farm, split_id, fv, preprocess_variant=preprocess_variant,
    )[BASELINE])


def _scores_dir(split_id: str, fv: str, farm: str, result_dir: Path,
                preprocess_variant: str | None = None) -> Path:
    if str(split_id) != "narrow_v1":
        from 实验配置 import ResultLayout
        return ResultLayout.scores_dir(split_id, fv, farm, BASELINE,
                                       preprocess_variant=preprocess_variant)
    return result_dir / "scores"


class _WTTCNWrapper(nn.Module):
    """tcn_input_residual: 输入级 α 门控 TCN 残差增强 (复用 SCADATCNResidualAdapter)。

    wt 输入 (B, L, D) → permute (B, D, L) → adapter(X+α·LN(TCN(X))) → permute 回 (B, L, D) → wt。
    与 TranADTCNWrapper 同一适配器/超参 (TCNProtocol), 参数名含 'tcn_adapter'/'_alpha_raw'。
    """

    def __init__(self, model: nn.Module, input_channels: int) -> None:
        super().__init__()
        _tcn_root = ROOT / "TCN-master"
        if str(_tcn_root) not in sys.path:
            sys.path.insert(0, str(_tcn_root))
        from TCN.scada_adapter import SCADATCNResidualAdapter
        self.model = model
        self.tcn_adapter = SCADATCNResidualAdapter.from_protocol(input_channels=input_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.permute(0, 2, 1).contiguous()          # (B, D, L)
        x = self.tcn_adapter(x).permute(0, 2, 1).contiguous()  # (B, L, D)
        return self.model(x)


def _build_model(module: str, input_dim: int, seq_len: int) -> nn.Module:
    # P1-fix#2 (2026-06-10): out_dim=D 全通道预测。旧单目标(齿轮油温)对 union 标签的另 17 通道
    #   事件全盲, 且正残差分数塌缩为 0 → val 选阈退化为全正预测 (实测 threshold=-0.0, tn=0)。
    base = WTTransformerTorch(input_dim=input_dim, seq_len=seq_len, out_dim=input_dim)
    if module == "tcn_input_residual":
        return _WTTCNWrapper(base, input_channels=input_dim)
    return base


def _forecast_scores(model: nn.Module, data: np.ndarray, labels: np.ndarray,
                     seq_len: int, target_index: int, batch_size: int,
                     device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    """跑一遍某段, 返回 (逐窗口异常分数 positive_residual_energy, 对齐标签, mse)。

    分数对齐到被预测点 stop=i+seq_len, 标签取 labels[seq_len:] (与三模型落盘口径一致)。
    """
    ds = _make_dataset(data, labels, seq_len, target_index)   # 评分恒 step=1 (密采样, 对齐 labels[seq_len:])
    dl = DataLoader(ds, batch_size=max(int(batch_size), 512), shuffle=False)   # 无梯度→大 batch 提速
    model.eval()
    scores, sq_sum, n = [], 0.0, 0
    with torch.no_grad():
        for x, y, _lab in dl:
            x = x.to(device, dtype=torch.float32)
            pred = model(x).detach().float().cpu().numpy()    # (B, D) 全通道预测
            y = y.numpy()                                      # (B, D)
            resid = y - pred
            # P1-fix#2: 全通道新息正能量 = mean_i max(0, resid_i)^2 (与已验证 bar 分数同形式;
            #   旧单通道分数大面积塌缩为 0 → 阈值退化全正预测)
            scores.append(np.mean(np.maximum(0.0, resid) ** 2, axis=1))
            sq_sum += float(np.sum(resid ** 2)); n += resid.size
    s = np.concatenate(scores) if scores else np.zeros(0, dtype=float)
    lab_aligned = np.asarray(labels[seq_len:seq_len + len(s)]).reshape(-1)
    mse = sq_sum / n if n else float("nan")
    return s.astype(np.float64), lab_aligned, mse


class _ArrayWindowDataset(torch.utils.data.Dataset):
    """与 ScadaNpyWindowDataset 同语义, 但直接吃内存数组 (避免重复落盘读)。"""

    def __init__(self, data: np.ndarray, labels: np.ndarray, n_steps: int, target_index: int, step: int = 1):
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.labels = np.asarray(labels)
        self.n = int(n_steps); self.ti = int(target_index); self.step = max(1, int(step))

    def __len__(self) -> int:
        return int((self.data.shape[0] - self.n - 1) // self.step + 1)

    def __getitem__(self, i: int):
        s = int(i) * self.step
        x = self.data[s:s + self.n]
        # P1-fix#2: 目标 = 下一步【全 D 通道】向量 (旧: 仅 self.ti 单通道 → 17 通道事件全盲)
        y = self.data[s + self.n, :]
        lab = np.int64(self.labels[s + self.n])
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), torch.tensor(lab)


def _make_dataset(data, labels, seq_len, target_index, step: int = 1):
    return _ArrayWindowDataset(data, labels, seq_len, target_index, step=step)


def _maybe_kalman_smooth(module: str, *score_arrays: np.ndarray):
    """kalman_score_smooth: 对各段 1D 异常分数做 Kalman 平滑 (复用项目 KF, 压抖动减碎片误报)。"""
    if module != "kalman_score_smooth":
        return score_arrays
    from 卡尔曼滤波.卡尔曼滤波 import kalman_filter_1d
    return tuple(kalman_filter_1d(np.asarray(s, dtype=np.float32),
                                  process_var=1e-4, measurement_var=1e-2).astype(np.float64)
                 for s in score_arrays)


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = _data_dir(args.farm, args.split_id, args.feature_version, args.preprocess_variant)
    meta = load_meta(data_dir)
    n_channels = int(meta.get("n_channels", 0)) or int(np.load(data_dir / "train.npy", mmap_mode="r").shape[1])
    target_index = target_index_from_meta(meta)
    seq_len = int(args.win_size)

    train = np.load(data_dir / "train.npy"); train_lab = np.load(data_dir / "train_labels.npy")
    val = np.load(data_dir / "val.npy");     val_lab = np.load(data_dir / "val_labels.npy")
    test = np.load(data_dir / "test.npy");   test_lab = np.load(data_dir / "test_labels.npy")
    if args.smoke:   # 链路验证: 截断到小样本
        train, train_lab = train[: seq_len + 2000], train_lab[: seq_len + 2000]
        val, val_lab = val[: seq_len + 1000], val_lab[: seq_len + 1000]
        test, test_lab = test[: seq_len + 1000], test_lab[: seq_len + 1000]
        args.epochs = 1

    result_dir = ensure_dir(Path(args.output_dir) if args.output_dir
                            else ROOT / "实验结果" / args.farm / BASELINE)
    jsonl = result_dir / "metrics.jsonl"
    csv_path = ROOT / "实验结果" / "metrics.csv"

    model = _build_model(args.module, n_channels, seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    loss_fn = nn.MSELoss()
    train_dl = DataLoader(_make_dataset(train, train_lab, seq_len, target_index, step=_TRAIN_STRIDE),
                          batch_size=int(args.batch_size), shuffle=True, drop_last=True)

    init_wandb_run(MODEL_NAME, args.farm, args.module, args.seed, args.epochs, args.batch_size)
    common = {"farm": args.farm, "module": args.module, "seed": args.seed, "subset": "SCADA",
              "n_channels": n_channels, "input_shape": [seq_len, n_channels],
              "split_id": args.split_id, "feature_version": args.feature_version,
              "run_kind": "smoke" if args.smoke else "formal"}
    add_preprocess_identity_to_metrics(
        common,
        identity=build_preprocess_identity(
            meta,
            split_id=args.split_id,
            feature_version=args.feature_version,
            preprocess_variant=getattr(args, "preprocess_variant", ""),
            n_channels=n_channels,
        ),
    )

    # 提速 Approach A: 矩阵(非grid)跳每-epoch【仅日志】val 打分, 只保留最后 epoch 打分 → 结果逐位不变。
    skip_epoch_eval = should_skip_epoch_eval()
    for epoch in range(1, int(args.epochs) + 1):
        model.train(); ep_loss, nb = 0.0, 0
        for x, y, _lab in train_dl:
            x = x.to(device, dtype=torch.float32); y = y.to(device, dtype=torch.float32)
            opt.zero_grad(); pred = model(x); loss = loss_fn(pred, y)
            loss.backward(); opt.step()
            ep_loss += float(loss.item()); nb += 1
        if skip_epoch_eval:   # 跳仅日志打分, 仍打 train loss 保留 loss 曲线
            print(f"  [{MODEL_NAME}] epoch {epoch}/{args.epochs} train_loss={ep_loss/max(nb,1):.4f} "
                  f"(skip per-epoch eval)", flush=True)
            continue
        # —— val 打分 + 选阈值(仅 val) + 指标 ——
        val_s, val_lab_a, val_mse = _forecast_scores(model, val, val_lab, seq_len, target_index,
                                                      int(args.batch_size), device)
        (val_s,) = _maybe_kalman_smooth(args.module, val_s)
        thr, thr_src, pol = choose_threshold_and_polarity_by_validation(val_lab_a, val_s, val_s)
        vo = orient_scores(val_s, "positive" if str(pol).startswith("positive") else "negative")
        vm = compute_binary_metrics(val_lab_a, scores=vo, threshold=thr)
        vm = augment_event_metrics(vm, labels=val_lab_a, preds=(vo > thr).astype(int))
        vm.update(common); vm.update({"loss": ep_loss / max(nb, 1), "mse": val_mse,
                                      "threshold": float(thr), "threshold_source": thr_src,
                                      "score_polarity": pol, "score_definition": "allchannel_innovation_pos_energy"})
        record_and_print_metric(jsonl, csv_path, MODEL_NAME, "val", epoch, vm)
        log_epoch_to_wandb("val", epoch, vm)

    # —— 最终 test: 用 val 选定阈值/极性, test 只评一次 ——
    val_s, val_lab_a, _ = _forecast_scores(model, val, val_lab, seq_len, target_index, int(args.batch_size), device)
    test_s, test_lab_a, test_mse = _forecast_scores(model, test, test_lab, seq_len, target_index, int(args.batch_size), device)
    train_s, train_lab_a, _ = _forecast_scores(model, train, train_lab, seq_len, target_index, int(args.batch_size), device)
    val_s, test_s, train_s = _maybe_kalman_smooth(args.module, val_s, test_s, train_s)
    thr, thr_src, pol = choose_threshold_and_polarity_by_validation(val_lab_a, val_s, train_s)
    base_pol = "positive" if str(pol).startswith("positive") else "negative"
    to = orient_scores(test_s, base_pol)
    tm = compute_binary_metrics(test_lab_a, scores=to, threshold=thr)
    tm = augment_event_metrics(tm, labels=test_lab_a, preds=(to > thr).astype(int))
    tm.update(common); tm.update({"loss": test_mse, "mse": test_mse, "threshold": float(thr),
                                  "threshold_source": thr_src, "score_polarity": pol,
                                  "score_definition": "allchannel_innovation_pos_energy"})
    record_and_print_metric(jsonl, csv_path, MODEL_NAME, "test", "final", tm)

    # —— 落盘 scores (统一命名: {split}_scores__{module}__seed{seed}.npy) ——
    sdir = ensure_dir(_scores_dir(
        args.split_id, args.feature_version, args.farm, result_dir, args.preprocess_variant,
    ))
    sfx = f"__{args.module}__seed{args.seed}"
    for nm, sc, lb in (("train", orient_scores(train_s, base_pol), train_lab_a),
                       ("val", orient_scores(val_s, base_pol), val_lab_a),
                       ("test", to, test_lab_a)):
        np.save(sdir / f"{nm}_scores{sfx}.npy", np.asarray(sc, dtype=np.float32))
        np.save(sdir / f"{nm}_labels{sfx}.npy", np.asarray(lb).reshape(-1))
    finish_wandb_run()
    print(f"[wt_transformer] done farm={args.farm} module={args.module} "
          f"test_f1={tm.get('f1')} affil_f1={tm.get('affiliation_f1')} auprc={tm.get('auprc')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
