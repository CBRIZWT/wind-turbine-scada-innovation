# MODULE_TCN.md — TCN 输入级残差增强模块

> **所属实验阶段**: Phase D (三 baseline 统一加模块对比)
> **模块代号**: `tcn_input_residual`
> **创建日期**: 2026-05-23
> **状态**: 已集成到 Anomaly Transformer / TranAD / TriTrackNet 三 baseline

---

## 1. 模块概述

TCN (Temporal Convolutional Network) 输入级残差增强模块在不修改各 baseline 核心模型结构的前提下，在输入端对多变量 SCADA 时间序列做膨胀因果卷积增强，形成 **X' = X + α·TCN(X)** 的残差连接，再送入原 baseline 模型。

### 1.1 模块角色

| 维度 | 说明 |
|---|---|
| 在 Phase D 中的角色 | `baseline_only` 的对比变体，验证"加输入级时序感受野扩展"能否提升轴承温度故障检测/预测性能 |
| 接入方式 | 输入级包装器 (wrapper)，不修改原模型 `__init__` 签名和 forward 内部计算图 |
| 适用 baseline | Anomaly Transformer / TranAD / TriTrackNet (三个均已适配) |

### 1.2 设计动机

- 三 baseline 的时间输入窗口长度各不相同 (AT=100, TranAD=36, TriTrackNet=96)，但都缺乏**显式的多尺度时序特征提取**；
  （2026-05-30 统一: 窗口大小唯一真源 = `实验配置.DatasetProtocol`；TranAD 由 10→36 以满足 TCN 感受野）；
- TCN 的膨胀因果卷积可以用少量参数覆盖较长历史感受野 (29 步 ≈ 4.83 小时 SCADA 数据)，精确匹配轴承热惯量级 1~6h；
- 残差连接 + sigmoid 门控 α (初始 0.01) 保证训练初期接近原 baseline 行为，由梯度自学决定是否放大 TCN 影响。

---

## 2. 算法基础

### 2.1 TCN 核心原理

TCN 由 Bai et al. (2018) 提出，核心特征：

1. **因果卷积 (Causal Convolution)**: 时刻 t 的输出只依赖 t 及之前的输入，不泄漏未来信息；
2. **膨胀卷积 (Dilated Convolution)**: 第 i 层膨胀因子 d = 2^i，使感受野随层数指数增长；
3. **残差连接 (Residual Connection)**: 每层 TemporalBlock 内包含 2 层膨胀卷积 + 残差连接，稳定深层训练。

### 2.2 Phase D 最优 TCN 配置 (2026-05-25 修订)

> **修订原因**: Phase D 第一轮 18 run 实测旧配置 (4 层 13×13、α=0.1、dropout=0.1、末尾 ReLU)
> 未达"模块有效"硬阈值,根因 5 项 (非负偏置 / α 过激 / 正则过弱 / RF 过大 / 缺归一化);
> 本节是修订后的最优配置,与 实验配置.py TCNProtocol 严格同步。

| 参数 | 值 | 说明 |
|---|---|---|
| `num_inputs` | 13 | SCADA 13 列核心信号 |
| `num_channels` | [26, 26, 13] | 3 层升降维 (中间扩 2× 给非线性留空间) |
| `kernel_size` | 3 | 最小奇数因果核 |
| `dropout` | 0.25 | 防短训练集 (~2700 行) 过拟合 |
| `alpha_init` | 0.01 | sigmoid 门控初值, 训练初期接近 baseline |
| `alpha_gate` | sigmoid | 保证 α ∈ (0, 1), 防发散到负/超 1 |
| `layer_norm_after_tcn` | True | 把 TCN(X) 标准化到 (mean=0, std=1) 再 α 加权 |
| `inner_activation` | gelu | 替代 ReLU, 允许负值激活 |
| `remove_final_relu` | True | 移除 TemporalBlock 末尾 ReLU, 输出可正可负 |

### 2.3 感受野计算

```
receptive_field = 1 + 2 × (k - 1) × Σ d_i
                = 1 + 2 × (3 - 1) × (1 + 2 + 4)
                = 1 + 4 × 7
                = 29 步
```

以 SCADA 10-min 采样率换算：29 × 10 / 60 = **4.83 小时**。

物理意义: 4.83h 精确落在轴承热惯量级 1~6h 区间,既能覆盖典型事件持续度
(20~68min ≈ 2~7 步), 也能捕获事件前 ~3h 的预兆信号。

---

## 3. 三 baseline 适配方案

### 3.1 共享适配器: `SCADATCNResidualAdapter`

```python
class SCADATCNResidualAdapter(nn.Module):
    """TCN 残差增强: (B, C, L) → (B, C, L) 形状不变."""
    def forward(self, x: Tensor) -> Tensor:
        return x + self.alpha * self.tcn(x)
```

位置: `{baseline}/modules/tcn_增强.py` (每个 baseline 目录下独立副本)

### 3.2 各 baseline 包装器

| Baseline | 包装器类 | 输入形状 | 适配逻辑 |
|---|---|---|---|
| Anomaly Transformer | `TCNInputWrapper` | (B, L, D) | permute → (B, D, L) → TCN → (B, D, L) → permute → (B, L, D) |
| TranAD | `TranADTCNWrapper` | (L, B, D) | permute → (B, D, L) → TCN → (B, D, L) → permute → (L, B, D) |
| TriTrackNet | `TriTrackNetTCNWrapper` | (B, C, L) | 直接 → TCN → (B, C, L) (无需转置) |

关键约束: 只做形状置换，不改 TCN 内部计算；三个 wrapper 共享同一 `SCADATCNResidualAdapter` 类定义。

---

## 4. 使用方式

### 4.1 通过启动器运行

```bash
# 仅 tcn_input_residual 模块, 默认使用 TrainingProtocol 的 5 seed / 10 epoch
python 启动.py --module tcn_input_residual

# baseline_only + tcn_input_residual 全部 (90 个 model run + 3 preprocess)
python 启动.py --module all

# 本轮只做轻量验证时使用 dry-run, 不启动训练
python 启动.py --dry-run
```

### 4.2 单 baseline 直接运行

```bash
# Anomaly Transformer + TCN
cd Anomaly-Transformer-main
python 实验.py --module tcn_input_residual --seed 0 --epochs 10

# TranAD + TCN
cd TranAD-main
python 实验.py --module tcn_input_residual --seed 0 --epochs 10

# TriTrackNet + TCN
cd TriTrackNet-main
python 实验.py --module tcn_input_residual --seed 0 --epochs 10
```

### 4.3 在代码中编程接入

```python
# 示例: 给 AnomalyTransformer 加 TCN 增强
from modules.tcn_增强 import TCNInputWrapper
model = AnomalyTransformer(...)
enhanced_model = TCNInputWrapper(model, input_channels=13)
# 此后训练/推理与原始模型相同
```

---

## 5. 新增参数量

| 组件 | 参数量 (近似) |
|---|---|
| 3 层 TemporalBlock: 13→26, 26→26, 26→13, kernel_size=3 (含 weight_norm + downsample) | ≈ 9,000 |
| LayerNorm(13) | 26 |
| 可学习 α (sigmoid 门控前的 raw 参数) | 1 |
| **总计** | ≈ 9,000 (约 36 KB, FP32) |

> 与 实验配置.py TCNProtocol.total_params_approx() 同步, 实测 forward 加载 9621 个参数
> (含 weight_norm 引入的 g/v 拆分参数)。

> 当前唯一实现位于 `TCN-master/TCN/scada_adapter.py`；三 baseline 的
> `modules/tcn_增强.py` 只负责 (B,L,D)/(L,B,D)/(B,C,L) 布局转换。

相对于三 baseline 原有参数量 (Anomaly Transformer ~300K, TranAD ~200K, TriTrackNet ~500K)，TCN 模块增加 < 3%，对训练速度和显存影响可忽略。

---

## 6. 与其它模块的关系

| 方面 | 说明 |
|---|---|
| 独立性 | TCN 模块与其它候选模块 (工况补偿/物理约束/退化感知 head) 正交，可任意叠加 |
| 顺序 | 若同时使用多种输入级增强，TCN 应放在最外层 (最接近原始 SCADA 数据) |
| 互斥性 | 与其它时序编码模块 (如 Informer encoder, PatchTST embedding) 互斥——同一次实验只选一种输入级增强 |

---

## 7. 预期效果与评估

### 7.1 预期增益

- **短期 (1h 级) 温度异常**: TCN 的局部时序感受野可能比纯注意力更擅长捕获温度导数变化 (升温速率信号)；
- **多尺度覆盖**: 3 层膨胀卷积覆盖 29 步 ≈ 4.83h 历史，精确匹配轴承热惯量级 1~6h；
- **TranAD 窗口已对齐 (2026-05-30)**: TranAD n_window 由 10→36 (6h) ≥ RF=29 步, TCN 第 3 层
  在三个 baseline 上都有完整真实感受野, 原"零填充妥协"已消除。窗口大小统一从
  `实验配置.DatasetProtocol.WIN_TRANAD` 取, 数据读入前由 `validate_windows()` 强制校验 ≥ 感受野。

### 7.2 评估方式

在切片 B (Train=01-01~01-19, Val=01-20~01-25 含 Evt-1, Test=03-15~04-15 含 Evt-2) 上，
固定 seeds=(0,1,2,3,4) (5 seed)，对比 `baseline_only` vs `tcn_input_residual` 在
判断 F1 / AUC / 预测 MSE 上的均值±标准差差异。

### 7.3 注意事项

- TCN 不改变 baseline 训练循环 (minimax / two-phase / PerturbOpt)，所有增益应归因于输入增强，不混淆训练策略变化；
- α 初始化为 0.01 (经 sigmoid 门控) 意味着第一轮训练接近 baseline_only；
  若训练后 α 仍 ≈ 0.01, 说明 TCN 未学到有用特征 (优化器认为不放大 α 更好);
- 监控建议: 在训练日志中记录 `adapter.alpha.item()` 的 epoch 序列, 评估 TCN 是否被有效利用。

---

## 8. 参考文献

1. Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. *arXiv:1803.01271*.
2. TCN 实现来源: <https://github.com/locuslab/TCN> (MIT License)
3. 本项目 TCN 适配设计文档: `E:\创新\实验配置.py` (TCNProtocol 类)
