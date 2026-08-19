# GitHub Report：风机 SCADA 温度异常研究项目主体

## 1. 项目定位

本项目面向风力发电机 SCADA 数据中的轴承/部件温度异常检测、预测与预后，重点研究时间序列建模、真实故障标签、事件级评价、提前量和低误报告警。主线包括 TriTrackNet、TranAD、Anomaly Transformer、TCN 适配器及统一评测基础设施。

建议仓库：`wind-turbine-scada-innovation`

## 2. 当前可公开内容

- `docs/项目架构.md`：当前架构和模块职责的主要说明。
- `运行手册.md`：实验 SOP、启动方式、输出规范和科研约束。
- `benchmark/`：评测合同、数据接口、指标和测试基础设施；需再次清理本地路径后上传。
- `codex/`：可复用的研究辅助代码。
- `tests/`：数据预处理、指标、事件评测、泄漏防护和模型适配测试。
- `TCN-master/`、`TriTrackNet-main/`、`TranAD-main/`、`Anomaly-Transformer-main/`、`wt-transformer-fault-prediction-main/`：仅保留源码、README、许可证和必要配置。
- 根目录的 Python 配置/工具脚本：上传前须逐个检查绝对路径、数据路径和本地环境依赖。

## 3. 证据等级与现状

| 内容 | 当前判断 | 公开表述 |
|---|---|---|
| 架构、协议、测试 | 有较完整本地文档和测试 | 可作为方法与工程基础公开 |
| 三模型/模块实验 | 存在全量和快速实验产物 | 必须附 manifest、split、seed 和版本信息 |
| 真实故障评测 | 有 real-fault 管线和事件级指标 | 仅报告已核验结果，不自动称为最终结论 |
| dry-run/组件测试 | 仅证明接线或局部行为 | 不得表述为全量性能或 SOTA |
| G0/G1 完整性门禁 | 仍需按当前状态重新核验 | 未通过的部分标记为不可用于确认性结论 |

## 4. 建议公开目录

```text
README.md
LICENSE
docs/
benchmark/
codex/
tests/
models/
scripts/
configs/
```

README 必须说明：数据不随仓库提供；实验依赖本地或合法数据源；路径通过配置传入；结果必须以 manifest 和哈希核验为准。

## 5. 明确排除

- `SCADA数据集/` 原始 CSV、预处理缓存和 `.npy/.npz`。
- `实验结果/` 中的原始日志、检查点、缓存和未审计中间产物。
- `.docx`、`.pptx`、论文全文和未确认版权的资料。
- 含 `E:\创新`、`C:\Users\...` 等绝对本地路径的原始 manifest；应改为相对路径或脱敏摘要。
- 所有凭据、令牌、私有配置、环境文件和本地 IDE 状态。

## 6. 复现入口与限制

建议先运行测试和 smoke/dry-run，再进行正式实验。正式训练、全量预处理和 sealed/confirmatory 评估不应在公开仓库 README 中暗示为已完成，除非对应 manifest、数据合同、split 和审计证据齐全。

## 7. 公开前门禁

- [ ] 代码不依赖本机绝对路径。
- [ ] `rg` 扫描未发现凭据或私有路径。
- [ ] 所有大文件按白名单排除。
- [ ] README 和 LICENSE 完整。
- [ ] 测试命令及预期结果可说明。
- [ ] 结果表区分探索性、快速实验和正式评估。
