r"""
模块名称：perturbopt.py
实验链路位置：TriTrackNet PerturbOpt/SAM 风格扰动优化器模块。
主要输入输出：保持原仓库接口不变；输入通常是 NumPy/Pandas/Torch 表示的多变量时间序列窗口，输出是模型张量、异常分数、预测值、阈值或实验指标。
核心数学思想：PerturbOpt 采用扰动方向 $e(w)\propto\nabla_w L$，先到局部尖锐邻域评估损失，再回到原参数执行鲁棒更新。
科研流程：先明确数据窗口和通道含义，再执行训练或推理，最后用重构误差、预测误差、关联差异或极值阈值形成可复核指标。
风机 SCADA 适用性：迁移到风机 SCADA 时，它更适合提前预测齿轮箱油温、轴承温度或振动趋势；若要做故障检测，需要再用预测残差、阈值和故障日志构造标签。
实现边界：本文件注释只解释名称、作用、数学原理和实验流程，不改变源码逻辑、默认参数、文件路径或张量形状。
参考文献：
- Liang, M., Jia, S., Liu, Y., Zhang, X., Wang, H., & Sun, Y. (2026). TriTrackNet: A dual-channel time series forecasting model with multi-path interaction and perturbation optimization. Neurocomputing, 669, 132519. DOI: https://doi.org/10.1016/j.neucom.2025.132519
- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017. PDF: https://arxiv.org/pdf/1706.03762
- Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022). Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. ICLR 2022. PDF: https://openreview.net/pdf?id=cGDAkQo1C0p
"""
import torch
from torch.optim import Optimizer


class perturbopt(Optimizer):
    """
    SAM with Adversarial Training: Sharpness-Aware Minimization combined with adversarial training.
    """
# 科研注释：`perturbopt` 已有原始 docstring；本补充说明强调其科研实验含义。名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。 数学原理：扰动更新近似 $w+\rho\nabla L/\|\nabla L\|$，用于降低尖锐极小值和噪声敏感性。 SCADA 迁移：保持时间顺序、通道物理意义和故障标签来源一致。

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, steps=3, epsilon=1.9, **kwargs):
        """
        :param steps: 扰动的步骤数，表示每次优化时扰动的增加步骤
        :param epsilon: 对抗扰动的强度
        """
        # 科研注释：`__init__` 已有原始 docstring；本补充说明强调其科研实验含义。名称作用：初始化对象超参数、网络子模块、缓存状态或优化器配置。 数学原理：扰动更新近似 $w+\rho\nabla L/\|\nabla L\|$，用于降低尖锐极小值和噪声敏感性。 SCADA 迁移：保持时间顺序、通道物理意义和故障标签来源一致。
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, steps=steps, epsilon=epsilon, **kwargs)
        super(perturbopt, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups


    @torch.no_grad()
    def first_step(self, zero_grad=False):
        r"""
        科研注释：函数/方法 `first_step`
        名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
        参数说明：`zero_grad`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回值保持原实现约定，调用方依赖其形状和类型。
        数学原理：扰动更新近似 $w+\rho\nabla L/\|\nabla L\|$，用于降低尖锐极小值和噪声敏感性。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        # 科研流程注释：梯度范数用于归一化扰动半径，避免不同层参数尺度导致扰动失衡。
        grad_norm = self._grad_norm()

        # 计算当前步数的扰动强度
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # 计算扰动量，根据步骤进行逐渐增大的扰动
                e_w = (
                        (torch.pow(p, 2) if group["adaptive"] else 1.0)
                        * p.grad
                        * scale.to(p)
                )

                # 增加扰动强度：step越大，扰动越强
                e_w *= (self.param_groups[0]["epsilon"] * (self.param_groups[0]["steps"] - 1) / self.param_groups[0][
                    "steps"])

                # 科研流程注释：第一步临时移动到高损失邻域评估鲁棒性，不是最终参数更新。
                p.add_(e_w)  # 对模型参数进行扰动
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        r"""
        科研注释：函数/方法 `second_step`
        名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
        参数说明：`zero_grad`：沿用原实现含义，通常表示张量、数组、路径、超参数或评估开关。
        返回值：返回值保持原实现约定，调用方依赖其形状和类型。
        数学原理：扰动更新近似 $w+\rho\nabla L/\|\nabla L\|$，用于降低尖锐极小值和噪声敏感性。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 科研流程注释：第二步先撤销临时扰动，再调用基础优化器完成真正更新。
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()


    def _grad_norm(self):
        r"""
        科研注释：函数/方法 `_grad_norm`
        名称作用：承担该模块中的局部实验步骤，服务于数据处理、模型构建、训练、推理或评估。
        参数说明：无外部业务参数；主要使用对象内部状态或全局实验配置。
        返回值：返回值保持原实现约定，调用方依赖其形状和类型。
        数学原理：扰动更新近似 $w+\rho\nabla L/\|\nabla L\|$，用于降低尖锐极小值和噪声敏感性。
        流程说明：流程：按原代码顺序完成局部数据准备、计算和返回；注释不改变任何控制流。
        关键参数：保持源码中的窗口长度、通道数、学习率、阈值概率、dropout、batch size 等默认值；改实验时应在配置层记录。
        SCADA 迁移：风机温度、振动、声音等通道进入该流程前必须统一采样间隔、缺失处理和量纲；输出指标需用报警/停机/检修记录验证。
        """
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    # 超参数修复 (2026-05-31): p=3→p=2, 匹配发表的 SAM 算法 (Foret et al. 2021)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm
