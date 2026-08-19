# -*- coding: utf-8 -*-
"""
回归测试: SchedulerProtocol.step_scheduler 对三类调度器都安全 (Bug#1 防回归)。

Bug#1 (2026-06-02): TranAD train_one_epoch 原以无参 `scheduler.step()` 结尾, 对
ReduceLROnPlateau 缺 metrics 实参会 TypeError, 在 grid-search 选中 plateau 时会让整个
90-run 的每个 TranAD run 崩在 epoch 末。修复: TranAD 改走统一入口
SchedulerProtocol.step_scheduler(scheduler, val_loss=...)。本测试锁定该入口对
cosine/steplr (无参 step) 与 plateau (需 val_loss) 均不抛异常, 并反证原始
plateau.step() 的确会崩 (说明修复的必要性)。
"""
import pytest
import torch

from 实验配置 import SchedulerProtocol


def _make_optimizer():
    p = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
    return torch.optim.Adam([p], lr=1e-3)


def test_step_scheduler_all_candidates_safe_with_val_loss():
    """统一入口对全部候选调度器 + val_loss 都不抛异常 (含 plateau)。"""
    for name in SchedulerProtocol.CANDIDATES:  # ("cosine", "plateau", "steplr")
        opt = _make_optimizer()
        sch = SchedulerProtocol.build_scheduler(name, opt, T_max=3)
        # 模拟两个 epoch 的调度推进
        SchedulerProtocol.step_scheduler(sch, val_loss=0.5)
        SchedulerProtocol.step_scheduler(sch, val_loss=0.4)


def test_plateau_raw_step_without_metric_raises():
    """反证 Bug#1: 直接对 plateau 调无参 .step() (TranAD 原写法) 必崩。"""
    opt = _make_optimizer()
    sch = SchedulerProtocol.build_scheduler("plateau", opt)
    assert "ReduceLROnPlateau" in type(sch).__name__
    with pytest.raises(TypeError):
        sch.step()  # 缺 metrics 实参 → 这正是被修复的崩溃点
