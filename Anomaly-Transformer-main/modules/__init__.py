# -*- coding: utf-8 -*-
"""
modules 包初始化 (Anomaly-Transformer baseline)

修改原因 (2026-05-25 清理): 旧 hook 注册系统 (基类/注册中心/配置.py) 已删除,
         本 __init__.py 仅保留 tcn_增强 的延迟 import 入口。
作用: 让 `from modules.tcn_增强 import TCNInputWrapper` 在 实验.py 中可用。
数学原理: 无 (Python package init)。
科研标准: 新增模块只需在本目录添加 <name>.py 并在 实验.py 的 --module choices 注册。
"""
# 不主动 import 任何具体模块, 让上层 实验.py 显式 import 触发, 避免循环依赖。
