r"""
模块名称：constants.py
实验链路位置：TranAD 数据集维度、窗口长度和输出目录常量模块。
主要输入输出：保持原仓库接口不变；输入通常是 NumPy/Pandas/Torch 表示的多变量时间序列窗口，输出是模型张量、异常分数、预测值、阈值或实验指标。
核心数学思想：异常分数主要来自预测/重构误差，后续用 POT 或标签搜索阈值转成二值异常判断。
科研流程：先明确数据窗口和通道含义，再执行训练或推理，最后用重构误差、预测误差、关联差异或极值阈值形成可复核指标。
风机 SCADA 适用性：迁移到风机 SCADA 时，可把温度、振动、声音或功率相关传感器作为多变量通道；异常分数不能自动等同故障真值，仍需状态/报警日志或检修记录校验。
实现边界：本文件注释只解释名称、作用、数学原理和实验流程，不改变源码逻辑、默认参数、文件路径或张量形状。
参考文献：
- Tuli, S., Casale, G., & Jennings, N. R. (2022). TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data. PVLDB, 15(6), 1201-1214. PDF: https://www.vldb.org/pvldb/vol15/p1201-tuli.pdf
- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017. PDF: https://arxiv.org/pdf/1706.03762
- Siffer, A., Fouque, P.-A., Termier, A., & Largouet, C. (2017). Anomaly Detection in Streams with Extreme Value Theory. KDD 2017. PDF: https://www.amossys.fr/wp-content/uploads/anomaly-detection-with-evt-1.pdf
- Nakamura, T., Imamura, M., Mercer, R., & Keogh, E. (2020). MERLIN: Parameter-Free Discovery of Arbitrary Length Anomalies in Massive Time Series Archives. ICDM 2020. PDF: https://www.cs.ucr.edu/~eamonn/MERLIN_Long_version_for_website.pdf
"""
from src.parser import *
from src.folderconstants import *

# Threshold parameters
lm_d = {
		'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
		'synthetic': [(0.999, 1), (0.999, 1)],
		'SWaT': [(0.993, 1), (0.993, 1)],
		'UCR': [(0.993, 1), (0.99935, 1)],
		'NAB': [(0.991, 1), (0.99, 1)],
		'SMAP': [(0.98, 1), (0.98, 1)],
		'MSL': [(0.97, 1), (0.999, 1.04)],
		'WADI': [(0.99, 1), (0.999, 1)],
		'MSDS': [(0.91, 1), (0.9, 1.04)],
		'MBA': [(0.87, 1), (0.93, 1.04)],
		# SCADA: 这些 POT/MERLIN 参数为占位值, 实际阈值由 实验.py 的 choose_threshold_and_polarity_by_validation 决定
		# SCADA实验修改:
		# 名称: SCADA POT 默认参数
		# 修改原因: 原 TranAD 常量表没有 SCADA,导入 src.models 时会按 args.dataset 查表并触发 KeyError。
		# 作用: 允许实验入口以 --dataset SCADA 导入 TranAD 类;实际统一实验阈值由 E:\创新\实验工具.py 的验证集策略决定。
		# 数学原理: 这里保留 POT 风险参数 (level, scale) 的占位含义,但不用于测试集调阈值。
		# 执行流程: constants.py 初始化 lm 后,实验.py 继续构建模型并使用独立验证集阈值。
		# 科研标准: 不使用测试集选择阈值,该参数仅保证原代码可导入和旧接口兼容。
		'SCADA': [(0.99, 1), (0.99, 1)],
	}
lm = lm_d[args.dataset][1 if 'TranAD' in args.model else 0]

# Hyperparameters
lr_d = {
		'SMD': 0.0001, 
		'synthetic': 0.0001, 
		'SWaT': 0.008, 
		'SMAP': 0.001, 
		'MSL': 0.002, 
		'WADI': 0.0001, 
		'MSDS': 0.001, 
		'UCR': 0.006, 
		'NAB': 0.009, 
		'MBA': 0.001, 
		# SCADA实验修改: SCADA 使用 TranAD 原论文常见的 1e-4 量级学习率,并在实验.py 中记录到指标日志。
		'SCADA': 0.0001,
	}
lr = lr_d[args.dataset]

# Debugging
percentiles = {
		'SMD': (98, 2000),
		'synthetic': (95, 10),
		'SWaT': (95, 10),
		'SMAP': (97, 5000),
		'MSL': (97, 150),
		'WADI': (99, 1200),
		'MSDS': (96, 30),
		'UCR': (98, 2),
		'NAB': (98, 2),
		'MBA': (99, 2),
		# SCADA实验修改: SCADA percentile/cvp 仅用于原 MERLIN/POT 兼容变量,统一实验不依赖该值调测试阈值。
		'SCADA': (99, 2),
	}
percentile_merlin = percentiles[args.dataset][0]
cvp = percentiles[args.dataset][1]
preds = []
# debug = 9  # (2026-05-31 移除) 未使用变量, 原用于调试幅度缩放的临时标记
