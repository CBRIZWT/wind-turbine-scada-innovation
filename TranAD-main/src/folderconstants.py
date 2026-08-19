r"""
模块名称：folderconstants.py
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
# Data folders
output_folder = 'processed'
data_folder = 'data'
