"""风机 SCADA 真实故障统一评测器。

本包只服务 `kelmarsh__realfault` 与 `penmanshiel__realfault`。A′、Hill
伪真值和 CARE 实验均不在本包的可执行范围内。
"""

from .contracts import CalibrationArtifact, DatasetBundle, RunRecord, ScoreView, TrainView

__all__ = [
    "CalibrationArtifact",
    "DatasetBundle",
    "RunRecord",
    "ScoreView",
    "TrainView",
]

