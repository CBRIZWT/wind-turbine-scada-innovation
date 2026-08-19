"""兼容既有 ``codex.repro_common`` 导入路径。

实际实现保留在论文复现目录；这里只做单一来源的显式重导出，避免复制逻辑。
"""
from 论文复现.repro_common import (  # noqa: F401
    FARMS,
    FarmSample,
    cumulative_scores,
    evaluate_scores,
    linear_residual_scores,
    load_farm_sample,
    preprocess_dir,
    project_root,
    reconstruction_scores,
    run_baseline_for_farms,
    sha256_file,
    write_result,
)

__all__ = [
    "FARMS", "FarmSample", "cumulative_scores", "evaluate_scores",
    "linear_residual_scores", "load_farm_sample", "preprocess_dir", "project_root",
    "reconstruction_scores", "run_baseline_for_farms", "sha256_file", "write_result",
]
