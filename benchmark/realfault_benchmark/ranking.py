from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TRUE_FAULT_FARMS = ("kelmarsh", "penmanshiel")
STRICT_SEED = 20260719
_REQUIRED = {
    "model_id", "farm", "status", "seed", "protocol_hash", "variant",
    "calibration_split", "data_hash", "calibration_hash", "local_equal4_score",
    "pr_auc", "publication_date",
}
_BOARD_COLUMNS = [
    "model_id", "macro_equal4_score", "worst_farm_equal4_score",
    "cross_farm_std", "macro_pr_auc", "publication_date", "seed",
    "protocol_hash", "variant", "calibration_split",
]


def _reason(model_id: str, code: str, detail: str) -> dict[str, str]:
    return {"model_id": str(model_id), "reason_code": code, "reason_detail": detail}


def build_true_fault_leaderboard(
    records: pd.DataFrame,
    *,
    return_exclusions: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """构建默认严格的双场真实故障榜，并保留所有排除原因。

    只有 Kelmarsh/Penmanshiel 各一条成功记录、固定种子、相同协议、真实
    ``realfault`` variant、验证集校准以及有限合法指标的模型才可入榜。
    """

    if "model_id" not in records.columns:
        raise ValueError("真实故障长表缺少 model_id，无法形成可审计排除记录")
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    global_missing = sorted(_REQUIRED - set(records.columns))

    for model_id, group in records.groupby("model_id", sort=False, dropna=False):
        mid = str(model_id)
        if global_missing:
            exclusions.append(_reason(mid, "missing_required_fields", ",".join(global_missing)))
            continue
        farm_group = group[group["farm"].isin(TRUE_FAULT_FARMS)].copy()
        good = farm_group[farm_group["status"] == "success"].copy()
        if len(good) != 2 or set(good["farm"].astype(str)) != set(TRUE_FAULT_FARMS):
            exclusions.append(
                _reason(mid, "not_successful_on_both_farms", "需要两场各一条 status=success")
            )
            continue
        if good["farm"].astype(str).duplicated().any():
            exclusions.append(_reason(mid, "duplicate_farm_record", "每场只能有一条成功记录"))
            continue

        seeds = pd.to_numeric(good["seed"], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(seeds)) or not np.all(seeds == STRICT_SEED):
            exclusions.append(_reason(mid, "seed_not_20260719", repr(good["seed"].tolist())))
            continue

        scores = pd.to_numeric(good["local_equal4_score"], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(scores)) or not np.all((scores >= 0.0) & (scores <= 1.0)):
            exclusions.append(_reason(mid, "invalid_local_equal4_score", repr(scores.tolist())))
            continue

        protocol_values = good["protocol_hash"].fillna("").astype(str)
        if (protocol_values == "").any():
            exclusions.append(_reason(mid, "missing_protocol_hash", "protocol_hash 不能为空"))
            continue
        if protocol_values.nunique() != 1:
            exclusions.append(_reason(mid, "incompatible_protocol_hash", repr(protocol_values.tolist())))
            continue
        variants = good["variant"].fillna("").astype(str)
        if set(variants) != {"realfault"}:
            exclusions.append(_reason(mid, "invalid_variant", repr(variants.tolist())))
            continue
        calibration_splits = good["calibration_split"].fillna("").astype(str)
        if set(calibration_splits) != {"val"}:
            exclusions.append(
                _reason(mid, "invalid_calibration_split", repr(calibration_splits.tolist()))
            )
            continue
        if (good["data_hash"].fillna("").astype(str) == "").any():
            exclusions.append(_reason(mid, "missing_data_hash", "两场 data_hash 均必须存在"))
            continue
        if (good["calibration_hash"].fillna("").astype(str) == "").any():
            exclusions.append(
                _reason(mid, "missing_calibration_hash", "两场 calibration_hash 均必须存在")
            )
            continue

        pr = pd.to_numeric(good["pr_auc"], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(pr)) or not np.all((pr >= 0.0) & (pr <= 1.0)):
            exclusions.append(_reason(mid, "invalid_pr_auc", repr(pr.tolist())))
            continue
        dates = pd.to_datetime(good["publication_date"], errors="coerce", utc=True)
        if dates.isna().any():
            exclusions.append(_reason(mid, "invalid_publication_date", repr(good["publication_date"].tolist())))
            continue

        rows.append(
            {
                "model_id": mid,
                "macro_equal4_score": float(scores.mean()),
                "worst_farm_equal4_score": float(scores.min()),
                "cross_farm_std": float(np.std(scores, ddof=0)),
                "macro_pr_auc": float(pr.mean()),
                "publication_date": dates.max(),
                "seed": STRICT_SEED,
                "protocol_hash": protocol_values.iloc[0],
                "variant": "realfault",
                "calibration_split": "val",
            }
        )

    board = pd.DataFrame(rows, columns=_BOARD_COLUMNS)
    if not board.empty:
        board = board.sort_values(
            ["macro_equal4_score", "worst_farm_equal4_score", "cross_farm_std",
             "macro_pr_auc", "publication_date"],
            ascending=[False, False, True, False, False], kind="stable",
        ).reset_index(drop=True)
    excluded = pd.DataFrame(
        exclusions, columns=["model_id", "reason_code", "reason_detail"],
    )
    board.attrs["exclusions"] = excluded.to_dict(orient="records")
    if return_exclusions:
        return board, excluded
    return board
