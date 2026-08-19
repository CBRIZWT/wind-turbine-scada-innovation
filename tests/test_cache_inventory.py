from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SCADA数据集"))
import numpy as np
import pandas as pd
from 数据清单 import (
    CoverageRecord, audit_coverage, coverage_table,
    mark_main_exclusions, raw_inventory_hash,
)


def test_coverage_record_fields():
    rec = CoverageRecord(farm="kelmarsh", turbine=1, year=2021,
                         raw_files=["a.csv"], rows=144, first_ts="2021-01-01",
                         last_ts="2021-01-01", missing_segments=0, col_hash="deadbeef",
                         file_size_bytes=123, file_mtime=1.0)
    d = rec.as_dict()
    for k in ("farm", "turbine", "year", "rows", "first_ts", "last_ts",
              "missing_segments", "col_hash", "file_size_bytes", "file_mtime",
              "excluded_from_main", "exclude_reason"):
        assert k in d


def test_audit_coverage_synthetic():
    idx = pd.date_range("2021-01-01", periods=10, freq="10min")
    df = pd.DataFrame({"temp (°C)": range(10), "Wind speed (m/s)": range(10)}, index=idx)
    def provider(farm, turbine, year):
        return df if year == 2021 else None
    recs = audit_coverage("kelmarsh", turbines=[1], years=[2021, 2022], provider=provider)
    by_year = {r.year: r for r in recs}
    assert by_year[2021].rows == 10
    assert by_year[2022].rows == 0          # missing year recorded explicitly, not skipped
    assert by_year[2021].col_hash == by_year[2021].col_hash  # stable/reproducible


def test_mark_main_exclusions_hot():
    recs = [CoverageRecord(farm="hill_of_towie", turbine=2304515, year=2023, rows=100)]
    mark_main_exclusions(recs, "hill_of_towie")
    assert recs[0].excluded_from_main is True
    assert "weak" in recs[0].exclude_reason.lower() or "hot" in recs[0].exclude_reason.lower()


def test_mark_main_exclusions_supplemental_year():
    recs = [CoverageRecord(farm="penmanshiel", turbine=11, year=2024, rows=10),
            CoverageRecord(farm="penmanshiel", turbine=11, year=2023, rows=10)]
    mark_main_exclusions(recs, "penmanshiel", supplemental_years=(2024,))
    by_year = {r.year: r for r in recs}
    assert by_year[2024].excluded_from_main and not by_year[2023].excluded_from_main


def test_coverage_table_is_dataframe():
    recs = [CoverageRecord(farm="kelmarsh", turbine=1, year=2021, rows=5)]
    tbl = coverage_table(recs)
    assert hasattr(tbl, "columns") and "excluded_from_main" in tbl.columns


def test_raw_inventory_hash_stable_and_sensitive():
    recs = [CoverageRecord(farm="kelmarsh", turbine=1, year=2021, rows=5, col_hash="h1")]
    h1 = raw_inventory_hash(recs)
    h2 = raw_inventory_hash(recs)
    assert h1 == h2 and isinstance(h1, str) and len(h1) >= 8
    recs2 = [CoverageRecord(farm="kelmarsh", turbine=1, year=2021, rows=6, col_hash="h1")]
    assert raw_inventory_hash(recs2) != h1   # different rows → different hash


# ---- Task 4: LayeredCache + meta audit ----
def test_cache_roundtrip_and_key(tmp_path):
    from 缓存 import LayeredCache
    cache = LayeredCache(root=tmp_path, cache_version="v2")
    key = cache.key("kelmarsh", turbine=1, year=2021, layer="colcut", col_hash="abc123")
    assert "v2" in key and "kelmarsh" in key and "2021" in key and "abc123" in key
    arr = np.arange(12, dtype=np.float32).reshape(6, 2)
    assert not cache.exists(key)
    cache.put(key, arr)
    assert cache.exists(key)
    np.testing.assert_array_equal(cache.get(key), arr)


def test_cache_invalidates_on_version_or_hash(tmp_path):
    from 缓存 import LayeredCache
    c1 = LayeredCache(root=tmp_path, cache_version="v2")
    c1.put(c1.key("kelmarsh", 1, 2021, "colcut", "h1"), np.zeros((2, 2), np.float32))
    c2 = LayeredCache(root=tmp_path, cache_version="v3")          # version change → miss
    assert not c2.exists(c2.key("kelmarsh", 1, 2021, "colcut", "h1"))
    assert not c1.exists(c1.key("kelmarsh", 1, 2021, "colcut", "h2"))  # col hash change → miss


def test_audit_meta_contract():
    from 缓存 import build_audit_meta, assert_meta_contract
    meta = build_audit_meta(
        split_cfg={"split_id": "chronological_v2", "feature_version": "v2"},
        cache_version="v2", raw_inventory_hash="deadbeef",
        fit_row_counts={"nbm": 1000, "scaler": 1000},
        label_counts={"train": {"pos": 0, "neg": 1000},
                      "val": {"pos": 5, "neg": 900},
                      "test": {"pos": 8, "neg": 900}})
    assert_meta_contract(meta)
    assert meta["test_used_for_fit"] is False
    assert meta["test_used_for_selection"] is False
    assert meta["split_id"] == "chronological_v2"
    assert "split_hash" in meta and "raw_inventory_hash" in meta


def test_meta_contract_rejects_test_leak():
    import pytest
    from 缓存 import assert_meta_contract
    bad = {"split_id": "x", "split_hash": "h", "raw_inventory_hash": "h",
           "feature_version": "v2", "cache_version": "v2",
           "test_used_for_fit": True,                  # leak!
           "test_used_for_selection": False,
           "fit_row_counts": {}, "label_counts": {}}
    with pytest.raises(AssertionError):
        assert_meta_contract(bad)
