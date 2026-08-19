from __future__ import annotations

from pathlib import Path

import pytest

from run_real_fault_benchmark import _guard_existing_run


def test_existing_stale_run_requires_explicit_rerun_permission(tmp_path: Path) -> None:
    record = tmp_path / "run_record.json"
    record.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="--allow-rerun"):
        _guard_existing_run(record, resumable=False, allow_rerun=False)
    assert _guard_existing_run(record, resumable=False, allow_rerun=True) == "run"


def test_resumable_or_absent_run_is_safe(tmp_path: Path) -> None:
    absent = tmp_path / "missing.json"
    assert _guard_existing_run(absent, resumable=False, allow_rerun=False) == "run"
    present = tmp_path / "run_record.json"
    present.write_text("{}", encoding="utf-8")
    assert _guard_existing_run(present, resumable=True, allow_rerun=False) == "resume"
