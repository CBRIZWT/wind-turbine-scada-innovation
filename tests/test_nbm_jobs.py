# -*- coding: utf-8 -*-
from __future__ import annotations


def test_nbm_job_cap_is_environment_controlled(monkeypatch):
    import 温度指标选择 as module

    monkeypatch.setenv("NBM_N_JOBS", "2")
    assert module._nbm_n_jobs(14) == 2
    monkeypatch.setenv("NBM_N_JOBS", "1")
    assert module._nbm_n_jobs(14) == 1
