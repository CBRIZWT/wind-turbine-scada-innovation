from __future__ import annotations
import sys
import os
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import pytest
import 启动


def _args(**kw):
    """构造一个最小可用的 argparse.Namespace (按 启动.py 的属性名)。
    默认 = 三 farm × 5 seed × 10 epoch 的 chronological_v2 正式矩阵 (即"正式运行")。"""
    ns = argparse.Namespace(
        split="chronological_v2", cache_version="v2", dry_run=False, smoke=False,
        confirm_full=False, force=False, farms="kelmarsh,penmanshiel,hill_of_towie",
        models="anomaly,tranad,tritrack,wt", module="all", seeds="0,1,2,3,4",
        norm="robust", epochs=10, batch_size=128, skip_preprocess=False,
        label_mode="real_fault_wl", a1_lead=72,
        model_python=sys.executable, no_resume=False,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


# ---- is_full_run: split 无关; 免许可只剩 dry-run 与单farm单seed1epoch smoke ----

def test_is_full_run_true_for_chrono_full():
    assert 启动.is_full_run(_args()) is True


def test_is_full_run_true_for_narrow_full():
    # 精炼 spec: narrow_v1 历史复现全量同样属正式运行, 需 --confirm-full。
    assert 启动.is_full_run(_args(split="narrow_v1")) is True


def test_is_full_run_false_for_dry_run():
    assert 启动.is_full_run(_args(dry_run=True)) is False
    assert 启动.is_full_run(_args(dry_run=True, split="narrow_v1")) is False


def test_unlicensed_smoke_is_exempt():
    a = _args(smoke=True, farms="kelmarsh", seeds="0", epochs=1)
    assert 启动.is_unlicensed_smoke(a) is True
    assert 启动.is_full_run(a) is False


def test_oversized_smoke_is_gated():
    # 多 farm / 多 seed / 多 epoch 的 smoke 不再免许可。
    assert 启动.is_full_run(_args(smoke=True)) is True
    assert 启动.is_full_run(_args(smoke=True, farms="kelmarsh", seeds="0", epochs=5)) is True
    assert 启动.is_full_run(_args(smoke=True, farms="kelmarsh", seeds="0,1", epochs=1)) is True


def test_force_is_gated_even_when_small():
    a = _args(smoke=True, farms="kelmarsh", seeds="0", epochs=1, force=True)
    assert 启动.is_unlicensed_smoke(a) is False
    assert 启动.is_full_run(a) is True


# ---- assert_full_run_permitted: 阻断 / 放行 ----

def test_gate_blocks_without_confirm():
    for kw in ({}, {"split": "narrow_v1"}, {"force": True}, {"smoke": True}):
        with pytest.raises(SystemExit):
            启动.assert_full_run_permitted(_args(**kw))


def test_gate_allows_with_confirm():
    启动.assert_full_run_permitted(_args(confirm_full=True))            # 不应抛异常


def test_gate_allows_dryrun_and_unlicensed_smoke():
    启动.assert_full_run_permitted(_args(dry_run=True))
    启动.assert_full_run_permitted(_args(smoke=True, farms="kelmarsh", seeds="0", epochs=1))


# ---- 协议签名 ----

def test_signature_has_split_hash_and_feature_version():
    sig = 启动.build_protocol_signature(_args())
    assert "split_hash" in sig and len(sig["split_hash"]) >= 8
    assert sig["feature_version"] == "v2"
    assert 启动.build_protocol_signature(_args(split="narrow_v1"))["feature_version"] == "v1"


def test_signature_tracks_a1_label_protocol():
    sig = 启动.build_protocol_signature(_args())
    assert sig["label_mode"] == "real_fault_wl"
    # [2026-07-26] A' 删除后 theta_pct/sustain 已无计算作用并从签名移除; 仅 lead(早警提前量)仍生效。
    assert sig["a1_label"] == {"lead": 72}


def test_preprocess_step_passes_a1_args_and_checks_wt_outputs(monkeypatch):
    monkeypatch.setattr(启动, "resolve_model_python", lambda value: sys.executable)
    steps = 启动.build_steps(_args(models="anomaly,tranad,tritrack,wt"), batch=0, force_all=False)
    preprocess = next(step for step in steps if step.name == "preprocess_kelmarsh")

    assert preprocess.command[preprocess.command.index("--label-mode") + 1] == "real_fault_wl"
    assert preprocess.command[preprocess.command.index("--a1-lead") + 1] == "72"

    # 2026-06 单一真源: 预处理只产出一份到 SCADA数据集/数据预处理/<farm>/ (四模型共读), 不再每模型一份。
    output_text = "\n".join(str(path) for path in preprocess.outputs)
    assert "数据预处理" in output_text
    assert any(path.name == "meta.json" and "数据预处理" in str(path) for path in preprocess.outputs)


# ---- 结构性护栏: 无 --confirm-full 的正式运行必须在任何副作用前退出 ----

def test_gate_blocks_before_side_effects(tmp_path):
    """bare 全量 (无 --confirm-full) 必须非零退出, 且不写 chronological_v2__v2 manifest。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    manifest = ROOT / "实验状态" / "chronological_v2__v2" / "manifest.json"
    existed_before = manifest.exists()
    r = subprocess.run(
        [sys.executable, str(ROOT / "启动.py"),
         "--split", "chronological_v2", "--cache-version", "v2"],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=180,
    )
    assert r.returncode != 0
    assert "权限门" in (r.stdout + r.stderr)
    # 门在 main() 最前触发 → 不应新建 manifest
    if not existed_before:
        assert not manifest.exists(), "权限门触发后仍写了 manifest, 副作用未被前置阻断"


def test_pipeline_run_full_requires_confirm(monkeypatch):
    """--pipeline --run-full 也必须先过权限门, 不能绕到编排器里产生副作用。"""
    import types

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "启动.py",
            "--pipeline",
            "--run-full",
            "--split",
            "chronological_v2",
            "--cache-version",
            "v2",
        ],
    )
    monkeypatch.setitem(
        sys.modules,
        "全量编排",
        types.SimpleNamespace(run_pipeline=lambda *a, **k: pytest.fail("权限门前不应进入 run_pipeline")),
    )

    with pytest.raises(SystemExit):
        启动.main()
