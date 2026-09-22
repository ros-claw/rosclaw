"""Digital Shadow v2 ClockAlignment + Residual v2 测试（MH21-B，
讨论总纲 §17-§19，红→绿）。

真实机器人数据：100Hz state / 30Hz camera / jitter / offset /
dropped samples——绝不是整齐同 t。ClockAlignment 支持
nearest/linear/hold-last-value；residual 分通道（qpos RMSE/P95/
max、qvel RMSE…），verdict 四级 MATCH/PARTIAL_MATCH/DIVERGED/
NOT_COMPARABLE。
"""

from __future__ import annotations

import numpy as np
import pytest

PEND_BASE = """<mujoco model="pend_base">
  <option timestep="0.002"/>
  <worldbody>
    <body name="rod" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
      <geom name="rod_g" type="capsule" size="0.02 0.25" pos="0 0 -0.25" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    (tmp_path / "base.xml").write_text(PEND_BASE, encoding="utf-8")
    return MujocoBackend(tmp_path)


def _observe(backend, qpos0: float = 0.6, duration_s: float = 1.0) -> str:
    source = backend.load_model("base.xml")
    dataset_ref = backend.record_dataset(
        source.model_ref,
        sequences=[{"controller": {"hold": True}, "duration_s": duration_s, "qpos0": [qpos0]}],
    )
    dataset = backend.store.get(dataset_ref)
    return dataset["sequences"][0]["trace_ref"]


def _shift_times(backend, trace_ref: str, offset_s: float) -> str:
    """把 trace 时间轴整体平移（人为 clock offset）。"""
    trace = backend.store.get(trace_ref)
    shifted = [{**row, "t": float(row["t"]) + offset_s} for row in trace["states"]]
    return backend.store.put("traces", {**trace, "states": shifted})


def _subsample(backend, trace_ref: str, every: int) -> str:
    """抽稀采样（不同采样率）。"""
    trace = backend.store.get(trace_ref)
    sampled = [row for i, row in enumerate(trace["states"]) if i % every == 0]
    return backend.store.put("traces", {**trace, "states": sampled})


def _drop_random(backend, trace_ref: str, ratio: float, seed: int = 7) -> str:
    """随机丢样本。"""
    trace = backend.store.get(trace_ref)
    rng = np.random.default_rng(seed)
    kept = [row for row in trace["states"] if rng.random() > ratio]
    return backend.store.put("traces", {**trace, "states": kept})


def test_sh02_clock_offset_alignment(backend) -> None:
    """SH02：观测整体 +35ms 偏移——ClockAlignment 后仍可比
    （estimated_offset ≈ 35ms，且 MATCH）。"""
    base = backend.load_model("base.xml")
    trace_ref = _observe(backend)
    shifted_ref = _shift_times(backend, trace_ref, 0.035)
    report = backend.shadow_compare(base.model_ref, shifted_ref)
    assert report["clock_alignment"]["estimated_offset_s"] == pytest.approx(0.035, abs=0.004)
    assert report["verdict"] == "MATCH"


def test_sh03_different_sample_rate_comparable(backend) -> None:
    """SH03：观测 100Hz（SIM 500Hz 的 1/5）——插值后可比且 MATCH。"""
    base = backend.load_model("base.xml")
    trace_ref = _observe(backend)
    sparse_ref = _subsample(backend, trace_ref, 5)
    report = backend.shadow_compare(base.model_ref, sparse_ref)
    assert report["verdict"] == "MATCH"
    assert report["clock_alignment"]["aligned_pairs"] > 40


def test_sh04_dropped_samples_bounded(backend) -> None:
    """SH04：随机丢 10%——仍有界 compare + 报告 drop ratio。"""
    base = backend.load_model("base.xml")
    trace_ref = _observe(backend)
    dropped_ref = _drop_random(backend, trace_ref, 0.1)
    report = backend.shadow_compare(base.model_ref, dropped_ref)
    assert report["verdict"] in ("MATCH", "PARTIAL_MATCH")
    alignment = report["clock_alignment"]
    assert alignment["dropped_samples"] > 0
    assert 0.0 < alignment["drop_ratio"] < 0.2


def test_residual_v2_per_channel(backend) -> None:
    """§18：residual 分通道（qpos RMSE/P95/max、qvel RMSE）——
    不再只有 max。"""
    base = backend.load_model("base.xml")
    trace_ref = _observe(backend)
    report = backend.shadow_compare(base.model_ref, trace_ref)
    channels = report["channels"]
    qpos = channels["qpos"]
    assert "rmse" in qpos and "p95" in qpos and "max" in qpos
    assert "rmse" in channels["qvel"]


def test_partial_match_semantics(backend, tmp_path) -> None:
    """四级判定：小偏差 → PARTIAL_MATCH（不是非黑即白）。

    构造：观测来自阻尼 0.05 的近邻模型（与 base 0.01 微差）——
    qpos 偏差小、qvel 偏差中等。"""

    near = PEND_BASE.replace('damping="0.01"', 'damping="0.05"').replace(
        'model="pend_base"', 'model="pend_near"'
    )
    (tmp_path / "near.xml").write_text(near, encoding="utf-8")
    base = backend.load_model("base.xml")
    near_model = backend.load_model("near.xml")
    dataset_ref = backend.record_dataset(
        near_model.model_ref,
        sequences=[{"controller": {"hold": True}, "duration_s": 1.0, "qpos0": [0.6]}],
    )
    dataset = backend.store.get(dataset_ref)
    report = backend.shadow_compare(
        base.model_ref,
        dataset["sequences"][0]["trace_ref"],
        partial_threshold=0.05,
    )
    assert report["verdict"] in ("PARTIAL_MATCH", "DIVERGED")
    assert report["verdict"] != "MATCH"  # 真分歧不得 MATCH
    # PARTIAL 语义：偏差在容差带内。
    if report["residual"] <= 0.05:
        assert report["verdict"] == "PARTIAL_MATCH"


def test_not_comparable_when_no_overlap(backend) -> None:
    """时间轴完全不重叠 → NOT_COMPARABLE（不硬凑）。"""
    base = backend.load_model("base.xml")
    trace_ref = _observe(backend)
    far_ref = _shift_times(backend, trace_ref, 100.0)  # +100s 完全错位
    report = backend.shadow_compare(base.model_ref, far_ref, allow_clock_search=False)
    assert report["verdict"] == "NOT_COMPARABLE"
    assert report["clock_alignment"]["aligned_pairs"] == 0
