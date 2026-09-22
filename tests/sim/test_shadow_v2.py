"""Digital Shadow v2 测试（MH21-A，讨论总纲 §13-§16，红→绿）。

ObservationTraceV2 provenance 强制 + joint name 映射：
- SIMULATION trace 只能 SHADOW_SELF_TEST，不能 REAL claim（SH06）；
- joint 顺序置换仍正确 compare（SH01）；
- body snapshot 不一致 → SHADOW_BODY_IDENTITY_MISMATCH（SH05）；
- joint schema 与模型不符 → SHADOW_JOINT_SCHEMA_MISMATCH。
"""

from __future__ import annotations

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

TWO_JOINT = """<mujoco model="two_joint">
  <option timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom name="g1" type="capsule" size="0.03 0.15" mass="0.8"/>
      <body name="b2" pos="0 0 0.2">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom name="g2" type="capsule" size="0.025 0.12" mass="0.5"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    (tmp_path / "base.xml").write_text(PEND_BASE, encoding="utf-8")
    (tmp_path / "two.xml").write_text(TWO_JOINT, encoding="utf-8")
    return MujocoBackend(tmp_path)


def _observe(backend, source_file: str, qpos0: list[float]) -> str:
    """录一段自由摆动观测（SIMULATION 域）。"""
    source = backend.load_model(source_file)
    dataset_ref = backend.record_dataset(
        source.model_ref,
        sequences=[{"controller": {"hold": True}, "duration_s": 0.5, "qpos0": qpos0}],
    )
    dataset = backend.store.get(dataset_ref)
    return dataset["sequences"][0]["trace_ref"]


def test_sh06_sim_trace_is_self_test_not_real(backend) -> None:
    """SH06：SIMULATION trace 只能 SHADOW_SELF_TEST——绝不输出
    REAL_SHADOW_COMPARE（SIM ≠ REAL 证据域强制）。"""
    base = backend.load_model("base.xml")
    trace_ref = _observe(backend, "base.xml", [0.6])
    report = backend.shadow_compare(base.model_ref, trace_ref)
    assert report["shadow_mode"] == "SELF_TEST"
    assert report["verdict"] == "MATCH"  # 自测一致
    # 绝不允许 REAL 语义。
    assert "REAL" not in report["shadow_mode"]


def test_hardware_recorded_enables_real_shadow(backend) -> None:
    """HARDWARE_RECORDED provenance 齐全 → REAL_SHADOW_COMPARE。"""
    base = backend.load_model("base.xml")
    trace_ref = _observe(backend, "base.xml", [0.6])
    obs_ref = backend.import_observation(
        trace_ref,
        evidence_domain="HARDWARE_RECORDED",
        body_id="pend_01",
        body_snapshot_hash="",  # 空 → import 时从源模型结构签名自动计算
        source={"type": "ros2", "robot": "pend_01", "session": "sess_1"},
        joint_schema=[{"joint": "hinge"}],
        clock={"domain": "ros_time", "epoch": "2026-09-21T00:00:00Z", "time_offset_estimate": 0.0},
        calibration_ref="",
    )
    report = backend.shadow_compare(base.model_ref, obs_ref)
    assert report["shadow_mode"] == "REAL_SHADOW_COMPARE"
    assert report["observation_ref"] == obs_ref


def test_sh05_body_identity_mismatch_rejected(backend) -> None:
    """SH05：body snapshot hash 与当前模型不符 →
    SHADOW_BODY_IDENTITY_MISMATCH（观测不是这个身体的）。"""
    base = backend.load_model("base.xml")
    trace_ref = _observe(backend, "base.xml", [0.6])
    obs_ref = backend.import_observation(
        trace_ref,
        evidence_domain="HARDWARE_RECORDED",
        body_id="pend_01",
        body_snapshot_hash="sha256:" + "ff" * 32,  # 伪造 hash
        source={"type": "ros2", "robot": "pend_01", "session": "sess_1"},
        joint_schema=[{"joint": "hinge"}],
        clock={"domain": "ros_time", "epoch": "2026-09-21T00:00:00Z", "time_offset_estimate": 0.0},
        calibration_ref="",
    )
    with pytest.raises(ValueError, match="SHADOW_BODY_IDENTITY_MISMATCH"):
        backend.shadow_compare(base.model_ref, obs_ref)


def test_joint_schema_mismatch_rejected(backend) -> None:
    """joint_schema 声明的关节名模型里不存在 →
    SHADOW_JOINT_SCHEMA_MISMATCH（不按数组位置猜）。"""
    base = backend.load_model("base.xml")
    trace_ref = _observe(backend, "base.xml", [0.6])
    with pytest.raises(ValueError, match="SHADOW_JOINT_SCHEMA_MISMATCH"):
        backend.import_observation(
            trace_ref,
            evidence_domain="HARDWARE_RECORDED",
            body_id="pend_01",
            body_snapshot_hash="",  # 由 import 计算
            source={"type": "ros2", "robot": "pend_01", "session": "sess_1"},
            joint_schema=[{"joint": "ghost_joint"}],
            clock={"domain": "ros_time", "epoch": "2026-09-21T00:00:00Z", "time_offset_estimate": 0.0},
            calibration_ref="",
            model_ref=base.model_ref,
        )


def test_sh01_joint_order_permutation_still_correct(backend) -> None:
    """SH01：REAL 观测的 joint 顺序与模型不同——按名映射后
    compare 仍正确（不按数组位置猜）。

    真置换：trace 数据列本身就是 j2,j1 序（REAL joint_state
    常见）——import 按 joint_order_in_trace 重排回模型序。"""
    two = backend.load_model("two.xml")
    trace_ref = _observe(backend, "two.xml", [0.5, -0.3])
    # 手工置换 trace 列（qpos/qvel 都是 [j1, j2] → [j2, j1]）。
    trace = backend.store.get(trace_ref)
    permuted_states = [
        {**row, "qpos": [row["qpos"][1], row["qpos"][0]], "qvel": [row["qvel"][1], row["qvel"][0]]}
        for row in trace["states"]
    ]
    permuted_ref = backend.store.put("traces", {**trace, "states": permuted_states})
    obs_ref = backend.import_observation(
        permuted_ref,
        evidence_domain="HARDWARE_RECORDED",
        body_id="two_joint_01",
        body_snapshot_hash="",
        source={"type": "ros2", "robot": "two_joint_01", "session": "sess_2"},
        joint_schema=[{"joint": "j2"}, {"joint": "j1"}],  # 观测声明序
        joint_order_in_trace=["j2", "j1"],  # trace 数据列的真实顺序
        clock={"domain": "ros_time", "epoch": "2026-09-21T00:00:00Z", "time_offset_estimate": 0.0},
        calibration_ref="",
        model_ref=two.model_ref,
    )
    report = backend.shadow_compare(two.model_ref, obs_ref)
    assert report["shadow_mode"] == "REAL_SHADOW_COMPARE"
    assert report["verdict"] == "MATCH"


def test_observation_v2_contract() -> None:
    """ObservationTraceV2 契约：schema literal + SIMULATED 之外的
    证据域字段 + digest 稳定。"""
    from rosclaw.sim.contracts import ObservationTraceV2

    obs = ObservationTraceV2()
    assert obs.schema_version == "rosclaw.observation_trace.v2"
    assert obs.evidence_domain == "SIMULATION"
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ObservationTraceV2(evidence_domain="PROBABLY_REAL")
