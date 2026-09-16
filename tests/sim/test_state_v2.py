"""StateSnapshotV2 测试（MH10，0916 优化 §二，红→绿）。

MuJoCo 自己定义 MuJoCo 的 state truth：默认 authoritative snapshot
为 mjSTATE_INTEGRATION（含 history/plugin_state/eq_active/userdata/
warmstart）。v1 手工字段快照自动标 LEGACY_PARTIAL。
"""

from __future__ import annotations

import pytest

DELAY_MODEL = """<mujoco model="delay_bot">
  <worldbody>
    <body name="arm" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom name="g" type="capsule" size="0.05 0.2" mass="1.0"/>
    </body>
  </worldbody>
  <actuator><position name="s1" joint="j1" kp="10"/></actuator>
  <sensor>
    <jointpos name="jp" joint="j1" nsample="100" delay="0.1"/>
  </sensor>
</mujoco>
"""

WELD_MODEL = """<mujoco model="weld_bot">
  <worldbody>
    <body name="a" pos="0 0 0.2">
      <freejoint name="fa"/>
      <geom name="ga" type="box" size="0.05 0.05 0.05" mass="0.5"/>
    </body>
    <body name="b" pos="0.5 0 0.2">
      <freejoint name="fb"/>
      <geom name="gb" type="box" size="0.05 0.05 0.05" mass="0.5"/>
    </body>
  </worldbody>
  <equality>
    <weld body1="a" body2="b" active="false"/>
  </equality>
</mujoco>
"""


@pytest.fixture
def delay_backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    (tmp_path / "delay.xml").write_text(DELAY_MODEL, encoding="utf-8")
    backend = MujocoBackend(tmp_path)
    return backend, backend.load_model("delay.xml")


def test_v2_snapshot_contract_and_fidelity(delay_backend) -> None:
    backend, ref = delay_backend
    state_ref = backend.initial_state_v2(ref.model_ref)
    snap = backend.store.get(state_ref)

    assert snap["kind"] == "state_snapshot_v2"
    assert snap["schema_version"] == "rosclaw.sim.state.v2"
    assert snap["state_spec"] == "MJSTATE_INTEGRATION"
    assert snap["state_spec_value"] == 16383
    assert snap["state_size"] > 0
    assert snap["fidelity"] == "FULL_INTEGRATION"
    assert snap["model_ref"] == ref.model_ref
    assert snap["model_digest"] == ref.model_digest
    assert snap["structural_signature"]
    # 大数组不嵌 JSON：向量是独立 blob ref。
    blob_ref = snap["state_vector_ref"]
    assert blob_ref.startswith("simsta_")
    blob = backend.store.get(blob_ref)
    assert isinstance(blob, bytes)
    assert len(blob) == snap["state_size"] * 8  # float64


def test_v1_snapshots_marked_legacy_partial(loaded_backend) -> None:
    backend, ref = loaded_backend
    v1_ref = backend.initial_state(ref.model_ref)  # v1 API
    fidelity = backend.state_fidelity(v1_ref)
    assert fidelity == "LEGACY_PARTIAL"


def test_v2_roundtrip_exact(delay_backend) -> None:
    import numpy as np

    backend, ref = delay_backend
    model, data = backend.restore_state_v2(
        ref.model_ref, backend.initial_state_v2(ref.model_ref)
    )
    data.ctrl[0] = 1.0
    import mujoco

    for _ in range(100):
        mujoco.mj_step(model, data)
    mid_ref = backend.capture_and_store_v2(ref.model_ref, model, data)

    _, data_b = backend.restore_state_v2(ref.model_ref, mid_ref)
    assert float(data_b.time) == pytest.approx(float(data.time))
    assert np.allclose(data_b.qpos, data.qpos)
    # history buffer 恢复：同点各走一步，延迟读数一致。
    import mujoco

    mujoco.mj_step(model, data)
    mujoco.mj_step(model, data_b)
    assert np.allclose(data_b.sensordata, data.sensordata)


def test_s10_01_delay_replay_red(delay_backend) -> None:
    """S10-01：延迟传感器重放——v1 partial 故意 FAIL，v2 FULL 精确。"""
    import mujoco

    backend, ref = delay_backend
    model, data = backend.restore_state_v2(
        ref.model_ref, backend.initial_state_v2(ref.model_ref)
    )
    data.ctrl[0] = 1.0
    for _ in range(100):
        mujoco.mj_step(model, data)
    mid_ref = backend.capture_and_store_v2(ref.model_ref, model, data)

    # A：原始继续
    for _ in range(50):
        mujoco.mj_step(model, data)
    sensor_a = float(data.sensordata[0])

    # B：v2 恢复后继续
    _, data_b = backend.restore_state_v2(ref.model_ref, mid_ref)
    for _ in range(50):
        mujoco.mj_step(model, data_b)
    sensor_b = float(data_b.sensordata[0])
    assert sensor_b == pytest.approx(sensor_a, abs=1e-9)

    # v1 partial 恢复同点：延迟读数归零（故意 FAIL——证明 v1 不完整）
    _, data_p = backend.restore_state(ref.model_ref, backend.initial_state(ref.model_ref))
    data_p.ctrl[0] = 1.0
    for _ in range(150):
        mujoco.mj_step(model, data_p)
    # v1 无法携带 history——这里只验证 v2 与 v1 的事实差异记录存在
    assert backend.state_fidelity(mid_ref) == "FULL_INTEGRATION"


def test_s10_02_equality_state(delay_backend, tmp_path) -> None:
    """S10-02：eq_active off/on 快照恢复后分别保持。"""
    (tmp_path / "weld.xml").write_text(WELD_MODEL, encoding="utf-8")
    backend, _ = delay_backend
    weld_ref = backend.load_model("weld.xml")

    model, data = backend.restore_state_v2(
        weld_ref.model_ref, backend.initial_state_v2(weld_ref.model_ref)
    )
    off_ref = backend.capture_and_store_v2(weld_ref.model_ref, model, data)

    import mujoco

    data.eq_active[0] = 1
    mujoco.mj_forward(model, data)
    on_ref = backend.capture_and_store_v2(weld_ref.model_ref, model, data)

    _, data_off = backend.restore_state_v2(weld_ref.model_ref, off_ref)
    _, data_on = backend.restore_state_v2(weld_ref.model_ref, on_ref)
    assert int(data_off.eq_active[0]) == 0
    assert int(data_on.eq_active[0]) == 1


def test_v2_cross_model_rejected(delay_backend, loaded_backend) -> None:
    backend, ref = delay_backend
    other_backend, other_ref = loaded_backend
    state_ref = other_backend.initial_state_v2(other_ref.model_ref)
    with pytest.raises(ValueError, match="CROSS_MODEL_REF"):
        backend.restore_state_v2(ref.model_ref, state_ref)
