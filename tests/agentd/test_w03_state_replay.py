"""W03 红测试（规格 2026-09-08 §7）：真实状态记录与完整回放。

§7.1 新记录必须保存完整 qpos（长度 nq）、qvel（长度 nv）、ctrl
（长度 nu），并显式声明维度与模型摘要——nq≠nv≠nu 的模型
（freejoint/被动关节）不得再被静默截断。

§7.2 视频回放 = 同模型状态恢复 + 相机 + 仅前向（不重仿真）；
旧部分关节 trace = 有标注的兼容回放或受信回放拒绝——绝不
补零冒充完整证据。

§7.2 续仿真检查点 = 状态维度/模型摘要校验，跨模型引用拒绝。
"""

from __future__ import annotations

import json

import pytest

from rosclaw.sandbox.backends import MujocoCpuBackend, RolloutRequest, ScenarioSpec
from rosclaw.sandbox.backends.fingerprints import file_hash
from rosclaw.sandbox.sandbox_api import Sandbox

HOME = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

FREEJOINT_MJCF = """<mujoco model="w03_fixture">
  <worldbody>
    <body name="cube" pos="0 0 0.1">
      <freejoint name="cube_free"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.5"/>
    </body>
    <body name="arm" pos="0.3 0 0.3">
      <joint name="shoulder" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.01 0.1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_motor" joint="shoulder" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


def _rollout(tmp_path):
    sandbox = Sandbox.create("ur5e", "empty", "mujoco")
    assert sandbox.has_physics, sandbox.load_error
    scenario = ScenarioSpec(
        scenario_id="w03-state-contract",
        robot_id="ur5e",
        world_id="empty",
        body_snapshot_hash="sha256:test-body-snapshot",
        model_hash=file_hash(sandbox.model_path),
        seed=7,
    )
    request = RolloutRequest(
        scenario=scenario,
        trajectory=[HOME],
        artifact_dir=tmp_path,
    )
    backend = MujocoCpuBackend(sandbox)
    try:
        receipt = backend.rollout(request)
    finally:
        sandbox.close()
    assert receipt.physics_executed is True
    return tmp_path / "trajectory_states.json"


def _model_dims():
    import mujoco

    sandbox = Sandbox.create("ur5e", "empty", "mujoco")
    assert sandbox.has_physics, sandbox.load_error
    model = sandbox.physics_model
    dims = (int(model.nq), int(model.nv), int(model.nu))
    sandbox.close()
    assert mujoco is not None
    return dims


class TestFullStateRecording:
    def test_states_file_declares_dims_and_model(self, tmp_path) -> None:
        """§7.1：trajectory_states.json 必须显式声明 nq/nv/nu 与
        模型摘要——维度不再是消费者猜出来的隐式契约。"""
        states_path = _rollout(tmp_path)
        payload = json.loads(states_path.read_text(encoding="utf-8"))
        nq, nv, nu = _model_dims()
        dims = payload.get("dims") or {}
        assert dims.get("nq") == nq, f"缺 dims.nq: {payload.keys()}"
        assert dims.get("nv") == nv
        assert dims.get("nu") == nu
        assert str(payload.get("model_digest", "")).startswith("sha256:"), (
            "缺模型摘要（回放无法校验同模型恢复）"
        )

    def test_samples_record_full_qpos_qvel_ctrl(self, tmp_path) -> None:
        """§7.1：每个采样保存完整 qpos(nq)/qvel(nv)/ctrl(nu)。"""
        states_path = _rollout(tmp_path)
        payload = json.loads(states_path.read_text(encoding="utf-8"))
        nq, nv, nu = _model_dims()
        samples = payload["states"]
        assert samples
        for sample in samples[:5] + samples[-2:]:
            assert len(sample["qpos"]) == nq, (
                f"qpos 长度 {len(sample['qpos'])} != nq={nq}（截断丢失状态）"
            )
            assert len(sample["qvel"]) == nv, (
                f"qvel 长度 {len(sample['qvel'])} != nv={nv}"
            )
            assert len(sample.get("ctrl", [])) == nu, (
                "缺完整 ctrl 记录（长度 nu）"
            )

    def test_recording_source_has_no_nu_truncation(self) -> None:
        """回归守卫：记录路径不得再出现按 nu 截断的 qpos/qvel。"""
        import inspect

        from rosclaw.sandbox.backends import mujoco_cpu

        src = inspect.getsource(mujoco_cpu)
        assert "data.qpos[:nu].copy()" not in src
        assert "data.qvel[:nu].copy()" not in src


class TestReplayStateRestore:
    """sim_render 回放侧：同模型完整恢复 + 维度校验 + 旧 trace 标注。"""

    def _model(self):
        sandbox = Sandbox.create("ur5e", "empty", "mujoco")
        assert sandbox.has_physics, sandbox.load_error
        return sandbox

    def test_restore_accepts_full_nq(self) -> None:
        from rosclaw.agentd.sim_render import restore_frame_state

        sandbox = self._model()
        try:
            model, data = sandbox.physics_model, sandbox.physics_data
            nq = int(model.nq)
            qpos = [0.1 * (i + 1) for i in range(nq)]
            outcome = restore_frame_state(model, data, qpos, declared_dims=None)
            assert outcome == "restored"
            assert abs(float(data.qpos[nq - 1]) - qpos[-1]) < 1e-9
        finally:
            sandbox.close()

    def test_restore_rejects_wrong_length_no_zero_pad(self) -> None:
        """维度不符 = 受信回放拒绝——绝不补零冒充完整状态。"""
        from rosclaw.agentd.sim_render import restore_frame_state

        sandbox = self._model()
        try:
            model, data = sandbox.physics_model, sandbox.physics_data
            with pytest.raises(ValueError, match="STATE_DIMENSION"):
                restore_frame_state(
                    model, data, [0.0, 0.0], declared_dims=None,
                )
        finally:
            sandbox.close()

    def test_restore_rejects_declared_dims_mismatch(self) -> None:
        """文件声明 dims 与模型不符（跨模型引用）→ 拒绝。"""
        from rosclaw.agentd.sim_render import restore_frame_state

        sandbox = self._model()
        try:
            model, data = sandbox.physics_model, sandbox.physics_data
            nq = int(model.nq)
            with pytest.raises(ValueError, match="MODEL_DIMS_MISMATCH"):
                restore_frame_state(
                    model,
                    data,
                    [0.0] * nq,
                    declared_dims={"nq": nq + 7, "nv": 6, "nu": 6},
                )
        finally:
            sandbox.close()

    def test_legacy_truncated_sample_marked_not_zero_padded(self) -> None:
        """旧 trace（qpos 按 nu 截断、无 dims 声明）在 nq>nu 模型上
        = LEGACY_PARTIAL_STATE 诚实拒绝——不是补零继续。"""
        import mujoco

        from rosclaw.agentd.sim_render import restore_frame_state

        model = mujoco.MjModel.from_xml_string(FREEJOINT_MJCF)
        data = mujoco.MjData(model)
        assert int(model.nq) == 8 and int(model.nu) == 1
        # 旧记录只存了 nu=1 维——nq=8 的模型上不得补零冒充。
        with pytest.raises(ValueError, match="LEGACY_PARTIAL_STATE"):
            restore_frame_state(model, data, [0.5], declared_dims=None)
        # 完整 nq=8 则正常恢复（兼容回放）。
        outcome = restore_frame_state(
            model, data, [0.0] * 8, declared_dims=None,
        )
        assert outcome == "restored"

    def test_replay_source_has_no_truncation(self) -> None:
        """W00 R2 的行为锚点：回放恢复不再按 nu 截断。"""
        import inspect

        from rosclaw.agentd import sim_render

        src = inspect.getsource(sim_render)
        assert "data.qpos[: int(model.nu)]" not in src


class TestContinueSimulationCheckpoint:
    """§7.2：续仿真检查点——状态维度与模型一致性校验。"""

    def test_initial_state_wrong_dims_rejected(self, tmp_path) -> None:
        from rosclaw.sim import api

        mjcf = tmp_path / "m.xml"
        mjcf.write_text(FREEJOINT_MJCF, encoding="utf-8")
        ref, desc = api.load_model(mjcf, task_root=tmp_path)
        assert desc["nq"] == 8
        bad_state = tmp_path / "models" / "state_bad.json"
        bad_state.write_text(json.dumps({"qpos": [0.0, 0.0]}))
        with pytest.raises(ValueError, match="STATE_DIMENSION"):
            api.submit_simulation(
                ref, "state_bad", {"hold": True}, 0.01, task_root=tmp_path,
            )

    def test_cross_model_state_rejected(self, tmp_path) -> None:
        """状态记录声明的模型摘要与目标模型不符 → 拒绝续仿真。"""
        from rosclaw.sim import api

        mjcf = tmp_path / "m.xml"
        mjcf.write_text(FREEJOINT_MJCF, encoding="utf-8")
        ref, desc = api.load_model(mjcf, task_root=tmp_path)
        foreign = tmp_path / "models" / "state_foreign.json"
        foreign.write_text(json.dumps({
            "model_digest": "sha256:" + "0" * 64,
            "qpos": [0.0] * desc["nq"],
        }))
        with pytest.raises(ValueError, match="CROSS_MODEL_REF"):
            api.submit_simulation(
                ref, "state_foreign", {"hold": True}, 0.01,
                task_root=tmp_path,
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
