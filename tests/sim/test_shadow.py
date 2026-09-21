"""Digital Shadow 测试（MH19，0916 优化 §三十一-§三十三，红→绿）。

predict → act → observe → compare → calibrate 闭环的 ROSClaw 侧
核心：SIM 预测与"REAL 观测"（日志/数据集轨迹）比对 → SIM/REAL
residual → 分歧即建议 SysID 候选 → 校准后复比。

边界（§三十二）：Agent 永不直接 ROS publish——ROS2 桥接属
Runtime 集成层，本机 ROS2 环境缺席时诚实 NOT_RUN。
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
PEND_TRUTH = PEND_BASE.replace('damping="0.01"', 'damping="0.3"').replace(
    'model="pend_base"', 'model="pend_truth"'
)


@pytest.fixture
def backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    (tmp_path / "base.xml").write_text(PEND_BASE, encoding="utf-8")
    (tmp_path / "truth.xml").write_text(PEND_TRUTH, encoding="utf-8")
    return MujocoBackend(tmp_path)


def _observation(backend, source_file: str, qpos0: float = 0.6) -> dict:
    """"REAL 观测"：从观测源（同款/扰动模型）录一段自由摆动。"""
    source = backend.load_model(source_file)
    dataset_ref = backend.record_dataset(
        source.model_ref,
        sequences=[{"controller": {"hold": True}, "duration_s": 1.0, "qpos0": [qpos0]}],
    )
    dataset = backend.store.get(dataset_ref)
    return {"trace_ref": dataset["sequences"][0]["trace_ref"], "dataset_ref": dataset_ref}


def test_shadow_match_when_model_faithful(backend) -> None:
    """模型忠实：SIM 预测 vs 同款观测 → MATCH（residual ≈ 0），
    不建议 SysID。"""
    base = backend.load_model("base.xml")
    observation = _observation(backend, "base.xml")
    report = backend.shadow_compare(base.model_ref, observation["trace_ref"])
    assert report["verdict"] == "MATCH"
    assert report["residual"] < 1e-6
    assert report["sysid_suggestion"] in (None, "")


def test_shadow_diverged_suggests_and_calibrates(backend) -> None:
    """模型失真（真阻尼 0.3 vs 以为 0.01）：DIVERGED → 建议 SysID
    候选 → run_sysid 校准 → 复比 MATCH——predict→act→observe→
    compare→calibrate 全闭环。"""
    base = backend.load_model("base.xml")
    observation = _observation(backend, "truth.xml")

    report = backend.shadow_compare(base.model_ref, observation["trace_ref"])
    assert report["verdict"] == "DIVERGED"
    assert report["residual"] > 0.01

    # 建议的 SysIDSpec 可直接消费（train/holdout 由 observation
    # 数据集划分——这里单序列时建议重录/切分的诚实提示）。
    suggestion = report["sysid_suggestion"]
    assert suggestion["schema_version"] == "rosclaw.sim.sysid_spec.v1"
    assert suggestion["base_model_ref"] == base.model_ref
    assert suggestion["parameters"]

    # 校准闭环：多录几段做 train/holdout，跑 SysID，复比 MATCH。
    truth = backend.load_model("truth.xml")
    dataset_ref = backend.record_dataset(
        truth.model_ref,
        sequences=[
            {"controller": {"hold": True}, "duration_s": 1.0, "qpos0": [0.6]},
            {"controller": {"hold": True}, "duration_s": 1.0, "qpos0": [-0.4]},
            {"controller": {"hold": True}, "duration_s": 1.0, "qpos0": [0.9]},
        ],
    )
    receipt = backend.run_sysid(
        {
            **suggestion,
            "dataset_ref": dataset_ref,
            "train_sequences": [0, 1],
            "holdout_sequences": [2],
        }
    )
    assert receipt["verdict"] == "IMPROVED"
    calibrated_ref = receipt["candidate_model_ref"]
    re_report = backend.shadow_compare(calibrated_ref, observation["trace_ref"])
    assert re_report["verdict"] == "MATCH"
    assert re_report["residual"] < report["residual"] * 0.1


def test_shadow_boundary_no_agent_ros_publish() -> None:
    """§三十二 边界：Agent 面（sim CLI 子命令 + P0_SIM_TOOLS）
    永远不含 ROS publish 动词——ROS2 集成属 Runtime 层。"""
    from rosclaw.agent.tool_catalog import P0_SIM_TOOLS
    from rosclaw.sim.cli import _SUBCOMMANDS

    forbidden = ("ros", "publish", "cmd_vel", "topic")
    for verb in forbidden:
        assert not any(verb in tool.lower() for tool in P0_SIM_TOOLS), verb
        assert not any(verb in cmd.lower() for cmd in _SUBCOMMANDS), verb


def test_ros2_bridge_honest_not_run(backend) -> None:
    """ROS2 桥：rclpy 缺席时诚实 NOT_RUN（不假装桥接成功）。"""
    from rosclaw.sim.shadow import ros2_bridge_status

    status = ros2_bridge_status()
    if status["available"]:
        pytest.skip("ROS2 环境在——live 桥接属另一 gate")
    assert status["status"] == "NOT_RUN"
    assert "rclpy" in status["reason"].lower() or "ros" in status["reason"].lower()


def test_ros2_bridge_survives_broken_namespace(monkeypatch) -> None:
    """CI 实证：ROS 在场但 rclpy 命名空间阴影（__spec__ 未设）时
    find_spec 抛 ValueError——必须按不可导入处理不炸测试。"""
    import importlib.util

    from rosclaw.sim.shadow import ros2_bridge_status

    original = importlib.util.find_spec

    def broken_find_spec(name, package=None):
        if name == "rclpy":
            raise ValueError("rclpy.__spec__ is not set")
        return original(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", broken_find_spec)
    status = ros2_bridge_status()
    assert status["status"] == "NOT_RUN"
    assert status["available"] is False
