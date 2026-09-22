"""ROS2 bridge 适配层测试（MH25，讨论总纲 §39-§46，红→绿）。

live bridge 诚实 NOT_RUN（本机：apt 索引 404/网络抖动 + 源码
main 分支 API=Rolling 与 Jazzy hardware_interface 不兼容——
不拿 Rolling 源码硬凑冒充 Jazzy 兼容）。适配层与 fault
fail-closed 语义先行落地：桥接入时直接走同一判据。

边界（§45）：Agent 永不直接 ROS publish——适配层属于 Runtime
集成层，模型面永无 ros publish 动词。
"""

from __future__ import annotations


def test_agent_surface_has_no_ros_publish() -> None:
    """§45 边界：模型面（CLI 子命令 + P0 工具面）永无
    ros/publish/cmd_vel 动词——architecture 永久锁定。"""
    from rosclaw.agent.tool_catalog import P0_SIM_TOOLS
    from rosclaw.sim.cli import _SUBCOMMANDS

    forbidden = ("ros2 topic", "cmd_vel", "ros2 pub", "ros2 run")
    for verb in forbidden:
        assert not any(verb in tool.lower() for tool in P0_SIM_TOOLS), verb
        assert not any(verb in cmd.lower() for cmd in _SUBCOMMANDS), verb


def test_ros2_bridge_status_honest() -> None:
    """桥状态探测：rclpy/bridge 缺失即 NOT_RUN + 具体原因——
    绝不假装桥接成功。"""
    from rosclaw.sim.shadow import ros2_bridge_status

    status = ros2_bridge_status()
    assert status["status"] in ("NOT_RUN", "AVAILABLE")
    if status["status"] == "NOT_RUN":
        assert status["reason"]


def test_ros2_adapter_layer_protocol() -> None:
    """适配层协议面（§41-§42）：exact-step/clock/reset/joint_state
    契约——接口契约测试（桥接入时直接消费）。"""
    from rosclaw.sim.ros2_adapter import Ros2BridgeProtocol

    proto = Ros2BridgeProtocol()
    # exact-step：pause → command → step N → compare。
    step = proto.exact_step_request(steps=100)
    assert step["steps"] == 100
    assert step["requires_pause"] is True
    # clock gate：pause/resume/reset 后 time 不得无 generation
    # 变化地回退（§43）。
    assert proto.clock_rule["no_rewind_without_generation_change"] is True
    # joint_state 必须按名映射不按数组位置（§16/§46 REAL Log
    # First——先 read/record 不给 REAL action authority）。
    mapping = proto.joint_mapping_rule
    assert mapping["by"] == "name"
    assert mapping["forbid_positional_zip"] is True
    assert proto.real_log_first is True


def test_fault_injection_fail_closed() -> None:
    """§44：controller crash / graph loss / joint_state stale /
    clock stale / sensor stops / bridge restart——全部 fail closed
    （绝不继续使用旧状态）。"""
    from rosclaw.sim.ros2_adapter import FaultPolicy

    policy = FaultPolicy()
    for fault in (
        "controller_crash",
        "ros_graph_disconnect",
        "joint_state_stale",
        "clock_stale",
        "sensor_stops",
        "bridge_restart",
    ):
        verdict = policy.on_fault(fault)
        assert verdict["fail_closed"] is True, fault
        assert verdict["use_stale_state"] is False, fault


def test_stale_observation_not_live() -> None:
    """stale observation 绝不被认为 live（§43 实证语义）。"""
    from rosclaw.sim.ros2_adapter import ObservationFreshness

    fresh = ObservationFreshness(max_age_s=0.1)
    assert fresh.judge(age_s=0.05)["live"] is True
    stale = fresh.judge(age_s=0.5)
    assert stale["live"] is False
    assert stale["fail_closed"] is True


def test_ros2_bridge_install_attempts_recorded() -> None:
    """安装实况留档（诚实过程证据）：binary 404 / 源码 API 不兼容
    的精确原因，不粉饰。"""
    from rosclaw.sim.ros2_adapter import bridge_install_notes

    notes = bridge_install_notes()
    assert notes["binary_apt_available"] in (True, False)
    assert notes["jazzy_branch_exists"] in (True, False)
    # 诚实留档：NOT_RUN 原因可机读。
    assert isinstance(notes["not_run_reasons"], list)
