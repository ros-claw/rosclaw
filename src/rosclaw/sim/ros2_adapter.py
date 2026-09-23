"""ROS2 bridge 适配层（MH25，讨论总纲 §39-§46）。

live bridge 诚实 NOT_RUN（本机实况见 bridge_install_notes）。
适配层协议面 + fault fail-closed 语义先行——桥接入时直接走
同一判据，不重写。

边界（§45）：适配层属 Runtime 集成层；Agent 面（CLI/工具面）
永无 ros publish 动词（architecture test 锁定）。
"""

from __future__ import annotations

import importlib.util
from typing import Any


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ValueError, ImportError):
        return False


class Ros2BridgeProtocol:
    """exact-step / clock / joint mapping / REAL-log-first 契约
    （§41-§43/§46）——桥接入时直接消费的协议面。"""

    def __init__(self) -> None:
        self.clock_rule = {
            "no_rewind_without_generation_change": True,
            "stale_is_not_live": True,
        }
        self.joint_mapping_rule = {
            "by": "name",
            "forbid_positional_zip": True,
        }
        self.real_log_first = True  # §46：先 read/record，不给 REAL action authority

    def exact_step_request(self, steps: int) -> dict[str, Any]:
        """§42 exact-step：pause → command → step N → compare。"""
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError("ROS2_STEP_INVALID: steps must be a positive int")
        return {
            "steps": steps,
            "requires_pause": True,
            "compare": ["joint_state", "native_mujoco_state", "clock"],
        }


class FaultPolicy:
    """§44 fault injection 判据：全部 fail closed——绝不继续
    使用旧状态。"""

    def on_fault(self, fault: str) -> dict[str, Any]:
        known = {
            "controller_crash",
            "ros_graph_disconnect",
            "joint_state_stale",
            "clock_stale",
            "sensor_stops",
            "bridge_restart",
        }
        if fault not in known:
            raise ValueError(f"ROS2_FAULT_UNKNOWN: {fault!r}")
        return {
            "fault": fault,
            "fail_closed": True,
            "use_stale_state": False,
            "action": "stop_and_report",
        }


class ObservationFreshness:
    """§43：stale observation 绝不被认为 live。"""

    def __init__(self, *, max_age_s: float) -> None:
        self.max_age_s = float(max_age_s)

    def judge(self, *, age_s: float) -> dict[str, Any]:
        age = float(age_s)
        if age <= self.max_age_s:
            return {"live": True, "fail_closed": False, "age_s": age}
        return {"live": False, "fail_closed": True, "age_s": age}


def bridge_install_notes() -> dict[str, Any]:
    """安装实况留档（诚实过程证据，不粉饰）。

    2026-09-23 更新：binary ros-jazzy-mujoco-ros2-control 0.1.1 已装通
    （tuna 镜像；MH25 时 stale index 404 已解除），G43 exact-step
    agreement live PASS（docs/reports/ros2-bridge/01）。下列
    not_run_reasons 为 MH25 当时的历史留档，现状以 G43 门表为准。
    """
    notes: dict[str, Any] = {
        "binary_apt_available": _importable("mujoco_ros2_control"),
        "jazzy_branch_exists": False,
        "rclpy_importable": _importable("rclpy"),
        "not_run_reasons": [],
        "resolution_2026_09_23": (
            "binary 0.1.1 经 tuna 镜像装通；G43 live AGREEMENT（单摆 "
            "500 步 qpos/qvel 逐位一致）——见 docs/reports/ros2-bridge/01"
        ),
    }
    if not notes["rclpy_importable"]:
        notes["not_run_reasons"].append(
            "rclpy not importable from rosclaw venv (needs ROS Jazzy setup sourced)"
        )
    if not notes["binary_apt_available"]:
        notes["not_run_reasons"].append(
            "binary ros-jazzy-mujoco-ros2-control 未安装：apt 索引 404（索引过期）"
            "且 packages.ros.org 镜像本轮网络不可达；源码 main 分支 API 面向"
            " Rolling（hardware_class_type/ResourceManagerParams），与 Jazzy"
            " hardware_interface 不兼容——不拿 Rolling 源码硬凑冒充 Jazzy 兼容"
        )
    return notes
