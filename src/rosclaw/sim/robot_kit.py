"""第一方 Robot Kit（七审 PR-SEVEN-1）：身份/能力/执行器/策略原子装配。

此前 AgentService 只从用户 config.yaml 的 mcp_servers 装配执行能力——
默认安装的 body 是 sim/ur5e，但 UR5e 能力包永远不会被激活（用户终端
实测 action capabilities 为 0）。Robot Kit 把 body 绑定、能力目录、
SIM 执行器、安全策略打成一个原子单元：

- 活跃 body 匹配 kit 的 body_instance_template 且用户未配置/未禁用
  → 自动激活（package-relative 模块入口，无源码路径）；
- 激活是事务：identity + capabilities + executor + policy + probes
  要么全 READY 要么 BROKEN——不再有"有 identity 没动作"的假就绪；
- 用户在 config.yaml 自配同名 server 时用户优先（不覆盖自定义）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RobotKitV1:
    kit_id: str
    robot_type: str
    display_name: str
    body_instance_template: str
    mode: str
    executor_module: str
    executor_identity: str
    action_tools: tuple[str, ...]
    observation_tools: tuple[str, ...]
    compute_tools: tuple[str, ...] = ()
    approval_policy: dict[str, str] = field(default_factory=dict)
    #: 七审 PR-SEVEN-5：自然语言 Robot Resolver 关键词（机械臂/arm →
    #: arm kit）。匹配只基于 manifest 声明——无匹配即诚实空候选。
    keywords: tuple[str, ...] = ()


def load_first_party_kits() -> list[RobotKitV1]:
    """加载包内 kit manifests（发行包数据文件）。"""
    kits_dir = Path(__file__).parent / "kits"
    kits: list[RobotKitV1] = []
    for path in sorted(kits_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        executor = raw.get("executor") or {}
        capabilities = raw.get("capabilities") or {}
        kits.append(
            RobotKitV1(
                kit_id=str(raw["kit_id"]),
                robot_type=str(raw["robot_type"]),
                display_name=str(raw.get("display_name") or raw["robot_type"]),
                body_instance_template=str(raw["body_instance_template"]),
                mode=str(raw.get("mode") or "SIMULATION"),
                executor_module=str(executor["module"]),
                executor_identity=str(executor["identity"]),
                action_tools=tuple(capabilities.get("action") or ()),
                observation_tools=tuple(capabilities.get("observation") or ()),
                compute_tools=tuple(capabilities.get("compute") or ()),
                approval_policy=dict(raw.get("approval_policy") or {}),
                keywords=tuple(str(k) for k in raw.get("keywords") or ()),
            )
        )
    return kits


def required_groups_for_goal(goal: str) -> list[str]:
    """七审 PR-SEVEN-5：任务目标 → 所需能力组（service.doctor_task 与
    CLI `rosclaw doctor task` 共用同一分类规则）。

    - 绘制/轨迹类目标：trajectory + executor + verifier；
    - 其余动作目标：executor。
    """
    import re

    text = goal.strip().lower()
    if re.search(r"画|draw|star|五角星|轨迹|trajectory|trace|circle|圆", text):
        return ["trajectory", "executor", "verifier"]
    return ["executor"]


def match_kits(query: str, kits: list[RobotKitV1] | None = None) -> list[RobotKitV1]:
    """七审 PR-SEVEN-5：自然语言 → kit 候选（Robot Resolver）。

    匹配只基于 manifest 声明的 kit_id/robot_type/display_name/keywords
    （大小写不敏感子串）；无匹配返回空列表——绝不伪造候选。候选按
    命中数降序。
    """
    text = query.strip().lower()
    if not text:
        return []
    scored: list[tuple[int, RobotKitV1]] = []
    for kit in kits if kits is not None else load_first_party_kits():
        terms = (kit.kit_id, kit.robot_type, kit.display_name, *kit.keywords)
        hits = sum(1 for term in terms if term and term.lower() in text)
        if hits:
            scored.append((hits, kit))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [kit for _, kit in scored]


def kit_for_body(body_id: str, kits: list[RobotKitV1] | None = None) -> RobotKitV1 | None:
    """按活跃 body 匹配第一方 kit。"""
    for kit in kits if kits is not None else load_first_party_kits():
        if kit.body_instance_template == body_id:
            return kit
    return None


def kit_server_spec(kit: RobotKitV1) -> dict[str, Any]:
    """kit → mcp_servers 配置项（package-relative 模块入口——禁止
    引用仓库源码路径）。"""
    import sys

    return {
        "name": kit.executor_identity.removeprefix("mcp:"),
        "command": sys.executable,
        "args": ["-m", kit.executor_module],
        "supported_modes": [kit.mode],
        "required_body_types": [kit.body_instance_template],
        "observation_tools": list(kit.observation_tools),
        "action_tools": list(kit.action_tools),
        "compute_tools": list(kit.compute_tools),
        "sim_executor": True,
        # 七审 §2.5：第一方 SIM kit 的动作只改仿真状态——POLICY_AUTO
        # 的效果域依据。
        "effect_domain": "SIMULATION_STATE_ONLY",
        # PR-N5B：第一方 kit 显式声明 output_schema（canonical 输出
        # 验证依据；N5E 将收紧为 binding manifest）。
        "output_schemas": dict(_UR5E_OUTPUT_SCHEMAS),
    }


#: UR5e 第一方 kit 观测/计算工具的 canonical 输出形状（与
#: ur5e_mcp.py executor 返回一一对应；required 只列稳定核心键）。
_UR5E_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "ur5e.get_joint_state": {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean", "const": True},
            "evidence_domain": {"type": "string", "const": "simulation"},
            "joints": {"type": "array", "items": {"type": "number"}},
            "moving": {"type": "boolean"},
            "observed_at": {"type": "string"},
        },
        "required": ["ok", "evidence_domain", "joints", "moving"],
        "additionalProperties": True,
    },
    "ur5e.get_end_effector_pose": {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean", "const": True},
            "evidence_domain": {"type": "string", "const": "simulation"},
            "pose": {"type": "object"},
            "observed_at": {"type": "string"},
        },
        "required": ["ok", "evidence_domain", "pose"],
        "additionalProperties": True,
    },
    "ur5e.get_cartesian_trace": {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean", "const": True},
            "evidence_domain": {"type": "string", "const": "simulation"},
            "evidence_level": {"type": "string", "const": "COMMAND_REPLAY"},
            "trace": {"type": "object"},
            "observed_at": {"type": "string"},
        },
        "required": ["ok", "evidence_domain", "trace"],
        "additionalProperties": True,
    },
    "ur5e.plan_cartesian_path": {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean", "const": True},
            "plan_id": {"type": "string"},
            "trajectory_hash": {"type": "string"},
            "summary": {"type": "string"},
            "point_count": {"type": "integer"},
            "workspace_ok": {"type": "boolean"},
        },
        "required": ["ok", "point_count", "workspace_ok"],
        "additionalProperties": True,
    },
    "ur5e.verify_drawing": {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean", "const": True},
            "verification": {"type": "object"},
        },
        "required": ["ok", "verification"],
        "additionalProperties": True,
    },
}
