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
    approval_policy: dict[str, str] = field(default_factory=dict)


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
                approval_policy=dict(raw.get("approval_policy") or {}),
            )
        )
    return kits


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
        "sim_executor": True,
        # 七审 §2.5：第一方 SIM kit 的动作只改仿真状态——POLICY_AUTO
        # 的效果域依据。
        "effect_domain": "SIMULATION_STATE_ONLY",
    }
