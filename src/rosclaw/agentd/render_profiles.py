"""RenderProfile 注册表（0902 审计 R2-2，§4.3）——渲染器对任意本体
的一等输入，不按本体名 hardcode。

每个登记本体一份 RenderProfileV1（契约见 contracts/agent/
render_spec.py）：sandbox robot 解析、root、qpos 映射、EEF frame、
默认相机。未登记本体 → RENDER_PROFILE_MISSING 诚实失败（不猜、
不回落默认本体冒充）。
"""

from __future__ import annotations

from rosclaw.contracts.agent.render_spec import RenderCameraV1, RenderProfileV1

#: 已登记本体的渲染档案（数据，不是代码分支——新本体加条目即可）。
_REGISTRY: dict[str, RenderProfileV1] = {
    "sim/ur5e": RenderProfileV1(
        body_id="sim/ur5e",
        root_body="base_link",
        # UR5e canonical MJCF 关节序（e-urdf-zoo/ur5e）。
        qpos_mapping={
            "shoulder_pan_joint": 0,
            "shoulder_lift_joint": 1,
            "elbow_joint": 2,
            "wrist_1_joint": 3,
            "wrist_2_joint": 4,
            "wrist_3_joint": 5,
        },
        eef_frame="ee_link",
        default_cameras=[RenderCameraV1(preset="follow")],
    ),
}

#: body_id → Sandbox robot_id（Sandbox._load_model 的别名表之外，
#: 渲染档案显式记录解析结果）。
_SANDBOX_ROBOT_ID: dict[str, str] = {
    "sim/ur5e": "ur5e",
}


def resolve_render_profile(body_id: str) -> RenderProfileV1:
    """body_id（如 sim/ur5e）→ RenderProfileV1；未登记诚实失败。"""
    profile = _REGISTRY.get(body_id)
    if profile is None:
        raise ValueError(
            f"RENDER_PROFILE_MISSING: 本体 {body_id!r} 无渲染档案"
            f"（已登记：{sorted(_REGISTRY)}）"
        )
    return profile


def sandbox_robot_id(body_id: str) -> str:
    """body_id → Sandbox.create 的 robot_id（与档案一致）。"""
    resolve_render_profile(body_id)  # 未登记即抛
    return _SANDBOX_ROBOT_ID[body_id]


__all__ = ["resolve_render_profile", "sandbox_robot_id"]
