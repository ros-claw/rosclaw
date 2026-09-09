"""可推导的身体档案（W02，规格 §6.1）——从加载的 MJCF 推出，
不从注册表或机器人名称猜。

力学模型（MJCF/URDF）是维度与关节顺序的唯一权威：内容摘要、
nq/nv/nu、joint 名称与类型、qpos/dof 地址、执行器→关节映射、
控制类型与限制、传感器、camera/site。RenderProfile 只放展示
偏好（默认视角/颜色/EEF 偏好），不复制这份力学真相。

没有夹爪不能从机器人名称猜有夹爪——`gripper` 只反映模型里
实际存在的执行器驱动夹持机构（本轮：存在名为 gripper 的
actuated joint 才为 True，否则恒 False）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelInspection:
    """从 MJCF 推出的身体档案（只读——不是 RenderProfile）。"""

    model_digest: str
    nq: int
    nv: int
    nu: int
    joints: list[dict[str, Any]] = field(default_factory=list)
    actuators: list[dict[str, Any]] = field(default_factory=list)
    sensors: list[dict[str, Any]] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    sites: list[str] = field(default_factory=list)
    gripper: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_digest": self.model_digest,
            "nq": self.nq, "nv": self.nv, "nu": self.nu,
            "joints": self.joints,
            "actuators": self.actuators,
            "sensors": self.sensors,
            "cameras": self.cameras,
            "sites": self.sites,
            "gripper": self.gripper,
        }


_JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


def inspect_mjcf(path: Path | str) -> ModelInspection:
    """加载 MJCF 并推出身体档案。加载失败给结构化错误。"""
    import mujoco

    path = Path(path)
    if not path.exists():
        raise ValueError(f"MODEL_NOT_FOUND: {path}")
    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:
        raise ValueError(f"MODEL_LOAD_ERROR: {path}: {exc}") from exc
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    joints: list[dict[str, Any]] = []
    for i in range(model.njnt):
        joints.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}",
            "type": _JOINT_TYPES.get(int(model.jnt_type[i]), str(int(model.jnt_type[i]))),
            "qpos_addr": int(model.jnt_qposadr[i]),
            "dof_addr": int(model.jnt_dofadr[i]),
        })

    actuators: list[dict[str, Any]] = []
    for i in range(model.nu):
        joint_id = int(model.actuator_trnid[i][0])
        joint_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_id >= 0 else ""
        )
        actuators.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}",
            "joint": joint_name or f"joint_{joint_id}",
            "ctrlrange": [float(model.actuator_ctrlrange[i][0]),
                          float(model.actuator_ctrlrange[i][1])],
        })

    sensors: list[dict[str, Any]] = []
    for i in range(model.nsensor):
        sensors.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) or f"sensor_{i}",
            "type": str(mujoco.mjtSensor(model.sensor_type[i]).name).removeprefix("mjSENS_").lower(),
        })

    cameras = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"camera_{i}"
        for i in range(model.ncam)
    ]
    sites = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or f"site_{i}"
        for i in range(model.nsite)
    ]

    # 夹爪：只反映实际存在的执行器驱动夹持机构——不从名字猜。
    actuator_joints = {a["joint"] for a in actuators}
    gripper = any("gripper" in j for j in actuator_joints)

    return ModelInspection(
        model_digest=digest,
        nq=int(model.nq), nv=int(model.nv), nu=int(model.nu),
        joints=joints, actuators=actuators, sensors=sensors,
        cameras=cameras, sites=sites, gripper=gripper,
    )


__all__ = ["ModelInspection", "inspect_mjcf"]
