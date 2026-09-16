"""Observation API（PR-MH3，ADR-0014，规格 §16）。

Maturity: experimental（ADR-0000 §4）。

语义化有界观测通道——不把 raw array 无界塞给 Agent（contact 最多
50 对；图像类返回 artifact ref，属后续渲染 PR）。未知通道 /
未知目标名全部 ``OBSERVE_CHANNEL_UNKNOWN`` fail closed。
"""

from __future__ import annotations

from typing import Any

import numpy as np

MAX_CONTACT_PAIRS = 50


def _quat_from_mat(mat) -> list[float]:  # noqa: ANN001
    """旋转矩阵 → xyzw 四元数。"""
    m = np.asarray(mat, dtype=float).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w, x, y, z = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    return [float(x), float(y), float(z), float(w)]


def _contact_info(model, data) -> list[dict[str, Any]]:  # noqa: ANN001
    import mujoco

    pairs = []
    for i in range(min(data.ncon, MAX_CONTACT_PAIRS)):
        contact = data.contact[i]
        pairs.append(
            {
                "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
                or f"geom_{contact.geom1}",
                "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
                or f"geom_{contact.geom2}",
                "dist": float(contact.dist),
            }
        )
    return pairs


def observe_channels(model, data, channels: list[str]) -> dict[str, Any]:  # noqa: ANN001
    """按通道名计算观测值。"""
    import mujoco

    values: dict[str, Any] = {}
    for channel in channels:
        if channel == "joint_positions":
            values[channel] = [float(v) for v in data.qpos]
        elif channel == "joint_velocities":
            values[channel] = [float(v) for v in data.qvel]
        elif channel == "joint_forces":
            values[channel] = [float(v) for v in data.qfrc_actuator]
        elif channel == "actuator_force":
            values[channel] = [float(v) for v in data.actuator_force]
        elif channel == "com":
            values[channel] = [float(v) for v in data.subtree_com[0]]
        elif channel == "energy":
            # 3.11：mj_energyPos/Vel(m, d) -> None，结果写入 data.energy。
            mujoco.mj_energyPos(model, data)
            potential = float(data.energy[0])
            mujoco.mj_energyVel(model, data)
            kinetic = float(data.energy[1])
            values[channel] = {
                "potential": potential,
                "kinetic": kinetic,
                "total": potential + kinetic,
            }
        elif channel.startswith("body_pose:"):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, channel.split(":", 1)[1])
            if body_id < 0:
                raise ValueError(f"OBSERVE_CHANNEL_UNKNOWN: {channel}")
            values[channel] = {
                "pos": [float(v) for v in data.xpos[body_id]],
                "quat": [float(v) for v in data.xquat[body_id]],
            }
        elif channel.startswith("body_velocity:"):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, channel.split(":", 1)[1])
            if body_id < 0:
                raise ValueError(f"OBSERVE_CHANNEL_UNKNOWN: {channel}")
            values[channel] = [float(v) for v in data.cvel[body_id]]
        elif channel.startswith("site_pose:"):
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, channel.split(":", 1)[1])
            if site_id < 0:
                raise ValueError(f"OBSERVE_CHANNEL_UNKNOWN: {channel}")
            values[channel] = {
                "pos": [float(v) for v in data.site_xpos[site_id]],
                "quat": _quat_from_mat(data.site_xmat[site_id]),
            }
        elif channel.startswith("sensor:"):
            sensor_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, channel.split(":", 1)[1]
            )
            if sensor_id < 0:
                raise ValueError(f"OBSERVE_CHANNEL_UNKNOWN: {channel}")
            adr = int(model.sensor_adr[sensor_id])
            dim = int(model.sensor_dim[sensor_id])
            values[channel] = [float(v) for v in data.sensordata[adr : adr + dim]]
        elif channel == "contact_summary":
            pairs = _contact_info(model, data)
            max_penetration = min((p["dist"] for p in pairs), default=0.0)
            values[channel] = {
                "count": int(data.ncon),
                "pairs": pairs,
                "max_penetration": float(max_penetration),
            }
        elif channel == "contact_pairs":
            values[channel] = _contact_info(model, data)
        elif channel == "contact_force":
            import numpy as _np

            total = 0.0
            force6 = _np.zeros(6)
            for i in range(min(data.ncon, MAX_CONTACT_PAIRS)):
                mujoco.mj_contactForce(model, data, i, force6)
                total += abs(float(force6[0]))
            values[channel] = {"total_normal": total}
        else:
            raise ValueError(f"OBSERVE_CHANNEL_UNKNOWN: {channel}")
    return values
