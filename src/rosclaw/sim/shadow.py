"""Digital Shadow（MH19，0916 优化 §三十一-§三十三）。

predict → act → observe → compare → calibrate 的 ROSClaw 侧核心：

- `shadow_compare`：SIM 预测（同控制器同初值重放）vs "REAL 观测"
  （日志/数据集 trace）→ SIM/REAL residual；MATCH/DIVERGED 判定；
  DIVERGED 时给出可直接消费的 SysIDSpec 建议（calibrate 入口）。
- `ros2_bridge_status`：ROS2 桥接诚实状态——rclpy 缺席即
  NOT_RUN（§三十二边界：Agent 永不直接 ROS publish，ROS2 属
  Runtime 集成层）。
"""

from __future__ import annotations

import hashlib as _hashlib
import json as _json
from typing import Any

import numpy as np

#: 确定性重放容差（同模型同初值同控制器 = 逐位一致；任何超过
#: 即真分歧，不是数值噪声）。
_MATCH_THRESHOLD = 1e-6


def _residual(predicted: list[dict[str, Any]], observed: list[dict[str, Any]]) -> dict[str, Any]:
    """分通道残差（MH21-B §18）：qpos/qvel 各 RMSE/P95/max——
    真实世界按通道有不同单位与噪声尺度，不再只有 max。"""
    import numpy as np

    qpos_errs: list[float] = []
    qvel_errs: list[float] = []
    for pred, obs in zip(predicted, observed, strict=True):
        qpos_errs.extend(abs(float(a) - float(b)) for a, b in zip(pred["qpos"], obs["qpos"], strict=True))
        qvel_errs.extend(abs(float(a) - float(b)) for a, b in zip(pred["qvel"], obs["qvel"], strict=True))

    def stats(errs: list[float]) -> dict[str, float]:
        arr = np.asarray(errs, dtype=float)
        if arr.size == 0:
            return {"rmse": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "rmse": float(np.sqrt(np.mean(np.square(arr)))),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        }

    qpos = stats(qpos_errs)
    qvel = stats(qvel_errs)
    return {
        "qpos": qpos,
        "qvel": qvel,
        # 兼容 MH19 读取方（channels.max_qpos_dev/max_qvel_dev）。
        "max_qpos_dev": qpos["max"],
        "max_qvel_dev": qvel["max"],
    }


def shadow_compare(
    backend,  # noqa: ANN001
    model_ref: str,
    observation_trace_ref: str,
    *,
    match_threshold: float = _MATCH_THRESHOLD,
    partial_threshold: float = 0.05,
    allow_clock_search: bool = True,
) -> dict[str, Any]:
    """SIM 预测 vs REAL 观测比对 → MATCH/DIVERGED + SysID 建议。

    重放语义：从观测 trace 第 0 行状态出发、用观测的控制序列
    重放 len-1 步，与第 1..n-1 行逐点比对（同模型确定性重放
    逐位一致——MATCH 容差 1e-6 是"真分歧"判据不是数值容差）。

    MH21 v2（讨论总纲 §15）：证据域强制——
    - 观测是 ObservationTraceV2 且 HARDWARE_RECORDED →
      REAL_SHADOW_COMPARE（身体身份 + joint schema 必须一致）；
    - 其余（SIMULATION/REPLAY/裸 trace）→ SELF_TEST——
      SIM 观测永远不能产出 REAL 结论（§3 SIM ≠ REAL）。
    """
    import mujoco

    from rosclaw.sim.backends.mujoco import rollout as rollout_mod

    manifest = backend._manifest(model_ref)
    spec = backend._spec_from_manifest(manifest)
    model, _ = backend._compile_smoke(spec)

    observation = backend.store.get(observation_trace_ref)
    shadow_mode = "SELF_TEST"
    observation_meta: dict[str, Any] = {}
    report_ref = observation_trace_ref  # 报告指向观测记录本身（不解包）
    if isinstance(observation, dict) and observation.get("kind") == "observation_trace_v2":
        observation_meta = observation
        domain = observation.get("evidence_domain", "SIMULATION")
        if domain == "HARDWARE_RECORDED":
            # 身体身份：观测侧结构签名 hash 与目标模型重算一致——
            # 不符即"观测不是这个身体的"（SH05）。
            if observation.get("body_snapshot_hash") != body_snapshot_hash(model):
                raise ValueError(
                    "SHADOW_BODY_IDENTITY_MISMATCH: observation 的 body_snapshot_hash "
                    "与目标模型结构签名不符"
                )
            # joint schema：按名校验（不按数组位置猜——§16）。
            schema_joints = {entry["joint"] for entry in observation.get("joint_schema", [])}
            model_joints = {entry["joint"] for entry in canonical_joint_schema(model)}
            if not schema_joints or not schema_joints <= model_joints:
                raise ValueError(
                    "SHADOW_JOINT_SCHEMA_MISMATCH: observation joint_schema 与模型不符"
                    f"（多余: {sorted(schema_joints - model_joints)}）"
                )
            shadow_mode = "REAL_SHADOW_COMPARE"
        observation_trace_ref = observation["trace_ref"]
        observation = backend.store.get(observation_trace_ref)

    states = observation.get("states") or []
    if len(states) < 2:
        raise ValueError(f"SHADOW_OBSERVATION_EMPTY: {observation_trace_ref}")
    controller = observation.get("controller") or {"hold": True}
    data = mujoco.MjData(model)
    data.qpos[:] = [float(v) for v in states[0]["qpos"]]
    data.qvel[:] = [float(v) for v in states[0]["qvel"]]
    mujoco.mj_forward(model, data)

    # 用观测 trace 的 ctrl 序列重放（预测路径与观测同控制输入）。
    ctrl_rows = [[float(v) for v in row.get("ctrl", [])] for row in states]
    if controller.get("ctrl_series"):
        plan = rollout_mod.validate_controller(controller, model.nu)
    elif model.nu and any(any(abs(v) > 0 for v in row) for row in ctrl_rows):
        plan = rollout_mod.validate_controller({"ctrl_series": ctrl_rows}, model.nu)
    else:
        plan = rollout_mod.validate_controller({"hold": True}, model.nu)
    # 观测与预测 trace 都是有界采样（stride = ceil(steps/
    # max_record_points)）——采样策略同源时行时间点一致，按 t 对齐
    # 逐点比对（实证：1.0s/500 步的 trace 只有 251 行）。
    dt = float(model.opt.timestep)
    total_steps = round((float(states[-1]["t"]) - float(states[0]["t"])) / dt)
    predicted, _ = rollout_mod.run_rollout(model, data, plan=plan, steps=total_steps)

    # MH21-B ClockAlignment（§17）：真实机器人数据一定有 jitter/
    # offset/dropped——观测时间映射到预测网格：先估计整体 offset
    # （观测首行 t 与预测网格原点之差），再逐行 nearest（容差
    # 半个 dt 内直接取）/线性插值（网格间）。
    obs_times = [float(row["t"]) for row in states]
    # allow_clock_search=False：不搜 offset（观测时钟域与 sim 网格
    # 无关时如实 NOT_COMPARABLE，不硬凑对齐）。
    estimated_offset = (obs_times[0] - float(predicted[0]["t"])) if allow_clock_search else 0.0
    pred_times = np.array([float(row["t"]) for row in predicted]) + estimated_offset

    def _predicted_at(tau: float, mode: str) -> dict[str, Any] | None:
        """在预测轨迹上取 τ 时刻的值（nearest / linear）。"""
        import bisect

        idx = bisect.bisect_left(pred_times, tau)
        candidates = [i for i in (idx - 1, idx) if 0 <= i < len(pred_times)]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda i: abs(pred_times[i] - tau))
        if mode == "nearest":
            return predicted[nearest] if abs(pred_times[nearest] - tau) <= dt / 2 + 1e-12 else None
        # linear：网格间线性插值。
        if idx == 0 or idx >= len(pred_times):
            return predicted[nearest] if abs(pred_times[nearest] - tau) <= dt / 2 + 1e-12 else None
        t_lo, t_hi = pred_times[idx - 1], pred_times[idx]
        if t_hi - t_lo <= 0:
            return predicted[idx - 1]
        alpha = (tau - t_lo) / (t_hi - t_lo)
        lo, hi = predicted[idx - 1], predicted[idx]
        return {
            "qpos": [float(a) * (1 - alpha) + float(b) * alpha for a, b in zip(lo["qpos"], hi["qpos"], strict=True)],
            "qvel": [float(a) * (1 - alpha) + float(b) * alpha for a, b in zip(lo["qvel"], hi["qvel"], strict=True)],
        }

    pairs = []
    dropped = 0
    for obs in states[1:]:
        aligned = _predicted_at(float(obs["t"]), "linear")
        if aligned is None:
            dropped += 1
            continue
        pairs.append((aligned, obs))
    aligned_pairs = len(pairs)
    # drop 统计 = 名义节拍缺口：观测自身 cadence（中位行间隔）
    # 推断期望行数，缺口即丢失样本（§17 dropped samples）。
    obs_intervals = np.diff(np.asarray(obs_times, dtype=float))
    nominal_dt = float(np.median(obs_intervals)) if obs_intervals.size else dt
    span = obs_times[-1] - obs_times[0]
    expected_rows = int(round(span / nominal_dt)) + 1 if nominal_dt > 0 else len(states)
    cadence_dropped = max(0, expected_rows - len(states))
    dropped += cadence_dropped
    if aligned_pairs == 0:
        report: dict[str, Any] = {
            "schema_version": "rosclaw.sim.shadow_report.v1",
            "model_ref": model_ref,
            "observation_ref": report_ref,
            "shadow_mode": shadow_mode,
            "observation_meta": observation_meta,
            "verdict": "NOT_COMPARABLE",
            "residual": None,
            "channels": {},
            "clock_alignment": {
                "estimated_offset_s": estimated_offset,
                "aligned_pairs": 0,
                "dropped_samples": dropped,
                "drop_ratio": 1.0,
            },
            "sysid_suggestion": None,
            "trust_level": "SIMULATED",
            "usable_for_real_execution": False,
        }
        return report
    residuals = _residual([p for p, _ in pairs], [o for _, o in pairs])
    residual = max(residuals["max_qpos_dev"], residuals["max_qvel_dev"])
    if residual <= match_threshold:
        verdict = "MATCH"
    elif residual <= partial_threshold:
        verdict = "PARTIAL_MATCH"
    else:
        verdict = "DIVERGED"
    clock_alignment = {
        "estimated_offset_s": estimated_offset,
        "aligned_pairs": aligned_pairs,
        "dropped_samples": dropped,
        "drop_ratio": dropped / max(1, dropped + aligned_pairs),
    }

    report: dict[str, Any] = {
        "schema_version": "rosclaw.sim.shadow_report.v1",
        "model_ref": model_ref,
        "observation_ref": report_ref,
        "shadow_mode": shadow_mode,
        "observation_meta": observation_meta,
        "verdict": verdict,
        "residual": residual,
        "channels": residuals,
        "clock_alignment": clock_alignment,
        "sysid_suggestion": None,
        "trust_level": "SIMULATED",
        "usable_for_real_execution": False,
    }
    if verdict == "DIVERGED":
        # calibrate 入口：SysIDSpec 建议（v1 参数族 = 全部 joint
        # damping——sim/real 最常见分歧源；train/holdout 由调用方
        # 按数据集划分，单序列时诚实标注需补录）。
        parameters = []
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and int(model.jnt_type[j]) in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                parameters.append(
                    {"type": "joint_damping", "joint": name, "min": 0.001, "max": 10.0}
                )
        report["sysid_suggestion"] = {
            "schema_version": "rosclaw.sim.sysid_spec.v1",
            "base_model_ref": model_ref,
            "dataset_ref": observation_trace_ref,
            "parameters": parameters,
            "train_sequences": [],
            "holdout_sequences": [],
            "note": "单序列观测不足以划分 train/holdout——请补录序列后填入索引（§26.3）",
        }
    return report


def _importable(module: str) -> bool:
    """importlib 探测的防御封装——命名空间阴影会让 find_spec
    抛 ValueError（CI 实证：ROS Jazzy 在场但 rclpy.__spec__ 未设），
    损坏状态按不可导入处理（诚实 NOT_RUN，绝不误判 AVAILABLE）。"""
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ValueError, ImportError):
        return False


def ros2_bridge_status() -> dict[str, Any]:
    """ROS2 桥接诚实状态（§三十一-§三十二）。

    rclpy/mujoco_ros2_control 缺席 → NOT_RUN（不假装桥接成功；
    ROS2 属 Runtime 集成层，Agent 面永远无 ROS publish 动词）。
    """
    if not _importable("rclpy"):
        return {
            "status": "NOT_RUN",
            "available": False,
            "reason": "rclpy not importable in this environment (ROS2 setup not sourced)",
        }
    if not _importable("mujoco_ros2_control"):
        return {
            "status": "NOT_RUN",
            "available": False,
            "reason": "mujoco_ros2_control not installed (ROS2 present but bridge missing)",
        }
    return {"status": "AVAILABLE", "available": True, "reason": "ros2_control bridge importable"}


# ---------------------------------------------------------------- MH21 v2


def body_snapshot_hash(model) -> str:  # noqa: ANN001, ANN202
    """身体结构签名 hash（joint 名/类型/qpos/dof 地址）——
    观测与模型比对的身份依据（不按数组位置猜）。"""
    import mujoco

    joints = [
        (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}",
            int(model.jnt_type[i]),
            int(model.jnt_qposadr[i]),
            int(model.jnt_dofadr[i]),
        )
        for i in range(model.njnt)
    ]
    canonical = _json.dumps(joints, separators=(",", ":"))
    return "sha256:" + _hashlib.sha256(canonical.encode()).hexdigest()


def canonical_joint_schema(model) -> list[dict[str, Any]]:  # noqa: ANN001
    """模型规范 joint schema（id 序）。"""
    import mujoco

    return [
        {
            "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}",
            "qpos_adr": int(model.jnt_qposadr[i]),
            "dof_adr": int(model.jnt_dofadr[i]),
        }
        for i in range(model.njnt)
    ]
