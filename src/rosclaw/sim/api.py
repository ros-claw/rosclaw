"""最小通用编程接口（W02，规格 §6.4）——Pi 写脚本组合能力的
可导入入口。六类入口复用同一执行器/引用系统，不各自维护
PlanStore。

```text
load_model(asset_ref | task_local_mjcf) -> model_ref + body_description
observe(model_ref, channels, at_time) -> observation_ref
submit_simulation(model_ref, initial_state_ref, controller_or_trajectory,
                  duration, recording_spec) -> operation_ref
read_trace(trace_ref, fields, interval) -> bounded_data | data_file
render(trace_ref, render_spec) -> operation_ref
export(artifact_ref, destination) -> exported_file
```

边界：
- task_local_mjcf 在 task_root 内解析/复制——模型导入不直接执行
  任意外部插件（native plugin 是另一权限类别）；
- 控制序列仿真记录**完整状态**（qpos nq / qvel nv / ctrl nu——
  W03 的状态契约在此先行落地）；
- 输出标 agent-generated experiment——模型编写脚本的输出不假装
  来自独立受信验证。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from rosclaw.sim.model_inspect import inspect_mjcf


def load_model(
    asset_ref: str | Path, *, task_root: Path | str,
) -> tuple[str, dict[str, Any]]:
    """加载模型（asset_ref 为 zoo 内名称或 task_root 内 MJCF 路径）。

    返回 (model_ref, body_description)。task-local 文件复制进
    task_root/models/（引用可跨进程解析；不执行外部插件）。
    """
    task_root = Path(task_root)
    path = Path(asset_ref)
    if not path.is_absolute():
        # zoo 内资产名（e-urdf-zoo/<name>/robot.mjcf.xml）或任务相对路径。
        from rosclaw.runtime.eurdf_loader import _default_zoo_path

        zoo_candidate = _default_zoo_path() / str(asset_ref) / "robot.mjcf.xml"
        task_candidate = task_root / str(asset_ref)
        if zoo_candidate.exists():
            path = zoo_candidate
        elif task_candidate.exists():
            path = task_candidate
        else:
            raise ValueError(
                f"MODEL_NOT_FOUND: {asset_ref!r}（已查 zoo 与 task_root）"
            )
    if not path.exists():
        raise ValueError(f"MODEL_NOT_FOUND: {path}")
    info = inspect_mjcf(path)
    models_dir = task_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    staged = models_dir / path.name
    if path.resolve() != staged.resolve():
        shutil.copy2(path, staged)
    digest_short = info.model_digest.removeprefix("sha256:")[:16]
    model_ref = f"model_{digest_short}"
    (models_dir / f"{model_ref}.json").write_text(json.dumps({
        "model_ref": model_ref,
        "path": str(staged),
        "source_path": str(path),
        **info.to_dict(),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return model_ref, info.to_dict()


def _load_model_record(model_ref: str, task_root: Path) -> dict[str, Any]:
    record_path = task_root / "models" / f"{model_ref}.json"
    if not record_path.exists():
        raise ValueError(f"REF_NOT_FOUND: model {model_ref!r} 不在 task_root")
    return json.loads(record_path.read_text(encoding="utf-8"))


def observe(
    model_ref: str,
    channels: list[str],
    *,
    at_time: float | None = None,
    task_root: Path | str,
    initial_state_ref: str | None = None,
) -> str:
    """只读观测（返回 observation_ref——内容随记录持久化）。

    channels：joint_positions / joint_velocities / eef_pose:<site>。
    at_time=None = 初始状态（或 initial_state_ref 指定状态）。
    """
    import mujoco
    import numpy as np

    task_root = Path(task_root)
    record = _load_model_record(model_ref, task_root)
    model = mujoco.MjModel.from_xml_path(record["path"])
    data = mujoco.MjData(model)
    if initial_state_ref:
        state = _load_state_record(initial_state_ref, task_root)
        data.qpos[:] = np.array(state["qpos"], dtype=float)
        data.qvel[:] = np.array(state.get("qvel", [0.0] * model.nv), dtype=float)
    mujoco.mj_forward(model, data)

    out: dict[str, Any] = {"model_ref": model_ref, "time": float(data.time)}
    if "joint_positions" in channels:
        out["joint_positions"] = [round(float(v), 8) for v in data.qpos]
    if "joint_velocities" in channels:
        out["joint_velocities"] = [round(float(v), 8) for v in data.qvel]
    for ch in channels:
        if ch.startswith("eef_pose:"):
            site_name = ch.split(":", 1)[1]
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                raise ValueError(f"SITE_NOT_FOUND: {site_name!r}")
            # site_xquat 在部分 mujoco 版本不存在——由 site_xmat
            # 旋转矩阵转显式命名的 xyzw 四元数（§6.2 命名契约）。
            from rosclaw.agentd.sim_trajectory import _mat_to_quat_xyzw

            mat = [
                [float(data.site_xmat[site_id][i]) for i in range(0, 3)],
                [float(data.site_xmat[site_id][i]) for i in range(3, 6)],
                [float(data.site_xmat[site_id][i]) for i in range(6, 9)],
            ]
            out[ch] = {
                "position": [round(float(v), 8) for v in data.site_xpos[site_id]],
                "quaternion_xyzw": _mat_to_quat_xyzw(mat),
            }
    obs_id = "obs_" + hashlib.sha256(
        json.dumps(out, sort_keys=True).encode()
    ).hexdigest()[:16]
    (task_root / "models" / f"{obs_id}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8",
    )
    return obs_id


def _load_state_record(state_ref: str, task_root: Path) -> dict[str, Any]:
    path = task_root / "models" / f"{state_ref}.json"
    if not path.exists():
        raise ValueError(f"REF_NOT_FOUND: state {state_ref!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def submit_simulation(
    model_ref: str,
    initial_state_ref: str | None,
    controller_or_trajectory: dict[str, Any],
    duration: float,
    recording_spec: dict[str, Any] | None = None,
    *,
    task_root: Path | str,
) -> str:
    """控制序列仿真——记录完整状态（qpos/qvel/ctrl，W03 契约）。

    controller_or_trajectory：{"ctrl_series": [[...], ...]}（每步
    nu 维）或 {"hold": true}（零控制保持）。模型编写的控制器
    不得写权威 metrics——输出标 agent-generated experiment。
    返回 operation_ref（trace 记录可经 read_trace/render 消费）。
    """
    import mujoco
    import numpy as np

    task_root = Path(task_root)
    record = _load_model_record(model_ref, task_root)
    model = mujoco.MjModel.from_xml_path(record["path"])
    data = mujoco.MjData(model)
    if initial_state_ref:
        state = _load_state_record(initial_state_ref, task_root)
        data.qpos[:] = np.array(state["qpos"], dtype=float)
        if state.get("qvel"):
            data.qvel[:] = np.array(state["qvel"], dtype=float)

    ctrl_series = controller_or_trajectory.get("ctrl_series")
    if ctrl_series is None and not controller_or_trajectory.get("hold"):
        raise ValueError(
            "CONTROLLER_INVALID: 需要 ctrl_series 或 hold（动态控制器"
            "进程是受限能力，见规格 §6.4）"
        )
    if ctrl_series:
        for step_ctrl in ctrl_series:
            if len(step_ctrl) != int(model.nu):
                raise ValueError(
                    f"CTRL_DIMENSION: 控制向量长度 {len(step_ctrl)} != "
                    f"nu={int(model.nu)}"
                )
    steps = max(1, int(round(float(duration) / float(model.opt.timestep))))
    states: list[dict[str, Any]] = []
    started = time.monotonic()
    for step in range(steps):
        if ctrl_series:
            data.ctrl[:] = np.array(
                ctrl_series[min(step, len(ctrl_series) - 1)], dtype=float,
            )
        mujoco.mj_step(model, data)
        states.append({
            "t": round(float(data.time), 6),
            "qpos": [round(float(v), 8) for v in data.qpos],
            "qvel": [round(float(v), 8) for v in data.qvel],
            "ctrl": [round(float(v), 8) for v in data.ctrl],
        })
    digest = hashlib.sha256(
        json.dumps(states, sort_keys=True).encode()
    ).hexdigest()[:16]
    op_id = f"op_{digest}"
    payload = {
        "operation_id": op_id,
        "kind": "agent_generated_experiment",
        "model_ref": model_ref,
        "model_digest": record["model_digest"],
        "duration_s": float(duration),
        "timestep_s": float(model.opt.timestep),
        "steps": steps,
        "wall_ms": round((time.monotonic() - started) * 1000, 1),
        "states_digest": "sha256:" + hashlib.sha256(
            json.dumps(states, sort_keys=True).encode()
        ).hexdigest(),
        "states": states,
    }
    (task_root / "models" / f"{op_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8",
    )
    return op_id


def read_trace(
    trace_ref: str,
    fields: list[str] | None = None,
    interval: float | None = None,
    *,
    task_root: Path | str,
    max_points: int = 480,
) -> dict[str, Any]:
    """读取仿真记录（有界——默认 decimate 到 max_points）。"""
    task_root = Path(task_root)
    payload = _load_state_record(trace_ref, task_root)
    states = payload.get("states") or []
    if interval:
        states = [s for s in states if s["t"] % max(float(interval), 1e-9)
                  < float(payload.get("timestep_s", 0.002))]
    if len(states) > max_points:
        step = len(states) / max_points
        states = [states[round(i * step)] for i in range(max_points)]
    if fields:
        states = [
            {k: v for k, v in s.items() if k == "t" or k in fields}
            for s in states
        ]
    return {
        "operation_id": payload.get("operation_id", trace_ref),
        "model_ref": payload.get("model_ref", ""),
        "model_digest": payload.get("model_digest", ""),
        "kind": payload.get("kind", ""),
        "states": states,
    }


def render(
    trace_ref: str,
    render_spec: dict[str, Any],
    *,
    task_root: Path | str,
) -> str:
    """渲染仿真记录（operation 目录独立——同 trace 不同视角/格式
    互不覆盖）。返回 render operation_ref。"""
    task_root = Path(task_root)
    payload = _load_state_record(trace_ref, task_root)
    states = payload.get("states") or []
    if not states:
        raise ValueError(f"RENDER_INPUT_MISSING: {trace_ref} 无 states")
    from rosclaw.agentd.sim_render import render_operation

    return render_operation(
        task_root, trace_ref, states,
        model_digest=str(payload.get("model_digest", "")),
        camera=str(render_spec.get("camera", "follow")),
        outputs=list(render_spec.get("outputs") or ["gif"]),
        fps=float(render_spec.get("fps", 12.0)),
        duration_s=float(payload.get("duration_s", 0.0)),
    )


def export(
    artifact_ref: str, destination: Path | str, *, task_root: Path | str,
) -> Path:
    """导出产物到指定路径（目标存在不静默覆盖）。"""
    task_root = Path(task_root)
    destination = Path(destination)
    source = task_root / "renders" / artifact_ref
    if not source.exists():
        source = task_root / "models" / f"{artifact_ref}.json"
    if not source.exists():
        raise ValueError(f"ARTIFACT_NOT_FOUND: {artifact_ref!r}")
    if destination.exists():
        raise ValueError(f"EXPORT_TARGET_EXISTS: {destination}（不静默覆盖）")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


__all__ = [
    "load_model", "observe", "submit_simulation",
    "read_trace", "render", "export",
]
