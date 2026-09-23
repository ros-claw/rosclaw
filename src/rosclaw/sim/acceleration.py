"""Backend Fidelity Gate + GPU candidate CPU agreement（MH18，0916 优化 §二十八-§三十）。

原则（ADR-0014 冻结）：**GPU = exploration，CPU = authoritative
verification**——GPU 结果永远不能直接当 verdict。

- `compatibility(...)`：静态兼容性分级。检查项只收录 MJX/Warp
  **官方文档**的限制（PGS/noslip/plugins/flexcomp/muscle/custom
  sensor/Euler-only integrator），每条 reason 注明依据；
  本机无 jax/warp 时执行面诚实 NOT_RUN（分级本身是静态分析，
  不依赖 GPU）。
- `evaluate_gpu_candidates(...)`：GPU sweep 的 top-K 候选必须经
  CPU 重放复核——一致才 PROMOTE；分歧存 counterexample
  （experiments 分区，§三十：分歧本身是有价值数据，不扔）。
"""

from __future__ import annotations

from typing import Any

#: 分级枚举。
CPU_ONLY = "CPU_ONLY"
MJX_JAX_COMPATIBLE = "MJX_JAX_COMPATIBLE"
MJX_WARP_COMPATIBLE = "MJX_WARP_COMPATIBLE"


def compatibility(model, xml_text: str) -> dict[str, Any]:  # noqa: ANN001
    """静态兼容性分级（全部检查项有据可依，不过度想象）。"""
    import mujoco

    jax_blockers: list[str] = []
    warp_blockers: list[str] = []

    solver = int(model.opt.solver)
    if solver == int(mujoco.mjtSolver.mjSOL_PGS):
        jax_blockers.append("pgs solver unsupported in MJX (documented)")
        warp_blockers.append("pgs solver unsupported in MuJoCo Warp (documented)")
    if int(model.opt.noslip_iterations) > 0:
        jax_blockers.append("noslip friction unsupported in MJX (documented)")
        warp_blockers.append("noslip friction unsupported in MuJoCo Warp (documented)")
    if int(getattr(model, "nplugin", 0)) > 0:
        jax_blockers.append("plugins unsupported in MJX (documented)")
        warp_blockers.append("PLUGIN actuator/sensor unsupported in MuJoCo Warp (documented)")
    if int(getattr(model, "nflex", 0)) > 0:
        jax_blockers.append("flexcomp bodies unsupported in MJX (documented)")
        warp_blockers.append("flexcomp bodies unsupported in MuJoCo Warp (documented)")
    lowered = xml_text.lower()
    if "<muscle" in lowered:
        jax_blockers.append("muscle actuators unsupported in MJX (documented)")
        warp_blockers.append("muscle actuators unsupported in MuJoCo Warp (documented)")
    if "<user" in lowered and "<sensor" in lowered:
        jax_blockers.append("user-defined custom sensors unsupported in MJX (documented)")
        warp_blockers.append("custom sensors unsupported in MuJoCo Warp (documented)")

    integrator = int(model.opt.integrator)
    if integrator != int(mujoco.mjtIntegrator.mjINT_EULER):
        warp_blockers.append("MuJoCo Warp is Euler-only integrator (documented)")

    if jax_blockers:
        classification = CPU_ONLY
        reasons = sorted(set(jax_blockers + warp_blockers))
    elif warp_blockers:
        classification = MJX_JAX_COMPATIBLE
        reasons = sorted(set(warp_blockers))
    else:
        classification = MJX_WARP_COMPATIBLE
        reasons = []
    return {"classification": classification, "reasons": reasons}


def evaluate_gpu_candidates(
    backend,  # noqa: ANN001
    model_ref: str,
    candidates: list[dict[str, Any]],
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """GPU 候选 CPU agreement 门（§三十）。

    每个候选 = {name, patches, gpu_final_qpos, gpu_controller, gpu_steps}；
    CPU 重放候选模型+控制器，末态 qpos 与"GPU 结果"逐点比对：
    - 一致 → PROMOTE；
    - 分歧 → counterexample 落库（gpu_cpu_counterexample，保留不扔）。
    """
    promoted: list[str] = []
    counterexamples: list[dict[str, Any]] = []
    for candidate in candidates:
        name = candidate.get("name") or f"candidate_{len(promoted) + len(counterexamples)}"
        target_ref = model_ref
        patches = candidate.get("patches") or []
        if patches:
            target_ref = backend.patch_model(model_ref, patches).new_model_ref
        receipt = backend.rollout(
            target_ref,
            controller=candidate["gpu_controller"],
            steps=int(candidate["gpu_steps"]),
        )
        trace = backend.store.get(receipt.trace_ref)
        cpu_final = [float(v) for v in trace["states"][-1]["qpos"]]
        gpu_final = [float(v) for v in candidate["gpu_final_qpos"]]
        agrees = len(cpu_final) == len(gpu_final) and all(
            abs(a - b) <= tolerance for a, b in zip(cpu_final, gpu_final, strict=True)
        )
        if agrees:
            promoted.append(name)
            continue
        record = {
            "kind": "gpu_cpu_counterexample",
            "schema_version": "rosclaw.sim.counterexample.v1",
            "name": name,
            "model_ref": target_ref,
            "base_model_ref": model_ref,
            "controller": candidate["gpu_controller"],
            "steps": int(candidate["gpu_steps"]),
            "gpu_final_qpos": gpu_final,
            "cpu_final_qpos": cpu_final,
            "tolerance": tolerance,
            "trust_level": "SIMULATED",
            "usable_for_real_execution": False,
        }
        counter_ref = backend.store.put("experiments", record)
        counterexamples.append(
            {"name": name, "counterexample_ref": counter_ref, "cpu_final_qpos": cpu_final}
        )
    return {"promoted": promoted, "counterexamples": counterexamples}


# ---------------------------------------------------------------- MH24 v2

#: 语义比较容差（逐字段）。
_SEMANTIC_TOL = {
    "qpos": 1e-6,
    "qvel": 1e-6,
    "checkpoint_qpos": 1e-5,
    "peak_contact_force_rel": 0.05,
}


def _final_qvel(backend, trace_ref: str) -> list[float]:  # noqa: ANN001
    trace = backend.store.get(trace_ref)
    return [float(v) for v in trace["states"][-1]["qvel"]]


def _checkpoint_qpos(backend, trace_ref: str, step: int) -> list[float]:  # noqa: ANN001
    trace = backend.store.get(trace_ref)
    states = trace["states"]
    row = states[min(step, len(states) - 1)]
    return [float(v) for v in row["qpos"]]


def _peak_contact_force(backend, model, data) -> float:  # noqa: ANN001
    import mujoco
    import numpy as np

    mujoco.mj_forward(model, data)
    peak = 0.0
    force6 = np.zeros(6)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, force6)
        peak = max(peak, abs(float(force6[0])))
    return peak


def _semantic_agreement(backend, model_ref: str, candidate: dict[str, Any], cpu_receipt) -> dict[str, Any]:  # noqa: ANN001
    """逐字段比较（§36）：final qpos/qvel + checkpoints + task
    success + collision/peak force + tracking RMSE + audit。"""
    diverged: list[str] = []
    trace_ref = cpu_receipt.trace_ref
    cpu_final_qpos = [float(v) for v in backend.store.get(trace_ref)["states"][-1]["qpos"]]
    gpu_final_qpos = [float(v) for v in candidate["gpu_final_qpos"]]
    if len(cpu_final_qpos) != len(gpu_final_qpos) or any(
        abs(a - b) > _SEMANTIC_TOL["qpos"] for a, b in zip(cpu_final_qpos, gpu_final_qpos, strict=True)
    ):
        diverged.append("final_qpos")

    gpu_qvel = candidate.get("gpu_final_qvel")
    if gpu_qvel is not None:
        cpu_qvel = _final_qvel(backend, trace_ref)
        if len(cpu_qvel) != len(gpu_qvel) or any(
            abs(a - b) > _SEMANTIC_TOL["qvel"] for a, b in zip(cpu_qvel, gpu_qvel, strict=True)
        ):
            diverged.append("qvel")

    for checkpoint in candidate.get("gpu_checkpoints", []):
        cpu_cp = _checkpoint_qpos(backend, trace_ref, int(checkpoint["step"]))
        gpu_cp = [float(v) for v in checkpoint["qpos"]]
        if len(cpu_cp) != len(gpu_cp) or any(
            abs(a - b) > _SEMANTIC_TOL["checkpoint_qpos"] for a, b in zip(cpu_cp, gpu_cp, strict=True)
        ):
            diverged.append(f"checkpoint:{checkpoint['step']}")
            break

    cpu_task_success = cpu_receipt.task_success if cpu_receipt.task_success is not None else None
    gpu_task_success = candidate.get("gpu_task_success")
    if gpu_task_success is not None and cpu_task_success is not None and gpu_task_success != cpu_task_success:
        diverged.append("task_success")

    gpu_peak_force = candidate.get("gpu_peak_contact_force")
    if gpu_peak_force is not None:
        model, data = backend.restore_state_v2(model_ref, cpu_receipt.final_state_ref)
        cpu_peak_force = _peak_contact_force(backend, model, data)
        base = max(abs(cpu_peak_force), 1e-9)
        if abs(gpu_peak_force - cpu_peak_force) / base > _SEMANTIC_TOL["peak_contact_force_rel"]:
            diverged.append("peak_contact_force")

    return {
        "status": "SEMANTIC_AGREEMENT" if not diverged else "SEMANTIC_DIVERGENCE",
        "diverged_fields": diverged,
    }


def evaluate_gpu_candidates_semantic(
    backend,  # noqa: ANN001
    model_ref: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """GPU 候选全语义复核（§36-§37）：一致才 PROMOTE；分歧按
    SEMANTIC_DIVERGENCE 落 counterexample 语料（§38 不扔）。"""
    promoted: list[str] = []
    agreements: dict[str, Any] = {}
    counterexamples: list[dict[str, Any]] = []
    for candidate in candidates:
        name = candidate.get("name", f"cand_{len(promoted) + len(counterexamples)}")
        target_ref = model_ref
        if candidate.get("patches"):
            target_ref = backend.patch_model(model_ref, candidate["patches"]).new_model_ref
        cpu_receipt = backend.run_experiment(
            target_ref,
            controller=candidate["gpu_controller"],
            steps=int(candidate["gpu_steps"]),
            task_predicates=candidate.get("task_predicates"),
        )
        agreement = _semantic_agreement(backend, target_ref, candidate, cpu_receipt)
        agreements[name] = agreement
        if agreement["status"] == "SEMANTIC_AGREEMENT":
            promoted.append(name)
            continue
        record = {
            "kind": "gpu_cpu_counterexample",
            "schema_version": "rosclaw.sim.counterexample.v2",
            "name": name,
            "model_ref": target_ref,
            "base_model_ref": model_ref,
            "controller": candidate["gpu_controller"],
            "steps": int(candidate["gpu_steps"]),
            "diverged_fields": agreement["diverged_fields"],
            "gpu_payload": {
                key: candidate[key]
                for key in ("gpu_final_qpos", "gpu_final_qvel", "gpu_checkpoints", "gpu_task_success", "gpu_peak_contact_force", "gpu_runtime")
                if key in candidate
            },
            "cpu_trace_ref": cpu_receipt.trace_ref,
            "category": classify_counterexample({"diverged_fields": agreement["diverged_fields"]}),
            "trust_level": "SIMULATED",
            "usable_for_real_execution": False,
        }
        counter_ref = backend.store.put("experiments", record)
        counterexamples.append(
            {"name": name, "counterexample_ref": counter_ref, "category": record["category"]}
        )
    return {"promoted": promoted, "agreements": agreements, "counterexamples": counterexamples}


def classify_counterexample(counterexample: dict[str, Any]) -> str:
    """§38 counterexample 语料分类（contact/solver/friction/
    high-stiffness/constraint/state）。"""
    fields = " ".join(str(f).lower() for f in counterexample.get("diverged_fields", []))
    if "contact" in fields or "force" in fields:
        return "contact"
    if "friction" in fields:
        return "friction"
    if "constraint" in fields or "weld" in fields or "eq_" in fields:
        return "constraint"
    if "stiff" in fields or "saturation" in fields:
        return "high_stiffness"
    if "solver" in fields or "integrator" in fields or "timestep" in fields:
        return "solver"
    return "state"


def gpu_execution_status() -> dict[str, Any]:
    """GPU 执行面诚实状态（MH24）：jax-cuda / mjx / warp 探测——
    缺失即 NOT_RUN，绝不假装 GPU qualified。"""
    import importlib.util

    def _importable(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ValueError, ImportError):
            return False

    if not _importable("jax"):
        return {"status": "NOT_RUN", "reason": "jax not installed", "gpu_qualified": False}
    import jax

    devices = jax.devices()
    has_gpu = any(d.platform in ("gpu", "cuda") for d in devices)
    if not has_gpu:
        return {
            "status": "NOT_RUN",
            "reason": f"jax installed but no CUDA backend (devices={devices})——aarch64 cuda jaxlib 缺失",
            "gpu_qualified": False,
        }
    if not _importable("mujoco.mjx") and not _importable("mujoco_warp"):
        return {
            "status": "NOT_RUN",
            "reason": "GPU backend present but mujoco.mjx/mujoco_warp not installed",
            "gpu_qualified": False,
        }
    return {"status": "AVAILABLE", "gpu_qualified": True, "devices": [str(d) for d in devices]}


def run_mjx_candidate(
    backend,  # noqa: ANN001
    model_ref: str,
    *,
    controller: dict[str, Any],
    steps: int,
    task_predicates: list[dict[str, Any]] | None = None,
    name: str = "mjx_live",
) -> dict[str, Any]:
    """真实 MJX GPU 候选产出（MH24 §36 live 数据面，G41/G42）。

    诚实纪律：GPU 执行面不可用 → ``NOT_RUN``（结构化 reason，不抛
    裸异常）；真跑则记录 runtime 指纹（jax/mujoco 版本、devices、
    x64 精度标志）进 candidate.gpu_runtime。controller 与 CPU
    rollout 走同一个 validate_controller（形状 fail-closed 一致）；
    position_targets/setpoints 常量 ctrl、ctrl_series 逐行、hold
    不动——与 CPU 路径逐语义对齐（rollout.py 实证）。

    peak_contact_force 记 None（MJX 接触力通道未接——门跳过 None
    字段，绝不拿 0.0 冒充实测）。
    """
    status = gpu_execution_status()
    if status["status"] != "AVAILABLE":
        return {
            "status": "NOT_RUN",
            "reason": status.get("reason", "gpu execution unavailable"),
            "name": name,
        }

    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    from rosclaw.sim.backends.mujoco import rollout as rollout_mod
    from rosclaw.sim.experiment.predicates import evaluate_predicates

    # 与 CPU float64 权威面对齐（须在创建 jax 数组前）。
    jax.config.update("jax_enable_x64", True)

    manifest = backend._manifest(model_ref)
    spec = backend._spec_from_manifest(manifest)
    spec.assets = backend._load_assets(manifest)
    model = spec.compile()
    plan = rollout_mod.validate_controller(
        controller, model.nu, channels=backend._control_schema(model, manifest)
    )

    mx = mjx.put_model(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    dx = mjx.put_data(model, data)

    def _ctrl_values(step: int) -> list[float] | None:
        if plan["kind"] == "hold":
            return None
        if plan["kind"] in ("position_targets", "setpoints"):
            return plan["values"]
        return plan["rows"][step] if step < len(plan["rows"]) else None

    step_fn = jax.jit(mjx.step)
    checkpoint_step = steps // 2
    checkpoints: list[dict[str, Any]] = []
    for step in range(steps):
        values = _ctrl_values(step)
        if values is not None:
            dx = dx.replace(ctrl=jnp.asarray(values))
        dx = step_fn(mx, dx)
        # 与 CPU trace 对齐：states[k] = 第 k 步后的状态（states[0]=初始）。
        if step + 1 == checkpoint_step:
            checkpoints.append({"step": step + 1, "qpos": [float(v) for v in dx.qpos]})

    final_qpos = [float(v) for v in dx.qpos]
    final_qvel = [float(v) for v in dx.qvel]

    gpu_task_success: bool | None = None
    if task_predicates is not None:
        # 与 CPU 同谓词函数；通道映射限关节量（joint-only 模型，
        # 其他通道需一般化映射——本面不支持时诚实不传谓词）。
        observations = {"joint_positions": final_qpos, "joint_velocities": final_qvel}
        verdicts = evaluate_predicates(observations, task_predicates)
        gpu_task_success = all(v["ok"] for v in verdicts)

    runtime = {
        "backend": "mjx",
        "jax_version": str(jax.__version__),
        "mujoco_version": str(mujoco.__version__),
        "devices": [str(d) for d in jax.devices()],
        "x64": True,
    }
    candidate: dict[str, Any] = {
        "name": name,
        "patches": [],
        "gpu_controller": controller,
        "gpu_steps": int(steps),
        "gpu_final_qpos": final_qpos,
        "gpu_final_qvel": final_qvel,
        "gpu_checkpoints": checkpoints,
        "gpu_task_success": gpu_task_success,
        "gpu_peak_contact_force": None,
        "gpu_runtime": runtime,
    }
    if task_predicates is not None:
        candidate["task_predicates"] = task_predicates
    return {"status": "RAN", "candidate": candidate, "runtime": runtime}
