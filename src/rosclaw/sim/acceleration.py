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
