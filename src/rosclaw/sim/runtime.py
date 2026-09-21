"""SimulationRuntime（PR-MH6，ADR-0014，规格 §6/§28）。

Maturity: experimental（ADR-0000 §4）。

Agent 面的仿真编排门面：属于 ROSClaw Native Runtime（进程内能力，
非独立进程、非 daemon 子系统）。Agent 只经 ToolGateway →
RuntimeClient 触达；返回全部 JSON 友好 dict（契约 canonical form +
usable_for_real_execution=false 恒成立）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rosclaw.sim.backends.mujoco.backend import MujocoBackend


def default_task_root() -> Path:
    """RuntimeClient 默认任务根：ROSCLAW_SIM_TASK_ROOT > $ROSCLAW_HOME/sim_tasks/agent。"""
    override = os.environ.get("ROSCLAW_SIM_TASK_ROOT")
    if override:
        return Path(override)
    home = os.environ.get("ROSCLAW_HOME") or str(Path.home() / ".rosclaw")
    return Path(home) / "sim_tasks" / "agent"


class SimulationRuntime:
    """Agent 工具面下的仿真运行时（当前唯一后端：MujocoBackend）。"""

    def __init__(self, task_root: Path | str | None = None) -> None:
        self._backend = MujocoBackend(task_root or default_task_root())

    @property
    def backend(self) -> MujocoBackend:
        return self._backend

    def get_capabilities(self) -> dict[str, Any]:
        caps = self._backend.capabilities().to_canonical_dict()
        caps["usable_for_real_execution"] = False
        return caps

    def load_model(self, asset_ref: str) -> dict[str, Any]:
        return self._backend.load_model(asset_ref).to_canonical_dict()

    def inspect_model(self, model_ref: str) -> dict[str, Any]:
        return self._backend.inspect_model(model_ref).to_canonical_dict()

    def patch_model(self, model_ref: str, patches: list[dict[str, Any]]) -> dict[str, Any]:
        return self._backend.patch_model(model_ref, patches).to_canonical_dict()

    def snapshot(self, model_ref: str, state_ref: str | None = None) -> dict[str, Any]:
        if state_ref is None:
            ref = self._backend.initial_state(model_ref)
        else:
            snap = self._backend.store.get(state_ref)
            if not isinstance(snap, dict):
                raise ValueError(f"REF_NOT_FOUND: {state_ref!r} is not a state snapshot")
            ref = self._backend.snapshot_state(model_ref, snap)
        return {"state_ref": ref, "model_ref": model_ref}

    def observe(self, model_ref: str, state_ref: str, channels: list[str]) -> dict[str, Any]:
        return self._backend.observe(model_ref, state_ref, channels).to_canonical_dict()

    def rollout(
        self,
        model_ref: str,
        *,
        controller: dict[str, Any],
        duration_s: float | None = None,
        steps: int | None = None,
        state_ref: str | None = None,
        seed: int = 0,
        task_predicates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self._backend.run_experiment(
            model_ref,
            state_ref=state_ref,
            controller=controller,
            duration_s=duration_s,
            steps=steps,
            seed=seed,
            task_predicates=task_predicates,
        ).to_canonical_dict()

    def audit(
        self,
        model_ref: str,
        *,
        checks: list[str] | None = None,
        trace_ref: str | None = None,
    ) -> dict[str, Any]:
        return self._backend.audit(
            model_ref, checks=checks, trace_ref=trace_ref
        ).to_canonical_dict()

    def compare(self, receipt_refs: list[str]) -> dict[str, Any]:
        return self._backend.compare_experiments(receipt_refs).to_canonical_dict()

    def branch_experiment(
        self,
        model_ref: str,
        *,
        branches: list[dict[str, Any]],
        controller: dict[str, Any],
        state_ref: str | None = None,
        duration_s: float | None = None,
        steps: int | None = None,
        seed: int = 0,
        task_predicates: list[dict[str, Any]] | None = None,
        parallel: bool = True,
    ) -> dict[str, Any]:
        """高层参数实验原语（PR-MH9/MH14）：
        base model + base state + patches[] → N branches → rollout。

        parallel=True 时同构分支走 `mujoco.rollout` 原生批量多线程
        （CPU batch，与权威 truth 同域）；异构自动回退串行并记录
        execution。branches: [{"name"?, "patches": [...]}]，空 patches
        为对照分支。
        """
        from rosclaw.sim.backends.mujoco.rollout import DEFAULT_BUDGETS

        if not isinstance(branches, list) or not branches:
            raise ValueError("BRANCHES_REQUIRED: branches must be a non-empty list")
        if len(branches) > DEFAULT_BUDGETS["max_branch_count"]:
            raise ValueError(
                f"SIM_BUDGET_EXCEEDED: branch count {len(branches)} > "
                f"{DEFAULT_BUDGETS['max_branch_count']}"
            )
        base_state = state_ref or self._backend.initial_state_v2(model_ref)
        fork = self._backend.fork_state(model_ref, base_state, len(branches))

        target_refs = []
        for index, branch in enumerate(branches):
            if not isinstance(branch, dict):
                raise ValueError(f"BRANCH_INVALID: branches[{index}] must be a mapping")
            patches = branch.get("patches", [])
            target_ref = model_ref
            if patches:
                patched_ref = self._backend.patch_model(model_ref, patches).new_model_ref
                if patched_ref is None:
                    raise ValueError("MODEL_PATCH_INVALID: patch produced no new model ref")
                target_ref = patched_ref
            target_refs.append(target_ref)

        execution = "serial"
        batch_results = None
        if parallel:
            try:
                batch_results = self._backend.rollout_batch(
                    target_refs,
                    controller=controller,
                    duration_s=duration_s,
                    steps=steps,
                )
                execution = "batch_parallel"
            except ValueError as exc:
                if "BATCH_NOT_HOMOGENEOUS" not in str(exc):
                    raise
                batch_results = None  # 异构回退串行

        receipts = []
        for index, target_ref in enumerate(target_refs):
            branch_state = self._backend.transplant_state(target_ref, base_state)
            if batch_results is not None:
                batch_result = batch_results[index]
                # 批量轨迹已有；审计与任务判定照常（指标为轨迹子集）。
                audit_result = self._backend.audit(target_ref, trace_ref=batch_result["trace_ref"])
                task_success = None
                if task_predicates is not None:
                    channels = sorted({p["channel"] for p in task_predicates})
                    observations = self._backend.observe(
                        target_ref, batch_result["final_state_ref"], channels
                    ).values
                    from rosclaw.sim.experiment.predicates import evaluate_predicates

                    verdicts = evaluate_predicates(observations, task_predicates)
                    task_success = all(v["ok"] for v in verdicts)
                receipts.append(
                    {
                        "model_ref": target_ref,
                        "trace_ref": batch_result["trace_ref"],
                        "final_state_ref": batch_result["final_state_ref"],
                        "states_digest": batch_result["states_digest"],
                        "steps": batch_result["steps"],
                        "audit_ref": audit_result.audit_ref,
                        "physical_audit_pass": audit_result.status == "PASS",
                        "task_success": task_success,
                        "metrics_mode": "batch_trajectory",
                        "execution": execution,
                        "trust_level": "SIMULATED",
                        "usable_for_real_execution": False,
                    }
                )
            else:
                receipts.append(
                    self._backend.run_experiment(
                        target_ref,
                        state_ref=branch_state,
                        controller=controller,
                        duration_s=duration_s,
                        steps=steps,
                        seed=seed,
                        task_predicates=task_predicates,
                    ).to_canonical_dict()
                )
        return {
            "fork_ref": fork["fork_ref"],
            "base_state_ref": base_state,
            "receipts": receipts,
            "count": len(receipts),
            "execution": execution,
        }

    def compile_world(self, worldspec: dict[str, Any], *, name: str = "world") -> dict[str, Any]:
        """高层世界编译原语（PR-MH9，0915 §七）：
        WorldSpec → validation → 能力声明→证明绑定 → compile → model_ref。"""
        from rosclaw.sim.world.compiler import compile_world

        return compile_world(self._backend, worldspec, name=name)

    def interact(
        self,
        model_ref: str,
        state_ref: str,
        interaction: dict[str, Any],
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行 typed interaction（官方 executor registry）。"""
        return self._backend.interact(model_ref, state_ref, interaction, payload)

    def load_menagerie(self, model_name: str, entry: str | None = None) -> dict[str, Any]:
        """Menagerie 导入（MH16 §二十四）：锁版本 + provenance 五元组。"""
        ref = self._backend.load_menagerie(model_name, entry=entry)
        return {
            "model_ref": ref.model_ref,
            "backend": ref.backend,
            "source": self._backend.store.get(ref.model_ref)["source"],
        }

    def scaffold_menagerie(self, model_name: str, out_dir: str) -> dict[str, Any]:
        """§24.3：e-URDF scaffold（能力 UNDECLARED 起步）。"""
        from pathlib import Path

        out = self._backend.scaffold_eurdf_from_menagerie(model_name, Path(out_dir))
        return {"scaffold_dir": str(out), "capability_default": "UNDECLARED"}

    def acceleration_compatibility(self, model_ref: str) -> dict[str, Any]:
        """Backend Fidelity Gate（MH18 §二十九）。"""
        return self._backend.acceleration_compatibility(model_ref)

    def record_dataset(self, model_ref: str, sequences: list[dict[str, Any]]) -> dict[str, Any]:
        """录制 SysID 数据集（MH17，内容寻址幂等）。"""
        dataset_ref = self._backend.record_dataset(model_ref, sequences=sequences)
        return {"dataset_ref": dataset_ref, "sequences": len(sequences)}

    def sysid(self, spec: dict[str, Any]) -> dict[str, Any]:
        """System Identification（MH17）：官方 sysid 工具箱 +
        holdout 独立复算（§26.3）→ SysIDReceipt。"""
        return self._backend.run_sysid(spec)

    def render(
        self,
        trace_ref: str,
        *,
        camera: str | None = None,
        width: int = 640,
        height: int = 480,
        max_frames: int = 16,
    ) -> dict[str, Any]:
        return self._backend.render(
            trace_ref, camera=camera, width=width, height=height, max_frames=max_frames
        )
