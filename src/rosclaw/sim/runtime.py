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
    ) -> dict[str, Any]:
        return self._backend.run_experiment(
            model_ref,
            state_ref=state_ref,
            controller=controller,
            duration_s=duration_s,
            steps=steps,
            seed=seed,
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
