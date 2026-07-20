"""Reusable service for evidence-bearing sandbox actions."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.kernel import ActionEnvelope, ExecutionMode, ExecutionReceipt


class SandboxConfigurationError(ValueError):
    """A sandbox request selects an unsupported or misleading backend."""


@dataclass(frozen=True)
class SandboxRunRequest:
    """One sandbox action submitted through the canonical Runtime gateway."""

    robot: str
    world: str = "empty"
    task: str = "reach"
    mode: ExecutionMode = ExecutionMode.SIMULATION
    backend: str = "mujoco"
    target: tuple[float, float, float] | None = None
    max_steps: int = 1200
    tolerance_m: float = 0.008
    seed: int = 0
    artifact_root: Path | None = None
    trace_id: str | None = None
    action_id: str | None = None
    actor_id: str = "rosclaw-cli"
    agent_framework: str = "cli"


def run_sandbox_action(request: SandboxRunRequest) -> ExecutionReceipt:
    """Run one real MuJoCo or explicit fixture action and return its receipt."""

    mode = ExecutionMode(request.mode)
    backend = request.backend.lower()
    if mode is ExecutionMode.FIXTURE:
        backend = "fixture"
    elif mode is ExecutionMode.SIMULATION and backend != "mujoco":
        raise SandboxConfigurationError(
            "SIMULATION mode requires backend 'mujoco'; use FIXTURE mode explicitly "
            "for synthetic execution."
        )
    elif mode is not ExecutionMode.SIMULATION:
        raise SandboxConfigurationError(
            "The sandbox reach service accepts only SIMULATION or FIXTURE mode."
        )

    artifact_root = request.artifact_root.expanduser().resolve() if request.artifact_root else None
    action_id = request.action_id or f"action_{uuid.uuid4().hex}"
    trace_home = artifact_root / action_id / "trace" if artifact_root is not None else None
    runtime = Runtime(
        RuntimeConfig(
            robot_id=request.robot,
            default_eurdf_robot="ur5e",
            enable_event_persistence=False,
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            sandbox_engine=backend,
            sandbox_world_id=request.world,
            sandbox_artifact_root=str(artifact_root) if artifact_root else None,
            trace_home=str(trace_home) if trace_home else None,
        )
    )
    receipt: ExecutionReceipt
    try:
        runtime.initialize()
        model_path = runtime.sandbox.model_path if runtime.sandbox is not None else None
        body_hash = ""
        if model_path is not None and model_path.is_file():
            body_hash = f"sha256:{hashlib.sha256(model_path.read_bytes()).hexdigest()}"

        arguments: dict[str, object] = {
            "task": request.task,
            "max_steps": request.max_steps,
            "tolerance_m": request.tolerance_m,
            "seed": request.seed,
        }
        if request.target is not None:
            arguments["target"] = list(request.target)
        action = ActionEnvelope(
            action_id=action_id,
            actor_id=request.actor_id,
            agent_framework=request.agent_framework,
            session_id=request.trace_id or f"cli_{uuid.uuid4().hex[:12]}",
            body_id=request.robot,
            body_snapshot_hash=body_hash,
            capability_id=f"sandbox.{request.task}",
            arguments=arguments,
            execution_mode=mode,
            parent_trace_id=request.trace_id,
        )
        receipt = runtime.submit_action(action)
    finally:
        runtime.stop()
    if trace_home is not None:
        trace_path = trace_home / "traces" / "live.jsonl"
        if trace_path.is_file() and trace_path.as_uri() not in receipt.artifacts:
            receipt.artifacts.append(trace_path.as_uri())
    return receipt


__all__ = [
    "SandboxConfigurationError",
    "SandboxRunRequest",
    "run_sandbox_action",
]
