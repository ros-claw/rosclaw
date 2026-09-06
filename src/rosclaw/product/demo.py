"""Official product demos backed by canonical Runtime execution."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rosclaw.kernel import ExecutionMode, ExecutionReceipt
from rosclaw.product.runs import ProductRunStore


class DemoNotFoundError(ValueError):
    """The requested official demo does not exist."""


class DemoConfigurationError(ValueError):
    """An official demo request contains unsafe or unbounded parameters."""


@dataclass(frozen=True)
class DemoDefinition:
    """Stable metadata for one official demo."""

    id: str
    title: str
    robot: str
    capability: str
    mode: ExecutionMode
    description: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "robot": self.robot,
            "capability": self.capability,
            "mode": self.mode.value,
            "description": self.description,
        }


DEMOS = {
    "ur5e-reach": DemoDefinition(
        id="ur5e-reach",
        title="UR5e Reach",
        robot="sim_ur5e",
        capability="sandbox.reach",
        mode=ExecutionMode.SIMULATION,
        description="Physics-backed MuJoCo reach with policy, collision, and task verification.",
    ),
    # 大道至简 R0-2a：固定五角星流程的唯一保留形态——显式 demo。
    # 它不再拦截用户对话（聊天路径的确定性路由已删除）；Runtime
    # 只报客观执行事实（轨迹执行+跟踪误差），不宣称用户目标完成。
    "ur5e-star": DemoDefinition(
        id="ur5e-star",
        title="UR5e Star Trajectory",
        robot="sim_ur5e",
        capability="sim.draw_path.star5",
        mode=ExecutionMode.SIMULATION,
        description=(
            "UR5e draws a five-pointed star in MuJoCo dynamics: plan → "
            "DLS-IK rollout → tracking verification → GIF render, with "
            "objective metrics (max/mean tracking error) and artifacts."
        ),
    ),
}


def list_demos() -> list[DemoDefinition]:
    """Return official demos in stable ID order."""

    return [DEMOS[key] for key in sorted(DEMOS)]


def run_demo(
    demo_id: str,
    *,
    home: Path | None = None,
    target: tuple[float, float, float] | None = None,
    max_steps: int = 1200,
    tolerance_m: float = 0.008,
    seed: int = 0,
    trace_id: str | None = None,
    actor_id: str = "rosclaw-cli",
    agent_framework: str = "cli",
) -> tuple[ExecutionReceipt, Path]:
    """Run one official demo and persist its canonical receipt."""

    definition = DEMOS.get(demo_id)
    if definition is None:
        choices = ", ".join(sorted(DEMOS))
        raise DemoNotFoundError(f"Unknown demo {demo_id!r}. Available demos: {choices}")
    if demo_id == "ur5e-star":
        return _run_star_demo(home=home, actor_id=actor_id,
                              agent_framework=agent_framework)
    _validate_configuration(
        target=target,
        max_steps=max_steps,
        tolerance_m=tolerance_m,
    )

    store = ProductRunStore(home)
    from rosclaw.sandbox.service import SandboxRunRequest, run_sandbox_action

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
    receipt = run_sandbox_action(
        SandboxRunRequest(
            robot=definition.robot,
            world="tabletop",
            task="reach",
            mode=definition.mode,
            backend="mujoco",
            target=target,
            max_steps=max_steps,
            tolerance_m=tolerance_m,
            seed=seed,
            artifact_root=store.root,
            trace_id=trace_id,
            action_id=run_id,
            actor_id=actor_id,
            agent_framework=agent_framework,
        )
    )
    receipt_path = store.save(receipt)
    return receipt, receipt_path


def _run_star_demo(
    *,
    home: Path | None,
    actor_id: str,
    agent_framework: str,
) -> tuple[ExecutionReceipt, Path]:
    """ur5e-star：显式五角星 demo——plan → 动力学 rollout → 跟踪验证
    → GIF 渲染。Receipt 只携带客观执行事实（轨迹/误差/交付物），
    不做任何用户目标语义宣称（大道至简 R0-2a）。"""
    from rosclaw.agentd.runtime_manager import RuntimeManager
    from rosclaw.agentd.sim_trajectory import SimTrajectoryService
    from rosclaw.kernel import ActionState, EvidenceDomain, EvidenceLevel

    store = ProductRunStore(home)
    # SimTrajectoryService 的 home 约定：<home>/sim/{plans,traces}——
    # 与产品 run store 同一 ROSCLAW_HOME。
    svc = SimTrajectoryService(
        store.home, runtime_manager=RuntimeManager(store.home),
    )
    plan = svc.generate_planar_path(
        shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.10,
    )
    rollout = svc.simulate_cartesian_trajectory(str(plan["plan_id"]))
    trace_id = str(rollout["trace_id"])
    verify = svc.verify_tracking(trace_id, max_tracking_error_m=0.05)
    gif = svc.render_trace(trace_id, format="gif")
    verdict = str(verify.get("verdict", "FAIL"))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
    metrics = dict(verify.get("metrics") or {})
    gif_path = str((gif.get("artifact") or {}).get("path") or "")
    mp4_path = str((gif.get("mp4_artifact") or {}).get("path") or "")
    receipt = ExecutionReceipt(
        action_id=run_id,
        trace_id=trace_id,
        mode=ExecutionMode.SIMULATION,
        body_id="sim_ur5e",
        body_snapshot_hash="",
        capability_id="sim.draw_path.star5",
        final_state=(
            ActionState.COMPLETED if verdict == "PASS" else ActionState.FAILED
        ),
        evidence_level=(
            EvidenceLevel.TASK_VERIFIED
            if verdict == "PASS"
            else EvidenceLevel.SYNTHETIC
        ),
        evidence_domain=EvidenceDomain.SIMULATION,
        simulation_result={
            "physics_executed": bool(rollout.get("physics_executed", True)),
            "plan_id": str(plan["plan_id"]),
            "point_count": int(rollout.get("point_count", 0) or 0),
            "is_safe": bool(rollout.get("is_safe", True)),
            # 客观事实纪律：误差数值随 receipt 可被第三方独立复核。
            "max_tracking_error_m": metrics.get("max_error_m"),
            "mean_tracking_error_m": metrics.get("mean_error_m"),
        },
        verification_result=verify,
        artifacts=[p for p in (gif_path, mp4_path) if p],
        dispatch_result={"actor_id": actor_id,
                         "agent_framework": agent_framework},
    )
    receipt_path = store.save(receipt)
    return receipt, receipt_path


def _validate_configuration(
    *,
    target: tuple[float, float, float] | None,
    max_steps: int,
    tolerance_m: float,
) -> None:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise DemoConfigurationError("max_steps must be an integer.")
    if not 1 <= max_steps <= 5000:
        raise DemoConfigurationError("max_steps must be between 1 and 5000.")
    if isinstance(tolerance_m, bool) or not isinstance(tolerance_m, (int, float)):
        raise DemoConfigurationError("tolerance_m must be a finite number.")
    if not math.isfinite(tolerance_m):
        raise DemoConfigurationError("tolerance_m must be a finite number.")
    if not 0.00001 <= tolerance_m <= 0.1:
        raise DemoConfigurationError("tolerance_m must be between 0.00001 and 0.1.")
    if target is None:
        return
    if len(target) != 3:
        raise DemoConfigurationError("target must contain exactly three coordinates.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in target
    ):
        raise DemoConfigurationError("target coordinates must be finite numbers.")


__all__ = [
    "DEMOS",
    "DemoConfigurationError",
    "DemoDefinition",
    "DemoNotFoundError",
    "list_demos",
    "run_demo",
]
