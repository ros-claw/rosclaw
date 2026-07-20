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
    )
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
