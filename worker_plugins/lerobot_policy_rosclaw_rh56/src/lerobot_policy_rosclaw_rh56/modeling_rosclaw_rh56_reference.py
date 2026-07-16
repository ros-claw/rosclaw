"""Deterministic RH56 reference policy (LeRobot ``PreTrainedPolicy`` plugin).

This policy validates the LeRobot → ROSClaw → RH56 deployment loop without any
training.  It is a finite-state, receding-horizon **single-step** controller:
every ``select_action`` call returns the next step target, moving from the
current observation toward the active waypoint by at most
``config.max_step_delta`` raw units per actuator.

Tasks (config.task or per-step ``batch["task"]`` override):

- ``hold_current`` — output equals the current observation (no motion)
- ``open_hand`` — move all actuators to the open pose (1000)
- ``micro_index_flex`` — index open → -20 → hold → return, cycling
- ``half_close`` — move all actuators to 500
- ``return_open`` — return to the open pose
- ``countdown_pose`` — 5 → 4 → 5 non-contact gesture (little closes to 100)
- ``ok_pose_safe`` — one-pose → synchronized thumb/index/thumb_rot approach to
  the validated OK pose (thumb=420, index=410, thumb_rot=300) → hold → open

Action order is always the RS485 canonical order:
``[little, ring, middle, index, thumb, thumb_rot]``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy

from lerobot_policy_rosclaw_rh56.configuration_rosclaw_rh56_reference import (
    RosclawRH56ReferenceConfig,
)

# Canonical RS485 action order indices.
LITTLE, RING, MIDDLE, INDEX, THUMB, THUMB_ROT = range(6)

OPEN = 1000
SAFE_CLOSED = 100

# Validated OK-contact pose (from the ROSClaw RH56 example telemetry).
OK_POSE = {THUMB: 420, INDEX: 410, THUMB_ROT: 300}
ONE_POSE = {INDEX: 500}


def _pose(overrides: dict[int, int] | None = None, base: int = OPEN) -> list[float]:
    pose = [float(base)] * 6
    for idx, value in (overrides or {}).items():
        pose[idx] = float(value)
    return pose


class _TaskProgram:
    """Waypoint sequence for one task; waypoints loop."""

    def __init__(self, waypoints: list[list[float]]):
        self.waypoints = waypoints


def _build_program(task: str) -> _TaskProgram:
    if task == "open_hand":
        return _TaskProgram([_pose()])
    if task == "half_close":
        return _TaskProgram([_pose(base=500)])
    if task == "return_open":
        return _TaskProgram([_pose()])
    if task == "micro_index_flex":
        return _TaskProgram(
            [
                _pose(),
                _pose({INDEX: OPEN - 20}),
                _pose({INDEX: OPEN - 20}),  # hold
                _pose(),
            ]
        )
    if task == "countdown_pose":
        # 5 → 4 → 5: little closes to the safe-closed position, then re-opens.
        return _TaskProgram(
            [
                _pose(),
                _pose({LITTLE: SAFE_CLOSED}),
                _pose({LITTLE: SAFE_CLOSED}),  # hold
                _pose(),
            ]
        )
    if task == "ok_pose_safe":
        return _TaskProgram(
            [
                _pose(),
                _pose(ONE_POSE),
                _pose(OK_POSE),
                _pose(OK_POSE),  # hold
                _pose(),
            ]
        )
    # hold_current handled dynamically (target = observation).
    return _TaskProgram([])


class RosclawRH56ReferencePolicy(PreTrainedPolicy):
    """Deterministic finite-state RH56 reference policy."""

    config_class = RosclawRH56ReferenceConfig
    name = "rosclaw_rh56_reference"

    def __init__(self, config: RosclawRH56ReferenceConfig, dataset_stats=None):
        super().__init__(config)
        config.validate_features()
        self.config = config
        # A registered buffer keeps save_pretrained / from_pretrained functional
        # even though the policy has no trainable weights.
        self.register_buffer("_identity", torch.zeros(1), persistent=True)
        self.reset()

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._waypoint_index = 0
        self._hold_counter = 0

    def _resolve_task(self, batch: dict) -> str:
        task = batch.get("task")
        if isinstance(task, (list, tuple)) and task:
            task = task[0]
        if isinstance(task, str) and task:
            return task
        return self.config.task

    def _step_toward(self, current: Tensor, target: list[float]) -> Tensor:
        """Move one bounded step from ``current`` toward ``target``."""
        tgt = torch.tensor(target, dtype=current.dtype, device=current.device)
        delta = tgt - current
        max_delta = float(self.config.max_step_delta)
        clamped = torch.clamp(delta, -max_delta, max_delta)
        return current + clamped

    def _reached(self, current: Tensor, target: list[float]) -> bool:
        tgt = torch.tensor(target, dtype=current.dtype, device=current.device)
        return bool(torch.all(torch.abs(tgt - current) <= 5.0).item())

    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        state = batch["observation.state"]
        if state.dim() == 1:
            state = state.unsqueeze(0)
        current = state[0].to(dtype=torch.float32)

        task = self._resolve_task(batch)
        if task == "hold_current":
            return current.unsqueeze(0).clone()

        program = _build_program(task)
        if not program.waypoints:
            return current.unsqueeze(0).clone()

        target = program.waypoints[self._waypoint_index % len(program.waypoints)]
        if self._reached(current, target):
            self._hold_counter += 1
            if self._hold_counter > self.config.hold_steps:
                self._hold_counter = 0
                self._waypoint_index += 1
                target = program.waypoints[self._waypoint_index % len(program.waypoints)]
        else:
            self._hold_counter = 0

        action = self._step_toward(current, target)
        action = torch.clamp(action, 0.0, 1000.0)
        return action.unsqueeze(0)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self.select_action(batch, **kwargs).unsqueeze(1)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        # Deterministic policy: zero loss, no training signal.
        loss = torch.zeros((), device=self._identity.device)
        return loss, {"task": self._resolve_task(batch)}

    def get_optim_params(self) -> dict:
        return {}
